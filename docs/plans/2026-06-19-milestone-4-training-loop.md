# Milestone 4 — training loop + overfit-a-batch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the Milestone-3 `Forward` to trainable GoMLX variables + AdamW + next-token cross-entropy, and prove the training loop drives the loss to ~0 on one tiny batch (the overfit-a-batch gate).

**Architecture:** A new `train` module declares one trainable variable per model weight in a `model.Scope` (`ModelVars`), runs `model.Forward` over those variable Nodes inside the GoMLX Trainer's `modelFn`, uses the library `loss.SparseCategoricalCrossEntropyLogits` (integer labels), AdamW, and a one-batch in-memory dataset. The overfit test asserts seed-independent convergence.

**Tech Stack:** Go 1.26, GoMLX `main` (`516689cbe913`) training machinery (`ml/model`, `ml/train`, `ml/train/optimizer`, `ml/train/loss`, `ml/dataset`, `ml/model/initializer`); reuses `model.Forward`; SimpleGo (`-tags noxla`, `GOMLX_BACKEND=go`).

## Global Constraints

- **Runtime boundary (ADR-0009):** `train/` (incl. `*_test.go`) must NOT import the runtime packages `"github.com/gomlx/compute"` (root), `gomlx/backends`, `go-xla`, `gomlx/gopjrt`. It MAY import the training vocabulary (`ml/model`, `ml/model/initializer`, `ml/train`, `ml/train/optimizer`, `ml/train/loss`, `ml/dataset`, `core/graph`, `core/tensors`, `compute/shapes`, `compute/dtypes`) and our `backend/gomlx` + `model`. Get the backend via `gomlx.New().Compute()` (type inferred — never write `compute.Backend`).
- **Per-package modules + go.work (ADR-0005):** `train` is its own module.
- **No seed-shopping (the M3 lesson):** the overfit assertion must be seed-independent — assert the loss fell substantially AND reached near zero, with thresholds set from the *observed* converged loss with comfortable margin. Never tune a threshold or seed to just-pass.
- **Determinism:** the batch is fixed; variable init is seeded (`model.ParamInitialSeed`) for reproducible debugging, but the assertion does not depend on the seed.
- **Commits:** terse, verb-first, no dashes, no Claude/Anthropic attribution.

## Verified GoMLX training API (pinned `516689cbe913`)

```go
import (
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/dataset"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/initializer"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/optimizer"
)

// variable -> node (inside a modelFn; gr := inputs[0].Graph())
v := scope.WithInitializer(initializer.RandomNormalFn(scope, 0.02)).
	VariableWithShape("name", shapes.Make(dtypes.Float32, dims...))
node := v.NodeValue(gr)                 // *g.Node ; v.ValueGraph is the deprecated alias
// initializers: initializer.One (a VariableInitializer func directly),
//   initializer.RandomNormalFn(scope, stddev), initializer.GlorotUniformFn(scope), initializer.Zero
// seed: store.RootScope().SetParam(model.ParamInitialSeed, int64(seed))  // 0 = nondeterministic

// loss for integer labels: labels MUST be [B,T,1] int, same rank as logits [B,T,V]
//   (it panics on [B,T]). It matches loss.LossFn(labels, predictions []*g.Node) *g.Node.
lossFn := loss.SparseCategoricalCrossEntropyLogits

// dataset: inputs[0] -> modelFn inputs[0]; labels[0] -> lossFn labels[0]; int tensors ok
ds, _ := dataset.InMemoryFromData(be, "name", []any{inputs}, []any{labels})
ds = ds.Infinite(true).BatchSize(B, false)   // one fixed batch: no Shuffle

opt := optimizer.Adam().WeightDecay(wd).LearningRate(lr).Done()
trainer := train.NewTrainer(be, store, modelFn, lossFn, opt, nil, nil) // 7 args
loop := train.NewLoop(trainer)
metrics, _ := loop.RunSteps(ds, n)           // metrics[0] is the loss (last step)
lossVal := shapes.ConvertTo[float64](metrics[0].Value())  // scalar -> float64
// modelFn signature: func(scope *model.Scope, spec any, inputs []*g.Node) []*g.Node
// shape of a tensor: t.Shape().Dimensions []int (same accessor used on *g.Node in model/)
```

`be` is `gomlx.New().Compute()` — the type is inferred, so it is never named (ADR-0009).

---

### Task 1: train module + ModelVars + Overfit driver

The `train` module, the variable declarations, and the training driver, with a smoke test that it runs and the loss decreases.

**Files:**
- Create: `train/go.mod`
- Modify: `go.work` (add `./train`)
- Create: `train/vars.go`
- Create: `train/train.go`
- Create: `train/train_test.go`
- Modify: `scripts/check.sh` (add `train` to the module loop)

**Interfaces:**
- Produces: `train.ModelVars(scope *model.Scope, gr *g.Graph, cfg model.Config) model.Weights`; `train.Overfit(cfg model.Config, inputs, labels *tensors.Tensor, steps int, lr, weightDecay float64, seed int64) (start, end float64, err error)`.

- [ ] **Step 1: Create the module and add it to the workspace**

`train/go.mod`:
```
module github.com/guygrigsby/lmkit-go/train

go 1.26
```
`go.work` — add `./train` to the `use` block (alongside backend/app/model). Then wire the local deps (with `go.work` present they resolve locally):
```bash
cd train && go get github.com/guygrigsby/lmkit-go/model github.com/gomlx/gomlx@516689cbe91329bf9aeac7750495a48bb85f9287 && cd ..
```

- [ ] **Step 2: Write ModelVars**

`train/vars.go`:
```go
// Package train wires the lmkit-go model to GoMLX variables, an optimizer, and a
// loss for training. It uses the GoMLX training vocabulary (ml/model, ml/train,
// ...) but not the runtime (ADR-0009): the backend comes from gomlx.New().Compute().
package train

import (
	"fmt"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/initializer"

	lmodel "github.com/guygrigsby/lmkit-go/model"
)

// ModelVars declares one trainable variable per model weight in scope and returns
// them as a model.Weights whose Nodes are the variables' graph values. Norm scales
// initialize to 1; projections and the embedding to a small random normal. Called
// inside the training modelFn (gr := inputs[0].Graph()). Variable names mirror the
// weight structure (embed, final_norm, layer{i}_Wq, ...).
func ModelVars(scope *model.Scope, gr *g.Graph, cfg lmodel.Config) lmodel.Weights {
	norm := scope.WithInitializer(initializer.One)
	proj := scope.WithInitializer(initializer.RandomNormalFn(scope, 0.02))
	mk := func(s *model.Scope, name string, dims ...int) *g.Node {
		return s.VariableWithShape(name, shapes.Make(dtypes.Float32, dims...)).NodeValue(gr)
	}
	h, nh, nkv, hd, ffn, vocab := cfg.Hidden, cfg.NHeads, cfg.NKVHeads, cfg.HeadDim, cfg.FFNHidden, cfg.VocabSize
	w := lmodel.Weights{
		Embed:     mk(proj, "embed", vocab, h),
		FinalNorm: mk(norm, "final_norm", h),
	}
	for i := 0; i < cfg.NLayers; i++ {
		p := fmt.Sprintf("layer%d_", i)
		w.Layers = append(w.Layers, lmodel.LayerWeights{
			AttnNorm: mk(norm, p+"attn_norm", h),
			Wq:       mk(proj, p+"Wq", h, nh*hd),
			Wk:       mk(proj, p+"Wk", h, nkv*hd),
			Wv:       mk(proj, p+"Wv", h, nkv*hd),
			Wo:       mk(proj, p+"Wo", nh*hd, h),
			FFNNorm:  mk(norm, p+"ffn_norm", h),
			Wgate:    mk(proj, p+"Wgate", h, ffn),
			Wup:      mk(proj, p+"Wup", h, ffn),
			Wdown:    mk(proj, p+"Wdown", ffn, h),
		})
	}
	return w
}
```

- [ ] **Step 3: Write the Overfit driver**

`train/train.go`:
```go
package train

import (
	"fmt"

	"github.com/gomlx/compute/shapes"
	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/dataset"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/optimizer"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	lmodel "github.com/guygrigsby/lmkit-go/model"
)

// Overfit trains a fresh model on one fixed batch for `steps` steps with AdamW and
// next-token cross-entropy, returning the loss at the first step and after `steps`.
// inputs is [B,T] int token ids; labels is [B,T,1] int next-token ids (the trailing
// 1 is required by SparseCategoricalCrossEntropyLogits). seed seeds variable init
// (for reproducible debugging; the loop converges for any seed).
func Overfit(cfg lmodel.Config, inputs, labels *tensors.Tensor, steps int, lr, weightDecay float64, seed int64) (start, end float64, err error) {
	bk, err := gomlx.New()
	if err != nil {
		return 0, 0, fmt.Errorf("train: backend: %w", err)
	}
	be := bk.Compute()

	store := model.NewStore()
	store.RootScope().SetParam(model.ParamInitialSeed, seed)

	tt := inputs.Shape().Dimensions[1] // T
	positions := make([]int, tt)
	for i := range positions {
		positions[i] = i
	}
	modelFn := func(scope *model.Scope, spec any, in []*g.Node) []*g.Node {
		w := ModelVars(scope, in[0].Graph(), cfg)
		return []*g.Node{lmodel.Forward(cfg, w, in[0], positions)}
	}

	ds, err := dataset.InMemoryFromData(be, "overfit", []any{inputs}, []any{labels})
	if err != nil {
		return 0, 0, fmt.Errorf("train: dataset: %w", err)
	}
	bsz := inputs.Shape().Dimensions[0] // B
	ds = ds.Infinite(true).BatchSize(bsz, false)

	opt := optimizer.Adam().WeightDecay(weightDecay).LearningRate(lr).Done()
	trainer := train.NewTrainer(be, store, modelFn, loss.SparseCategoricalCrossEntropyLogits, opt, nil, nil)
	loop := train.NewLoop(trainer)

	m0, err := loop.RunSteps(ds, 1) // first step: loss at (near) init
	if err != nil {
		return 0, 0, fmt.Errorf("train: initial step: %w", err)
	}
	start = shapes.ConvertTo[float64](m0[0].Value())
	mN, err := loop.RunSteps(ds, steps) // `steps` more
	if err != nil {
		return 0, 0, fmt.Errorf("train: steps: %w", err)
	}
	end = shapes.ConvertTo[float64](mN[0].Value())
	return start, end, nil
}
```

- [ ] **Step 4: Write the smoke test (failing)**

`train/train_test.go`:
```go
package train_test

import (
	"testing"

	"github.com/gomlx/gomlx/core/tensors"

	lmodel "github.com/guygrigsby/lmkit-go/model"
	"github.com/guygrigsby/lmkit-go/train"
)

// tinyCfg + a fixed next-token batch shared by the train tests.
func tinyCfg() lmodel.Config {
	return lmodel.Config{
		Hidden: 16, NLayers: 2, NHeads: 4, NKVHeads: 2, HeadDim: 4,
		FFNHidden: 32, VocabSize: 32, RopeBase: 10000, RMSEps: 1e-5,
	}
}

// fixedBatch builds one deterministic batch: a base sequence of B*(T+1) ids, with
// inputs = seq[:, :T] and labels = seq[:, 1:] (next token), labels shaped [B,T,1].
func fixedBatch(b, t, vocab int) (inputs, labels *tensors.Tensor) {
	in := make([]int32, b*t)
	lab := make([]int32, b*t)
	for bi := 0; bi < b; bi++ {
		seq := make([]int32, t+1)
		for k := range seq {
			seq[k] = int32((bi*(t+1)+k)*13+7) % int32(vocab)
		}
		for ti := 0; ti < t; ti++ {
			in[bi*t+ti] = seq[ti]
			lab[bi*t+ti] = seq[ti+1]
		}
	}
	return tensors.FromFlatDataAndDimensions(in, b, t),
		tensors.FromFlatDataAndDimensions(lab, b, t, 1)
}

func TestOverfitRuns(t *testing.T) {
	cfg := tinyCfg()
	inputs, labels := fixedBatch(2, 8, cfg.VocabSize)
	start, end, err := train.Overfit(cfg, inputs, labels, 50, 3e-3, 0.0, 42)
	if err != nil {
		t.Fatalf("Overfit: %v", err)
	}
	t.Logf("loss start=%.4f end=%.4f", start, end)
	if start <= 0 || end <= 0 {
		t.Errorf("losses should be positive, got start=%v end=%v", start, end)
	}
	if end >= start {
		t.Errorf("loss did not decrease over 50 steps: start=%v end=%v", start, end)
	}
}
```

- [ ] **Step 5: Run the smoke test — FAIL then implement then PASS**

Run: `cd train && GOMLX_BACKEND=go go test -tags noxla . -run TestOverfitRuns -v`
First expected: FAIL (package/functions undefined) → after Steps 2-3 are in place, run again.
Then expected: PASS — logs a start loss near `ln(32)≈3.47` and an end loss below it after 50 steps.
(If `v.NodeValue`, `shapes.Make`, `shapes.ConvertTo`, `model.ParamInitialSeed`, or `loss.SparseCategoricalCrossEntropyLogits` don't resolve against the pinned commit, fix at the single call site and report what changed — these are source-verified but not yet compiled. The label-shape `[B,T,1]` is mandatory; a `[B,T]` label panics in the loss.)

- [ ] **Step 6: Add train to the gate**

In `scripts/check.sh`, add `train` to the module loop:
```bash
for mod in backend model train app; do
```
Run: `make check`
Expected: backend, model, train, app build+test; boundary clean; `OK`.

- [ ] **Step 7: Commit**

```bash
git add train/ go.work scripts/check.sh
git commit -m "train: model variables + Overfit driver

New train module: ModelVars declares a trainable variable per model weight
and feeds model.Forward; Overfit runs AdamW + sparse cross-entropy on one
fixed batch via the GoMLX Trainer. Smoke test: loss decreases over 50 steps."
```

---

### Task 2: the overfit-a-batch gate

Harden the smoke test into the milestone's validation gate: enough steps to memorize the batch, with a seed-independent convergence assertion.

**Files:**
- Create: `train/overfit_test.go`

**Interfaces:**
- Consumes: `train.Overfit`, `tinyCfg`/`fixedBatch` patterns (re-declared here or shared — see Step 1).

- [ ] **Step 1: Write the overfit-a-batch gate test**

`train/overfit_test.go` (the helpers `tinyCfg`/`fixedBatch` live in `train_test.go` from Task 1, same `train_test` package, so reuse them directly):
```go
package train_test

import (
	"testing"

	"github.com/guygrigsby/lmkit-go/train"
)

// TestOverfitBatch is the Milestone-4 gate: a working loop drives the loss on one
// tiny fixed batch to ~0 (memorization). The assertion is seed-independent — it
// checks the loss fell substantially AND reached near zero, both of which a broken
// loop (no gradient flow, wrong loss, dead optimizer) cannot achieve, and which a
// correct loop achieves for any initialization. The absolute threshold below is set
// from the observed converged loss with margin (NOT tuned to a lucky seed/step).
func TestOverfitBatch(t *testing.T) {
	cfg := tinyCfg()
	inputs, labels := fixedBatch(2, 8, cfg.VocabSize)
	start, end, err := train.Overfit(cfg, inputs, labels, 2000, 3e-3, 0.0, 42)
	if err != nil {
		t.Fatalf("Overfit: %v", err)
	}
	t.Logf("loss start=%.4f end=%.6f", start, end)
	// Substantial fall: a correct loop reduces a memorizable batch far below init.
	if end >= 0.1*start {
		t.Errorf("loss did not fall substantially: end=%v, start=%v (want end < 0.1*start)", end, start)
	}
	// Near zero: the batch is memorized. THRESHOLD IS PROVISIONAL — set it from the
	// observed end loss with comfortable margin in Step 2.
	if end >= 0.05 {
		t.Errorf("batch not memorized: end loss=%v (want < 0.05)", end)
	}
}
```

- [ ] **Step 2: Run, observe convergence, set the threshold honestly**

Run: `cd train && GOMLX_BACKEND=go go test -tags noxla . -run TestOverfitBatch -v`
Read the logged `end` loss. A correct overfit of this tiny batch should drive `end` very low (expect well under 0.05, likely ~1e-3 or smaller after 2000 steps).
- If `end` is comfortably below 0.05 (e.g. < 0.005), the `< 0.05` threshold has ~10× margin — keep it; PASS.
- If `end` is near 0.05, do NOT loosen the threshold or change the seed. Instead give the loop more headroom for a true overfit: increase `steps` (e.g. 4000) and/or shrink the batch (`fixedBatch(1, 6, ...)`) and/or raise `lr` slightly — then re-observe and set the threshold to ~10× the new converged loss. A true overfit reaches ~0; if it plateaus high, that's a real wiring bug to debug (gradient flow, loss shape, optimizer), not a threshold to relax.
Record the final observed `end` and the chosen threshold in the test comment.

- [ ] **Step 3: Confirm seed-independence**

Run the test twice with different seeds to confirm the assertion holds regardless of init (the gate must not depend on a lucky seed):
```bash
cd train
GOMLX_BACKEND=go go test -tags noxla . -run TestOverfitBatch -v   # seed 42 (in code)
```
Then temporarily change the seed arg in the test to `7`, re-run, confirm PASS, and restore `42`. (Both must pass; if one fails, the threshold/steps are not honest — fix per Step 2, do not seed-shop.)

- [ ] **Step 4: Run the full gate**

Run: `cd train && GOMLX_BACKEND=go go test -tags noxla . -v` then `make check`
Expected: `TestOverfitRuns` + `TestOverfitBatch` pass; `make check` prints `OK`.

- [ ] **Step 5: Commit**

```bash
git add train/overfit_test.go
git commit -m "train: overfit-a-batch gate

Seed-independent convergence test: AdamW drives the loss on one tiny fixed
batch to ~0 (memorized). Threshold set from observed convergence with margin,
not seed-shopped. The validation-philosophy gate for the training loop."
```

---

## Self-Review

**Spec coverage** (against `2026-06-19-milestone-4-training-loop-design.md`):
- new `train` module in go.work → Task 1.
- `ModelVars` declares a variable per weight, feeds `model.Forward` → Task 1.
- next-token cross-entropy for integer labels → `loss.SparseCategoricalCrossEntropyLogits` with `[B,T,1]` labels (Task 1 driver). The spec's `Loss` is satisfied by the library function (YAGNI — no custom loss needed); noted.
- overfit driver → Task 1 `Overfit`.
- overfit-a-batch seed-independent gate → Task 2.
- boundary clean / make check / train in gate → Task 1 Step 6 + Task 2 Step 4.

**Out of scope, correctly absent:** real data/tokenizer, checkpoint/resume, LR schedule, grad clip/accum, metrics.jsonl, multi-GPU.

**Placeholder scan:** all Go is complete and uses source-verified training symbols; the one provisional value (the `< 0.05` overfit threshold) is explicitly set from observation in Task 2 Step 2 with a documented procedure — not a vague placeholder, and the seed-independence check (Step 3) guards against the M3 seed-shopping trap.

**Type consistency:** `ModelVars(scope, gr, cfg) model.Weights`, `Overfit(cfg, inputs, labels, steps, lr, wd, seed) (start, end float64, err error)`, the `tinyCfg`/`fixedBatch` helpers, and `model.Config`/`model.Weights`/`model.Forward` are used identically across `vars.go`, `train.go`, and both test files. Labels are `[B,T,1]` int everywhere (the loss's hard requirement).
</content>
