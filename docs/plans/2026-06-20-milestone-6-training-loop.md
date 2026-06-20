# Milestone 6 — production pretrain loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A generic `train` pretrain loop mirroring `lmkit.pretrain` (WSD LR, grad-accum, global-norm clip, AdamW, eval, checkpoint/resume, metrics.jsonl, bf16), plus an `examples/lm-100m-en` consumer (config + gated long integration tests) that launches the replica run.

**Architecture:** Hand-rolled Go loop over compiled `model.Exec` step graphs. Variables stay fp32 (master); compute casts to bf16 on CUDA via `ConvertDType` (differentiable). Grad accumulation uses on-device accumulator variables (mirroring GoMLX `accgradients.go`); the global-norm-clipped mean grads go to `optimizer.Adam()...Done().UpdateGraphWithGradients` (reusing verified AdamW). `checkpoint.Build(store)` persists weights + Adam moments + step. All lm-100m specifics live in `examples/lm-100m-en` behind `//go:build integration`.

**Tech Stack:** Go 1.26, GoMLX `main` (`516689cbe913`); reuses `model` + `data`; `github.com/NVIDIA/go-nvml` (peak-VRAM, dlopen); `encoding/json` (config).

## Global Constraints

- **Generic `train`, no lm-100m constants** — all baseline values live in `examples/lm-100m-en` (ADR per the M6 spec).
- **Runtime boundary (ADR-0009):** `train`/`examples` may use `core/graph` + `ml/...` op/training vocabulary; backend via `gomlx.New().Compute()` (inferred — never name `compute.Backend`). Boundary test (`-count=1`) covers them.
- **bf16 = fp32 master + bf16 compute** on CUDA (cast weights→bf16 in-graph; XLA GEMM accumulates fp32; grads flow back fp32 through `ConvertDType`'s VJP); **fp32 on CPU/SimpleGo**. Loss computed in fp32.
- **Metrics:** exact `metrics.jsonl` event schema from `lmkit.pretrain` (below). No Aim.
- **Robustness:** non-finite loss → save latest + `nan` event + exit 2; SIGTERM/SIGINT → save latest + `sigterm` + exit 0.
- **Checkpoint Save() is NOT atomic in GoMLX** — wrap saves temp-dir + rename, or accept `Backup()` snapshots; the plan wraps.
- **Per-step LR:** set the lr *variable* value each step (SetParam only re-applies on graph rebuild).
- Commits: terse, verb-first, no dashes, no Claude/Anthropic attribution.

## Verified GoMLX API (pinned `516689cbe913`) — the hand-rolled training step

```go
// variable write-back (THE primitive): any var touched with SetNodeValue in the graph
// is materialized + persisted by a plain model.Exec after the run.
v.SetNodeValue(node)                 // ml/model/variable.go
// grads in trainable-variable order:
grads := scope.BuildTrainableVariablesGradientsGraph(loss) // []*graph.Node
// apply externally-computed (accumulated+clipped) grads with verified AdamW:
opt := optimizer.Adam().WeightDecay(wd).Done()             // ml/train/optimizer
opt.(optimizer.OptimizeWithGradients).UpdateGraphWithGradients(scope, grads, lossDType) // accgradients.go:21
// grad accumulation idiom (mirror ml/train/accgradients.go): acc_grads/<path> vars,
//   acc.SetNodeValue(Add(acc.NodeValue(g), grad)); on apply: mean -> clip -> Update -> zero (Zeros(g,shape)).
// dtype cast with gradient:
g.ConvertDType(x, dtypes.BFloat16)   // core/graph/ops.go; VJP casts back (rev_autodiff.go)
// dtypes: import "github.com/gomlx/compute/dtypes" -> dtypes.Float32, dtypes.BFloat16
// checkpoint (saves ALL store vars incl Adam moments + global_step; auto-resumes):
h, _ := checkpoint.Build(store).Dir(dir).Keep(n).Done()    // ml/model/checkpoint; h.Save()
// global step variable: optimizer.GetGlobalStep(store) (int64 var "global_step")
// lr variable: set its value each step (path from optimizer.LearningRateVar)
// loss readback: shapes.ConvertTo[float64](out.Value())  ("github.com/gomlx/compute/shapes")
```
Exec returns the loss output AND performs the var-update side effects in one `Call`. No `train.Trainer`/`Loop`.

---

### Task 1: `train.Config` (struct + JSON) + WSD `getLR`

Pure config + schedule — fast, no GoMLX.

**Files:** Modify `train/config.go` (create if absent); Create `train/config_test.go`.

**Interfaces:**
- Produces: `train.Config` struct; `train.LoadConfig(path string) (Config, error)`; `train.getLR(step int, c Config) float64` (unexported; tested in-package).

- [ ] **Step 1: failing tests**

`train/config_test.go`:
```go
package train

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadConfig(t *testing.T) {
	p := filepath.Join(t.TempDir(), "c.json")
	os.WriteFile(p, []byte(`{"lr":4e-4,"min_lr":4e-5,"warmup_steps":1000,"max_steps":200000,"decay_frac":0.0,"batch_size":2,"grad_accum":32,"grad_clip":1.0,"weight_decay":0.1,"beta1":0.9,"beta2":0.95,"dtype":"bfloat16","seed":1337,"eval_interval":2000,"eval_iters":100,"log_interval":20,"save_interval":200,"snapshot_interval":25000,"keep_last_snapshots":3,"out_dir":"ckpt","data_dir":"data"}`), 0o644)
	c, err := LoadConfig(p)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if c.LR != 4e-4 || c.GradAccum != 32 || c.Dtype != "bfloat16" || c.MaxSteps != 200000 {
		t.Errorf("parsed wrong: %+v", c)
	}
}

func TestGetLR_WSD(t *testing.T) {
	c := Config{LR: 4e-4, MinLR: 4e-5, WarmupSteps: 1000, MaxSteps: 200000, DecayFrac: 0.0}
	// warmup: linear from 0
	if got := getLR(500, c); math.Abs(got-2e-4) > 1e-12 {
		t.Errorf("warmup mid = %v want 2e-4", got)
	}
	if got := getLR(0, c); got != 0 {
		t.Errorf("step0 = %v want 0", got)
	}
	// stable (decay_frac=0): constant lr after warmup, forever
	if got := getLR(1000, c); got != 4e-4 {
		t.Errorf("stable start = %v want 4e-4", got)
	}
	if got := getLR(199999, c); got != 4e-4 {
		t.Errorf("stable late = %v want 4e-4 (decay_frac=0)", got)
	}
	// with a decay tail: ends at min_lr
	c2 := c
	c2.DecayFrac = 0.5
	if got := getLR(c2.MaxSteps, c2); math.Abs(got-c2.MinLR) > 1e-12 {
		t.Errorf("decay end = %v want min_lr", got)
	}
}
```
Run: `cd train && go test ./... -run 'TestLoadConfig|TestGetLR' -v` → FAIL.

- [ ] **Step 2: implement**

`train/config.go`:
```go
package train

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// Config mirrors the Python lmkit TrainConfig. Loaded from JSON; no lm-100m
// constants live here (those are in examples/lm-100m-en).
type Config struct {
	OutDir            string  `json:"out_dir"`
	DataDir           string  `json:"data_dir"`
	LR                float64 `json:"lr"`
	MinLR             float64 `json:"min_lr"`
	WarmupSteps       int     `json:"warmup_steps"`
	MaxSteps          int     `json:"max_steps"`
	DecayFrac         float64 `json:"decay_frac"`
	BatchSize         int     `json:"batch_size"`
	GradAccum         int     `json:"grad_accum"`
	GradClip          float64 `json:"grad_clip"`
	WeightDecay       float64 `json:"weight_decay"`
	Beta1             float64 `json:"beta1"`
	Beta2             float64 `json:"beta2"`
	Dtype             string  `json:"dtype"` // "bfloat16" | "float32"
	Seed              int64   `json:"seed"`
	EvalInterval      int     `json:"eval_interval"`
	EvalIters         int     `json:"eval_iters"`
	LogInterval       int     `json:"log_interval"`
	SaveInterval      int     `json:"save_interval"`
	SnapshotInterval  int     `json:"snapshot_interval"`
	KeepLastSnapshots int     `json:"keep_last_snapshots"`
}

// LoadConfig reads a JSON train config.
func LoadConfig(path string) (Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Config{}, err
	}
	var c Config
	if err := json.Unmarshal(b, &c); err != nil {
		return Config{}, fmt.Errorf("train: config %s: %w", path, err)
	}
	return c, nil
}

// getLR is the WSD schedule (exact port of lmkit.training.get_lr): linear warmup →
// constant lr → cosine decay to min_lr over the final decay_frac of max_steps.
func getLR(step int, c Config) float64 {
	if step < c.WarmupSteps {
		return c.LR * float64(step) / math.Max(1, float64(c.WarmupSteps))
	}
	decaySteps := int(c.DecayFrac * float64(c.MaxSteps))
	decayStart := c.MaxSteps - decaySteps
	if step < decayStart {
		return c.LR
	}
	if step >= c.MaxSteps {
		return c.MinLR
	}
	progress := float64(step-decayStart) / math.Max(1, float64(decaySteps))
	coeff := 0.5 * (1.0 + math.Cos(math.Pi*progress))
	return c.MinLR + coeff*(c.LR-c.MinLR)
}

var _ = json.Marshal // keep encoding/json imported if unused elsewhere
```
Run → PASS. Commit:
```bash
git add train/config.go train/config_test.go
git commit -m "train: Config (JSON) + WSD getLR schedule"
```

---

### Task 2: dtype-aware model ops (fp32-internal norm/softmax) for bf16

Make RMSNorm + attention numerically fp32-internal so bf16 training matches the baseline. Keep fp32 parity green.

**Files:** Modify `model/rmsnorm.go`, `model/attention.go`; Create `model/dtype_test.go`.

**Interfaces:** unchanged signatures; RMSNorm/attention internally upcast to fp32 for the sensitive reductions regardless of input dtype.

- [ ] **Step 1: failing dtype test**

`model/dtype_test.go`:
```go
package model_test

import (
	"math"
	"testing"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/model"
)

// RMSNorm on bf16 input must stay close to the fp32 result (fp32-internal reduction).
func TestRMSNormBF16CloseToFP32(t *testing.T) {
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	h := 16
	xData := make([]float32, h)
	sc := make([]float32, h)
	for i := range xData {
		xData[i] = float32(i%7) - 3.0
		sc[i] = 1.0
	}
	run := func(dt dtypes.DType) []float32 {
		exec := g.MustNewExec(be.Compute(), func(x, scale *g.Node) *g.Node {
			if dt == dtypes.BFloat16 {
				x = g.ConvertDType(x, dtypes.BFloat16)
				scale = g.ConvertDType(scale, dtypes.BFloat16)
			}
			out := model.RMSNorm(x, scale, 1e-5)
			return g.ConvertDType(out, dtypes.Float32)
		})
		out := exec.MustCall1(
			tensors.FromFlatDataAndDimensions(xData, 1, 1, h),
			tensors.FromFlatDataAndDimensions(sc, h))
		return tensors.MustCopyFlatData[float32](out)
	}
	f32, bf16 := run(dtypes.Float32), run(dtypes.BFloat16)
	for i := range f32 {
		if d := math.Abs(float64(f32[i] - bf16[i])); d > 5e-2 {
			t.Errorf("elem %d: fp32 %v vs bf16 %v (diff %v > 5e-2)", i, f32[i], bf16[i], d)
		}
	}
}
```
Run: `cd model && GOMLX_BACKEND=go go test -tags noxla . -run TestRMSNormBF16 -v` → FAIL or imprecise until RMSNorm upcasts.

- [ ] **Step 2: make RMSNorm fp32-internal**

In `model/rmsnorm.go`, upcast to fp32 for the reduction, downcast back to the input dtype before scaling:
```go
package model

import (
	"github.com/gomlx/compute/dtypes"
	g "github.com/gomlx/gomlx/core/graph"
)

// RMSNorm normalizes x [B,T,H] over its last axis and scales by scale [H].
// The mean/rsqrt run in fp32 (upcast) even for bf16 inputs, matching the baseline's
// fp32-internal norm; the normalized result is returned in x's dtype.
func RMSNorm(x, scale *g.Node, eps float32) *g.Node {
	inDT := x.DType()
	xf := x
	if inDT != dtypes.Float32 {
		xf = g.ConvertDType(x, dtypes.Float32)
	}
	ms := g.ReduceAndKeep(g.Mul(xf, xf), g.ReduceMean, -1) // [B,T,1] fp32
	inv := g.Reciprocal(g.Sqrt(g.AddScalar(ms, eps)))      // Rsqrt has no VJP (M4)
	normed := g.Mul(xf, inv)                               // fp32
	if inDT != dtypes.Float32 {
		normed = g.ConvertDType(normed, inDT)
	}
	return g.Mul(normed, g.Reshape(scale, 1, 1, -1))
}
```
Run → PASS (bf16 within 5e-2 of fp32). Also re-run the existing fp32 parity: `cd model && GOMLX_BACKEND=go go test -tags noxla . -run 'TestRMSNormParity|TestForwardParity' -v` → still PASS (fp32 path unchanged: `inDT==Float32` skips casts).

- [ ] **Step 3: make attention softmax fp32-internal**

In `model/attention.go`, where the softmax over scores happens, upcast scores to fp32 for the softmax and back:
```go
	// scores computed as before (bf16 on CUDA); softmax in fp32 for stability/fidelity.
	sdt := scores.DType()
	sf := scores
	if sdt != dtypes.Float32 {
		sf = g.ConvertDType(scores, dtypes.Float32)
	}
	probs := g.Softmax(sf, -1)
	if sdt != dtypes.Float32 {
		probs = g.ConvertDType(probs, sdt)
	}
```
(Add the `dtypes` import to attention.go.) Run `cd model && GOMLX_BACKEND=go go test -tags noxla ./... ` → all 7 parity + paritytest tests still PASS (fp32 unaffected). Commit:
```bash
git add model/rmsnorm.go model/attention.go model/dtype_test.go
git commit -m "model: fp32-internal RMSNorm + softmax for bf16 fidelity"
```

---

### Task 3: training-step machinery (grad-accum + global-norm clip + AdamW)

The on-device accumulate/apply step graphs reusing `UpdateGraphWithGradients`, driven from Go. The riskiest task — validated by a CPU "loss descends" test through the production path.

**Files:** Create `train/step.go`, `train/clip.go`, `train/step_test.go`.

**Interfaces:**
- Consumes: `model.ModelVars`, `model.Forward`, `model.Config`, `data` (in the test, synthetic).
- Produces: `train.modelLoss(scope, gr, mcfg, x, y, computeDT) *g.Node`; `train.clipByGlobalNorm(grads []*g.Node, clip float64) []*g.Node`; a `train.stepper` that builds the accumulate-only + accumulate-then-apply Execs and runs one optimizer step over `GradAccum` micro-batches, returning the loss.

- [ ] **Step 1: implement the global-norm clip (with a unit test)**

`train/clip.go`:
```go
package train

import g "github.com/gomlx/gomlx/core/graph"

// clipByGlobalNorm scales grads so their global L2 norm <= clip (no-op if already
// under). Mirrors torch.nn.utils.clip_grad_norm_. GoMLX has no built-in for this.
func clipByGlobalNorm(grads []*g.Node, clip float64) []*g.Node {
	if len(grads) == 0 {
		return grads
	}
	gr := grads[0].Graph()
	total := g.ReduceAllSum(g.Mul(grads[0], grads[0]))
	for _, gd := range grads[1:] {
		total = g.Add(total, g.ReduceAllSum(g.Mul(gd, gd)))
	}
	norm := g.Sqrt(total)
	// scale = min(1, clip/(norm+eps))
	scale := g.MinScalar(g.DivScalar(g.AddScalar(norm, 1e-6), 1.0), 1.0) // placeholder; replaced below
	_ = scale
	scaleNode := g.MinScalar(divConst(gr, clip, g.AddScalar(norm, 1e-6)), 1.0)
	out := make([]*g.Node, len(grads))
	for i, gd := range grads {
		out[i] = g.Mul(gd, scaleNode)
	}
	return out
}

// divConst returns clip / denom as a node.
func divConst(gr *g.Graph, clip float64, denom *g.Node) *g.Node {
	return g.Div(g.Const(gr, float32(clip)), denom)
}
```
(Note: simplify to `scaleNode := g.MinScalar(g.Div(g.Const(gr, float32(clip)), g.AddScalar(norm, 1e-6)), 1.0)` and drop the placeholder line; confirm `g.Div`, `g.MinScalar`, `g.Const(gr, float32)` resolve — the research confirmed `MinScalar`, `Div`/`DivScalar`, `Const`.)

`train/clip_test.go`:
```go
package train

import (
	"math"
	"testing"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
)

func TestClipByGlobalNorm(t *testing.T) {
	be, _ := gomlx.New()
	// grad = [3,4] (norm 5); clip 1 -> scaled to norm 1 -> [0.6,0.8].
	exec := g.MustNewExec(be.Compute(), func(in *g.Node) *g.Node {
		return clipByGlobalNorm([]*g.Node{in}, 1.0)[0]
	})
	out := exec.MustCall1(tensors.FromFlatDataAndDimensions([]float32{3, 4}, 2))
	got := tensors.MustCopyFlatData[float32](out)
	if math.Abs(float64(got[0]-0.6)) > 1e-4 || math.Abs(float64(got[1]-0.8)) > 1e-4 {
		t.Errorf("clipped = %v want ~[0.6 0.8]", got)
	}
	// under the clip: unchanged
	exec2 := g.MustNewExec(be.Compute(), func(in *g.Node) *g.Node {
		return clipByGlobalNorm([]*g.Node{in}, 100.0)[0]
	})
	out2 := tensors.MustCopyFlatData[float32](exec2.MustCall1(tensors.FromFlatDataAndDimensions([]float32{3, 4}, 2)))
	if math.Abs(float64(out2[0]-3)) > 1e-4 {
		t.Errorf("under-clip changed: %v", out2)
	}
}
```
Run: `cd train && GOMLX_BACKEND=go go test -tags noxla . -run TestClipByGlobalNorm -v` → implement until PASS (norm 5 → [0.6,0.8]; under-clip unchanged).

- [ ] **Step 2: implement the loss builder + stepper**

`train/step.go` — builds the per-micro-batch loss and the accumulate/apply step graphs. **Mirror `ml/train/accgradients.go`** for the accumulator-variable iteration (read it at the pinned commit; the exact internal helper `iterTrainableAndAccumulatorVariables` is the pattern to replicate: for each trainable var in `scope` order, an `acc_grads/<path>` accumulator of the same shape):
```go
package train

import (
	"github.com/gomlx/compute/dtypes"
	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/optimizer"

	lmodel "github.com/guygrigsby/lmkit-go/model"
)

// modelLoss builds the fp32 next-token CE loss for one micro-batch. Variables are
// fp32 (master); on CUDA the weights are cast to computeDT (bf16) for the matmuls,
// logits upcast to fp32 for the loss.
func modelLoss(scope *model.Scope, gr *g.Graph, mcfg lmodel.Config, x, y *g.Node, computeDT dtypes.DType, positions []int) *g.Node {
	w := lmodel.ModelVars(scope, gr, mcfg) // fp32 vars
	if computeDT != dtypes.Float32 {
		w = castWeights(w, computeDT) // ConvertDType each weight node
	}
	logits := lmodel.Forward(mcfg, w, x, positions) // computeDT on CUDA
	logits = g.ConvertDType(logits, dtypes.Float32)  // loss in fp32
	return loss.SparseCategoricalCrossEntropyLogits([]*g.Node{y}, []*g.Node{logits})
}

// castWeights returns a copy of w with every weight node cast to dt (for bf16 compute).
func castWeights(w lmodel.Weights, dt dtypes.DType) lmodel.Weights {
	c := func(n *g.Node) *g.Node { return g.ConvertDType(n, dt) }
	out := lmodel.Weights{Embed: c(w.Embed), FinalNorm: c(w.FinalNorm)}
	for _, lw := range w.Layers {
		out.Layers = append(out.Layers, lmodel.LayerWeights{
			AttnNorm: c(lw.AttnNorm), Wq: c(lw.Wq), Wk: c(lw.Wk), Wv: c(lw.Wv), Wo: c(lw.Wo),
			FFNNorm: c(lw.FFNNorm), Wgate: c(lw.Wgate), Wup: c(lw.Wup), Wdown: c(lw.Wdown),
		})
	}
	return out
}
```
Then the stepper: an `accumulate-only` graph fn (`grads := scope.BuildTrainableVariablesGradientsGraph(loss)`; add each into its `acc_grads/` accumulator via `acc.SetNodeValue(Add(acc.NodeValue(gr), grads[i]))`; return loss) and an `accumulate-then-apply` graph fn (same accumulate; then read accumulators, divide by `GradAccum`, `clipByGlobalNorm`, `opt.(optimizer.OptimizeWithGradients).UpdateGraphWithGradients(scope, clipped, dtypes.Float32)`, then `acc.SetNodeValue(Zeros(gr, acc.Shape()))`; return loss). Compile each once with `model.NewExec`/`g.MustNewExec` over the store backend; the Go driver runs accumulate (GradAccum-1)× then apply 1×, reading the loss from the apply call. (The exact accumulator-iteration + `UpdateGraphWithGradients` call mirror `accgradients.go:81-150`; read that file at the pinned commit and replicate — this is the one spot to verify against source, like prior milestones.)

- [ ] **Step 3: failing "production step descends" test**

`train/step_test.go` — a tiny model + one fixed batch; run ~200 optimizer steps via the stepper; assert the loss falls substantially (the production path trains, end-to-end through grad-accum+clip+AdamW). (Reuse a tiny `model.Config` like M4's; build a synthetic `(x,y)` batch with `y` shaped `[B,T,1]` int32.) Assert `endLoss < 0.3*startLoss` (seed-independent; do not seed-shop). Run on CPU SimpleGo (fp32, `computeDT=Float32`). Run: `cd train && GOMLX_BACKEND=go go test -tags noxla . -run TestStepDescends -v` → implement stepper until PASS.

- [ ] **Step 4: commit**

```bash
git add train/step.go train/clip.go train/step_test.go train/clip_test.go
git commit -m "train: grad-accum + global-norm clip + AdamW step (apply via UpdateGraphWithGradients)"
```

---

### Task 4: `train.Run` — the full loop (resume, eval, checkpoint, metrics, signals)

**Files:** Create `train/run.go`, `train/metrics.go`, `train/checkpoint.go`, `train/run_test.go`.

**Interfaces:**
- Consumes: `Config`, `getLR`, the stepper (Task 3), `data.Loader`, `model.Config`, GoMLX `checkpoint`.
- Produces: `train.Run(cfg Config, mcfg model.Config, trainLoader, valLoader *data.Loader) (int, error)` (returns process exit code: 0 done/sigterm, 2 non-finite).

- [ ] **Step 1: metrics + checkpoint helpers**

`train/metrics.go`: `emit(path, map[string]any)` (append one JSON line with a `ts`); `peakVRAMGB() float64` via `go-nvml` (returns 0 if NVML unavailable); `tflops(nParams int, tokPerSec float64) float64 = 6*nParams*tokPerSec/1e12`. (Add `github.com/NVIDIA/go-nvml/pkg/nvml`; on non-CUDA, `nvml.Init()` errors → return 0.)

`train/checkpoint.go`: thin wrappers over GoMLX `checkpoint` — `saveCheckpoint(store, dir)` that saves to a temp dir then `os.Rename` into `dir` (GoMLX `Save()` is not atomic — wrap it), `resumeIfPresent(store, dir)` (uses `checkpoint.Build(store).Dir(dir).Done()` to auto-load + returns the restored step from `optimizer.GetGlobalStep`), `pruneSnapshots(outDir, keep)`.

- [ ] **Step 2: implement `Run`**

`train/run.go` — the loop mirroring `lmkit.pretrain.run` exactly:
- `signal.Notify` SIGTERM/SIGINT → a `stop` flag.
- build the store + model vars + optimizer (`optimizer.Adam().WeightDecay(cfg.WeightDecay).Betas(cfg.Beta1,cfg.Beta2).Done()`); `resumeIfPresent` → `step`, `bestVal`.
- emit `start` (or `resume`).
- `for step < cfg.MaxSteps`: if stop → save latest + `sigterm` → return 0; set lr variable = `getLR(step,cfg)`; if `step%EvalInterval==0` → eval (mean CE over `EvalIters` val batches via a no-grad loss exec) → `val_loss`/`val_perplexity`/best (+ save `best` on improvement) → emit `eval`; if `step%SaveInterval==0 && step>0` → save `latest`; if `SnapshotInterval` cadence → save `step_NNNNNN` + prune; run one optimizer step (Task 3 stepper, `GradAccum` micro-batches from `trainLoader`); if loss non-finite → save latest + emit `nan` → return 2; if `step%LogInterval==0` → emit `train` with the full metric set (`train_loss/lr/grad_norm/tok_per_sec/step_time_ms/tokens_seen/tflops/peak_vram_gb`); `step++`.
- after loop: save `final` + `latest`, emit `done`, return 0.

(grad_norm for the metric: have the apply step also return the pre-clip global norm; small add.)

- [ ] **Step 3: CPU smoke test**

`train/run_test.go` (`//go:build !integration` or plain — it's fast): tiny config, tiny synthetic shards (write `train_0.bin`/`val_0.bin` via the `data` test helper pattern, or point at in-memory), `MaxSteps` ~30, `EvalInterval` 10, `SaveInterval` 10. Assert: `Run` returns 0; `metrics.jsonl` exists with `start`,`eval`,`train`,`done` events and the right fields; train loss at the last `train` event < the first; a **resume** test (run 15 steps, `Run` again with same OutDir → resumes at 15, finishes) gives a checkpoint round-trip; a **non-finite** path (inject via a config that NaNs, or a unit on the nan branch) returns 2; a **SIGTERM** path (send the signal mid-run in a goroutine) returns 0 and leaves `latest`. Run: `cd train && GOMLX_BACKEND=go go test -tags noxla . -v`.

- [ ] **Step 4: extend gate + commit**

`make check` (train already in `scripts/check.sh`). Commit:
```bash
git add train/run.go train/metrics.go train/checkpoint.go train/run_test.go train/go.mod train/go.sum
git commit -m "train: Run loop (resume, eval, checkpoint, metrics.jsonl, signals)"
```

---

### Task 5: `cmd/lmkit train` (generic CLI)

**Files:** Modify `app/cmd/lmkit/main.go` (add the `train` subcommand); Create `app/cmd/lmkit/train.go`.

**Interfaces:** `lmkit train --config <train.json> --model <model.json> [--data <dir>]` → loads configs, builds model + `data.Loader`s over `--data`/`cfg.DataDir`'s `train_*.bin`/`val_*.bin`, calls `train.Run`, exits with its code.

- [ ] **Step 1: implement the subcommand**

`app/cmd/lmkit/train.go`: parse flags; `train.LoadConfig`; load `model.Config` (its own small JSON loader or reuse a `model.LoadConfig`); glob `filepath.Glob(dataDir+"/train_*.bin")` and `val_*.bin`; `data.New(...)` train + val loaders (BlockSize from model block, BatchSize from cfg); `os.Exit(train.Run(...))`. Wire it into `main.go`'s subcommand switch next to `quickstart`.

- [ ] **Step 2: build + smoke**

Run: `go build -tags noxla ./app/cmd/lmkit && ./lmkit train --help` (or a no-args usage). A real run is exercised by the example's integration test (Task 6); here just confirm it builds + dispatches. `make check` green.

- [ ] **Step 3: commit**

```bash
git add app/cmd/lmkit/
git commit -m "app: lmkit train subcommand (generic, config-driven)"
```

---

### Task 6: `examples/lm-100m-en` (config + gated long integration tests)

All lm-100m specifics + the heavy runs, isolated behind `//go:build integration`.

**Files:** Create `examples/lm-100m-en/go.mod`; Modify `go.work`; Create `examples/lm-100m-en/config.json`, `examples/lm-100m-en/model.json`, `examples/lm-100m-en/run_integration_test.go`, `examples/lm-100m-en/README.md`.

**Interfaces:** the build-tagged tests drive the generic `train.Run`/CLI with the lm-100m config against real shards on a CUDA host.

- [ ] **Step 1: module + configs**

`examples/lm-100m-en/go.mod` (module `github.com/guygrigsby/lmkit-go/examples/lm-100m-en`); add to `go.work`. `config.json` (the baseline TrainConfig values) + `model.json` (the lm-100m ModelConfig: `hidden 768, n_layer 12, n_head 12, n_kv_heads 4, head_dim 64, ffn 2048, block 2048, vocab 32000, rope_base 10000, rms_eps 1e-5`). `DataDir`/`OutDir` are placeholders set via the test/CLI flags (no hard-coded host paths).

- [ ] **Step 2: gated integration test**

`examples/lm-100m-en/run_integration_test.go` (first line `//go:build integration`): a test that builds the lm-100m model + loaders over `$LMKIT_DATA` (env, skip if unset) and runs `train.Run` for a SHORT step budget (e.g. `MaxSteps=200`, eval once) — asserts it descends, writes `metrics.jsonl`, checkpoints, and (on CUDA) reports non-zero `peak_vram_gb`. Skipped in normal CI (no `integration` tag); run on demand:
```bash
# on a CUDA host with the shards at $LMKIT_DATA:
GOMLX_BACKEND=xla:cuda LMKIT_DATA=<shard-dir> go test -tags integration ./examples/lm-100m-en/ -run TestShortBF16 -v
```

- [ ] **Step 3: README + launch doc**

`examples/lm-100m-en/README.md`: how to launch the full durable run (`lmkit train --config examples/lm-100m-en/config.json --model examples/lm-100m-en/model.json --data <shard-dir>` under a process supervisor on a CUDA host; resumes from `latest`), and that the val curve toward 1.7337 is monitored via `metrics.jsonl`. No lab hostnames — generic `$GPU_HOST`/`$LMKIT_DATA` placeholders.

- [ ] **Step 4: verify excluded from `make check`, commit**

Run: `make check` (the `examples/lm-100m-en` integration tests must NOT run — they're behind the `integration` tag; the module still builds). Confirm `go build ./examples/lm-100m-en/` is clean.
```bash
git add examples/ go.work
git commit -m "examples: lm-100m-en config + gated long integration tests + launch doc"
```

- [ ] **Step 5 (operational, not a code gate): launch the run**

On a CUDA host with the shards: build the CUDA binary (cross-compile per the established artifact-deploy method, or build on the host), and launch `lmkit train` under a supervisor (durable/resumable). Record the launch + first `metrics.jsonl` lines in the example README's run log. The multi-day val curve toward 1.7337 is monitored separately.

---

## Self-Review

**Spec coverage** (against `2026-06-20-milestone-6-training-loop-design.md`):
- `train.Config` + JSON + `LoadConfig`, generic → Task 1.
- WSD `getLR` → Task 1. AdamW (wd, β via optimizer) → Task 3/4. grad-accum + global-norm clip → Task 3. eval + checkpoint(latest/best/snapshot+prune+atomic)+resume + metrics.jsonl(schema) + nan→exit2 + sigterm→exit0 → Task 4. bf16 fp32-master+bf16-compute + dtype-aware norm/softmax/loss → Task 2 (+ casts in Task 3). generic CLI → Task 5. examples/lm-100m-en config + gated integration tests + launch → Task 6.
- Boundary clean (no `compute.Backend` named; backend via `gomlx.New().Compute()`) → all train/example tasks.

**Out of scope, correctly absent:** SFT, generation/KV-cache, Aim, multi-GPU, io/export, the multi-day 1.7337 result.

**Placeholder scan:** complete code for config/getLR/clip/dtype-aware ops/CLI; the two intricate spots (the accumulator-variable iteration + `UpdateGraphWithGradients` call in Task 3, and the `Run` loop body in Task 4) are specified with the exact verified primitives + a cite to `accgradients.go`/`lmkit.pretrain` to mirror — precise source-read points (like M1/M4), not vague placeholders. The `train/clip.go` snippet has a noted simplification to apply (drop the placeholder line; use `g.Div(Const, AddScalar(norm,eps))`).

**Type consistency:** `train.Config` fields, `getLR(step,Config)`, `modelLoss`/`castWeights`/`clipByGlobalNorm`/`Run` signatures, and reuse of `model.ModelVars`/`model.Forward`/`model.Weights`/`data.Loader` are consistent across files. bf16 cast via `g.ConvertDType` + `dtypes.{Float32,BFloat16}` used uniformly. Labels `[B,T,1]` int for the loss (from M5's loader `y` reshaped) — the loader yields `[B,T]`; the step reshapes `y` to `[B,T,1]` before the loss (note in Task 3).
