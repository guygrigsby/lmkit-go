# Milestone 4 — training loop + overfit-a-batch (spec)

**Status:** ready for plan (self-reviewed; authored autonomously while the user is
away, per explicit delegation). Sub-project 4 of the context map. Decisions:
ADR-0001..0009.

## Goal

Wire the Milestone-3 `Forward` to trainable GoMLX variables + an AdamW optimizer
and a next-token cross-entropy loss, and prove the training loop drives the loss to
~0 on a single tiny batch — the **overfit-a-batch** gate from the project's
validation philosophy. This catches optimizer/autodiff/variable-wiring bugs before
any real run.

## In scope

- A new **`train` module** (own `go.mod`, ADR-0005) depending on `backend`,
  `model`, and the GoMLX training machinery (`ml/model`, `ml/train`,
  `ml/train/optimizer`, `ml/train/loss`/`losses`, `ml/dataset`). Added to `go.work`.
- **`ModelVars`** — declare one trainable variable per model weight (matching the
  `model.Weights`/`model.LayerWeights` shapes) in a `model.Scope` and return them as
  a `model.Weights` (graph Nodes), so the same `model.Forward` runs over variables.
  This is the seam Milestone 3 deliberately left.
- **`Loss`** — next-token cross-entropy: logits `[B,T,V]` vs integer target ids
  `[B,T]`.
- **The overfit driver** — assemble the model-over-variables `modelFn`, the loss,
  AdamW, and a one-batch dataset; run N steps; return the start/end loss.
- **The overfit-a-batch test** — the milestone gate (below).

## Out of scope (later milestones)

- Real data / tokenizer / corpus, shards, the `DataLoader` (data milestone).
- Checkpointing / resume, LR schedule (WSD/warmup), gradient clipping, gradient
  accumulation, `metrics.jsonl`, peak-VRAM, multi-GPU (the *full* train milestone).
- FlashAttention, bf16 training validation.

This milestone proves the loop **works and can overfit**; the production training
features come next.

## Boundary (ADR-0009)

`train` may import the GoMLX training vocabulary — `ml/model` (variables/scope),
`ml/train`, `ml/train/optimizer`, the loss package, `ml/dataset` — none of which are
the forbidden *runtime* packages (`"github.com/gomlx/compute"` root, `gomlx/backends`,
`go-xla`, `pjrt`). It obtains the backend via `gomlx.New().Compute()` (inferred type,
never naming `compute.Backend`), exactly the pattern the parity tests use. The
backend-boundary test (now runtime-specific and run uncached, `-count=1`) covers the
new module automatically.

## Components

- **`ModelVars(scope *model.Scope, cfg model.Config) model.Weights`** — declares (or
  fetches, on later calls) a trainable variable per weight with the right shape,
  random-initialized, and returns a `model.Weights` whose Nodes are the variables'
  graph values. Variable names mirror the weight structure (`embed`, `final_norm`,
  `layer{i}_attn_norm`, `layer{i}_Wq`, …). Used inside the training `modelFn` and
  reusable by later milestones. (Exact variable-declaration + node-fetch API is a
  plan-time detail; see open questions.)
- **`Loss(logits, targetIDs *g.Node) *g.Node`** — mean next-token cross-entropy over
  the vocabulary axis: `logits [B,T,V]`, `targetIDs [B,T]` int. Uses GoMLX's
  sparse-categorical-cross-entropy-from-logits if available; otherwise the verified
  composition (log-softmax over V, gather the target logit, negate, mean). Adapted to
  the trainer's `lossFn(labels, predictions []*g.Node) *g.Node` signature.
- **The driver** — builds a `modelFn` that calls `ModelVars(scope, cfg)` then
  `model.Forward(cfg, w, inputs[0], positions)` returning logits; wires
  `train.NewTrainer(be, store, modelFn, Loss, optimizer.Adam().WeightDecay(wd).LearningRate(lr).Done(), nil, nil)`
  with a one-batch in-memory dataset of `(inputs, targets)` int tensors; runs the
  loop; returns the loss at step 0 (or 1) and after N steps. (Trainer path is the
  M1-proven route; the plan picks Trainer vs a hand-rolled `g.Gradient` + optimizer
  `UpdateGraph` loop based on the verified API.)

## The overfit-a-batch gate

A tiny model (e.g. `NLayers=2`, `Hidden=16`, small vocab) and one fixed batch of
random token ids; train AdamW for enough steps to memorize it. Choose the batch
small enough (small `B`, `T`, `VocabSize`) that the model has ample capacity to
memorize it — the test must measure the *loop*, not fight model capacity.

Assert **seed-independently** the two things a working loop guarantees and a broken
one (no gradient flow, wrong loss, dead optimizer) cannot:
- the loss fell **substantially** from its random-init value (`~ln(VocabSize)`) —
  e.g. final `< 0.1 ×` initial; AND
- the loss reached **near zero** in absolute terms — i.e. the batch is memorized.

The concrete thresholds are set in the plan from the *observed* converged loss for a
correct loop at the chosen tiny size, with comfortable margin (do **not** tune them
to a lucky seed or step count — the M3 lesson). Overfitting one tiny batch converges
for any initialization, so a correctly-chosen threshold passes seed-independently.
If the chosen size can't drive absolute loss very low, shrink the batch rather than
loosen the claim — a true overfit reaches ~0. Document the reasoning in the test.

Next-token convention: inputs `= ids[:, :-1]`, targets `= ids[:, 1:]` (predict the
next token), so the objective is the real LM objective on the toy batch.

## Structure

```
train/
  go.mod          (module .../train; requires backend, model, gomlx ml/train etc.)
  vars.go         (ModelVars)
  loss.go         (Loss: next-token cross-entropy)
  train.go        (the overfit driver / step runner)
  overfit_test.go (the overfit-a-batch gate)
```

Each file one responsibility. SimpleGo (`-tags noxla`, `GOMLX_BACKEND=go`).

## Validation philosophy (DESIGN, non-negotiable)

- **Overfit-a-batch** is this milestone's gate — the loop must drive loss → ~0 on one
  tiny batch before any full run. This IS that test.
- Deterministic *structure* (fixed batch); the convergence assertion is
  seed-independent so it never depends on a lucky initialization.

## Done criteria

- [ ] `train` module in `go.work`; builds clean; boundary still clean (ADR-0009).
- [ ] `ModelVars` declares variables for the full `model.Weights` and feeds
      `model.Forward`.
- [ ] `Loss` computes next-token cross-entropy for integer targets.
- [ ] The overfit-a-batch test drives final loss `< 0.1` and `< 0.25 ×` initial,
      seed-independently, on SimpleGo.
- [ ] `make check` (train included) ends OK.

## Open questions to resolve during the plan (API specifics)

- Exact GoMLX `main` API for: declaring a trainable variable with a shape + random
  initializer in a `model.Scope` and obtaining its value as a `*graph.Node`; seeding
  the init RNG (not required for the seed-independent assertion, but useful for
  debugging).
- Whether GoMLX provides a sparse-categorical-cross-entropy-from-logits for integer
  labels and the exact label shape it expects (`[B,T]` vs `[B,T,1]`), or whether to
  compose it from log-softmax + gather.
- How to read the scalar loss after `RunSteps` (is the returned metric the mean
  loss?) vs evaluating the loss with a separate exec; and reading a trained
  variable's value back.
- Trainer/Loop path vs a hand-rolled `g.Gradient` + `optimizer.UpdateGraph` loop —
  pick whichever the verified API makes cleaner; the Trainer path is proven from M1's
  `FitConstant`.
</content>
