# Milestone 6 — production pretrain loop + lm-100m-en example (spec)

**Status:** ready for plan. Sub-project 6 of the context map (the `train` context's
production form) + an `examples/lm-100m-en` consumer. Decisions: ADR-0001..0009.
Mirrors the Python `lmkit.pretrain` + `lmkit.training` mechanics exactly.

## Goal

A **generic pretrain loop** (any model/config) that reproduces the lm-100m-en
baseline's training mechanics, **plus** an `examples/lm-100m-en` package holding all
lm-100m-specific wiring, runnable as **gated long integration tests** that drive the
short bf16 smoke and launch the full replica run on the CUDA GPU host toward val loss **1.7337**.

Clean split — exactly like the Python side (`lmkit.pretrain` is generic;
`training/lm-100m-en/` is the consumer):
- **`train`** = the reusable loop + `train.Config`. No lm-100m specifics.
- **`examples/lm-100m-en`** = the lm-100m config JSON + the run wiring + long
  integration tests. The only place lm-100m values live.

"Done" = the loop **validated** (fast CPU tests in `make check` + the example's short
CUDA GPU host bf16 integration test) and the full lm-100m-en run **launched durably**; the
multi-day val curve lands later.

## Part A — generic `train` loop (the library)

Extends the existing `train` module (which has `ModelVars` + the M4 overfit driver).
Mirrors `lmkit.pretrain`/`lmkit.training` field- and mechanic-for-mechanic. **No
lm-100m constants here** — everything comes from `train.Config` + `model.Config`.

### Config (struct + JSON, stdlib)
- **`train.Config`** mirrors the Python `TrainConfig`: `OutDir, DataDir, LR, MinLR,
  WarmupSteps, MaxSteps, DecayFrac, BatchSize, GradAccum, GradClip, WeightDecay,
  Beta1, Beta2, Dtype, Seed, EvalInterval, EvalIters, LogInterval, SaveInterval,
  SnapshotInterval, KeepLastSnapshots`. Loaded from JSON (`encoding/json`).
- A `LoadConfig(path)` helper. (The lm-100m JSON file itself lives in the example,
  Part B — not in `train`.)

### `train.Run(cfg, modelCfg, ...)` — hand-rolled loop
A Go `for`-loop over a compiled step graph (not GoMLX `Trainer.RunSteps`, which owns
the loop) so eval/checkpoint/metrics/per-step-LR interleave and global-norm clip can
be injected. Mechanics mirrored verbatim:
- **Resume** from `latest` (model + optimizer state + step + best_val); emit `resume`.
- **WSD `getLR(step, cfg)`** — exact `get_lr`: warmup → constant `lr` → cosine decay
  to `min_lr` over the final `decay_frac` of `max_steps` (`decay_frac=0` = pure stable
  trunk). Set optimizer LR per step.
- **Eval** every `EvalInterval`: mean CE over `EvalIters` val batches → `val_loss`,
  `val_perplexity = exp(min(vl,20))`; checkpoint `best` on improvement.
- **Checkpoint**: atomic `latest` (`SaveInterval`); `step_NNNNNN` snapshots
  (`SnapshotInterval`, prune to `KeepLastSnapshots`); `final` at end. Each = model
  vars + optimizer state + step + best_val + model config.
- **Grad accumulation**: `GradAccum` micro-batches, loss `/= GradAccum`, accumulate,
  one optimizer step (effective batch = `BatchSize*GradAccum`).
- **Global-norm grad clip** (`GradClip`): hand-rolled (GoMLX has only clip-by-value).
- **AdamW**: weight decay on 2-D+ params only, `β=(Beta1,Beta2)`.
- **metrics.jsonl** (`OutDir/metrics.jsonl`): exact event schema — `start`, `resume`,
  `eval` (`val_loss/val_perplexity/best_val/train_loss/lr/improved`), `train`
  (`train_loss/lr/grad_norm/tok_per_sec/step_time_ms/tokens_seen/tflops/peak_vram_gb`),
  `nan`, `sigterm`, `done`. Existing Grafana/ops CLI work unchanged. (No Aim.)
- **Robustness**: non-finite loss → save the diverged state to a separate `nan/` dir
  (NOT `latest/`, so the last good checkpoint stays resumable) + `nan` event + exit 2;
  SIGTERM/SIGINT → clean save + `sigterm` + exit 0. Durable/resumable (the WSD trunk
  runs indefinitely). (Revised from the original "save `latest`": overwriting `latest/`
  with NaN weights would poison the resume path; the supervisor leaves exit 2 for manual
  inspection.)
- **tflops** = `6*nParams*tok_per_sec/1e12`; **peak_vram_gb** via NVML
  (`github.com/NVIDIA/go-nvml`, dlopen; 0 where NVML absent).

### bf16 (confirmed: fp32 master + bf16 compute)
Mirror the autocast recipe: **variables fp32** (master; AdamW updates fp32),
**compute bf16 on CUDA** (cast weights→bf16 in the modelFn before matmuls; XLA's bf16
GEMM accumulates fp32; grads flow back fp32 through the cast), **fp32 on
CPU/SimpleGo**. Requires small **dtype-aware** edits in `model` (spans model + train):
- **RMSNorm** upcasts to fp32 for `mean(x²)`/`rsqrt`, downcasts (DESIGN's fp32-internal
  norm; M2 was fp32-only).
- **Attention softmax** upcasts to fp32.
- **Cross-entropy loss** in fp32 (upcast logits first).
- All else (projections, residuals, embeddings) bf16 on CUDA. Model parity tests
  (fp32) must stay green.

### Generic CLI — `cmd/lmkit train --config <file> [--model <file>]`
Loads `train.Config` + `model.Config`, builds the model (`ModelVars`) + train/val
`data.DataLoader` over `DataDir`'s `train_*.bin`/`val_*.bin`, calls `train.Run`.
Generic — no lm-100m hardcoding; the example points it at its config.

## Part B — `examples/lm-100m-en` (the consumer)

A new package (own module, ADR-0005; in `go.work`) holding **only** lm-100m-specific
things:
- **`config.json`** (+ model config) — the baseline values: `lr 4e-4, min_lr 4e-5,
  warmup 1000, max_steps 200000, decay_frac 0, batch 2, grad_accum 32, grad_clip 1.0,
  wd 0.1, β 0.9/0.95, dtype bfloat16, seed 1337, eval_interval 2000, eval_iters 100,
  log_interval 20, save_interval 200, snapshot_interval 25000, keep_last 3`; model:
  `hidden 768, n_layer 12, n_head 12, n_kv 4, head_dim 64, ffn 2048, block 2048,
  vocab 32000, rope_base 10000, rms_eps 1e-5, tie_embeddings`.
- **Run wiring** — a thin entry that points the generic loop at the lm-100m shards +
  config (or just documents the `lmkit train --config examples/lm-100m-en/config.json`
  invocation).
- **Long integration tests** (build tag `//go:build integration`, **excluded from
  `make check`**, run on demand with `go test -tags integration ./examples/lm-100m-en/`):
  - **short bf16 smoke (CUDA GPU host):** a few hundred steps of the real config —
    loss descending, bf16 path works, `peak_vram_gb` read, checkpoint→resume on CUDA.
  - **(optional) tiny CPU end-to-end:** a scaled-down lm-100m config on SimpleGo
    confirming the full wiring (model+data+loop+config) runs and checkpoints.
  - **the launch** is an operational step the integration test (or a documented
    command) kicks off durably; the multi-day 1.7337 curve is monitored separately.

This isolates every lm-100m constant + the heavy/slow runs from the fast generic
gate, and mirrors the Python `lm-100m-en/` consumer exactly.

## Out of scope (later)
- SFT, generation/sampling, KV cache, Aim/W&B, multi-GPU/distributed.
- safetensors/GGUF/HF export (`io` milestone).
- The actual 1.7337 *result* (arrives over days on the launched run).

## GoMLX-mechanics research (plan-gated, like M1/M4)
Pin the exact GoMLX `main` API before coding: grad accumulation across micro-batches;
global-L2-norm clip over grad nodes + conditional scale; per-step LR; `ConvertDType`
with a gradient (bf16 cast); checkpoint save/load of variables **+ optimizer state** +
step; reading the loss scalar. Approach: hand-rolled step graph (`graph.Gradient` over
the model vars) + optimizer application + GoMLX checkpoint.

## Testing
- **Fast (in `make check`):** `train` unit tests — `getLR`/WSD formula (warmup/stable/
  decay points), `Config` JSON round-trip, a tiny synthetic end-to-end loop smoke
  (few steps, loss decreases, **checkpoint→resume** round-trips, `metrics.jsonl`
  schema correct, eval runs, forced-NaN→exit2 + a SIGTERM save path). All fp32 on
  SimpleGo. No lm-100m specifics.
- **Long integration (gated, not in `make check`):** the `examples/lm-100m-en`
  build-tagged tests above — the short CUDA GPU host bf16 smoke + (optional) tiny CPU
  end-to-end + the launch.

## Done criteria
- [ ] `train.Config` (struct + JSON) + `LoadConfig`; generic, no lm-100m constants.
- [ ] `train.Run`: resume, WSD LR, AdamW (wd-on-2D), grad-accum, global-norm clip,
      eval, latest/best/snapshot checkpoint + prune, metrics.jsonl (exact schema),
      non-finite→exit2, SIGTERM→exit0.
- [ ] bf16 path: fp32 master + bf16 compute (CUDA) / fp32 (CPU); dtype-aware
      RMSNorm/softmax/loss; model parity tests still green.
- [ ] `cmd/lmkit train --config` builds model + loaders + runs (generic).
- [ ] `examples/lm-100m-en`: config JSON + build-tagged long integration tests; the
      short CUDA GPU host bf16 smoke green; full run launched durably.
- [ ] Fast `train` tests green in `make check`; integration tests excluded from it;
      boundary clean.

## Open questions to resolve in the plan
- The GoMLX API specifics above (grad-accum, global-norm clip, per-step LR, optimizer
  checkpoint, `ConvertDType` gradient).
- Where the bf16 cast lives + the exact dtype-aware upcast points so fp32 parity and
  bf16 both hold.
- Checkpoint layout (GoMLX checkpoint dir vs single-file `latest`) matching the
  baseline's `latest`/`best`/`step_NNNNNN` closely enough for the ops CLI.
- `examples/lm-100m-en` as its own module vs under `app`; how the integration test
  reaches the GPU host (cross-compile+rsync, per the established method) vs running there.
</content>
