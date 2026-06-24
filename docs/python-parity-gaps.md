# lmkit-go — feature parity gaps vs the Python lmkit / PyTorch ecosystem

What lmkit-go must add to mirror the Python `lmkit` surface (its stated parity
target) and, beyond that, the broader PyTorch/HF ecosystem. Grounded in the
`lmkit` package layout (`pretrain`, `training`, `shard`, `tokenizer`, `sft`,
`eval/{gen,judge}`, `observability`, `push`, `cli`), the `lm-100m-en` config, and
the mid-2026 Go-vs-Python ecosystem research.

**Status legend:** ✅ built · 🟡 partial · ⬜ missing · 🧊 deferred by ADR (intentional)

## Built so far
- ✅ **backend** — XLA via GoMLX, CUDA-validated; autodiff; AdamW; bf16 dtype.
- ✅ **model** — Llama blocks (RMSNorm, RoPE, GQA, SwiGLU, tied emb) + full decoder
  forward, all parity-verified vs PyTorch. cuDNN flash attention on the bf16 CUDA
  path. Config matches lm-100m-en exactly.
- ✅ **train** — full production loop: gradient accumulation, global-norm clip, WSD
  schedule, AdamW (betas + decoupled decay on rank≥2 params), bf16 on CUDA, periodic
  eval, checkpoint/resume, JSONL metrics, signal handling, optional gradient
  checkpointing.
- ✅ **data / tokenizer / CLI** — mmap `uint16` shard loader (prefetching,
  deterministic); byte-level BPE load + encode/decode; `lmkit` CLI with
  `quickstart`/`train`/`shard`.

The lm-100m-en Chinchilla-budget pretrain is **running** on a single ~8GB CUDA GPU,
so section A below is closed. See [`examples/lm-100m-en`](./../examples/lm-100m-en).

---

## A. Critical path to the lm-100m-en reproduction — DONE
All built and exercised by the live run. lm-100m-en config values in parens.

- ✅ **DataLoader** — mmap `uint16` shards → `(x,y)` block batches (`block_size=2048`),
  prefetching, deterministic.
- ✅ **Tokenizer load** — 32k byte-level BPE `tokenizer.json`, encode/decode,
  round-tripped against a committed Python fixture.
- ✅ **Gradient accumulation** (`grad_accum=32`, effective batch 64, 131k tok/step).
- ✅ **Gradient clipping by global norm** (`grad_clip=1.0`) — hand-rolled global-L2
  clip (`train/clip.go`).
- ✅ **WSD LR schedule** — warmup → stable (constant) → optional decay
  (`warmup_steps=1000`, `lr=4e-4`, `min_lr=4e-5`, `decay_frac=0`), hand-rolled
  piecewise.
- ✅ **AdamW full config** — `beta1=0.9 beta2=0.95 weight_decay=0.1`, decoupled decay
  on rank≥2 params only.
- ✅ **bf16 training** (`dtype=bfloat16`) with fp32-internal norms, gated on the CUDA
  device.
- ✅ **Periodic eval** (`eval_interval=2000`, `eval_iters=100`) — held-out val loss,
  the signal compared against the baseline.
- ✅ **Checkpointing + resume** — rolling `latest` (`save_interval=200`), permanent
  snapshots (`snapshot_interval=25000`, `keep_last_snapshots=3`), auto-resume from
  latest.
- ✅ **metrics.jsonl** — emits `step/train_loss/val_loss/lr/tok_per_sec/peak_vram/`
  `grad_norm/tflops`; peak VRAM via `go-nvml` (CUDA-only).
- ✅ **The pretrain driver** — `train.Run`: the step loop tying accumulation, clip,
  schedule, eval, checkpoint, metrics, and resume together.
- ✅ **TrainConfig + config loading** — JSON config mirroring the dataclass
  (lr/schedule/batch/intervals/optimizer/seed/dtype).
- ✅ **Run on the GPU host** — the live Chinchilla-budget run toward the val curve
  (single ~8GB CUDA GPU, bf16, ~days), durable/resumable per the WSD config.

## B. Tokenizer (lmkit/tokenizer.py, lmkit/shard.py)
- ✅ **Load + encode/decode** byte-level BPE (`tokenizer/`), gated by a Python
  equivalence fixture.
- ✅ **Go corpus sharder** (`lmkit shard`: gzipped JSONL → tokenize → `uint16` shards).
- ⬜ **Train a fresh BPE** from a corpus (`train_tokenizer.py --vocab-size 32000`).
  We load an existing vocab; training one in Go is a later/optional add.
- 🧊 **Byte-exact without cgo** — pure-Go loader + Python equivalence gate (ADR-0003);
  cgo `daulet/tokenizers` is the exact-but-cgo alternative we rejected.
- ⬜ **Corpus pull** (`data.py`: stream SmolLM-corpus + wikipedia → jsonl.gz). Exists
  in Python; out of scope for Go (one-time data prep).

## C. Eval / generation (lmkit/eval/gen.py, judge.py) — post-reproduction
- ⬜ **Autoregressive generation / sampling** — greedy/temperature/top-k/top-p decode.
- ⬜ **KV cache** for efficient inference/generation (the model is forward-only now).
- ⬜ **LLM-judge eval** (`eval/judge.py`) — quality eval via a judge model.
- ⬜ **Eval harness / metrics** beyond val loss.

## D. SFT — supervised fine-tuning (lmkit/sft.py, sft_data.py)
- ⬜ **SFT loop** — fine-tune off a pretrained base (`SFTConfig`, `init_from` weights-
  only + fresh optimizer).
- ⬜ **Instruct data** handling (`instruct.jsonl.gz`, chat templating, loss masking on
  prompt tokens).

## E. IO / hub (lmkit/push.py)
- ⬜ **safetensors** read/write (~200-line hand-roll; the format is a JSON header +
  raw bytes).
- ⬜ **GGUF export** (F16/F32 hand-roll; quantize via llama.cpp `llama-quantize`).
- ⬜ **HF Hub push** (`lmkit.push`) — upload model + tokenizer (net/http + LFS, or
  shell the `hf` CLI first).
- ⬜ **Checkpoint format** interop (read Python `.pt`?) — likely N/A; we use GoMLX
  checkpoints. Cross-stack weight loading would go via safetensors.

## F. Backend / kernels / hardware (mostly ADR-0004, intentional)
- ✅ **FlashAttention** — cuDNN `__cudnn$fmha` forward + flash backward via StableHLO
  custom-call (gomlx/compute/go-xla forks), parity-verified on sm_86, scores never
  materialized. Earns ~1.4x on its own and unblocks seq-2048 training on the 8GB card
  (the memory enabler). See `docs/specs/2026-06-22-flash-attention-design.md`.
- ✅ **Throughput gap closed** — the ~26x residual the flash work could not explain
  was a GoMLX autodiff bug: weight-gradient matmuls (`DotGeneral` VJP) lowered as
  multiply+reduce on CUDA cores instead of tensor-core gemms. Fixed upstream
  (gomlx #428, 2026-06-24); the live run now sustains ~29k tok/s, close to the PyTorch
  reference (~32k). Supersedes the "broad per-kernel inefficiency" read in the flash
  design doc's 2026-06-23 Outcome.
- 🧊 **ROCm** (an AMD/ROCm GPU) training path — PJRT-ROCm validation pending.
- 🧊 **Metal** (Apple GPU) training — revive the PJRT-Metal bridge (jax-metal is
  abandoned). Workstream, not on the repro critical path.
- ✅ **peak-VRAM** query — `go-nvml` poll (CUDA-only), wired into `metrics.jsonl`.
- 🧊 **Hand-written kernels** (Triton/CUTLASS-style) — deferred unless profiling
  forces it.
- ⬜ **Multi-GPU / distributed** (FSDP/DDP) — single-GPU for the repro; later.
- 🧊 **gotch fallback** — second `Backend` impl, kept real by the boundary, unbuilt.

## G. CLI / ergonomics (lmkit/cli.py, quickstart.py)
- 🟡 **CLI** — `quickstart`, `train`, `shard` subcommands built (`cmd/lmkit`); an
  `eval` subcommand is the remaining gap.
- ✅ **quickstart** — bring-up demo (backend proof + device select).
- ✅ **observability** — `metrics.jsonl` structured run logging (mirrors lmkit's
  `observability`), consumable by the existing Grafana dashboard + ops CLI unchanged.

## Net read
The model and the full production training loop are built, and the lm-100m-en
Chinchilla-budget reproduction is running on the 8GB GPU. The GoMLX gemm-gradient fix
(#428) closed the throughput gap to roughly the PyTorch reference, retiring the "broad
per-kernel inefficiency" concern. The remaining surface is post-reproduction:
generation + KV cache, LLM-judge eval, SFT, and IO/hub (safetensors/GGUF/HF push).
Optional completeness: a Go BPE trainer and an `eval` subcommand. Multi-GPU,
ROCm/Metal, and hand-kernels stay intentional ADR-0004 deferrals.
