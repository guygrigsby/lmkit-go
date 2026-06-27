# lmkit-go — design

**Status:** design. No code yet. This document is the brief for the first
implementer.

**Name** `lmkit-go` — the Go sibling to the Python `lmkit` stack. Descriptive,
not a codename. Lives under `guygrigsby`/Aeryx (research). Module path
`github.com/guygrigsby/lmkit-go`. MIT.

## What this is

A from-scratch large-language-model **training** toolkit written in Go, as an
open-source give-back. The Go ecosystem has tensor math (via XLA bindings) but
lacks the *ergonomic training layer* — tokenization, data sharding, transformer
building blocks, a real training loop, checkpointing, eval, hub interop. This
project fills that gap. `lmkit-go` mirrors the Python `lmkit` surface, package
for package.

## What this is NOT

- **Not** a replacement for the existing Python/PyTorch stack (`lmkit`). That
  stack keeps doing the real research training. This is a parallel, for-fun,
  community project.
- **Not** a new autograd engine or a new set of GPU kernels. See the core bet.

## The core architectural bet

**Do not write kernels or autograd.** Reimplementing PyTorch's math in Go is a
multi-year effort that loses to torch / JAX / Candle. The leverage is the
*compiler backend*:

- Build on **GoMLX** (`github.com/gomlx/gomlx`), which binds **XLA/PJRT** via
  `gopjrt`. That gives real graph compilation, op fusion, and CUDA/CPU (and
  potentially TPU/Metal) execution for free — the same backend JAX uses.
- gotrain owns the *layer above*: model definitions, the training loop, data,
  tokenization, IO, ergonomics. XLA owns the math and the hardware.

If GoMLX proves too limiting, the fallback backend is gotch (libtorch bindings),
but that re-introduces a C++ dependency and torch coupling — prefer GoMLX.

**Explicitly rejected:**
- *Gorgonia* — immature, weak modern-GPU training story, effectively a dead end.
- *gotch / libtorch* — you inherit C++ binding lag and still depend on torch.
- *Pure-Go tensor libs (no XLA)* — no competitive GPU path.

(If the goal were "leave Python" rather than "do it in Go," **Rust + Candle** is
the more mature systems-language training framework. We chose Go on XLA because
the *ergonomic Go layer is the novel, unfilled contribution*.)

## North-star artifact (the first real milestone)

Reproduce a tiny-Llama baseline in pure Go-on-XLA:

- Target: the ~100M `Aeryx-ai/lm-100m-en` vanilla Llama (English-only, fresh 32k
  BPE, GQA), trained to a **Chinchilla token budget**. The original pretraining
  corpus is no longer available, so the baseline is re-established on a freshly
  assembled English corpus rather than the original run's exact validation loss.
  (Architecture/config/tokenizer recipe from the Python stack's `lm-100m-en` project.)
- Success = a from-scratch Go-on-XLA run reproduces a comparable val curve on that
  corpus + tokenizer + config.

Why this target: tiny enough to train cheaply, fully specified, and we already
own ground-truth numbers to validate against. The OSS story becomes
"from-scratch LLM training in pure Go, reproducing a published baseline" — not a
vaporware general framework.

## Component map (the lmkit-shaped surface to build)

Mirror what `lmkit` provides in Python, each as a Go package:

1. **backend** — thin wrapper over GoMLX context/graph/exec; device + dtype
   (bf16) selection; the autodiff + optimizer-step plumbing XLA exposes.
2. **tokenizer** — train/load a 32k BPE. Options: bind HF `tokenizers` (Rust)
   via `github.com/daulet/tokenizers`, or a pure-Go BPE. Must load the existing
   lm-100m-en `tokenizer.json` for the reproduction milestone.
3. **data** — port the `shard.py` concept: corpus → tokenized `uint16` `.bin`
   shards; a memory-mapped `DataLoader` yielding `(x, y)` block batches; a
   content-hash train/val split. Byte-compatibility with the Python shards is a
   nice-to-have for exact reproduction.
4. **model** — Llama blocks in GoMLX ops: RMSNorm (fp32-internal), RoPE, GQA
   attention (prefer an XLA fused-attention path; else explicit
   softmax(QK^T/√d)V with a causal mask), SwiGLU FFN, tied embeddings.
   Config-driven (vocab/hidden/layers/heads/kv-heads/block/rope-base).
5. **train** — the loop: AdamW, grad accumulation, grad clip, warmup → stable
   (WSD, no-decay trunk) lr schedule, periodic eval, `latest`/`best`/snapshot
   checkpointing, resume. Emits metrics (mirror lmkit's `metrics.jsonl`:
   step/train_loss/val_loss/lr/tok_per_sec/peak_vram so the existing Grafana
   Workers dashboard and `lmkit` ops CLI work unchanged).
6. **io** — safetensors read/write; optionally GGUF export. Hub push to HF.
7. **cmd/lmkit** — CLI (binary `lmkit`): `shard`, `train`, `eval` (+
   `quickstart` demo).

## Validation philosophy (non-negotiable)

Prove correctness with tests, numerically, before claiming it works:

- **Layer parity:** each block (RMSNorm, RoPE, attention, SwiGLU) gets a test
  asserting its forward output matches a small PyTorch reference within a tight
  tolerance (e.g. 1e-4 bf16 / 1e-6 fp32). Capture reference tensors from torch,
  commit them as golden fixtures.
- **Overfit-a-batch:** the training loop must drive loss → ~0 on a single tiny
  batch before any full run (catches optimizer/autodiff wiring bugs).
- **End-to-end:** the reproduction run's val curve tracks the published baseline.

## Suggested milestones (for the implementer)

1. **Bring-up:** `go.mod`, GoMLX hello-world (a matmul on CPU then CUDA via the
   PJRT plugin). Confirm bf16 + a backward pass + an AdamW step run on GPU.
2. **Model parity:** implement the Llama blocks; pass layer-parity tests vs a
   torch reference; assemble the full model and match a reference forward.
3. **Loop:** training loop + overfit-a-batch test green.
4. **Data + tokenizer:** load the lm-100m-en tokenizer; shard a freshly assembled
   corpus; DataLoader.
5. **Reproduce:** full training run at the lm-100m-en config/tokenizer; compare the
   val curve to a Chinchilla-budget baseline.
6. **IO + hub:** safetensors export, push to HF; (optional) GGUF for local
   inference.

## Open questions to resolve early

- GoMLX maturity for: bf16 training, a fused/efficient attention, multi-GPU
  (the reproduction is single-GPU; multi-GPU is later).
- Tokenizer: bind Rust `tokenizers` (fast, exact-match to the Python side) vs a
  pure-Go BPE (no cgo, but must match the existing vocab).
- How close to byte-identical the Go shards can be to the Python `.bin` shards
  (affects whether the reproduction is exact or just statistically equivalent).

## Environment notes

- Train on a CUDA (NVIDIA/Linux) GPU host; a single ~8 GB card is enough for the
  100M target. NVIDIA/CUDA is the natural first PJRT target; an AMD/ROCm card also
  works (validated on RDNA4/gfx1201, decomposed attention; see `docs/backends.md`).
  Don't stack GPU jobs; confirm the GPU is idle and serialize per your environment.
- Reuse existing monitoring: emit lmkit-compatible `metrics.jsonl` so a Grafana
  dashboard + the `lmkit` ops CLI manage runs with zero extra work.
