# lmkit-go — context map & decomposition

**Status:** active. Supersedes the informal component list in `DESIGN.md` §"Component
map". `DESIGN.md` remains the project brief and north-star statement.

This document decomposes lmkit-go into bounded contexts, fixes the boundaries
and the shared language, and sets the build order. It is the map; each
sub-project gets its own spec under `docs/specs/`.

## Why a map first

`DESIGN.md` describes a program (7 packages, 6 milestones, ending in a 100M-model
reproduction), not a single project. Built blind it would drift. The decomposition
exists so each piece has one purpose, a typed boundary, and can be built and
tested on its own.

## The bet, restated (see ADR-0001..0004)

Build the *ergonomic Go training layer* on top of XLA via GoMLX. We do **not**
write GPU kernels or an autograd engine — XLA generates kernels and calls vendor
libraries (cuBLAS/cuDNN, rocBLAS/MIOpen, MPSGraph), exactly as JAX does. What we
*do* build in Go is everything above the XLA op layer, plus the Go-native
dependencies the ecosystem is missing (BPE trainer, safetensors, GGUF, hub
upload, dataloader) and FlashAttention expressed as an XLA op-graph. Multi-GPU is
a question of PJRT *plugins* (bridges to vendor kernels), not kernels.

## Bounded contexts

| Context | Responsibility | Vendor it shields |
|---|---|---|
| **backend** | The anti-corruption layer over XLA. Owns the domain types `Tensor`, `Device`, `DType` and the operations `model`/`train` need: build graph, run, gradient, optimizer-step. Selects device + PJRT plugin. | GoMLX / go-xla / PJRT |
| **tokenizer** | Train and load a 32k BPE; encode/decode. Must load an existing HF `tokenizer.json`. | — (Go-native; see ADR-0003) |
| **data** | corpus → `uint16` `.bin` shards; mmap `DataLoader` → `(x,y)` block batches; content-hash train/val split. | mmap, file layout |
| **model** | Llama blocks (RMSNorm, RoPE, GQA attention, SwiGLU FFN, tied embeddings), config-driven. FlashAttention as an XLA op-graph. Expressed **only** in `backend` types. | — (pure domain) |
| **train** | The loop: AdamW, grad accumulation, grad clip (global-norm), WSD schedule, periodic eval, checkpoint/resume, `metrics.jsonl`. | — |
| **io** | safetensors read/write, GGUF export, HF Hub push. | file formats, HF API |
| **cmd/lmkit** | CLI composition root: `shard`, `train`, `eval`, `quickstart`. | flag parsing |

## The one non-negotiable rule: the backend boundary

`backend` is the anti-corruption layer for the whole project. `model` and `train`
depend on `backend`'s domain types and **never import `gomlx`, `go-xla`, or PJRT
directly.** `backend` translates: our types in, vendor calls out, vendor results
back to our types.

Enforcement is a test: `grep -rE 'gomlx|go-xla|pjrt'` over the source tree returns
hits **only** under `backend/`.

Why it earns its keep here specifically:
- The vendor stack is effectively one author and churns (gopjrt→go-xla rename, a
  post-v0.27.3 `main` API refactor). That churn is absorbed in one package.
- It keeps the `gotch` fallback in `DESIGN.md` *real* — a second `Backend` impl,
  not a theoretical escape hatch.
- The three PJRT plugins (CUDA/ROCm/Metal) plug in behind one interface;
  `model`/`train` never know which GPU they run on.

See ADR-0002.

## Ubiquitous language — collisions to fix in code

| Term | Meaning A | Meaning B | Resolution |
|---|---|---|---|
| **block** | transformer decoder block (`model`) | sequence/context window (`data`) | `Block`/`Layer` vs `SeqLen` / `block_size` |
| **model** | the nn definition (`model` pkg) | saved weights file (`io`) | *model* vs *checkpoint* |
| **step** | optimizer step | grad-accum micro-step | *step* vs *microstep* |
| **shard** | data `.bin` shard | (reserved — no other use) | data shard only |

## Hardware & backend reality (mid-2026)

Three GPU architectures are available; their GoMLX training paths differ sharply.

| GPU | Vendor kernels | PJRT plugin → GoMLX | Status for training |
|---|---|---|---|
| **CUDA** (3070 Ti on `trig`, linux/amd64) | cuBLAS/cuDNN | works | **known-good** — primary target |
| **ROCm** (R9700, AMD) | rocBLAS/MIOpen | exists in OpenXLA, untested via GoMLX | **validate** (ADR-0004) |
| **Metal** (this Mac, Apple Silicon) | MPSGraph | jax-metal plugin abandoned/version-broken | **revive bridge** (ADR-0004) |
| **CPU** (any) | — | XLA-CPU + pure-Go SimpleGo | dev + parity tests |

Local Mac dev runs on SimpleGo (pure-Go, no cgo) or XLA-CPU for correctness and
layer-parity work. Real training currently runs on CUDA (`trig`). ROCm and Metal
GPU training are explicit workstreams, not assumptions — see ADR-0004.

## Dependency strategy

Build the missing pieces in Go rather than shelling out to Python / llama.cpp /
the `hf` CLI (ADR-0003): a pure-Go BPE trainer, safetensors read/write, a GGUF
writer, an HF Hub uploader (net/http + LFS), the DataLoader. This *is* the OSS
contribution — the gap the Go ecosystem has. Off-the-shelf Go libraries are used
where they're mature (e.g. `edsrzf/mmap-go`); everything central is owned.

## Build order

Dependency-driven critical path to the north star (reproduce val loss 1.7337):

```
backend ──► model ──► train (overfit-a-batch, synthetic data)
                          │
        tokenizer ──► data ┘
                          ▼
                      reproduce ──► io
```

- **backend** first — no internal deps, de-risks the bet (Milestone 1).
- **model** needs only `backend`.
- **train** proves itself with overfit-a-batch on *synthetic* data before
  `data`/`tokenizer` exist.
- **tokenizer** + **data** join for the real corpus.
- **io** is last (export/hub), post-reproduction.

Parallel workstream, not on the critical path: **ROCm validation** and **Metal
PJRT-plugin revival** (ADR-0004), gated behind a working CUDA baseline.

## Validation philosophy (from DESIGN.md, non-negotiable)

- **Layer parity:** each block matches a small PyTorch reference within tight
  tolerance; golden fixtures committed.
- **Overfit-a-batch:** the loop drives loss → ~0 on one tiny batch before any full
  run.
- **End-to-end:** the reproduction val curve tracks the published baseline.

## Sub-project specs (each its own document)

1. `YYYY-MM-DD-milestone-1-bringup-design.md` — backend bring-up (next).
2. model parity — *to be written*.
3. train loop + overfit-a-batch — *to be written*.
4. data + tokenizer — *to be written*.
5. reproduction run — *to be written*.
6. io (safetensors/GGUF/hub) — *to be written*.
7. Metal PJRT-plugin revival — *to be written* (parallel).

## Decisions captured as ADRs

- ADR-0001 — Build on GoMLX/XLA; track `main`.
- ADR-0002 — The backend anti-corruption layer (the no-leak rule).
- ADR-0003 — Build dependencies in Go, not shell-outs.
- ADR-0004 — Multi-GPU via PJRT plugins; Metal = revive the bridge; hand-written
  kernels deferred.
</content>
</invoke>
