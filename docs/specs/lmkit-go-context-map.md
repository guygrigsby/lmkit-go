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
| **model** | Composable transformer blocks (RMSNorm, RoPE, GQA attention, SwiGLU FFN, tied embeddings), config-driven — a new architecture is a Go `Forward` + the odd block, Config-dispatched. **Arbitrary architectures intended; Llama is the first/reproduction target, not the ceiling.** FlashAttention as an XLA op-graph. Built in GoMLX `core/graph` ops (ADR-0009); no runtime/device imports. | GoMLX op vocabulary |
| **train** | The loop: AdamW, grad accumulation, grad clip (global-norm), WSD schedule, periodic eval, checkpoint/resume, `metrics.jsonl`. | — |
| **io** | safetensors read/write, GGUF export, HF Hub push. | file formats, HF API |
| **app** (`cmd/lmkit` + `internal/`) | CLI composition root: `shard`, `train`, `eval`, `quickstart`; lmkit-specific reproduction glue. | flag parsing |

## Repository layout — per-package modules (ADR-0005)

One repo, but each reusable context is its own Go module, tied by a root `go.work`
and versioned independently (`module/vX.Y.Z` tags). This keeps the Go-native deps
(ADR-0003) dependency-light: importing the pure-Go libs must not drag in XLA.

```
github.com/guygrigsby/lmkit-go/
  go.work
  tokenizer/    go.mod  (pure Go)
  safetensors/  go.mod  (pure Go)
  gguf/         go.mod  (pure Go)
  hub/          go.mod  (net/http only)
  data/         go.mod  (+edsrzf/mmap-go)
  backend/      go.mod  (+gomlx/go-xla)
  model/        go.mod  (+backend, +gomlx core/graph)
  train/        go.mod  (+backend, +model, +gomlx core/graph)
  io/           go.mod  (+safetensors, +gguf, +hub)
  app/          go.mod  (cmd/lmkit + internal/ — lmkit-specific glue)
```

`internal/` exists **only** in the `app` module. Nothing general-purpose hides
there — every reusable dep is a public package in its own module.

## The one non-negotiable rule: the runtime boundary

`backend/gomlx/` is the anti-corruption layer for the **runtime**: backend
construction (`compute.New`), device/PJRT-plugin selection, `go-xla`, execution,
optimizer wiring, and checkpointing live there and nowhere else. This is the
swappable, churn-prone, device-specific surface.

Refined by ADR-0009 (supersedes ADR-0002's strict rule): `model` and `train` MAY
import GoMLX's op/type vocabulary (`core/graph`, `core/tensors`, `compute/dtypes`,
`compute/shapes`) to express layer math — building a parallel op-DSL behind the
boundary was over-engineering. They must NOT import the runtime packages.

Enforcement is a test: the **runtime** imports
(`"github.com/gomlx/compute"`, `gomlx/backends`, `go-xla`, `pjrt`) appear **only**
under `backend/gomlx/`. The op-vocabulary packages are allowed everywhere.

Why it earns its keep here specifically:
- The vendor stack is effectively one author and churns (gopjrt→go-xla rename, a
  post-v0.27.3 `main` API refactor). That churn is absorbed in one package.
- It keeps the `gotch` fallback in `DESIGN.md` *real* — a second `Backend` impl,
  not a theoretical escape hatch.
- The three PJRT plugins (CUDA/ROCm/Metal) plug in behind one interface;
  `model`/`train` never know which GPU they run on.

See ADR-0002 and ADR-0009.

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
| **CUDA** (a CUDA GPU, e.g. ~8GB, linux/amd64) | cuBLAS/cuDNN | works | **known-good** — primary target |
| **ROCm** (an AMD/ROCm GPU) | rocBLAS/MIOpen | exists in OpenXLA, untested via GoMLX | **validate** (ADR-0004) |
| **Metal** (Apple Silicon, local dev) | MPSGraph | jax-metal plugin abandoned/version-broken | **revive bridge** (ADR-0004) |
| **CPU** (any) | — | XLA-CPU + pure-Go SimpleGo | dev + parity tests |

Local dev runs on SimpleGo (pure-Go, no cgo) or XLA-CPU for correctness and
layer-parity work. Real training currently runs on CUDA (the GPU host). ROCm and Metal
GPU training are explicit workstreams, not assumptions — see ADR-0004.

## Dependency strategy

Build the missing pieces in Go rather than shelling out to Python / llama.cpp /
the `hf` CLI (ADR-0003): a pure-Go BPE trainer, safetensors read/write, a GGUF
writer, an HF Hub uploader (net/http + LFS), the DataLoader. This *is* the OSS
contribution — the gap the Go ecosystem has. Off-the-shelf Go libraries are used
where they're mature (e.g. `edsrzf/mmap-go`); everything central is owned.

## Build order

Dependency-driven critical path to the north star (reproduce the Chinchilla-budget baseline):

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
- ADR-0005 — Per-package Go modules in a monorepo (reusable deps stay
  dependency-light; `internal/` only in `app`).
- ADR-0006 — Aggressive upgrade cadence (Go + all deps bumped often and
  automatically; the named tax of a bleeding-edge stack).
- ADR-0007 — Test rigor as the safety net (backend boundary contract tests catch
  vendor refactors in one place; the green gate is the merge bar for every bump).
- ADR-0008 — Public API evolves additively (grow, don't remove; under-expose first;
  removals cost a major-version bump).
- ADR-0009 — The boundary is the runtime, not the op vocabulary (`model`/`train`
  may use GoMLX `core/graph` ops; backend construction/device/exec/optimizer/
  checkpoint stay in `backend/`). Supersedes ADR-0002's strict grep rule.
</content>
</invoke>
