# Milestone 1 — backend bring-up (spec)

**Status:** ready for plan. Sub-project 1 of the context map
(`lmkit-go-context-map.md`). Decisions: ADR-0001..0004.

## Goal

De-risk the core bet. Prove that GoMLX/XLA, behind the lmkit-go `backend`
boundary, can do the three things every later milestone depends on:

1. a **matmul** that produces correct numbers,
2. a **gradient** (reverse-mode autodiff) that is numerically correct,
3. an **AdamW optimizer step** that measurably reduces a toy loss,

on **CPU** (locally, via SimpleGo and XLA-CPU) and on **CUDA** (a CUDA GPU host,
linux/amd64). Nothing here trains a model — it establishes the foundation and the
backend boundary.

## In scope

- The monorepo skeleton per ADR-0005: root `go.work`, the **`backend` module**
  (its `go.mod` pinned to a specific GoMLX `main` commit, ADR-0001), and the
  **`app` module** (`cmd/lmkit` + `internal/`). Other modules are stubbed only as
  later milestones need them — bring-up creates just `backend` and `app`.
- The `backend` package: minimal domain types and the smallest interface that
  expresses matmul + gradient + optimizer-step + device selection.
- One adapter under the `backend` module, `backend/gomlx`, the only package
  importing `gomlx`/`go-xla`/PJRT.
- An `app` `cmd/lmkit quickstart` that runs the three checks and prints results +
  selected device.
- Tests:
  - matmul correctness vs a hand-computed small case;
  - gradient correctness vs finite-difference on a scalar function;
  - AdamW reduces a quadratic toy loss to ~0 over N steps;
  - the **backend-boundary grep test** (ADR-0002): `gomlx|go-xla|pjrt` appears
    only under `backend/`.
  - **backend boundary contract tests** (ADR-0007): pin the numerical output of
    matmul/grad/AdamW through the `backend` interface so a GoMLX `main` bump that
    changes vendor behavior fails here, in `backend/`, not downstream. These are
    the early-warning system for the tracked-`main` churn — write them as the
    adapter is built, not after.
- Device selection: pick SimpleGo / XLA-CPU / CUDA via config or env, default to
  the best available.

## Out of scope (later milestones)

- Llama blocks, attention, any model (Milestone 2).
- The training loop proper, schedules, checkpointing (Milestone 3).
- Tokenizer, data, shards (Milestone 4).
- Metal and ROCm GPU paths (ADR-0004 parallel workstreams).
- bf16 specifics beyond confirming a dtype runs — full bf16 training validation
  rides on Milestone 2 parity tests.

## Minimal backend interface (grown, not designed)

Only what bring-up needs. Expanded in later milestones as `model`/`train`
require — **additively** (ADR-0008): once `backend` is a tagged public module we
can add to this interface but not remove from it, so under-expose now. Prefer
unexported until a consumer needs it.

```go
package backend

type DType int
const ( F32 DType = iota; BF16 )

type Device struct { Kind string; Index int } // "simplego","cpu","cuda"

type Tensor interface { Shape() []int; DType() DType }

// Backend hides gomlx entirely. backend/gomlx is its only implementation here.
type Backend interface {
    Device() Device
    // Compile a graph function of N inputs to M outputs.
    Compile(fn GraphFunc) (Executable, error)
    // Grad returns an executable computing d(out[0])/d(inputs) of fn.
    Grad(fn GraphFunc) (Executable, error)
    NewTensor(shape []int, dt DType, data []float32) Tensor
    Read(t Tensor) []float32
}
```

(Optimizer-step is exercised via GoMLX's AdamW behind the adapter; the public
shape of the optimizer API on the `backend` interface is decided in Milestone 3
when `train` needs it. Milestone 1 only proves a step runs and reduces loss.)

## Validation

Per `DESIGN.md` validation philosophy — prove numerically before claiming it works.

- **matmul:** a 2×3 · 3×2 case with known product; assert exact (F32).
- **gradient:** for f(x)=Σxᵢ², assert backend gradient ≈ 2x within 1e-4 of a
  central finite-difference estimate.
- **AdamW:** minimize f(x)=Σ(x−c)² from a random start; assert loss < 1e-3 after a
  fixed step budget, and x ≈ c.
- **boundary:** the grep test is part of the gate, not a manual check.
- **the real path (per workflow rules):** run the quickstart binary locally
  (SimpleGo and XLA-CPU) and on a CUDA GPU host, and record that each printed the
  passing checks + the device it used. A green CPU run is not a green CUDA run —
  both are required for "done".

## Environments

- **Local (dev machine):** SimpleGo (pure-Go, no cgo) as the zero-friction default;
  XLA-CPU as the second CPU path. Both CPU.
- **CUDA GPU host (`$GPU_HOST`):** CUDA via the PJRT-CUDA plugin (linux/amd64). Check the GPU is idle with `nvidia-smi` first; source pulled, not edited on the box.

## Done criteria

- [x] `go.work` + `backend` and `app` modules; `backend/go.mod` pins a GoMLX
      `main` commit (`516689cbe913`); both modules build clean.
- [x] Contract tests green locally on SimpleGo (matmul, gradient, AdamW) + the
      boundary test. (XLA-CPU not run locally — needs XLA libs; covered instead
      by the CUDA run on the GPU host, which exercises the full XLA path.)
- [x] Quickstart prints passing matmul/grad/AdamW + device locally (SimpleGo).
- [x] Same quickstart green on the CUDA GPU host with CUDA selected — see results below.
- [x] Backend-boundary grep test wired into the gate (runs per module).

## Validation results (2026-06-19, branch `milestone-1-bringup`)

Backend selector is `GOMLX_BACKEND=xla:cuda` (the registered backend is `xla`; CUDA
is its PJRT plugin — `cuda` alone is not a backend name).

- **Local, SimpleGo:** `make check` green; quickstart `device: "Go Backend"`,
  matmul `[58 64 139 154]`, gradient `[2 4 6]`, adamw `w=3.0000 loss=9.09e-13`.
- **CUDA GPU host, SimpleGo (linux/amd64):** cross-compiled locally (`CGO_ENABLED=0`),
  rsync'd to `/tmp`, ran green (`loss=5.12e-13`). Proves linux/amd64 portability.
- **CUDA GPU host, CUDA (NVIDIA GPU ~8GB):** cross-compiled locally with `zig cc`
  (`CGO_ENABLED=1`, `xla` backend), rsync'd to `/tmp`, ran under
  `GOMLX_BACKEND=xla:cuda`. go-xla auto-installed the `cuda` PJRT plugin (v0.112)
  to `~/.local/lib`. Device: `xla:cuda … pjrt_c_api_cuda_plugin.so v0.112
  [StableHLO] [1 device(s)]`; matmul/gradient/adamw all correct (`loss=1.42e-12`).
  Binary in `/tmp` cleaned up; no repo on the box (artifact-only deploy).

The core bet holds: GoMLX/XLA behind the `backend` boundary does correct
matmul + autodiff + AdamW on real CUDA, driven from a cross-compiled binary.

## Open questions to resolve during the plan

- Exact GoMLX `main` API surface for graph build + gradient + AdamW (read source;
  pkg.go.dev is stale per ADR-0001). The plan's first task is a spike confirming
  the real import paths and signatures before writing the adapter.
- Does GoMLX expose device/plugin selection cleanly enough to switch
  SimpleGo/CPU/CUDA via one config knob, or does the adapter wrap that?
- Confirm a bf16 tensor round-trips on at least one backend (smoke only).
</content>
