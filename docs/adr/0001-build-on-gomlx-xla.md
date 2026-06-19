# 1. Build on GoMLX/XLA; track main

Status: accepted (2026-06-19)

## Context

lmkit-go is a from-scratch LLM *training* toolkit in Go. The Go ecosystem has no
ergonomic training layer, but it does have one credible path to real graph
compilation and GPU autodiff: GoMLX (`github.com/gomlx/gomlx`), which binds
XLA/PJRT via `go-xla` (the successor to the deprecated `gopjrt`).

Alternatives evaluated (mid-2026, web-verified):
- **gotch** (libtorch): effectively abandoned — last commit Nov 2023, pinned to
  libtorch 2.1, checkpointing saves weights without optimizer state. Kept only as
  a documented fallback.
- **Gorgonia**: hibernating; no bf16; disqualified for modern LLM training.
- **No independent Go XLA/JAX binding exists** — every XLA path is the gomlx org.

GoMLX itself: Apache-2.0, active through mid-2026, ~1.5k stars, reverse-mode
autodiff, AdamW (correctly decoupled), grad accumulation, and — verified against a
committed checkpoint — full checkpoint/resume *including optimizer state, step,
and RNG*. A pure-Go `SimpleGo` CPU backend trains with zero cgo.

Two facts force a version decision: (1) GoMLX did a large API refactor on `main`
(commit "New modelling api", 2026-05-24) after the v0.27.3 tag; pkg.go.dev shows
the old layout. (2) `main` adds built-in LR warmup and the newer model API.

## Decision

Build the ergonomic Go layer on GoMLX/XLA. Do not write an autograd engine or GPU
kernels (see ADR-0004). Keep gotch as a documented fallback behind the backend
boundary (ADR-0002), never a direct dependency of `model`/`train`.

**Track GoMLX `main`, not the v0.27.3 tag.** Pin a specific commit in `go.mod` and
bump deliberately.

## Consequences

- We inherit the same kernels JAX uses (cuBLAS/cuDNN on CUDA) for free.
- Single biggest risk: bus-factor — the whole credible Go-on-XLA stack is
  effectively one author and churns. Mitigated by the backend boundary (ADR-0002),
  which confines churn to one package and keeps the gotch fallback real.
- Tracking `main` means a moving target; the backend boundary absorbs the API
  drift, and a pinned commit keeps builds reproducible.
- `main` uses `ml/model` (`Scope`/`Store`) and `ml/train/optimizer` (singular),
  not the `ml/context` / `optimizers` layout pkg.go.dev documents. Read source,
  not the stale docs.
</content>
