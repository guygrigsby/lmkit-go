# 5. Per-package Go modules in a monorepo

Status: accepted (2026-06-19)

## Context

lmkit-go is one large repo, but the Go-native dependencies it builds (ADR-0003) —
BPE tokenizer, safetensors, GGUF, HF Hub uploader, dataloader — are meant for
reuse by others. Their value *is* being dependency-light. If they share a module
with `backend` (which requires gomlx/go-xla/cgo), a consumer doing
`go get .../safetensors` drags the entire XLA stack into their module graph,
defeating the reuse goal.

`internal/` only protects within its own module, so it cannot be the boundary that
keeps reusable packages public — module structure has to.

## Decision

Each reusable context is its own Go module, tied for local development by a
`go.work` at the repo root. Modules version independently via Go's
`module/vX.Y.Z` tag convention (e.g. `safetensors/v0.1.0`).

```
github.com/guygrigsby/lmkit-go/
  go.work
  tokenizer/    go.mod  (pure Go)
  safetensors/  go.mod  (pure Go)
  gguf/         go.mod  (pure Go)
  hub/          go.mod  (net/http only)
  data/         go.mod  (+edsrzf/mmap-go)
  backend/      go.mod  (+gomlx/go-xla)
  model/        go.mod  (+backend)
  train/        go.mod  (+backend, +model)
  io/           go.mod  (+safetensors, +gguf, +hub)
  app/          go.mod  (cmd/lmkit + internal/ — lmkit-specific glue)
```

Rules:
- Reusable code is a public package in its own module. `internal/` exists **only**
  in the `app` module, for the lmkit reproduction glue and CLI wiring particular to
  this project — nothing general-purpose hides there.
- A module's dependencies are the minimum it needs. The pure-Go libs
  (`tokenizer`, `safetensors`, `gguf`, `hub`) never depend on gomlx/go-xla.
- The backend boundary (ADR-0002) holds at the module level too: only `backend`
  requires gomlx/go-xla; `model`/`train` require `backend`, not the vendor.

## Consequences

- Consumers import exactly one lib with a minimal dependency tree
  (`.../safetensors` is pure Go, no XLA).
- Overhead: many `go.mod` files, per-module version tags, and `go.work` for local
  cross-module development (and CI must build each module). Accepted as the cost of
  the reuse goal.
- Releasing a lib means tagging `name/vX.Y.Z`; modules can rev independently rather
  than being chained to the app's version.
- New shared utility used by 2+ modules → its own module, not a dumping-ground
  `common` package, to avoid recreating the heavy-dep coupling this ADR removes.
- The backend-boundary grep test (ADR-0002) runs per module in CI.
</content>
