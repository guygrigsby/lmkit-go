# 2. The backend anti-corruption layer

Status: accepted (2026-06-19)

## Context

The vendor stack (GoMLX / go-xla / PJRT) is effectively one author and visibly
churns: `gopjrt` was renamed to `go-xla`, and a `main` API refactor moved
`ml/context`→`ml/model` and `optimizers`→`optimizer` after the last tag. We also
keep gotch as a documented fallback (ADR-0001) and intend to run three PJRT
plugins (CUDA/ROCm/Metal, ADR-0004). If vendor types leak into `model` and
`train`, every one of those changes ripples across the tree, and the fallback
becomes theoretical.

## Decision

`backend` is the anti-corruption layer for the entire project. It defines our own
domain types (`Tensor`, `Device`, `DType`) and an interface expressing the
operations `model`/`train` need (build graph, run, gradient, optimizer-step,
device/plugin selection). The adapter under `backend/gomlx` is the **only** package
permitted to import `gomlx`/`go-xla`/PJRT, and it translates in both directions.

`model` and `train` import `backend` only.

Enforce with a test: `grep -rE 'gomlx|go-xla|pjrt'` over the source tree must hit
**only** under `backend/`. Wire it into the project gate.

## Consequences

- Vendor churn (renames, the `main` refactor) is absorbed in one package.
- The gotch fallback is a real second implementation of the `Backend` interface,
  not a hope.
- The three PJRT plugins plug in behind one interface; `model`/`train` are
  GPU-agnostic.
- Cost: a translation layer to write and maintain, and some XLA expressiveness is
  mediated through our interface rather than used raw. Accepted — this is the
  single rule the "build it in Go on a volatile vendor stack" plan rests on.
- The interface is grown as `model`/`train` need ops, not designed up front (YAGNI).
</content>
