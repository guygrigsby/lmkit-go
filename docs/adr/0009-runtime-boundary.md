# 9. The boundary is the runtime, not the op vocabulary

Status: accepted (2026-06-19). Supersedes the strict "only `backend/**` imports
gomlx" rule of ADR-0002; ADR-0002's runtime-centralization intent stands.

## Context

Milestone 2 builds Llama blocks (RMSNorm, RoPE, GQA attention, SwiGLU), each
composed of dozens of tensor ops (`rsqrt`, `mul`, `einsum`, `softmax`, gather,
rotate, …). ADR-0002 said only `backend/**` may import GoMLX. Honoring that
literally for the model layer forces one of two costs:

- Build a full op-graph DSL behind the `backend` interface (~40 autodiff-correct op
  methods in lmkit types) so `model/` never imports GoMLX — reinventing most of
  GoMLX's `core/graph` surface, and (per ADR-0008) committing to it as additive-only
  forever.

The strict rule existed to keep the GoMLX→gotch fallback (ADR-0001) cheap. But
gotch is an unlikely, documented last resort, and the model layers are inherently
expressed in *some* op vocabulary; paying a large permanent abstraction cost to
keep one improbable swap trivial is over-engineering. The ecosystem agrees:
`gomlx/gemma` writes its layers directly in `core/graph` ops.

## Decision

Redraw the anti-corruption boundary at the **runtime**, not the op vocabulary.

`backend/gomlx/` exclusively owns the swappable, churn-prone, device-specific
runtime:
- backend construction (`compute.New`/`MustNew`), `backends/default`,
- device / PJRT-plugin selection, `go-xla`, `pjrt`,
- graph compilation + execution, optimizer wiring, checkpointing.

`model/` and `train/` MAY import the op/type **vocabulary** to express layers:
- `github.com/gomlx/gomlx/core/graph` (ops), `core/tensors`,
- `github.com/gomlx/compute/dtypes`, `github.com/gomlx/compute/shapes`.

They must NOT import the runtime packages above; they obtain a compiled/executable
graph and a device through `backend`.

**Boundary test (updated):** grep for the *runtime* imports and forbid them outside
`backend/gomlx/` — the patterns are the root `"github.com/gomlx/compute"`
(the `Backend` constructor module), `github.com/gomlx/gomlx/backends`,
`github.com/gomlx/go-xla`, and `pjrt`. The op-vocabulary packages
(`core/graph`, `core/tensors`, `compute/dtypes`, `compute/shapes`) are allowed
everywhere and are NOT matched.

## Consequences

- `model/` layers are GoMLX-op-coded, so a gotch swap would require rewriting the
  layers. Accepted — gotch is a last resort (ADR-0001), and layer math is small,
  well-specified, and re-derivable.
- The boundary still centralizes what actually varies and churns: device/plugin
  selection (CUDA/ROCm/Metal, ADR-0004), execution, optimizer, checkpoint — all in
  `backend/`. That is the surface the gopjrt→go-xla rename and the `main` refactor
  touched; the op vocabulary is comparatively stable.
- The grep gate becomes more specific (runtime packages), which is a tighter, truer
  statement of the invariant than "no gomlx anywhere."
- `model` and `train` modules depend on GoMLX `core/graph` in their `go.mod` — they
  are not in the "dependency-light pure-Go libs" set (ADR-0005); that set is
  `tokenizer`/`safetensors`/`gguf`/`hub`, which stay GoMLX-free.
</content>
