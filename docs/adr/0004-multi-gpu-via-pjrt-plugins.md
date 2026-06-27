# 4. Multi-GPU via PJRT plugins; Metal = revive the bridge; kernels deferred

Status: accepted (2026-06-19)

## Context

Three GPU architectures are available: CUDA (a CUDA GPU, e.g. ~8GB), ROCm (an AMD/ROCm GPU), and
Metal (local dev machine). A natural but wrong assumption is that supporting all three means
writing GPU kernels. It does not — and Python doesn't either.

Vendor kernels already exist for all three: cuBLAS/cuDNN (NVIDIA),
rocBLAS/MIOpen (AMD), MPSGraph (Apple). PyTorch and JAX *call* these; they don't
reimplement them (the rare hand-written kernel, e.g. Tri Dao's flash-attn, is a
specialist last-20% optimization exposed as a library). JAX writes essentially
nothing — XLA generates kernels or calls the vendor libraries.

What actually differs per GPU is the **PJRT plugin**: the C++ bridge translating
XLA's StableHLO into vendor calls.

| GPU | Vendor kernels | PJRT bridge → GoMLX |
|---|---|---|
| CUDA | cuBLAS/cuDNN | works |
| ROCm | rocBLAS/MIOpen | exists in OpenXLA, untested via GoMLX |
| Metal | MPSGraph | jax-metal frozen Oct 2024, version-incompatible with current GoMLX StableHLO |

The local dev machine trains today via Python MLX/MPS — a different path that calls MPSGraph
directly, not through XLA/PJRT. The broken piece for GoMLX-on-Metal is the *bridge*,
not the kernels.

## Decision

Treat multi-GPU as a PJRT-plugin coverage problem, behind the backend boundary
(ADR-0002). Do **not** write GPU kernels.

- **CUDA** is the primary training target; reproduce the north-star baseline there
  first.
- **ROCm**: validate the existing OpenXLA PJRT-ROCm plugin against GoMLX as a
  parallel workstream, gated behind the CUDA baseline.
- **Metal**: an explicit workstream to *revive the bridge* — rebuild/fork a
  StableHLO→MPSGraph PJRT plugin (jax-metal or MetalHLO lineage) that current
  GoMLX can load. This calls Apple's existing Metal kernels; it does not write
  them. C++/bridge work, not kernel work.
- **FlashAttention** is expressed as an XLA op-graph (tiled online-softmax) and/or
  left to XLA-GPU's cuDNN fusion — portable across whatever plugin runs, no custom
  kernel.
- **Hand-writing GPU kernels** (Triton/Pallas/CUTLASS-style) is deferred to its own
  milestone, opened only if profiling proves a specific op is the bottleneck and
  neither XLA codegen nor a vendor library covers it.

## Consequences

- "Three GPUs" is achievable without a kernel-authoring program, consistent with
  ADR-0001's "no kernels" bet.
- The Metal goal is real but uncertain: reviving a PJRT plugin on a moving XLA
  StableHLO target is non-trivial C++ work on an unmaintained upstream. Scoped as a
  parallel workstream, not on the north-star critical path.
- ROCm support is contingent on the upstream plugin behaving; if it doesn't, ROCm
  becomes a bridge-fix workstream like Metal.
- If profiling later forces a hand-written kernel, that reverses part of this ADR
  and gets its own superseding ADR.

## Update (2026-06-27): ROCm validated

The upstream plugin behaved. lmkit-go runs on an RDNA4 card (gfx1201) via the JAX
ROCm PJRT plugin loaded by go-xla, with the ROCm 7 userspace supplied as pip wheels
(no system ROCm, no code change). `quickstart` passes matmul, autodiff and AdamW.
The cuDNN flash custom-call does not carry over (NVIDIA-only), so attention falls
back to the decomposed path on ROCm. Setup directions: `docs/backends.md`. ROCm is
no longer a deferred/contingent item; an AMD fused-attention path (MIOpen) remains a
later optimization.
</content>
