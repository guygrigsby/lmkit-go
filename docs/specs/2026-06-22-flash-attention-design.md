# Flash attention for the GoMLX/XLA path

**Status:** design. Central question RESOLVED (see below): XLA lowers the cuDNN
flash custom-calls on this exact stack, so the path is design (a), a Go-only
implementation in go-xla, no kernel writing.

## Why this matters (measured)

lmkit-go trains the lm-100m model (100M-param Llama, seq 2048, bf16) roughly **100x
slower** than the PyTorch reference on the same 8GB Ampere GPU (sm_86). The reference
does the Chinchilla budget (~2B tokens) in ~17 hours (~32k tok/s). lmkit-go at the
real config measures **~310 tok/s**, and is unstable (intermittent OOM). At 2048 the
reproduction is a multi-week crash loop; with flash it is an overnight run.

The gap is dominated by one thing: no flash attention. Breakdown of the ~100x:

- **Naive attention vs flash (~10-20x).** PyTorch runs FlashAttention: one fused
  kernel keeping each Q/K tile in SRAM, never writing the `[B,nH,T,T]` scores to HBM.
  lmkit-go materializes the full score matrix in HBM and runs matmul -> softmax ->
  matmul as separate ops with HBM round-trips. At T=2048 attention is
  memory-bandwidth-bound, so we shovel a huge matrix to/from HBM repeatedly. Worse as
  context grows (O(T^2)).
- **The memory cascade this forces (~3-5x).** Holding the full `[B,nH,T,T]` matrices
  blows past 8GB, forcing gradient checkpointing (a second forward per step), XLA's
  own rematerialization near the memory ceiling, and ragged-edge execution that is
  slow and OOM-prone. PyTorch+flash never approaches the wall, so none of this.
- **General fusion gap (~2-3x).** More, smaller XLA kernels with launch overhead;
  grad_accum=32 means 32 exec calls per step; less-tuned GEMMs than cuBLASLt/cuDNN.

None of this is a Go or XLA ceiling. XLA is the compiler JAX uses to match PyTorch on
this hardware. The 100x is one missing kernel plus the cascade it drags in.

## What we already ruled out (measured)

- **Gradient checkpointing** (gomlx `Node.Checkpoint()`, gomlx #425): fixes *memory*
  (fits 2048 at ~7.5GB) but not speed. It is the memory enabler, not the throughput
  fix. Checkpointing always trades compute for memory.
- **Chunked / online-softmax attention** (loop-unrolled in Go, so differentiable, no
  `While`): does not help. Training OOMs (the backward re-materializes the per-chunk
  scores), forward is a wash at 2048/4096 (the `[B,nH,T,T]` block is not the peak
  there, the weights and the 32k-vocab logits are), and the unroll does not scale
  (dies in compile at 8192). The speed win of flash is SRAM staging that lives in a
  fused kernel; online-softmax is the *algorithm* that enables tiling, but expressed
  as StableHLO ops it does not produce the kernel, XLA schedules slice/dot/exp/reduce
  as separate kernels with the tiles round-tripping HBM. Confirms: flash needs a
  kernel, not HLO ops.

## Probe: there is no flash path in the stack today (confirmed three ways)

1. Optimized HLO on sm_86: **0 custom-calls**, **0 fmha/cudnn** in the computation,
   just dot + exp + reduce-max (naive matmul -> softmax -> matmul). XLA is not
   auto-fusing our attention into anything.
2. `xla_gpu_enable_cudnn_fmha` is **reserved** in this XLA's DebugOptions proto: the
   old auto-rewrite flag was removed. Modern XLA dropped the implicit fMHA rewrite in
   favor of explicit custom-calls.
3. go-xla's `FusedScaledDotProductAttention` returns **ErrNotImplemented**. The
   fused-SDPA backend op is a stub, so gomlx's `attention.Core` falls back to the
   decomposed (naive) version on CUDA.

## The central question (gates the design)

**Does XLA's GPU backend still lower the `__cudnn$fmha` StableHLO `custom_call`
target to cuDNN flash (forward and backward), even though the auto-rewrite pass is
gone?** This is how `jax.nn.dot_product_attention(implementation="cudnn")` works: JAX
emits the custom_call directly; XLA's GPU backend lowers it to cuDNN's fused MHA. The
auto-rewrite (pattern-match naive attention -> custom-call) and the *lowering* of an
already-emitted custom-call are two different passes; only the rewrite is reserved.

go-xla's PJRT plugin is the jax-cuda PJRT plugin, so JAX on the same host uses the
*identical* XLA backend go-xla compiles through. So this is empirically testable: run
JAX flash attention, dump the StableHLO, and see whether it emits `__cudnn$fmha*` and
runs on sm_86.

### RESOLVED: yes (measured)

JAX 0.10.2 (the *exact* PJRT plugin version go-xla bundles, `jax-cuda13-pjrt_0.10.2`)
with `dot_product_attention(implementation="cudnn")` on the 3070 Ti (sm_86, cuDNN
9.23, CUDA 13):

- Forward emits `__cudnn$fmhaSoftmax` and **runs**.
- Backward emits `__cudnn$fmhaSoftmax` + `__cudnn$fmhaSoftmaxBackward` and **runs**.

So the lowering is alive; only the auto-rewrite was removed. Design (a) it is.

The exact custom_call spec (full HLO in `reference/fmha_fwd.hlo`, `fmha_bwd.hlo`):

- **Targets:** `__cudnn$fmhaSoftmax` (fwd), `__cudnn$fmhaSoftmaxBackward` (bwd).
- **Layouts:** inputs q,k,v are `bf16[B,S,H,D]` (BSHD). Forward output is
  `bf16[B,H,S,D]` plus a `u8[0]` workspace; backward output is dQ,dK,dV
  (`bf16[B,H,S,D]`) plus a ~25MB `u8` workspace.
- **Backward operands:** q, k, v, forward-output, dO, and the softmax stats
  `f32[B,H,S]` (the per-row logsumexp the forward must also return for the backward
  to recompute, this is the flash residual).
- **api_version:** `API_VERSION_STATUS_RETURNING`.
- **backend_config** (`cudnn_fmha_backend_config`): the bmm1/bmm2 dot dimension
  numbers (Q·Kᵀ then P·V, batch over B,H, contract over D), `fmha_scale` = 1/sqrt(D)
  = 0.125, `mask_type` = CAUSAL, `is_flash_attention` = true, `dropout_rate` 0,
  `algorithm.is_cudnn_frontend` true / `math_type` TENSOR_OP_MATH, plus the four
  grad-gemm dimension numbers on the backward. This is a serialized proto XLA already
  defines (`CudnnfMHABackendConfig`), so go-xla constructs it, no new C ABI.

## Two designs, picked by the answer

- **(a) XLA still lowers the custom_call.** Then flash is Go-only, no kernel writing:
  - go-xla `stablehlo`: emit a `custom_call` op with target `__cudnn$fmha-forward`
    (and `-backward`), the right operand layout, and the cuDNN `backend_config`.
  - go-xla `compute/xla`: implement `FusedScaledDotProductAttention` (and its VJP via
    the `-backward` custom-call) on top of that.
  - gomlx `attention.Core`: already routes to the fused op via `InternalFusedOpCaller`
    with a decomposed fallback, so once the backend implements it, gomlx uses it.
  - This is the tractable, few-hundred-lines-of-Go path. It is what jan flagged
    ("figure out how the undocumented custom-calls Jax uses can be used").
- **(b) XLA does not lower it / it is unsupported on sm_86.** Then we ship our own
  kernel (Pallas/Triton or hand-CUDA) behind a *generic* custom-call, which first
  needs generic custom-call plumbing added to go-xla (a C-ABI boundary). Much bigger.

## lmkit-go integration

The heavy work is upstream (go-xla, gomlx). lmkit-go consumes it at the existing
attention call site: `model.Attention` swaps the manual einsum-softmax-einsum for
gomlx's fused SDPA (`attention.Core`) when the backend implements it; the naive path
stays as the CPU/SimpleGo fallback and the fp32 parity reference. No new boundary;
the vendor seam is the gomlx attention op.

## Plan (design a)

1. **go-xla `stablehlo`:** add a `CustomCall` emitter (the op is already in
   go-xla's optype enum, just not exposed in the builder), able to set target,
   operands, tuple result shapes, `operand_layout_constraints`, `api_version`, and a
   `backend_config`.
2. **go-xla `compute/xla`:** implement `FusedScaledDotProductAttention` to emit
   `__cudnn$fmhaSoftmax` with the `CudnnfMHABackendConfig` (scale, causal, dot
   dimension numbers from the spec above), returning output + softmax-stats; and its
   VJP via `__cudnn$fmhaSoftmaxBackward`. Replaces the current `ErrNotImplemented`.
3. **gomlx:** `attention.Core` already routes to the backend fused op via
   `InternalFusedOpCaller` with a decomposed fallback, so it starts using cuDNN flash
   on CUDA once the backend implements it; SimpleGo/CPU keeps the decomposed path.
4. **lmkit-go:** `model.Attention` calls gomlx's fused SDPA instead of the manual
   einsum-softmax-einsum; the naive path stays as the fp32 parity reference and the
   CPU fallback. Validate parity (fp32) and re-measure throughput at 2048 on the 8GB
   GPU; expect the ~100x to mostly close and the memory wall to vanish (no T^2 held).

Steps 1-2 are the bulk and live in go-xla (jan's territory; he invited us to take a
stab). Verifying each step against the JAX reference HLO in `reference/` keeps it
honest. Open risks: GQA (numKVHeads != numHeads) may need the head-repeat before the
call or a config flag; bf16-only on the fast path; the softmax-stats residual must be
threaded from forward to backward.

## Outcome (measured, 2026-06-23)

Design (a) shipped and works, but the throughput model above was wrong about where
the gap lives. Record:

**Built and verified.** The three-layer custom-call path is implemented across the
forks and PR'd: `attention.FlashAttention` (gomlx, PR #427), `compute.CustomCallSpec`
+ Go backend (compute, PR #13), and `Function.CustomCall` that lowers to the StableHLO
`custom_call` (go-xla, PR #37). lmkit-go pins all three via `replace` (commits 350b408,
4872014). **Critical gotcha:** without the go-xla pin, `FlashAttention` catches
`ErrNotImplemented` in a `TryCatch` and *silently* falls back to naive fp32 attention
(materialized `[B,H,T,T]` scores) — it compiles and runs green but flash is a no-op.
Verify engagement by grepping the *optimized* HLO for `__cudnn$fmhaSoftmax`; 0 hits =
silent fallback. With all three pinned: optimized HLO shows `__cudnn$fmhaSoftmax` +
`__cudnn$fmhaSoftmaxBackward`, the fp32 score buffers are gone, and parity (fwd +
dQ/dK/dV) matches the decomposed reference to ~0.2-0.4% rel error on sm_86 (within bf16
tolerance). PR #427's own cuda parity tests pass.

**The "~100x is one missing kernel" thesis was wrong.** Flash took the lm-100m step
(B=2, T=2048, ga=1, ckpt=true) from ~875 to ~1237 tok/s — about **1.4x**, not the
10-20x the breakdown above attributed to attention. The remaining gap to the PyTorch
reference (~32k tok/s) is still ~26x, and it is NOT attention, memory, or batch:

- **B=2 is the reference's own config.** PyTorch hits 32k tok/s *at* B=2, and B=4 OOMs
  in PyTorch too (per `lmkit.toml`). So the micro-batch was never the bottleneck.
- The ~26x residual is **per-token kernel efficiency at fixed B=2** — the "general
  fusion gap" this doc estimated at 2-3x is actually the dominant factor. At 100% GPU
  util we sustain ~1 TFLOP/s vs PyTorch's ~19-35; the step is ~1300 small
  XLA kernels (norms, RoPE, SwiGLU, residuals, fp32<->bf16 converts, the 32k-vocab
  LM-head) where PyTorch runs far fewer, fatter kernels.

**Memory work that did NOT help (negative results, all reverted).** The LM-head
backward (`dH = dLogits @ table`, K=V=32000) lowers as an XLA split-K gemm that
materializes a ~1.5 GB `[128,768,B*T]` partial-sum buffer, the ceiling that pins B=2.
Three attempts to remove it: (1) an `OptimizationBarrier` on `dLogits` — barrier VJP is
pass-through, did not redirect the lowering, regressed to 704 tok/s; (2) `TiedLogits`
as an explicit 2-D matmul — still split-K (just reshaped), regressed B=2 to OOM;
(3) chunked online-softmax cross-entropy (`TiedLogitsCrossEntropyChunked`) — numerically
exact (parity + overfit pass) and it *did* eliminate the split-K buffer, but the unrolled
chunks' activations replaced it one-for-one (no net memory win) and it regressed to 560
tok/s. All reverted: unpinning B=2 was the wrong goal, and this mirrors the chunked-
attention lesson above (online-softmax as HLO ops does not produce a kernel win).

**Next:** per-kernel profiling at B=2 (nsys/CUPTI or XLA's profiler trace) to locate the
26x by actual kernel time, instead of inferring from buffer sizes and kernel counts.
