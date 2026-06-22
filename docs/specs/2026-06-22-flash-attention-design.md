# Flash attention for the GoMLX/XLA path

**Status:** design / research. No code yet. This is the justification and the design
fork; the central question (below) gates which design we actually build.

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

## Next step

Answer the central question empirically: install / locate JAX against the go-xla PJRT
plugin on a CUDA host (sm_86), run `jax.nn.dot_product_attention(implementation=
"cudnn")` forward and backward, dump the StableHLO and the optimized HLO, and confirm
(1) it emits a `__cudnn$fmha*` custom_call, (2) it compiles and runs on sm_86, (3) the
exact target name, operand layout, and `backend_config`. That confirms design (a) and
hands us the precise custom_call spec go-xla must emit.
