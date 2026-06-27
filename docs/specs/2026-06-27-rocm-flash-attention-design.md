# ROCm flash attention for lmkit-go (AOTriton via XLA custom_call)

Status: design. Feasibility spikes passed (2026-06-27).

## Context

lmkit-go runs on the AMD R9700 (gfx1201/RDNA4) via the jax-rocm PJRT plugin, but
attention falls back to the decomposed graph: XLA-ROCm emits no fused-attention
custom_call (confirmed by lowering `jax.nn.dot_product_attention` on the card, zero
custom_calls). That makes ROCm ~6x slower than the CUDA arm (4.7k vs 28.8k tok/s);
the entire gap is flash attention.

The CUDA path gets flash because XLA ships the cuDNN handler and emits a custom_call
to `__cudnn$fmhaSoftmax`. ROCm has no equivalent in XLA, but the flash *kernel* for
gfx1201 exists: AOTriton, the same library PyTorch's ROCm `scaled_dot_product_attention`
uses. The goal is to wire that existing kernel into our XLA stack, mirroring the typed
`FusedScaledDotProductAttention` + `...VJP` design already used for cuDNN, gated on
ROCm. Lands in `gomlx/go-xla` so every GoMLX-on-ROCm user benefits.

Out of scope: upstreaming a fusion pass to OpenXLA (a separate, heavier effort);
non-causal / bias / dropout variants; dtypes other than bf16.

## Feasibility (spiked 2026-06-27)

- **FFI dispatch on the prebuilt plugin — green.** A minimal FFI handler (device
  memcpy on the HIP stream), compiled with the rocm-sdk `amdclang++` against
  `xla/ffi/api/ffi.h`, registered via the FFI mechanism under platform `ROCM`, ran a
  `custom_call` on the R9700 and returned output == input. The prebuilt jax-rocm PJRT
  plugin dispatches to handlers we register and hands them the HIP stream + device
  buffers. No custom plugin/XLA build required.
- **AOTriton kernel — green.** `libaotriton_v2.so` (0.11.0) ships a public C++ API
  (`aotriton::v2::flash::attn_fwd` / `attn_bwd` in `aotriton/flash.h`), gfx1201 is in
  its compiled kernel targets, and PyTorch's flash forward runs on the card at the
  lm-100m shape `[2,12,2048,64]` causal bf16.
- **Toolchain — green.** `amdclang++` (clang-23), HIP headers, and the XLA FFI headers
  are all present in the rocm-sdk / jaxlib wheels.

## Architecture

Mirror the cuDNN path. One genuinely new piece: we supply the kernel *handler*,
because XLA ships cuDNN's but not an AMD one.

```
gomlx attention.Core  (unchanged: typed FusedSDPA + VJP, decomposed fallback)
        │  compute.Backend.FusedScaledDotProductAttention[VJP]
        ▼
go-xla  FusedScaledDotProductAttention[VJP]  (ROCm branch, new)
        │  emits stablehlo.custom_call("lmkit$flash_fwd" / "_bwd", backend_config)
        ▼
go-xla  FFI handler (C++/cgo, new)  ── registered at backend init via PJRT FFI ext
        │  unpacks buffers + HIP stream + attrs
        ▼
AOTriton attn_fwd / attn_bwd  ──►  gfx1201 flash kernels
```

`compute` and `gomlx` need no change: the typed interface and the `attention.Core`
fallback already exist from the cuDNN work. Everything is in `go-xla`.

## Components

1. **ROCm emitter** (`compute/xla/flash.go`, Go). Add an `IsROCm` branch to
   `FusedScaledDotProductAttention` / `...VJP`. On ROCm, emit a `custom_call` to our
   own target (`lmkit$flash_fwd` / `lmkit$flash_bwd`) with a `backend_config` carrying
   the params AOTriton needs (head counts, head_dim, scale, causal, dtype). Forward is
   multi-output `(output, softmax_lse)`; the VJP threads `softmax_lse` into the
   backward. Add an `IsROCm` helper to `pjrt/plugins.go` (today only `IsCUDA`/`IsCPU`;
   `device.platform` reports `gpu`, canonical platform is `ROCM`).
2. **FFI handler** (`compute/xla/rocm_flash_handler.cc` + cgo, the new part). C++
   functions bound via `XLA_FFI_DEFINE_HANDLER_SYMBOL` that take
   `Ctx<PlatformStream<hipStream_t>>` + the operand/result `AnyBuffer`s + decoded
   attrs, fill the AOTriton `attn_fwd_params` / `attn_bwd_params`, and call AOTriton on
   the stream. The forward writes the LSE buffer; the backward consumes it.
3. **Handler registration** (`compute/xla/` backend init). At ROCm backend
   construction, register the two handlers with the plugin through the PJRT FFI
   extension (`XLA_FFI_Handler_Register`), the Go/cgo equivalent of the spike's
   `jax.ffi.register_ffi_target`. Walk the plugin's `PJRT_Api` extension chain to find
   the FFI extension.
4. **AOTriton binding + build.** cgo links `libaotriton_v2` + AOTriton headers. Bundle
   or fetch `libaotriton_v2.so` the way go-xla already fetches the CUDA PJRT plugin, so
   it is transparent for downstream users; build tag / runtime guard so non-ROCm builds
   do not require it.
5. **Capability gating + fallback.** Advertise `FusedScaledDotProductAttention` in the
   ROCm backend capabilities only when AOTriton is present and the plugin is ROCm;
   otherwise return `ErrNotImplemented` so `attention.Core` keeps the decomposed path.

## Data flow and the LSE detail

The cuDNN path already threads softmax stats (log-sum-exp) from forward into the
backward custom_call; the typed interface returns `softmaxStats` for exactly this. The
one ROCm-specific care: AOTriton's LSE layout/shape and the `attn_bwd_params` field
expectations must match what the forward produced. Forward output and LSE are produced
by `attn_fwd`; the backward takes `q,k,v,output,LSE,dOutput` and returns `dq,dk,dv`,
same operand set as the cuDNN backward. bf16 in/out, fp32 LSE.

## Error handling

- AOTriton missing at runtime, or non-ROCm plugin, or unsupported config
  (non-causal, explicit mask, non-bf16, head_dim AOTriton does not cover): return
  `ErrNotImplemented`; `attention.Core` falls back to decomposed (already wired).
- AOTriton call returns non-success: surface as an FFI `Error`, fail the op (do not
  silently produce garbage).

## Testing

- **Parity** (the gate): forward + gradient parity vs the decomposed `Core` attention
  on the R9700, to bf16 tolerance, at the lm-100m shape and head dims 64 and 128.
  Reuse the existing flash parity tests (`TestAttentionFlash*`) which already run
  whatever backend `GOMLX_BACKEND` selects, now `xla:rocm`.
- **Fallback**: on a non-ROCm / AOTriton-absent build, the same tests pass via
  decomposed.
- **Throughput**: re-run the lm-100m step benchmark on the R9700; expect a multi-x
  jump from the 4.7k tok/s decomposed baseline.

## Phasing

1. **Forward**: emitter + FFI handler + registration + `attn_fwd` binding; forward
   parity on the R9700.
2. **Backward**: `attn_bwd` binding + LSE threading through the VJP; gradient parity.
3. **Packaging**: bundle `libaotriton`, capability gating, graceful fallback, build
   tags so CUDA/CPU builds are unaffected.
4. **Validate + measure**: parity green, then the lm-100m throughput run on the R9700.

## Residual risks

- **PJRT FFI extension from Go.** The spike registered via jaxlib; go-xla must register
  through the PJRT FFI extension directly. Standard, but the first integration step to
  prove. If the prebuilt plugin does not expose the extension, registration would need
  the process-global XLA FFI registry instead (still no plugin rebuild).
- **AOTriton API stability + packaging size.** `libaotriton_v2` is large (compiled
  kernel images per gfx); bundling adds heft. Pin the version; document the gfx
  coverage.
- **attn params mapping.** Getting `attn_fwd_params`/`attn_bwd_params` (layout, LSE,
  scale, causal) exactly right is the fiddly part; the parity test is the guard.
