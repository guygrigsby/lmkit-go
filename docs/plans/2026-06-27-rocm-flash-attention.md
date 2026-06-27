# ROCm Flash Attention (AOTriton via XLA custom_call) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give lmkit-go flash attention on AMD ROCm (R9700 / gfx1201) by binding AOTriton's flash kernels into `gomlx/go-xla` through an XLA `custom_call` + FFI handler, mirroring the existing cuDNN typed `FusedScaledDotProductAttention` + VJP path, gated on ROCm.

**Architecture:** All changes are in `go-xla`. On a ROCm plugin, `FusedScaledDotProductAttention`/`...VJP` emit a `custom_call` to our own target (`lmkit$flash_fwd` / `lmkit$flash_bwd`); a C++ FFI handler (compiled with the rocm-sdk `amdclang++`, linking `libaotriton_v2`) is registered with the plugin at backend construction via the PJRT FFI extension, and dispatches to AOTriton `attn_fwd`/`attn_bwd` on the HIP stream. `compute` and `gomlx` are unchanged: the typed interface and `attention.Core` decomposed fallback already exist from the cuDNN work.

**Tech Stack:** Go + cgo, C++ (amdclang++ / clang-23), HIP, AOTriton `libaotriton_v2` 0.11.0, XLA FFI (`xla/ffi/api/ffi.h`), PJRT C API (vendored in `go-xla/pjrt/pjrt_c_api.h`), the jax-rocm 0.10.2 PJRT plugin.

## Global Constraints

- **Matching go-xla's exact style and patterns is paramount.** This lands as a PR in `gomlx/go-xla` (maintainer reviews for duplicated/misplaced code). Before writing any Go, re-read and mirror: `compute/xla/flash.go` (the cuDNN emitter, backend_config builders, layout constants, `IsCUDA` gating), `compute/xla/ops.go` (op method shape, `verifyAndCastValues`, `newNode`, `xslices.Map`), `compute/xla/xla.go` (capability set at backend construction), `pjrt/plugins.go` (`IsCUDA`/`IsCPU` via `isCuda`), and the cgo style in `pjrt/*.go` + `pjrt/common.c`. Errors use `errors.Wrapf(compute.ErrNotImplemented, ...)`. Comment density, naming and file layout follow go-xla, not lmkit-go or generic Go.
- **The ROCm flash path only RUNS on the R9700 (trig).** go-xla compiles on the Mac (the forks are at `/Users/guygrigsby/projects/forks/{compute,go-xla,gomlx}`, tied by `…/forks/go.work`), but every parity/throughput check runs on trig. Validate by pushing the go-xla fork branch + repinning lmkit-go, or by building natively on trig.
- **Scope: bf16, causal, BSHD, equal-or-GQA heads only.** Any other config returns `errors.Wrapf(compute.ErrNotImplemented, ...)` and falls back to decomposed (already wired in `attention.Core`). No bias, no dropout, no non-causal.
- **AOTriton version pinned to the one PyTorch-ROCm ships** (`libaotriton_v2.so.0.11.0`). gfx1201 is in its compiled targets; do not assume other gfx.
- **The gate at every phase is the existing parity tests** (`TestAttentionFlashMatchesDecomposed`, `TestAttentionFlashGradMatchesDecomposed` in the lmkit-go `model` package, plus the gomlx `ml/layers/attention` flash tests) run with `GOMLX_BACKEND=xla:rocm` on the R9700. Green parity vs decomposed is "done"; anything else is not.
- **Spike artifacts on trig** (reference, already working): `/tmp/ffi_probe.cc` (an FFI handler that dispatches on the R9700), AOTriton headers at `~/venvs/rocm/lib/python3.12/site-packages/torch/include/aotriton/` and lib `…/torch/lib/libaotriton_v2.so`, rocm-sdk `amdclang++` at `~/venvs/jaxrocm/lib/python3.12/site-packages/_rocm_sdk_core/lib/llvm/bin/amdclang++`, the XLA FFI headers at `~/venvs/jaxrocm/lib/python3.12/site-packages/jaxlib/include/xla/ffi/api/`.

## File structure (all in `go-xla`)

- `pjrt/pjrt_c_api.h` — **modify**: add the `PJRT_FFI_Extension` struct (the enum `PJRT_Extension_Type_FFI` already exists; the struct does not). Source of truth: the upstream PJRT C API header that ships with the jax-rocm plugin / jaxlib.
- `pjrt/ffi.go` + `pjrt/ffi.c`/`.h` — **create**: cgo to walk `PJRT_Api.extension_start` to the FFI extension and register a handler by name. One responsibility: PJRT FFI handler registration.
- `pjrt/plugins.go` — **modify**: add `IsROCm()` next to `IsCUDA`/`IsCPU`, and an `isRocm(name)` helper.
- `compute/xla/rocm_flash.go` — **create**: the ROCm emitter + handler registration wiring. Mirrors `compute/xla/flash.go`.
- `compute/xla/rocm_flash_handler.cc` + `.h` — **create**: the C++ FFI handlers (`lmkit_flash_fwd`/`_bwd`) that build AOTriton tensor views and call `attn_fwd`/`attn_bwd`.
- `compute/xla/flash.go` — **modify**: route `FusedScaledDotProductAttention`/`...VJP` to the ROCm path when `IsROCm()` (today they only handle `IsCUDA`).
- `compute/xla/xla.go` — **modify**: in the backend constructor, set the FusedSDPA capability for ROCm (when AOTriton is available) and register the handlers.

---

### Task 1: `IsROCm` plugin helper

**Files:**
- Modify: `pjrt/plugins.go` (next to `IsCUDA`/`IsCPU`, ~line 152–162)
- Test: `pjrt/plugins_test.go`

**Interfaces:**
- Produces: `func (p *Plugin) IsROCm() bool` and `func isRocm(name string) bool`.

- [ ] **Step 1: Write the failing test** — mirror any existing `isCuda`/name test in `pjrt/plugins_test.go`; if none, add:

```go
func TestIsRocm(t *testing.T) {
	for name, want := range map[string]bool{
		"rocm": true, "ROCM": true, "xla_rocm_plugin": true,
		"cuda": false, "cpu": false,
	} {
		if got := isRocm(name); got != want {
			t.Errorf("isRocm(%q)=%v want %v", name, got, want)
		}
	}
}
```

- [ ] **Step 2: Run it, verify it fails**

Run (from `pjrt/`, in the forks workspace): `go test ./pjrt/ -run TestIsRocm`
Expected: FAIL (`isRocm` undefined).

- [ ] **Step 3: Implement, mirroring `isCuda`/`IsCPU` exactly**

```go
// IsROCm attempts to guess if the plugin is based on AMD ROCm.
// It's not perfect.
func (p *Plugin) IsROCm() bool {
	return isRocm(p.name)
}
```

Add `isRocm` next to `isCuda` (same file/pattern as `isCuda`):

```go
func isRocm(name string) bool {
	return strings.Contains(strings.ToLower(name), "rocm")
}
```

- [ ] **Step 4: Run, verify pass**

Run: `go test ./pjrt/ -run TestIsRocm`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add pjrt/plugins.go pjrt/plugins_test.go
git commit -m "pjrt: add Plugin.IsROCm helper, mirroring IsCUDA"
```

---

### Task 2: PJRT FFI handler registration from Go (TOP RISK — first integration milestone)

This proves the prebuilt plugin lets go-xla (not jaxlib) register an FFI handler. The Python spike already proved the plugin *dispatches*; this proves the *Go registration path*. Do this before any AOTriton work — if it is red, the whole approach changes.

**Files:**
- Modify: `pjrt/pjrt_c_api.h` (add the `PJRT_FFI_Extension` struct)
- Create: `pjrt/ffi.go`, `pjrt/ffi.c`, `pjrt/ffi.h`
- Create (test fixture): `compute/xla/testdata/ffi_copy_probe.cc` (the memcpy handler from the spike, `/tmp/ffi_probe.cc` on trig)
- Test: `pjrt/ffi_rocm_test.go` (build-tagged / skipped off ROCm)

**Interfaces:**
- Produces: `func (p *Plugin) RegisterFFIHandler(targetName string, handler unsafe.Pointer) error` — registers `handler` (an `XLA_FFI_Handler*`) under `targetName` for this plugin's platform via the FFI extension.

- [ ] **Step 1: Add the FFI extension struct to the vendored header.** Copy the `PJRT_FFI_Extension` / `PJRT_FFI_Register_Args` definitions verbatim from the upstream `pjrt_c_api.h` that ships with the jax-rocm plugin (find it under the jaxlib include dir on trig). Match the existing extension-struct style already in `pjrt/pjrt_c_api.h`. Keep it minimal: only the FFI extension base + the register-handler entry.

- [ ] **Step 2: Write the cgo extension-chain walk + register** in `pjrt/ffi.c`/`.h`: given `PJRT_Api*`, walk `extension_start` (a linked list of `PJRT_Extension_Base{ size, type, next }`) until `type == PJRT_Extension_Type_FFI`, then call its register-handler function with `{ target_name, platform_name, handler }`. Mirror the cgo call style in `pjrt/common.c` / `pjrt/gen_api_calls.c`.

- [ ] **Step 3: Write `RegisterFFIHandler` in `pjrt/ffi.go`** wrapping the C call; platform name from the plugin (`"ROCM"`, the canonical name proven in the spike — `device.platform` reports `gpu`). Return a wrapped error on a missing extension so callers can fall back.

- [ ] **Step 4: Build the probe handler fixture on trig** (reuse `/tmp/ffi_probe.cc`, already verified) and write a ROCm-only Go test that loads the rocm plugin, calls `RegisterFFIHandler("copy_probe", &CopyProbe)`, runs a one-op `custom_call` program, and asserts output == input.

```go
//go:build rocm
// run only where GOMLX_BACKEND=xla:rocm and the plugin is present
func TestRegisterFFIHandlerDispatches(t *testing.T) { /* load plugin, register, run copy_probe, assert out==in */ }
```

- [ ] **Step 5: Run on trig.** Push the branch + build natively on trig (or repin). Run with `GOMLX_BACKEND=xla:rocm`, plugin path + `LD_LIBRARY_PATH` per `docs/backends.md`.
Expected: PASS (out == in), proving Go-side FFI registration + dispatch.

- [ ] **Step 6: Commit**

```bash
git add pjrt/pjrt_c_api.h pjrt/ffi.go pjrt/ffi.c pjrt/ffi.h pjrt/ffi_rocm_test.go compute/xla/testdata/ffi_copy_probe.cc
git commit -m "pjrt: register XLA FFI custom-call handlers via the PJRT FFI extension"
```

> **Gate:** if Step 5 cannot register/dispatch from Go (e.g. the plugin does not expose the FFI extension), STOP and reassess — the fallback is the process-global XLA FFI registry, still no plugin rebuild, but it changes Tasks 4–8. Do not proceed until this is green.

---

### Task 3: AOTriton forward handler (C++)

**Files:**
- Create: `compute/xla/rocm_flash_handler.cc`, `compute/xla/rocm_flash_handler.h`
- Build: a `Makefile`/script step compiling it to `libgoxla_rocm_flash.so` with `amdclang++`, linking `libaotriton_v2` (see Task 7 for packaging; here just compile + a smoke test).

**Interfaces:**
- Produces: C symbol `XLA_FFI_Handler* lmkit_flash_fwd` — an FFI handler with binding `Ctx<PlatformStream<hipStream_t>>().Arg(q).Arg(k).Arg(v).Ret(out).Ret(lse)` plus attrs (`scale: f64`, `causal: bool`, `num_heads`, `num_kv_heads`, `head_dim: i64`).

- [ ] **Step 1: Read `aotriton/flash.h` (exact path in Global Constraints).** The forward needs `attn_fwd_params{ Q,K,V (T4), Sm_scale (float), L (T2, the LSE output), Out (T4), causal_type (int8), … }` and `attn_fwd(params, attn_fwd_params::kVersion, AOTRITON_NS::Stream{hipStream_t}, nullptr)`. Construct AOTriton `T4`/`T2` views from the FFI `AnyBuffer`s (device ptr + dims; bf16 for Q/K/V/Out, fp32 for L). Set `causal_type` to the causal constant, leave bias/dropout/varlen empty.

- [ ] **Step 2: Write `lmkit_flash_fwd`** using the spike's handler skeleton (`/tmp/ffi_probe.cc`) for the FFI boilerplate (`XLA_FFI_DEFINE_HANDLER_SYMBOL`, `Ctx<PlatformStream<hipStream_t>>`, `AnyBuffer`), filling and calling `attn_fwd`. Return `ffi::Error` from the `hipError_t`.

- [ ] **Step 3: Compile on trig**

```
amdclang++ -std=c++17 -fPIC -shared -D__HIP_PLATFORM_AMD__ rocm_flash_handler.cc \
  -o libgoxla_rocm_flash.so -I<jaxlib/include> -I<rocm_sdk_core/include> \
  -I<torch/include> -L<torch/lib> -laotriton_v2
```
Expected: compiles; `nm -D` shows `lmkit_flash_fwd`.

- [ ] **Step 4: Standalone smoke test on trig** — a tiny C++/Python harness that calls `lmkit_flash_fwd` on `[2,12,2048,64]` bf16 causal inputs and checks the output is finite and matches a NumPy reference to bf16 tolerance (before wiring into go-xla). Keeps the kernel-binding bug surface separate from the Go integration.

- [ ] **Step 5: Commit**

```bash
git add compute/xla/rocm_flash_handler.cc compute/xla/rocm_flash_handler.h
git commit -m "compute/xla: AOTriton flash forward FFI handler (lmkit_flash_fwd)"
```

---

### Task 4: ROCm forward emitter + wire-up → forward parity

**Files:**
- Create: `compute/xla/rocm_flash.go` (emitter + registration, mirrors `compute/xla/flash.go`)
- Modify: `compute/xla/flash.go` (`FusedScaledDotProductAttention`: add the `IsROCm()` branch)
- Modify: `compute/xla/xla.go` (register `lmkit_flash_fwd` at construction when `IsROCm()` + AOTriton present; set the FusedSDPA capability)

**Interfaces:**
- Consumes: `Plugin.IsROCm()` (Task 1), `Plugin.RegisterFFIHandler` (Task 2), `lmkit_flash_fwd` (Task 3), the `customCall` helper + layout constants already in `compute/xla/flash.go`.
- Produces: `FusedScaledDotProductAttention` returns `(output, softmaxStats, err)` on ROCm via the custom_call.

- [ ] **Step 1: Confirm forward parity is currently red on ROCm** — run `TestAttentionFlashMatchesDecomposed` with `GOMLX_BACKEND=xla:rocm` on trig; it passes today only because it falls back to decomposed. Add an assertion / log that the fused path was taken (e.g. a capability check) so "fell back" can't masquerade as "passed".

- [ ] **Step 2: Write `rocm_flash.go`** mirroring `flash.go`: a `backend_config`/attrs builder for AOTriton (scale, causal, head counts, head_dim), layout constants (reuse BSHD/BHSD/stats from `flash.go`), and the emit via the existing private `customCall(target, apiVersion, backendConfig, operandLayouts, resultLayouts, outputShapes, operands...)`. Target `lmkit$flash_fwd`, outputs `(output BSHD bf16, lse [B,H,S] f32)`.

- [ ] **Step 3: Add the `IsROCm()` branch to `FusedScaledDotProductAttention`** in `flash.go`, mirroring the `IsCUDA` branch shape; same guard set (causal, no mask, BSHD), else `ErrNotImplemented`.

- [ ] **Step 4: Register the handler + set capability in `xla.go`**, mirroring how the FusedSDPA capability is set for CUDA: when `backend.plugin.IsROCm()` and AOTriton loads, `backend.plugin.RegisterFFIHandler("lmkit$flash_fwd", C.lmkit_flash_fwd)` and `capabilities.Operations[compute.OpTypeFusedScaledDotProductAttention] = true`.

- [ ] **Step 5: Run forward parity on trig**

Run: `GOMLX_BACKEND=xla:rocm … go test ./model/ -run TestAttentionFlashMatchesDecomposed -v` (on trig, against the repinned forks).
Expected: PASS, fused path taken, rel-error within bf16 tolerance at head dims 64 and 128.

- [ ] **Step 6: Commit**

```bash
git add compute/xla/rocm_flash.go compute/xla/flash.go compute/xla/xla.go
git commit -m "compute/xla: emit AOTriton flash forward on ROCm (FusedScaledDotProductAttention)"
```

---

### Task 5: AOTriton backward handler (C++)

**Files:**
- Modify: `compute/xla/rocm_flash_handler.cc`/`.h` (add `lmkit_flash_bwd`)

**Interfaces:**
- Produces: C symbol `XLA_FFI_Handler* lmkit_flash_bwd` — binding `Ctx<PlatformStream<hipStream_t>>().Arg(q).Arg(k).Arg(v).Arg(out).Arg(lse).Arg(dOut).Ret(dq).Ret(dk).Ret(dv)` + the same attrs.

- [ ] **Step 1: Read `attn_bwd_params` in `aotriton/flash.h`** — it needs `Q,K,V,Out,DO (dOutput),DK,DV,DQ (T4), L (T2, the LSE from forward), D (mutable LT2 delta), DQ_ACC (mutable LT4 fp32), Sm_scale, causal_type`. The `D` and `DQ_ACC` are scratch the handler must allocate on-device (use the FFI `ScratchAllocator` ctx, or `hipMallocAsync` on the stream and free after — match whichever AOTriton's PyTorch adapter does; see `torch/include/.../aotriton_adapter.h` on trig for the reference construction).

- [ ] **Step 2: Write `lmkit_flash_bwd`**, constructing the params and calling `attn_bwd(params, kVersion, Stream, nullptr)`.

- [ ] **Step 3: Compile on trig** (same command as Task 3 Step 3); `nm -D` shows `lmkit_flash_bwd`.

- [ ] **Step 4: Standalone smoke test** — call `lmkit_flash_bwd` and finite-check dq/dk/dv against a NumPy decomposed-attention gradient reference to bf16 tolerance.

- [ ] **Step 5: Commit**

```bash
git add compute/xla/rocm_flash_handler.cc compute/xla/rocm_flash_handler.h
git commit -m "compute/xla: AOTriton flash backward FFI handler (lmkit_flash_bwd)"
```

---

### Task 6: ROCm VJP emitter + LSE threading → gradient parity

**Files:**
- Modify: `compute/xla/rocm_flash.go` (the VJP emit)
- Modify: `compute/xla/flash.go` (`FusedScaledDotProductAttentionVJP`: `IsROCm()` branch)
- Modify: `compute/xla/xla.go` (register `lmkit_flash_bwd`)

**Interfaces:**
- Consumes: `lmkit_flash_bwd` (Task 5), the forward's `softmaxStats` (Task 4).
- Produces: `FusedScaledDotProductAttentionVJP` returns `(dQuery, dKey, dValue, err)` on ROCm.

- [ ] **Step 1: Add the VJP emit to `rocm_flash.go`** — operands `q,k,v,output,lse,dOutput`, target `lmkit$flash_bwd`, outputs `(dq,dk,dv BSHD bf16)`, mirroring the cuDNN backward emit in `flash.go`.

- [ ] **Step 2: Add the `IsROCm()` branch to `FusedScaledDotProductAttentionVJP`** in `flash.go`, threading the `softmaxStats` from the forward (same operand set as the cuDNN VJP).

- [ ] **Step 3: Register `lmkit_flash_bwd` in `xla.go`** alongside the forward handler.

- [ ] **Step 4: Run gradient parity on trig**

Run: `GOMLX_BACKEND=xla:rocm … go test ./model/ -run TestAttentionFlashGradMatchesDecomposed -v` (on trig).
Expected: PASS — dX/dWq/dWk/dWv/dWo within bf16 tolerance vs decomposed, at head dims 64 and 128.

- [ ] **Step 5: Commit**

```bash
git add compute/xla/rocm_flash.go compute/xla/flash.go compute/xla/xla.go
git commit -m "compute/xla: AOTriton flash backward VJP on ROCm (threads softmax LSE)"
```

---

### Task 7: Packaging — bundle libaotriton, build tags, graceful fallback

**Files:**
- Create: `compute/xla/rocm_flash_cgo.go` (the `#cgo` directives + the `//go:build` tag) and a `rocm_flash_stub.go` for non-rocm builds.
- Modify: build/fetch script (mirror how `pjrt/cuda_linux.go` locates/fetches the CUDA plugin) to locate `libaotriton_v2.so`.

**Interfaces:**
- Produces: on a build without AOTriton, `FusedScaledDotProductAttention`/`VJP` return `ErrNotImplemented` and the capability stays false (decomposed fallback).

- [ ] **Step 1: Split the cgo behind a build tag** so CUDA/CPU/Mac builds never need HIP/AOTriton: `//go:build rocm` on the cgo file, a stub (`//go:build !rocm`) returning `ErrNotImplemented`. Confirm `go build ./...` on the Mac (no rocm tag) is unaffected.

- [ ] **Step 2: Locate `libaotriton_v2.so` at runtime**, mirroring `cuda_linux.go`'s plugin-dir search; if absent, log once and leave the capability false. No hard dependency when the lib is missing.

- [ ] **Step 3: Verify the fallback** — on the Mac (`GOMLX_BACKEND=go`) and on trig without the rocm tag, `make check` and the parity tests pass via decomposed.

- [ ] **Step 4: Commit**

```bash
git add compute/xla/rocm_flash_cgo.go compute/xla/rocm_flash_stub.go
git commit -m "compute/xla: gate ROCm flash behind a build tag + AOTriton presence, decomposed fallback"
```

---

### Task 8: Validate parity + lm-100m throughput on the R9700

**Files:** none (validation).

- [ ] **Step 1: Full parity sweep on trig** — `TestAttentionFlash*` at head dims 64 and 128, batch 2, seq 2048, `GOMLX_BACKEND=xla:rocm`. All green vs decomposed.

- [ ] **Step 2: lm-100m throughput** — run the lm-100m step benchmark (`train` `TestFlashFullStep2048`) with the ROCm flash path on the R9700. Record tok/s vs the 4.7k decomposed baseline (expect a multi-x jump). Confirm peak VRAM drops (no materialized scores).

- [ ] **Step 3: Record the result** in `examples/lm-100m-en/README.md` run log (lab-agnostic) and update `docs/python-parity-gaps.md` ROCm line (decomposed → flash) and `docs/backends.md` (ROCm now has flash).

- [ ] **Step 4: Commit + open the go-xla PR** (separate ask) — a `gomlx/go-xla` PR mirroring the cuDNN-flash PR style, referencing `gomlx/gomlx#422`.

---

## Self-review

- **Spec coverage:** emitter (Task 4/6), FFI handler (Task 3/5), registration (Task 2), AOTriton binding (Task 3/5), capability gating + fallback (Task 7), LSE threading (Task 6), parity + throughput tests (Task 8), `IsROCm` (Task 1). All spec sections map to a task.
- **Risk-first:** the PJRT-FFI-from-Go gate is Task 2, before any AOTriton work, with an explicit STOP.
- **Style rule** restated in Global Constraints and referenced per task (mirror `flash.go`/`ops.go`/`xla.go`/`plugins.go`).
- **Types consistent:** `lmkit_flash_fwd`/`lmkit_flash_bwd`, `RegisterFFIHandler`, `IsROCm`, target names `lmkit$flash_fwd`/`_bwd` used identically across tasks.
- **Known non-placeholder gaps the implementer resolves by reading the named headers** (exact paths given): the AOTriton `T4`/`T2` view construction (`aotriton/flash.h` + `aotriton_adapter.h`), the `PJRT_FFI_Extension` struct (upstream `pjrt_c_api.h`), the backward scratch allocation. These are reads of real files, not invented APIs.
