# Fused Attention Upstreaming — Shared Design Contract

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute the per-repo plans (01–05). This file is the cross-repo API contract every plan depends on; read it first. Steps in the per-repo plans use checkbox (`- [ ]`) syntax.

**Goal:** Land janpfeifer's review on PRs gomlx/compute#13, gomlx/go-xla#37, gomlx/gomlx#427, plus a new go-huggingface PR — generalizing flash attention into a "fused" abstraction with seqlen masking and the full cuDNN fmha variant set.

**Through-line (Jan's core ask):** Nothing flash-/CUDA-specific leaks above the `go-xla` backend. The cross-backend `compute` surface and the `gomlx` layers know only "FusedSDPA: supported or `ErrNotImplemented`". The `__cudnn$fmha*` target mapping lives entirely in `go-xla`.

## Repo dependency order (do NOT reorder)

```
compute#13  (interface: carry new params)
   └─> go-xla#37   (xla backend: CustomCallV2 + seqlen + variant dispatch)
          └─> gomlx#427   (layers: WithFusion gate, seqlen builder, generalized tests)
                 ├─> lmkit-go   (consumer: migrate the deleted FlashAttention call site)
                 └─> go-huggingface  (consumer: thread seqlens into transformer attention)
```

Each downstream repo consumes the upstream repo via a local `replace` directive (already present in the forks). After committing an upstream repo locally, the downstream `go.mod` replace points at the local checkout, so no push is needed to integrate.

## Staging (decided 2026-06-29)

The work ships in two stages. **Do Stage 1 in full — review — then Stage 2.** Each stage is independently green and reviewable.

**Stage 1 — strict refactor to Jan's proposed API. No new attention variants.**
- compute: add `QuerySeqLen`/`KeyValueSeqLen` config fields only; reference honors them; **CPU SDPA capability stays `false`** (see Contract A note); reference tested directly.
- go-xla: sync branch; `CustomCallV2`; route the existing standard `__cudnn$fmhaSoftmax` target through it; seqlen `mask_type`; `client.Plugin().IsCUDA()` test; relax `flashSupported` for seqlens. No bias/dropout dispatch.
- gomlx: delete `flash.go`; `WithFusion` + `useFusion` through `Core`; `WithSeqLens`; `SimpleAttention`; generalize tests over `TestOfficialBackends` + `backendSupportsFusion` + `GetOfficialBackend`; bench fixes (`FinalizeAll`, `for range`, `TestFusionThroughput`, flags not env).
- lmkit-go: migrate `model/attention.go` `FlashAttention(...)` → `SimpleAttention(...)` (the deletion in gomlx breaks this build; same stage).
- go-huggingface: thread seqlens into transformer attention (kept in-scope per Guy 2026-06-29, though Jan offered to take it).

**Stage 2 — added variants: bias, dropout, bias+dropout (all bf16). FP8 stays paused.**
- compute: add `Bias`/`DropoutRate`/`DropoutSeed`/`DropoutOffset` fields + reference + direct unit tests.
- go-xla: `selectFMHAVariant` bias/dropout/bias+dropout branches; per-variant operand sets; `[cuda]` exec tests.
- gomlx: new `WithAttentionBias(*Node)` (distinct from the existing `UseProjectionBias`); **relax `Core`'s `!dropoutActive` fused-branch gate** so it fuses with dropout when the backend supports fused dropout; thread bias + dropout seed/offset into the config.

Within each per-repo plan, every task is tagged **[S1]** or **[S2]**.

## Global Constraints (apply to every task in every per-repo plan)

- **Go 1.26**, module paths: `github.com/gomlx/compute`, `github.com/gomlx/go-xla`, `github.com/gomlx/gomlx`, `github.com/gomlx/go-huggingface`.
- **No push, no PR.** Every plan ends at `git commit` on the PR branch in the local fork. Guy reviews all diffs manually before anything reaches a remote. (Branches: compute=`flash-customcall`, go-xla=`flash-attention`, gomlx=`flash-attention`, go-huggingface=new `fused-seqlen`.)
- **CUDA-gated tests run on the CUDA host, not the Mac.** CI has no CUDA. Local Mac validation covers the `go` (CPU) backend reference and all non-fused fallback paths. Any task whose test needs cuDNN is marked **[cuda]** and its verification step runs there (build natively on the CUDA host, `GOMLX_BACKEND=xla:cuda`, serialize GPU jobs if the card is shared).
- **Fallback is the contract.** Any unsupported config (wrong dtype, unsupported variant, unsupported layout) returns a wrapped `compute.ErrNotImplemented`; the caller falls back to the decomposed path. Never panic on an unsupported-but-valid request.
- Voice rules for commits: terse, verb-first, no em/en dashes, keep each repo's existing prefix scheme (`attention:`, `compute/xla:`, `fused_ops:`), no Claude attribution.
- **go-xla `flash-attention` branch must be synced to `fork/flash-attention` first** (Jan merged his backend-options work in). This is Task 0 of plan 02.

---

## Contract A — `compute` interface (plan 01 produces, 02/03 consume)

Extend the existing options struct rather than the positional signature (keeps every existing caller compiling). Current struct has only `QuantizedMatmuls bool`.

```go
// ScaledDotProductAttentionConfig — optional fused-attention parameters.
// Backends that cannot honor a set field MUST return ErrNotImplemented so the
// caller falls back to the decomposed path. nil/zero means "unused".
type ScaledDotProductAttentionConfig struct {
	QuantizedMatmuls bool

	// [S1] QuerySeqLen, KeyValueSeqLen are optional per-batch actual sequence
	// lengths (int32 tensors, shape [B]). When set, the backend masks by
	// sequence length (padding mask) instead of a materialized [S,Skv] mask.
	// Combined with causal=true this is a padding-causal mask. nil = unused.
	QuerySeqLen, KeyValueSeqLen Value

	// [S2] Bias is an optional additive attention-score bias broadcast to
	// [B,H,S,Skv] (ALiBi / relative-position). NOT the Q/K/V projection bias.
	// Selects the fmhaScaleBias* variants in the xla backend. nil = unused.
	Bias Value

	// [S2] DropoutRate in [0,1); 0 disables. Nonzero selects the fmha*Dropout
	// variants. Seed/Offset feed the backend RNG (Value: int64 scalars).
	DropoutRate                float64
	DropoutSeed, DropoutOffset Value
}
```

Stage 1 adds the `QuerySeqLen`/`KeyValueSeqLen` fields only; Stage 2 adds `Bias`/`DropoutRate`/`DropoutSeed`/`DropoutOffset`.

The forward/VJP **method signatures on `FusedOps` do not change** — all new params ride inside `*ScaledDotProductAttentionConfig`. FP8 is selected by the dtype of `query/key/value` (float8_e4m3fn / float8_e5m2), not a config field; fp8 is paused (see Contract C).

**The CPU `go`-backend SDPA capability stays `false`.** It is disabled on purpose — `capabilities.go:147` documents that the fused CPU path is ~3x slower than the SIMD matmul + decomposed path; the maintainer turned it off pending a SIMD fused kernel. Do **not** flip it to make tests run. The reference impl in `internal/gobackend/fusedops/sdpa.go` is extended for correctness and exercised by **direct unit tests on the `fusedops` functions**, not through the capability-gated backend path. On CPU, `BackendFusedScaledDotProductAttention` therefore still returns `ErrNotImplemented` → decomposed fallback. Re-enable only alongside a SIMD fused kernel, or if Jan asks.

CPU `go` backend reference (`internal/gobackend/fusedops/sdpa.go`) support matrix (tested directly, capability stays off):
- QuerySeqLen/KeyValueSeqLen [S1]: implement (build padding mask from lengths).
- Bias [S2]: implement (additive before softmax).
- DropoutRate [S2]: implement deterministically from Seed/Offset (so CPU-vs-CPU tests are stable).
- FP8 input dtype: return `ErrNotImplemented` (paused; never wired on CPU).

## Contract B — `go-xla` CustomCallV2 (plan 02 produces, internal to go-xla)

Replace the string-layout `CustomCall` with a typed, versioned form. Keep version in the name (V2 = STATUS_RETURNING); a future V4 gets its own function.

```go
// stablehlo/customcall.go
// CustomCallV2 emits an XLA custom-call (API version 2, status-returning).
// operandLayouts/outputLayouts are minor-to-major dim orders, paired by index
// with operands/outputShapes. A nil entry (or nil slice) defaults to row-major
// (decreasing order over the operand/result rank).
func CustomCallV2(
	target string,
	backendConfig string,
	operands []*Value,
	operandLayouts [][]int,
	outputShapes []shapes.Shape,
	outputLayouts [][]int,
) ([]*Value, error)
```

The existing MLIR `dense<[...]>` string rendering moves *inside* CustomCallV2 (build it from `[][]int`). The old `CustomCall` string API is removed; `compute/xla/ops.go`'s `customCall` wrapper and `flash.go` switch to CustomCallV2.

## Contract C — `go-xla` fmha variant dispatch (plan 02 internal)

`flash.go` selects the cuDNN target from the config, in this precedence (forward / backward):

| Condition (first match wins) | Forward target | Backward target |
|---|---|---|
| Bias set + DropoutRate>0 | `__cudnn$fmhaScaleBiasSoftmaxDropout` | `…Backward` |
| Bias set | `__cudnn$fmhaScaleBiasSoftmax` | `…Backward` |
| DropoutRate>0 | `__cudnn$fmhaSoftmaxDropout` | `…Backward` |
| else | `__cudnn$fmhaSoftmax` | `__cudnn$fmhaSoftmaxBackward` |

Dtype gate: all wired variants accept `float16`/`bfloat16` only. Anything else → `ErrNotImplemented`.

**FP8 is paused, not wired** (decided 2026-06-29). The only fp8-capable hardware is Hopper/Ada (sm_8.9+); the local CUDA card is an RTX 3070 Ti (sm_8.6, Ampere) which has no fp8 tensor cores, so the F8 fmha path cannot execute on local hardware. Rather than ship an untested kernel, fp8 input dtype (`float8_e4m3fn`/`float8_e5m2`) falls through the dtype gate to `ErrNotImplemented` → decomposed fallback. The dispatch table above is the extension point: adding the `__cudnn$fmhaSoftmaxF8` / `…BackwardF8` row later (on a Hopper box, or by janpfeifer) is a pure addition, not a refactor. PR note hands fp8 to Jan explicitly.

Masking: `mask_type` in backend_config is `CAUSAL` (causal, no seqlens), `PADDING` (seqlens, no causal), `PADDING_CAUSAL` (both), or `NO_MASK`. Seqlen tensors are appended as operands when `mask_type` includes PADDING.

## Contract D — `gomlx` layers API (plan 03 produces, 04 consumes)

Builder methods follow the existing `WithX` convention (`WithPreProjected`, `WithQueryKeyScale`, `WithDropout`, `WithCausalMask`). The fusion toggle is therefore `WithFusion`, not `UseFusion` — matching the convention and Jan's inline comment.

```go
// ml/layers/attention/multiheadattention.go

// [S1] WithFusion controls whether the backend fused path may be used (default true).
// When false, Core always takes the decomposed path. Threaded into Core.
func (b *MultiHeadAttentionBuilder) WithFusion(enabled bool) *MultiHeadAttentionBuilder

// [S1] WithSeqLens supplies per-batch actual sequence lengths (int32 [B] nodes) for
// padding masking. Mutually exclusive with an explicit queryKeyMatrixMask.
func (b *MultiHeadAttentionBuilder) WithSeqLens(querySeqLen, keyValueSeqLen *Node) *MultiHeadAttentionBuilder

// [S1] SCOPE-FREE thin helper (decided 2026-06-29, option A). Fuses when the
// backend supports it, decomposes otherwise — implemented directly over
// InternalFusedOpCaller(fused, decomposed), NOT the MultiHeadAttention builder.
// The pre-projected, externally-weighted drop-in for the deleted FlashAttention:
// backend-generic (no "flash" in the name), causal, equal q/kv heads (caller
// repeats KV first), NOT a parallel full API. lmkit-go — scope-free by design,
// no model.Scope anywhere — is its first and motivating caller.
//
// DIVERGES from Jan's inline sketch (which took a *model.Scope and returned a
// *MultiHeadAttentionBuilder). Flag in the PR. Rationale: pre-projected inputs
// have no projection variables, so a scope is pure ceremony, and the only real
// caller is scope-free; forcing a throwaway scope reads worse than this helper.
func SimpleAttention(query, key, value *Node, scale float64) *Node

// [S2] WithAttentionBias supplies an additive attention-score bias [B,H,S,Skv]
// (ALiBi / relative-position). DISTINCT from UseProjectionBias (the Q/K/V dense
// bias). Populates ScaledDotProductAttentionConfig.Bias. nil = unused.
func (b *MultiHeadAttentionBuilder) WithAttentionBias(bias *Node) *MultiHeadAttentionBuilder
```

`attention.Core` gains a trailing `useFusion bool` param (appended after `scoreSoftCap` — keeps the single production call site + test calls' positional args stable). The existing `Core` fused branch (attention.go:268-280) additionally gates on `useFusion`. `GOMLX_FUSION` env stays as a global override in `InternalFusedOpCaller`; `WithFusion(false)` is the per-call override.

**[S2] dropout fused-gate.** Core's fused branch is currently gated `!dropoutActive` (active dropout forces the decomposed path). The dropout variant relaxes this: when the backend supports fused dropout, Core may fuse *with* dropout, threading the rate + seed/offset into the config. This gate change is Stage 2, not Stage 1.

Capability check for tests (replaces `isCUDABackend`): a backend "supports fusion" iff a probe `BackendFusedScaledDotProductAttention` on a tiny causal bf16 input does not return `ErrNotImplemented`. (On CPU this is always false — capability stays off — so the CPU run takes the decomposed path; that is correct and intended.) Add `testutil.GetOfficialBackend(name string) compute.Backend` (returns the named official backend if present in the system, else skips) for the one `xla:cuda` specific test.

## Contract E — `go-huggingface` consumer (plan 04) [S1]

Find the transformer attention call sites (they call `attention.MultiHeadAttention(...).…Done()`), add an optional seqlen path: where the model already computes an attention/padding mask from token lengths, pass those lengths through `WithSeqLens` instead of (or in addition to) materializing the mask, guarded so models without lengths are unchanged.

## Contract F — `lmkit-go` consumer (Stage 1, same stage as the gomlx deletion)

`lmkit-go/model/attention.go:47` calls `attention.FlashAttention(q, k, v, scale)`, which gomlx Stage 1 deletes. Migrate it to the scope-free `attention.SimpleAttention(q, k, v, scale)` (returns `*Node`; pre-projected, bf16, causal — the same path `FlashAttention` took, now via `InternalFusedOpCaller` fused-or-decomposed). The `UseFlashAttention && bf16` guard collapses: fusion is automatic when the backend supports it, decomposed otherwise. Verify lmkit's existing attention parity test still passes (CPU decomposed) and, on the CUDA host, that the bf16 path still fuses.

---

## Verification gates (per plan)

- **01 compute:** `go test ./...` green on Mac. New seqlen/bias/dropout reference code is covered by **direct unit tests on the `fusedops` functions** (capability stays `false`; the backendtest SDPA group stays skipped, as it is today). fp8 → `ErrNotImplemented`.
- **02 go-xla:** `go test ./...` green on Mac for non-CUDA (CustomCallV2 rendering, fallback). **[cuda]** `go test ./compute/xla ./pjrt -run FMHA|Flash` green under `xla:cuda` for each wired variant (S1: standard + seqlen; S2: + bias/dropout).
- **03 gomlx:** `go test ./ml/layers/attention/...` green on Mac (decomposed + fallback + `WithFusion(false)`). **[cuda]** the `xla:cuda` fusion-parity test green.
- **lmkit-go (S1):** `go build ./...` green after the gomlx deletion; existing attention parity test green; **[cuda]** bf16 path still fuses.
- **04 go-huggingface:** `go test ./...` green on Mac; one model integration test asserting seqlen path produces output within tolerance of the mask-matrix path.
