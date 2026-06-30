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
- gomlx: delete `flash.go` (no replacement helper — see Contract D); `WithFusion` + `useFusion` through `Core`; `WithSeqLens`; generalize tests over `TestOfficialBackends` + `backendSupportsFusion` + `GetOfficialBackend`; bench fixes (`FinalizeAll`, `for range`, `TestFusionThroughput`, flags not env).
- lmkit-go: migrate `model/attention.go` `FlashAttention(...)` → `InternalFusedOpCaller(fused, decomposed)` over the scope-free graph primitives (the deletion in gomlx breaks this build; same stage). See Contract F.
- go-huggingface: thread seqlens into transformer attention (kept in-scope per Guy 2026-06-29, though Jan offered to take it).

**Stage 2 — additive attention bias only (bf16). Dropout, bias+dropout, and FP8 are all CUT (decided 2026-06-30): no caller needs them, and fused dropout has no local use. They stay `NotImplemented` seams.**
- compute: add the `Bias` field + reference (additive before softmax) + direct unit test. NO dropout fields.
- go-xla: `selectFMHAVariant` adds the bias branch (`__cudnn$fmhaScaleBiasSoftmax`/`…Backward`); bias operand set; `[cuda]` exec test. NO dropout/bias+dropout targets.
- gomlx: new `WithAttentionBias(*Node)` (distinct from the existing `UseProjectionBias`); thread bias to `Core` as a `*Node` (built into the fused config in the fused closure, added to scores in the decomposed fallback — same pattern as seqlens). NO dropout fused-gate change.

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
	// Selects the fmhaScaleBias* variant in the xla backend. nil = unused.
	Bias Value
}
```

Stage 1 adds the `QuerySeqLen`/`KeyValueSeqLen` fields only; Stage 2 adds `Bias`. (Dropout fields are cut — see Staging.)

The forward/VJP **method signatures on `FusedOps` do not change** — all new params ride inside `*ScaledDotProductAttentionConfig`. FP8 is selected by the dtype of `query/key/value` (float8_e4m3fn / float8_e5m2), not a config field; fp8 is paused (see Contract C).

**The CPU `go`-backend SDPA capability stays `false`.** It is disabled on purpose — `capabilities.go:147` documents that the fused CPU path is ~3x slower than the SIMD matmul + decomposed path; the maintainer turned it off pending a SIMD fused kernel. Do **not** flip it to make tests run. The reference impl in `internal/gobackend/fusedops/sdpa.go` is extended for correctness and exercised by **direct unit tests on the `fusedops` functions**, not through the capability-gated backend path. On CPU, `BackendFusedScaledDotProductAttention` therefore still returns `ErrNotImplemented` → decomposed fallback. Re-enable only alongside a SIMD fused kernel, or if Jan asks.

CPU `go` backend reference (`internal/gobackend/fusedops/sdpa.go`) support matrix (tested directly, capability stays off):
- QuerySeqLen/KeyValueSeqLen [S1]: implement (build padding mask from lengths).
- Bias [S2]: implement (additive before softmax).
- FP8 input dtype: return `ErrNotImplemented` (paused; never wired on CPU).
- Dropout: cut (not a config field; no reference). `NotImplemented` if ever requested.

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
| Bias set [S2] | `__cudnn$fmhaScaleBiasSoftmax` | `__cudnn$fmhaScaleBiasSoftmaxBackward` |
| else | `__cudnn$fmhaSoftmax` | `__cudnn$fmhaSoftmaxBackward` |

Dtype gate: all wired variants accept `float16`/`bfloat16` only. Anything else → `ErrNotImplemented`.

**Dropout / bias+dropout cut** (decided 2026-06-30): the `__cudnn$fmha*Dropout` targets are not wired. A nonzero dropout request (there is no dropout config field) never reaches here. Bias combines with causal and (where the kernel allows) seqlens via `mask_type`; a bias+seqlen combination the kernel can't fuse falls through to `ErrNotImplemented` → decomposed.

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

// [S2] WithAttentionBias supplies an additive attention-score bias [B,H,S,Skv]
// (ALiBi / relative-position). DISTINCT from UseProjectionBias (the Q/K/V dense
// bias). Populates ScaledDotProductAttentionConfig.Bias. nil = unused.
func (b *MultiHeadAttentionBuilder) WithAttentionBias(bias *Node) *MultiHeadAttentionBuilder
```

`attention.Core` gains trailing params appended after `scoreSoftCap` (keeps the single production call site + test calls' positional args stable): `useFusion bool` (Stage 1, Task 2) and the seqlens as graph nodes `querySeqLen, keyValueSeqLen *Node` (Stage 1, Task 4). The existing `Core` fused branch additionally gates on `useFusion`.

**Critical: Core takes the seqlens as `*Node`, not a pre-built `compute.Value` config.** Both the fused and decomposed paths need the seqlens in *different* forms, and a single materialized mask cannot serve both (a materialized mask passed to the fused path trips go-xla's `flashSupported` `mask != nil` rejection → no fusion). So Core builds each representation internally: the FUSED closure calls `core/graph.NewSeqLenAttentionConfig(querySeqLen, keyValueSeqLen)` (`*Node`→`compute.Value` via in-package `outputOps[0]`) and passes the config with `attentionMask=nil` (so it fuses, PADDING/PADDING_CAUSAL); the DECOMPOSED fallback builds a boolean padding mask from the same `*Node`s (Iota/LessThan, AND-combined with causal) so it masks correctly. `GOMLX_FUSION` env stays as a global override in `InternalFusedOpCaller`; `WithFusion(false)` is the per-call override. (S2 bias follows the same pattern — passed to Core as a `*Node`, built into the config inside the fused closure and added to the decomposed scores, not materialized up in `doneInternal`.)

**No `SimpleAttention` helper** (decided 2026-06-29, on upstream merits — NOT a downstream consideration). Jan only hedged it ("if the use case is useful, we could..."). Any concrete version either (a) delegates to the builder/`Core` and therefore needs a `*model.Scope`, or (b) reimplements the fused-or-decomposed logic standalone — which is a *parallel path*, the exact duplication that deleting `flash.go` removes. The real primitives already cover the need: the `MultiHeadAttention` builder for the full path, and the scope-free `graph.BackendFusedScaledDotProductAttention` + `graph.InternalFusedOpCaller` for the bare fused-or-decomposed op. Don't add a third surface; if Jan wants the convenience helper he can shape it (builder-preconfiguring) on his own terms. The PR notes this as a deliberate omission.

**[S2] bias threading.** `WithAttentionBias(bias *Node)` passes the bias to `Core` as a trailing `*Node` param (same pattern as seqlens): the FUSED closure puts it in the config (`Bias` field → cuDNN `fmhaScaleBiasSoftmax`); the DECOMPOSED fallback adds it to the scores before softmax. `Core`'s `!dropoutActive` gate is UNCHANGED — dropout fusion is cut (no dropout config field). Bias is independent of `WithFusion`/seqlens (it may combine with causal; a bias+seqlen combo the kernel can't fuse falls back to decomposed, which handles both).

Capability check for tests (replaces `isCUDABackend`): a backend "supports fusion" iff a probe `BackendFusedScaledDotProductAttention` on a tiny causal bf16 input does not return `ErrNotImplemented`. (On CPU this is always false — capability stays off — so the CPU run takes the decomposed path; that is correct and intended.) Add `testutil.GetOfficialBackend(name string) compute.Backend` (returns the named official backend if present in the system, else skips) for the one `xla:cuda` specific test.

## Contract E — `go-huggingface` consumer (plan 04) [S1]

Find the transformer attention call sites (they call `attention.MultiHeadAttention(...).…Done()`), add an optional seqlen path: where the model already computes an attention/padding mask from token lengths, pass those lengths through `WithSeqLens` instead of (or in addition to) materializing the mask, guarded so models without lengths are unchanged.

## Contract F — `lmkit-go` consumer (Stage 1, same stage as the gomlx deletion)

`lmkit-go/model/attention.go:47` calls `attention.FlashAttention(q, k, v, scale)`, which gomlx Stage 1 deletes. With no upstream `SimpleAttention`, lmkit migrates to the scope-free graph primitives directly (lmkit is scope-free and already owns a decomposed branch):

```go
out = g.InternalFusedOpCaller(
    func() *g.Node { // fused
        return g.BackendFusedScaledDotProductAttention(
            q, k, v, nil, nH, nH, compute.AxesLayoutBSHD, scale, true /*causal*/, nil)
    },
    func() *g.Node { return decomposedAttention(q, k, v, scale) }, // lmkit's existing else-branch, factored out
)
```

Factor lmkit's current decomposed else-branch (attention.go:48-66) into `decomposedAttention(q,k,v,scale)` so both the fused fallback and the non-bf16 path share it (removes lmkit's own duplication). `InternalFusedOpCaller` runs `fused`, catches `compute.IsNotImplemented`, falls back to `decomposed` — exactly what `FlashAttention` did internally. Keep the `UseFlashAttention && bf16` guard selecting fused-vs-direct-decomposed. This introduces no new gomlx API and no parallel path. Verify lmkit's existing parity test still passes (CPU → fallback) and, on the CUDA host, the bf16 path fuses. (`InternalFusedOpCaller`/`BackendFusedScaledDotProductAttention` carry an "Internal"/"Backend" prefix — they are gomlx's documented escape hatch, the same one `FlashAttention` used; acceptable from a consumer.)

---

## Verification gates (per plan)

- **01 compute:** `go test ./...` green on Mac. New seqlen/bias reference code is covered by **direct unit tests on the `fusedops` functions** (capability stays `false`; the backendtest SDPA group stays skipped, as it is today). fp8 → `ErrNotImplemented`.
- **02 go-xla:** `go test ./...` green on Mac for non-CUDA (CustomCallV2 rendering, fallback). **[cuda]** `go test ./compute/xla ./pjrt -run FMHA|Flash` green under `xla:cuda` for each wired variant (S1: standard + seqlen; S2: + bias).
- **03 gomlx:** `go test ./ml/layers/attention/...` green on Mac (decomposed + fallback + `WithFusion(false)`). **[cuda]** the `xla:cuda` fusion-parity test green.
- **lmkit-go (S1):** `go build ./...` green after the gomlx deletion; existing attention parity test green; **[cuda]** bf16 path still fuses.
- **04 go-huggingface:** `go test ./...` green on Mac; one model integration test asserting seqlen path produces output within tolerance of the mask-matrix path.
