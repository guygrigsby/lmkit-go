# Fused Attention Upstreaming — Shared Design Contract

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to execute the per-repo plans (01–04). This file is the cross-repo API contract every plan depends on; read it first. Steps in the per-repo plans use checkbox (`- [ ]`) syntax.

**Goal:** Land janpfeifer's review on PRs gomlx/compute#13, gomlx/go-xla#37, gomlx/gomlx#427, plus a new go-huggingface PR — generalizing flash attention into a "fused" abstraction with seqlen masking and the full cuDNN fmha variant set.

**Through-line (Jan's core ask):** Nothing flash-/CUDA-specific leaks above the `go-xla` backend. The cross-backend `compute` surface and the `gomlx` layers know only "FusedSDPA: supported or `ErrNotImplemented`". The `__cudnn$fmha*` target mapping lives entirely in `go-xla`.

## Repo dependency order (do NOT reorder)

```
compute#13  (interface: carry new params)
   └─> go-xla#37   (xla backend: implement variants + seqlen + CustomCallV2)
          └─> gomlx#427   (layers: UseFusion gate, seqlen builder, generalized tests)
                 └─> go-huggingface  (consumer: thread seqlens into transformer attention)
```

Each downstream repo consumes the upstream repo via a local `replace` directive (already present in the forks). After committing an upstream repo locally, the downstream `go.mod` replace points at the local checkout, so no push is needed to integrate.

## Global Constraints (apply to every task in every per-repo plan)

- **Go 1.26**, module paths: `github.com/gomlx/compute`, `github.com/gomlx/go-xla`, `github.com/gomlx/gomlx`, `github.com/gomlx/go-huggingface`.
- **No push, no PR.** Every plan ends at `git commit` on the PR branch in the local fork. Guy reviews all diffs manually before anything reaches a remote. (Branches: compute=`flash-customcall`, go-xla=`flash-attention`, gomlx=`flash-attention`, go-huggingface=new `fused-seqlen`.)
- **CUDA-gated tests run on trig, not the Mac.** CI has no CUDA. Local Mac validation covers the `go` (CPU) backend reference and all non-fused fallback paths. Any task whose test needs cuDNN is marked **[trig]** and its verification step runs there (build natively on trig, `GOMLX_BACKEND=xla:cuda`, wrap GPU work in `gputex run`).
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

	// Bias is an optional additive attention bias broadcast to [B,H,S,Skv].
	// Selects the fmhaScaleBias* variants in the xla backend. nil = unused.
	Bias Value

	// QuerySeqLen, KeyValueSeqLen are optional per-batch actual sequence
	// lengths (int32 tensors, shape [B]). When set, the backend masks by
	// sequence length (padding mask) instead of a materialized [S,Skv] mask.
	// Combined with causal=true this is a padding-causal mask. nil = unused.
	QuerySeqLen, KeyValueSeqLen Value

	// DropoutRate in [0,1); 0 disables. Nonzero selects the fmha*Dropout
	// variants. Seed/Offset feed the backend RNG (Value: int64 scalars).
	DropoutRate           float64
	DropoutSeed, DropoutOffset Value
}
```

The forward/VJP **method signatures on `FusedOps` do not change** — all new params ride inside `*ScaledDotProductAttentionConfig`. FP8 is selected by the dtype of `query/key/value` (float8_e4m3fn / float8_e5m2), not a config field.

CPU `go` backend reference (`internal/gobackend/fusedops/sdpa.go`) support matrix:
- Bias: implement (additive before softmax).
- QuerySeqLen/KeyValueSeqLen: implement (build padding mask from lengths).
- DropoutRate: implement deterministically from Seed/Offset (so CPU-vs-CPU tests are stable).
- FP8 input dtype: return `ErrNotImplemented`.

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
| FP8 input dtype | `__cudnn$fmhaSoftmaxF8` | `__cudnn$fmhaSoftmaxBackwardF8` |
| Bias set + DropoutRate>0 | `__cudnn$fmhaScaleBiasSoftmaxDropout` | `…Backward` |
| Bias set | `__cudnn$fmhaScaleBiasSoftmax` | `…Backward` |
| DropoutRate>0 | `__cudnn$fmhaSoftmaxDropout` | `…Backward` |
| else | `__cudnn$fmhaSoftmax` | `__cudnn$fmhaSoftmaxBackward` |

Dtype gate: standard/bias/dropout variants accept `float16`/`bfloat16` only; F8 variants accept `float8_e4m3fn`/`float8_e5m2`. Anything else → `ErrNotImplemented`.

Masking: `mask_type` in backend_config is `CAUSAL` (causal, no seqlens), `PADDING` (seqlens, no causal), `PADDING_CAUSAL` (both), or `NO_MASK`. Seqlen tensors are appended as operands when `mask_type` includes PADDING.

## Contract D — `gomlx` layers API (plan 03 produces, 04 consumes)

```go
// ml/layers/attention/multiheadattention.go
// UseFusion controls whether the backend fused path may be used (default true).
// When false, Core always takes the decomposed path. Threaded into Core.
func (b *MultiHeadAttentionBuilder) UseFusion(enabled bool) *MultiHeadAttentionBuilder

// WithSeqLens supplies per-batch actual sequence lengths (int32 [B] nodes) for
// padding masking. Mutually exclusive with an explicit queryKeyMatrixMask.
func (b *MultiHeadAttentionBuilder) WithSeqLens(querySeqLen, keyValueSeqLen *Node) *MultiHeadAttentionBuilder

// Optional thin helper Jan suggested. Preconfigures MultiHeadAttention for the
// pre-projected case so existing callers transparently get fusion.
func SimpleAttention(scope *model.Scope, query, key, value *Node, scale float64) *MultiHeadAttentionBuilder
```

`attention.Core` gains a trailing `useFusion bool` param (before `scoreSoftCap`, or appended — plan 03 picks one and is internally consistent). The existing `Core` fused branch (attention.go:268-280) additionally gates on `useFusion`. `GOMLX_FUSION` env stays as a global override in `InternalFusedOpCaller`; `UseFusion(false)` is the per-call override.

Capability check for tests (replaces `isCUDABackend`): a backend "supports fusion" iff a probe `BackendFusedScaledDotProductAttention` on a tiny causal bf16 input does not return `ErrNotImplemented`. Add `testutil.GetOfficialBackend(name string) compute.Backend` (returns the named official backend if present in the system, else skips) for the one `xla:cuda` specific test.

## Contract E — `go-huggingface` consumer (plan 04)

Find the transformer attention call sites (they call `attention.MultiHeadAttention(...).…Done()`), add an optional seqlen path: where the model already computes an attention/padding mask from token lengths, pass those lengths through `WithSeqLens` instead of (or in addition to) materializing the mask, guarded so models without lengths are unchanged.

---

## Verification gates (per plan)

- **01 compute:** `go test ./...` green on Mac (CPU backend exercises bias/seqlen/dropout reference + fp8 NotImplemented).
- **02 go-xla:** `go test ./...` green on Mac for non-CUDA (CustomCallV2 rendering, fallback). **[trig]** `go test ./compute/xla ./pjrt -run FMHA|Flash` green under `xla:cuda` for each variant present in the dtype/variant matrix.
- **03 gomlx:** `go test ./ml/layers/attention/...` green on Mac (decomposed + fallback + UseFusion(false)). **[trig]** the `xla:cuda` fusion-parity test green.
- **04 go-huggingface:** `go test ./...` green on Mac; one model integration test asserting seqlen path produces output within tolerance of the mask-matrix path.
