# Fused Attention — compute#13 (Contract A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the `compute` fused-attention interface (`ScaledDotProductAttentionConfig`) with sequence-length padding masking (Stage 1), then additive bias and deterministic dropout (Stage 2), and implement them in the CPU `go` backend reference (`internal/gobackend/fusedops/sdpa.go`) so downstream repos (go-xla#37, gomlx#427) can consume the new surface.

**Architecture:** All new parameters ride inside `*ScaledDotProductAttentionConfig`; the `FusedOps` method signatures do not change (keeps every existing caller compiling). The CPU `go` backend gains real reference implementations of the seqlen padding mask, additive bias, and seeded dropout; FP8 input dtype returns wrapped `ErrNotImplemented`.

**The CPU SDPA capability stays `false`.** `internal/gobackend/capabilities.go:147` has `OpTypeFusedScaledDotProductAttention: false` and `sdpa.go`'s `init()` keeps its `if false` registration guard. Per the maintainer's note at `capabilities.go:147`, the fused CPU path is ~3x slower than the SIMD matmul + decomposed path, so it was disabled on purpose pending a SIMD fused kernel. We do NOT flip it. Consequently the capability-gated backend test group (`support/backendtest/fusedops.go`) stays skipped exactly as it is today. Every new reference path in this plan is verified by **direct unit tests on the `fusedops` package functions** (calling `fusedops.FusedScaledDotProductAttention(...)` / the exported reference through a test-local backend), each comparing against an in-test decomposed reference.

**Tech Stack:** Go 1.26, `github.com/gomlx/compute` (branch `flash-customcall`, local fork `/Users/guygrigsby/projects/forks/compute`). Reference math uses `math` stdlib and the existing `dtypes`/`shapes` packages.

## Global Constraints

See `docs/plans/2026-06-29-fused-attn-00-contract.md` for the full cross-repo contract and the "Staging" section. Constraints that bind every task here:

- Go 1.26; module `github.com/gomlx/compute`; branch `flash-customcall`; repo `/Users/guygrigsby/projects/forks/compute`.
- This plan implements **Contract A** (the `compute` interface). The forward/VJP method signatures on `FusedOps` MUST NOT change; new params go inside `*ScaledDotProductAttentionConfig`.
- **The CPU `go`-backend SDPA capability stays `false`.** Do not re-enable the `if false` registration in `sdpa.go`, do not flip `capabilities.go:147`. The reference is extended for correctness and tested directly. On CPU, `BackendFusedScaledDotProductAttention` still returns `ErrNotImplemented` → decomposed fallback. This is intentional (the SIMD matmul + decomposed path is ~3x faster); re-enable only alongside a SIMD fused kernel or if Jan asks.
- **Stage order:** do all **[S1]** tasks (seqlen + the struct/equality plumbing they require + FP8 gate), then all **[S2]** tasks (bias, dropout). Each stage is independently green and reviewable.
- **No push, no PR.** Every task ends at `git commit` on `flash-customcall`. Guy reviews diffs before anything reaches a remote.
- **Fallback is the contract.** Any unsupported config (here: FP8 input dtype) returns a `compute.ErrNotImplemented` wrapped with a stack via `errors.Wrapf`. Never panic on an unsupported-but-valid request.
- Commit messages: terse, verb-first, no em/en dashes, prefix `fused_ops:`, no Claude attribution.
- All tests run on the Mac CPU `go` backend (no CUDA in this plan).

## Grounded code state (verified)

- `ScaledDotProductAttentionConfig` is at `fused_ops.go:206-218`; today it holds only `QuantizedMatmuls bool`.
- The forward method is `fused_ops.go:320-326`; the VJP is `fused_ops.go:336-343`. Neither changes.
- The CPU reference is `internal/gobackend/fusedops/sdpa.go`. The exported entry point is `FusedScaledDotProductAttention(f *gobackend.Function, query, key, value, mask compute.Value, numHeads, numKVHeads int, axesLayout compute.AxesLayout, scale float64, causal bool, options *compute.ScaledDotProductAttentionConfig) (output, softmaxStats compute.Value, err error)` (sdpa.go:41-52). It delegates to `buildSDPANode` (sdpa.go:90-117); execution is `execFusedScaledDotProductAttention` (sdpa.go:128-171); the kernels are `sdpaMultiHeadGeneric` (sdpa.go:368) and `sdpaGeneric` (sdpa.go:241).
- **The op is disabled on the go backend and STAYS disabled:** `init()` is gated `if false` (sdpa.go:54-62) and `internal/gobackend/capabilities.go:147` has `OpTypeFusedScaledDotProductAttention: false`. Today `go test ./gobackend/ -run 'TestCompliance/FusedOps/FusedScaledDotProductAttention'` **SKIPs** ("Backend ... does not support operation"); it will keep SKIPping after this plan. We do not touch either site.
- Because the capability stays off, the new reference paths are tested by **direct unit tests** that register the executor on a test-local backend (a `*_test.go` file inside the `fusedops` package whose `TestMain`/setup registers `execFusedScaledDotProductAttention` and calls `FusedScaledDotProductAttention` directly), bypassing the global capability gate.
- `equalOptions` (sdpa.go:80-88) compares options with `*d.options == *o.options`. After adding `Value` (= `any`) fields this struct comparison can panic at runtime when a field holds a non-comparable concrete type. **[S1]** rewrites it to compare field-by-field. **This is a real bug introduced by adding the fields, not a nicety.**
- `compute.ErrNotImplemented` / `compute.IsNotImplemented` are at `compute.go:110-115`.
- `Value` is `any` (`builder.go:12`).
- F8 dtypes: `dtypes.F8E4M3FN` (=17), `dtypes.F8E5M2` (=16). The go backend rejects an F8 **parameter** at creation, but `fn.ConvertDType(f32param, dtypes.F8E4M3FN)` produces an F8-typed node that reaches the SDPA builder — this is how the FP8 test feeds F8 input (verified).
- Test helpers available in `support/testutil/`: `testutil.Exec1(backend, []any{...}, fn)`, `testutil.IsInDelta(want, got, delta)`. `testutil.SkipIfMissing` is NOT used here (it would skip, since the capability is off); the direct-unit-test backend registers the executor itself.

---

## Direct-unit-test harness (used by every task below)

Because the capability stays `false`, the shared `backendtest` SDPA group is skipped. Every test in this plan instead lives in a new file `internal/gobackend/fusedops/sdpa_direct_test.go` (package `fusedops`, so it can register and call unexported and package-level symbols directly). The file's `init()` registers the executor on the `go` backend so that `FusedScaledDotProductAttention(...)` actually runs in-process, then each test calls the exported reference and compares to an in-test decomposed reference.

This harness file is created in **[S1] Task 1, Step 1** (the first test that needs it) and reused by every later task. It does NOT flip the global capability flag; it registers the node executor locally for the test binary only.

---

## File map

- `fused_ops.go` — extend `ScaledDotProductAttentionConfig`: `QuerySeqLen`/`KeyValueSeqLen` ([S1] Task 1), `Bias`/`DropoutRate`/`DropoutSeed`/`DropoutOffset` ([S2] Task 4).
- `internal/gobackend/fusedops/sdpa.go` — rewrite `equalOptions` + node flags ([S1] Task 1); seqlen padding mask ([S1] Task 2); FP8 gate ([S1] Task 3); additive bias ([S2] Task 4); deterministic dropout ([S2] Task 5). **`init()` and `capabilities.go` are NOT touched.**
- `internal/gobackend/fusedops/sdpa_direct_test.go` — NEW direct-unit-test file (package `fusedops`): registers the executor for the test binary, holds `WithSeqLens`, `WithSeqLensCausal`, `FP8NotImplemented` ([S1]); `WithBias`, `WithDropoutDeterministic` ([S2]).
- `support/backendtest/fusedops.go` — **untouched** (its SDPA group stays skipped because the capability is off).

---

## STAGE 1 — seqlen padding mask + config plumbing + FP8 gate

### [S1] Task 1: Extend ScaledDotProductAttentionConfig (seqlen fields) and fix option equality

**Why:** Stage 1 adds `QuerySeqLen`/`KeyValueSeqLen` (Contract A, S1) and fixes the `equalOptions` comparison that would otherwise panic once the struct holds `Value` (interface) fields. Also stands up the direct-unit-test harness.

**Files:**
- Modify: `fused_ops.go:206-218` (struct definition — add the two S1 seqlen fields)
- Modify: `internal/gobackend/fusedops/sdpa.go:80-88` (`equalOptions`)
- Create: `internal/gobackend/fusedops/sdpa_direct_test.go` (harness + a `ConfigFieldsCompile` direct test)

**Interfaces:**
- Consumes: nothing.
- Produces: `compute.ScaledDotProductAttentionConfig` gains `QuerySeqLen, KeyValueSeqLen Value`; `equalOptions` no longer panics on interface fields; the direct-test backend can run the reference.

- [ ] **Step 1: Write the failing test (and the harness file)**

Create `internal/gobackend/fusedops/sdpa_direct_test.go`. The `init()` registers the executor on the `go` backend for this test binary only (it does NOT touch the global capability flag), then the first test sets the new config fields to nil and asserts the reference still runs:

```go
package fusedops

import (
	"math"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/internal/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/testutil"
)

// init registers the SDPA executor on the go backend for this test binary only.
// The production init() in sdpa.go is gated `if false` (the CPU fused path is ~3x
// slower than SIMD matmul + decomposed, so the capability stays off). These tests
// exercise the reference directly, bypassing the capability gate, without enabling
// the op for real callers.
func init() {
	gobackend.RegisterFusedScaledDotProductAttention.Register(FusedScaledDotProductAttention, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeFusedScaledDotProductAttention, gobackend.PriorityTyped, execFusedScaledDotProductAttention)
}

// newGoBackend builds the CPU go backend for direct reference tests.
func newGoBackend(t *testing.T) compute.Backend {
	t.Helper()
	b, err := compute.NewBackend("go")
	if err != nil {
		t.Fatalf("NewBackend(go): %+v", err)
	}
	return b
}

func TestSDPADirect_ConfigFieldsCompile(t *testing.T) {
	b := newGoBackend(t)
	// Setting every new config field with nil/zero values must be accepted
	// (no panic in option equality, output equals the no-config result).
	q := [][][][]float32{{{{1}, {1}}}} // [1,1,2,1]
	k := [][][][]float32{{{{1}, {1}}}}
	v := [][][][]float32{{{{10}, {20}}}}
	cfg := &compute.ScaledDotProductAttentionConfig{
		QuantizedMatmuls: false,
		QuerySeqLen:      nil,
		KeyValueSeqLen:   nil,
	}
	got, err := testutil.Exec1(b, []any{q, k, v}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		out, _, err := f.FusedScaledDotProductAttention(params[0], params[1], params[2], nil, 1, 1, compute.AxesLayoutBHSD, 1.0, true, cfg)
		return out, err
	})
	if err != nil {
		t.Fatalf("SDPA with zero-value config failed: %+v", err)
	}
	// Causal: q0 sees k0 only -> 10; q1 sees k0,k1 (raw scores equal) -> 15.
	want := [][][][]float32{{{{10}, {15}}}}
	if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
		t.Errorf("SDPA zero-value config mismatch:\n%s", diff)
	}
	_ = math.Exp // keep math imported for later tasks in this file
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./internal/gobackend/fusedops/ -run 'TestSDPADirect_ConfigFieldsCompile' -v`
Expected: COMPILE FAILURE — `unknown field 'QuerySeqLen' in struct literal of type compute.ScaledDotProductAttentionConfig` (and `KeyValueSeqLen`).

- [ ] **Step 3: Add the seqlen fields to the struct**

In `fused_ops.go`, replace the `ScaledDotProductAttentionConfig` struct (lines 206-218) with the version below. Only the S1 seqlen fields are added now; the S2 bias/dropout fields are added in Task 4 to keep each stage's diff self-contained:

```go
// ScaledDotProductAttentionConfig holds optional optimization hints and fused-attention
// parameters for FusedScaledDotProductAttention.
// A nil *ScaledDotProductAttentionConfig means "use defaults" (all optimizations disabled).
// Backends that cannot honor a set field MUST return ErrNotImplemented so the caller
// falls back to the decomposed path. nil/zero means "unused".
type ScaledDotProductAttentionConfig struct {
	// QuantizedMatmuls: if true, the backend may use dynamic per-head symmetric
	// affine quantization (scale-only, no zero point) to convert float32 Q/K/V slices
	// to uint8 for the Q@K^T and attn@V matmul stages. Accumulation is done in int32,
	// then dequantized back to float32. Softmax and masking remain in float32.
	// This matches ONNX DynamicQuantizeLinear semantics and trades some numerical
	// precision for throughput on hardware with fast integer dot-product instructions
	// (e.g. ARM SDOT/UDOT, x86 VNNI). Backends that do not support quantized matmuls
	// ignore this flag and use float arithmetic.
	QuantizedMatmuls bool

	// QuerySeqLen, KeyValueSeqLen are optional per-batch actual sequence lengths
	// (int32 tensors, shape [B]). When set, the backend masks by sequence length
	// (padding mask) instead of a materialized [S,Skv] mask. Combined with causal=true
	// this is a padding-causal mask. nil = unused.
	QuerySeqLen, KeyValueSeqLen Value
}
```

- [ ] **Step 4: Fix equalOptions to avoid interface-comparison panic**

In `internal/gobackend/fusedops/sdpa.go`, replace `equalOptions` (lines 80-88) with a field-by-field comparison that uses pointer identity for the `Value` fields (two graph nodes are "the same" iff they are the same `Value`):

```go
func (d *nodeScaledDotProductAttention) equalOptions(o *nodeScaledDotProductAttention) bool {
	if d.options == nil && o.options == nil {
		return true
	}
	if d.options == nil || o.options == nil {
		return false
	}
	a, b := d.options, o.options
	return a.QuantizedMatmuls == b.QuantizedMatmuls &&
		a.QuerySeqLen == b.QuerySeqLen &&
		a.KeyValueSeqLen == b.KeyValueSeqLen
}
```

Note: the `Value` (== `any`) comparisons compare the interface values the caller passed. For the go backend these are `*gobackend.Value` pointers (comparable), so `==` is pointer identity and never panics. If a backend ever stores a non-comparable concrete value, dedup falls back to "not equal", which is safe. (Task 4 extends this with the S2 bias/dropout fields.)

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./internal/gobackend/fusedops/ -run 'TestSDPADirect_ConfigFieldsCompile' -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/guygrigsby/projects/forks/compute
git add fused_ops.go internal/gobackend/fusedops/sdpa.go internal/gobackend/fusedops/sdpa_direct_test.go
git commit -m "fused_ops: add seqlen fields to attention config and direct test harness"
```

---

### [S1] Task 2: CPU reference — sequence-length padding mask

**Why:** Contract A (S1): `QuerySeqLen`/`KeyValueSeqLen` (int32 `[B]` vectors) build a padding mask `[B,1,S,Skv]` (query position `s` valid iff `s < QuerySeqLen[b]`; key position `kv` valid iff `kv < KeyValueSeqLen[b]`), combined with `causal` to give a padding-causal mask.

**Files:**
- Modify: `internal/gobackend/fusedops/sdpa.go` (`buildSDPANode`, `execFusedScaledDotProductAttention`, `sdpaMultiHeadGeneric`, `sdpaGeneric`)
- Test: `internal/gobackend/fusedops/sdpa_direct_test.go` (new tests `TestSDPADirect_WithSeqLens`, `TestSDPADirect_WithSeqLensCausal`)

**Interfaces:**
- Consumes: Task 1 (`options.QuerySeqLen`, `options.KeyValueSeqLen compute.Value`).
- Produces: when both seqlen vectors are set, the go backend masks (weight 0) any score where the query index >= `QuerySeqLen[b]` or the key index >= `KeyValueSeqLen[b]`. A query row whose every key is masked yields an all-zero output row (existing `sum==0 -> invSum=0` guard). Combined with `causal=true`, both constraints apply.

- [ ] **Step 1: Write the failing tests**

Append these two tests to `internal/gobackend/fusedops/sdpa_direct_test.go`:

```go
func TestSDPADirect_WithSeqLens(t *testing.T) {
	b := newGoBackend(t)
	// batch=1, 1 head, seqLen=2 queries, kvLen=2 keys. KeyValueSeqLen=1 means
	// only key 0 is valid; the padding mask must match a materialized mask
	// that allows only key 0. QuerySeqLen=2 (both queries valid).
	q := [][][][]float32{{{{1}, {1}}}}   // [1,1,2,1]
	k := [][][][]float32{{{{1}, {1}}}}   // [1,1,2,1]
	v := [][][][]float32{{{{10}, {20}}}} // [1,1,2,1]
	qLen := []int32{2}
	kvLen := []int32{1}
	got, err := testutil.Exec1(b, []any{q, k, v, qLen, kvLen}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		cfg := &compute.ScaledDotProductAttentionConfig{QuerySeqLen: params[3], KeyValueSeqLen: params[4]}
		out, _, err := f.FusedScaledDotProductAttention(params[0], params[1], params[2], nil, 1, 1, compute.AxesLayoutBHSD, 1.0, false, cfg)
		return out, err
	})
	if err != nil {
		t.Fatalf("SDPA with seqlens failed: %+v", err)
	}
	// Decomposed reference: only key0 valid -> every query attends to key0 only -> output 10.
	want := [][][][]float32{{{{10}, {10}}}}
	if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
		t.Errorf("SDPA seqlens padding mask mismatch:\n%s", diff)
	}
}

func TestSDPADirect_WithSeqLensCausal(t *testing.T) {
	b := newGoBackend(t)
	// causal + KeyValueSeqLen=2 (no key padding) reduces to plain causal.
	// QuerySeqLen=2. query0 sees key0 only (causal) -> 10; query1 sees key0,key1 -> 15.
	q := [][][][]float32{{{{1}, {1}}}} // [1,1,2,1]
	k := [][][][]float32{{{{1}, {1}}}}
	v := [][][][]float32{{{{10}, {20}}}}
	qLen := []int32{2}
	kvLen := []int32{2}
	got, err := testutil.Exec1(b, []any{q, k, v, qLen, kvLen}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		cfg := &compute.ScaledDotProductAttentionConfig{QuerySeqLen: params[3], KeyValueSeqLen: params[4]}
		out, _, err := f.FusedScaledDotProductAttention(params[0], params[1], params[2], nil, 1, 1, compute.AxesLayoutBHSD, 1.0, true, cfg)
		return out, err
	})
	if err != nil {
		t.Fatalf("SDPA with seqlens+causal failed: %+v", err)
	}
	want := [][][][]float32{{{{10}, {15}}}}
	if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
		t.Errorf("SDPA seqlens+causal mismatch:\n%s", diff)
	}
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./internal/gobackend/fusedops/ -run 'TestSDPADirect_WithSeqLens' -v`
Expected: FAIL — seqlens ignored. `WithSeqLens` returns the plain mean `15` for both rows instead of `10`. (`WithSeqLensCausal` passes by accident since `kvLen==2` is no padding, but keep it; it pins the causal-combination once Step 5 lands.)

- [ ] **Step 3: Thread seqlens through buildSDPANode as extra operands**

In `internal/gobackend/fusedops/sdpa.go`, in `buildSDPANode`, after appending `mask` to `values` (after the `if mask != nil` block, ~line 100), validate and append the two seqlen operands when both are set (they are mutually-required: a padding mask needs both length vectors):

```go
	if options != nil && (options.QuerySeqLen != nil) != (options.KeyValueSeqLen != nil) {
		return nil, errors.Errorf("%s: QuerySeqLen and KeyValueSeqLen must both be set or both nil", opName)
	}
	hasSeqLens := options != nil && options.QuerySeqLen != nil && options.KeyValueSeqLen != nil
	if hasSeqLens {
		values = append(values, options.QuerySeqLen, options.KeyValueSeqLen)
	}
```

Extend `nodeScaledDotProductAttention` (lines 64-71) with the operand-presence flags exec needs to index `inputs` correctly:

```go
type nodeScaledDotProductAttention struct {
	numHeads   int
	numKVHeads int
	axesLayout compute.AxesLayout
	scale      float64
	causal     bool
	hasMask    bool
	hasSeqLens bool
	options    *compute.ScaledDotProductAttentionConfig
}
```

Set the flags in the `data` literal (replace the `data := &nodeScaledDotProductAttention{...}` line ~114):

```go
	data := &nodeScaledDotProductAttention{
		numHeads: numHeads, numKVHeads: numKVHeads, axesLayout: axesLayout,
		scale: scale, causal: causal,
		hasMask: mask != nil, hasSeqLens: hasSeqLens,
		options: options,
	}
```

Add the flags to `EqualNodeData` (lines 73-78):

```go
func (d *nodeScaledDotProductAttention) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*nodeScaledDotProductAttention)
	return d.numHeads == o.numHeads && d.numKVHeads == o.numKVHeads &&
		d.axesLayout == o.axesLayout && d.scale == o.scale && d.causal == o.causal &&
		d.hasMask == o.hasMask && d.hasSeqLens == o.hasSeqLens &&
		d.equalOptions(o)
}
```

- [ ] **Step 4: Locate operands in exec and read the lengths**

In `execFusedScaledDotProductAttention` (lines 128-171), replace the mask-locating block (lines 134-137) with flag-driven operand indexing that locates mask then the two seqlen buffers:

```go
	data := node.Data.(*nodeScaledDotProductAttention)
	query := inputs[0]
	key := inputs[1]
	value := inputs[2]
	next := 3
	var mask *gobackend.Buffer
	if data.hasMask {
		mask = inputs[next]
		next++
	}
	var querySeqLen, keyValueSeqLen []int32
	if data.hasSeqLens {
		querySeqLen = inputs[next].Flat.([]int32)
		next++
		keyValueSeqLen = inputs[next].Flat.([]int32)
		next++
	}
	_ = next
```

The existing BSHD rank-4 mask transpose block (lines 142-148) is unchanged. Pass the two length slices into both dtype calls of `sdpaMultiHeadGeneric`:

```go
	switch query.RawShape.DType {
	case dtypes.Float32:
		sdpaMultiHeadGeneric[float32](query, key, value, mask, output, data, maskBatchStride, maskHeadStride, querySeqLen, keyValueSeqLen)
	case dtypes.Float64:
		sdpaMultiHeadGeneric[float64](query, key, value, mask, output, data, maskBatchStride, maskHeadStride, querySeqLen, keyValueSeqLen)
	default:
		return nil, errors.Errorf("FusedScaledDotProductAttention: unsupported dtype %s", query.RawShape.DType)
	}
```

- [ ] **Step 5: Apply the padding mask in the kernels**

Extend `sdpaMultiHeadGeneric` (line 368) to accept the two length slices:

```go
func sdpaMultiHeadGeneric[T float32 | float64](query, key, value, mask, output *gobackend.Buffer, data *nodeScaledDotProductAttention, maskBatchStride, maskHeadStride int, querySeqLen, keyValueSeqLen []int32) {
```

Inside the `for batchIdx` loop, before the `for kvHeadIdx` loop (after line 426 `for kvHeadIdx := range numKVHeads {` is entered — compute per-batch limits just inside `for batchIdx`, before the kvHead loop, around line 426), compute this batch's effective query/key limits (default to full length when no seqlens):

```go
	for batchIdx := range batchSize {
		qLimit := seqLen
		kvLimit := kvLen
		if len(querySeqLen) > 0 {
			qLimit = int(querySeqLen[batchIdx])
		}
		if len(keyValueSeqLen) > 0 {
			kvLimit = int(keyValueSeqLen[batchIdx])
		}
		for kvHeadIdx := range numKVHeads {
			// ... existing mask-slice block unchanged ...
```

Pass `qLimit, kvLimit` into `sdpaGeneric` at the call site (lines 447-453):

```go
			sdpaGeneric(
				q, k, v, qOff, kvOff, qSeqStride, kvSeqStride, qHeadStride,
				additiveMaskSlice, booleanMaskSlice, maskGroupStride,
				scores,
				out,
				groupSize, seqLen, kvLen, headDim, scale, causal,
				qLimit, kvLimit,
			)
```

Extend `sdpaGeneric` (lines 241-249) with `qLimit, kvLimit int` after `causal bool`:

```go
	groupSize, seqLen, kvLen, headDim int, scale T, causal bool,
	qLimit, kvLimit int,
) {
```

Inside `sdpaGeneric`, fold the key limit into `kvLenUnmasked` (replace lines 259-262):

```go
			kvLenUnmasked := kvLen
			if kvLimit < kvLenUnmasked {
				kvLenUnmasked = kvLimit
			}
			if causal {
				kvLenUnmasked = min(kvLenUnmasked, qIdx+1)
			}
```

The query limit zeroes whole rows. At the top of the `for qIdx := range seqLen` loop body, after `maskIdxBase := gMaskOff + qIdx*kvLen` (line 257), skip-and-zero padded query rows:

```go
			if qIdx >= qLimit {
				// Padded query position: emit a zero output row, skip attention.
				// Local padBase has its own scope, distinct from the outBase the
				// accumulation loop declares later, so there is no redeclaration.
				padBase := gQOff + qIdx*qSeqStride
				for d := range headDim {
					output[padBase+d] = 0
				}
				continue
			}
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./internal/gobackend/fusedops/ -run 'TestSDPADirect_WithSeqLens' -v`
Expected: PASS for both `TestSDPADirect_WithSeqLens` and `TestSDPADirect_WithSeqLensCausal`.

- [ ] **Step 7: Run the whole fusedops + gobackend suite to confirm no regression**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./internal/gobackend/... -run 'SDPA|FusedOps' -v`
Expected: PASS for the new direct tests; the capability-gated `TestCompliance/FusedOps/FusedScaledDotProductAttention` still SKIPs (capability off).

- [ ] **Step 8: Commit**

```bash
cd /Users/guygrigsby/projects/forks/compute
git add internal/gobackend/fusedops/sdpa.go internal/gobackend/fusedops/sdpa_direct_test.go
git commit -m "fused_ops: implement seqlen padding mask in go backend"
```

---

### [S1] Task 3: CPU reference — FP8 input dtype returns ErrNotImplemented

**Why:** Contract A: the go backend does not implement FP8 attention; an FP8 (e4m3/e5m2) input dtype must return a wrapped `compute.ErrNotImplemented` so the caller falls back to the decomposed path. Never panic. This is pure dtype gating and ships in Stage 1.

**Files:**
- Modify: `internal/gobackend/fusedops/sdpa.go` (`buildSDPANode`)
- Test: `internal/gobackend/fusedops/sdpa_direct_test.go` (new test `TestSDPADirect_FP8NotImplemented`)

**Interfaces:**
- Consumes: Task 1 (struct in place); the `buildSDPANode` dtype check sees `qNode.Shape.DType`.
- Produces: `FusedScaledDotProductAttention` returns `err` with `compute.IsNotImplemented(err) == true` when `query` dtype is `dtypes.F8E4M3FN` or `dtypes.F8E5M2`.

- [ ] **Step 1: Write the failing test**

Append this test to `internal/gobackend/fusedops/sdpa_direct_test.go`:

```go
func TestSDPADirect_FP8NotImplemented(t *testing.T) {
	b := newGoBackend(t)
	// The go backend rejects F8 parameters at creation, so feed F8 by converting
	// a float32 param to F8E4M3FN inside the graph (verified path), then assert
	// SDPA reports NotImplemented for the F8 dtype.
	builder := b.Builder("fused_fp8_test")
	mainFn := builder.Main()
	p, err := mainFn.Parameter("q", shapes.Make(dtypes.Float32, 1, 1, 2, 1), nil)
	if err != nil {
		t.Fatalf("Parameter failed: %+v", err)
	}
	q8, err := mainFn.ConvertDType(p, dtypes.F8E4M3FN)
	if err != nil {
		t.Fatalf("ConvertDType to F8E4M3FN failed: %+v", err)
	}
	_, _, err = mainFn.FusedScaledDotProductAttention(q8, q8, q8, nil, 1, 1, compute.AxesLayoutBHSD, 1.0, true, nil)
	if err == nil {
		t.Fatalf("SDPA with F8 input must return an error, got nil")
	}
	if !compute.IsNotImplemented(err) {
		t.Errorf("SDPA with F8 input must return ErrNotImplemented, got: %+v", err)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./internal/gobackend/fusedops/ -run 'TestSDPADirect_FP8NotImplemented' -v`
Expected: FAIL — with no dtype gate, `buildSDPANode` accepts the F8 node and returns `err == nil`, tripping "must return an error, got nil". (Without the gate the node would later fail in exec with an unsupported-dtype `errors.Errorf`, which is not `ErrNotImplemented`.)

- [ ] **Step 3: Add the FP8 dtype gate in buildSDPANode**

In `internal/gobackend/fusedops/sdpa.go`, in `buildSDPANode`, after the rank check (after line 109, `if qNode.Shape.Rank() != 4 { ... }`), add:

```go
	switch qNode.Shape.DType {
	case dtypes.F8E4M3FN, dtypes.F8E5M2:
		return nil, errors.Wrapf(compute.ErrNotImplemented,
			"%s: float8 input dtype %s is not implemented in the go backend", opName, qNode.Shape.DType)
	}
```

(`dtypes`, `compute`, and `errors` (`github.com/pkg/errors`) are already imported in this file.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./internal/gobackend/fusedops/ -run 'TestSDPADirect_FP8NotImplemented' -v`
Expected: PASS.

- [ ] **Step 5: Run the whole fusedops suite to confirm no regression**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./internal/gobackend/fusedops/ -v`
Expected: PASS for all `TestSDPADirect_*` tests.

- [ ] **Step 6: Commit**

```bash
cd /Users/guygrigsby/projects/forks/compute
git add internal/gobackend/fusedops/sdpa.go internal/gobackend/fusedops/sdpa_direct_test.go
git commit -m "fused_ops: return ErrNotImplemented for float8 attention dtype"
```

---

## Stage 1 gate

Before starting Stage 2, confirm Stage 1 is independently green:

- [ ] `cd /Users/guygrigsby/projects/forks/compute && go build ./...` — no output, exit 0.
- [ ] `cd /Users/guygrigsby/projects/forks/compute && go test ./...` — all packages `ok` (or `[no test files]`), no `FAIL`. The capability-gated `TestCompliance/FusedOps/FusedScaledDotProductAttention` still SKIPs (intended).
- [ ] `cd /Users/guygrigsby/projects/forks/compute && go vet ./...` — no output, exit 0.

Stop here for review before Stage 2.

---

## STAGE 2 — additive bias + deterministic dropout

### [S2] Task 4: Add bias/dropout config fields and implement additive bias

**Why:** Stage 2 adds `Bias`/`DropoutRate`/`DropoutSeed`/`DropoutOffset` (Contract A, S2) and implements the additive pre-softmax bias in the CPU reference. (Dropout impl is Task 5; this task adds its struct fields alongside bias so the struct extension is one diff.)

**Files:**
- Modify: `fused_ops.go` (extend struct with the four S2 fields)
- Modify: `internal/gobackend/fusedops/sdpa.go` (`equalOptions`, `buildSDPANode`, `execFusedScaledDotProductAttention`, `sdpaMultiHeadGeneric`, `sdpaGeneric` — bias path)
- Test: `internal/gobackend/fusedops/sdpa_direct_test.go` (new test `TestSDPADirect_WithBias`)

**Interfaces:**
- Consumes: Task 2 (operand-threading pattern; `options.Bias compute.Value`).
- Produces: when `options.Bias != nil`, the go backend adds `bias[b,h,s,kv]` to the scaled score before softmax, broadcast to `[B,H,S,Skv]` using the same stride convention as the additive `mask` (`sdpaComputeMaskStrides`). Bias and `mask` may both be set; both are added.

- [ ] **Step 1: Write the failing test**

Append this test to `internal/gobackend/fusedops/sdpa_direct_test.go`:

```go
func TestSDPADirect_WithBias(t *testing.T) {
	b := newGoBackend(t)
	// One query attends to two keys; bias makes key 0 dominate.
	// q=[1], k=[[1],[1]], scale=1 -> raw scores [1,1].
	// bias=[[5,0]] -> scores [6,1]. softmax([6,1]) weights key0 heavily.
	q := [][][][]float32{{{{1}}}}        // [1,1,1,1]
	k := [][][][]float32{{{{1}, {1}}}}   // [1,1,2,1]
	v := [][][][]float32{{{{10}, {20}}}} // [1,1,2,1]
	bias := [][][][]float32{{{{5, 0}}}}  // [1,1,1,2] broadcast over heads
	got, err := testutil.Exec1(b, []any{q, k, v, bias}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
		cfg := &compute.ScaledDotProductAttentionConfig{Bias: params[3]}
		out, _, err := f.FusedScaledDotProductAttention(params[0], params[1], params[2], nil, 1, 1, compute.AxesLayoutBHSD, 1.0, false, cfg)
		return out, err
	})
	if err != nil {
		t.Fatalf("SDPA with bias failed: %+v", err)
	}
	// Decomposed reference: w = softmax([1*1+5, 1*1+0]) = softmax([6,1]).
	e0, e1 := math.Exp(6), math.Exp(1)
	w0, w1 := e0/(e0+e1), e1/(e0+e1)
	want := [][][][]float32{{{{float32(w0*10 + w1*20)}}}}
	if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
		t.Errorf("SDPA with bias mismatch:\n%s", diff)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./internal/gobackend/fusedops/ -run 'TestSDPADirect_WithBias' -v`
Expected: COMPILE FAILURE first — `unknown field 'Bias'`. After Step 3 adds the field but before Step 5 wires it, FAIL — bias ignored, output is the plain mean `15` not `~10.007`.

- [ ] **Step 3: Add the S2 fields to the struct**

In `fused_ops.go`, append the four S2 fields to `ScaledDotProductAttentionConfig` (after `QuerySeqLen, KeyValueSeqLen Value`):

```go
	// Bias is an optional additive attention-score bias broadcast to [B,H,S,Skv],
	// added to the scaled scores before softmax (ALiBi / relative-position). NOT the
	// Q/K/V projection bias. Selects the fmhaScaleBias* variants in the xla backend.
	// nil = unused.
	Bias Value

	// DropoutRate in [0,1); 0 disables. Nonzero selects the fmha*Dropout variants.
	// DropoutSeed and DropoutOffset feed the backend RNG (Value: int64 scalars) so the
	// dropout pattern is deterministic for a given (seed, offset). nil seed/offset with a
	// nonzero rate is treated as seed=0, offset=0.
	DropoutRate                float64
	DropoutSeed, DropoutOffset Value
```

- [ ] **Step 4: Extend equalOptions with the S2 fields**

In `internal/gobackend/fusedops/sdpa.go`, extend `equalOptions` to compare the new fields:

```go
	a, b := d.options, o.options
	return a.QuantizedMatmuls == b.QuantizedMatmuls &&
		a.QuerySeqLen == b.QuerySeqLen &&
		a.KeyValueSeqLen == b.KeyValueSeqLen &&
		a.Bias == b.Bias &&
		a.DropoutRate == b.DropoutRate &&
		a.DropoutSeed == b.DropoutSeed &&
		a.DropoutOffset == b.DropoutOffset
```

- [ ] **Step 5: Thread Bias through buildSDPANode and exec**

In `buildSDPANode`, append the bias operand right after the mask append (before the seqlen block from Task 2, so operand order is mask, bias, seqlens, dropout):

```go
	hasBias := options != nil && options.Bias != nil
	if hasBias {
		values = append(values, options.Bias)
	}
```

Add `hasBias bool` to `nodeScaledDotProductAttention`, set `hasBias: hasBias` in the `data` literal, and add `&& d.hasBias == o.hasBias` to `EqualNodeData`.

In `execFusedScaledDotProductAttention`, locate the bias operand after mask and before seqlens (the operand order in `buildSDPANode` is mask, bias, seqlens):

```go
	var bias *gobackend.Buffer
	if data.hasBias {
		bias = inputs[next]
		next++
	}
```

(Insert this between the mask block and the seqlen block, matching the append order.) Transpose a rank-4 BSHD bias the same way the mask is transposed:

```go
	if data.axesLayout == compute.AxesLayoutBSHD && bias != nil && bias.RawShape.Rank() == 4 {
		var err error
		if bias, err = transposeBuffer(backend, bias, []int{0, 2, 1, 3}); err != nil {
			return nil, err
		}
	}
```

Compute bias strides next to the mask strides:

```go
	var biasBatchStride, biasHeadStride int
	if bias != nil {
		biasBatchStride, biasHeadStride = sdpaComputeMaskStrides(bias.RawShape.Dimensions)
	}
```

Pass `bias, biasBatchStride, biasHeadStride` into both `sdpaMultiHeadGeneric` dtype calls (alongside the seqlen slices from Task 2).

- [ ] **Step 6: Apply bias in the kernels**

Extend `sdpaMultiHeadGeneric` signature to accept the bias buffer and strides:

```go
func sdpaMultiHeadGeneric[T float32 | float64](query, key, value, mask, bias, output *gobackend.Buffer, data *nodeScaledDotProductAttention, maskBatchStride, maskHeadStride, biasBatchStride, biasHeadStride int, querySeqLen, keyValueSeqLen []int32) {
```

After the `var additiveMask []T; var booleanMask []bool` block, extract the bias slice:

```go
	var biasFlat []T
	if bias != nil {
		biasFlat = bias.Flat.([]T)
	}
```

Inside the `for kvHeadIdx` loop, after the mask-slice block, compute the per-group bias slice and stride exactly like the additive mask:

```go
			var biasSlice []T
			biasGroupStride := 0
			if len(biasFlat) > 0 {
				biasOffset := batchIdx*biasBatchStride + kvHeadIdx*groupSize*biasHeadStride
				biasEnd := biasOffset + maskSliceLen
				if biasHeadStride > 0 && groupSize > 1 {
					biasEnd = biasOffset + (groupSize-1)*biasHeadStride + maskSliceLen
					biasGroupStride = biasHeadStride
				}
				biasSlice = biasFlat[biasOffset:biasEnd]
			}
```

Pass `biasSlice, biasGroupStride` into `sdpaGeneric`. Extend `sdpaGeneric` with `bias []T, biasGroupStride int` (after `maskGroupStride int`):

```go
	additiveMask []T,
	booleanMask []bool,
	maskGroupStride int,
	bias []T,
	biasGroupStride int,
	scores []T,
	output []T,
	groupSize, seqLen, kvLen, headDim int, scale T, causal bool,
	qLimit, kvLimit int,
) {
```

Inside `sdpaGeneric`, after `gMaskOff := gIdx * maskGroupStride` (line 252) add the bias group offset:

```go
		gBiasOff := gIdx * biasGroupStride
```

After `maskIdxBase := gMaskOff + qIdx*kvLen` (line 257) add:

```go
			biasIdxBase := gBiasOff + qIdx*kvLen
```

In the score loop, after the additive-mask add (`if len(additiveMask) > 0 { s += additiveMask[maskIdx] }`), add the bias term:

```go
				if len(bias) > 0 {
					s += bias[biasIdxBase+kvIdx]
				}
```

- [ ] **Step 7: Run test to verify it passes**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./internal/gobackend/fusedops/ -run 'TestSDPADirect_WithBias' -v`
Expected: PASS.

- [ ] **Step 8: Run the whole fusedops suite to confirm no regression**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./internal/gobackend/fusedops/ -v`
Expected: PASS for all `TestSDPADirect_*` tests.

- [ ] **Step 9: Commit**

```bash
cd /Users/guygrigsby/projects/forks/compute
git add fused_ops.go internal/gobackend/fusedops/sdpa.go internal/gobackend/fusedops/sdpa_direct_test.go
git commit -m "fused_ops: add bias/dropout config fields and implement additive bias"
```

---

### [S2] Task 5: CPU reference — deterministic dropout

**Why:** Contract A (S2): implement `DropoutRate` deterministically from `DropoutSeed`/`DropoutOffset` so CPU-vs-CPU tests are stable (same seed -> same output; different seed -> different).

**Files:**
- Modify: `internal/gobackend/fusedops/sdpa.go` (`buildSDPANode`, `execFusedScaledDotProductAttention`, `sdpaMultiHeadGeneric`, `sdpaGeneric` — dropout path; add `dropoutUniform` helper)
- Test: `internal/gobackend/fusedops/sdpa_direct_test.go` (new test `TestSDPADirect_WithDropoutDeterministic`)

**Interfaces:**
- Consumes: Task 4 (operand-threading pattern; `options.DropoutRate float64`, `options.DropoutSeed`, `options.DropoutOffset compute.Value`).
- Produces: when `options.DropoutRate > 0`, each post-softmax weight is independently dropped with probability `DropoutRate`, surviving weights scaled by `1/(1-DropoutRate)` (inverted dropout), via a deterministic per-element PRNG keyed on `(seed, offset, b, h, qIdx, kvIdx)`. Seed/offset are optional int64 scalar `Value`s; nil means 0. The pattern is identical across runs for a fixed `(seed, offset)`.

- [ ] **Step 1: Write the failing test**

Append this test to `internal/gobackend/fusedops/sdpa_direct_test.go`:

```go
func TestSDPADirect_WithDropoutDeterministic(t *testing.T) {
	b := newGoBackend(t)
	// Wide attention (8 keys, rate 0.5) so dropout changes the output with high
	// probability. Same seed -> identical output; different seed -> different.
	q := [][][][]float32{{{{1}}}} // [1,1,1,1]
	k := [][][][]float32{{{{1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}}}} // [1,1,8,1]
	v := [][][][]float32{{{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}}}} // [1,1,8,1]
	run := func(seed int64) any {
		out, err := testutil.Exec1(b, []any{q, k, v, seed}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
			cfg := &compute.ScaledDotProductAttentionConfig{DropoutRate: 0.5, DropoutSeed: params[3]}
			o, _, err := f.FusedScaledDotProductAttention(params[0], params[1], params[2], nil, 1, 1, compute.AxesLayoutBHSD, 1.0, false, cfg)
			return o, err
		})
		if err != nil {
			t.Fatalf("SDPA with dropout failed: %+v", err)
		}
		return out
	}
	a1 := run(7)
	a2 := run(7)
	c1 := run(42)
	if ok, _ := testutil.IsInDelta(a1, a2, 1e-9); !ok {
		t.Errorf("same seed must produce identical dropout output: %v vs %v", a1, a2)
	}
	if ok, _ := testutil.IsInDelta(a1, c1, 1e-9); ok {
		t.Errorf("different seeds must produce different dropout output, both = %v", a1)
	}
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./internal/gobackend/fusedops/ -run 'TestSDPADirect_WithDropoutDeterministic' -v`
Expected: FAIL — dropout ignored, so seed 7 and seed 42 give the identical (undropped) output, tripping "different seeds must produce different".

- [ ] **Step 3: Thread dropout params through buildSDPANode**

In `buildSDPANode`, after the seqlen-append block, append the optional dropout seed/offset operands (operand order: mask, bias, seqlens, dropout seed, dropout offset). Scalar seed/offset are passed as operands so the graph can carry runtime RNG state; nil defaults to 0 captured in node data:

```go
	hasDropout := options != nil && options.DropoutRate > 0
	if hasDropout {
		if options.DropoutSeed != nil {
			values = append(values, options.DropoutSeed)
		}
		if options.DropoutOffset != nil {
			values = append(values, options.DropoutOffset)
		}
	}
```

Extend `nodeScaledDotProductAttention` with the flags needed to locate these operands:

```go
	hasDropout     bool
	hasDropoutSeed bool
	hasDropoutOff  bool
```

Set them in the `data` literal:

```go
		hasDropout:     hasDropout,
		hasDropoutSeed: hasDropout && options.DropoutSeed != nil,
		hasDropoutOff:  hasDropout && options.DropoutOffset != nil,
```

Add all three to `EqualNodeData` (`&& d.hasDropout == o.hasDropout && d.hasDropoutSeed == o.hasDropoutSeed && d.hasDropoutOff == o.hasDropoutOff`).

- [ ] **Step 4: Read seed/offset in exec and pass dropout params down**

In `execFusedScaledDotProductAttention`, after the seqlen block, read the optional seed/offset scalars (default 0) and the rate:

```go
	var dropoutRate float64
	var dropoutSeed, dropoutOffset int64
	if data.hasDropout {
		dropoutRate = data.options.DropoutRate
		if data.hasDropoutSeed {
			dropoutSeed = inputs[next].Flat.([]int64)[0]
			next++
		}
		if data.hasDropoutOff {
			dropoutOffset = inputs[next].Flat.([]int64)[0]
			next++
		}
	}
	_ = next
```

Pass `dropoutRate, dropoutSeed, dropoutOffset` into both `sdpaMultiHeadGeneric` dtype calls (alongside the seqlen + bias args from Tasks 2 and 4).

- [ ] **Step 5: Add a deterministic per-element PRNG and apply inverted dropout**

At the top of `internal/gobackend/fusedops/sdpa.go`, after the imports, add a pure splitmix64-based hash: given the key tuple it returns a uniform float64 in `[0,1)`:

```go
// dropoutUniform returns a deterministic uniform value in [0,1) for the given key.
// splitmix64 finalizer over a mixed 64-bit key: stable across runs and platforms.
func dropoutUniform(seed, offset int64, b, h, q, kv int) float64 {
	x := uint64(seed)
	mix := func(v uint64) {
		x += v * 0x9E3779B97F4A7C15
		x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9
		x = (x ^ (x >> 27)) * 0x94D049BB133111EB
		x ^= x >> 31
	}
	mix(uint64(offset))
	mix(uint64(b))
	mix(uint64(h))
	mix(uint64(q))
	mix(uint64(kv))
	// Top 53 bits -> [0,1).
	return float64(x>>11) / float64(uint64(1)<<53)
}
```

Extend `sdpaMultiHeadGeneric` to accept dropout params:

```go
func sdpaMultiHeadGeneric[T float32 | float64](query, key, value, mask, bias, output *gobackend.Buffer, data *nodeScaledDotProductAttention, maskBatchStride, maskHeadStride, biasBatchStride, biasHeadStride int, querySeqLen, keyValueSeqLen []int32, dropoutRate float64, dropoutSeed, dropoutOffset int64) {
```

Dropout is keyed per (batch, head, query, key). `sdpaGeneric` works per KV-head group; pass the absolute base head index plus dropout params so the PRNG can recover the absolute head. Extend `sdpaGeneric` (after `qLimit, kvLimit int`):

```go
	qLimit, kvLimit int,
	dropoutRate float64, dropoutSeed, dropoutOffset int64,
	batchIdx, baseHeadIdx int,
) {
```

At the call site in `sdpaMultiHeadGeneric`, pass `dropoutRate, dropoutSeed, dropoutOffset, batchIdx, kvHeadIdx*groupSize`.

Inside `sdpaGeneric`, after the softmax normalization writes the final per-key weights (after the block that sets `scores[scoreIdx] *= invSum`, before the output accumulation that starts at `outBase := gQOff + qIdx*qSeqStride`), apply inverted dropout when `dropoutRate > 0`:

```go
			if dropoutRate > 0 {
				keep := 1.0 - dropoutRate
				invKeep := T(1.0 / keep)
				absHead := baseHeadIdx + gIdx
				for kvIdx := range kvLenUnmasked {
					scoreIdx := scoreIdxBase + kvIdx
					if dropoutUniform(dropoutSeed, dropoutOffset, batchIdx, absHead, qIdx, kvIdx) < dropoutRate {
						scores[scoreIdx] = 0
					} else {
						scores[scoreIdx] *= invKeep
					}
				}
			}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./internal/gobackend/fusedops/ -run 'TestSDPADirect_WithDropoutDeterministic' -v`
Expected: PASS (same seed identical, different seeds differ).

- [ ] **Step 7: Run the whole fusedops suite to confirm no regression**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./internal/gobackend/fusedops/ -v`
Expected: PASS for all `TestSDPADirect_*` tests (dropout off by default leaves prior tests unchanged).

- [ ] **Step 8: Commit**

```bash
cd /Users/guygrigsby/projects/forks/compute
git add internal/gobackend/fusedops/sdpa.go internal/gobackend/fusedops/sdpa_direct_test.go
git commit -m "fused_ops: implement deterministic dropout in go backend"
```

---

## Verification gate (01 compute)

Per the contract's revised "01 compute" gate: `go test ./...` green on Mac; the new seqlen/bias/dropout reference code is covered by **direct unit tests on the `fusedops` functions** (`TestSDPADirect_*`); the **CPU SDPA capability stays `false`** and the **capability-gated `backendtest` SDPA group stays skipped** exactly as it is today; FP8 input → `ErrNotImplemented`.

- [ ] **Full module build:**

Run: `cd /Users/guygrigsby/projects/forks/compute && go build ./...`
Expected: no output, exit 0.

- [ ] **Full test suite green:**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./...`
Expected: all packages `ok` (or `[no test files]`), no `FAIL`.

- [ ] **Direct fusedops unit tests all pass:**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./internal/gobackend/fusedops/ -v`
Expected: PASS for `TestSDPADirect_ConfigFieldsCompile`, `TestSDPADirect_WithSeqLens`, `TestSDPADirect_WithSeqLensCausal`, `TestSDPADirect_FP8NotImplemented`, `TestSDPADirect_WithBias`, `TestSDPADirect_WithDropoutDeterministic`.

- [ ] **Capability-gated SDPA group still SKIPs (capability off, intended):**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./gobackend/ -run 'TestCompliance/FusedOps/FusedScaledDotProductAttention' -v`
Expected: SKIP ("does not support operation FusedScaledDotProductAttention"). This MUST still skip; the capability stays `false`.

- [ ] **Vet clean:**

Run: `cd /Users/guygrigsby/projects/forks/compute && go vet ./...`
Expected: no output, exit 0.

When green, the downstream go-xla#37 plan (02) can consume the new `ScaledDotProductAttentionConfig` fields via the local `replace` directive. No push, no PR from this plan.

---

## Self-review

**Spec coverage vs contract.**
- Contract A "CPU SDPA capability stays `false`": honored — no edit to `capabilities.go` or the `if false` `init()` guard; the verification gate asserts the gated group still SKIPs. The old Task 0 that re-enabled the op is REMOVED.
- Contract A reference support matrix: QuerySeqLen/KeyValueSeqLen ([S1] Task 2), Bias ([S2] Task 4), DropoutRate deterministic from seed/offset ([S2] Task 5), FP8 → `ErrNotImplemented` ([S1] Task 3). All covered.
- Staging: tasks ordered S1 (struct+equality plumbing Task 1, seqlen Task 2, FP8 gate Task 3) then S2 (bias Task 4, dropout Task 5), with an explicit Stage 1 gate between them. FP8 kept in S1 as decided (pure dtype gating). Every task is tagged [S1]/[S2].
- Tests are DIRECT unit tests on the `fusedops` package functions (`internal/gobackend/fusedops/sdpa_direct_test.go`, package `fusedops`, registers the executor in a test-local `init()` and calls `FusedScaledDotProductAttention(...)` / `buildSDPANode` through the graph), NOT through the capability-gated `support/backendtest/fusedops.go` (which stays untouched and skipped). Each test compares to an in-test decomposed reference.
- Verification gate matches the contract's revised "01 compute": `go test ./...` green; direct unit tests cover the new paths; capability stays false; backendtest SDPA group stays skipped; FP8 → ErrNotImplemented.

**Placeholder scan.** No `TODO`, no `...` stand-ins, no "implement X here" — every step gives full, pasteable Go. Exported/unexported names match the verified source: `FusedScaledDotProductAttention`, `buildSDPANode`, `execFusedScaledDotProductAttention`, `sdpaMultiHeadGeneric`, `sdpaGeneric`, `equalOptions`, `EqualNodeData`, `nodeScaledDotProductAttention`, `sdpaComputeMaskStrides`, `transposeBuffer` all verified against `sdpa.go`.

**Type consistency.**
- New config fields are `Value` (= `any`) for tensors (`QuerySeqLen`, `KeyValueSeqLen`, `Bias`, `DropoutSeed`, `DropoutOffset`) and `float64` for `DropoutRate`, matching Contract A verbatim.
- `equalOptions` compares `Value` fields with `==` (pointer identity on `*gobackend.Value`, comparable; safe non-equal otherwise) — no `*d.options == *o.options` struct compare that would panic on interface fields. This is the real bug fix.
- Operand order is fixed and consistent everywhere it is read: `query, key, value [, mask] [, bias] [, querySeqLen, keyValueSeqLen] [, dropoutSeed] [, dropoutOffset]`. `exec` walks them with a single `next` cursor gated by the node `has*` flags, matching the `buildSDPANode` append order.
- Kernel signatures stay generic `[T float32 | float64]`; int32 seqlen slices and int64 dropout scalars are read in `exec` (not generic), passed as concrete types — no dtype mismatch.
- FP8 gate uses `dtypes.F8E4M3FN`/`dtypes.F8E5M2` and `errors.Wrapf(compute.ErrNotImplemented, ...)`; all three packages already imported in `sdpa.go`.

**One risk flagged.** The seqlen/dropout scalars are fed as runtime operands (graph parameters), so the FP8 and seqlen tests rely on the test-local `init()` registering the executor. If a future refactor moves registration behind the capability flag, these direct tests would start exercising nothing — the gate's "capability-gated group still SKIPs" check plus the direct tests asserting real numeric output together catch that.
