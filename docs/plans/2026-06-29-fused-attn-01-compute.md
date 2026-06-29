# Fused Attention — compute#13 (Contract A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the `compute` fused-attention interface (`ScaledDotProductAttentionConfig`) with bias, sequence-length padding masking, and deterministic dropout, and implement them in the CPU `go` backend reference so downstream repos (go-xla#37, gomlx#427) can consume the new surface.

**Architecture:** All new parameters ride inside `*ScaledDotProductAttentionConfig`; the `FusedOps` method signatures do not change (keeps every existing caller compiling). The CPU `go` backend (`internal/gobackend/fusedops/sdpa.go`) gains real implementations of bias, seqlen padding mask, and seeded dropout; FP8 input dtype returns wrapped `ErrNotImplemented`. The shared backend test suite (`support/backendtest/fusedops.go`) gains subtests that compare the fused output against an in-test decomposed reference.

**Tech Stack:** Go 1.26, `github.com/gomlx/compute` (branch `flash-customcall`, local fork `/Users/guygrigsby/projects/forks/compute`). Reference math uses `math` stdlib and the existing `dtypes`/`shapes` packages.

## Global Constraints

See `docs/plans/2026-06-29-fused-attn-00-contract.md` for the full cross-repo contract. Constraints that bind every task here:

- Go 1.26; module `github.com/gomlx/compute`; branch `flash-customcall`; repo `/Users/guygrigsby/projects/forks/compute`.
- This plan implements **Contract A** (the `compute` interface). The forward/VJP method signatures on `FusedOps` MUST NOT change; new params go inside `*ScaledDotProductAttentionConfig`.
- **No push, no PR.** Every task ends at `git commit` on `flash-customcall`. Guy reviews diffs before anything reaches a remote.
- **Fallback is the contract.** Any unsupported config (here: FP8 input dtype) returns a `compute.ErrNotImplemented` wrapped with a stack via `errors.Wrapf`. Never panic on an unsupported-but-valid request.
- Commit messages: terse, verb-first, no em/en dashes, prefix `fused_ops:`, no Claude attribution.
- All tests run on the Mac CPU `go` backend (no CUDA in this plan).

## Grounded code state (verified)

- `ScaledDotProductAttentionConfig` is at `fused_ops.go:206-218`; today it holds only `QuantizedMatmuls bool`.
- The forward method is `fused_ops.go:320-326`; the VJP is `fused_ops.go:336-343`. Neither changes.
- The CPU reference is `internal/gobackend/fusedops/sdpa.go`. The op is currently **disabled** on the go backend: `init()` is gated `if false` (sdpa.go:54-62) and `internal/gobackend/capabilities.go:147` has `OpTypeFusedScaledDotProductAttention: false`. Today `go test ./gobackend/ -run 'TestCompliance/FusedOps/FusedScaledDotProductAttention'` **SKIPs** ("Backend ... does not support operation"). Task 0 re-enables it so the new subtests actually run.
- `equalOptions` (sdpa.go:80-88) compares options with `*d.options == *o.options`. After adding `Value` (= `any`) fields this struct comparison can panic at runtime when a field holds a non-comparable concrete type. Task 1 rewrites it to compare field-by-field. **This is a real bug introduced by adding the fields, not a nicety.**
- `compute.ErrNotImplemented` / `compute.IsNotImplemented` are at `compute.go:110-115`.
- `Value` is `any` (`builder.go:12`).
- F8 dtypes: `dtypes.F8E4M3FN` (=17), `dtypes.F8E5M2` (=16). The go backend rejects an F8 **parameter** at creation, but `fn.ConvertDType(f32param, dtypes.F8E4M3FN)` produces an F8-typed node that reaches the SDPA builder — this is how the FP8 test feeds F8 input (verified).
- Go backend test entrypoint: `go test ./gobackend/ -run TestCompliance/FusedOps` (constructs the `go` backend in `gobackend/gobackend_test.go`, runs the shared `backendtest.TestFusedOps` suite).
- Test helpers: `testutil.Exec1(backend, []any{...}, fn)`, `testutil.IsInDelta(want, got, delta)`, `testutil.SkipIfMissing(t, b, op)` (all in `support/testutil/`).

---

## File map

- `fused_ops.go` — extend `ScaledDotProductAttentionConfig` (Task 1).
- `internal/gobackend/capabilities.go` — flip SDPA capability to `true` (Task 0).
- `internal/gobackend/fusedops/sdpa.go` — re-enable registration (Task 0); rewrite `equalOptions` (Task 1); bias (Task 2), seqlen padding mask (Task 3), dropout (Task 4), FP8 gate (Task 5).
- `support/backendtest/fusedops.go` — new subtests `WithBias`, `WithSeqLens`, `WithSeqLensCausal`, `WithDropoutDeterministic`, `FP8NotImplemented` (Tasks 2-5).

---

### Task 0: Re-enable SDPA on the go backend

**Why:** Every new subtest in this plan exercises `FusedScaledDotProductAttention` on the `go` backend. Today the op is disabled, so `SkipIfMissing` skips the whole group and no new test can prove anything. Re-enabling it makes the existing `BHSD_Causal`/`BSHD_Causal`/`WithBooleanMask`/`QuantizedMatmuls` subtests run for real (they become the regression baseline for this plan).

**Files:**
- Modify: `internal/gobackend/capabilities.go:147`
- Modify: `internal/gobackend/fusedops/sdpa.go:54-62`
- Test (existing harness): `support/backendtest/fusedops.go` (no edit; just runs)

**Interfaces:**
- Consumes: nothing.
- Produces: the `go` backend now reports `OpTypeFusedScaledDotProductAttention` capable; `testutil.SkipIfMissing(t, b, compute.OpTypeFusedScaledDotProductAttention)` no longer skips.

- [ ] **Step 1: Run the existing SDPA subtest to confirm it SKIPs today**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./gobackend/ -run 'TestCompliance/FusedOps/FusedScaledDotProductAttention' -v`
Expected: output contains `SKIP: TestCompliance/FusedOps/FusedScaledDotProductAttention` and `does not support operation FusedScaledDotProductAttention`.

- [ ] **Step 2: Flip the capability flag to true**

In `internal/gobackend/capabilities.go`, change line 147 from:

```go
		compute.OpTypeFusedScaledDotProductAttention: false,
```

to:

```go
		compute.OpTypeFusedScaledDotProductAttention: true,
```

- [ ] **Step 3: Re-enable registration in the init()**

In `internal/gobackend/fusedops/sdpa.go`, replace the `if false { ... }` body (lines 54-62) with an unconditional registration:

```go
func init() {
	gobackend.RegisterFusedScaledDotProductAttention.Register(FusedScaledDotProductAttention, gobackend.PriorityGeneric)
	gobackend.SetNodeExecutor(compute.OpTypeFusedScaledDotProductAttention, gobackend.PriorityTyped, execFusedScaledDotProductAttention)
}
```

- [ ] **Step 4: Run the existing SDPA subtests to confirm they now PASS**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./gobackend/ -run 'TestCompliance/FusedOps/FusedScaledDotProductAttention' -v`
Expected: `PASS` for `BHSD_Causal`, `BSHD_Causal`, `WithBooleanMask`, `QuantizedMatmuls` (no SKIP).

- [ ] **Step 5: Commit**

```bash
cd /Users/guygrigsby/projects/forks/compute
git add internal/gobackend/capabilities.go internal/gobackend/fusedops/sdpa.go
git commit -m "fused_ops: re-enable scaled dot product attention on go backend"
```

---

### Task 1: Extend ScaledDotProductAttentionConfig and fix option equality

**Why:** Adds the new fields (Contract A) and fixes the `equalOptions` comparison that would otherwise panic once the struct holds `Value` (interface) fields.

**Files:**
- Modify: `fused_ops.go:206-218` (struct definition)
- Modify: `internal/gobackend/fusedops/sdpa.go:80-88` (`equalOptions`)
- Test: `support/backendtest/fusedops.go` (new subtest `ConfigFieldsCompile` under `FusedScaledDotProductAttention`)

**Interfaces:**
- Consumes: Task 0 (SDPA enabled on go backend).
- Produces: `compute.ScaledDotProductAttentionConfig` now has exported fields
  `QuantizedMatmuls bool`, `Bias compute.Value`, `QuerySeqLen compute.Value`, `KeyValueSeqLen compute.Value`, `DropoutRate float64`, `DropoutSeed compute.Value`, `DropoutOffset compute.Value`. Later tasks set these fields.

- [ ] **Step 1: Write the failing test**

Add this subtest inside the existing `t.Run("FusedScaledDotProductAttention", ...)` block in `support/backendtest/fusedops.go`, after the `QuantizedMatmuls` subtest (before the closing `})` of the SDPA block, around line 280):

```go
		t.Run("ConfigFieldsCompile", func(t *testing.T) {
			// Setting every new config field with nil/zero values must be accepted
			// (no panic in option equality, output equals the no-config result).
			q := [][][][]float32{{{{1}, {1}}}} // [1,1,2,1]
			k := [][][][]float32{{{{1}, {1}}}}
			v := [][][][]float32{{{{10}, {20}}}}
			cfg := &compute.ScaledDotProductAttentionConfig{
				QuantizedMatmuls: false,
				Bias:             nil,
				QuerySeqLen:      nil,
				KeyValueSeqLen:   nil,
				DropoutRate:      0,
				DropoutSeed:      nil,
				DropoutOffset:    nil,
			}
			got, err := testutil.Exec1(b, []any{q, k, v}, func(f compute.Function, params []compute.Value) (compute.Value, error) {
				out, _, err := f.FusedScaledDotProductAttention(params[0], params[1], params[2], nil, 1, 1, compute.AxesLayoutBHSD, 1.0, true, cfg)
				return out, err
			})
			if err != nil {
				t.Fatalf("SDPA with zero-value config failed: %+v", err)
			}
			want := [][][][]float32{{{{10}, {15}}}}
			if ok, diff := testutil.IsInDelta(want, got, fusedTestTolerance); !ok {
				t.Errorf("SDPA zero-value config mismatch:\n%s", diff)
			}
		})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./gobackend/ -run 'TestCompliance/FusedOps/FusedScaledDotProductAttention/ConfigFieldsCompile' -v`
Expected: COMPILE FAILURE — `unknown field 'Bias' in struct literal of type compute.ScaledDotProductAttentionConfig` (and similarly for the other new fields).

- [ ] **Step 3: Add the fields to the struct**

In `fused_ops.go`, replace the `ScaledDotProductAttentionConfig` struct (lines 206-218) with:

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

	// Bias is an optional additive attention bias broadcast to [B,H,S,Skv], added to
	// the scaled scores before softmax. Selects the fmhaScaleBias* variants in the xla
	// backend. nil = unused.
	Bias Value

	// QuerySeqLen, KeyValueSeqLen are optional per-batch actual sequence lengths
	// (int32 tensors, shape [B]). When set, the backend masks by sequence length
	// (padding mask) instead of a materialized [S,Skv] mask. Combined with causal=true
	// this is a padding-causal mask. nil = unused.
	QuerySeqLen, KeyValueSeqLen Value

	// DropoutRate in [0,1); 0 disables. Nonzero selects the fmha*Dropout variants.
	// DropoutSeed and DropoutOffset feed the backend RNG (Value: int64 scalars) so the
	// dropout pattern is deterministic for a given (seed, offset). nil seed/offset with a
	// nonzero rate is treated as seed=0, offset=0.
	DropoutRate                float64
	DropoutSeed, DropoutOffset Value
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
		a.DropoutRate == b.DropoutRate &&
		a.Bias == b.Bias &&
		a.QuerySeqLen == b.QuerySeqLen &&
		a.KeyValueSeqLen == b.KeyValueSeqLen &&
		a.DropoutSeed == b.DropoutSeed &&
		a.DropoutOffset == b.DropoutOffset
}
```

Note: the `Value` (== `any`) comparisons here compare the interface values the caller passed. For the go backend these are `*gobackend.Value` pointers (comparable), so `==` is pointer identity and never panics. If a backend ever stores a non-comparable concrete value, dedup falls back to "not equal", which is safe.

- [ ] **Step 5: Run test to verify it passes**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./gobackend/ -run 'TestCompliance/FusedOps/FusedScaledDotProductAttention/ConfigFieldsCompile' -v`
Expected: PASS.

- [ ] **Step 6: Run the whole SDPA group to confirm no regression**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./gobackend/ -run 'TestCompliance/FusedOps/FusedScaledDotProductAttention' -v`
Expected: PASS for all subtests (`BHSD_Causal`, `BSHD_Causal`, `WithBooleanMask`, `QuantizedMatmuls`, `ConfigFieldsCompile`).

- [ ] **Step 7: Commit**

```bash
cd /Users/guygrigsby/projects/forks/compute
git add fused_ops.go internal/gobackend/fusedops/sdpa.go support/backendtest/fusedops.go
git commit -m "fused_ops: add bias/seqlen/dropout fields to attention config"
```

---

### Task 2: CPU reference — additive Bias before softmax

**Why:** Contract A: the go backend implements `Bias` (additive pre-softmax bias broadcast to `[B,H,S,Skv]`).

**Files:**
- Modify: `internal/gobackend/fusedops/sdpa.go` (`buildSDPANode`, `execFusedScaledDotProductAttention`, `sdpaMultiHeadGeneric`, `sdpaGeneric`)
- Test: `support/backendtest/fusedops.go` (new subtest `WithBias`)

**Interfaces:**
- Consumes: Task 1 (`options.Bias compute.Value`).
- Produces: when `options.Bias != nil`, the go backend adds `bias[b,h,s,kv]` to the scaled score before softmax. Bias is a float tensor broadcastable to the score shape `[B,H,S,Skv]` using the same stride convention as the additive `mask` (`sdpaComputeMaskStrides`). Bias and `mask` may both be set; both are added.

- [ ] **Step 1: Write the failing test**

Add this subtest inside the SDPA block in `support/backendtest/fusedops.go`, after `ConfigFieldsCompile`:

```go
		t.Run("WithBias", func(t *testing.T) {
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
		})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./gobackend/ -run 'TestCompliance/FusedOps/FusedScaledDotProductAttention/WithBias' -v`
Expected: FAIL — the bias is ignored, so the output is the plain mean `(10+20)/2 = 15`, not `~10.007`. Failure message shows want `~10.0067` got `15`.

- [ ] **Step 3: Thread Bias through buildSDPANode as an extra operand**

In `internal/gobackend/fusedops/sdpa.go`, in `buildSDPANode`, after appending `mask` to `values` (after the `if mask != nil` block, around line 100), append the bias operand and record whether it is present so exec can find it:

```go
	values := []compute.Value{query, key, value}
	if mask != nil {
		values = append(values, mask)
	}
	hasBias := options != nil && options.Bias != nil
	if hasBias {
		values = append(values, options.Bias)
	}
```

The node data needs to know whether a mask and a bias operand are present (so exec can index `inputs` correctly). Extend `nodeScaledDotProductAttention` (lines 64-71) with two bool flags:

```go
type nodeScaledDotProductAttention struct {
	numHeads   int
	numKVHeads int
	axesLayout compute.AxesLayout
	scale      float64
	causal     bool
	hasMask    bool
	hasBias    bool
	options    *compute.ScaledDotProductAttentionConfig
}
```

Set them when constructing `data` (replace the `data := &nodeScaledDotProductAttention{...}` line ~114):

```go
	data := &nodeScaledDotProductAttention{
		numHeads: numHeads, numKVHeads: numKVHeads, axesLayout: axesLayout,
		scale: scale, causal: causal, hasMask: mask != nil, hasBias: hasBias,
		options: options,
	}
```

Add the two flags to `EqualNodeData` (lines 73-78):

```go
func (d *nodeScaledDotProductAttention) EqualNodeData(other gobackend.NodeDataComparable) bool {
	o := other.(*nodeScaledDotProductAttention)
	return d.numHeads == o.numHeads && d.numKVHeads == o.numKVHeads &&
		d.axesLayout == o.axesLayout && d.scale == o.scale && d.causal == o.causal &&
		d.hasMask == o.hasMask && d.hasBias == o.hasBias &&
		d.equalOptions(o)
}
```

- [ ] **Step 4: Locate the bias operand in exec and pass it down**

In `execFusedScaledDotProductAttention` (lines 128-171), replace the mask-locating block (lines 134-137) with logic that locates both mask and bias by their flags, then transpose the bias the same way the mask is transposed for BSHD rank-4, and pass a bias buffer plus its strides into the kernel:

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
	var bias *gobackend.Buffer
	if data.hasBias {
		bias = inputs[next]
		next++
	}

	// For rank-4 BSHD masks/bias [batch, seq, heads, kvLen], transpose to BHSD so
	// per-head data is contiguous [seqLen, kvLen]. These tensors are small.
	if data.axesLayout == compute.AxesLayoutBSHD {
		var err error
		if mask != nil && mask.RawShape.Rank() == 4 {
			if mask, err = transposeBuffer(backend, mask, []int{0, 2, 1, 3}); err != nil {
				return nil, err
			}
		}
		if bias != nil && bias.RawShape.Rank() == 4 {
			if bias, err = transposeBuffer(backend, bias, []int{0, 2, 1, 3}); err != nil {
				return nil, err
			}
		}
	}

	output, err := backend.GetBuffer(query.RawShape.Clone())
	if err != nil {
		return nil, err
	}

	// Compute mask/bias strides for broadcasting (BHSD convention).
	var maskBatchStride, maskHeadStride int
	if mask != nil {
		maskBatchStride, maskHeadStride = sdpaComputeMaskStrides(mask.RawShape.Dimensions)
	}
	var biasBatchStride, biasHeadStride int
	if bias != nil {
		biasBatchStride, biasHeadStride = sdpaComputeMaskStrides(bias.RawShape.Dimensions)
	}

	switch query.RawShape.DType {
	case dtypes.Float32:
		sdpaMultiHeadGeneric[float32](query, key, value, mask, bias, output, data, maskBatchStride, maskHeadStride, biasBatchStride, biasHeadStride)
	case dtypes.Float64:
		sdpaMultiHeadGeneric[float64](query, key, value, mask, bias, output, data, maskBatchStride, maskHeadStride, biasBatchStride, biasHeadStride)
	default:
		return nil, errors.Errorf("FusedScaledDotProductAttention: unsupported dtype %s", query.RawShape.DType)
	}

	return output, nil
```

- [ ] **Step 5: Add bias to the kernel signatures and apply it before softmax**

Extend `sdpaMultiHeadGeneric` (line 368) to accept the bias buffer and strides, slice it per KV-head group exactly like the additive mask, and forward a per-group bias slice into `sdpaGeneric`. Replace the signature and the mask-slicing/`sdpaGeneric` call. New signature:

```go
func sdpaMultiHeadGeneric[T float32 | float64](query, key, value, mask, bias, output *gobackend.Buffer, data *nodeScaledDotProductAttention, maskBatchStride, maskHeadStride, biasBatchStride, biasHeadStride int) {
```

Just after the existing `var additiveMask []T; var booleanMask []bool` block (after line 381), add bias extraction:

```go
	var biasFlat []T
	if bias != nil {
		biasFlat = bias.Flat.([]T)
	}
```

Inside the `for kvHeadIdx` loop, after the mask-slice block (after line 446, before the `sdpaGeneric(...)` call), compute the bias slice and group stride the same way:

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

Replace the `sdpaGeneric(...)` call (lines 447-453) to pass bias:

```go
			sdpaGeneric(
				q, k, v, qOff, kvOff, qSeqStride, kvSeqStride, qHeadStride,
				additiveMaskSlice, booleanMaskSlice, maskGroupStride,
				biasSlice, biasGroupStride,
				scores,
				out,
				groupSize, seqLen, kvLen, headDim, scale, causal,
			)
```

Extend `sdpaGeneric` (lines 241-249) to accept `bias []T, biasGroupStride int`. New signature (insert the two bias params after `maskGroupStride int`):

```go
func sdpaGeneric[T float32 | float64](
	q, k, v []T, qOff, kvOff, qSeqStride, kvSeqStride, qGroupStride int,
	additiveMask []T,
	booleanMask []bool,
	maskGroupStride int,
	bias []T,
	biasGroupStride int,
	scores []T,
	output []T,
	groupSize, seqLen, kvLen, headDim int, scale T, causal bool,
) {
```

Inside the head loop, the bias offset mirrors the mask offset. After `gMaskOff := gIdx * maskGroupStride` (line 252) add:

```go
		gBiasOff := gIdx * biasGroupStride
```

And after `maskIdxBase := gMaskOff + qIdx*kvLen` (line 257) add:

```go
			biasIdxBase := gBiasOff + qIdx*kvLen
```

In the score computation loop, after the additive-mask add (lines 286-288, `if len(additiveMask) > 0 { s += additiveMask[maskIdx] }`), add the bias term:

```go
				if len(bias) > 0 {
					s += bias[biasIdxBase+kvIdx]
				}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./gobackend/ -run 'TestCompliance/FusedOps/FusedScaledDotProductAttention/WithBias' -v`
Expected: PASS.

- [ ] **Step 7: Run the whole SDPA group to confirm no regression**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./gobackend/ -run 'TestCompliance/FusedOps/FusedScaledDotProductAttention' -v`
Expected: PASS for all subtests.

- [ ] **Step 8: Commit**

```bash
cd /Users/guygrigsby/projects/forks/compute
git add internal/gobackend/fusedops/sdpa.go support/backendtest/fusedops.go
git commit -m "fused_ops: implement additive attention bias in go backend"
```

---

### Task 3: CPU reference — sequence-length padding mask

**Why:** Contract A: `QuerySeqLen`/`KeyValueSeqLen` (int32 `[B]` vectors) build a padding mask `[B,1,S,Skv]` (query position `s` valid iff `s < QuerySeqLen[b]`; key position `kv` valid iff `kv < KeyValueSeqLen[b]`), combined with `causal` to give a padding-causal mask. This task implements the padding masking and its combination with causal.

**Files:**
- Modify: `internal/gobackend/fusedops/sdpa.go` (`buildSDPANode`, `execFusedScaledDotProductAttention`, `sdpaMultiHeadGeneric`, `sdpaGeneric`)
- Test: `support/backendtest/fusedops.go` (new subtests `WithSeqLens`, `WithSeqLensCausal`)

**Interfaces:**
- Consumes: Task 2 (bias plumbing pattern; `options.QuerySeqLen`, `options.KeyValueSeqLen compute.Value`).
- Produces: when both `options.QuerySeqLen != nil` and `options.KeyValueSeqLen != nil`, the go backend masks (sets weight 0) any score where the query index >= `QuerySeqLen[b]` or the key index >= `KeyValueSeqLen[b]`. A masked query row whose every key is masked yields an all-zero output row (the existing `sum==0 -> invSum=0` guard). Combined with `causal=true`, both constraints apply.

- [ ] **Step 1: Write the failing tests**

Add these two subtests inside the SDPA block in `support/backendtest/fusedops.go`, after `WithBias`:

```go
		t.Run("WithSeqLens", func(t *testing.T) {
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
			// Equivalent materialized mask: only key0 valid -> every query attends to key0 only -> output 10.
			want := [][][][]float32{{{{10}, {10}}}}
			if ok, diff := testutil.IsInDelta(want, got, 1e-5); !ok {
				t.Errorf("SDPA seqlens padding mask mismatch:\n%s", diff)
			}
		})

		t.Run("WithSeqLensCausal", func(t *testing.T) {
			// causal + KeyValueSeqLen=2 (no key padding) reduces to plain causal.
			// QuerySeqLen=2. query0 sees key0 only (causal) -> 10; query1 sees key0,key1 -> 15.
			q := [][][][]float32{{{{1}, {1}}}}   // [1,1,2,1]
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
		})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./gobackend/ -run 'TestCompliance/FusedOps/FusedScaledDotProductAttention/WithSeqLens' -v`
Expected: FAIL — seqlens ignored. `WithSeqLens` returns the plain mean `15` for both rows instead of `10`. (`WithSeqLensCausal` passes by accident since `kvLen==2` is no padding, but keep it; it pins the causal-combination behavior once Step 3 lands.)

- [ ] **Step 3: Thread seqlens through buildSDPANode as extra operands**

In `internal/gobackend/fusedops/sdpa.go`, in `buildSDPANode`, after the bias-append block from Task 2, append the two seqlen operands when both are set (they are mutually-required: a padding mask needs both length vectors):

```go
	hasSeqLens := options != nil && options.QuerySeqLen != nil && options.KeyValueSeqLen != nil
	if (options != nil && (options.QuerySeqLen != nil) != (options.KeyValueSeqLen != nil)) {
		return nil, errors.Errorf("%s: QuerySeqLen and KeyValueSeqLen must both be set or both nil", opName)
	}
	if hasSeqLens {
		values = append(values, options.QuerySeqLen, options.KeyValueSeqLen)
	}
```

Add a `hasSeqLens` flag to `nodeScaledDotProductAttention`:

```go
type nodeScaledDotProductAttention struct {
	numHeads    int
	numKVHeads  int
	axesLayout  compute.AxesLayout
	scale       float64
	causal      bool
	hasMask     bool
	hasBias     bool
	hasSeqLens  bool
	options     *compute.ScaledDotProductAttentionConfig
}
```

Set it in the `data := &nodeScaledDotProductAttention{...}` literal (`hasSeqLens: hasSeqLens,`), and add it to `EqualNodeData` (`&& d.hasSeqLens == o.hasSeqLens`).

- [ ] **Step 4: Locate seqlen operands in exec and pass lengths down**

In `execFusedScaledDotProductAttention`, after the bias-locating block from Task 2, locate the two seqlen buffers (they follow bias in operand order) and read them into `[]int32` slices:

```go
	var querySeqLen, keyValueSeqLen []int32
	if data.hasSeqLens {
		qLenBuf := inputs[next]
		next++
		kvLenBuf := inputs[next]
		next++
		querySeqLen = qLenBuf.Flat.([]int32)
		keyValueSeqLen = kvLenBuf.Flat.([]int32)
	}
```

Pass both slices into `sdpaMultiHeadGeneric` (extend both the `case dtypes.Float32` and `case dtypes.Float64` calls). New calls:

```go
	case dtypes.Float32:
		sdpaMultiHeadGeneric[float32](query, key, value, mask, bias, output, data, maskBatchStride, maskHeadStride, biasBatchStride, biasHeadStride, querySeqLen, keyValueSeqLen)
	case dtypes.Float64:
		sdpaMultiHeadGeneric[float64](query, key, value, mask, bias, output, data, maskBatchStride, maskHeadStride, biasBatchStride, biasHeadStride, querySeqLen, keyValueSeqLen)
```

- [ ] **Step 5: Apply the padding mask in the kernel**

Extend `sdpaMultiHeadGeneric` signature to accept the two length slices:

```go
func sdpaMultiHeadGeneric[T float32 | float64](query, key, value, mask, bias, output *gobackend.Buffer, data *nodeScaledDotProductAttention, maskBatchStride, maskHeadStride, biasBatchStride, biasHeadStride int, querySeqLen, keyValueSeqLen []int32) {
```

Inside the `for batchIdx` loop (before the `for kvHeadIdx` loop, after line 425), compute this batch's effective query/key limits (default to full length when no seqlens):

```go
		qLimit := seqLen
		kvLimit := kvLen
		if len(querySeqLen) > 0 {
			qLimit = int(querySeqLen[batchIdx])
		}
		if len(keyValueSeqLen) > 0 {
			kvLimit = int(keyValueSeqLen[batchIdx])
		}
```

Pass `qLimit, kvLimit` into `sdpaGeneric`. Extend its signature with `qLimit, kvLimit int` (after `causal bool`):

```go
	groupSize, seqLen, kvLen, headDim int, scale T, causal bool,
	qLimit, kvLimit int,
) {
```

Update the `sdpaGeneric(...)` call site to pass `qLimit, kvLimit`.

Inside `sdpaGeneric`, apply both limits. The key limit narrows the unmasked key range; fold it into `kvLenUnmasked` (replace lines 259-262):

```go
			kvLenUnmasked := kvLen
			if kvLimit < kvLenUnmasked {
				kvLenUnmasked = kvLimit
			}
			if causal {
				kvLenUnmasked = min(kvLenUnmasked, qIdx+1)
			}
```

The query limit zeroes whole rows: at the top of the `for qIdx := range seqLen` loop body, before computing scores, skip-and-zero padded query rows. Insert right after the `biasIdxBase := gBiasOff + qIdx*kvLen` line (added in Task 2 Step 5), add:

```go
			if qIdx >= qLimit {
				// Padded query position: emit a zero output row, skip attention.
				// The block has its own scope, so this padBase does not collide with
				// the outBase declared later in the loop body.
				padBase := gQOff + qIdx*qSeqStride
				for d := range headDim {
					output[padBase+d] = 0
				}
				continue
			}
```

(Placement: this `continue` comes before the score loop. It uses its own local `padBase`, distinct from the `outBase` the accumulation loop declares later, so there is no redeclaration.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./gobackend/ -run 'TestCompliance/FusedOps/FusedScaledDotProductAttention/WithSeqLens' -v`
Expected: PASS for both `WithSeqLens` and `WithSeqLensCausal`.

- [ ] **Step 7: Run the whole SDPA group to confirm no regression**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./gobackend/ -run 'TestCompliance/FusedOps/FusedScaledDotProductAttention' -v`
Expected: PASS for all subtests.

- [ ] **Step 8: Commit**

```bash
cd /Users/guygrigsby/projects/forks/compute
git add internal/gobackend/fusedops/sdpa.go support/backendtest/fusedops.go
git commit -m "fused_ops: implement seqlen padding mask in go backend"
```

---

### Task 4: CPU reference — deterministic dropout

**Why:** Contract A: implement `DropoutRate` deterministically from `DropoutSeed`/`DropoutOffset` so CPU-vs-CPU tests are stable (same seed -> same output; different seed -> different).

**Files:**
- Modify: `internal/gobackend/fusedops/sdpa.go` (`buildSDPANode`, `execFusedScaledDotProductAttention`, `sdpaMultiHeadGeneric`, `sdpaGeneric`)
- Test: `support/backendtest/fusedops.go` (new subtest `WithDropoutDeterministic`)

**Interfaces:**
- Consumes: Task 3 (operand-threading pattern; `options.DropoutRate float64`, `options.DropoutSeed`, `options.DropoutOffset compute.Value`).
- Produces: when `options.DropoutRate > 0`, each post-softmax attention weight is independently dropped with probability `DropoutRate` and surviving weights are scaled by `1/(1-DropoutRate)` (inverted dropout), using a deterministic per-element PRNG keyed on `(seed, offset, b, h, qIdx, kvIdx)`. `DropoutSeed`/`DropoutOffset` are optional int64 scalar `Value`s; nil means 0. The pattern is identical across runs for a fixed `(seed, offset)`.

- [ ] **Step 1: Write the failing test**

Add this subtest inside the SDPA block in `support/backendtest/fusedops.go`, after `WithSeqLensCausal`:

```go
		t.Run("WithDropoutDeterministic", func(t *testing.T) {
			// Wide attention (8 keys, rate 0.5) so dropout changes the output with high
			// probability. Same seed -> identical output; different seed -> different.
			q := [][][][]float32{{{{1}}}} // [1,1,1,1]
			k := [][][][]float32{{{{1}, {1}, {1}, {1}, {1}, {1}, {1}, {1}}}}      // [1,1,8,1]
			v := [][][][]float32{{{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}}}}      // [1,1,8,1]
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
			b1 := run(42)
			if ok, _ := testutil.IsInDelta(a1, a2, 1e-9); !ok {
				t.Errorf("same seed must produce identical dropout output: %v vs %v", a1, a2)
			}
			if ok, _ := testutil.IsInDelta(a1, b1, 1e-9); ok {
				t.Errorf("different seeds must produce different dropout output, both = %v", a1)
			}
		})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./gobackend/ -run 'TestCompliance/FusedOps/FusedScaledDotProductAttention/WithDropoutDeterministic' -v`
Expected: FAIL — dropout ignored, so seed 7 and seed 42 give the identical (undropped) output, tripping the "different seeds must produce different" assertion.

- [ ] **Step 3: Thread dropout params through buildSDPANode**

In `internal/gobackend/fusedops/sdpa.go`, in `buildSDPANode`, after the seqlen-append block from Task 3, append the optional dropout seed/offset operands. Read the scalar seed/offset are passed as operands so the graph can carry runtime RNG state; nil defaults to a constant 0 captured in node data:

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
	hasDropout      bool
	hasDropoutSeed  bool
	hasDropoutOff   bool
```

Set them in the `data` literal:

```go
		hasDropout:     hasDropout,
		hasDropoutSeed: hasDropout && options.DropoutSeed != nil,
		hasDropoutOff:  hasDropout && options.DropoutOffset != nil,
```

Add all three to `EqualNodeData` (`&& d.hasDropout == o.hasDropout && d.hasDropoutSeed == o.hasDropoutSeed && d.hasDropoutOff == o.hasDropoutOff`).

- [ ] **Step 4: Read seed/offset in exec and pass dropout params down**

In `execFusedScaledDotProductAttention`, after the seqlen block from Task 3, read the optional seed/offset scalars (default 0) and the rate:

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

Pass `dropoutRate, dropoutSeed, dropoutOffset` into `sdpaMultiHeadGeneric` (both dtype cases):

```go
	case dtypes.Float32:
		sdpaMultiHeadGeneric[float32](query, key, value, mask, bias, output, data, maskBatchStride, maskHeadStride, biasBatchStride, biasHeadStride, querySeqLen, keyValueSeqLen, dropoutRate, dropoutSeed, dropoutOffset)
	case dtypes.Float64:
		sdpaMultiHeadGeneric[float64](query, key, value, mask, bias, output, data, maskBatchStride, maskHeadStride, biasBatchStride, biasHeadStride, querySeqLen, keyValueSeqLen, dropoutRate, dropoutSeed, dropoutOffset)
```

- [ ] **Step 5: Add a deterministic per-element PRNG and apply inverted dropout**

At the top of `internal/gobackend/fusedops/sdpa.go`, after the imports, add a splitmix64-based hash that is pure (no state): given the key tuple it returns a uniform float64 in `[0,1)`.

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

Extend `sdpaMultiHeadGeneric` signature to accept dropout params:

```go
func sdpaMultiHeadGeneric[T float32 | float64](query, key, value, mask, bias, output *gobackend.Buffer, data *nodeScaledDotProductAttention, maskBatchStride, maskHeadStride, biasBatchStride, biasHeadStride int, querySeqLen, keyValueSeqLen []int32, dropoutRate float64, dropoutSeed, dropoutOffset int64) {
```

Dropout is applied per (batch, head, query, key). The kernel `sdpaGeneric` works per KV-head group; to key the PRNG on the absolute head index, pass the absolute base head index plus dropout params into `sdpaGeneric`. Add params to its signature (after `qLimit, kvLimit int`):

```go
	qLimit, kvLimit int,
	dropoutRate float64, dropoutSeed, dropoutOffset int64,
	batchIdx, baseHeadIdx int,
) {
```

At the call site in `sdpaMultiHeadGeneric`, pass `dropoutRate, dropoutSeed, dropoutOffset, batchIdx, kvHeadIdx*groupSize`.

Inside `sdpaGeneric`, after the softmax normalization produces the final per-key weights and before the output accumulation (i.e. after the block that writes `scores[scoreIdx] *= invSum`, around lines 322-340), apply inverted dropout to the normalized weights when `dropoutRate > 0`:

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

(Place this block inside the `for qIdx` loop, after the normalization branches and before the `output[qIdx][d]` accumulation loop that starts at `outBase := gQOff + qIdx*qSeqStride`.)

- [ ] **Step 6: Run test to verify it passes**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./gobackend/ -run 'TestCompliance/FusedOps/FusedScaledDotProductAttention/WithDropoutDeterministic' -v`
Expected: PASS (same seed identical, different seeds differ).

- [ ] **Step 7: Run the whole SDPA group to confirm no regression**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./gobackend/ -run 'TestCompliance/FusedOps/FusedScaledDotProductAttention' -v`
Expected: PASS for all subtests (dropout off by default leaves prior subtests unchanged).

- [ ] **Step 8: Commit**

```bash
cd /Users/guygrigsby/projects/forks/compute
git add internal/gobackend/fusedops/sdpa.go support/backendtest/fusedops.go
git commit -m "fused_ops: implement deterministic dropout in go backend"
```

---

### Task 5: CPU reference — FP8 input dtype returns ErrNotImplemented

**Why:** Contract A: the go backend does not implement FP8 attention; an FP8 (e4m3/e5m2) input dtype must return a wrapped `compute.ErrNotImplemented` so the caller falls back to the decomposed path. Never panic.

**Files:**
- Modify: `internal/gobackend/fusedops/sdpa.go` (`buildSDPANode`)
- Test: `support/backendtest/fusedops.go` (new subtest `FP8NotImplemented`)

**Interfaces:**
- Consumes: Task 1 (struct in place); the `buildSDPANode` dtype check sees `qNode.Shape.DType`.
- Produces: `FusedScaledDotProductAttention` (and its VJP) return `err` with `compute.IsNotImplemented(err) == true` when `query` dtype is `dtypes.F8E4M3FN` or `dtypes.F8E5M2`.

- [ ] **Step 1: Write the failing test**

Add this subtest inside the SDPA block in `support/backendtest/fusedops.go`, after `WithDropoutDeterministic`:

```go
		t.Run("FP8NotImplemented", func(t *testing.T) {
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
		})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./gobackend/ -run 'TestCompliance/FusedOps/FusedScaledDotProductAttention/FP8NotImplemented' -v`
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

(`dtypes` and the `compute` package are already imported in this file; `errors` is `github.com/pkg/errors`, also already imported.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./gobackend/ -run 'TestCompliance/FusedOps/FusedScaledDotProductAttention/FP8NotImplemented' -v`
Expected: PASS.

- [ ] **Step 5: Run the whole SDPA group to confirm no regression**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./gobackend/ -run 'TestCompliance/FusedOps/FusedScaledDotProductAttention' -v`
Expected: PASS for all subtests.

- [ ] **Step 6: Commit**

```bash
cd /Users/guygrigsby/projects/forks/compute
git add internal/gobackend/fusedops/sdpa.go support/backendtest/fusedops.go
git commit -m "fused_ops: return ErrNotImplemented for float8 attention dtype"
```

---

## Verification gate (01 compute)

Per the contract, plan 01 is done when the full suite is green on the Mac CPU backend, with the new subtests exercising bias/seqlen/dropout reference paths and the FP8 NotImplemented fallback.

- [ ] **Full module build:**

Run: `cd /Users/guygrigsby/projects/forks/compute && go build ./...`
Expected: no output, exit 0.

- [ ] **Full test suite green:**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./...`
Expected: all packages `ok` (or `[no test files]`), no `FAIL`.

- [ ] **SDPA subtests all run (no SKIP) and pass:**

Run: `cd /Users/guygrigsby/projects/forks/compute && go test ./gobackend/ -run 'TestCompliance/FusedOps/FusedScaledDotProductAttention' -v`
Expected: PASS (no SKIP) for `BHSD_Causal`, `BSHD_Causal`, `WithBooleanMask`, `QuantizedMatmuls`, `ConfigFieldsCompile`, `WithBias`, `WithSeqLens`, `WithSeqLensCausal`, `WithDropoutDeterministic`, `FP8NotImplemented`.

- [ ] **Vet clean:**

Run: `cd /Users/guygrigsby/projects/forks/compute && go vet ./...`
Expected: no output, exit 0.

When green, the downstream go-xla#37 plan (02) can consume the new `ScaledDotProductAttentionConfig` fields via the local `replace` directive. No push, no PR from this plan.
