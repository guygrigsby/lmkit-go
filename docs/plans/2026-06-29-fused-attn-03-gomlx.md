# Fused Attention — Plan 03: gomlx Layers (UseFusion, SeqLens, generalized tests)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the CUDA-shaped `FlashAttention` wrapper, expose fusion through the `MultiHeadAttention` builder (`UseFusion`, `WithSeqLens`, `SimpleAttention`), and make the attention tests backend-general instead of `isCUDABackend`-gated, per Contract D.

**Architecture:** The fused capability already lives in `attention.Core`'s fused branch (`InternalFusedOpCaller(fused, decomposed)`), so the standalone `FlashAttention` function is redundant and CUDA-flavored — delete it. Add a per-call `UseFusion` gate threaded `builder → doneInternal → Core → fused branch`, a `WithSeqLens` path that populates `compute.ScaledDotProductAttentionConfig.QuerySeqLen/KeyValueSeqLen`, and rewrite the tests to probe each official backend for fusion support (`backendSupportsFusion`) rather than sniff the backend name.

**Tech Stack:** Go 1.26, `github.com/gomlx/gomlx` (fork, branch `flash-attention`), depends on `github.com/gomlx/compute` (fork, branch `flash-customcall`) and `github.com/gomlx/go-xla` (fork, branch `flash-attention`) via local `replace`.

**Contract:** This plan **produces Contract D** and **consumes Contracts A and C** of `docs/plans/2026-06-29-fused-attn-00-contract.md`. Read the contract first. Global constraints (Go 1.26, no push/no PR, CUDA tests `[trig]` only, fallback-on-`ErrNotImplemented`, voice-rule commits with `attention:` prefix and no Claude attribution) are defined there and apply to every task below; they are not restated per task.

## Global Constraints

- Repo `/Users/guygrigsby/projects/forks/gomlx`, branch `flash-attention`, module `github.com/gomlx/gomlx`, Go 1.26.
- This plan depends on Contract A landing first: `compute.ScaledDotProductAttentionConfig` must already carry `QuerySeqLen, KeyValueSeqLen Value` (plan 01). Task 0 verifies the local `replace` points at the updated compute fork and the fields exist; if they do not, stop — plan 01 is not done.
- Commit prefix `attention:` (existing scheme in this package). Terse, verb-first, no em/en dashes, no Claude attribution.
- CUDA execution is `[trig]` only; the Mac CI has no cuDNN. Mark every step whose `Run:` needs `xla:cuda` with **[trig]**. Everything else runs on the Mac `go` (CPU) backend.
- Fallback is the contract: an unsupported-but-valid request returns wrapped `compute.ErrNotImplemented` and falls back to the decomposed path. Never panic on unsupported-but-valid.

---

## File Structure

- `ml/layers/attention/flash.go` — **DELETE**. `FlashAttention` is superseded by `Core`'s fused branch; `naiveCausalAttention` and `repeatKVHeads` are test-only helpers that relocate to a test file.
- `ml/layers/attention/attention.go` — `Core` gains a trailing `useFusion bool` param; the fused branch additionally gates on it and threads a `*compute.ScaledDotProductAttentionConfig` (built from seqlens) into `BackendFusedScaledDotProductAttention`.
- `ml/layers/attention/multiheadattention.go` — builder gains `useFusion bool` (default true, set in constructor), `querySeqLen/keyValueSeqLen *Node` fields, methods `UseFusion`, `WithSeqLens`, and package function `SimpleAttention`; `doneInternal` builds the config and passes `useFusion` to `Core`.
- `ml/layers/attention/fusion_test.go` — **NEW**. Houses the relocated reference helpers (`naiveCausalAttention`, `naiveGQAReference`, `repeatKVHeads`), the `backendSupportsFusion` probe, the CPU-testable builder/wiring tests, and the backend-general parity tests (loop `TestOfficialBackends`, skip when unsupported).
- `ml/layers/attention/flash_test.go` — **DELETE** (its content moves to `fusion_test.go`, generalized).
- `ml/layers/attention/flash_bench_test.go` → rename to `ml/layers/attention/fusion_bench_test.go`; `attentionStep` toggles fusion via `UseFusion`, not a separate code path; env-var probe gating becomes Go test flags.
- `support/testutil/testutil.go` — add `GetOfficialBackend(name string) compute.Backend`.

---

## Task 0: Verify Contract A landed (no code)

**Files:**
- Inspect: `/Users/guygrigsby/projects/forks/gomlx/go.mod` (the `replace github.com/gomlx/compute` directive)
- Inspect: the compute fork `fused_ops.go` reachable through that replace.

**Interfaces:**
- Consumes (Contract A): `compute.ScaledDotProductAttentionConfig` with fields `QuerySeqLen, KeyValueSeqLen compute.Value`.

- [ ] **Step 1: Confirm the local replace and the fields exist**

Run:
```bash
cd /Users/guygrigsby/projects/forks/gomlx
grep -n 'replace.*gomlx/compute' go.mod
go doc github.com/gomlx/compute.ScaledDotProductAttentionConfig
```
Expected: `go.mod` has a `replace github.com/gomlx/compute => /Users/guygrigsby/projects/forks/compute` (or equivalent local path), and `go doc` lists `QuerySeqLen Value` and `KeyValueSeqLen Value` on the struct.

- [ ] **Step 2: Decision gate**

If the replace is missing or the fields are absent, STOP and report: "plan 01 (compute) is not integrated; cannot proceed with plan 03." Otherwise continue to Task 1. (No commit — this is a read-only gate.)

---

## Task 1: Relocate test-only helpers, then delete flash.go

This task moves the three helpers `FlashAttention`'s file shares with the tests (`naiveCausalAttention`, `naiveGQAReference`, `repeatKVHeads`) into a new test file, then deletes `flash.go` and `flash_test.go`. `naiveGQAReference` is already test-only (it lives in `flash_test.go`); it moves verbatim. `naiveCausalAttention` and `repeatKVHeads` currently live in `flash.go` (non-test) but are referenced only by tests once `FlashAttention` is gone — verify that and relocate them to the test file so production code carries nothing CUDA-shaped.

**Why deletion loses no behavior:** `FlashAttention(q,k,v,scale)` did exactly three things — validate rank-4 / equal-head shapes, repeat KV heads for GQA, then call `BackendFusedScaledDotProductAttention(..., causal=true, options=nil)` with a `naiveCausalAttention` fallback on `compute.IsNotImplemented`. `Core`'s fused branch (`attention.go:268-280`) already calls `BackendFusedScaledDotProductAttention` through `InternalFusedOpCaller`, which performs the identical `ErrNotImplemented → decomposed` fallback, and `Core`'s decomposed path already handles GQA via `reshapeQueryForGQA` (no `repeatKVHeads` needed in production). So every capability `FlashAttention` provided is reachable through `MultiHeadAttention(...).UseCausalMask().Done()` / `Core(..., useCausalMask=true, ...)`. The deletion only removes a redundant, CUDA-named entry point.

**Files:**
- Create: `ml/layers/attention/fusion_test.go` (helpers only, for now)
- Delete: `ml/layers/attention/flash.go`
- Delete: `ml/layers/attention/flash_test.go`
- Inspect: `ml/layers/attention/flash_bench_test.go` (still references `naiveCausalAttention`, `repeatKVHeads` — keeps compiling because they now live in `fusion_test.go`, same package).

**Interfaces:**
- Produces (test-internal, same package `attention`):
  - `func naiveCausalAttention(query, key, value *Node, scale float64) *Node`
  - `func naiveGQAReference(query, key, value *Node, numKVHeads int, scale float64) *Node`
  - `func repeatKVHeads(x *Node, group int) *Node`
- Removes: `func FlashAttention(query, key, value *Node, scale float64) *Node` (no production caller; grep confirmed zero usages outside the deleted files).

- [ ] **Step 1: Write `fusion_test.go` with the relocated helpers**

Create `ml/layers/attention/fusion_test.go`:
```go
// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
)

// naiveCausalAttention is the decomposed reference and fallback: softmax(scale*QK^T + causal)*V
// in float32. query/key/value are [B,S,H,D] with equal heads.
func naiveCausalAttention(query, key, value *Node, scale float64) *Node {
	g := query.Graph()
	q := ConvertDType(query, dtypes.Float32)
	k := ConvertDType(key, dtypes.Float32)
	v := ConvertDType(value, dtypes.Float32)
	dims := q.Shape().Dimensions
	batch, seqLen, heads := dims[0], dims[1], dims[2]

	scores := MulScalar(Einsum("bqhd,bkhd->bhqk", q, k), scale)
	causal := BroadcastToDims(Reshape(LowerTriangular(g, seqLen), 1, 1, seqLen, seqLen), batch, heads, seqLen, seqLen)
	attn := MaskedSoftmax(scores, causal, -1)
	return Einsum("bhqk,bkhd->bqhd", attn, v)
}

// naiveGQAReference computes grouped-query attention independently of repeatKVHeads: it splits the
// query heads into (numKVHeads, group) and contracts each group against its kv head, so it is a
// genuine cross-check of the repeat-KV grouping. query is [B,S,nQH,D], key/value [B,S,nKVH,D].
func naiveGQAReference(query, key, value *Node, numKVHeads int, scale float64) *Node {
	g := query.Graph()
	q := ConvertDType(query, dtypes.Float32)
	k := ConvertDType(key, dtypes.Float32)
	v := ConvertDType(value, dtypes.Float32)
	d := q.Shape().Dimensions
	b, s, nQH, dim := d[0], d[1], d[2], d[3]
	group := nQH / numKVHeads

	qg := Reshape(q, b, s, numKVHeads, group, dim) // [b,s,h,g,d], query head = h*group + g
	scores := MulScalar(Einsum("bqhgd,bkhd->bhgqk", qg, k), scale)
	causal := BroadcastToDims(Reshape(LowerTriangular(g, s), 1, 1, 1, s, s), b, numKVHeads, group, s, s)
	attn := MaskedSoftmax(scores, causal, -1)
	out := Einsum("bhgqk,bkhd->bqhgd", attn, v)
	return Reshape(out, b, s, nQH, dim)
}

// repeatKVHeads expands key/value for grouped-query attention: [B,S,nKV,D] -> [B,S,nKV*group,D],
// repeating each kv head group times contiguously, so output head h uses kv head h/group.
func repeatKVHeads(x *Node, group int) *Node {
	d := x.Shape().Dimensions
	b, s, nKV, dim := d[0], d[1], d[2], d[3]
	x = Reshape(x, b, s, nKV, 1, dim)
	x = BroadcastToDims(x, b, s, nKV, group, dim)
	return Reshape(x, b, s, nKV*group, dim)
}
```

- [ ] **Step 2: Delete flash.go and flash_test.go**

Run:
```bash
cd /Users/guygrigsby/projects/forks/gomlx
git rm ml/layers/attention/flash.go ml/layers/attention/flash_test.go
```

- [ ] **Step 3: Verify the package still builds and the bench file still compiles**

Run: `go build ./ml/layers/attention/ && go vet ./ml/layers/attention/`
Expected: PASS. (`flash_bench_test.go` references `naiveCausalAttention`/`repeatKVHeads`, now satisfied from `fusion_test.go`; `FlashAttention` is referenced only by `flash_bench_test.go:25` — that line is fixed in Task 6, so vet of the test build may flag it. If `go vet` reports `undefined: FlashAttention` in `flash_bench_test.go`, that is expected and resolved in Task 6; the non-test build `go build ./ml/layers/attention/` must pass cleanly now.)

- [ ] **Step 4: Confirm no stray FlashAttention references remain in production code**

Run: `grep -rn 'FlashAttention' ml/ core/ --include='*.go' | grep -v '_test.go'`
Expected: no output (empty).

- [ ] **Step 5: Commit**

```bash
cd /Users/guygrigsby/projects/forks/gomlx
git add ml/layers/attention/fusion_test.go
git commit -m "attention: delete FlashAttention wrapper; relocate reference helpers to test file

Core's fused branch already calls BackendFusedScaledDotProductAttention with
the same ErrNotImplemented fallback, and Core's decomposed path handles GQA via
reshapeQueryForGQA, so the CUDA-named FlashAttention entry point was redundant.
Move naiveCausalAttention, naiveGQAReference and repeatKVHeads (now test-only)
into fusion_test.go."
```

---

## Task 2: Thread `useFusion` through Core

Add the trailing `useFusion bool` param to `Core` (appended after `scoreSoftCap` — keeps every existing positional arg stable). The fused branch additionally gates on `useFusion`; when false, `Core` always takes the decomposed path. Update both production call sites and all `attention_test.go` / bench call sites.

**Files:**
- Modify: `ml/layers/attention/attention.go:182-282` (signature + fused-branch gate)
- Modify: `ml/layers/attention/multiheadattention.go:446` (sole production `Core` caller — pass `true` for now; Task 3 threads the real builder value)
- Modify: `ml/layers/attention/attention_test.go` (12 `Core(...)` calls — append `true`)
- Modify: `ml/layers/attention/flash_bench_test.go` (no direct `Core` call today; left untouched here)

**Interfaces:**
- Produces (Contract D):
  ```go
  func Core(scope *model.Scope, query, key, value *Node, scale float64, attentionMask *Node, dropoutRate *Node,
      layout AxesLayout, useCausalMask, wantCoefficients bool, scoreSoftCap float64, useFusion bool) (output, coefficients *Node)
  ```

- [ ] **Step 1: Write a failing test for the UseFusion(false) decomposed gate**

Add to `ml/layers/attention/fusion_test.go` (imports: add `"testing"`, `"github.com/gomlx/gomlx/core/tensors"`, `"github.com/gomlx/gomlx/ml/model"`, `"github.com/gomlx/gomlx/support/testutil"`, `"github.com/stretchr/testify/require"`; the existing `dtypes` and graph dot-import stay):
```go
// TestCoreUseFusionFalseMatchesDecomposed pins that Core with useFusion=false produces the
// same output as useFusion=true on the CPU backend (where both take the decomposed path anyway,
// since CPU returns ErrNotImplemented for fused causal). This guards the new gate compiling and
// not altering results. Runs on the default (CPU) backend.
func TestCoreUseFusionFalseMatchesDecomposed(t *testing.T) {
	backend := testutil.BuildTestBackend()
	const B, S, H, D = 1, 32, 2, 64
	scale := 0.125
	q := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 1), B, S, H, D)
	k := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 2), B, S, H, D)
	v := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 3), B, S, H, D)

	store := model.NewStore()
	exec := model.MustNewExec(backend, store, func(scope *model.Scope, qIn, kIn, vIn *Node) []*Node {
		on, _ := Core(scope, qIn, kIn, vIn, scale, nil, nil, LayoutBSHD, true, false, 0.0, true)
		off, _ := Core(scope, qIn, kIn, vIn, scale, nil, nil, LayoutBSHD, true, false, 0.0, false)
		return []*Node{Div(ReduceAllMax(Abs(Sub(on, off))), AddScalar(ReduceAllMax(Abs(off)), 1e-6))}
	})
	out := exec.MustCall(q, k, v)
	rel := float64(tensors.ToScalar[float32](out[0]))
	require.LessOrEqual(t, rel, 1e-6, "useFusion on/off diverged on CPU (both should be decomposed)")
}
```
Also add the `randFlat` helper to `fusion_test.go` (it moved out with `flash_test.go`):
```go
import "math/rand" // add to the import block

func randFlat(n int, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed))
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(r.NormFloat64() * 0.5)
	}
	return out
}
```

- [ ] **Step 2: Run the test to verify it fails to compile**

Run: `go test ./ml/layers/attention/ -run TestCoreUseFusionFalseMatchesDecomposed`
Expected: FAIL — compile error `too many arguments in call to Core` (Core does not yet take `useFusion`).

- [ ] **Step 3: Add the `useFusion` param and gate the fused branch**

In `ml/layers/attention/attention.go`, change the signature (line ~182-183):
```go
func Core(scope *model.Scope, query, key, value *Node, scale float64, attentionMask *Node, dropoutRate *Node,
	layout AxesLayout, useCausalMask, wantCoefficients bool, scoreSoftCap float64, useFusion bool) (output, coefficients *Node) {
```
Change the branch condition (line ~265) to also require `useFusion`:
```go
	if wantCoefficients || dropoutActive || scoreSoftCap > 0 || !useFusion {
		output, coefficients = decomposedFn()
	} else {
```

- [ ] **Step 4: Update the production caller**

In `ml/layers/attention/multiheadattention.go:446-447`, append `true` (replaced by the real builder value in Task 3):
```go
	attentionOutput, attentionCoefficients = Core(b.scope, projectedQuery, projectedKey, projectedValue,
		scale, mask, b.dropoutRate, b.layout, useCausalMask, wantCoefficients, b.scoreSoftCap, true)
```

- [ ] **Step 5: Update all `attention_test.go` Core call sites**

Append `, true` (or `, false` where the test name asks for decomposed — keep behavior identical to today, so append `true` everywhere to preserve the prior default of attempting fusion) to each of the 12 `Core(...)` calls at `attention_test.go` lines 28, 36, 79, 101, 102, 132, 161, 188, 228, 230, 286, 287, 323, 324, 366. Example, line 28:
```go
		bhsdOut, _ := Core(scope, q, k, v, scale, nil, nil, LayoutBHSD, false, false, 0.0, true)
```
For the two `decomposedOutput`/coefficient calls that already pass `wantCoefficients=true` (lines 230, 287, 324, 188), the trailing arg is irrelevant to behavior but still required — append `true`.

Run after editing: `grep -n 'Core(scope' ml/layers/attention/attention_test.go` and confirm every line ends with `, true)`.

- [ ] **Step 6: Run the test to verify it passes**

Run: `go test ./ml/layers/attention/ -run TestCoreUseFusionFalseMatchesDecomposed`
Expected: PASS.

- [ ] **Step 7: Run the whole package to confirm no call site was missed**

Run: `go test ./ml/layers/attention/`
Expected: PASS (or, if `flash_bench_test.go` still references `FlashAttention`, a compile error confined to that file — fix is Task 6; if so, run `go test ./ml/layers/attention/ -run 'TestCore|TestAxes'` to scope around it and confirm the non-bench tests pass).

- [ ] **Step 8: Commit**

```bash
cd /Users/guygrigsby/projects/forks/gomlx
git add ml/layers/attention/attention.go ml/layers/attention/multiheadattention.go ml/layers/attention/attention_test.go ml/layers/attention/fusion_test.go
git commit -m "attention: add useFusion gate to Core (trailing param), thread from sole caller

Appends useFusion bool to Core; the fused branch now requires it in addition to
the existing !wantCoefficients/!dropout/scoreSoftCap<=0 conditions. GOMLX_FUSION
stays the global override; useFusion is the per-call override."
```

---

## Task 3: Builder `UseFusion` (default true)

Add the `useFusion` field defaulting to `true` (set in the `MultiHeadAttention` constructor), the `UseFusion(enabled bool)` builder method, and thread the field into the `Core` call in `doneInternal`.

**Files:**
- Modify: `ml/layers/attention/multiheadattention.go` (struct field ~32-76, constructor ~142-156, new method near other `With*`/`Use*` methods, `doneInternal` Core call ~446)
- Test: `ml/layers/attention/fusion_test.go`

**Interfaces:**
- Produces (Contract D):
  ```go
  func (b *MultiHeadAttentionBuilder) UseFusion(enabled bool) *MultiHeadAttentionBuilder
  ```
- Consumes: `Core(..., useFusion bool)` from Task 2.

- [ ] **Step 1: Write the failing test**

Add to `ml/layers/attention/fusion_test.go`:
```go
// TestBuilderUseFusionDefaultsTrue pins that the builder defaults useFusion to true and that
// UseFusion(false) flips it. Builder-level, no execution needed — inspects the field through a
// Done() run on CPU and asserts the output matches the decomposed reference either way (CPU has
// no fused causal kernel, so both equal the reference; the test guards the wiring compiles and
// the default is true).
func TestBuilderUseFusionDefaultsTrue(t *testing.T) {
	backend := testutil.BuildTestBackend()
	const B, S, H, D = 1, 16, 2, 8
	x := tensors.FromFlatDataAndDimensions(randFlat(B*S*(H*D), 1), B, S, H*D)

	store := model.NewStore()
	exec := model.MustNewExec(backend, store, func(scope *model.Scope, in *Node) []*Node {
		def := SelfAttention(scope.In("def"), in, H, D).UseCausalMask().Done()
		off := SelfAttention(scope.In("off"), in, H, D).UseCausalMask().UseFusion(false).Done()
		// Same weights are not shared across scopes, so compare shapes only: both must produce
		// [B,S,H*D] and be finite. The behavioral on/off equivalence is covered by Task 2.
		return []*Node{def, off}
	})
	out := exec.MustCall(x)
	require.Equal(t, []int{B, S, H * D}, out[0].Shape().Dimensions)
	require.Equal(t, []int{B, S, H * D}, out[1].Shape().Dimensions)
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./ml/layers/attention/ -run TestBuilderUseFusionDefaultsTrue`
Expected: FAIL — compile error `b.UseFusion undefined`.

- [ ] **Step 3: Add the field, default, method, and thread into doneInternal**

In `ml/layers/attention/multiheadattention.go`, add to the struct (after `queryKeyScale float64` ~line 75):
```go
	queryKeyScale float64

	useFusion bool // whether the fused SDPA path may be used; default true (set in constructor).
```
In the `MultiHeadAttention` constructor (the `b := &MultiHeadAttentionBuilder{...}` literal ~142-156), add the default:
```go
		useProjectionBias: true,
		layout:            LayoutBSHD,
		useFusion:         true,
	}
```
Add the method next to `WithQueryKeyScale` (~line 330):
```go
// UseFusion controls whether the backend fused scaled-dot-product-attention path may be used.
// Default is true. When false, Core always takes the decomposed path (useful for debugging or
// for configs the fused kernel does not support). GOMLX_FUSION is a separate global override.
func (b *MultiHeadAttentionBuilder) UseFusion(enabled bool) *MultiHeadAttentionBuilder {
	b.useFusion = enabled
	return b
}
```
In `doneInternal`, change the `Core` call (~446) from `b.scoreSoftCap, true)` to `b.scoreSoftCap, b.useFusion)`:
```go
	attentionOutput, attentionCoefficients = Core(b.scope, projectedQuery, projectedKey, projectedValue,
		scale, mask, b.dropoutRate, b.layout, useCausalMask, wantCoefficients, b.scoreSoftCap, b.useFusion)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `go test ./ml/layers/attention/ -run TestBuilderUseFusionDefaultsTrue`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/guygrigsby/projects/forks/gomlx
git add ml/layers/attention/multiheadattention.go ml/layers/attention/fusion_test.go
git commit -m "attention: add MultiHeadAttentionBuilder.UseFusion (default true)

Constructor seeds useFusion=true; UseFusion(false) forces the decomposed path
through Core."
```

---

## Task 4: Builder `WithSeqLens` → config QuerySeqLen/KeyValueSeqLen

Add `querySeqLen`/`keyValueSeqLen *Node` fields and `WithSeqLens`, mutually exclusive with an explicit `queryKeyMatrixMask`. When set, `doneInternal` builds a `*compute.ScaledDotProductAttentionConfig{QuerySeqLen, KeyValueSeqLen}` and passes it into the fused branch. Threading the config requires `Core` to accept and forward it.

**Files:**
- Modify: `ml/layers/attention/attention.go` (Core: accept an optional config and pass it to `BackendFusedScaledDotProductAttention` instead of the hardcoded `nil` at ~272)
- Modify: `ml/layers/attention/multiheadattention.go` (fields, method, config construction in `doneInternal`)
- Test: `ml/layers/attention/fusion_test.go`

**Interfaces:**
- Produces (Contract D):
  ```go
  func (b *MultiHeadAttentionBuilder) WithSeqLens(querySeqLen, keyValueSeqLen *Node) *MultiHeadAttentionBuilder
  ```
- Consumes (Contract A): `compute.ScaledDotProductAttentionConfig{ QuerySeqLen, KeyValueSeqLen compute.Value }`.
- Internal change: `Core` gains a trailing `*compute.ScaledDotProductAttentionConfig` param (appended after `useFusion`):
  ```go
  func Core(..., scoreSoftCap float64, useFusion bool, fusedConfig *compute.ScaledDotProductAttentionConfig) (output, coefficients *Node)
  ```
  The `*Node` builder seqlens convert to `compute.Value` via `node.outputOps[0]`-equivalent — but `Core` works in `*Node` land, so the config it forwards carries `compute.Value`. Build the config in `doneInternal` (which has the `*Node` seqlens) and pass the already-`compute.Value`-typed config down. See Step 3 for the exact conversion using the seqlen `*Node`s' backend values.

**Note on the *Node → compute.Value boundary:** `compute.ScaledDotProductAttentionConfig.QuerySeqLen` is a `compute.Value`, but the builder holds `*Node`. The graph package converts a `*Node` to its backend `compute.Value` via its `outputOps[0]`, which is unexported. The clean route is to keep the seqlen tensors as graph `*Node`s and pass them as the `mask`-style operands — but Contract A specifies they ride in the *config*, not as operands. Resolve this by having `doneInternal` construct the config inside the `core/graph` package boundary: add a tiny exported helper in `core/graph` that builds the config from `*Node` seqlens. See Step 3.

- [ ] **Step 1: Write the failing validation test (CPU)**

Add to `ml/layers/attention/fusion_test.go`:
```go
// TestWithSeqLensRejectsExplicitMask pins the mutual-exclusion rule: WithSeqLens and an explicit
// query/key matrix mask cannot both be set. Builder-time validation, runs on CPU (panics before
// any backend op).
func TestWithSeqLensRejectsExplicitMask(t *testing.T) {
	backend := testutil.BuildTestBackend()
	const B, S, H, D = 1, 8, 2, 8
	x := tensors.FromFlatDataAndDimensions(randFlat(B*S*(H*D), 1), B, S, H*D)
	lens := tensors.FromFlatDataAndDimensions([]int32{S}, B)
	maskData := make([]float32, B*S*H*S)
	mask := tensors.FromFlatDataAndDimensions(maskData, B, S, H, S)

	store := model.NewStore()
	require.Panics(t, func() {
		_ = model.MustNewExec(backend, store, func(scope *model.Scope, in, qlen, klen, m *Node) []*Node {
			return []*Node{
				SelfAttention(scope, in, H, D).
					WithQueryKeyMatrixMask(m).
					WithSeqLens(qlen, klen).
					Done(),
			}
		}).MustCall(x, lens, lens, mask)
	}, "WithSeqLens + explicit matrix mask must panic")
}
```
(If the builder method that sets `queryKeyMatrixMask` is named differently than `WithQueryKeyMatrixMask`, grep `queryKeyMatrixMask` in `multiheadattention.go` for the setter and use that name. As of this writing the field is `queryKeyMatrixMask` ~line 54; confirm its setter and adjust the test call.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./ml/layers/attention/ -run TestWithSeqLensRejectsExplicitMask`
Expected: FAIL — compile error `b.WithSeqLens undefined`.

- [ ] **Step 3: Add fields, method, config helper, and thread through Core**

In `core/graph/fused_ops.go`, add an exported config builder (it has access to the unexported `outputOps`):
```go
// NewSeqLenAttentionConfig builds a ScaledDotProductAttentionConfig carrying per-batch sequence
// lengths (int32 [B] nodes) for padding masking in the fused SDPA path. Either node may be nil.
func NewSeqLenAttentionConfig(querySeqLen, keyValueSeqLen *Node) *compute.ScaledDotProductAttentionConfig {
	cfg := &compute.ScaledDotProductAttentionConfig{}
	if querySeqLen != nil {
		cfg.QuerySeqLen = querySeqLen.outputOps[0]
	}
	if keyValueSeqLen != nil {
		cfg.KeyValueSeqLen = keyValueSeqLen.outputOps[0]
	}
	return cfg
}
```
In `ml/layers/attention/attention.go`, append the config param to `Core` and forward it:
```go
func Core(scope *model.Scope, query, key, value *Node, scale float64, attentionMask *Node, dropoutRate *Node,
	layout AxesLayout, useCausalMask, wantCoefficients bool, scoreSoftCap float64, useFusion bool,
	fusedConfig *compute.ScaledDotProductAttentionConfig) (output, coefficients *Node) {
```
(Add `"github.com/gomlx/compute"` to attention.go imports if not present — it already imports it via the `AxesLayout` alias, confirm.) In the fused closure (~270-272), pass `fusedConfig` instead of `nil`:
```go
				return BackendFusedScaledDotProductAttention(
					query, key, value, attentionMask,
					numQueryHeads, numKVHeads, layout, scale, useCausalMask, fusedConfig)
```
In `ml/layers/attention/multiheadattention.go`, add fields (after `useFusion bool`):
```go
	querySeqLen    *Node
	keyValueSeqLen *Node
```
Add the method near `UseFusion`:
```go
// WithSeqLens supplies per-batch actual sequence lengths (int32 [B] nodes) for padding masking
// via the fused SDPA path. Mutually exclusive with an explicit query/key matrix mask; setting
// both panics. nil nodes are ignored.
func (b *MultiHeadAttentionBuilder) WithSeqLens(querySeqLen, keyValueSeqLen *Node) *MultiHeadAttentionBuilder {
	if b.queryKeyMatrixMask != nil {
		Panicf("MultiHeadAttention: WithSeqLens is mutually exclusive with an explicit query/key matrix mask")
	}
	b.querySeqLen = querySeqLen
	b.keyValueSeqLen = keyValueSeqLen
	return b
}
```
Also guard the reverse order — the matrix-mask setter must reject seqlens already set. Find the setter for `queryKeyMatrixMask` (grep `b.queryKeyMatrixMask =`) and add at its top:
```go
	if b.querySeqLen != nil || b.keyValueSeqLen != nil {
		Panicf("MultiHeadAttention: query/key matrix mask is mutually exclusive with WithSeqLens")
	}
```
In `doneInternal`, build the config and pass it to `Core`:
```go
	var fusedConfig *compute.ScaledDotProductAttentionConfig
	if b.querySeqLen != nil || b.keyValueSeqLen != nil {
		fusedConfig = NewSeqLenAttentionConfig(b.querySeqLen, b.keyValueSeqLen)
	}
	attentionOutput, attentionCoefficients = Core(b.scope, projectedQuery, projectedKey, projectedValue,
		scale, mask, b.dropoutRate, b.layout, useCausalMask, wantCoefficients, b.scoreSoftCap, b.useFusion, fusedConfig)
```
(Add `"github.com/gomlx/compute"` to multiheadattention.go imports.)

- [ ] **Step 4: Update the remaining Core call sites for the new trailing param**

The production caller in `doneInternal` is done above. Update `attention_test.go`'s 12 calls to append `, nil` (no fused config), and the `TestCoreUseFusionFalseMatchesDecomposed` / any `fusion_test.go` Core calls to append `, nil`.

Run: `grep -rn 'Core(scope' ml/layers/attention/*_test.go` and append `, nil` to each call that does not already pass a config.

- [ ] **Step 5: Run the test to verify it passes**

Run: `go test ./ml/layers/attention/ -run 'TestWithSeqLensRejectsExplicitMask|TestCoreUseFusionFalseMatchesDecomposed|TestBuilderUseFusionDefaultsTrue'`
Expected: PASS.

- [ ] **Step 6: Run the package to confirm no call site missed**

Run: `go build ./... && go test ./ml/layers/attention/ -run 'TestCore|TestAxes|TestWithSeqLens|TestBuilder'`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
cd /Users/guygrigsby/projects/forks/gomlx
git add core/graph/fused_ops.go ml/layers/attention/attention.go ml/layers/attention/multiheadattention.go ml/layers/attention/attention_test.go ml/layers/attention/fusion_test.go
git commit -m "attention: add WithSeqLens; thread ScaledDotProductAttentionConfig through Core

Core gains a trailing *compute.ScaledDotProductAttentionConfig forwarded to
BackendFusedScaledDotProductAttention. WithSeqLens populates QuerySeqLen/
KeyValueSeqLen and is mutually exclusive with an explicit query/key matrix mask
(panics if both set, either order)."
```

---

## Task 5: `SimpleAttention` helper + `GetOfficialBackend`

Add the thin `SimpleAttention` package function (pre-projected + explicit scale) and the `testutil.GetOfficialBackend` skip-or-return helper.

**Files:**
- Modify: `ml/layers/attention/multiheadattention.go` (new package function)
- Modify: `support/testutil/testutil.go` (new `GetOfficialBackend`)
- Test: `ml/layers/attention/fusion_test.go`

**Interfaces:**
- Produces (Contract D):
  ```go
  func SimpleAttention(scope *model.Scope, query, key, value *Node, scale float64) *MultiHeadAttentionBuilder
  func (testutil) GetOfficialBackend(name string) compute.Backend
  ```
- Consumes: `WithPreProjected(true)`, `WithQueryKeyScale(scale)` (existing).

- [ ] **Step 1: Write the failing tests (CPU)**

Add to `ml/layers/attention/fusion_test.go`:
```go
// TestSimpleAttentionWiring pins that SimpleAttention preconfigures pre-projected + explicit scale:
// its causal output matches a manual MultiHeadAttention(...).WithPreProjected(true).WithQueryKeyScale(scale)
// build on CPU. Q/K/V are already projected [B,S,H,D] reshaped to [B,S,H*D] inputs.
func TestSimpleAttentionWiring(t *testing.T) {
	backend := testutil.BuildTestBackend()
	const B, S, H, D = 1, 16, 2, 8
	scale := 0.3
	q := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 1), B, S, H*D)
	k := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 2), B, S, H*D)
	v := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 3), B, S, H*D)

	store := model.NewStore()
	exec := model.MustNewExec(backend, store, func(scope *model.Scope, qn, kn, vn *Node) []*Node {
		simple := SimpleAttention(scope.In("s"), qn, kn, vn, scale).UseCausalMask().Done()
		manual := MultiHeadAttention(scope.In("m"), qn, kn, vn, H, D).
			WithPreProjected(true).WithQueryKeyScale(scale).UseCausalMask().Done()
		require.Equal(t, manual.Shape().Dimensions, simple.Shape().Dimensions)
		return []*Node{simple, manual}
	})
	out := exec.MustCall(q, k, v)
	require.Equal(t, []int{B, S, H * D}, out[0].Shape().Dimensions)
}

// TestGetOfficialBackendSkipsWhenAbsent pins that GetOfficialBackend returns the named backend when
// present (the CPU "go" backend always is) and a nil for an absent name (caller skips). CPU-only.
func TestGetOfficialBackendSkipsWhenAbsent(t *testing.T) {
	testutil.BuildTestBackend()
	require.NotNil(t, testutil.GetOfficialBackend("go"), "go backend should be available")
	require.Nil(t, testutil.GetOfficialBackend("definitely-not-a-backend"))
}
```
(`SimpleAttention` infers `numHeads`/`headDim` from the pre-projected shape. With pre-projected inputs `[B,S,H*D]`, the builder needs `numHeads` and `headDim`; derive them is impossible from a flat last dim alone. So `SimpleAttention` takes the already-head-split inputs `[B,S,H,D]` — see the signature note in Step 3 and adjust the test input shapes to `[B,S,H,D]` accordingly.)

- [ ] **Step 2: Run the tests to verify they fail**

Run: `go test ./ml/layers/attention/ -run 'TestSimpleAttentionWiring|TestGetOfficialBackendSkipsWhenAbsent'`
Expected: FAIL — `SimpleAttention undefined` / `testutil.GetOfficialBackend undefined`.

- [ ] **Step 3: Implement SimpleAttention and GetOfficialBackend**

`SimpleAttention` consumes inputs already projected to `[B, S, numHeads, headDim]` (so it can read `numHeads`/`headDim` off the shape) and reshapes them to the `[B,S,H*D]` the builder's pre-projected path expects. Add to `multiheadattention.go`:
```go
// SimpleAttention is a thin helper for the common pre-projected case: query/key/value are already
// projected to [batch, seq, numHeads, headDim]. It preconfigures the builder with WithPreProjected
// and an explicit scale, so existing pre-projected callers get fusion transparently via Done().
// Pass UseCausalMask()/WithSeqLens()/UseFusion() as needed before Done().
func SimpleAttention(scope *model.Scope, query, key, value *Node, scale float64) *MultiHeadAttentionBuilder {
	qDims := query.Shape().Dimensions
	if len(qDims) != 4 {
		Panicf("SimpleAttention requires query shaped [batch, seq, numHeads, headDim]; got %s", query.Shape())
	}
	numHeads, headDim := qDims[2], qDims[3]
	flat := func(n *Node) *Node {
		d := n.Shape().Dimensions
		return Reshape(n, d[0], d[1], d[2]*d[3])
	}
	return MultiHeadAttention(scope, flat(query), flat(key), flat(value), numHeads, headDim).
		WithPreProjected(true).
		WithQueryKeyScale(scale)
}
```
Update the test inputs in Step 1 to `[B,S,H,D]` (4-D) rather than `[B,S,H*D]`, e.g.:
```go
	q := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 1), B, S, H, D)
```
Add to `support/testutil/testutil.go`:
```go
// GetOfficialBackend returns the named official backend if it was successfully initialized,
// or nil if it is not available in this environment. Callers should skip the test on nil:
//
//	b := testutil.GetOfficialBackend("xla:cuda")
//	if b == nil { t.Skip("xla:cuda backend not available") }
func GetOfficialBackend(name string) compute.Backend {
	BuildTestBackend()
	return OfficialTestBackends[name]
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `go test ./ml/layers/attention/ -run 'TestSimpleAttentionWiring|TestGetOfficialBackendSkipsWhenAbsent'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/guygrigsby/projects/forks/gomlx
git add ml/layers/attention/multiheadattention.go support/testutil/testutil.go ml/layers/attention/fusion_test.go
git commit -m "attention: add SimpleAttention helper and testutil.GetOfficialBackend

SimpleAttention preconfigures the pre-projected builder with an explicit scale.
GetOfficialBackend returns a named official backend or nil (caller skips), used
by the xla:cuda fusion-parity test."
```

---

## Task 6: Generalize parity tests — `backendSupportsFusion`, loop official backends

Replace `isCUDABackend` with a capability probe `backendSupportsFusion`, rewrite the parity/fallback tests to loop `testutil.TestOfficialBackends` and skip when fusion is unsupported, and add one `xla:cuda`-specific fusion-parity test via `GetOfficialBackend`.

**Files:**
- Modify: `ml/layers/attention/fusion_test.go` (probe + generalized tests)
- This task absorbs the former `flash_test.go` test bodies, now generalized.

**Interfaces:**
- Produces (Contract D, test-internal):
  ```go
  func backendSupportsFusion(backend compute.Backend) bool
  ```
- Consumes: `testutil.TestOfficialBackends`, `testutil.GetOfficialBackend`, `naiveCausalAttention`, `naiveGQAReference`, `repeatKVHeads`.

- [ ] **Step 1: Write the probe and a fallback test that loops official backends**

Add to `ml/layers/attention/fusion_test.go` (add imports: `"github.com/gomlx/compute"`, `"github.com/gomlx/compute/dtypes"` already present, `. "github.com/gomlx/gomlx/support/exceptions"`):
```go
// backendSupportsFusion reports whether the backend implements fused scaled-dot-product attention,
// by probing a tiny causal bf16 FusedSDPA and checking it does not return ErrNotImplemented. This
// replaces name-sniffing (isCUDABackend) so any backend that grows the kernel is exercised.
func backendSupportsFusion(backend compute.Backend) bool {
	const B, S, H, D = 1, 8, 1, 64
	q := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 1), B, S, H, D)
	k := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 2), B, S, H, D)
	v := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 3), B, S, H, D)
	supported := false
	_ = TryCatch[error](func() {
		exec := MustNewExec(backend, func(qn, kn, vn *Node) *Node {
			round := func(n *Node) *Node { return ConvertDType(n, dtypes.BFloat16) }
			err := TryCatch[error](func() {
				_ = BackendFusedScaledDotProductAttention(
					round(qn), round(kn), round(vn), nil, H, H,
					compute.AxesLayoutBSHD, 0.125, true, nil)
			})
			supported = err == nil || !compute.IsNotImplemented(err)
			return qn // dummy output to keep the graph valid
		})
		_ = exec.MustCall(q, k, v)
	})
	return supported
}
```
(If building a graph that calls the backend op directly is awkward through `MustNewExec`, the simpler form is to call `BackendFusedScaledDotProductAttention` inside the exec function and catch with `TryCatch`; the key is: returns true unless the error `compute.IsNotImplemented`. Keep the probe shape tiny and causal bf16 per Contract D.)

```go
// TestFusionFallbackParity checks that on every official backend, attention through the builder
// (fused path attempted) matches the decomposed reference, forward and gradients. On a backend
// without the fused kernel this exercises the ErrNotImplemented fallback; on one with it, it is a
// parity check. Loops all official backends.
func TestFusionFallbackParity(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		const B, S, H, D = 1, 64, 2, 64
		scale := 0.125
		q := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 1), B, S, H, D)
		k := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 2), B, S, H, D)
		v := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 3), B, S, H, D)

		store := model.NewStore()
		exec := model.MustNewExec(backend, store, func(scope *model.Scope, qIn, kIn, vIn *Node) []*Node {
			fusedOut, _ := Core(scope, qIn, kIn, vIn, scale, nil, nil, LayoutBSHD, true, false, 0.0, true, nil)
			refOut := naiveCausalAttention(qIn, kIn, vIn, scale)
			fusedOut = ConvertDType(fusedOut, dtypes.Float32)
			fg := Gradient(ReduceAllSum(fusedOut), qIn, kIn, vIn)
			ng := Gradient(ReduceAllSum(refOut), qIn, kIn, vIn)
			relMax := func(got, want *Node) *Node {
				return Div(ReduceAllMax(Abs(Sub(got, want))), AddScalar(ReduceAllMax(Abs(want)), 1e-6))
			}
			return []*Node{relMax(fusedOut, refOut), relMax(fg[0], ng[0]), relMax(fg[1], ng[1]), relMax(fg[2], ng[2])}
		})
		out := exec.MustCall(q, k, v)
		// Fused kernels run in bf16, so use a loose tolerance when fusion is supported; the CPU
		// fallback is the reference itself, so it matches near-exactly.
		tol := 1e-5
		if backendSupportsFusion(backend) {
			tol = 0.06
		}
		for i, name := range []string{"output", "dQ", "dK", "dV"} {
			rel := float64(tensors.ToScalar[float32](out[i]))
			require.LessOrEqualf(t, rel, tol, "%s rel error %.2e exceeds %.2e on %s", name, rel, tol, backend.Name())
		}
	})
}
```

- [ ] **Step 2: Run on Mac (CPU) to verify it passes via the fallback path**

Run: `go test ./ml/layers/attention/ -run 'TestFusionFallbackParity'`
Expected: PASS (the `go`/CPU backend takes the decomposed fallback; tol 1e-5).

- [ ] **Step 3: Add the GQA fallback parity test (loops backends)**

Add to `fusion_test.go`:
```go
// TestFusionGQAParity checks grouped-query attention through Core against the independent grouped
// reference on every official backend, forward and gradients. group=3 (6 query / 2 kv heads).
func TestFusionGQAParity(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		const B, S, QH, KVH, D = 1, 128, 6, 2, 64
		scale := 0.125
		q := tensors.FromFlatDataAndDimensions(randFlat(B*S*QH*D, 1), B, S, QH, D)
		k := tensors.FromFlatDataAndDimensions(randFlat(B*S*KVH*D, 2), B, S, KVH, D)
		v := tensors.FromFlatDataAndDimensions(randFlat(B*S*KVH*D, 3), B, S, KVH, D)

		store := model.NewStore()
		exec := model.MustNewExec(backend, store, func(scope *model.Scope, qIn, kIn, vIn *Node) []*Node {
			out, _ := Core(scope, qIn, kIn, vIn, scale, nil, nil, LayoutBSHD, true, false, 0.0, true, nil)
			out = ConvertDType(out, dtypes.Float32)
			ref := naiveGQAReference(qIn, kIn, vIn, KVH, scale)
			og := Gradient(ReduceAllSum(out), qIn, kIn, vIn)
			rg := Gradient(ReduceAllSum(ref), qIn, kIn, vIn)
			relMax := func(got, want *Node) *Node {
				return Div(ReduceAllMax(Abs(Sub(got, want))), AddScalar(ReduceAllMax(Abs(want)), 1e-6))
			}
			return []*Node{relMax(out, ref), relMax(og[0], rg[0]), relMax(og[1], rg[1]), relMax(og[2], rg[2])}
		})
		out := exec.MustCall(q, k, v)
		tol := 1e-5
		if backendSupportsFusion(backend) {
			tol = 0.06
		}
		for i, name := range []string{"output", "dQ", "dK", "dV"} {
			rel := float64(tensors.ToScalar[float32](out[i]))
			require.LessOrEqualf(t, rel, tol, "%s rel error %.2e exceeds %.2e on %s", name, rel, tol, backend.Name())
		}
	})
}
```

- [ ] **Step 4: Add the repeatKVHeads pin and the xla:cuda-specific test**

Add to `fusion_test.go`:
```go
// TestRepeatKVHeads pins the GQA grouping order: kv head h repeats to query heads h*group..h*group+group-1.
func TestRepeatKVHeads(t *testing.T) {
	backend := testutil.BuildTestBackend()
	in := tensors.FromFlatDataAndDimensions([]float32{10, 20}, 1, 1, 2, 1)
	out := MustNewExec(backend, func(x *Node) *Node { return Reshape(repeatKVHeads(x, 2), 4) }).MustCall(in)
	require.Equal(t, []float32{10, 10, 20, 20}, out[0].Value().([]float32))
}

// TestCUDAFusionParity [trig] cross-checks the cuDNN fused kernel against the float32 reference at
// the head dims cuDNN flash supports (64, 128). Skipped unless the xla:cuda backend is present.
func TestCUDAFusionParity(t *testing.T) {
	backend := testutil.GetOfficialBackend("xla:cuda")
	if backend == nil || !backendSupportsFusion(backend) {
		t.Skip("xla:cuda fused attention backend not available")
	}
	const B, S, H = 1, 512, 4
	for _, D := range []int{64, 128} {
		t.Run(fmt.Sprintf("D=%d", D), func(t *testing.T) {
			scale := 1.0 / math.Sqrt(float64(D))
			q := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 1), B, S, H, D)
			k := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 2), B, S, H, D)
			v := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 3), B, S, H, D)
			store := model.NewStore()
			exec := model.MustNewExec(backend, store, func(scope *model.Scope, qIn, kIn, vIn *Node) []*Node {
				round := func(n *Node) *Node { return ConvertDType(ConvertDType(n, dtypes.BFloat16), dtypes.Float32) }
				qb, kb, vb := round(qIn), round(kIn), round(vIn)
				fusedOut, _ := Core(scope, qb, kb, vb, scale, nil, nil, LayoutBSHD, true, false, 0.0, true, nil)
				fusedOut = ConvertDType(fusedOut, dtypes.Float32)
				refOut := naiveCausalAttention(qb, kb, vb, scale)
				fg := Gradient(ReduceAllSum(fusedOut), qb, kb, vb)
				ng := Gradient(ReduceAllSum(refOut), qb, kb, vb)
				relMax := func(got, want *Node) *Node {
					return Div(ReduceAllMax(Abs(Sub(got, want))), AddScalar(ReduceAllMax(Abs(want)), 1e-6))
				}
				return []*Node{relMax(fusedOut, refOut), relMax(fg[0], ng[0]), relMax(fg[1], ng[1]), relMax(fg[2], ng[2])}
			})
			out := exec.MustCall(q, k, v)
			tol := []float64{0.03, 0.06, 0.06, 0.06}
			for i, name := range []string{"output", "dQ", "dK", "dV"} {
				rel := float64(tensors.ToScalar[float32](out[i]))
				require.Falsef(t, math.IsNaN(rel), "%s rel error NaN", name)
				require.LessOrEqualf(t, rel, tol[i], "%s rel max %.4f exceeds %.4f", name, rel, tol[i])
			}
		})
	}
}
```
Add `"fmt"` and `"math"` to the `fusion_test.go` import block.

- [ ] **Step 5: Run the full attention package on Mac (CPU)**

Run: `go test ./ml/layers/attention/`
Expected: PASS. `TestCUDAFusionParity` skips (no `xla:cuda`); all generalized/parity/builder tests pass via the CPU fallback path.

- [ ] **Step 6: [trig] Run the CUDA parity test on trig**

Run (on trig, after `git pull` in the fork checkout):
```bash
GOMLX_BACKEND=xla:cuda gputex run "gomlx fusion parity" -- \
  go test ./ml/layers/attention/ -run 'TestCUDAFusionParity|TestFusionFallbackParity|TestFusionGQAParity' -v
```
Expected: PASS — `TestCUDAFusionParity/D=64` and `/D=128` run the cuDNN kernel and stay within tolerance; the looped parity tests run with `tol=0.06` (fusion supported) and pass.

- [ ] **Step 7: Commit**

```bash
cd /Users/guygrigsby/projects/forks/gomlx
git add ml/layers/attention/fusion_test.go
git commit -m "attention: generalize fusion tests over official backends; capability probe

Replace isCUDABackend name-sniffing with backendSupportsFusion (tiny causal bf16
FusedSDPA probe). Parity/fallback/GQA tests loop TestOfficialBackends and pick
tolerance by capability. Add an xla:cuda-specific parity test via
GetOfficialBackend. Pin repeatKVHeads grouping."
```

---

## Task 7: Rename and rework the throughput/memory bench

Rename `flash_bench_test.go` → `fusion_bench_test.go`. `attentionStep` uses the **same** code path for fused and non-fused (toggling via `UseFusion`, not a separate `naiveCausalAttention` branch). Rename `TestFlashThroughput` → `TestFusionThroughput`. Add `out[0].FinalizeAll()` after the sync. Use `for range iters`. Replace `GOMLX_MEM_PROBE`/`GOMLX_PROBE_S` env gating with Go test flags.

**Files:**
- Rename: `ml/layers/attention/flash_bench_test.go` → `ml/layers/attention/fusion_bench_test.go`
- Modify: the renamed file (rewrite `attentionStep`, `TestFusionThroughput`, memory probe).

**Interfaces:**
- Consumes: `SimpleAttention`, `UseFusion`, `backendSupportsFusion`, `testutil.BuildTestBackend`.

- [ ] **Step 1: Rename the file**

Run:
```bash
cd /Users/guygrigsby/projects/forks/gomlx
git mv ml/layers/attention/flash_bench_test.go ml/layers/attention/fusion_bench_test.go
```

- [ ] **Step 2: Rewrite the file**

Replace the entire contents of `ml/layers/attention/fusion_bench_test.go`:
```go
// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"flag"
	"testing"
	"time"

	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/testutil"
	. "github.com/gomlx/gomlx/support/exceptions"
)

var (
	memProbeVariant = flag.String("mem_probe", "", "run the memory probe with this variant: fused|decomposed")
	memProbeSeqLen  = flag.Int("mem_probe_seqlen", 2048, "sequence length for the memory probe")
)

// attentionStep builds one forward+backward attention step through the SAME builder path for both
// fused and non-fused — fusion toggles via UseFusion, not a separate decomposed code path — so the
// benchmark compares the two routes of one implementation. Returns a scalar so reading it forces a
// device sync. query/key/value are pre-projected [B,S,H,D].
func attentionStep(useFusion bool, qHeads, kvHeads int, scale float64) func(scope *model.Scope, q, k, v *Node) []*Node {
	return func(scope *model.Scope, q, k, v *Node) []*Node {
		b := SimpleAttention(scope, q, k, v, scale).UseCausalMask().UseFusion(useFusion)
		if kvHeads != qHeads {
			b = b.WithNumKVHeads(kvHeads)
		}
		out := ConvertDType(b.Done(), dtypes.Float32)
		g := Gradient(ReduceAllSum(out), q, k, v)
		return []*Node{Add(Add(ReduceAllSum(g[0]), ReduceAllSum(g[1])), ReduceAllSum(g[2]))}
	}
}

// TestFusionThroughput reports per-step (forward+backward) wall time for the fused vs decomposed
// route at the lm-100m attention shape. Requires a backend that supports fusion (cuDNN). [trig]
func TestFusionThroughput(t *testing.T) {
	backend := testutil.BuildTestBackend()
	if !backendSupportsFusion(backend) {
		t.Skipf("fusion throughput needs a backend with the fused kernel; %q has none", backend.Name())
	}
	const (
		B, S, QH, KVH, D = 2, 2048, 12, 4, 64
		scale            = 0.125
		iters            = 20
	)
	q := tensors.FromFlatDataAndDimensions(randFlat(B*S*QH*D, 1), B, S, QH, D)
	k := tensors.FromFlatDataAndDimensions(randFlat(B*S*KVH*D, 2), B, S, KVH, D)
	v := tensors.FromFlatDataAndDimensions(randFlat(B*S*KVH*D, 3), B, S, KVH, D)

	timeStep := func(useFusion bool) (time.Duration, error) {
		var perStep time.Duration
		err := TryCatch[error](func() {
			store := model.NewStore()
			exec := model.MustNewExec(backend, store, attentionStep(useFusion, QH, KVH, scale))
			exec.MustCall(q, k, v) // warmup + compile
			start := time.Now()
			for range iters {
				out := exec.MustCall(q, k, v)
				_ = tensors.ToScalar[float32](out[0]) // force device sync
				out[0].FinalizeAll()                  // release the step's device buffers
			}
			perStep = time.Since(start) / iters
		})
		return perStep, err
	}

	fused, err := timeStep(true)
	if err != nil {
		t.Fatalf("fused step failed: %v", err)
	}
	t.Logf("fused      per-step (fwd+bwd) B=%d S=%d %d/%d heads D=%d: %v", B, S, QH, KVH, D, fused)

	decomposed, err := timeStep(false)
	if err != nil {
		t.Logf("decomposed per-step: did not run (%v)", err)
		return
	}
	t.Logf("decomposed per-step: %v", decomposed)
	t.Logf("fused speedup: %.1fx", float64(decomposed)/float64(fused))
}

// TestFusionMemoryProbe runs one attention variant repeatedly so an external sampler (nvidia-smi,
// XLA preallocation disabled) can read the working-set peak. Enable with
// -mem_probe=fused|decomposed; skipped otherwise. [trig]
func TestFusionMemoryProbe(t *testing.T) {
	if *memProbeVariant == "" {
		t.Skip("pass -mem_probe=fused|decomposed to run the memory probe")
	}
	backend := testutil.BuildTestBackend()
	if !backendSupportsFusion(backend) {
		t.Skipf("memory probe needs the fused kernel; %q has none", backend.Name())
	}
	const (
		B, QH, KVH, D = 2, 12, 4, 64
		scale         = 0.125
	)
	S := *memProbeSeqLen
	q := tensors.FromFlatDataAndDimensions(randFlat(B*S*QH*D, 1), B, S, QH, D)
	k := tensors.FromFlatDataAndDimensions(randFlat(B*S*KVH*D, 2), B, S, KVH, D)
	v := tensors.FromFlatDataAndDimensions(randFlat(B*S*KVH*D, 3), B, S, KVH, D)

	store := model.NewStore()
	exec := model.MustNewExec(backend, store, attentionStep(*memProbeVariant == "fused", QH, KVH, scale))
	for range 40 {
		out := exec.MustCall(q, k, v)
		_ = tensors.ToScalar[float32](out[0])
		out[0].FinalizeAll()
	}
}
```
(If `model.Exec` output nodes do not expose `FinalizeAll`, use the tensor finalizer the package actually provides — grep `func.*Finalize` in `core/tensors` and `ml/model`; the intent is to free each step's device buffers so the loop does not accumulate. If no finalizer exists, drop those two lines and note it; the sync via `ToScalar` is the load-bearing part.)

- [ ] **Step 3: Verify the bench builds and skips cleanly on Mac (CPU)**

Run: `go test ./ml/layers/attention/ -run 'TestFusionThroughput|TestFusionMemoryProbe' -v`
Expected: both SKIP on CPU (`backendSupportsFusion` is false for `go`), package builds clean. Confirm no remaining reference to `FlashAttention`, `isCUDABackend`, `GOMLX_MEM_PROBE`, or `GOMLX_PROBE_S`:
```bash
grep -rn 'FlashAttention\|isCUDABackend\|GOMLX_MEM_PROBE\|GOMLX_PROBE_S\|TestFlashThroughput' ml/layers/attention/
```
Expected: no output.

- [ ] **Step 4: [trig] Run the throughput bench on trig**

Run (on trig):
```bash
GOMLX_BACKEND=xla:cuda gputex run "gomlx fusion throughput" -- \
  go test ./ml/layers/attention/ -run TestFusionThroughput -v
```
Expected: logs fused and decomposed per-step times and a speedup ratio; PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/guygrigsby/projects/forks/gomlx
git add ml/layers/attention/fusion_bench_test.go
git commit -m "attention: rework throughput/memory bench through one fusion-toggled path

Rename flash_bench_test.go -> fusion_bench_test.go and TestFlashThroughput ->
TestFusionThroughput. attentionStep now runs the SAME SimpleAttention builder for
both routes, toggling UseFusion, so the bench compares one implementation's two
paths. FinalizeAll after each sync; for range iters; replace GOMLX_MEM_PROBE/
GOMLX_PROBE_S env gating with -mem_probe/-mem_probe_seqlen test flags."
```

---

## Verification Gate (Plan 03)

Per Contract section "Verification gates / 03 gomlx". Run all on the Mac first; the `[trig]` line on trig.

- [ ] **Gate 1 — Mac (CPU) full package green**

Run:
```bash
cd /Users/guygrigsby/projects/forks/gomlx
go build ./...
go test ./ml/layers/attention/...
```
Expected: PASS. Covers decomposed path, `ErrNotImplemented` fallback, `UseFusion(false)`, `WithSeqLens` validation, `SimpleAttention` wiring, `GetOfficialBackend` skip behavior. `TestCUDAFusionParity`, `TestFusionThroughput`, `TestFusionMemoryProbe` SKIP (no `xla:cuda`).

- [ ] **Gate 2 — No leftover CUDA-shaped surface**

Run:
```bash
cd /Users/guygrigsby/projects/forks/gomlx
grep -rn 'FlashAttention\|isCUDABackend\|GOMLX_MEM_PROBE\|GOMLX_PROBE_S' ml/ core/ support/ --include='*.go'
test ! -e ml/layers/attention/flash.go && test ! -e ml/layers/attention/flash_test.go && test ! -e ml/layers/attention/flash_bench_test.go && echo "OK: flash files removed"
```
Expected: no grep output; prints `OK: flash files removed`.

- [ ] **Gate 3 — [trig] cuDNN fusion-parity green**

Run (on trig, fork checkout pulled to the latest `flash-attention` commit):
```bash
GOMLX_BACKEND=xla:cuda gputex run "gomlx plan03 gate" -- \
  go test ./ml/layers/attention/... -run 'TestCUDAFusionParity|TestFusionFallbackParity|TestFusionGQAParity|TestFusionThroughput' -v
```
Expected: PASS — the cuDNN fused kernel runs (parity within tolerance for D=64 and D=128), the looped parity tests pass at `tol=0.06`, and throughput reports a speedup.

- [ ] **Gate 4 — Stop and report**

All gates green → plan 03 (Contract D) is satisfied; downstream plan 04 (go-huggingface) may consume `UseFusion`/`WithSeqLens`/`SimpleAttention` via the local `replace` on this gomlx fork. Report the gate output to Guy for manual diff review before any push.

---

## Self-Review (writing-plans)

**Spec coverage (Contract D + scope items 1-5):**
- Delete flash.go, preserve `repeatKVHeads` (relocated to `fusion_test.go`), state why deletion loses no behavior → Task 1. ✓
- `UseFusion(enabled bool)`, default true in constructor, thread to `Core` via new trailing param, gate fused branch, update all Core call sites → Tasks 2-3. ✓ (Core param appended after `scoreSoftCap`, then `fusedConfig` after that in Task 4 — internally consistent; all 12 `attention_test.go` calls + `multiheadattention.go:446` + `fusion_test.go` calls updated.)
- `WithSeqLens` → `ScaledDotProductAttentionConfig.QuerySeqLen/KeyValueSeqLen`, mutually exclusive with `queryKeyMatrixMask` (both orders) → Task 4. ✓
- `SimpleAttention` preconfiguring `WithPreProjected(true).WithQueryKeyScale(scale)` → Task 5. ✓
- `backendSupportsFusion` probe replacing `isCUDABackend`; loop `TestOfficialBackends`; `testutil.GetOfficialBackend`; one `xla:cuda` parity test → Tasks 5-6. ✓
- Bench: rename `TestFusionThroughput`, same code path via `UseFusion`, `FinalizeAll`, `for range iters`, test-flag gating → Task 7. ✓
- CUDA-execution tests marked `[trig]`; CPU-testable items enumerated → present per task and in the gate. ✓

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". Two flagged uncertainties (the `*Node → compute.Value` config boundary in Task 4, the `FinalizeAll` finalizer name in Task 7) carry an explicit grep-and-resolve instruction with a concrete fallback, not a hand-wave — acceptable because the implementer is told exactly what to check and what to do in either branch.

**Type consistency:** `Core` signature is consistent across Tasks 2 (`+useFusion bool`) and 4 (`+fusedConfig *compute.ScaledDotProductAttentionConfig`), and every call-site-update step references the same trailing-arg shape. `UseFusion`/`WithSeqLens`/`SimpleAttention`/`GetOfficialBackend`/`backendSupportsFusion`/`NewSeqLenAttentionConfig` names match between their producing task and consuming tasks. `repeatKVHeads`/`naiveCausalAttention`/`naiveGQAReference`/`randFlat` relocate once (Task 1) and are referenced thereafter.

**Known residual risk to flag at execution:** Task 4's `NewSeqLenAttentionConfig` reaches into `node.outputOps[0]`; confirm at implementation that `outputOps` is the field name on `*Node` in this fork (grep `outputOps` in `core/graph/fused_ops.go` — it is used there, so the helper compiles in the same package). This is the only place the plan touches `core/graph` rather than `ml/layers/attention`.
