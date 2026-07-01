# Fused Attention — Plan 03: gomlx Layers (WithFusion, SeqLens, generalized tests)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Delete the CUDA-shaped `FlashAttention` wrapper, expose fusion through the `MultiHeadAttention` builder (`WithFusion`, `WithSeqLens`), make the attention tests backend-general instead of `isCUDABackend`-gated, and (Stage 2) add `WithAttentionBias`, per Contract D. No replacement convenience helper is added (Contract D, "No SimpleAttention helper"): the bare primitives `graph.BackendFusedScaledDotProductAttention` + `graph.InternalFusedOpCaller` and the `MultiHeadAttention` builder already cover the need.

**Architecture:** The fused capability already lives in `attention.Core`'s fused branch (`InternalFusedOpCaller(fused, decomposed)`), so the standalone `FlashAttention` function is redundant and CUDA-flavored — delete it with no replacement helper. Add a per-call `WithFusion` gate threaded `builder → doneInternal → Core` (`Core` carries it as a plain `useFusion bool` param), a `WithSeqLens` path that populates `compute.ScaledDotProductAttentionConfig.QuerySeqLen/KeyValueSeqLen`, and rewrite the tests to probe each official backend for fusion support (`backendSupportsFusion`) rather than sniff the backend name. Stage 2 adds the score-bias variant (`WithAttentionBias` → `config.Bias`).

**Amendment A1 (2026-07-01, see contract doc) — gomlx-side changes, part of Stage 1:**
- `BackendFusedScaledDotProductAttention` (`core/graph/fused_ops.go:179`) returns `(output *Node, statesForVJP []*Node)` instead of hiding `softmaxStats` — a `Backend<Op>` escape hatch mirrors the backend op and does not hide outputs.
- Register the VJP in `VJPRegistration` (`core/graph/rev_autodiff.go`), NOT via `customVJP` on the parent node (`fused_ops.go:187-188`). A small wrapper around `backendFusedScaledDotProductAttentionVJP` registers centrally; delete the `customVJP`-on-parent mechanism.
- Inline + unexport `NewSeqLenAttentionConfig` (`fused_ops.go:159`): used once, exists only to reach the unexported `.outputOps[0]` (`*Node`→`compute.Value`); inline at the call site, drop the exported name.
- `TestSeqLenFusedParity_cuda` (`fusion_test.go:535`): loop over `testutil.TestOfficialBackends`, drop the `_cuda` name (non-fusing backends pass trivially; future Go SIMD fused kernel covered for free).

**Tech Stack:** Go 1.26, `github.com/gomlx/gomlx` (fork, branch `flash-attention`), depends on `github.com/gomlx/compute` (fork, branch `flash-customcall`) and `github.com/gomlx/go-xla` (fork, branch `flash-attention`) via local `replace`.

**Contract:** This plan **produces Contract D** and **consumes Contracts A and C** of `docs/plans/2026-06-29-fused-attn-00-contract.md`. Read the contract first — especially the **Staging** section (Stage 1 = strict API refactor; Stage 2 = variants) and the revised **Contract D** (`WithFusion`, the "No SimpleAttention helper" note, and the S2 `WithAttentionBias`). Global constraints (Go 1.26, no push/no PR, CUDA tests `[cuda]` only, fallback-on-`ErrNotImplemented`, voice-rule commits with `attention:` prefix and no Claude attribution) are defined there and apply to every task below; they are not restated per task.

## Staging

Per the contract's Staging section, this plan ships in two stages, each independently green and reviewable. **Do Stage 1 in full — review — then Stage 2.**

- **[S1] (strict API refactor, no variants):** Tasks 0–7 below. Delete `flash.go` (no replacement helper); relocate (do not lose) the test-only decomposed oracles `naiveCausalAttention`/`naiveGQAReference`/`repeatKVHeads`; `WithFusion` builder method + `useFusion bool` through `Core`; `WithSeqLens` threading into `compute.ScaledDotProductAttentionConfig.QuerySeqLen/KeyValueSeqLen`; `testutil.GetOfficialBackend`; generalize tests (`backendSupportsFusion` probe, loop `TestOfficialBackends`, `testutil.GetOfficialBackend`); bench fixes (`FinalizeAll`, `for range`, rename `TestFusionThroughput`, single `WithFusion`-toggled code path, flags not env). In S1 the only fused-config fields that exist are `QuerySeqLen/KeyValueSeqLen` (per compute Stage 1).
- **[S2] (variants — depends on compute Stage 2 fields):** Tasks 8–9 below. Add `WithAttentionBias(bias *Node)` populating `config.Bias` (DISTINCT from the existing `UseProjectionBias`). This depends on the compute Stage 2 field (`Bias`) existing on `ScaledDotProductAttentionConfig`; do **not** start S2 until plan 01 Stage 2 has landed that field through the local `replace`.

S1 tasks (0–7) come first; S2 tasks (8–9) come last.

## Global Constraints

- Repo `/Users/guygrigsby/projects/forks/gomlx`, branch `flash-attention`, module `github.com/gomlx/gomlx`, Go 1.26.
- This plan depends on Contract A landing first: `compute.ScaledDotProductAttentionConfig` must already carry `QuerySeqLen, KeyValueSeqLen Value` (plan 01 Stage 1). Task 0 verifies the local `replace` points at the updated compute fork and the S1 fields exist; if they do not, stop — plan 01 Stage 1 is not done. The S2 field (`Bias`) is verified separately by Task 8 Step 0.
- Commit prefix `attention:` (existing scheme in this package). Terse, verb-first, no em/en dashes, no Claude attribution.
- CUDA execution is `[cuda]` only; the Mac CI has no cuDNN. Mark every step whose `Run:` needs `xla:cuda` with **[cuda]**. Everything else runs on the Mac `go` (CPU) backend.
- Fallback is the contract: an unsupported-but-valid request returns wrapped `compute.ErrNotImplemented` and falls back to the decomposed path. Never panic on unsupported-but-valid.

---

## File Structure

- `ml/layers/attention/flash.go` — **DELETE** [S1]. The `FlashAttention` function is superseded by `Core`'s fused branch; no replacement helper is added (Contract D, "No SimpleAttention helper"). Its decomposed half, `naiveCausalAttention`, has no runtime caller after the deletion (Core has its own inline decomposed path), so it survives only as a test-only decomposed oracle and moves to a `_test.go`, alongside the already-test-only helpers (`naiveGQAReference`, `repeatKVHeads`).
- `ml/layers/attention/attention.go` — `Core` gains a trailing `useFusion bool` param [S1] and a trailing `*compute.ScaledDotProductAttentionConfig` param [S1]; the fused branch additionally gates on `useFusion` and threads the config into `BackendFusedScaledDotProductAttention`.
- `ml/layers/attention/multiheadattention.go` — builder gains `useFusion bool` (default true, set in constructor), `querySeqLen/keyValueSeqLen *Node` fields, methods `WithFusion`, `WithSeqLens` [S1]; later `attentionBias *Node` + `WithAttentionBias` [S2]; `doneInternal` builds the config and passes `useFusion` to `Core`.
- `ml/layers/attention/fusion_test.go` — **NEW** [S1], test-only. Houses the relocated decomposed oracles (`naiveCausalAttention`, `naiveGQAReference`, `repeatKVHeads`), the `backendSupportsFusion` probe, the CPU-testable builder/wiring tests, and the backend-general parity tests (loop `TestOfficialBackends`, skip when unsupported).
- `ml/layers/attention/flash_test.go` — **DELETE** [S1] (its content moves to `fusion_test.go`, generalized).
- `ml/layers/attention/flash_bench_test.go` → rename to `ml/layers/attention/fusion_bench_test.go` [S1]; `attentionStep` toggles fusion via `WithFusion`, not a separate code path; env-var probe gating becomes Go test flags.
- `support/testutil/testutil.go` — add `GetOfficialBackend(name string) compute.Backend` [S1].

**Core call-site blast radius (real count, grepped):** exactly **one production caller** — `multiheadattention.go:446` — plus **15 test calls** in `attention_test.go` (lines 28, 36, 79, 101, 102, 132, 161, 188, 228, 230, 286, 287, 323, 324, 366) and the new `fusion_test.go` calls. Changing `Core`'s signature is small: one production line, fifteen test lines, plus the new tests this plan adds.

---

## Task 0: Verify Contract A (Stage 1) landed (no code) [S1]

**Files:**
- Inspect: `/Users/guygrigsby/projects/forks/gomlx/go.mod` (the `replace github.com/gomlx/compute` directive)
- Inspect: the compute fork `fused_ops.go` reachable through that replace.

**Interfaces:**
- Consumes (Contract A, Stage 1): `compute.ScaledDotProductAttentionConfig` with fields `QuerySeqLen, KeyValueSeqLen compute.Value`.

- [ ] **Step 1: Confirm the local replace and the S1 fields exist**

Run:
```bash
cd /Users/guygrigsby/projects/forks/gomlx
grep -n 'replace.*gomlx/compute' go.mod
go doc github.com/gomlx/compute.ScaledDotProductAttentionConfig
```
Expected: `go.mod` has a `replace github.com/gomlx/compute => /Users/guygrigsby/projects/forks/compute` (or equivalent local path), and `go doc` lists `QuerySeqLen Value` and `KeyValueSeqLen Value` on the struct. (The S2 field `Bias` may or may not be present yet — it is not required for Stage 1; Task 8 verifies it.)

- [ ] **Step 2: Decision gate**

If the replace is missing or the S1 fields are absent, STOP and report: "plan 01 Stage 1 (compute) is not integrated; cannot proceed with plan 03 Stage 1." Otherwise continue to Task 1. (No commit — this is a read-only gate.)

---

## Task 1: Move flash.go helpers to a test file (decomposed oracles), delete the wrapper [S1]

`FlashAttention` is deleted with **no replacement helper** (Contract D, "No SimpleAttention helper"). After the deletion there is **no runtime caller** for `naiveCausalAttention`: `Core`'s fused branch falls back to its own inline decomposed path, and nothing in the package calls `naiveCausalAttention` outside tests. So all three flash helpers — `naiveCausalAttention`, `naiveGQAReference`, `repeatKVHeads` — are **test-only** decomposed oracles and move into `ml/layers/attention/fusion_test.go`. No non-test `fusion.go` is created.

This task: (a) move `naiveCausalAttention`, `naiveGQAReference`, and `repeatKVHeads` into `fusion_test.go`; (b) delete the `FlashAttention` function and `flash.go`/`flash_test.go`.

**Why all three are test-only:**
- `naiveCausalAttention` — **test-only oracle.** The generalized parity tests (`TestFusionFallbackParity`, `TestCUDAFusionParity`, Task 6) use it as the decomposed reference to compare `Core`'s output against. No production code calls it: `Core` has its own inline decomposed path, and there is no `SimpleAttention` to use it as a fallback.
- `naiveGQAReference` — **test-only**, already lives in `flash_test.go`. The GQA parity test (`TestFusionGQAParity`, Task 6) uses it as the independent grouped reference. Move verbatim to `fusion_test.go`.
- `repeatKVHeads` — **test-only.** `Core`'s decomposed path uses `reshapeQueryForGQA`, not `repeatKVHeads`. It survives only to pin the GQA grouping order in `TestRepeatKVHeads` (Task 6). Move it to `fusion_test.go`. **It must be preserved (relocated, not deleted).**

**Why deleting `FlashAttention` loses no behavior:** `FlashAttention(q,k,v,scale)` did three things — validate rank-4/equal-head shapes, repeat KV heads for GQA, then call `BackendFusedScaledDotProductAttention(..., causal=true, options=nil)` with a `naiveCausalAttention` fallback on `compute.IsNotImplemented`. Its consumers are covered without a replacement: `Core`'s fused branch (`attention.go:268-280`) calls the same backend op through `InternalFusedOpCaller` (identical fallback) with GQA via `reshapeQueryForGQA` in the decomposed path; and a consumer that needs the bare pre-projected fused-or-decomposed op (e.g. lmkit-go, Contract F) calls `graph.InternalFusedOpCaller` + `graph.BackendFusedScaledDotProductAttention` directly. The deletion only removes the redundant, CUDA-named entry point.

**Files:**
- Create: `ml/layers/attention/fusion_test.go` (test-only oracles `naiveCausalAttention`, `naiveGQAReference`, `repeatKVHeads`).
- Delete: `ml/layers/attention/flash.go`
- Delete: `ml/layers/attention/flash_test.go`
- Inspect: `ml/layers/attention/flash_bench_test.go` (still references `naiveCausalAttention`/`repeatKVHeads` — keeps compiling because both move to `fusion_test.go`, same package `attention`; `flash_bench_test.go`'s `FlashAttention` reference is fixed in Task 7).

**Interfaces:**
- Produces (test-internal, package `attention`, in `fusion_test.go`):
  - `func naiveCausalAttention(query, key, value *Node, scale float64) *Node`
  - `func naiveGQAReference(query, key, value *Node, numKVHeads int, scale float64) *Node`
  - `func repeatKVHeads(x *Node, group int) *Node`
- Removes: `func FlashAttention(query, key, value *Node, scale float64) *Node` (no production caller; grep confirmed zero usages outside the deleted files).

- [ ] **Step 1: Create `fusion_test.go` with the test-only oracles**

Create `ml/layers/attention/fusion_test.go`:
```go
// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

import (
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
)

// naiveCausalAttention is the decomposed reference oracle: softmax(scale*QK^T + causal)*V in float32.
// query/key/value are [B,S,H,D] with equal heads. Test-only: Core has its own inline decomposed path,
// so this exists only as the reference the parity tests compare Core's output against.
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
// Test-only: production GQA goes through Core's reshapeQueryForGQA; this survives only to pin the
// grouping order in TestRepeatKVHeads.
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
Expected: PASS for `go build` (the non-test build has no `naiveCausalAttention` — it is test-only now, and nothing in the non-test build references it). `flash_bench_test.go` references `naiveCausalAttention`/`repeatKVHeads` (both in `fusion_test.go`, same package) and `FlashAttention` (referenced only by `flash_bench_test.go:25` — fixed in Task 7). If `go vet` reports `undefined: FlashAttention` in `flash_bench_test.go`, that is expected and resolved in Task 7; the non-test build `go build ./ml/layers/attention/` must pass cleanly now.

- [ ] **Step 4: Confirm no stray FlashAttention references remain in production code, and naiveCausalAttention is test-only**

Run:
```bash
grep -rn 'FlashAttention' ml/ core/ --include='*.go' | grep -v '_test.go'
grep -l 'func naiveCausalAttention' ml/layers/attention/*.go
```
Expected: first grep empty; second prints `ml/layers/attention/fusion_test.go` (a `_test.go` file — `naiveCausalAttention` is test-only).

- [ ] **Step 5: Commit**

```bash
cd /Users/guygrigsby/projects/forks/gomlx
git add ml/layers/attention/fusion_test.go
git commit -m "attention: delete FlashAttention wrapper; move decomposed oracles to tests

FlashAttention is gone with no replacement helper: Core's fused branch covers its
behavior through InternalFusedOpCaller, and consumers needing the bare pre-projected
fused-or-decomposed op call graph.InternalFusedOpCaller + BackendFusedScaledDotProductAttention
directly. naiveCausalAttention has no runtime caller after the deletion, so it joins
naiveGQAReference and repeatKVHeads as test-only decomposed oracles in fusion_test.go."
```

---

## Task 2: Thread `useFusion` through Core [S1]

Add the trailing `useFusion bool` param to `Core` (appended after `scoreSoftCap` — keeps every existing positional arg stable). The fused branch additionally gates on `useFusion`; when false, `Core` always takes the decomposed path. Update the single production call site and all `attention_test.go` call sites.

**Files:**
- Modify: `ml/layers/attention/attention.go:182-282` (signature + fused-branch gate)
- Modify: `ml/layers/attention/multiheadattention.go:446` (the **sole** production `Core` caller — pass `true` for now; Task 3 threads the real builder value)
- Modify: `ml/layers/attention/attention_test.go` (15 `Core(scope ...)` calls — append `true`)
- Modify: `ml/layers/attention/flash_bench_test.go` (no direct `Core` call today; left untouched here)

**Interfaces:**
- Produces (Contract D):
  ```go
  func Core(scope *model.Scope, query, key, value *Node, scale float64, attentionMask *Node, dropoutRate *Node,
      layout AxesLayout, useCausalMask, wantCoefficients bool, scoreSoftCap float64, useFusion bool) (output, coefficients *Node)
  ```
  Note: `Core` carries the toggle as a plain `useFusion bool` param. The builder-facing method is `WithFusion` (Task 3) — the param name `useFusion` is internal and stays a bool.

- [ ] **Step 1: Write a failing test for the WithFusion(false) decomposed gate**

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

Append `, true` to each of the **15** `Core(scope ...)` calls at `attention_test.go` lines 28, 36, 79, 101, 102, 132, 161, 188, 228, 230, 286, 287, 323, 324, 366. Keep behavior identical to today, so append `true` everywhere to preserve the prior default of attempting fusion. For the calls that already pass `wantCoefficients=true` (lines 188, 230, 287, 324), the trailing arg is irrelevant to behavior but still required — append `true`. Example, line 28:
```go
		bhsdOut, _ := Core(scope, q, k, v, scale, nil, nil, LayoutBHSD, false, false, 0.0, true)
```

Run after editing: `grep -n 'Core(scope' ml/layers/attention/attention_test.go` and confirm every line ends with `, true)`.

- [ ] **Step 6: Run the test to verify it passes**

Run: `go test ./ml/layers/attention/ -run TestCoreUseFusionFalseMatchesDecomposed`
Expected: PASS.

- [ ] **Step 7: Run the whole package to confirm no call site was missed**

Run: `go test ./ml/layers/attention/`
Expected: PASS (or, if `flash_bench_test.go` still references `FlashAttention`, a compile error confined to that file — fix is Task 7; if so, run `go test ./ml/layers/attention/ -run 'TestCore|TestAxes'` to scope around it and confirm the non-bench tests pass).

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

## Task 3: Builder `WithFusion` (default true) [S1]

Add the `useFusion` field defaulting to `true` (set in the `MultiHeadAttention` constructor), the `WithFusion(enabled bool)` builder method, and thread the field into the `Core` call in `doneInternal`. The method is `WithFusion` to match the existing `WithX` convention (`WithPreProjected`, `WithQueryKeyScale`, `WithDropout`) and Jan's inline comment; the underlying field/param stays `useFusion bool`.

**Files:**
- Modify: `ml/layers/attention/multiheadattention.go` (struct field ~32-76, constructor ~142-156, new method near other `With*` methods, `doneInternal` Core call ~446)
- Test: `ml/layers/attention/fusion_test.go`

**Interfaces:**
- Produces (Contract D):
  ```go
  func (b *MultiHeadAttentionBuilder) WithFusion(enabled bool) *MultiHeadAttentionBuilder
  ```
- Consumes: `Core(..., useFusion bool)` from Task 2.

- [ ] **Step 1: Write the failing test**

Add to `ml/layers/attention/fusion_test.go`:
```go
// TestBuilderWithFusionDefaultsTrue pins that the builder defaults useFusion to true and that
// WithFusion(false) flips it. Builder-level — inspects the field through a Done() run on CPU and
// asserts the output shape is correct either way (CPU has no fused causal kernel, so both equal
// the reference; the test guards the wiring compiles and the default is true).
func TestBuilderWithFusionDefaultsTrue(t *testing.T) {
	backend := testutil.BuildTestBackend()
	const B, S, H, D = 1, 16, 2, 8
	x := tensors.FromFlatDataAndDimensions(randFlat(B*S*(H*D), 1), B, S, H*D)

	store := model.NewStore()
	exec := model.MustNewExec(backend, store, func(scope *model.Scope, in *Node) []*Node {
		def := SelfAttention(scope.In("def"), in, H, D).UseCausalMask().Done()
		off := SelfAttention(scope.In("off"), in, H, D).UseCausalMask().WithFusion(false).Done()
		// Weights are not shared across scopes, so compare shapes only: both must produce
		// [B,S,H*D]. The behavioral on/off equivalence is covered by Task 2.
		return []*Node{def, off}
	})
	out := exec.MustCall(x)
	require.Equal(t, []int{B, S, H * D}, out[0].Shape().Dimensions)
	require.Equal(t, []int{B, S, H * D}, out[1].Shape().Dimensions)
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./ml/layers/attention/ -run TestBuilderWithFusionDefaultsTrue`
Expected: FAIL — compile error `b.WithFusion undefined`.

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
// WithFusion controls whether the backend fused scaled-dot-product-attention path may be used.
// Default is true. When false, Core always takes the decomposed path (useful for debugging or
// for configs the fused kernel does not support). GOMLX_FUSION is a separate global override.
func (b *MultiHeadAttentionBuilder) WithFusion(enabled bool) *MultiHeadAttentionBuilder {
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

Run: `go test ./ml/layers/attention/ -run TestBuilderWithFusionDefaultsTrue`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/guygrigsby/projects/forks/gomlx
git add ml/layers/attention/multiheadattention.go ml/layers/attention/fusion_test.go
git commit -m "attention: add MultiHeadAttentionBuilder.WithFusion (default true)

Constructor seeds useFusion=true; WithFusion(false) forces the decomposed path
through Core. Method name follows the WithX builder convention."
```

---

## Task 4: Builder `WithSeqLens` → config QuerySeqLen/KeyValueSeqLen [S1]

Add `querySeqLen`/`keyValueSeqLen *Node` fields and `WithSeqLens`, mutually exclusive with an explicit `queryKeyMatrixMask`. When set, `doneInternal` builds a `*compute.ScaledDotProductAttentionConfig{QuerySeqLen, KeyValueSeqLen}` and passes it into the fused branch. Threading the config requires `Core` to accept and forward it. **Stage 1 only populates `QuerySeqLen/KeyValueSeqLen`** — the only fused-config fields that exist at this stage (per compute Stage 1). `Bias` config plumbing is Stage 2 (Task 8), and that compute field does not exist until compute Stage 2.

**Files:**
- Modify: `ml/layers/attention/attention.go` (Core: accept an optional config and pass it to `BackendFusedScaledDotProductAttention` instead of the hardcoded `nil` at ~272)
- Modify: `ml/layers/attention/multiheadattention.go` (fields, method, config construction in `doneInternal`)
- Test: `ml/layers/attention/fusion_test.go`

**Interfaces:**
- Produces (Contract D):
  ```go
  func (b *MultiHeadAttentionBuilder) WithSeqLens(querySeqLen, keyValueSeqLen *Node) *MultiHeadAttentionBuilder
  ```
- Consumes (Contract A, Stage 1): `compute.ScaledDotProductAttentionConfig{ QuerySeqLen, KeyValueSeqLen compute.Value }`.
- Internal change: `Core` gains a trailing `*compute.ScaledDotProductAttentionConfig` param (appended after `useFusion`):
  ```go
  func Core(..., scoreSoftCap float64, useFusion bool, fusedConfig *compute.ScaledDotProductAttentionConfig) (output, coefficients *Node)
  ```

**Note on the *Node → compute.Value boundary:** `compute.ScaledDotProductAttentionConfig.QuerySeqLen` is a `compute.Value`, but the builder holds `*Node`. The graph package converts a `*Node` to its backend `compute.Value` via its `outputOps[0]`, which is unexported. Resolve this by having `doneInternal` construct the config through a tiny exported helper in `core/graph` that builds the config from `*Node` seqlens. See Step 3.

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
(Confirm `outputOps` is the field name on `*Node` in this fork — grep `outputOps` in `core/graph/fused_ops.go`; it is used there, so the helper compiles in the same package.)

In `ml/layers/attention/attention.go`, append the config param to `Core` and forward it:
```go
func Core(scope *model.Scope, query, key, value *Node, scale float64, attentionMask *Node, dropoutRate *Node,
	layout AxesLayout, useCausalMask, wantCoefficients bool, scoreSoftCap float64, useFusion bool,
	fusedConfig *compute.ScaledDotProductAttentionConfig) (output, coefficients *Node) {
```
(Add `"github.com/gomlx/compute"` to attention.go imports if not present — confirm; the package already imports `compute/dtypes` and `compute/shapes`.) In the fused closure (~270-272), pass `fusedConfig` instead of `nil`:
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
Add the method near `WithFusion`:
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

The production caller in `doneInternal` is done above. Update `attention_test.go`'s 15 calls to append `, nil` (no fused config), and the `TestCoreUseFusionFalseMatchesDecomposed` / any `fusion_test.go` Core calls to append `, nil`.

Run: `grep -rn 'Core(scope' ml/layers/attention/*_test.go` and append `, nil` to each call that does not already pass a config.

- [ ] **Step 5: Run the test to verify it passes**

Run: `go test ./ml/layers/attention/ -run 'TestWithSeqLensRejectsExplicitMask|TestCoreUseFusionFalseMatchesDecomposed|TestBuilderWithFusionDefaultsTrue'`
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

## Task 5: `GetOfficialBackend` helper [S1]

Add the `testutil.GetOfficialBackend` skip-or-return helper. It returns a named official backend if present in the system, else nil so the caller skips — used by the one `xla:cuda`-specific fusion-parity test (Task 6). No replacement for the deleted `FlashAttention` is added (Contract D, "No SimpleAttention helper"): consumers that need the bare pre-projected fused-or-decomposed op call `graph.InternalFusedOpCaller` + `graph.BackendFusedScaledDotProductAttention` directly (Contract F shows lmkit-go doing exactly this), and the full path stays on the `MultiHeadAttention` builder.

**Files:**
- Modify: `support/testutil/testutil.go` (new `GetOfficialBackend`)
- Test: `ml/layers/attention/fusion_test.go`

**Interfaces:**
- Produces (Contract D):
  ```go
  func GetOfficialBackend(name string) compute.Backend
  ```

- [ ] **Step 1: Write the failing test (CPU)**

Add to `ml/layers/attention/fusion_test.go`:
```go
// TestGetOfficialBackendSkipsWhenAbsent pins that GetOfficialBackend returns the named backend when
// present (the CPU "go" backend always is) and a nil for an absent name (caller skips). CPU-only.
func TestGetOfficialBackendSkipsWhenAbsent(t *testing.T) {
	testutil.BuildTestBackend()
	require.NotNil(t, testutil.GetOfficialBackend("go"), "go backend should be available")
	require.Nil(t, testutil.GetOfficialBackend("definitely-not-a-backend"))
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./ml/layers/attention/ -run 'TestGetOfficialBackendSkipsWhenAbsent'`
Expected: FAIL — `testutil.GetOfficialBackend undefined`.

- [ ] **Step 3: Implement GetOfficialBackend**

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

- [ ] **Step 4: Run the test to verify it passes**

Run: `go test ./ml/layers/attention/ -run 'TestGetOfficialBackendSkipsWhenAbsent'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/guygrigsby/projects/forks/gomlx
git add support/testutil/testutil.go ml/layers/attention/fusion_test.go
git commit -m "attention: add testutil.GetOfficialBackend

GetOfficialBackend returns a named official backend or nil (caller skips), used by
the xla:cuda fusion-parity test. No SimpleAttention helper is added (Contract D):
consumers needing the bare pre-projected fused-or-decomposed op call
graph.InternalFusedOpCaller + BackendFusedScaledDotProductAttention directly."
```

---

## Task 6: Generalize parity tests — `backendSupportsFusion`, loop official backends [S1]

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
// TestFusionFallbackParity checks that on every official backend, attention through Core (fused path
// attempted) matches the decomposed reference, forward and gradients. On a backend without the fused
// kernel this exercises the ErrNotImplemented fallback; on one with it, it is a parity check. Loops
// all official backends.
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

// TestCUDAFusionParity [cuda] cross-checks the cuDNN fused kernel against the float32 reference at
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

- [ ] **Step 6: [cuda] Run the CUDA parity test on the CUDA host**

Run (on the CUDA host, after `git pull` in the fork checkout):
```bash
GOMLX_BACKEND=xla:cuda \
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

## Task 7: Rename and rework the throughput/memory bench [S1]

Rename `flash_bench_test.go` → `fusion_bench_test.go`. `attentionStep` uses the **same** code path for fused and non-fused (toggling via `WithFusion`, not a separate `naiveCausalAttention` branch). Rename `TestFlashThroughput` → `TestFusionThroughput`. Add `out[0].FinalizeAll()` after the sync. Use `for range iters`. Replace `GOMLX_MEM_PROBE`/`GOMLX_PROBE_S` env gating with Go test flags.

**Files:**
- Rename: `ml/layers/attention/flash_bench_test.go` → `ml/layers/attention/fusion_bench_test.go`
- Modify: the renamed file (rewrite `attentionStep`, `TestFusionThroughput`, memory probe).

**Interfaces:**
- Consumes: `MultiHeadAttention` builder (`WithPreProjected`/`WithQueryKeyScale`/`WithFusion`/`WithNumKVHeads`/`UseCausalMask`), `backendSupportsFusion`, `testutil.BuildTestBackend`. (The bench needs the WithFusion toggle and GQA, so it goes through the builder.)

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
// fused and non-fused — fusion toggles via WithFusion, not a separate decomposed code path — so the
// benchmark compares the two routes of one implementation. Returns a scalar so reading it forces a
// device sync. query/key/value are pre-projected [B,S,H,D].
//
// The bench uses the MultiHeadAttention builder directly because it needs the WithFusion toggle and
// GQA (WithNumKVHeads) — the builder is the right surface for a fused-vs-decomposed micro-benchmark.
func attentionStep(useFusion bool, qHeads, kvHeads int, scale float64) func(scope *model.Scope, q, k, v *Node) []*Node {
	return func(scope *model.Scope, q, k, v *Node) []*Node {
		// Pre-projected [B,S,H,D] in, flatten the head dims for the builder's pre-projected path.
		flat := func(n *Node) *Node { d := n.Shape().Dimensions; return Reshape(n, d[0], d[1], d[2]*d[3]) }
		b := MultiHeadAttention(scope, flat(q), flat(k), flat(v), qHeads, q.Shape().Dimensions[3]).
			WithPreProjected(true).WithQueryKeyScale(scale).UseCausalMask().WithFusion(useFusion)
		if kvHeads != qHeads {
			b = b.WithNumKVHeads(kvHeads)
		}
		out := ConvertDType(b.Done(), dtypes.Float32)
		g := Gradient(ReduceAllSum(out), q, k, v)
		return []*Node{Add(Add(ReduceAllSum(g[0]), ReduceAllSum(g[1])), ReduceAllSum(g[2]))}
	}
}

// TestFusionThroughput reports per-step (forward+backward) wall time for the fused vs decomposed
// route at the lm-100m attention shape. Requires a backend that supports fusion (cuDNN). [cuda]
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
// -mem_probe=fused|decomposed; skipped otherwise. [cuda]
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

- [ ] **Step 4: [cuda] Run the throughput bench on the CUDA host**

Run (on the CUDA host):
```bash
GOMLX_BACKEND=xla:cuda \
  go test ./ml/layers/attention/ -run TestFusionThroughput -v
```
Expected: logs fused and decomposed per-step times and a speedup ratio; PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/guygrigsby/projects/forks/gomlx
git add ml/layers/attention/fusion_bench_test.go
git commit -m "attention: rework throughput/memory bench through one fusion-toggled path

Rename flash_bench_test.go -> fusion_bench_test.go and TestFlashThroughput ->
TestFusionThroughput. attentionStep now runs the SAME MultiHeadAttention builder
for both routes, toggling WithFusion, so the bench compares one implementation's
two paths. FinalizeAll after each sync; for range iters; replace GOMLX_MEM_PROBE/
GOMLX_PROBE_S env gating with -mem_probe/-mem_probe_seqlen test flags."
```

---

## Stage 1 Verification Gate (Plan 03)

Per Contract section "Verification gates / 03 gomlx". Run all on the Mac first; the `[cuda]` line on the CUDA host. **This gate closes Stage 1. Stop here for Guy's review before starting Stage 2.**

- [ ] **Gate 1 — Mac (CPU) full package green**

Run:
```bash
cd /Users/guygrigsby/projects/forks/gomlx
go build ./...
go test ./ml/layers/attention/...
```
Expected: PASS. Covers decomposed path, `ErrNotImplemented` fallback, `WithFusion(false)`, `WithSeqLens` validation, `GetOfficialBackend` skip behavior. `TestCUDAFusionParity`, `TestFusionThroughput`, `TestFusionMemoryProbe` SKIP (no `xla:cuda`).

- [ ] **Gate 2 — No leftover CUDA-shaped surface**

Run:
```bash
cd /Users/guygrigsby/projects/forks/gomlx
grep -rn 'FlashAttention\|isCUDABackend\|GOMLX_MEM_PROBE\|GOMLX_PROBE_S' ml/ core/ support/ --include='*.go'
test ! -e ml/layers/attention/flash.go && test ! -e ml/layers/attention/flash_test.go && test ! -e ml/layers/attention/flash_bench_test.go && echo "OK: flash files removed"
```
Expected: no grep output; prints `OK: flash files removed`.

- [ ] **Gate 3 — [cuda] cuDNN fusion-parity green**

Run (on the CUDA host, fork checkout pulled to the latest `flash-attention` commit):
```bash
GOMLX_BACKEND=xla:cuda \
  go test ./ml/layers/attention/... -run 'TestCUDAFusionParity|TestFusionFallbackParity|TestFusionGQAParity|TestFusionThroughput' -v
```
Expected: PASS — the cuDNN fused kernel runs (parity within tolerance for D=64 and D=128), the looped parity tests pass at `tol=0.06`, and throughput reports a speedup.

- [ ] **Gate 4 — Stop and report**

All gates green → plan 03 Stage 1 (Contract D, S1 surface) is satisfied; downstream plan 04 (go-huggingface) and Contract F (lmkit-go) may consume `WithFusion`/`WithSeqLens` and the bare `graph.InternalFusedOpCaller` + `graph.BackendFusedScaledDotProductAttention` primitives via the local `replace` on this gomlx fork. Report the gate output to Guy for manual diff review before any push, and before starting Stage 2.

---

## Task 8: Builder `WithAttentionBias` → config Bias [S2]

**Depends on compute Stage 2:** `compute.ScaledDotProductAttentionConfig` must carry the `Bias Value` field (plan 01 Stage 2). Do not start until it does.

Add an `attentionBias *Node` field and a `WithAttentionBias(bias *Node)` builder method that populates `config.Bias`. This is the additive attention-score bias broadcast to `[B,H,S,Skv]` (ALiBi / relative-position). **It is DISTINCT from the existing `UseProjectionBias`** (the Q/K/V / output dense bias) and MUST NOT be conflated: `UseProjectionBias` toggles a learned bias term inside the linear projections; `WithAttentionBias` adds a caller-supplied tensor to the attention scores before softmax. They are independent and may both be set. The method threads the bias node into the fused config (the score bias rides in `ScaledDotProductAttentionConfig.Bias`, which only exists after compute Stage 2).

**Files:**
- Inspect: the compute fork `fused_ops.go` (verify `Bias Value` field exists)
- Modify: `core/graph/fused_ops.go` (extend `NewSeqLenAttentionConfig` or add a sibling helper that also sets `Bias`)
- Modify: `ml/layers/attention/multiheadattention.go` (field, method, config construction in `doneInternal`)
- Test: `ml/layers/attention/fusion_test.go`

**Interfaces:**
- Produces (Contract D):
  ```go
  func (b *MultiHeadAttentionBuilder) WithAttentionBias(bias *Node) *MultiHeadAttentionBuilder
  ```
- Consumes (Contract A, Stage 2): `compute.ScaledDotProductAttentionConfig{ Bias Value }`.

- [ ] **Step 0: Verify compute Stage 2 `Bias` field exists**

Run:
```bash
cd /Users/guygrigsby/projects/forks/gomlx
go doc github.com/gomlx/compute.ScaledDotProductAttentionConfig
```
Expected: lists `Bias Value`. If absent, STOP and report: "plan 01 Stage 2 (compute) is not integrated; cannot proceed with plan 03 Stage 2."

- [ ] **Step 1: Write the failing test (CPU)**

The fused bias path is `ErrNotImplemented` on CPU (capability stays off), so the CPU test pins that bias flows through the decomposed path and is NOT conflated with the projection bias. Add to `ml/layers/attention/fusion_test.go`:
```go
// TestWithAttentionBiasIsDistinctFromProjectionBias pins that WithAttentionBias adds a score bias
// (changing the output) and is independent of UseProjectionBias. On CPU the fused path falls back to
// decomposed; the decomposed path must honor the additive score bias. A zero bias is a no-op; a
// nonzero bias shifts the output.
func TestWithAttentionBiasIsDistinctFromProjectionBias(t *testing.T) {
	backend := testutil.BuildTestBackend()
	const B, S, H, D = 1, 16, 2, 8
	scale := 0.125
	q := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 1), B, S, H, D)
	k := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 2), B, S, H, D)
	v := tensors.FromFlatDataAndDimensions(randFlat(B*S*H*D, 3), B, S, H, D)
	// Nonzero bias [B,H,S,S].
	biasData := randFlat(B*H*S*S, 9)
	bias := tensors.FromFlatDataAndDimensions(biasData, B, H, S, S)

	// preProj builds the pre-projected builder from [B,S,H,D] inputs. WithAttentionBias is a builder
	// method, so these S2 tests use the MultiHeadAttention builder on the pre-projected path.
	flat := func(n *Node) *Node { d := n.Shape().Dimensions; return Reshape(n, d[0], d[1], d[2]*d[3]) }
	preProj := func(scope *model.Scope, qn, kn, vn *Node) *MultiHeadAttentionBuilder {
		return MultiHeadAttention(scope, flat(qn), flat(kn), flat(vn), H, D).
			WithPreProjected(true).WithQueryKeyScale(scale)
	}

	store := model.NewStore()
	exec := model.MustNewExec(backend, store, func(scope *model.Scope, qn, kn, vn, bn *Node) []*Node {
		base := preProj(scope.In("base"), qn, kn, vn).Done()
		biased := preProj(scope.In("biased"), qn, kn, vn).WithAttentionBias(bn).Done()
		// A nonzero score bias must change the output.
		return []*Node{ReduceAllMax(Abs(Sub(base, biased)))}
	})
	out := exec.MustCall(q, k, v, bias)
	delta := float64(tensors.ToScalar[float32](out[0]))
	require.Greater(t, delta, 1e-4, "nonzero attention bias must change the output (decomposed path)")
}
```
(This requires the decomposed path to honor `fusedConfig.Bias`. If the decomposed path does not yet read a bias from the config, add it: in `Core`'s `decomposedFn`, after `scores = MulScalar(...)` and before the mask, `if fusedConfig != nil && fusedConfig.Bias is set { scores = Add(scores, biasNode) }`. The bias arrives as a `compute.Value` in the config; thread the original `*Node` into `Core` alongside the config — simplest is to pass the bias as the existing `attentionMask` additive-float operand when no explicit mask is set, OR add a dedicated `biasNode *Node` param to `Core`. Pick the dedicated param to keep mask semantics clean; grep how `attentionMask` additive-float is applied at `attention.go:237-240` and mirror it. State the chosen wiring in the commit.)

- [ ] **Step 2: Run the test to verify it fails**

Run: `go test ./ml/layers/attention/ -run TestWithAttentionBiasIsDistinctFromProjectionBias`
Expected: FAIL — compile error `b.WithAttentionBias undefined`.

- [ ] **Step 3: Add the field, method, config plumbing, and decomposed-path bias**

In `ml/layers/attention/multiheadattention.go`, add the field (after `keyValueSeqLen *Node`):
```go
	attentionBias *Node // additive attention-score bias [B,H,S,Skv]; distinct from useProjectionBias.
```
Add the method near `WithSeqLens`:
```go
// WithAttentionBias supplies an additive attention-score bias broadcast to [B,H,S,Skv]
// (ALiBi / relative-position), added to the scores before softmax. This is DISTINCT from
// UseProjectionBias, which toggles the learned bias term inside the Q/K/V and output dense
// projections. The two are independent and may both be set. Populates
// ScaledDotProductAttentionConfig.Bias for the fused path; the decomposed fallback adds it
// directly to the scores. nil = unused.
func (b *MultiHeadAttentionBuilder) WithAttentionBias(bias *Node) *MultiHeadAttentionBuilder {
	b.attentionBias = bias
	return b
}
```
Extend `core/graph/fused_ops.go`'s config helper (or add a sibling) so it also sets `Bias` when a bias node is present:
```go
// NewFusedAttentionConfig builds a ScaledDotProductAttentionConfig from optional seqlen and bias
// nodes (all may be nil). Supersedes NewSeqLenAttentionConfig once bias is in play.
func NewFusedAttentionConfig(querySeqLen, keyValueSeqLen, bias *Node) *compute.ScaledDotProductAttentionConfig {
	cfg := &compute.ScaledDotProductAttentionConfig{}
	if querySeqLen != nil {
		cfg.QuerySeqLen = querySeqLen.outputOps[0]
	}
	if keyValueSeqLen != nil {
		cfg.KeyValueSeqLen = keyValueSeqLen.outputOps[0]
	}
	if bias != nil {
		cfg.Bias = bias.outputOps[0]
	}
	return cfg
}
```
In `doneInternal`, build the config with the bias and thread the bias `*Node` into `Core` (see Step 1's wiring note — pass a dedicated `biasNode *Node` param to `Core` so the decomposed fallback can add it to the scores):
```go
	var fusedConfig *compute.ScaledDotProductAttentionConfig
	if b.querySeqLen != nil || b.keyValueSeqLen != nil || b.attentionBias != nil {
		fusedConfig = NewFusedAttentionConfig(b.querySeqLen, b.keyValueSeqLen, b.attentionBias)
	}
	attentionOutput, attentionCoefficients = Core(b.scope, projectedQuery, projectedKey, projectedValue,
		scale, mask, b.dropoutRate, b.layout, useCausalMask, wantCoefficients, b.scoreSoftCap, b.useFusion,
		fusedConfig, b.attentionBias)
```
In `Core` (`attention.go`), add the trailing `biasNode *Node` param and, in `decomposedFn`, add the bias to the scores (mirror the additive-float-mask handling at lines 237-240):
```go
	scores = MulScalar(scores, scale)
	if biasNode != nil {
		scores = Add(scores, biasNode) // additive score bias; broadcasts [B,H,S,Skv] / [B,S,H,Skv] per layout.
	}
```
Update all `Core(...)` call sites (1 production + 15 in `attention_test.go` + the `fusion_test.go` calls) to append `, nil` for the new `biasNode` param.

Run after editing: `grep -rn 'Core(scope' ml/layers/attention/*_test.go` and confirm each non-bias call appends the trailing `nil`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `go test ./ml/layers/attention/ -run TestWithAttentionBiasIsDistinctFromProjectionBias`
Expected: PASS (decomposed path applies the bias on CPU).

- [ ] **Step 5: Run the package**

Run: `go build ./... && go test ./ml/layers/attention/`
Expected: PASS.

- [ ] **Step 6: [cuda] Bias fused-parity on the CUDA host**

Add an `xla:cuda` bias-parity case (mirror `TestCUDAFusionParity`, but build `Core` with a nonzero bias and compare against `naiveCausalAttention` with the same bias added to its scores). Run on the CUDA host:
```bash
GOMLX_BACKEND=xla:cuda \
  go test ./ml/layers/attention/ -run 'TestCUDAFusionParity|TestWithAttentionBias' -v
```
Expected: PASS — the `fmhaScaleBiasSoftmax` variant runs and stays within tolerance.

- [ ] **Step 7: Commit**

```bash
cd /Users/guygrigsby/projects/forks/gomlx
git add core/graph/fused_ops.go ml/layers/attention/attention.go ml/layers/attention/multiheadattention.go ml/layers/attention/attention_test.go ml/layers/attention/fusion_test.go
git commit -m "attention: add WithAttentionBias (score bias, distinct from UseProjectionBias)

WithAttentionBias adds a caller-supplied additive score bias [B,H,S,Skv] before
softmax, populating ScaledDotProductAttentionConfig.Bias for the fused path and
added directly to scores in the decomposed fallback. Independent of the projection
bias (UseProjectionBias); both may be set. Core gains a trailing biasNode param."
```

---

## Task 9: Relax Core's `!dropoutActive` fused-branch gate [S2]

CUT (2026-06-30): fused dropout removed; no caller needs it. Stays a NotImplemented seam. See contract Staging.

---

## Stage 2 Verification Gate (Plan 03)

Run after Tasks 8–9. Run on the Mac first; `[cuda]` lines on the CUDA host.

- [ ] **Gate 1 — Mac (CPU) full package green**

Run:
```bash
cd /Users/guygrigsby/projects/forks/gomlx
go build ./...
go test ./ml/layers/attention/...
```
Expected: PASS. `WithAttentionBias` decomposed bias falls back correctly on CPU; CUDA tests SKIP.

- [ ] **Gate 2 — [cuda] bias fused-parity green**

Run (on the CUDA host):
```bash
GOMLX_BACKEND=xla:cuda \
  go test ./ml/layers/attention/... -run 'TestCUDAFusionParity|TestWithAttentionBias' -v
```
Expected: PASS — `fmhaScaleBiasSoftmax` variant runs within tolerance.

- [ ] **Gate 3 — Stop and report**

All Stage 2 gates green → Contract D is fully satisfied (S1 + S2). Report gate output to Guy for manual diff review before any push.

---

## Self-Review (writing-plans)

**Spec coverage (Contract D + the four decided changes):**
- **Rename `UseFusion → WithFusion`:** builder method, all test names (`TestBuilderWithFusionDefaultsTrue`, `TestCoreUseFusionFalseMatchesDecomposed` keeps `UseFusion` only in the *internal* identifier per the param name), and prose use `WithFusion`. The `Core` param stays `useFusion bool` (plain bool, appended after `scoreSoftCap`). → Tasks 2-3, bench Task 7. ✓
- **Real Core call-site count:** grepped — **1 production caller** (`multiheadattention.go:446`) + **15** `Core(scope ...)` calls in `attention_test.go` (lines 28, 36, 79, 101, 102, 132, 161, 188, 228, 230, 286, 287, 323, 324, 366) + the new `fusion_test.go` calls. Stated in File Structure and Tasks 2/4/8. The earlier "12" figure was wrong and is removed. ✓
- **Stage tags:** every task carries `[S1]` or `[S2]`. S1 = Tasks 0-7 (refactor, no variants); S2 = Tasks 8-9 (variants). S1 ordered first, S2 last, with a Stage 1 gate between them. ✓
- Delete flash.go's `FlashAttention` with **no replacement helper** (Contract D, "No SimpleAttention helper"). `naiveCausalAttention` has no runtime caller after the deletion (Core has its own inline decomposed path; no `SimpleAttention` to use it as a fallback), so all three flash helpers — `naiveCausalAttention`, `naiveGQAReference`, `repeatKVHeads` — are test-only decomposed oracles and move to `fusion_test.go`. No non-test `fusion.go` is created. **Preserve `repeatKVHeads`** (relocated, pinned by `TestRepeatKVHeads`). State why deletion loses no behavior → Task 1 [S1]. ✓
- `WithFusion(enabled bool)`, default true in constructor, thread to `Core` via trailing `useFusion bool`, gate fused branch, update all Core call sites → Tasks 2-3 [S1]. ✓
- `WithSeqLens` → `ScaledDotProductAttentionConfig.QuerySeqLen/KeyValueSeqLen` only (the only S1 config fields), mutually exclusive with `queryKeyMatrixMask` (both orders) → Task 4 [S1]. ✓
- **No `SimpleAttention` helper** (Contract D, decided 2026-06-29 on upstream merits — DROPPED entirely, reversing the earlier option-A entry). Any concrete version is either a parallel duplicate of the fused-or-decomposed logic or needs a scope; the bare `graph.BackendFusedScaledDotProductAttention` + `graph.InternalFusedOpCaller` primitives and the `MultiHeadAttention` builder already cover the need. Contract F shows lmkit-go consuming the bare primitives directly. → noted across Task 1 (deletion, no replacement) and Task 5 (`GetOfficialBackend` only). ✓
- `backendSupportsFusion` probe replacing `isCUDABackend`; loop `TestOfficialBackends`; `testutil.GetOfficialBackend` (Task 5); one `xla:cuda` parity test → Tasks 5-6 [S1]. ✓
- Bench: rename `TestFusionThroughput`, single `WithFusion`-toggled code path, `FinalizeAll`, `for range iters`, flags-not-env → Task 7 [S1]. ✓
- **`WithAttentionBias(bias *Node)` → `config.Bias`, explicitly DISTINCT from `UseProjectionBias`** (independent, may both be set; the task spells out the difference) → Task 8 [S2]. ✓
- **Fused dropout: CUT (2026-06-30).** Fused-with-dropout removed; no caller needs it. `Core`'s `!dropoutActive` gate stays UNCHANGED (still declines to fuse when decomposed dropout is active); Task 9 is a NotImplemented seam → Task 9 [S2]. ✓
- The S2 task notes its dependency on the compute Stage 2 config field (`Bias`); Task 8 Step 0 verifies it before code. ✓
- CUDA-execution tests marked `[cuda]`; CPU-testable items enumerated → present per task and in both gates. ✓

**Placeholder scan:** No "TBD"/"handle edge cases"/"similar to Task N". Flagged uncertainties — the `*Node → compute.Value` config boundary (Task 4), the `FinalizeAll` finalizer name (Task 7), the bias `*Node` wiring into `Core`'s decomposed path (Task 8) — each carry an explicit grep-and-resolve instruction with a concrete fallback, not a hand-wave. Acceptable: the implementer is told exactly what to check and what to do either way.

**Type consistency:** `Core`'s signature grows monotonically — Task 2 appends `useFusion bool`, Task 4 appends `fusedConfig *compute.ScaledDotProductAttentionConfig`, Task 8 appends `biasNode *Node` — and every call-site-update step references the same trailing-arg shape at that point in the plan. `WithFusion`/`WithSeqLens`/`GetOfficialBackend`/`backendSupportsFusion`/`NewSeqLenAttentionConfig`/`NewFusedAttentionConfig`/`WithAttentionBias` names match between producing and consuming tasks. No `SimpleAttention` is produced — the deletion has no replacement helper. `naiveCausalAttention`/`naiveGQAReference`/`repeatKVHeads` relocate once to `fusion_test.go` (Task 1) as test-only decomposed oracles, referenced by the parity tests (Task 6); `randFlat` is added to `fusion_test.go` (Task 2) and referenced thereafter.

**Known residual risk to flag at execution:**
- Task 4/8's `NewSeqLenAttentionConfig`/`NewFusedAttentionConfig` reach into `node.outputOps[0]`; confirm `outputOps` is the field name on `*Node` in this fork (grep `outputOps` in `core/graph/fused_ops.go` — it is used there, so the helpers compile in the same package). The only places the plan touches `core/graph` rather than `ml/layers/attention`.
