# Fused Attention 02 — go-xla Backend (CustomCallV2 + fmha variants + seqlen) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Read the shared contract first: `docs/plans/2026-06-29-fused-attn-00-contract.md`.

**Goal:** Land the `go-xla` half of fused-attention upstreaming (PR gomlx/go-xla#37): a typed `CustomCallV2`, cuDNN fmha variant dispatch (bias, bf16), and seqlen padding masks — keeping every `__cudnn$fmha*` detail inside `go-xla`. FP8 is paused (no local sm_8.9+ hardware); fp8 input falls to `ErrNotImplemented` and is handed to janpfeifer — see Contract C.

**Staging.** This plan ships in two stages, per the contract's "Staging" section. **Do Stage 1 in full — review — then Stage 2.** Every task below is tagged **[S1]** or **[S2]**:
- **[S1]** (strict refactor to Jan's API; no new attention variants): Task 0 (sync), Task 1 (`CustomCallV2`), Task 2 (dispatch seam routing ONLY the standard `__cudnn$fmhaSoftmax`/`Backward` target + seqlen `mask_type`), the seqlen operand append (Task 3 [S1] part) + `flashSupported` relaxation (Task 4), and the `IsCUDA` test migration (Task 5). The fp8-paused dtype-gate unit test is also [S1].
- **[S2]** (added bias variant, bf16; fp8 stays paused): the bias branch that EXTENDS `selectFMHAVariant`, its target constants, the per-variant bias operand set, and the `[cuda]` per-variant exec test.

**Amendment A1 (2026-07-01, see contract doc).** The fused forward's stats output and the VJP's stats input change from a single `softmaxStats Value` to `statesForVJP []Value`. In this plan that means the VJP operand set built from the state (Task ~; the `operands := []compute.Value{q, k, v, softmaxStats, dOut, out}` snippets) now unpacks `statesForVJP` instead of a single `softmaxStats`. For the current cuDNN path `statesForVJP` carries the one softmax-stats tensor (plus a `workspace` byte tensor if/when a wired variant needs it); the go-xla backend op returns/consumes it as a slice.

**Cross-stage compile dependency (CRITICAL).** The S1 `selectFMHAVariant` **must not** reference `cfg.Bias`. That `compute.ScaledDotProductAttentionConfig` field **does not exist until compute Stage 2** (Contract A is staged: S1 adds only `QuerySeqLen`/`KeyValueSeqLen`). So the S1 selector reads only `cfg.QuerySeqLen`/`cfg.KeyValueSeqLen` + the `causal` bool, and the S1 `fmhaVariant` struct omits the bias field. Stage 2 ADDS that read, the struct field, and the variant target constants. Both versions are shown in full (Task 2 = S1 minimal, Task 2b = S2 extension). Referencing an S2 field in S1 code will not compile against the Stage-1 compute fork.

**Architecture:** This plan implements Contract B (CustomCallV2) and Contract C (fmha variant dispatch + seqlen masking) from the shared contract, and consumes Contract A (the `compute.ScaledDotProductAttentionConfig` fields produced by plan 01, staged: S1 fields first). The string-layout `CustomCall` is replaced by a `[][]int`-layout `CustomCallV2` that renders MLIR `dense<...>` internally; `compute/xla/flash.go` rewires onto it and grows a config-driven target selector. Nothing flash-/CUDA-specific leaks above the backend: callers still see only "FusedSDPA: supported or `ErrNotImplemented`".

**Tech Stack:** Go 1.26, `github.com/gomlx/go-xla` (PR branch `flash-attention`), StableHLO emitter (`stablehlo` package), PJRT (`pjrt` package), cuDNN fMHA custom-calls. Consumes `github.com/gomlx/compute` (PR branch `flash-customcall`, via local `replace`).

## Global Constraints

(Verbatim from the shared contract — every task's requirements implicitly include these.)

- **Go 1.26.** Module: `github.com/gomlx/go-xla`. Consumes `github.com/gomlx/compute` via the local `replace` already in this fork's `go.mod`/`go.work`.
- **No push, no PR.** Every task ends at `git commit` on branch `flash-attention` in `/Users/guygrigsby/projects/forks/go-xla`. Guy reviews all diffs before anything reaches a remote.
- **CUDA-gated tests run on the CUDA host, not the Mac.** CI has no CUDA. Tasks whose test needs cuDNN execution are marked **[cuda]**: build natively on the CUDA host, `GOMLX_BACKEND=xla:cuda`, serialize GPU jobs if the card is shared. The CustomCallV2 layout-rendering unit test is CPU-only and runs on the Mac.
- **Fallback is the contract.** Any unsupported config (wrong dtype, unsupported variant/layout) returns a wrapped `compute.ErrNotImplemented`; the caller falls back to the decomposed path. Never panic on an unsupported-but-valid request.
- **Commit voice:** terse, verb-first, no em/en dashes, prefix `compute/xla:` (or `stablehlo:` for the customcall.go-only task), no Claude attribution.

**Grounded facts (verified against `fork/flash-attention`, the post-merge target):**
- `stablehlo/customcall.go`: `type literalStr string` (statement.go:177); attrs set `operand_layouts`/`result_layouts` to `literalStr(...)`. Const `CustomCallAPIVersionStatusReturning = 2`.
- `compute/xla/ops.go:826`: `customCall` wrapper; imports `shapes "github.com/gomlx/go-xla/types/shapes"` and aliases the stablehlo shapes as `stablehloshapes`. It maps `compute.Value` → stablehlo and back via `f.verifyAndCastValues` / `f.newNode`.
- `compute/xla/flash.go`: uses `github.com/gomlx/compute/shapes` and `github.com/gomlx/compute/dtypes`. `flashSupported` gates on `IsCUDA`, causal, nil mask, BSHD, equal heads. `flashFwdBackendConfig`/`flashBwdBackendConfig` build JSON via `flashBackendConfig(b,h,s,scale,dotDimNumbers)` with hardcoded `"mask_type": "CAUSAL"`, `"dropout_rate": 0.0`.
- `compute` dtypes (consumed): `dtypes.Float16`, `dtypes.BFloat16`, `dtypes.F8E4M3FN`, `dtypes.F8E5M2`.
- `pjrt/fmha_test.go`: gates `if *FlagPluginName != "cuda"`. `getPJRTClient(t) *Client` (pjrt_test.go:65); `(*Client).Plugin() *Plugin` (clients.go:212); `(*Plugin).IsCUDA() bool` (plugins.go:154).

---

## Task 0 [S1]: Sync local branch to `fork/flash-attention` (NOT SKIPPABLE)

The local checkout is on a different branch (`rocm-flash-attention`); `fork/flash-attention` has the maintainer's backend-options merge. Every downstream task assumes this synced state (the line numbers and the `ScaledDotProductAttentionConfig` param already present in `flash.go` come from the merge).

**Files:**
- Modify: working tree of `/Users/guygrigsby/projects/forks/go-xla` (branch checkout only; no source edits)

- [ ] **Step 1: Fetch and switch to the PR branch**

```bash
cd /Users/guygrigsby/projects/forks/go-xla
git fetch fork
git checkout flash-attention
```

- [ ] **Step 2: Fast-forward to the maintainer's merged work**

Try ff-only first; fall back to rebase only if ff is refused.

```bash
git merge --ff-only fork/flash-attention || git rebase fork/flash-attention
```

Expected: branch now at `fork/flash-attention` HEAD (commit `28d1e9b` "Merge branch 'main' ... into flash-attention", or later).

- [ ] **Step 3: Verify the tree builds**

```bash
go build ./...
```

Expected: exit 0, no output. If it fails, the rebase/merge produced a broken tree — STOP and resolve before any downstream task.

- [ ] **Step 4: Confirm the grounded preconditions hold**

```bash
git grep -n "func CustomCall" stablehlo/customcall.go
git grep -n "func (f \*Function) customCall" compute/xla/ops.go
git grep -n "ScaledDotProductAttentionConfig" compute/xla/flash.go
```

Expected: the string `CustomCall` exists in `stablehlo/customcall.go`; `customCall` wrapper at ~`compute/xla/ops.go:826`; `flash.go` references `*compute.ScaledDotProductAttentionConfig` in both fused method signatures. If `ScaledDotProductAttentionConfig` is absent, plan 01 (compute) has not been integrated — STOP; this plan consumes Contract A.

Stage check: confirm the integrated compute fork matches the stage you are executing. For Stage 1, the struct carries `QuerySeqLen`/`KeyValueSeqLen` but **not** `Bias` (which lands in compute Stage 2):

```bash
git grep -n "QuerySeqLen\|KeyValueSeqLen" $(go env GOMODCACHE >/dev/null 2>&1; go list -m -f '{{.Dir}}' github.com/gomlx/compute)/scaleddotproductattention.go 2>/dev/null \
  || echo "check the compute fork's config struct manually"
```

Expected (Stage 1): `QuerySeqLen`/`KeyValueSeqLen` present, bias field absent. If the bias field is already present, compute Stage 2 has landed — the S1/S2 split below is then a code-organization concern only, but still keep S1 `selectFMHAVariant` free of the S2 field read so the two stages stay independently reviewable.

- [ ] **Step 5: No commit**

Task 0 is a branch sync, not a code change. Do not commit. Downstream tasks commit their own work.

---

## Task 1 [S1]: `CustomCallV2` with typed `[][]int` layouts (Contract B)

Replace the string-layout `CustomCall` with the typed, versioned `CustomCallV2`. Move the MLIR `dense<...>` rendering inside it (built from `[][]int`; a nil entry/slice means row-major = decreasing order over the rank). The layout-rendering helper is the CPU-testable unit. Then remove the old `CustomCall` and rewire the `compute/xla/ops.go` `customCall` wrapper onto `CustomCallV2`.

**Files:**
- Modify: `stablehlo/customcall.go`
- Test: `stablehlo/customcall_test.go` (create)
- Modify: `compute/xla/ops.go` (the `customCall` wrapper, ~line 826)
- Modify: `compute/xla/flash.go` (the two `f.customCall(...)` call sites — minimal signature adaptation; full rewire is Task 2)

**Interfaces:**
- Consumes: `type literalStr string`, `const CustomCallAPIVersionStatusReturning = 2` (both in `stablehlo`); `shapes.Shape` = `github.com/gomlx/go-xla/types/shapes`; `*stablehlo.Value`.
- Produces:
  ```go
  // stablehlo/customcall.go
  func renderLayouts(layouts [][]int, ranks []int) string            // "[dense<[3, 2, 1, 0]> : tensor<4xindex>, ...]"; "" if layouts==nil
  func CustomCallV2(
      target string,
      backendConfig string,
      operands []*Value,
      operandLayouts [][]int,
      outputShapes []shapes.Shape,
      outputLayouts [][]int,
  ) ([]*Value, error)
  ```
  `renderLayouts(nil, _)` returns `""` (omit the attribute). A nil/empty entry at index i renders the row-major default for `ranks[i]`: `dense<[r-1, ..., 1, 0]> : tensor<Rxindex>` (for r==1, `dense<0> : tensor<1xindex>`).

- [ ] **Step 1: Write the failing unit test for layout rendering (CPU-only, runs on Mac)**

`stablehlo/customcall_test.go`:

```go
package stablehlo

import "testing"

func TestRenderLayouts(t *testing.T) {
	for _, tc := range []struct {
		name    string
		layouts [][]int
		ranks   []int
		want    string
	}{
		{
			name:    "nil layouts omits attribute",
			layouts: nil,
			ranks:   []int{4, 4, 4},
			want:    "",
		},
		{
			name:    "explicit BSHD and BHSD",
			layouts: [][]int{{3, 2, 1, 0}, {3, 1, 2, 0}},
			ranks:   []int{4, 4},
			want:    "[dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 1, 2, 0]> : tensor<4xindex>]",
		},
		{
			name:    "nil entry defaults row-major for its rank",
			layouts: [][]int{nil, {2, 1, 0}, nil},
			ranks:   []int{4, 3, 1},
			want:    "[dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[2, 1, 0]> : tensor<3xindex>, dense<0> : tensor<1xindex>]",
		},
		{
			name:    "rank-1 explicit",
			layouts: [][]int{{0}},
			ranks:   []int{1},
			want:    "[dense<0> : tensor<1xindex>]",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := renderLayouts(tc.layouts, tc.ranks)
			if got != tc.want {
				t.Errorf("renderLayouts(%v, %v):\n got %q\nwant %q", tc.layouts, tc.ranks, got, tc.want)
			}
		})
	}
}
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/guygrigsby/projects/forks/go-xla
go test ./stablehlo/ -run TestRenderLayouts -v
```

Expected: FAIL — `undefined: renderLayouts`.

- [ ] **Step 3: Implement `renderLayouts` + `CustomCallV2`, remove old `CustomCall`**

Replace the body of `stablehlo/customcall.go` (keep the package, imports, and the `CustomCallAPIVersionStatusReturning`/`CustomCallAPIVersionTypedFFI` consts) with:

```go
// renderLayouts renders the MLIR operand_layouts/result_layouts array attribute from
// minor-to-major dim orders. layouts==nil returns "" (omit the attribute). Each entry is
// paired by index with ranks; a nil/empty entry defaults to row-major (decreasing order
// over its rank), e.g. rank 4 -> [3, 2, 1, 0]. Output form:
//   "[dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<0> : tensor<1xindex>]"
func renderLayouts(layouts [][]int, ranks []int) string {
	if layouts == nil {
		return ""
	}
	var sb strings.Builder
	sb.WriteByte('[')
	for i, rank := range ranks {
		if i > 0 {
			sb.WriteString(", ")
		}
		order := layouts[i]
		if len(order) == 0 {
			order = make([]int, rank)
			for j := range order {
				order[j] = rank - 1 - j
			}
		}
		sb.WriteString("dense<")
		if len(order) == 1 {
			sb.WriteString(strconv.Itoa(order[0]))
		} else {
			sb.WriteByte('[')
			for j, d := range order {
				if j > 0 {
					sb.WriteString(", ")
				}
				sb.WriteString(strconv.Itoa(d))
			}
			sb.WriteByte(']')
		}
		sb.WriteString("> : tensor<")
		sb.WriteString(strconv.Itoa(rank))
		sb.WriteString("xindex>")
	}
	sb.WriteByte(']')
	return sb.String()
}

// CustomCallV2 emits a stablehlo.custom_call (API version 2 = STATUS_RETURNING) to the named
// target (e.g. "__cudnn$fmhaSoftmax").
//
//   - backendConfig: the raw backend_config string (serialized proto / JSON); "" omits it.
//   - operandLayouts/outputLayouts: minor-to-major dim orders, paired by index with
//     operands/outputShapes. A nil entry (or nil slice) defaults to row-major. Passing a nil
//     slice omits the layout attribute entirely.
//   - outputShapes: one shape per result (multi-output: e.g. attention output + scratch).
//
// Returns one output Value per outputShape, in order.
func CustomCallV2(
	target string,
	backendConfig string,
	operands []*Value,
	operandLayouts [][]int,
	outputShapes []shapes.Shape,
	outputLayouts [][]int,
) ([]*Value, error) {
	op := optypes.CustomCall
	if len(operands) == 0 {
		return nil, errors.Errorf("%s requires at least one operand", op)
	}
	if len(outputShapes) == 0 {
		return nil, errors.Errorf("%s requires at least one output shape", op)
	}
	fn, err := innerMostFunction(operands...)
	if err != nil {
		return nil, err
	}
	if fn.Returned {
		return nil, errors.Errorf("cannot add operation %s after returning, in function %q", op, fn.Name)
	}

	operandRanks := make([]int, len(operands))
	for i, v := range operands {
		operandRanks[i] = v.shape.Rank()
	}
	outputRanks := make([]int, len(outputShapes))
	for i, s := range outputShapes {
		outputRanks[i] = s.Rank()
	}

	stmt := fn.addMultiOp(op, outputShapes, operands)
	attrs := map[string]any{
		"call_target_name": target,
		"api_version":      int32(CustomCallAPIVersionStatusReturning),
	}
	if backendConfig != "" {
		attrs["backend_config"] = backendConfig
	}
	if s := renderLayouts(operandLayouts, operandRanks); s != "" {
		attrs["operand_layouts"] = literalStr(s)
	}
	if s := renderLayouts(outputLayouts, outputRanks); s != "" {
		attrs["result_layouts"] = literalStr(s)
	}
	stmt.Attributes = attrs
	return stmt.Outputs, nil
}
```

Add `"strconv"` and `"strings"` to the import block in `customcall.go`. Verify `(*Value).shape.Rank()` exists (the stablehlo `Value` carries a `shape shapes.Shape`; `shapes.Shape` has `Rank()`). If the field is named differently, adapt `operandRanks` accordingly — grep `git grep -n "shape" stablehlo/value.go`.

- [ ] **Step 4: Run the unit test to verify it passes**

```bash
cd /Users/guygrigsby/projects/forks/go-xla
go test ./stablehlo/ -run TestRenderLayouts -v
```

Expected: PASS (all four sub-tests).

- [ ] **Step 5: Rewire the `customCall` wrapper in `compute/xla/ops.go` onto `CustomCallV2`**

Replace the `func (f *Function) customCall(...)` body (ops.go:826) with the `[][]int`-layout form:

```go
// customCall emits a StableHLO custom_call (API version 2, status-returning). It is private:
// the only callers are the cuDNN flash fused ops in flash.go, so the cuDNN
// target/backend_config/layout mapping never leaves this package. Multi-output: one result
// Value per outputShapes entry. operandLayouts/outputLayouts are minor-to-major dim orders
// (nil entry/slice -> row-major); see stablehlo.CustomCallV2.
func (f *Function) customCall(target string, backendConfig string,
	operandLayouts [][]int, outputShapes []shapes.Shape, outputLayouts [][]int,
	operands ...compute.Value) ([]compute.Value, error) {
	if len(operands) == 0 {
		return nil, errors.New("customCall requires at least one operand")
	}
	if len(outputShapes) == 0 {
		return nil, errors.New("customCall requires at least one output shape")
	}
	nodes, err := f.verifyAndCastValues("customCall", operands...)
	if err != nil {
		return nil, err
	}
	operandValues := xslices.Map(nodes, func(n *Node) *stablehlo.Value { return n.value })
	outShapes := xslices.Map(outputShapes, func(s shapes.Shape) stablehloshapes.Shape {
		return stablehloshapes.Make(s.DType, s.Dimensions...)
	})
	values, err := stablehlo.CustomCallV2(target, backendConfig, operandValues,
		operandLayouts, outShapes, outputLayouts)
	if err != nil {
		return nil, err
	}
	outputNodes := xslices.Map(values, func(v *stablehlo.Value) compute.Value { return f.newNode(v) })
	return outputNodes, nil
}
```

Note: the `apiVersion` param is dropped (V2 is fixed). The `stablehloshapes.Make` conversion is unchanged.

- [ ] **Step 6: Adapt the two `flash.go` call sites minimally so the package compiles**

`flash.go` still has the string-layout constants from before. For now, convert them to `[][]int` inline at the two call sites so the build is green (Task 2 replaces these constants entirely). In `FusedScaledDotProductAttention`, replace the forward call:

```go
	// Forward operand layouts: q,k,v BSHD [3,2,1,0]. Result layouts: output BHSD [3,1,2,0],
	// stats [2,1,0], scratch u8 [0].
	fwdOperandLayouts := [][]int{{3, 2, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0}}
	fwdResultLayouts := [][]int{{3, 1, 2, 0}, {2, 1, 0}, {0}}
	outs, err := f.customCall(fmhaForwardTarget, flashFwdBackendConfig(b, h, s, scale),
		fwdOperandLayouts, []shapes.Shape{bhsd, stats, scratch}, fwdResultLayouts, q, k, v)
```

In `FusedScaledDotProductAttentionVJP`, replace the backward call:

```go
	// Backward operands: q,k,v BSHD, stats [2,1,0], dOutput BSHD, output BSHD.
	bwdOperandLayouts := [][]int{{3, 2, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0}, {2, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0}}
	// Backward results: dQ, dK, dV BHSD, scratch u8.
	bwdResultLayouts := [][]int{{3, 1, 2, 0}, {3, 1, 2, 0}, {3, 1, 2, 0}, {0}}
	grads, err := f.customCall(fmhaBackwardTarget, flashBwdBackendConfig(b, h, s, scale),
		bwdOperandLayouts, []shapes.Shape{bhsd, bhsd, bhsd, scratch}, bwdResultLayouts,
		q, k, v, softmaxStats, dOut, out)
```

Delete the now-unused string layout constants (`layoutBSHD`, `layoutBHSD`, `layoutStats`, `layoutU8`, `flashFwdOperandLayouts`, `flashFwdResultLayouts`, `flashBwdOperandLayouts`, `flashBwdResultLayouts`) and the `fmhaAPIVersion` const from `flash.go`.

- [ ] **Step 7: Build the whole module and run the CPU test suite**

```bash
cd /Users/guygrigsby/projects/forks/go-xla
go build ./...
go test ./stablehlo/ ./compute/... 2>&1 | tail -20
```

Expected: build exit 0; `stablehlo` and `compute/xla` tests pass (CUDA-gated tests skip on the Mac with no cuda plugin). The `pjrt/fmha_test.go` still references the old `stablehlo.CustomCall` — it will fail to compile here. That is expected and fixed in Task 5; for this task, scope the run to `./stablehlo/ ./compute/...` to confirm the rewire.

- [ ] **Step 8: Commit**

```bash
git add stablehlo/customcall.go stablehlo/customcall_test.go compute/xla/ops.go compute/xla/flash.go
git commit -m "compute/xla: typed CustomCallV2 with [][]int layouts, render MLIR internally"
```

---

## Task 2 [S1]: Rewire `flash.go` onto config-driven targets and backend_config — standard target only (Contract C scaffolding)

Restructure `flash.go` so the cuDNN target and `mask_type` come from the config rather than hardcoded constants. This task introduces the dispatch seam (functions + a variant struct) and routes the existing causal-bf16 path through it, with no behavior change yet. **S1 routes ONLY the standard `__cudnn$fmhaSoftmax`/`__cudnn$fmhaSoftmaxBackward` target**; `mask_type` derives from `causal` + seqlens. Task 4 relaxes `flashSupported` for seqlens onto this same seam. The bias branch that extends this selector is **Task 2b [S2]**.

**Cross-stage compile dependency (read before writing code):** the S1 `selectFMHAVariant` here reads **only** `cfg.QuerySeqLen`/`cfg.KeyValueSeqLen` (compute S1 fields) and the `causal` bool. It **must not** reference `cfg.Bias` — that field is not on the struct until compute Stage 2, so a reference would not compile against the Stage-1 compute fork. The `fmhaVariant` struct here is correspondingly minimal (no `hasBias`). Task 2b adds that read, field, and the variant target constants.

**Files:**
- Modify: `compute/xla/flash.go`
- Test: `compute/xla/flash_dispatch_test.go` (create) — CPU-testable pure-function tests for the selector and backend_config builder

**Interfaces:**
- Consumes: `*compute.ScaledDotProductAttentionConfig` (Contract A **S1 fields only**: `QuerySeqLen`, `KeyValueSeqLen`); `dtypes.BFloat16`, `dtypes.Float16`.
- Produces (S1 minimal forms; Task 2b extends them):
  ```go
  // compute/xla/flash.go
  type fmhaVariant struct {
      fwdTarget, bwdTarget string
      maskType             string // "CAUSAL" | "PADDING" | "PADDING_CAUSAL" | "NO_MASK"
      hasSeqLens           bool
      // [S2] adds: hasBias bool
  }
  // selectFMHAVariant maps dtype + (causal, seqlens) -> standard target/maskType, or ErrNotImplemented.
  // [S2] extends with the bias branch.
  func selectFMHAVariant(op string, qkvDType dtypes.DType, causal bool,
      cfg *compute.ScaledDotProductAttentionConfig) (fmhaVariant, error)
  // flashBackendConfigV builds the cudnn_fmha_backend_config for a variant.
  func flashBackendConfigV(b, h, s int, scale float64, dotDimNumbers string, v fmhaVariant) string
  ```

- [ ] **Step 1: Write the failing dispatch test (CPU-only, runs on Mac)**

`compute/xla/flash_dispatch_test.go`:

```go
package xla

import (
	"errors"
	"strings"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
)

func TestSelectFMHAVariant_StandardCausal(t *testing.T) {
	v, err := selectFMHAVariant("op", dtypes.BFloat16, true, nil)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if v.fwdTarget != "__cudnn$fmhaSoftmax" || v.bwdTarget != "__cudnn$fmhaSoftmaxBackward" {
		t.Errorf("targets = %q / %q", v.fwdTarget, v.bwdTarget)
	}
	if v.maskType != "CAUSAL" {
		t.Errorf("maskType = %q, want CAUSAL", v.maskType)
	}
}

func TestSelectFMHAVariant_NoMaskWhenNotCausal(t *testing.T) {
	v, err := selectFMHAVariant("op", dtypes.Float16, false, nil)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if v.maskType != "NO_MASK" {
		t.Errorf("maskType = %q, want NO_MASK", v.maskType)
	}
}

func TestSelectFMHAVariant_RejectsF32(t *testing.T) {
	_, err := selectFMHAVariant("op", dtypes.Float32, true, nil)
	if !errors.Is(err, compute.ErrNotImplemented) {
		t.Errorf("err = %v, want ErrNotImplemented", err)
	}
}

func TestFlashBackendConfigV_MaskTypeFromVariant(t *testing.T) {
	v := fmhaVariant{maskType: "NO_MASK"}
	cfg := flashBackendConfigV(2, 12, 2048, 0.125, `"x": 1`, v)
	if !strings.Contains(cfg, `"mask_type": "NO_MASK"`) {
		t.Errorf("backend_config missing NO_MASK mask_type:\n%s", cfg)
	}
	if !strings.Contains(cfg, `"dropout_rate": 0`) {
		t.Errorf("backend_config missing dropout_rate:\n%s", cfg)
	}
}
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /Users/guygrigsby/projects/forks/go-xla
go test ./compute/xla/ -run 'TestSelectFMHAVariant|TestFlashBackendConfigV' -v
```

Expected: FAIL — `undefined: selectFMHAVariant`, `undefined: fmhaVariant`, `undefined: flashBackendConfigV`.

- [ ] **Step 3: Add the variant struct, selector, and parameterized backend_config builder (S1 minimal)**

Add to `flash.go` (replace the `fmhaForwardTarget`/`fmhaBackwardTarget` const block with the standard-target pair, and add the new functions). **S1 defines only the standard `fmhaSoftmaxFwd`/`fmhaSoftmaxBwd` pair**; Task 2b [S2] adds the `fmhaScaleBias*` constants. The selector here reads only seqlens + causal — no `cfg.Bias` (that struct field does not exist in the Stage-1 compute fork).

```go
// cuDNN fused-attention custom-call targets. S1 wires the standard softmax pair only.
// [S2] (Task 2b) adds the fmhaScaleBias* target rows.
// FP8 targets (__cudnn$fmhaSoftmaxF8 / …BackwardF8) are intentionally NOT defined:
// fp8 fmha is paused (no local sm_8.9+ hardware to test). fp8 input dtype falls to
// ErrNotImplemented in selectFMHAVariant. Add that row when wiring fp8 on Hopper/Ada.
const (
	fmhaSoftmaxFwd = "__cudnn$fmhaSoftmax"
	fmhaSoftmaxBwd = "__cudnn$fmhaSoftmaxBackward"
)

// fmhaVariant captures the config-derived custom-call selection: the fwd/bwd targets, the
// backend_config mask_type, and the operand-set flags. Built by selectFMHAVariant.
// S1 fields only; [S2] (Task 2b) adds `hasBias bool`.
type fmhaVariant struct {
	fwdTarget, bwdTarget string
	maskType             string // "CAUSAL" | "PADDING" | "PADDING_CAUSAL" | "NO_MASK"
	hasSeqLens           bool
}

// selectFMHAVariant maps the q/k/v dtype and (causal, seqlens) to a cuDNN variant. S1 routes the
// standard softmax target only. Dtype gate: f16/bf16 only; anything else (incl. fp8 e4m3fn/e5m2 —
// paused, no local hardware) -> ErrNotImplemented, and the caller falls back to the decomposed path.
// mask_type derives from causal + seqlens: PADDING_CAUSAL (both), PADDING (seqlens only),
// CAUSAL (causal only), NO_MASK (neither).
//
// S1 reads ONLY cfg.QuerySeqLen/cfg.KeyValueSeqLen (the compute Stage-1 fields). It must not touch
// cfg.Bias — that lands in compute Stage 2 and is wired here by Task 2b [S2], which extends this
// function with the bias branch.
func selectFMHAVariant(op string, qkvDType dtypes.DType, causal bool,
	cfg *compute.ScaledDotProductAttentionConfig) (fmhaVariant, error) {
	var v fmhaVariant
	hasSeqLens := cfg != nil && cfg.QuerySeqLen != nil && cfg.KeyValueSeqLen != nil

	switch qkvDType {
	case dtypes.Float16, dtypes.BFloat16:
		v.fwdTarget, v.bwdTarget = fmhaSoftmaxFwd, fmhaSoftmaxBwd
	default:
		// fp8 (e4m3fn/e5m2) lands here too: paused, not wired. NotImplemented -> decomposed.
		return v, errors.Wrapf(compute.ErrNotImplemented,
			"%s: cuDNN fmha needs f16/bf16, got %s", op, qkvDType)
	}

	switch {
	case causal && hasSeqLens:
		v.maskType = "PADDING_CAUSAL"
	case hasSeqLens:
		v.maskType = "PADDING"
	case causal:
		v.maskType = "CAUSAL"
	default:
		v.maskType = "NO_MASK"
	}
	v.hasSeqLens = hasSeqLens
	return v, nil
}

// flashBackendConfigV builds a cudnn_fmha_backend_config for the given variant: mask_type comes
// from v, the score-matrix dims [B,H,S,S] and fmha_scale from the shape, dotDimNumbers carries the
// bmm dot_dimension_numbers JSON (the fwd/bwd-specific part). Dropout is cut, so dropout_rate is
// always the literal 0 (a fixed cuDNN backend_config field, not a variant).
func flashBackendConfigV(b, h, s int, scale float64, dotDimNumbers string, v fmhaVariant) string {
	return fmt.Sprintf(`{"operation_queue_id": "0", "cudnn_fmha_backend_config": {"algorithm": {"algo_id": "0", "math_type": "TENSOR_OP_MATH", "tuning_knobs": {"17": "1", "24": "0"}, "is_cudnn_frontend": true, "workspace_size": "0"}, "fmha_scale": %s, "intermediate_tensor_shape": {"element_type": "BF16", "dimensions": ["%d", "%d", "%d", "%d"], "tuple_shapes": [], "layout": {"dim_level_types": [], "dim_unique": [], "dim_ordered": [], "minor_to_major": ["3", "2", "1", "0"], "tiles": [], "element_size_in_bits": "0", "memory_space": "0", "index_primitive_type": "PRIMITIVE_TYPE_INVALID", "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID", "dynamic_shape_metadata_prefix_bytes": "0"}, "is_dynamic_dimension": [false, false, false, false]}, "is_flash_attention": true, "mask_type": "%s", %s, "dropout_rate": 0, "seed": 42, "sliding_window_length": 0, "max_seg_per_batch": 1, "is_paged_attention": false}}`,
		formatScale(scale), b, h, s, s, v.maskType, dotDimNumbers)
}
```

Now update the two `flashFwdBackendConfig`/`flashBwdBackendConfig` helpers to call `flashBackendConfigV` with a variant. Change their signatures to take the variant:

```go
func flashFwdBackendConfig(b, h, s int, scale float64, v fmhaVariant) string {
	return flashBackendConfigV(b, h, s, scale,
		`"bmm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["3"], "lhs_batch_dimensions": ["0", "2"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}`, v)
}

func flashBwdBackendConfig(b, h, s int, scale float64, v fmhaVariant) string {
	return flashBackendConfigV(b, h, s, scale,
		`"bmm1_grad_gemm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["2"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm1_grad_gemm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_grad_gemm1_dot_dimension_numbers": {"lhs_contracting_dimensions": ["2"], "rhs_contracting_dimensions": ["1"], "lhs_batch_dimensions": ["0", "1"], "rhs_batch_dimensions": ["0", "2"]}, "bmm2_grad_gemm2_dot_dimension_numbers": {"lhs_contracting_dimensions": ["3"], "rhs_contracting_dimensions": ["3"], "lhs_batch_dimensions": ["0", "2"], "rhs_batch_dimensions": ["0", "2"]}`, v)
}
```

Delete the old `flashBackendConfig` (now superseded by `flashBackendConfigV`).

- [ ] **Step 4: Route the two fused methods through `selectFMHAVariant`**

In `FusedScaledDotProductAttention`, after `flashSupported`, derive the variant from the original query dtype (before the bf16 cast) and use its targets/config. Capture the query dtype first:

```go
	const op = "FusedScaledDotProductAttention"
	if err = f.flashSupported(op, mask, numHeads, numKVHeads, axesLayout, causal, options); err != nil {
		return nil, nil, err
	}
	qDType, err := f.dtypeOf(op, query)
	if err != nil {
		return nil, nil, err
	}
	variant, err := selectFMHAVariant(op, qDType, causal, options)
	if err != nil {
		return nil, nil, err
	}
	b, s, h, d, err := f.bshdDims(op, query)
	// ... bf16 casts unchanged ...
	outs, err := f.customCall(variant.fwdTarget, flashFwdBackendConfig(b, h, s, scale, variant),
		fwdOperandLayouts, []shapes.Shape{bhsd, stats, scratch}, fwdResultLayouts, q, k, v)
```

Add a `dtypeOf` helper to `flash.go`:

```go
// dtypeOf returns the element dtype of a value (used to pick the cuDNN fmha variant before
// any bf16 cast).
func (f *Function) dtypeOf(op string, v compute.Value) (dtypes.DType, error) {
	nodes, err := f.verifyAndCastValues(op, v)
	if err != nil {
		return 0, err
	}
	return nodes[0].shape.DType, nil
}
```

Mirror in `FusedScaledDotProductAttentionVJP`: derive `variant` the same way and pass `variant.bwdTarget` + `flashBwdBackendConfig(b, h, s, scale, variant)`.

Note `flashSupported` now takes `options` (added in Task 4). For this task, add the trailing `options *compute.ScaledDotProductAttentionConfig` param to `flashSupported`'s signature and ignore it (Task 4 uses it). Update its body's signature line only:

```go
func (f *Function) flashSupported(op string, mask compute.Value, numHeads, numKVHeads int, axesLayout compute.AxesLayout, causal bool, options *compute.ScaledDotProductAttentionConfig) error {
```

- [ ] **Step 5: Run the dispatch tests and the package build**

```bash
cd /Users/guygrigsby/projects/forks/go-xla
go test ./compute/xla/ -run 'TestSelectFMHAVariant|TestFlashBackendConfigV' -v
go build ./compute/...
```

Expected: PASS; build exit 0. (`pjrt/fmha_test.go` still broken — fixed in Task 5.)

- [ ] **Step 6: Commit**

```bash
git add compute/xla/flash.go compute/xla/flash_dispatch_test.go
git commit -m "compute/xla: config-driven fmha target + mask_type selection seam"
```

---

## Task 2b [S2]: Extend `selectFMHAVariant` with the bias target branch (Contract C)

**Stage 2.** Now that compute Stage 2 has added `Bias` to `ScaledDotProductAttentionConfig`, extend the S1 selector (Task 2) with the bias target branch and its target constants. This is a pure addition onto the S1 seam — the standard-target path and the `mask_type` derivation are unchanged. The bias operand set lands in Task 3b [S2]; this task only picks the target and threads `hasBias` into the variant.

**Cross-stage note:** this task is the *only* place `cfg.Bias` may be read. It does not compile against the Stage-1 compute fork (that field is absent there) — it is gated behind compute Stage 2 by the dependency order.

**Files:**
- Modify: `compute/xla/flash.go` (the target const block, `fmhaVariant`, `selectFMHAVariant`)
- Test: `compute/xla/flash_dispatch_test.go` (extend) — CPU dispatch-unit test for the new bias branch

**Interfaces:**
- Consumes: `*compute.ScaledDotProductAttentionConfig` **S2 field** (`Bias`); the S1 `fmhaVariant`/`selectFMHAVariant` (Task 2).
- Produces: extended `fmhaVariant` (adds `hasBias`), extended `selectFMHAVariant` (bias branch), the `fmhaScaleBias*` target constants.

- [ ] **Step 1: Write the failing dispatch tests for the new precedence (CPU, runs on Mac)**

Add to `compute/xla/flash_dispatch_test.go`:

```go
func TestSelectFMHAVariant_Bias(t *testing.T) {
	cfg := &compute.ScaledDotProductAttentionConfig{Bias: dummyValue(t)}
	v, err := selectFMHAVariant("op", dtypes.BFloat16, true, cfg)
	if err != nil {
		t.Fatalf("unexpected err: %v", err)
	}
	if v.fwdTarget != "__cudnn$fmhaScaleBiasSoftmax" || v.bwdTarget != "__cudnn$fmhaScaleBiasSoftmaxBackward" {
		t.Errorf("targets = %q / %q", v.fwdTarget, v.bwdTarget)
	}
	if !v.hasBias {
		t.Error("hasBias = false, want true")
	}
}
```

`dummyValue(t)` returns any non-nil `compute.Value` for the bias-set check (the selector only tests `cfg.Bias != nil`); a tiny graph constant on a CPU backend suffices — grep the package's other CPU `_test.go` files for the established way to mint a `compute.Value` and reuse it rather than inventing one.

- [ ] **Step 2: Run to verify it fails**

```bash
cd /Users/guygrigsby/projects/forks/go-xla
go test ./compute/xla/ -run 'TestSelectFMHAVariant_Bias' -v
```

Expected: FAIL — the bias branch and the `fmhaScaleBias*` constants do not exist yet; `fmhaVariant` has no `hasBias` field.

- [ ] **Step 3: Add the variant target constants, extend the struct and the selector**

Extend the target const block (add to the S1 pair):

```go
const (
	fmhaSoftmaxFwd = "__cudnn$fmhaSoftmax"
	fmhaSoftmaxBwd = "__cudnn$fmhaSoftmaxBackward"
	// [S2] bias variant.
	fmhaScaleBiasSoftmaxFwd = "__cudnn$fmhaScaleBiasSoftmax"
	fmhaScaleBiasSoftmaxBwd = "__cudnn$fmhaScaleBiasSoftmaxBackward"
)
```

Extend `fmhaVariant` with the S2 fields:

```go
type fmhaVariant struct {
	fwdTarget, bwdTarget string
	maskType             string // "CAUSAL" | "PADDING" | "PADDING_CAUSAL" | "NO_MASK"
	hasSeqLens           bool
	hasBias              bool // [S2]
}
```

Replace the dtype switch's bf16/f16 arm in `selectFMHAVariant` with the bias branch (the fp8/default `ErrNotImplemented` arm and the `mask_type` block stay exactly as S1 left them):

```go
	hasBias := cfg != nil && cfg.Bias != nil

	switch qkvDType {
	case dtypes.Float16, dtypes.BFloat16:
		// Contract C precedence: bias > standard.
		switch {
		case hasBias:
			v.fwdTarget, v.bwdTarget = fmhaScaleBiasSoftmaxFwd, fmhaScaleBiasSoftmaxBwd
		default:
			v.fwdTarget, v.bwdTarget = fmhaSoftmaxFwd, fmhaSoftmaxBwd
		}
	default:
		return v, errors.Wrapf(compute.ErrNotImplemented,
			"%s: cuDNN fmha needs f16/bf16, got %s", op, qkvDType)
	}
```

And before `return v, nil`, set the new field:

```go
	v.hasBias = hasBias
```

- [ ] **Step 4: Run the dispatch tests and build**

```bash
cd /Users/guygrigsby/projects/forks/go-xla
go test ./compute/xla/ -run 'TestSelectFMHAVariant|TestFlashBackendConfigV' -v
go build ./compute/...
```

Expected: PASS (S1 + S2 dispatch tests); build exit 0.

- [ ] **Step 5: Commit**

```bash
git add compute/xla/flash.go compute/xla/flash_dispatch_test.go
git commit -m "compute/xla: extend fmha selector with bias variant target"
```

---

## Task 3 [S1]: Seqlen operand set + fp8-paused dtype gate (Contract C)

Build the standard-variant operand list: q/k/v always, plus the query/kv seqlen operands when the variant carries them. The bias operand set is **Task 3b [S2]**. The dtype gate already lives in `selectFMHAVariant` (Task 2); this task adds the seqlen operands + the CPU fp8-paused unit test that pins the gate's default branch (present from S1).

**Cross-stage note:** the S1 operand builder branches only on `variant.hasSeqLens` (an S1 struct field). It must not reference `variant.hasBias` — that field is added by Task 2b [S2], and `options.Bias` does not exist on the Stage-1 compute config. Task 3b adds that branch.

**Files:**
- Modify: `compute/xla/flash.go`
- Test: `compute/xla/flash_dispatch_test.go` (the CPU dispatch-unit-test file from Task 2) — add the fp8-paused `TestSelectFMHAVariant_FP8NotImplemented` here (CPU, no cuda)

**Interfaces:**
- Consumes: `fmhaVariant.hasSeqLens` (Task 2); `cfg.QuerySeqLen`, `cfg.KeyValueSeqLen` (Contract A S1); `f.customCall` ([][]int layouts, Task 1).
- Produces: forward/backward operand builders inside the two fused methods (no new exported names; the variant struct drives them).

- [ ] **Step 1: Pin the fp8-paused seam (CPU unit test, runs on Mac)**

In `compute/xla/flash_dispatch_test.go` (CPU file, Mac-runnable), pin the fp8-paused seam. This exercises the S1 dtype gate's `default` branch — no bias field needed, so it is fully [S1]:

```go
// fp8 is paused (no local sm_8.9+ hardware). Pin the seam: an fp8 dtype must select
// no variant and surface ErrNotImplemented so the caller falls back to decomposed.
// Pure unit test on selectFMHAVariant — CPU, runs on the Mac (no [cuda]).
func TestSelectFMHAVariant_FP8NotImplemented(t *testing.T) {
	_, err := selectFMHAVariant("fmha", dtypes.F8E4M3FN, true, nil)
	require.True(t, compute.IsNotImplemented(err), "fp8 must be NotImplemented (paused), got %v", err)
	_, err = selectFMHAVariant("fmha", dtypes.F8E5M2, true, nil)
	require.True(t, compute.IsNotImplemented(err), "fp8 e5m2 must be NotImplemented (paused), got %v", err)
}
```

Run it (Mac):

```bash
cd /Users/guygrigsby/projects/forks/go-xla
go test ./compute/xla/ -run TestSelectFMHAVariant_FP8NotImplemented -v
```

Expected: PASS (the S1 selector already returns `ErrNotImplemented` for non-half dtypes).

- [ ] **Step 2: Build the seqlen operand set in the forward method**

In `FusedScaledDotProductAttention`, after casting q/k/v, assemble operands and matching layouts. **S1: only the seqlen branch** (the bias branch is added by Task 3b):

```go
	// Operand order cuDNN expects: q, k, v, [seqQ, seqKV]. [S2] inserts [bias] before seqlens.
	operands := []compute.Value{q, k, v}
	operandLayouts := [][]int{{3, 2, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0}}
	if variant.hasSeqLens {
		operands = append(operands, options.QuerySeqLen, options.KeyValueSeqLen)
		operandLayouts = append(operandLayouts, nil, nil) // int32 [B], row-major
	}
	fwdResultLayouts := [][]int{{3, 1, 2, 0}, {2, 1, 0}, {0}}
	outs, err := f.customCall(variant.fwdTarget, flashFwdBackendConfig(b, h, s, scale, variant),
		operandLayouts, []shapes.Shape{bhsd, stats, scratch}, fwdResultLayouts, operands...)
```

All wired variants are bf16, so q/k/v always cast to bf16 (fp8 is paused — `selectFMHAVariant`
already returned `ErrNotImplemented` before reaching here, so this method only runs for half dtypes):

```go
	var q, k, v2 compute.Value
	if q, err = f.bf16(query); err != nil { return nil, nil, err }
	if k, err = f.bf16(key); err != nil { return nil, nil, err }
	if v2, err = f.bf16(value); err != nil { return nil, nil, err }
```

(rename the local `v` for value to `v2` to avoid shadowing `variant`'s short name if present; keep names consistent with the existing code's `q, k, v`).

- [ ] **Step 3: Mirror the seqlen operand set in the backward (VJP) method**

In `FusedScaledDotProductAttentionVJP`, append the seqlen operands after the fixed `q, k, v, softmaxStats, dOut, out`:

```go
	operands := []compute.Value{q, k, v, softmaxStats, dOut, out}
	operandLayouts := [][]int{{3, 2, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0}, {2, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0}}
	if variant.hasSeqLens {
		operands = append(operands, options.QuerySeqLen, options.KeyValueSeqLen)
		operandLayouts = append(operandLayouts, nil, nil)
	}
	bwdResultLayouts := [][]int{{3, 1, 2, 0}, {3, 1, 2, 0}, {3, 1, 2, 0}, {0}}
	grads, err := f.customCall(variant.bwdTarget, flashBwdBackendConfig(b, h, s, scale, variant),
		operandLayouts, []shapes.Shape{bhsd, bhsd, bhsd, scratch}, bwdResultLayouts, operands...)
```

- [ ] **Step 4: Confirm the Mac CPU build/test still green**

```bash
cd /Users/guygrigsby/projects/forks/go-xla
go build ./... # pjrt still references old CustomCall; expect only that failure
go vet ./compute/xla/
go test ./compute/xla/ -run 'TestSelectFMHAVariant|TestFlashBackendConfigV' -v
```

Expected: `compute/xla` builds and its CPU tests pass; `pjrt` fails only on `CustomCall` (Task 5). The seqlen execution test runs in Task 4 [cuda] after the gate is relaxed.

- [ ] **Step 5: Commit**

```bash
git add compute/xla/flash.go compute/xla/flash_dispatch_test.go
git commit -m "compute/xla: seqlen fmha operand set; fp8 paused as NotImplemented"
```

---

## Task 3b [S2]: Bias operand set + [cuda] variant exec test (Contract C)

**Stage 2.** Add the bias operand branch onto the S1 operand builder (Task 3), and the `[cuda]` bias-variant execution test. Bias adds a `[B,H,S,Skv]` operand before the seqlens. Depends on compute Stage 2 (`options.Bias`) and the extended `fmhaVariant` (`hasBias`) from Task 2b.

**Execution-order note:** this S2 task is numbered before Task 4 in the document for narrative grouping, but stages gate execution — all of Stage 1 (Tasks 0,1,2,3,4,5) lands and is reviewed first, so Task 4's shared `[cuda]` harness already exists when this task runs.

**Files:**
- Modify: `compute/xla/flash.go` (insert the bias branch into both operand builders)
- Test: `compute/xla/flash_variants_cuda_test.go` (create) — **[cuda]** execution test for the bias variant

**Interfaces:**
- Consumes: `fmhaVariant.hasBias` (Task 2b); `cfg.Bias` (Contract A S2); `f.customCall` (Task 1).
- Produces: the bias branch inside the two fused-method operand builders.

- [ ] **Step 1: Write the [cuda] variant execution tests**

`compute/xla/flash_variants_cuda_test.go` (no `//go:build` tag needed — it self-skips on non-cuda via `getFusionBackend`; each test is **[cuda]**):

```go
package xla

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
)

// [cuda] runs under GOMLX_BACKEND=xla:cuda. Each builds a tiny fused-attention graph for one
// variant and asserts it executes (output finite, right shape). q=k=v=ones makes the softmax
// uniform so the bias-free output is all ~1.0; with bias the output stays finite.
func TestFMHAVariant_Bias_cuda(t *testing.T) {
	be := getFusionBackend(t) // helper from Task 3; skips if backend can't fuse bf16 causal
	cfg := &compute.ScaledDotProductAttentionConfig{Bias: makeBias(t, be)}
	out := runFusedFwd(t, be, dtypes.BFloat16, true, cfg)
	assertFiniteBSHD(t, out)
}
```

Reuse the shared `[cuda]` harness introduced in Task 4 (`getFusionBackend`, `runFusedFwd`, `assertFiniteBSHD` in `flash_seqlen_cuda_test.go`); add only the bias-specific helper here: `makeBias` (a `[B,H,S,Skv]` bf16 node). Same package, so the Task 4 helpers are visible without redeclaration.

- [ ] **Step 2: Run on the CUDA host to verify they fail [cuda]**

On the CUDA host, branch synced, in the go-xla checkout:

```bash
env GOMLX_BACKEND=xla:cuda go test ./compute/xla/ -run 'TestFMHAVariant_.*_cuda' -v
```

Expected: FAIL — bias variant emits the bias target but passes no bias operand (cuDNN rejects operand count), or the helpers reference operand-building not yet present. Capture the actual failure.

- [ ] **Step 3: Insert the bias branch into the forward operand builder**

In `FusedScaledDotProductAttention`, extend the Task 3 [S1] operand builder. Bias goes *before* the seqlens (the cuDNN operand order is q,k,v,[bias],[seqQ,seqKV]):

```go
	operands := []compute.Value{q, k, v}
	operandLayouts := [][]int{{3, 2, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0}}
	if variant.hasBias { // [S2]
		bias, err := f.bf16(options.Bias)
		if err != nil {
			return nil, nil, err
		}
		operands = append(operands, bias)
		operandLayouts = append(operandLayouts, nil) // bias rank-4 [B,H,S,Skv], row-major
	}
	if variant.hasSeqLens {
		operands = append(operands, options.QuerySeqLen, options.KeyValueSeqLen)
		operandLayouts = append(operandLayouts, nil, nil) // int32 [B], row-major
	}
```

(the `f.customCall` call and `fwdResultLayouts` are unchanged from Task 3.)

- [ ] **Step 4: Mirror the bias branch in the backward (VJP) operand builder**

In `FusedScaledDotProductAttentionVJP`, extend the Task 3 [S1] backward builder the same way:

```go
	operands := []compute.Value{q, k, v, softmaxStats, dOut, out}
	operandLayouts := [][]int{{3, 2, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0}, {2, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0}}
	if variant.hasBias { // [S2]
		bias, err := f.bf16(options.Bias)
		if err != nil { return nil, nil, nil, err }
		operands = append(operands, bias)
		operandLayouts = append(operandLayouts, nil)
	}
	if variant.hasSeqLens {
		operands = append(operands, options.QuerySeqLen, options.KeyValueSeqLen)
		operandLayouts = append(operandLayouts, nil, nil)
	}
```

- [ ] **Step 5: Run the variant tests on the CUDA host to verify they pass [cuda]**

```bash
env GOMLX_BACKEND=xla:cuda go test ./compute/xla/ -run 'TestFMHAVariant_.*_cuda' -v
```

Expected: PASS for the bias variant on the CUDA host GPU. Document that it ran green.

- [ ] **Step 6: Confirm the Mac CPU build/test still green**

```bash
cd /Users/guygrigsby/projects/forks/go-xla
go build ./... # pjrt still references old CustomCall until Task 5
go vet ./compute/xla/
```

Expected: `compute/xla` builds; `pjrt` fails only on `CustomCall` (Task 5).

- [ ] **Step 7: Commit**

```bash
git add compute/xla/flash.go compute/xla/flash_variants_cuda_test.go
git commit -m "compute/xla: bias fmha operand set"
```

---

## Task 4 [S1]: Seqlen padding masks — relax `flashSupported` (Contract C)

Allow seqlens through `flashSupported` and let the variant's `mask_type` (`PADDING` / `PADDING_CAUSAL`) drive masking. The forward/backward operand append for seqlens already landed in Task 3; this task is the gate relaxation plus the seqlen execution test.

**Files:**
- Modify: `compute/xla/flash.go` (`flashSupported`)
- Test: `compute/xla/flash_seqlen_cuda_test.go` (create) — **[cuda]**

**Interfaces:**
- Consumes: `cfg.QuerySeqLen`, `cfg.KeyValueSeqLen` (Contract A); `fmhaVariant.maskType` (Task 2).
- Produces: relaxed `flashSupported` that accepts seqlens; no new exported names.

- [ ] **Step 1: Write the [cuda] seqlen test**

`compute/xla/flash_seqlen_cuda_test.go`:

```go
package xla

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
)

// [cuda] PADDING_CAUSAL: per-batch lengths shorter than S must mask the padding rows. With
// q=k=v=ones, masking changes which keys contribute, so the masked output differs from the
// unmasked all-ones output for the shortened batch element. Runs under xla:cuda.
func TestFMHA_SeqLenPaddingCausal_cuda(t *testing.T) {
	be := getFusionBackend(t)
	const B, S = 2, 8
	cfg := &compute.ScaledDotProductAttentionConfig{
		QuerySeqLen:    makeI32Vec(t, be, []int32{S, S / 2}), // batch 1 padded
		KeyValueSeqLen: makeI32Vec(t, be, []int32{S, S / 2}),
	}
	out := runFusedFwdShaped(t, be, dtypes.BFloat16, true, cfg, B, S)
	assertFiniteBSHD(t, out)
}
```

**This [S1] task introduces the shared `[cuda]` test harness** (`getFusionBackend`, `runFusedFwd`, `runFusedFwdShaped`, `makeI32Vec`, `assertFiniteBSHD`) in `flash_seqlen_cuda_test.go` — it is the first `[cuda]` execution test in the S1 stage. `getFusionBackend` probes `BackendFusedScaledDotProductAttention` on a tiny causal bf16 input and `t.Skip`s if it returns `compute.ErrNotImplemented` (mirrors the contract's capability check). Build the graph through the public `compute.Backend`/`compute.Function` API the package already uses in its other `_cuda_test.go` files — grep `git grep -l "GOMLX_BACKEND\|NewBackend" compute/xla/*_test.go` for the established harness and reuse it rather than inventing one. Task 3b [S2] reuses these helpers and adds only the bias-specific one (`makeBias`).

- [ ] **Step 2: Run on the CUDA host to verify it fails [cuda]**

```bash
env GOMLX_BACKEND=xla:cuda go test ./compute/xla/ -run 'TestFMHA_SeqLen.*_cuda' -v
```

Expected: FAIL — `flashSupported` returns `ErrNotImplemented` because seqlens are set (current gate only allows nil mask + no extra config), so the fused path is never taken / the test's fusion probe rejects it.

- [ ] **Step 3: Relax `flashSupported` to allow seqlens**

The gate must still reject an explicit materialized `mask` (seqlens replace it) and still require BSHD + equal heads + cuda. Seqlens are now allowed. Replace the body:

```go
// flashSupported reports whether the cuDNN flash path can serve this call. cuDNN fMHA here is
// f16/bf16 (fp8 paused), BSHD-layout, equal-head, on a cuda plugin. Causality and
// per-batch seqlen padding are supported (mask_type derives from them in selectFMHAVariant);
// an explicit materialized mask is not (use seqlens instead). Anything else -> ErrNotImplemented.
func (f *Function) flashSupported(op string, mask compute.Value, numHeads, numKVHeads int, axesLayout compute.AxesLayout, causal bool, options *compute.ScaledDotProductAttentionConfig) error {
	if !f.builder.backend.plugin.IsCUDA() {
		return errors.Wrapf(compute.ErrNotImplemented, "%s: cuDNN flash needs the cuda plugin, have %q", op, f.builder.backend.pluginName)
	}
	if mask != nil {
		return errors.Wrapf(compute.ErrNotImplemented,
			"%s: cuDNN flash path takes seqlens, not a materialized mask", op)
	}
	if axesLayout != compute.AxesLayoutBSHD || numKVHeads != numHeads {
		return errors.Wrapf(compute.ErrNotImplemented,
			"%s: cuDNN flash path supports BSHD layout, equal q/kv heads only (got layout=%v heads=%d/%d)",
			op, axesLayout, numHeads, numKVHeads)
	}
	// One of QuerySeqLen/KeyValueSeqLen set without the other is ambiguous.
	if options != nil && (options.QuerySeqLen != nil) != (options.KeyValueSeqLen != nil) {
		return errors.Wrapf(compute.ErrNotImplemented,
			"%s: cuDNN flash padding mask needs both QuerySeqLen and KeyValueSeqLen", op)
	}
	return nil
}
```

(The dtype/variant gate stays in `selectFMHAVariant`; `flashSupported` no longer rejects non-causal, since `NO_MASK` is a valid variant.)

- [ ] **Step 4: Run on the CUDA host to verify it passes [cuda]**

```bash
env GOMLX_BACKEND=xla:cuda go test ./compute/xla/ -run 'TestFMHA_SeqLen.*_cuda' -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add compute/xla/flash.go compute/xla/flash_seqlen_cuda_test.go
git commit -m "compute/xla: allow seqlen padding masks in cuDNN flash path"
```

---

## Task 5 [S1]: Migrate `pjrt/fmha_test.go` to `CustomCallV2` and `IsCUDA`

The existing fmha execution test still calls the removed `stablehlo.CustomCall` and gates on `*FlagPluginName != "cuda"`. Update both: call `CustomCallV2` with `[][]int` layouts, and gate on `client.Plugin().IsCUDA()`.

**Files:**
- Modify: `pjrt/fmha_test.go`

**Interfaces:**
- Consumes: `stablehlo.CustomCallV2` (Task 1); `(*Client).Plugin() *Plugin` (clients.go:212); `(*Plugin).IsCUDA() bool` (plugins.go:154).

- [ ] **Step 1: Run the test as-is to confirm it fails to compile**

```bash
cd /Users/guygrigsby/projects/forks/go-xla
go test ./pjrt/ -run TestFMHAForwardExecute 2>&1 | tail -10
```

Expected: build FAIL — `undefined: stablehlo.CustomCall`.

- [ ] **Step 2: Rewrite the custom-call site and the gate**

In `pjrt/fmha_test.go`, replace the string layout constants and the `CustomCall` invocation. Old:

```go
	const layIn = "[dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>, dense<[3, 2, 1, 0]> : tensor<4xindex>]"
	const layOut = "[dense<[3, 1, 2, 0]> : tensor<4xindex>, dense<0> : tensor<1xindex>]"
	...
	res := must1(stablehlo.CustomCall("__cudnn$fmhaSoftmax", stablehlo.CustomCallAPIVersionStatusReturning,
		fmhaFwdBackendConfig, layIn, layOut, []shapes.Shape{out, scratch}, q, k, v))
```

New:

```go
	layIn := [][]int{{3, 2, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0}}
	layOut := [][]int{{3, 1, 2, 0}, {0}}
	...
	res := must1(stablehlo.CustomCallV2("__cudnn$fmhaSoftmax", fmhaFwdBackendConfig,
		[]*stablehlo.Value{q, k, v}, layIn, []shapes.Shape{out, scratch}, layOut))
```

Replace the gate (note: the gate must be evaluated after the client exists, since `IsCUDA` is on the plugin behind the client). Move the skip to after `getPJRTClient`:

```go
func TestFMHAForwardExecute(t *testing.T) {
	client := getPJRTClient(t)
	if !client.Plugin().IsCUDA() {
		t.Skipf("fmha is cuDNN-only; run with -plugin cuda (have %q)", *FlagPluginName)
	}
	...
}
```

(Confirm the `must1` types still line up: `CustomCallV2` returns `([]*stablehlo.Value, error)`, same as `CustomCall` did.)

- [ ] **Step 3: Verify it compiles and skips cleanly on the Mac**

```bash
cd /Users/guygrigsby/projects/forks/go-xla
go test ./pjrt/ -run TestFMHAForwardExecute -v
```

Expected: SKIP with "fmha is cuDNN-only" (Mac has no cuda plugin). The key result is that it compiles and skips, not fails.

- [ ] **Step 4: Run on the CUDA host to verify the execution path passes [cuda]**

```bash
# NOTE: the pjrt fmha tests gate on a -plugin flag (FlagPluginName), NOT GOMLX_BACKEND.
# The flag must come after -args or go test rejects it; the env-var form silently SKIPs.
go test ./pjrt/ -run TestFMHAForwardExecute -v -args -plugin cuda
```

Expected: PASS (output all ~1.0, per the test's invariant). VALIDATED on RTX 3070 Ti (sm_8.6, cuDNN 9.23).

- [ ] **Step 5: Commit**

```bash
git add pjrt/fmha_test.go
git commit -m "compute/xla: migrate fmha pjrt test to CustomCallV2 and IsCUDA gate"
```

---

## Verification Gate (Plan 02, from the contract)

Two gates: run the **Stage 1 gate** before declaring S1 complete (review checkpoint), then the **Stage 2 gate** before declaring S2 complete. Each stage is independently green.

### Stage 1 gate (Tasks 0,1,2,3,4,5)

- [ ] **Mac (CPU) — full non-CUDA suite green [S1]:**

```bash
cd /Users/guygrigsby/projects/forks/go-xla
go build ./...
go test ./stablehlo/ ./compute/... ./pjrt/ 2>&1 | tail -30
```

Expected: build exit 0 against the **Stage-1 compute fork** (config has `QuerySeqLen`/`KeyValueSeqLen`, not bias); all packages pass. CUDA-gated tests (`TestFMHAForwardExecute`, `TestFMHA_SeqLen*_cuda`) SKIP on the Mac; the `CustomCallV2` rendering unit test (`TestRenderLayouts`), the S1 dispatch tests (`TestSelectFMHAVariant_StandardCausal`/`_NoMaskWhenNotCausal`/`_RejectsF32`/`_FP8NotImplemented`, `TestFlashBackendConfigV_MaskTypeFromVariant`) PASS. Verify the S1 `selectFMHAVariant` references no S2 fields: `git grep -n 'cfg.Bias' compute/xla/flash.go` must return nothing.

- [ ] **the CUDA host (`xla:cuda`) — S1 matrix green [cuda][S1]:**

```bash
# compute/xla selects cuda via GOMLX_BACKEND; pjrt selects it via -plugin (after -args). Run separately:
env GOMLX_BACKEND=xla:cuda go test ./compute/xla -run 'FMHA_SeqLen' -v
go test ./pjrt -run 'TestFMHA(Forward|Backward)Execute' -v -args -plugin cuda
```

Expected: PASS for standard causal (`TestFMHAForwardExecute`/`BackwardExecute`) and seqlen padding-causal (`TestFMHA_SeqLenPaddingCausal_cuda`). No bias variant in the S1 matrix. VALIDATED on RTX 3070 Ti.

### Stage 2 gate (Tasks 2b, 3b)

- [ ] **Mac (CPU) — S2 dispatch tests green [S2]:**

```bash
cd /Users/guygrigsby/projects/forks/go-xla
go build ./...
go test ./compute/xla/ -run 'TestSelectFMHAVariant|TestFlashBackendConfigV' -v
```

Expected: build exit 0 against the **Stage-2 compute fork** (config now carries `Bias`); the S1 + S2 dispatch tests (`_Bias`) PASS.

- [ ] **the CUDA host (`xla:cuda`) — full variant matrix green [cuda][S2]:**

```bash
# compute/xla via GOMLX_BACKEND; pjrt via -plugin (after -args). Run separately:
env GOMLX_BACKEND=xla:cuda go test ./compute/xla -run 'FMHA|Flash' -v
go test ./pjrt -run 'FMHA' -v -args -plugin cuda
```

Expected: PASS for each wired variant (standard causal, bias, seqlen padding-causal). Variants unsupported by the installed cuDNN/card `t.Skip` rather than fail — note which ran green. (fp8 is not in this matrix — paused.)

- [ ] **Downstream integration unblocked:** confirm `compute/xla` exports nothing CUDA-specific above the backend (the only public surface change is `stablehlo.CustomCallV2`; `flash.go`'s methods keep their Contract-A signatures). Plan 03 (gomlx) consumes this via the local `replace` — its S1 work can start after the Stage 1 gate, without waiting on S2.

---

## Self-Review

**Stage tagging:** every task carries [S1] or [S2]. S1: Task 0 (sync), 1 (`CustomCallV2`), 2 (selector seam, standard target only), 3 (seqlen operands + fp8 CPU test), 4 (`flashSupported` relax + seqlen [cuda] test), 5 (`IsCUDA` test). S2: Task 2b (bias selector branch + target consts), 3b (bias operand + [cuda] exec test). Matches the contract's decided split exactly: S1 = strict API refactor (no bias variant), S2 = the bias variant.

**Cross-stage compile dependency (the load-bearing constraint):** the S1 `selectFMHAVariant` (Task 2) reads only `cfg.QuerySeqLen`/`cfg.KeyValueSeqLen` + `causal`, and the S1 `fmhaVariant` struct has only `fwdTarget`/`bwdTarget`/`maskType`/`hasSeqLens`. It references **none** of `cfg.Bias` — that field is absent from the Stage-1 compute config (Contract A is staged), so any reference would fail to compile. Task 2b [S2] is the single place that adds that read, the `hasBias` struct field, and the variant target constants. Both versions of `selectFMHAVariant` are shown in full (Task 2 minimal, Task 2b extended); `flashBackendConfigV` hardcodes `dropout_rate: 0` (dropout is cut, so it stays a fixed cuDNN backend_config field). The Stage-1 gate greps `flash.go` to prove no S2 field leaked into S1.

**Spec coverage** (against contract Contracts B, C, the SCOPE list, and the Staging section):
- Contract B `CustomCallV2` + internal MLIR rendering + remove old `CustomCall` + rewire `ops.go`/`flash.go` → Task 1 [S1]. CPU layout-rendering unit test → Task 1 Step 1.
- Contract C dtype gate (f16/bf16 only, else ErrNotImplemented incl. paused fp8) → Task 2 [S1], pinned by `TestSelectFMHAVariant_FP8NotImplemented` (Task 3 [S1], CPU — it is just the S1 default branch).
- Contract C `mask_type` CAUSAL/PADDING/PADDING_CAUSAL/NO_MASK from causal+seqlens → Task 2 [S1]; seqlen operands → Task 3 [S1]; gate relaxation + [cuda] seqlen exec → Task 4 [S1].
- Contract C variant dispatch precedence (Bias / Softmax) → Task 2b [S2] (extends `selectFMHAVariant`). Per-variant operand set (bias operand) → Task 3b [S2] (operands). VJP mirrors operands → Task 3 Step 3 [S1] (seqlen) + Task 3b Step 4 [S2] (bias).
- Test fix `client.Plugin().IsCUDA()` → Task 5 [S1].
- Task 0 sync is first and non-skippable; its Step 4 adds a stage check on the integrated compute config.
- [cuda] markers present on every cuDNN-execution test; layout-rendering and dispatch tests are CPU-only on the Mac. The shared [cuda] harness is introduced in Task 4 [S1] (first S1 [cuda] test); Task 3b [S2] reuses it and adds only `makeBias`.

**Placeholder scan:** no TBD/TODO; every code step shows full code. The [cuda] test-helper harness defers to the package's existing `_cuda_test.go` harness (grep instruction given) rather than inventing one — this is the one place exact code depends on the synced tree's existing helpers; flagged explicitly so the implementer reuses, not guesses.

**Type consistency:** `fmhaVariant`, `selectFMHAVariant`, `flashBackendConfigV`, `flashFwdBackendConfig(…, v fmhaVariant)`, `flashBwdBackendConfig(…, v fmhaVariant)`, `dtypeOf`, `renderLayouts`, `CustomCallV2`, and the `customCall` wrapper signature (dropped `apiVersion`, `[][]int` layouts) are used identically across the tasks. `fmhaVariant`/`selectFMHAVariant`/`flashBackendConfigV` deliberately have a smaller S1 shape that S2 widens — the only place a type grows between stages, called out above. `flashSupported` gains the trailing `options` param in Task 2 [S1] and uses it in Task 4 [S1] — consistent. dtype names `dtypes.Float16`/`dtypes.BFloat16` (and `dtypes.F8E4M3FN`/`dtypes.F8E5M2`, used only by the fp8-paused NotImplemented test) match the compute module enum.

**FP8 unchanged:** still paused → `ErrNotImplemented` via the S1 dtype-gate default branch; no F8 target constants, no F8 operand set, handed to janpfeifer per Contract C. The fp8 CPU unit test is [S1] (it tests the default branch that exists from S1).
