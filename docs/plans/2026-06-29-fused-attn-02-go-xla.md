# Fused Attention 02 — go-xla Backend (CustomCallV2 + fmha variants + seqlen) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Read the shared contract first: `docs/plans/2026-06-29-fused-attn-00-contract.md`.

**Goal:** Land the `go-xla` half of fused-attention upstreaming (PR gomlx/go-xla#37): a typed `CustomCallV2`, cuDNN fmha variant dispatch (bias / dropout / bias+dropout, all bf16), and seqlen padding masks — keeping every `__cudnn$fmha*` detail inside `go-xla`. FP8 is paused (no local sm_8.9+ hardware); fp8 input falls to `ErrNotImplemented` and is handed to janpfeifer — see Contract C.

**Architecture:** This plan implements Contract B (CustomCallV2) and Contract C (fmha variant dispatch + seqlen masking) from the shared contract, and consumes Contract A (the `compute.ScaledDotProductAttentionConfig` fields produced by plan 01). The string-layout `CustomCall` is replaced by a `[][]int`-layout `CustomCallV2` that renders MLIR `dense<...>` internally; `compute/xla/flash.go` rewires onto it and grows a config-driven target selector. Nothing flash-/CUDA-specific leaks above the backend: callers still see only "FusedSDPA: supported or `ErrNotImplemented`".

**Tech Stack:** Go 1.26, `github.com/gomlx/go-xla` (PR branch `flash-attention`), StableHLO emitter (`stablehlo` package), PJRT (`pjrt` package), cuDNN fMHA custom-calls. Consumes `github.com/gomlx/compute` (PR branch `flash-customcall`, via local `replace`).

## Global Constraints

(Verbatim from the shared contract — every task's requirements implicitly include these.)

- **Go 1.26.** Module: `github.com/gomlx/go-xla`. Consumes `github.com/gomlx/compute` via the local `replace` already in this fork's `go.mod`/`go.work`.
- **No push, no PR.** Every task ends at `git commit` on branch `flash-attention` in `/Users/guygrigsby/projects/forks/go-xla`. Guy reviews all diffs before anything reaches a remote.
- **CUDA-gated tests run on trig, not the Mac.** CI has no CUDA. Tasks whose test needs cuDNN execution are marked **[trig]**: build natively on trig, `GOMLX_BACKEND=xla:cuda`, wrap GPU work in `gputex run`. The CustomCallV2 layout-rendering unit test is CPU-only and runs on the Mac.
- **Fallback is the contract.** Any unsupported config (wrong dtype, unsupported variant/layout) returns a wrapped `compute.ErrNotImplemented`; the caller falls back to the decomposed path. Never panic on an unsupported-but-valid request.
- **Commit voice:** terse, verb-first, no em/en dashes, prefix `compute/xla:` (or `stablehlo:` for the customcall.go-only task), no Claude attribution.

**Grounded facts (verified against `fork/flash-attention`, the post-merge target):**
- `stablehlo/customcall.go`: `type literalStr string` (statement.go:177); attrs set `operand_layouts`/`result_layouts` to `literalStr(...)`. Const `CustomCallAPIVersionStatusReturning = 2`.
- `compute/xla/ops.go:826`: `customCall` wrapper; imports `shapes "github.com/gomlx/go-xla/types/shapes"` and aliases the stablehlo shapes as `stablehloshapes`. It maps `compute.Value` → stablehlo and back via `f.verifyAndCastValues` / `f.newNode`.
- `compute/xla/flash.go`: uses `github.com/gomlx/compute/shapes` and `github.com/gomlx/compute/dtypes`. `flashSupported` gates on `IsCUDA`, causal, nil mask, BSHD, equal heads. `flashFwdBackendConfig`/`flashBwdBackendConfig` build JSON via `flashBackendConfig(b,h,s,scale,dotDimNumbers)` with hardcoded `"mask_type": "CAUSAL"`, `"dropout_rate": 0.0`.
- `compute` dtypes (consumed): `dtypes.Float16`, `dtypes.BFloat16`, `dtypes.F8E4M3FN`, `dtypes.F8E5M2`.
- `pjrt/fmha_test.go`: gates `if *FlagPluginName != "cuda"`. `getPJRTClient(t) *Client` (pjrt_test.go:65); `(*Client).Plugin() *Plugin` (clients.go:212); `(*Plugin).IsCUDA() bool` (plugins.go:154).

---

## Task 0: Sync local branch to `fork/flash-attention` (NOT SKIPPABLE)

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

- [ ] **Step 5: No commit**

Task 0 is a branch sync, not a code change. Do not commit. Downstream tasks commit their own work.

---

## Task 1: `CustomCallV2` with typed `[][]int` layouts (Contract B)

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

## Task 2: Rewire `flash.go` onto config-driven targets and backend_config (Contract C scaffolding)

Restructure `flash.go` so the cuDNN target and `mask_type` come from the config rather than hardcoded constants. This task introduces the dispatch seam (functions + a variant struct) and routes the existing causal-bf16 path through it, with no behavior change yet. Tasks 3 and 4 add the bias/dropout/seqlen branches onto this seam.

**Files:**
- Modify: `compute/xla/flash.go`
- Test: `compute/xla/flash_dispatch_test.go` (create) — CPU-testable pure-function tests for the selector and backend_config builder

**Interfaces:**
- Consumes: `*compute.ScaledDotProductAttentionConfig` (Contract A: fields `Bias`, `QuerySeqLen`, `KeyValueSeqLen`, `DropoutRate`, `DropoutSeed`, `DropoutOffset`); `dtypes.BFloat16`, `dtypes.Float16`.
- Produces:
  ```go
  // compute/xla/flash.go
  type fmhaVariant struct {
      fwdTarget, bwdTarget string
      maskType             string // "CAUSAL" | "PADDING" | "PADDING_CAUSAL" | "NO_MASK"
      dropoutRate          float64
      hasBias, hasSeqLens  bool
  }
  // selectFMHAVariant maps dtype + config -> targets/maskType, or ErrNotImplemented.
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

- [ ] **Step 3: Add the variant struct, selector, and parameterized backend_config builder**

Add to `flash.go` (replace the `fmhaForwardTarget`/`fmhaBackwardTarget` const block with the full target set, and add the new functions):

```go
// cuDNN fused-attention custom-call targets. Forward/backward pairs per variant.
const (
	fmhaSoftmaxFwd                  = "__cudnn$fmhaSoftmax"
	fmhaSoftmaxBwd                  = "__cudnn$fmhaSoftmaxBackward"
	fmhaScaleBiasSoftmaxFwd         = "__cudnn$fmhaScaleBiasSoftmax"
	fmhaScaleBiasSoftmaxBwd         = "__cudnn$fmhaScaleBiasSoftmaxBackward"
	fmhaSoftmaxDropoutFwd           = "__cudnn$fmhaSoftmaxDropout"
	fmhaSoftmaxDropoutBwd           = "__cudnn$fmhaSoftmaxDropoutBackward"
	fmhaScaleBiasSoftmaxDropoutFwd  = "__cudnn$fmhaScaleBiasSoftmaxDropout"
	fmhaScaleBiasSoftmaxDropoutBwd  = "__cudnn$fmhaScaleBiasSoftmaxDropoutBackward"
	// FP8 targets (__cudnn$fmhaSoftmaxF8 / …BackwardF8) are intentionally NOT defined:
	// fp8 fmha is paused (no local sm_8.9+ hardware to test). fp8 input dtype falls to
	// ErrNotImplemented in selectFMHAVariant. Add the row here when wiring fp8 on Hopper/Ada.
)

// fmhaVariant captures the config-derived custom-call selection: the fwd/bwd targets, the
// backend_config mask_type, and the operand-set flags. Built by selectFMHAVariant.
type fmhaVariant struct {
	fwdTarget, bwdTarget string
	maskType             string // "CAUSAL" | "PADDING" | "PADDING_CAUSAL" | "NO_MASK"
	dropoutRate          float64
	hasBias, hasSeqLens  bool
}

// selectFMHAVariant maps the q/k/v dtype and config to a cuDNN variant, following Contract C
// precedence (bias+dropout > bias > dropout > standard). Dtype gate: all wired variants accept
// f16/bf16 only. Anything else (incl. fp8 e4m3fn/e5m2 — paused, no local hardware) ->
// ErrNotImplemented, and the caller falls back to the decomposed path.
// mask_type derives from causal + seqlens: PADDING_CAUSAL (both), PADDING (seqlens only),
// CAUSAL (causal only), NO_MASK (neither).
func selectFMHAVariant(op string, qkvDType dtypes.DType, causal bool,
	cfg *compute.ScaledDotProductAttentionConfig) (fmhaVariant, error) {
	var v fmhaVariant
	hasBias := cfg != nil && cfg.Bias != nil
	dropout := 0.0
	if cfg != nil {
		dropout = cfg.DropoutRate
	}
	hasSeqLens := cfg != nil && cfg.QuerySeqLen != nil && cfg.KeyValueSeqLen != nil

	isHalf := qkvDType == dtypes.Float16 || qkvDType == dtypes.BFloat16

	switch {
	case isHalf:
		switch {
		case hasBias && dropout > 0:
			v.fwdTarget, v.bwdTarget = fmhaScaleBiasSoftmaxDropoutFwd, fmhaScaleBiasSoftmaxDropoutBwd
		case hasBias:
			v.fwdTarget, v.bwdTarget = fmhaScaleBiasSoftmaxFwd, fmhaScaleBiasSoftmaxBwd
		case dropout > 0:
			v.fwdTarget, v.bwdTarget = fmhaSoftmaxDropoutFwd, fmhaSoftmaxDropoutBwd
		default:
			v.fwdTarget, v.bwdTarget = fmhaSoftmaxFwd, fmhaSoftmaxBwd
		}
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
	v.dropoutRate = dropout
	v.hasBias = hasBias
	v.hasSeqLens = hasSeqLens
	return v, nil
}

// flashBackendConfigV builds a cudnn_fmha_backend_config for the given variant: mask_type and
// dropout_rate come from v, the score-matrix dims [B,H,S,S] and fmha_scale from the shape,
// dotDimNumbers carries the bmm dot_dimension_numbers JSON (the fwd/bwd-specific part).
func flashBackendConfigV(b, h, s int, scale float64, dotDimNumbers string, v fmhaVariant) string {
	return fmt.Sprintf(`{"operation_queue_id": "0", "cudnn_fmha_backend_config": {"algorithm": {"algo_id": "0", "math_type": "TENSOR_OP_MATH", "tuning_knobs": {"17": "1", "24": "0"}, "is_cudnn_frontend": true, "workspace_size": "0"}, "fmha_scale": %s, "intermediate_tensor_shape": {"element_type": "BF16", "dimensions": ["%d", "%d", "%d", "%d"], "tuple_shapes": [], "layout": {"dim_level_types": [], "dim_unique": [], "dim_ordered": [], "minor_to_major": ["3", "2", "1", "0"], "tiles": [], "element_size_in_bits": "0", "memory_space": "0", "index_primitive_type": "PRIMITIVE_TYPE_INVALID", "pointer_primitive_type": "PRIMITIVE_TYPE_INVALID", "dynamic_shape_metadata_prefix_bytes": "0"}, "is_dynamic_dimension": [false, false, false, false]}, "is_flash_attention": true, "mask_type": "%s", %s, "dropout_rate": %s, "seed": 42, "sliding_window_length": 0, "max_seg_per_batch": 1, "is_paged_attention": false}}`,
		formatScale(scale), b, h, s, s, v.maskType, dotDimNumbers, formatScale(v.dropoutRate))
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

## Task 3: Bias, dropout, and seqlen operand sets (Contract C)

Build the right operand list per variant: bias adds a `[B,H,S,Skv]` bias operand; dropout adds the seed/offset operands and threads `dropout_rate` into the config; seqlens append the query/kv length operands. The dtype gate already lives in `selectFMHAVariant` (Task 2); this task wires the operands and layouts. (FP8 is paused — `selectFMHAVariant` returns `ErrNotImplemented` for fp8 before any operand work, so there is no F8 operand set here.)

**Files:**
- Modify: `compute/xla/flash.go`
- Test: `compute/xla/flash_variants_cuda_test.go` (create) — **[trig]** execution tests, one per wired variant (bias, dropout)
- Test: `compute/xla/flash_dispatch_test.go` (the CPU dispatch-unit-test file from Task 2) — add the fp8-paused `TestSelectFMHAVariant_FP8NotImplemented` here (CPU, no cuda)

**Interfaces:**
- Consumes: `fmhaVariant` (Task 2); `cfg.Bias`, `cfg.DropoutSeed`, `cfg.DropoutOffset` (Contract A); `f.customCall` ([][]int layouts, Task 1).
- Produces: forward/backward operand builders inside the two fused methods (no new exported names; the variant struct drives them).

- [ ] **Step 1: Write the [trig] variant execution tests**

`compute/xla/flash_variants_cuda_test.go` (the `//go:build` tag keeps it off the Mac build is unnecessary — it self-skips on non-cuda; mark each as **[trig]**):

```go
package xla

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
)

// [trig] runs under GOMLX_BACKEND=xla:cuda. Each builds a tiny fused-attention graph for one
// variant and asserts it executes (output finite, right shape). q=k=v=ones makes the softmax
// uniform so the bias-free output is all ~1.0; with bias the output stays finite.
func TestFMHAVariant_Bias_trig(t *testing.T) {
	be := getFusionBackend(t) // helper below; skips if backend can't fuse bf16 causal
	cfg := &compute.ScaledDotProductAttentionConfig{Bias: makeBias(t, be)}
	out := runFusedFwd(t, be, dtypes.BFloat16, true, cfg)
	assertFiniteBSHD(t, out)
}

func TestFMHAVariant_Dropout_trig(t *testing.T) {
	be := getFusionBackend(t)
	cfg := &compute.ScaledDotProductAttentionConfig{
		DropoutRate:   0.1,
		DropoutSeed:   makeI64Scalar(t, be, 42),
		DropoutOffset: makeI64Scalar(t, be, 0),
	}
	out := runFusedFwd(t, be, dtypes.BFloat16, true, cfg)
	assertFiniteBSHD(t, out)
}

```

And in `compute/xla/flash_dispatch_test.go` (CPU file, Mac-runnable), pin the fp8-paused seam:

```go
// fp8 is paused (no local sm_8.9+ hardware). Pin the seam: an fp8 dtype must select
// no variant and surface ErrNotImplemented so the caller falls back to decomposed.
// Pure unit test on selectFMHAVariant — CPU, runs on the Mac (no [trig]).
func TestSelectFMHAVariant_FP8NotImplemented(t *testing.T) {
	_, err := selectFMHAVariant("fmha", dtypes.F8E4M3FN, true, nil)
	require.True(t, compute.IsNotImplemented(err), "fp8 must be NotImplemented (paused), got %v", err)
	_, err = selectFMHAVariant("fmha", dtypes.F8E5M2, true, nil)
	require.True(t, compute.IsNotImplemented(err), "fp8 e5m2 must be NotImplemented (paused), got %v", err)
}
```

Add the test helpers (`getFusionBackend`, `runFusedFwd`, `makeBias`, `makeI64Scalar`, `assertFiniteBSHD`, `assertFiniteBSHD`) in the same file. `getFusionBackend` probes `BackendFusedScaledDotProductAttention` on a tiny causal bf16 input and `t.Skip`s if it returns `compute.ErrNotImplemented` (mirrors the contract's capability check). Build the graph through the public `compute.Backend`/`compute.Function` API the package already uses in its other `_cuda_test.go` files — grep `git grep -l "GOMLX_BACKEND\|getFusionBackend\|NewBackend" compute/xla/*_test.go` for the established harness and reuse it rather than inventing a new one.

- [ ] **Step 2: Run on trig to verify they fail [trig]**

On trig, branch synced, in the go-xla checkout:

```bash
gputex run "fmha-variants" -- env GOMLX_BACKEND=xla:cuda go test ./compute/xla/ -run 'TestFMHAVariant_.*_trig' -v
```

Expected: FAIL — bias variant emits the bias target but passes no bias operand (cuDNN rejects operand count), or the helpers reference operand-building not yet present. Capture the actual failure.

- [ ] **Step 3: Build the per-variant operand sets in the forward method**

In `FusedScaledDotProductAttention`, after casting q/k/v, assemble operands and matching layouts from the variant:

```go
	// Operand order cuDNN expects: q, k, v, [bias], [seqQ, seqKV], [dropout seed, offset].
	operands := []compute.Value{q, k, v}
	operandLayouts := [][]int{{3, 2, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0}}
	if variant.hasBias {
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
	if variant.dropoutRate > 0 {
		operands = append(operands, options.DropoutSeed, options.DropoutOffset)
		operandLayouts = append(operandLayouts, nil, nil) // int64 scalars
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

- [ ] **Step 4: Mirror the operand set in the backward (VJP) method**

In `FusedScaledDotProductAttentionVJP`, build the backward operands appending the same optional operands after the fixed `q, k, v, softmaxStats, dOut, out`:

```go
	operands := []compute.Value{q, k, v, softmaxStats, dOut, out}
	operandLayouts := [][]int{{3, 2, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0}, {2, 1, 0}, {3, 2, 1, 0}, {3, 2, 1, 0}}
	if variant.hasBias {
		bias, err := f.bf16(options.Bias)
		if err != nil { return nil, nil, nil, err }
		operands = append(operands, bias)
		operandLayouts = append(operandLayouts, nil)
	}
	if variant.hasSeqLens {
		operands = append(operands, options.QuerySeqLen, options.KeyValueSeqLen)
		operandLayouts = append(operandLayouts, nil, nil)
	}
	if variant.dropoutRate > 0 {
		operands = append(operands, options.DropoutSeed, options.DropoutOffset)
		operandLayouts = append(operandLayouts, nil, nil)
	}
	bwdResultLayouts := [][]int{{3, 1, 2, 0}, {3, 1, 2, 0}, {3, 1, 2, 0}, {0}}
	grads, err := f.customCall(variant.bwdTarget, flashBwdBackendConfig(b, h, s, scale, variant),
		operandLayouts, []shapes.Shape{bhsd, bhsd, bhsd, scratch}, bwdResultLayouts, operands...)
```

- [ ] **Step 5: Run the variant tests on trig to verify they pass [trig]**

```bash
gputex run "fmha-variants" -- env GOMLX_BACKEND=xla:cuda go test ./compute/xla/ -run 'TestFMHAVariant_.*_trig' -v
```

Expected: PASS for the variants cuDNN supports on the trig GPU (bias, dropout). Document which variants ran green.

- [ ] **Step 6: Confirm the Mac CPU build/test still green**

```bash
cd /Users/guygrigsby/projects/forks/go-xla
go build ./... # pjrt still references old CustomCall; expect only that failure
go vet ./compute/xla/
```

Expected: `compute/xla` builds; `pjrt` fails only on `CustomCall` (Task 5).

- [ ] **Step 7: Commit**

```bash
git add compute/xla/flash.go compute/xla/flash_variants_cuda_test.go
git commit -m "compute/xla: per-variant fmha operand sets (bias, dropout, seqlens); fp8 paused as NotImplemented"
```

---

## Task 4: Seqlen padding masks — relax `flashSupported` (Contract C)

Allow seqlens through `flashSupported` and let the variant's `mask_type` (`PADDING` / `PADDING_CAUSAL`) drive masking. The forward/backward operand append for seqlens already landed in Task 3; this task is the gate relaxation plus the seqlen execution test.

**Files:**
- Modify: `compute/xla/flash.go` (`flashSupported`)
- Test: `compute/xla/flash_seqlen_cuda_test.go` (create) — **[trig]**

**Interfaces:**
- Consumes: `cfg.QuerySeqLen`, `cfg.KeyValueSeqLen` (Contract A); `fmhaVariant.maskType` (Task 2).
- Produces: relaxed `flashSupported` that accepts seqlens; no new exported names.

- [ ] **Step 1: Write the [trig] seqlen test**

`compute/xla/flash_seqlen_cuda_test.go`:

```go
package xla

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
)

// [trig] PADDING_CAUSAL: per-batch lengths shorter than S must mask the padding rows. With
// q=k=v=ones, masking changes which keys contribute, so the masked output differs from the
// unmasked all-ones output for the shortened batch element. Runs under xla:cuda.
func TestFMHA_SeqLenPaddingCausal_trig(t *testing.T) {
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

Reuse the harness helpers from Task 3; add `makeI32Vec` and `runFusedFwdShaped` (a `runFusedFwd` variant taking B,S) in the Task 3 helper file or here.

- [ ] **Step 2: Run on trig to verify it fails [trig]**

```bash
gputex run "fmha-seqlen" -- env GOMLX_BACKEND=xla:cuda go test ./compute/xla/ -run 'TestFMHA_SeqLen.*_trig' -v
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

- [ ] **Step 4: Run on trig to verify it passes [trig]**

```bash
gputex run "fmha-seqlen" -- env GOMLX_BACKEND=xla:cuda go test ./compute/xla/ -run 'TestFMHA_SeqLen.*_trig' -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add compute/xla/flash.go compute/xla/flash_seqlen_cuda_test.go
git commit -m "compute/xla: allow seqlen padding masks in cuDNN flash path"
```

---

## Task 5: Migrate `pjrt/fmha_test.go` to `CustomCallV2` and `IsCUDA`

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

- [ ] **Step 4: Run on trig to verify the execution path passes [trig]**

```bash
gputex run "fmha-fwd-exec" -- env GOMLX_BACKEND=xla:cuda go test ./pjrt/ -run TestFMHAForwardExecute -plugin cuda -v
```

Expected: PASS (output all ~1.0, per the test's invariant).

- [ ] **Step 5: Commit**

```bash
git add pjrt/fmha_test.go
git commit -m "compute/xla: migrate fmha pjrt test to CustomCallV2 and IsCUDA gate"
```

---

## Verification Gate (Plan 02, from the contract)

Run before declaring plan 02 complete.

- [ ] **Mac (CPU) — full non-CUDA suite green:**

```bash
cd /Users/guygrigsby/projects/forks/go-xla
go build ./...
go test ./stablehlo/ ./compute/... ./pjrt/ 2>&1 | tail -30
```

Expected: build exit 0; all packages pass. CUDA-gated tests (`TestFMHA*`, `TestFMHAVariant_*_trig`, `TestFMHA_SeqLen*_trig`) SKIP on the Mac (no cuda plugin); the `CustomCallV2` rendering unit test (`TestRenderLayouts`) and the dispatch tests (`TestSelectFMHAVariant*`, `TestFlashBackendConfigV`) PASS. No `undefined` / fallback regressions.

- [ ] **trig (`xla:cuda`) — every variant in the matrix green [trig]:**

On trig, branch synced, native build:

```bash
gputex run "fmha-gate" -- env GOMLX_BACKEND=xla:cuda go test ./compute/xla ./pjrt -run 'FMHA|Flash' -plugin cuda -v
```

Expected: PASS for each wired variant (standard causal, bias, dropout, bias+dropout, seqlen padding-causal). Variants unsupported by the installed cuDNN/card `t.Skip` rather than fail — note which ran green. (fp8 is not in this matrix — paused.)

- [ ] **Downstream integration unblocked:** confirm `compute/xla` exports nothing CUDA-specific above the backend (the only public surface change is `stablehlo.CustomCallV2`; `flash.go`'s methods keep their Contract-A signatures). Plan 03 (gomlx) consumes this via the local `replace`.

---

## Self-Review

**Spec coverage** (against contract Contracts B, C and the SCOPE list):
- Contract B `CustomCallV2` + internal MLIR rendering + remove old `CustomCall` + rewire `ops.go`/`flash.go` → Task 1. CPU layout-rendering unit test → Task 1 Step 1.
- Contract C variant dispatch (Bias+Dropout / Bias / Dropout / Softmax precedence) → Task 2 (`selectFMHAVariant`). Dtype gate (f16/bf16 only, else ErrNotImplemented incl. paused fp8) → Task 2, pinned by `TestSelectFMHAVariant_FP8NotImplemented` (Task 3, CPU). Per-variant operand sets (bias operand, dropout seed/offset + rate) → Task 3. Seqlen operands + `mask_type` CAUSAL/PADDING/PADDING_CAUSAL/NO_MASK → Task 2 (mask_type) + Task 3 (operands) + Task 4 (gate relaxation). VJP mirrors operands → Task 3 Step 4.
- Test fix `client.Plugin().IsCUDA()` → Task 5.
- Task 0 sync is first and non-skippable.
- [trig] markers present on every cuDNN-execution test; layout-rendering and dispatch tests are CPU-only on the Mac.

**Placeholder scan:** no TBD/TODO; every code step shows full code. The Task 3/4 test-helper harness defers to the package's existing `_cuda_test.go` harness (grep instruction given) rather than inventing one — this is the one place exact code depends on the synced tree's existing helpers; flagged explicitly so the implementer reuses, not guesses.

**Type consistency:** `fmhaVariant`, `selectFMHAVariant`, `flashBackendConfigV`, `flashFwdBackendConfig(…, v fmhaVariant)`, `flashBwdBackendConfig(…, v fmhaVariant)`, `dtypeOf`, `renderLayouts`, `CustomCallV2`, and the `customCall` wrapper signature (dropped `apiVersion`, `[][]int` layouts) are used identically across Tasks 1–5. `flashSupported` gains the trailing `options` param in Task 2 and uses it in Task 4 — consistent. dtype names `dtypes.Float16`/`dtypes.BFloat16` (and `dtypes.F8E4M3FN`/`dtypes.F8E5M2`, used only by the fp8-paused NotImplemented test) match the compute module enum.
