# Fused Attention 05 — lmkit-go Consumer Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. Read the shared contract first: `docs/plans/2026-06-29-fused-attn-00-contract.md` (Contract F).

**Goal:** Migrate lmkit-go's one `attention.FlashAttention` call site — which gomlx Stage 1 deletes — to the scope-free graph primitives `graph.InternalFusedOpCaller` + `graph.BackendFusedScaledDotProductAttention`, so lmkit still builds and fuses (with an automatic decomposed fallback) after the deletion.

**Stage:** Stage 1 only. Same stage as the gomlx deletion — gomlx Stage 1 breaks this build, this plan fixes it. No Stage 2 work (lmkit uses neither attention bias nor dropout).

**Architecture:** lmkit-go is scope-free by design (no `model.Scope`/context anywhere; weights passed explicitly). Upstream drops `FlashAttention` with no replacement helper (Contract D — a helper would duplicate `Core` or need a scope). So lmkit calls the bare primitives directly: `InternalFusedOpCaller(fused, decomposed)` runs the fused path and falls back to the decomposed closure on `ErrNotImplemented` — exactly what `FlashAttention` did internally. lmkit's existing fp32-parity decomposed else-branch is factored into a shared `decomposedAttention` helper used by both the fallback and the non-bf16 path (removes lmkit's own duplication).

**Tech Stack:** Go 1.26 workspace (`go.work`). Module under `model/`. Consumes `github.com/gomlx/gomlx/core/graph` (already imported as `g`) + `github.com/gomlx/compute` (for `AxesLayoutBSHD`).

## Global Constraints

(From the shared contract.)
- **Go 1.26.** Workspace modules under `go.work`. gomlx/compute/go-xla `replace` directives live in `model/go.mod` and `backend/go.mod`.
- **No push, no PR.** End at `git commit`. Guy reviews before any remote.
- **CUDA-gated checks** are tagged **[cuda]**: run on a CUDA GPU host (validated on an RTX 3070 Ti, sm_8.6, Ampere); serialize GPU jobs if the card is shared. CI/Mac runs the CPU decomposed path only.
- **Commit voice:** terse, verb-first, no em/en dashes, keep lmkit's prefix scheme, no Claude attribution.

**Grounded facts (verified):**
- `model/attention.go:21` `func Attention(cfg Config, x, wQ, wK, wV, wO *g.Node, positions []int) *g.Node`. Imports `g "github.com/gomlx/gomlx/core/graph"` and `github.com/gomlx/compute/dtypes`. KV heads already repeated to `nH` (lines 37-38), so q/k/v are equal-head BSHD `[B,T,nH,hd]`.
- Line 42-67: `if UseFlashAttention && q.DType() == dtypes.BFloat16 { out = attention.FlashAttention(q, k, v, scale) } else { <decomposed: einsum scores, MulScalar scale, Add causalMask(x.Graph(), tt) cast to dtype, fp32 Softmax, einsum output> }`. `causalMask(gr, tt)` helper exists in package.
- gomlx Stage 1 (consumed): `graph.InternalFusedOpCaller(fused, decomposed func() *Node) *Node` and `graph.BackendFusedScaledDotProductAttention(query, key, value, mask *Node, numHeads, numKVHeads int, axesLayout compute.AxesLayout, scale float64, causal bool, options *compute.ScaledDotProductAttentionConfig) *Node` (both in `core/graph`, scope-free; the same primitives the deleted `FlashAttention` used). `compute.AxesLayoutBSHD` is the BSHD layout constant.
- `model/attention_flash_test.go`: `TestAttentionFlashMatchesDecomposed` and `TestAttentionFlashGradMatchesDecomposed` toggle `model.UseFlashAttention` and assert the flash/fused path == decomposed path on identical bf16 inputs (falls back off-cuda, so they run in CI). These are the migration's parity gate; cuda detected via `Description()` containing "cuda".

---

## Task 0 [S1]: Point the gomlx replace at the local Stage-1 fork (NOT SKIPPABLE)

lmkit pins gomlx to a published fork pseudo-version that still HAS `flash.go`. To consume gomlx Stage 1 (flash.go deleted, primitives intact), repoint the `replace` to the local fork checkout. Dev-local edit for integration; do not push it (Guy reviews; the pseudo-version pin is restored when the fork is tagged).

**Files:**
- Modify: `model/go.mod` (the gomlx replace; compute/go-xla too if their Stage-1 commits are needed)
- Modify: `backend/go.mod` (mirror, so the workspace resolves consistently)

- [ ] **Step 1: Confirm gomlx Stage 1 is committed and flash.go is gone**

```bash
git -C /Users/guygrigsby/projects/forks/gomlx log --oneline -3
test -f /Users/guygrigsby/projects/forks/gomlx/ml/layers/attention/flash.go && echo "flash.go still present — gomlx deletion not done; STOP" || echo "flash.go deleted (good)"
git -C /Users/guygrigsby/projects/forks/gomlx grep -l "func BackendFusedScaledDotProductAttention\|func InternalFusedOpCaller" core/graph/ || echo "primitives MISSING — wrong gomlx state; STOP"
```

Expected: `flash.go` deleted; both primitives present in `core/graph`. If not, gomlx plan 03 Stage 1 is incomplete — STOP.

- [ ] **Step 2: Repoint the replace to the local path**

In `model/go.mod` and `backend/go.mod`, change the gomlx replace target to the local checkout:

```
replace github.com/gomlx/gomlx => /Users/guygrigsby/projects/forks/gomlx
```

(Repoint `compute`/`go-xla` to their local fork paths the same way IF lmkit must build against their Stage-1 commits; otherwise leave the pseudo-version pins.)

- [ ] **Step 3: Verify the build now FAILS on the deleted symbol (RED)**

```bash
cd /Users/guygrigsby/projects/lmkit-go/model && go build ./...
```

Expected: FAIL — `undefined: attention.FlashAttention` at `attention.go:47`. Confirms the local Stage-1 gomlx is wired in and the migration is needed. If it builds, the replace did not take effect (`go mod tidy` / check `go.work`).

- [ ] **Step 4: No commit yet** — Task 1 makes it green and commits the pair.

---

## Task 1 [S1]: Migrate to InternalFusedOpCaller; factor the decomposed branch

**Files:**
- Modify: `model/attention.go` (factor `decomposedAttention`; rewrite the if-branch; drop the now-unused `attention` import if nothing else in the file uses it — check)
- Test: `model/attention_flash_test.go` (existing — the parity gate; unchanged)

**Interfaces:**
- Consumes (gomlx Stage 1): `g.InternalFusedOpCaller(fused, decomposed func() *g.Node) *g.Node`; `g.BackendFusedScaledDotProductAttention(q,k,v,mask *g.Node, numHeads,numKVHeads int, axesLayout compute.AxesLayout, scale float64, causal bool, options *compute.ScaledDotProductAttentionConfig) *g.Node`.
- Produces: nothing downstream; lmkit is a leaf consumer.

- [ ] **Step 1: Confirm the parity test fails to build (RED)**

```bash
cd /Users/guygrigsby/projects/lmkit-go/model && go test -run TestAttentionFlashMatchesDecomposed ./...
```

Expected: build FAIL (`undefined: attention.FlashAttention`).

- [ ] **Step 2: Add the `github.com/gomlx/compute` import**

In `model/attention.go`'s import block, add `"github.com/gomlx/compute"` (for `compute.AxesLayoutBSHD`). Remove `"github.com/gomlx/gomlx/ml/layers/attention"` only if no other reference to `attention.` remains in the file (grep first).

- [ ] **Step 3: Factor the decomposed else-branch into a helper**

Extract the current else-branch body (attention.go:48-66) verbatim into a package function. It already derives everything from q/k/v:

```go
// decomposedAttention is the fp32-parity reference path: scaled q·kᵀ, causal mask,
// fp32 softmax, value aggregation. q,k,v are [B,T,nH,hd] (BSHD), equal heads.
func decomposedAttention(q, k, v *g.Node, scale float64) *g.Node {
	tt := q.Shape().Dimensions[1]
	scores := g.Einsum("btnh,bsnh->bnts", q, k)
	scores = g.MulScalar(scores, float32(scale))
	scores = g.Add(scores, g.ConvertDType(causalMask(q.Graph(), tt), scores.DType()))
	sdt := scores.DType()
	sf := scores
	if sdt != dtypes.Float32 {
		sf = g.ConvertDType(scores, dtypes.Float32)
	}
	probs := g.Softmax(sf, -1)
	if sdt != dtypes.Float32 {
		probs = g.ConvertDType(probs, sdt)
	}
	return g.Einsum("bnts,bsnh->btnh", probs, v)
}
```

- [ ] **Step 4: Rewrite the branch in `Attention` to use the primitives**

Replace the `if UseFlashAttention ... { out = attention.FlashAttention(...) } else { <inline decomposed> }` block (attention.go:42-67) with:

```go
	var out *g.Node // [B,T,nH,hd]
	if UseFlashAttention && q.DType() == dtypes.BFloat16 {
		// bf16 fused compute path: backend fused attention (scores never materialized)
		// when supported; InternalFusedOpCaller falls back to the decomposed path off-GPU.
		// q,k,v are [B,T,nH,hd] (BSHD), kv already repeated to nH above. Causal.
		out = g.InternalFusedOpCaller(
			func() *g.Node {
				return g.BackendFusedScaledDotProductAttention(
					q, k, v, nil, nH, nH, compute.AxesLayoutBSHD, scale, true, nil)
			},
			func() *g.Node { return decomposedAttention(q, k, v, scale) },
		)
	} else {
		out = decomposedAttention(q, k, v, scale)
	}
```

- [ ] **Step 5: Build green**

```bash
cd /Users/guygrigsby/projects/lmkit-go/model && go build ./...
```

Expected: exit 0.

- [ ] **Step 6: Parity tests pass on CPU (decomposed both sides)**

```bash
go test -run 'TestAttentionFlash(Matches|GradMatches)Decomposed' ./... -v
```

Expected: PASS. Off-cuda, the fused closure returns `ErrNotImplemented`, so `InternalFusedOpCaller` falls back to `decomposedAttention` — the same reference both branches use, so the parity is exact. This asserts the wiring (shapes, scale, head merge, VJP) is preserved.

- [ ] **Step 7: Commit**

```bash
git add model/attention.go model/go.mod backend/go.mod
git commit -m "model: migrate attention to gomlx fused primitives (FlashAttention removed upstream)"
```

---

## Verification gate (Stage 1)

- `cd model && go build ./...` exit 0; `go test ./...` green on Mac (CPU decomposed path; `TestAttentionFlash*Decomposed` parity holds via fallback).
- **[cuda]** on the CUDA GPU host (RTX 3070 Ti / sm_8.6): with a cuda backend, `TestAttentionFlashGradMatchesDecomposed` validates the real fused kernel + VJP against the decomposed reference. Confirm the bf16 path actually fuses (the backend does not return `ErrNotImplemented`).

## Self-review

- **Spec coverage:** Contract F (migrate the call site via `InternalFusedOpCaller`, factor the decomposed branch) → Task 1. No upstream `SimpleAttention` is consumed (correctly — it does not exist). ✓
- **Minimum change:** one branch rewrite + extracting the existing else-branch into a shared helper + one import + the dev-local replace repoint. The decomposed logic is unchanged (moved verbatim), and is now shared instead of inlined — a net reduction. ✓
- **No side effects:** the decomposed numerics are byte-identical (same code, relocated); only the bf16 branch swaps `FlashAttention` for the primitives it used to wrap. `UseFlashAttention` toggle preserved. ✓
- **Type consistency:** `BackendFusedScaledDotProductAttention(...) *g.Node` and `InternalFusedOpCaller(func() *g.Node, func() *g.Node) *g.Node` return `*g.Node` matching `out`; `decomposedAttention(q,k,v,scale) *g.Node` matches its uses. `compute.AxesLayoutBSHD` is the imported layout constant. ✓
