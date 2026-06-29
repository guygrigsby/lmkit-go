# Fused Attention 05 — lmkit-go Consumer Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. Read the shared contract first: `docs/plans/2026-06-29-fused-attn-00-contract.md` (Contract F + the `SimpleAttention` entry in Contract D).

**Goal:** Migrate lmkit-go's one `attention.FlashAttention` call site to the scope-free `attention.SimpleAttention`, so lmkit-go still builds and fuses after gomlx Stage 1 deletes `flash.go`.

**Stage:** Stage 1 only. Same stage as the gomlx deletion — gomlx Stage 1 breaks this build, this plan fixes it. No Stage 2 work (lmkit uses neither attention bias nor dropout).

**Architecture:** lmkit-go is scope-free by design (no `model.Scope`/context anywhere; weights passed explicitly). The deleted `FlashAttention(q,k,v,scale)` is replaced by `SimpleAttention(q,k,v,scale)` — same scope-free, causal, equal-head, pre-projected shape (Contract D option A). lmkit keeps its own fp32-parity decomposed else-branch unchanged; only the `UseFlashAttention && bf16` if-branch swaps the call.

**Tech Stack:** Go 1.26 workspace (`go.work`). Module `github.com/lmkit/...` under `model/`. Consumes `github.com/gomlx/gomlx` via the `replace` in `model/go.mod`.

## Global Constraints

(From the shared contract.)
- **Go 1.26.** Workspace modules under `go.work`. The gomlx/compute/go-xla `replace` directives live in `model/go.mod` and `backend/go.mod`.
- **No push, no PR.** End at `git commit`. Guy reviews before any remote.
- **CUDA-gated checks** are tagged **[cuda]**: run on a CUDA GPU host (validated on an RTX 3070 Ti, sm_8.6, Ampere); serialize GPU jobs if the card is shared. CI/Mac runs the CPU decomposed path only.
- **Commit voice:** terse, verb-first, no em/en dashes, keep lmkit's prefix scheme, no Claude attribution.

**Grounded facts (verified):**
- `model/attention.go:21` `func Attention(cfg Config, x, wQ, wK, wV, wO *g.Node, positions []int) *g.Node`. Line 47: `out = attention.FlashAttention(q, k, v, scale)` inside `if UseFlashAttention && q.DType() == dtypes.BFloat16`. KV heads already repeated to `nH` (lines 37-38), so q/k/v are equal-head BSHD `[B,T,nH,hd]`. The `else` branch (lines 48-66) is the fp32-parity decomposed path — leave it untouched.
- `model/go.mod:26` `replace github.com/gomlx/gomlx => github.com/guygrigsby/gomlx <pseudo-version>` (also compute:28, go-xla:30). `backend/go.mod` has the same three.
- `model/attention_flash_test.go`: `TestAttentionFlashMatchesDecomposed` (toggles `model.UseFlashAttention`, asserts flash path == decomposed path on identical bf16 inputs; falls back to decomposed off-cuda so it runs in CI) and `TestAttentionFlashGradMatchesDecomposed` (same for the VJP). These are the migration's parity gate. Cuda detected via `Description()` containing "cuda".

---

## Task 0 [S1]: Point the gomlx replace at the local Stage-1 fork (NOT SKIPPABLE)

lmkit currently pins gomlx to a published fork pseudo-version that still HAS `flash.go`. To consume gomlx Stage 1 (flash.go deleted, `SimpleAttention` added), repoint the `replace` to the local fork checkout. This is a dev-local edit for integration; do not push it (Guy reviews; the published-pseudo-version pin is restored when the fork is tagged).

**Files:**
- Modify: `model/go.mod` (the gomlx replace; compute/go-xla too if their Stage-1 commits are needed for the build)
- Modify: `backend/go.mod` (mirror, so the workspace resolves consistently)

- [ ] **Step 1: Confirm gomlx Stage 1 is committed in the local fork**

```bash
git -C /Users/guygrigsby/projects/forks/gomlx log --oneline -3
git -C /Users/guygrigsby/projects/forks/gomlx grep -l "func SimpleAttention" ml/layers/attention/ || echo "SimpleAttention MISSING — gomlx Stage 1 not done; STOP"
test -f /Users/guygrigsby/projects/forks/gomlx/ml/layers/attention/flash.go && echo "flash.go still present — gomlx deletion not done; STOP" || echo "flash.go deleted (good)"
```

Expected: `SimpleAttention` present, `flash.go` deleted. If not, gomlx plan 03 Stage 1 is incomplete — STOP.

- [ ] **Step 2: Repoint the replace to the local path**

In `model/go.mod` and `backend/go.mod`, change the gomlx replace target to the local checkout:

```
replace github.com/gomlx/gomlx => /Users/guygrigsby/projects/forks/gomlx
```

(Repoint `github.com/gomlx/compute` and `github.com/gomlx/go-xla` to their local fork paths the same way IF lmkit must build against their Stage-1 commits; otherwise leave the pseudo-version pins.)

- [ ] **Step 3: Verify the build now FAILS on the deleted symbol (this is the RED state)**

```bash
cd /Users/guygrigsby/projects/lmkit-go/model && go build ./...
```

Expected: FAIL — `undefined: attention.FlashAttention` at `attention.go:47`. This confirms the local Stage-1 gomlx is wired in and the migration is needed. If it builds, the replace did not take effect (run `go mod tidy` / check `go.work`).

- [ ] **Step 4: No commit yet** — Task 1 makes the build green and commits the pair together.

---

## Task 1 [S1]: Swap FlashAttention → SimpleAttention

**Files:**
- Modify: `model/attention.go:47`
- Test: `model/attention_flash_test.go` (existing — the parity gate; no new test needed, the wiring is already covered)

**Interfaces:**
- Consumes (gomlx Stage 1, Contract D): `func attention.SimpleAttention(query, key, value *g.Node, scale float64) *g.Node` — scope-free, causal, equal-head, fuses when the backend supports it else decomposes.
- Produces: nothing downstream; lmkit is a leaf consumer.

- [ ] **Step 1: Confirm the parity test fails to build (RED)**

```bash
cd /Users/guygrigsby/projects/lmkit-go/model && go test -run TestAttentionFlashMatchesDecomposed ./...
```

Expected: build FAIL (`undefined: attention.FlashAttention`). The test itself is unchanged; it cannot compile until the call site is migrated.

- [ ] **Step 2: Replace the call**

In `model/attention.go`, line 47:

```go
		out = attention.SimpleAttention(q, k, v, scale)
```

Update the adjacent comment to drop the "flash"-specific framing:

```go
		// bf16 fused compute path: backend fused attention (scores never materialized)
		// when supported; transparently decomposed off-GPU. q,k,v are [B,T,nH,hd] (BSHD),
		// kv already repeated to nH above. Causal.
```

Leave the `UseFlashAttention` toggle and the `else` decomposed parity branch exactly as they are.

- [ ] **Step 3: Build green**

```bash
cd /Users/guygrigsby/projects/lmkit-go/model && go build ./...
```

Expected: exit 0.

- [ ] **Step 4: Parity tests pass on CPU (decomposed both sides)**

```bash
go test -run 'TestAttentionFlash(Matches|GradMatches)Decomposed' ./... -v
```

Expected: PASS. Off-cuda, `SimpleAttention` falls back to decomposed, so this asserts the wiring (shapes, scale, head merge, VJP) matches lmkit's own decomposed path — exactly what it checked before the migration.

- [ ] **Step 5: Commit**

```bash
git add model/attention.go model/go.mod backend/go.mod
git commit -m "model: migrate attention to gomlx SimpleAttention (FlashAttention removed upstream)"
```

---

## Verification gate (Stage 1)

- `cd model && go build ./...` exit 0; `go test ./...` green on Mac (CPU decomposed path; `TestAttentionFlash*Decomposed` parity holds).
- **[cuda]** on the CUDA GPU host (RTX 3070 Ti / sm_8.6): with a cuda backend, `TestAttentionFlashGradMatchesDecomposed` validates the real fused kernel + VJP against the decomposed reference (the test logs a WARNING and is trivial off-cuda). Confirm the bf16 path actually fuses (the backend does not return `ErrNotImplemented`).

## Self-review

- **Spec coverage:** Contract F (migrate the FlashAttention call site) → Task 1. The scope-free `SimpleAttention` signature (Contract D option A) is consumed exactly. ✓
- **Minimum change:** one call-site line + its comment + the dev-local replace repoint. lmkit's decomposed parity branch and `UseFlashAttention` toggle are untouched. No new test (the existing parity test is the gate). ✓
- **No side effects:** the `else` fp32-parity path is byte-for-byte unchanged; only the bf16 fused branch swaps one function whose contract (fuse-or-decompose, causal, equal-head) matches the deleted `FlashAttention`. ✓
- **Type consistency:** `SimpleAttention(q, k, v *g.Node, scale float64) *g.Node` matches the call `attention.SimpleAttention(q, k, v, scale)` with `out *g.Node`. ✓
