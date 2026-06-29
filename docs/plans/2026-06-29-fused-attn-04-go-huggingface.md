# Fused Attention — go-huggingface Consumer (plan 04) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thread per-batch sequence lengths from transformer attention call sites into `gomlx`'s `MultiHeadAttentionBuilder.WithSeqLens(...)`, so encoder models use padding masking via the fused path instead of materializing an `[S,Skv]` mask — guarded so models that supply no lengths are byte-for-byte unchanged.

**Stage:** entirely **Stage 1** (seqlen plumbing only; no bias/dropout variants). Every task here is [S1]. Depends on gomlx Stage 1 (`WithSeqLens`).

**Architecture:** This is the leaf consumer of the fused-attention chain (see contract repo dependency order). It produces no new public API; it rewires existing attention call sites in the transformers/encoder layers to optionally pass sequence lengths. Each rewrite is a guarded, opt-in branch: lengths present → `WithSeqLens`; lengths absent → the existing mask-matrix path, unchanged.

**Tech Stack:** Go 1.26, `github.com/gomlx/go-huggingface`, `github.com/gomlx/gomlx` (local fork on branch `flash-attention`, supplying `WithSeqLens` per Contract D), `github.com/gomlx/go-xla`, `github.com/gomlx/compute`.

## Contract reference

This plan implements **Contract E** and consumes **Contract D** from
`docs/plans/2026-06-29-fused-attn-00-contract.md`. Read that file first.

- **Consumes (Contract D, gomlx layers):**
  ```go
  func (b *MultiHeadAttentionBuilder) WithSeqLens(querySeqLen, keyValueSeqLen *Node) *MultiHeadAttentionBuilder
  func (b *MultiHeadAttentionBuilder) WithFusion(enabled bool) *MultiHeadAttentionBuilder
  ```
  `WithSeqLens` takes int32 `[B]` nodes and is **mutually exclusive** with an explicit query/key matrix mask. Supplied by the local gomlx fork.
- **Produces (Contract E):** the rewired call sites — no exported symbols. The deliverable is behavior-preserving when no lengths are supplied, and a fused padding-mask path when they are.

Global constraints (Go 1.26, no push/no PR, fallback-is-the-contract, voice rules
for commits, CUDA tests on the CUDA host) are defined once in the contract and apply to every
task here. Not restated.

## Recon dependency note (read before executing)

The go-huggingface repo is **not yet cloned** at plan-writing time, so the exact
file paths, line numbers, package names, the attention block's struct/field names,
and the repo's commit-message prefix convention **cannot be pinned from source**.
This plan is therefore structured as **recon-then-implement**:

- **Task 0** forks and clones the repo and wires the local gomlx replace.
- **Task 1** is an explicit reconnaissance step. It greps the cloned repo and produces
  a written list of attention call sites that materialize a padding mask from sequence
  lengths (file:line). **You must fill that list before doing any Task 2..N.**
- **Tasks 2..N** are templated: one task per recon'd call site. Exactly **one** is fully
  coded as a worked example (Task 2, the most likely site: a BERT/encoder-style attention
  block where lengths come from the attention mask). The remaining per-site tasks
  (Task 3..N) reuse Task 2's transformation; you instantiate them from the Task 1 list.

This plan is allowed **one** level of "fill from recon" indirection: the *list of sites*.
Everything downstream of that list — the per-site code transformation, the test
fixtures, the `go test` invocations, the assertions — is concrete here. If the recon
in Task 1 contradicts an assumption in Task 2 (e.g. the attention block does not exist,
or lengths are not derivable from an attention mask), **stop and report** rather than
inventing a call site.

---

### Task 0: Fork, clone, and wire the local gomlx replace

**Files:**
- Create: `/Users/guygrigsby/projects/forks/go-huggingface/` (clone)
- Modify: `/Users/guygrigsby/projects/forks/go-huggingface/go.mod`

**Interfaces:**
- Consumes: nothing (bootstrap).
- Produces: a buildable local checkout on branch `fused-seqlen` whose `go.mod`
  resolves `github.com/gomlx/gomlx` to the local fork that exports `WithSeqLens`.

- [ ] **Step 1: Fork upstream (no local clone via gh)**

```bash
gh repo fork gomlx/go-huggingface --clone=false
```

Expected: prints `✓ Created fork guygrigsby/go-huggingface` (or `already exists`).

- [ ] **Step 2: Clone the fork to the canonical forks dir**

```bash
mkdir -p /Users/guygrigsby/projects/forks
git clone git@github.com:guygrigsby/go-huggingface.git /Users/guygrigsby/projects/forks/go-huggingface
```

Expected: clone succeeds, `main` (or the repo's default branch) checked out.

- [ ] **Step 3: Add upstream remote and the PR branch**

```bash
git -C /Users/guygrigsby/projects/forks/go-huggingface remote add upstream https://github.com/gomlx/go-huggingface.git
git -C /Users/guygrigsby/projects/forks/go-huggingface fetch upstream
git -C /Users/guygrigsby/projects/forks/go-huggingface checkout -b fused-seqlen
```

Expected: on branch `fused-seqlen`; `git remote -v` lists both `origin` (guygrigsby) and `upstream` (gomlx).

- [ ] **Step 4: Add local replaces so it builds against the WithSeqLens gomlx**

Append to `/Users/guygrigsby/projects/forks/go-huggingface/go.mod` (these point at the
sibling forks that carry the contract's upstream work; the gomlx fork is on its
`flash-attention` branch and exports `WithSeqLens`):

```
replace github.com/gomlx/gomlx => /Users/guygrigsby/projects/forks/gomlx

replace github.com/gomlx/go-xla => /Users/guygrigsby/projects/forks/go-xla

replace github.com/gomlx/compute => /Users/guygrigsby/projects/forks/compute
```

Then sync the module graph:

```bash
cd /Users/guygrigsby/projects/forks/go-huggingface && go mod tidy
```

Note: if go-huggingface does not currently require `go-xla`/`compute` directly,
`go mod tidy` will drop the unused replaces — that is fine; keep only the `gomlx`
replace if the others are pruned. The load-bearing one is `gomlx`.

- [ ] **Step 5: Confirm WithSeqLens resolves from the local fork**

```bash
cd /Users/guygrigsby/projects/forks/go-huggingface && \
  go doc github.com/gomlx/gomlx/ml/layers/attention.MultiHeadAttentionBuilder.WithSeqLens
```

Expected: prints the `WithSeqLens(querySeqLen, keyValueSeqLen *Node) *MultiHeadAttentionBuilder`
signature. If it errors with "no such symbol", the gomlx fork/branch is wrong —
stop and verify the gomlx replace target is the `flash-attention` branch checkout
before continuing.

- [ ] **Step 6: Baseline build + test green before touching anything**

```bash
cd /Users/guygrigsby/projects/forks/go-huggingface && go build ./... && go test ./... 2>&1 | tail -30
```

Expected: builds; tests pass (or the pre-existing baseline — record any already-failing
tests now so later tasks don't get blamed for them).

- [ ] **Step 7: Commit the scaffold**

```bash
cd /Users/guygrigsby/projects/forks/go-huggingface
git add go.mod go.sum
git commit -m "build: pin local gomlx fork for WithSeqLens (fused-seqlen)"
```

(Commit prefix here is provisional — Task 1 discovers the repo's real prefix
convention; if it differs, match it from Task 2 onward.)

---

### Task 1: Reconnaissance — enumerate sites that materialize a padding mask from sequence lengths

**Files:**
- Read-only across `/Users/guygrigsby/projects/forks/go-huggingface/`
- Output (working notes, do not commit): the filled call-site list below.

**Interfaces:**
- Consumes: Task 0's clone.
- Produces: a written list `RECON_SITES` of `file:line` locations where a model
  builds an attention/padding mask from token sequence lengths and feeds it into
  `attention.MultiHeadAttention(...)`. Tasks 2..N are instantiated from this list.

- [ ] **Step 1: Find attention builder call sites**

```bash
cd /Users/guygrigsby/projects/forks/go-huggingface
grep -rn --include='*.go' 'attention\.MultiHeadAttention(' .
grep -rn --include='*.go' '\.WithCausalMask(' .
grep -rn --include='*.go' '\.Done()' . | grep -i attention
```

- [ ] **Step 2: Find where masks are constructed from sequence lengths / attention masks**

```bash
cd /Users/guygrigsby/projects/forks/go-huggingface
grep -rinE 'attention[_ ]?mask|padding[_ ]?mask|seq[_ ]?len|sequence[_ ]?length' --include='*.go' .
grep -rinE 'WithQueryKeyMatrixMask|MatrixMask|key_padding|invertedMask|\(1\s*-\s*mask\)' --include='*.go' .
```

- [ ] **Step 3: Find where tokenizer output becomes a mask in the graph**

```bash
cd /Users/guygrigsby/projects/forks/go-huggingface
grep -rinE 'AttentionMask|attention_mask|input_ids|ReduceSum.*mask|mask.*ReduceSum' --include='*.go' .
```

- [ ] **Step 4: Discover the repo's commit-message prefix convention**

```bash
cd /Users/guygrigsby/projects/forks/go-huggingface && git log --oneline -40
```

Record the dominant prefix scheme (e.g. `bert:`, `models:`, `attention:`, or none).
Use it for all commits in Tasks 2..N. Keep the voice rules: terse, verb-first, no
em/en dashes, no Claude attribution.

- [ ] **Step 5: Write the recon list**

Fill this table from Steps 1-3. A site qualifies only if it (a) calls the gomlx
attention builder and (b) has, in scope, a per-batch notion of sequence length
(an attention mask `[B,Skv]`, a `seqLens` `[B]` node, or token-count derivation):

```
RECON_SITES (fill before Task 2):
| # | file:line (builder call) | how lengths are available in scope | mask currently built at file:line |
|---|--------------------------|------------------------------------|-----------------------------------|
| 2 | <fill, this is the worked example — pick the BERT/encoder block> | <e.g. attention_mask [B,Skv] from tokenizer> | <fill> |
| 3 | <fill> | <fill> | <fill> |
| … | …                        | …                                  | …                                 |
```

- [ ] **Step 6: Decide concrete scope and STOP for review**

If the list is empty (no site both calls the builder and has lengths in scope),
**stop and report** — there is nothing to thread, and Contract E is satisfied
vacuously; do not invent a site. Otherwise proceed: Task 2 is the BERT/encoder
row, Tasks 3..N are the remaining rows, each a copy of Task 2's transformation
retargeted to that file:line.

(No commit — recon produces notes, not code.)

---

### Task 2: Worked example — seqlen path in the BERT/encoder attention block

This is the fully-coded template. Tasks 3..N copy this transformation to the other
`RECON_SITES` rows. The exact file path and the attention-block struct/field names
come from Task 1; placeholders below are marked `‹from-recon›` and there is exactly
one indirection level (the file path). Everything else — the helper, the guard, the
test — is concrete.

**Files:**
- Create: `/Users/guygrigsby/projects/forks/go-huggingface/‹recon-pkg›/seqlens.go` (the shared helper, so Tasks 3..N reuse it — DRY)
- Modify: `‹RECON_SITES row 2 file›` (the BERT/encoder attention block) at the builder call site
- Test: `/Users/guygrigsby/projects/forks/go-huggingface/‹recon-pkg›/seqlens_test.go`

**Interfaces:**
- Consumes (Contract D): `attention.MultiHeadAttentionBuilder.WithSeqLens(q, kv *Node) *MultiHeadAttentionBuilder`.
- Produces: a package-local helper
  ```go
  // seqLensFromMask returns per-batch key/value lengths (int32 [B]) from a
  // boolean-or-0/1 attention mask of shape [B, Skv], or nil if mask is nil.
  func seqLensFromMask(mask *graph.Node) *graph.Node
  ```
  and a guarded branch at the builder call site that uses `WithSeqLens` when a mask
  is present and falls back to the existing mask-matrix path when it is not.

- [ ] **Step 1: Write the failing test (parity: seqlen path vs mask-matrix path)**

Create `‹recon-pkg›/seqlens_test.go`. The fixture is small and deterministic: batch
of 2, key length 4, with a padding mask that keeps 3 then 2 valid positions. It asserts
(a) `seqLensFromMask` returns `[3, 2]` and (b) running attention through `WithSeqLens`
with those lengths matches running it through an explicit additive `[B,1,Sq,Skv]` mask
within tolerance.

```go
package ‹reconpkg›

import (
	"testing"

	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/layers/attention"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
)

// maskFixture: B=2, Skv=4. Row 0 valid for first 3 keys, row 1 for first 2.
func maskFixture(g *graph.Graph) *graph.Node {
	return graph.Const(g, [][]float32{{1, 1, 1, 0}, {1, 1, 0, 0}}) // [B, Skv]
}

func TestSeqLensFromMask(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	got := graph.ExecOnce(backend, func(g *graph.Graph) *graph.Node {
		return seqLensFromMask(maskFixture(g))
	})
	require.Equal(t, []int32{3, 2}, got.Value())
}

func TestSeqLensParityWithMaskMatrix(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	const B, Sq, Skv, H, Dh = 2, 3, 4, 2, 8

	// Deterministic Q/K/V so both paths see identical inputs.
	qkv := func(g *graph.Graph) (q, k, v *graph.Node) {
		q = graph.Iota(g, shapesOf(B, Sq, H, Dh), -1)
		k = graph.Iota(g, shapesOf(B, Skv, H, Dh), -1)
		v = graph.Iota(g, shapesOf(B, Skv, H, Dh), -1)
		return
	}

	maskPath := graph.ExecOnce(backend, func(g *graph.Graph) *graph.Node {
		q, k, v := qkv(g)
		mask := maskFixture(g)                              // [B, Skv], 1=keep
		add := graph.MulScalar(graph.OneMinus(mask), -1e9)  // [B, Skv] additive
		add = graph.ExpandAxes(add, 1, 1)                   // [B,1,1,Skv]
		return attention.MultiHeadAttention(g, q, k, v).
			SetKeyQueryDimension(Dh).WithQueryKeyMatrixMask(add).Done()
	})

	seqlenPath := graph.ExecOnce(backend, func(g *graph.Graph) *graph.Node {
		q, k, v := qkv(g)
		kvLen := seqLensFromMask(maskFixture(g))            // int32 [B]
		qLen := graph.Const(g, []int32{Sq, Sq})             // all queries valid
		return attention.MultiHeadAttention(g, q, k, v).
			SetKeyQueryDimension(Dh).WithSeqLens(qLen, kvLen).Done()
	})

	requireAllClose(t, maskPath.Value(), seqlenPath.Value(), 1e-4)
}
```

> Note: `shapesOf`, `requireAllClose`, `SetKeyQueryDimension`, and the exact
> `MultiHeadAttention` constructor arity come from Task 1 recon — match the repo's
> actual builder surface and its existing test helpers. If the repo already has a
> close-comparison helper, use it instead of `requireAllClose` (DRY). The test's
> *shape* (two paths, same inputs, assert within tolerance) is the load-bearing part.

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/guygrigsby/projects/forks/go-huggingface && \
  go test ./‹recon-pkg›/ -run 'TestSeqLens' -v 2>&1 | tail -30
```

Expected: FAIL — `undefined: seqLensFromMask` (helper not written yet).

- [ ] **Step 3: Write the helper**

Create `‹recon-pkg›/seqlens.go`:

```go
package ‹reconpkg›

import (
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gopjrt/dtypes"
)

// seqLensFromMask converts a key/value attention mask of shape [B, Skv]
// (1/true = keep, 0/false = pad) into per-batch valid lengths int32 [B],
// for attention.MultiHeadAttentionBuilder.WithSeqLens. Returns nil when
// mask is nil so callers can guard on it. Assumes left-aligned padding
// (the standard tokenizer layout): length == count of kept positions.
func seqLensFromMask(mask *graph.Node) *graph.Node {
	if mask == nil {
		return nil
	}
	g := mask.Graph()
	kept := graph.ConvertDType(graph.GreaterThanScalar(mask, 0.5), dtypes.Int32) // [B, Skv]
	lens := graph.ReduceSum(kept, -1)                                            // [B]
	return lens
}
```

> `GreaterThanScalar`/`ConvertDType`/`ReduceSum` are the gomlx graph ops; if recon
> shows the mask is already boolean, drop the `GreaterThanScalar` and convert directly.
> Keep the helper independent of any model struct so Tasks 3..N reuse it verbatim.

- [ ] **Step 4: Run the helper-level test to verify it passes**

```bash
cd /Users/guygrigsby/projects/forks/go-huggingface && \
  go test ./‹recon-pkg›/ -run 'TestSeqLensFromMask' -v 2>&1 | tail -20
```

Expected: PASS (`[3, 2]`).

- [ ] **Step 5: Wire the guarded branch into the BERT/encoder attention block**

At `‹RECON_SITES row 2 file›`, the existing call materializes a mask and passes it via
`WithQueryKeyMatrixMask` (or equivalent — confirm from recon). Replace that with a guard:
when a per-batch attention mask is in scope, prefer `WithSeqLens`; otherwise keep the
existing mask-matrix call **unchanged**.

```go
// BEFORE (existing, mask-matrix path — keep as the fallback arm):
//   b := attention.MultiHeadAttention(g, q, k, v).SetKeyQueryDimension(headDim)
//   b = b.WithQueryKeyMatrixMask(additiveMask)   // additiveMask from attentionMask
//   out := b.Done()

// AFTER (guarded; behavior-preserving when attentionMask == nil):
b := attention.MultiHeadAttention(g, q, k, v).SetKeyQueryDimension(headDim)
if useSeqLens && attentionMask != nil {
	kvLen := seqLensFromMask(attentionMask)            // int32 [B]
	qLen := seqLensFromMask(queryMask)                 // int32 [B]; or all-Sq if queries unpadded
	b = b.WithSeqLens(qLen, kvLen)
} else if additiveMask != nil {
	b = b.WithQueryKeyMatrixMask(additiveMask)         // unchanged legacy path
}
out := b.Done()
```

Guard rules (Contract E — must hold):
- `useSeqLens` defaults such that the **mask path is taken unless a model explicitly
  opts in / supplies lengths**, so existing models are byte-for-byte unchanged. If the
  repo has no config knob, gate purely on `attentionMask != nil` AND a build-time
  package default (a `const useSeqLens = false` you flip per-model), so the default
  binary is identical. Confirm the exact toggle mechanism from recon.
- `WithSeqLens` and `WithQueryKeyMatrixMask` are mutually exclusive (Contract D) —
  the `if/else` above guarantees only one is ever called.
- If `queryMask` is unavailable (encoder queries are typically unpadded), pass
  `qLen = graph.BroadcastToShape(graph.Scalar(g, dtypes.Int32, Sq), shapesOf(B))` —
  i.e. all queries valid. Match the repo's idiom for a constant `[B]` from recon.

- [ ] **Step 6: Run the full parity test + package suite to verify green**

```bash
cd /Users/guygrigsby/projects/forks/go-huggingface && \
  go test ./‹recon-pkg›/ -run 'TestSeqLens' -v 2>&1 | tail -30 && \
  go test ./... 2>&1 | tail -20
```

Expected: `TestSeqLensFromMask` and `TestSeqLensParityWithMaskMatrix` PASS; full suite
PASS (no regression vs the Task 0 baseline).

- [ ] **Step 7: Verify the default path is byte-for-byte unchanged**

```bash
cd /Users/guygrigsby/projects/forks/go-huggingface && \
  go test ./... -run 'Bert|Encoder|Model' -count=1 2>&1 | tail -20
```

Expected: PASS. With `useSeqLens` defaulting off, the existing model integration/golden
tests must be untouched. If a golden test shifts, the guard is leaking — fix the default
before committing.

- [ ] **Step 8: Commit**

```bash
cd /Users/guygrigsby/projects/forks/go-huggingface
git add ‹recon-pkg›/seqlens.go ‹recon-pkg›/seqlens_test.go ‹RECON_SITES-row-2-file›
git commit -m "‹prefix›: optional WithSeqLens padding path in encoder attention"
```

(`‹prefix›` = the convention discovered in Task 1 Step 4. Voice rules: terse,
verb-first, no em/en dashes, no Claude attribution.)

---

### Task 3..N: Apply the seqlen path to each remaining RECON_SITES row

One task per remaining row in the Task 1 `RECON_SITES` table. Each is the Task 2
transformation retargeted — **no new helper** (reuse `seqLensFromMask`), same guard
rules, same parity-test shape.

**Files (per row r):**
- Modify: `‹RECON_SITES row r file›` at its builder call site.
- Test: add a `TestSeqLensParity_‹site›` to that package's `_test.go` (or extend
  `seqlens_test.go` if same package), mirroring `TestSeqLensParityWithMaskMatrix`
  with that site's actual Q/K/V shapes and mask source.

**Interfaces:**
- Consumes: `seqLensFromMask` (Task 2), `WithSeqLens` (Contract D).
- Produces: the rewired site; no new exported symbols.

Per row, repeat the Task 2 cycle:

- [ ] **Step 1:** Write a parity test for this site (copy `TestSeqLensParityWithMaskMatrix`,
  swap in this site's batch/seq/head dims and its real mask source). Code it out — do
  not write "similar to Task 2".
- [ ] **Step 2:** Run it, confirm it FAILs (the site still uses only the mask path).
  `go test ./‹pkg›/ -run 'TestSeqLensParity_‹site›' -v`.
- [ ] **Step 3:** Insert the guarded `if useSeqLens && mask != nil { … WithSeqLens … } else { …legacy… }`
  branch at the builder call (Task 2 Step 5 shape), reusing `seqLensFromMask`.
- [ ] **Step 4:** Run the test, confirm PASS.
- [ ] **Step 5:** Run `go test ./...` and the default-path golden check
  (`go test ./... -run 'Bert|Encoder|Model' -count=1`); confirm no regression.
- [ ] **Step 6:** Commit with the recon'd prefix:
  `git commit -m "‹prefix›: optional WithSeqLens padding path in ‹site›"`.

If the row turns out not to have lengths in scope after closer reading, drop it from
scope and note why — do not force a `WithSeqLens` where there is no length source.

---

## Verification gate (Contract gate 04)

From `docs/plans/2026-06-29-fused-attn-00-contract.md`, gate **04 go-huggingface**:

- [ ] **Mac, full suite green:**

```bash
cd /Users/guygrigsby/projects/forks/go-huggingface && go test ./... 2>&1 | tail -20
```

Expected: PASS on the Mac (CPU `go` backend). This exercises the decomposed/fallback
paths; no CUDA required. The fused path resolves to the CPU `go` backend reference
(Contract A implements seqlen masking on CPU), so the parity tests run and pass on the Mac.

- [ ] **Integration parity assertion present and green:** the worked-example test
  `TestSeqLensParityWithMaskMatrix` (Task 2) — and each `TestSeqLensParity_‹site›`
  (Tasks 3..N) — asserts the seqlen path output is within `1e-4` of the mask-matrix
  path on a small fixture. This is the contract's required "one model integration test
  asserting seqlen path produces output within tolerance of the mask-matrix path."

- [ ] **Default-path invariance:** existing model golden/integration tests unchanged
  with `useSeqLens` off (Task 2 Step 7 / Task 3..N Step 5). This is the Contract E guard:
  models that supply no lengths are byte-for-byte unchanged.

- [ ] **No push, no PR.** Plan ends at `git commit` on branch `fused-seqlen` in the
  local fork. Guy reviews diffs before anything reaches a remote.

(No **[cuda]** step in plan 04: the contract's 04 gate is Mac-only. cuDNN variant
coverage is gated in plans 02/03; here the CPU backend reference makes the seqlen
parity test runnable on the Mac.)

---

## Self-Review (concrete portions only)

Per the recon-dependency note, only Task 0 and the concrete scaffolding of Tasks 1-2
are fully pinnable now; the per-site fan-out (Tasks 3..N) and exact paths/prefixes are
recon-bound by design.

**1. Spec/contract coverage:**
- Contract E (thread seqlens into transformer attention, guarded) → Tasks 2, 3..N. ✓
- Contract D consumption (`WithSeqLens`) → Task 0 Step 5 (resolves) + Task 2 Step 5 (calls). ✓
- Gate 04 (Mac suite green + one parity integration test within tolerance) → Verification gate. ✓
- "models without lengths unchanged" → Task 2 Step 7 + guard rules. ✓

**2. Placeholder scan:** The only intentional placeholders are the recon indirections
(`‹recon-pkg›`, `‹RECON_SITES row file›`, `‹prefix›`), each explicitly defined as
"fill from Task 1" and bounded to one indirection level per the recon note. No `TODO`,
no "implement later", no "add error handling" — the helper, guard, and tests are fully
coded. Tasks 3..N repeat the Task 2 code rather than saying "similar to Task 2"
(the cycle steps are spelled out).

**3. Type consistency:** `seqLensFromMask(mask *graph.Node) *graph.Node` defined in
Task 2 Step 3, consumed identically in Task 2 Step 5 and Tasks 3..N. `WithSeqLens(q, kv)`
matches Contract D's signature everywhere (Task 0 Step 5, Task 2 tests + wiring). int32
`[B]` length type is consistent across helper return, test assertions, and the builder call.
