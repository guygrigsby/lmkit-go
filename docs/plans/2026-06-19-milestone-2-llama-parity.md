# Milestone 2 — Llama block parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the five Llama building blocks (RMSNorm, RoPE, GQA attention, SwiGLU, tied embeddings) in Go using GoMLX `core/graph` ops, each proven to match a PyTorch reference within tight tolerance.

**Architecture:** A new `model` Go module (own `go.mod`, in `go.work`) writes each block as a pure function over `*graph.Node`. A committed PyTorch generator emits JSON golden fixtures; Go parity tests run each block on the SimpleGo backend and compare elementwise. Per ADR-0009 the runtime boundary moves to `backend/`: `model/` may import GoMLX's op vocabulary (`core/graph`, `core/tensors`, `compute/dtypes`, `compute/shapes`) but not the runtime (`compute.New`, `backends`, `go-xla`, `pjrt`).

**Tech Stack:** Go 1.26, GoMLX `core/graph` at the pinned commit `516689cbe913…`; PyTorch (CPU) for the golden generator; SimpleGo backend (`-tags noxla`, `GOMLX_BACKEND=go`) for tests.

## Global Constraints

- **Runtime boundary (ADR-0009):** `model/` (incl. its `*_test.go`) must NOT import the runtime packages `"github.com/gomlx/compute"` (root), `github.com/gomlx/gomlx/backends`, `github.com/gomlx/go-xla`, `pjrt`. It MAY import `core/graph`, `core/tensors`, `compute/dtypes`, `compute/shapes`, and our `backend/gomlx` package. Tests get a backend via `gomlx.New().Compute()` (inferred type, so no runtime import is named).
- **Per-package modules + go.work (ADR-0005):** `model` is its own module.
- **Additive API (ADR-0008):** adding `(*gomlx.Backend).Compute()` is additive; don't change existing `backend`/`gomlx` symbols.
- **Determinism:** blocks take explicit weights as `*graph.Node`; no variable creation, no RNG inside blocks. The generator is seeded.
- **Tolerance:** parity asserts elementwise `|got-expected| <= tol`; default `tol=1e-5`, document a per-block relaxation to `5e-5` only if a block's accumulation needs it.
- **Build/test:** SimpleGo only — `-tags noxla`, `GOMLX_BACKEND=go`.
- **Commits:** terse, verb-first, no dashes, no Claude/Anthropic attribution.

## Verified GoMLX `core/graph` op reference (pinned `516689cbe913`)

Package `g "github.com/gomlx/gomlx/core/graph"`. Reductions DROP reduced axes (use `ReduceAndKeep` to keep). Binary ops need EQUAL rank (lift with `Reshape`).

```go
g.Mul(a,b), g.Add(a,b), g.Sub(a,b), g.Neg(x), g.Exp(x)
g.Rsqrt(x)                               // 1/sqrt(x)
g.MulScalar(x, f), g.AddScalar(x, f), g.DivScalar(x, f)   // f is a Go number
g.Sigmoid(x)                             // == Logistic; SiLU = g.Mul(x, g.Sigmoid(x))
g.ReduceMean(x, axes...), g.ReduceMax(x, axes...), g.ReduceSum(x, axes...)  // axes drop
g.ReduceAndKeep(x, g.ReduceMean, -1)     // keep the reduced axis as size 1
g.Reshape(x, dims...)                    // one -1 allowed (inferred)
g.Transpose(x, axisA, axisB)             // swap two axes
g.Concatenate([]*g.Node{a,b}, axis)      // takes a SLICE, not variadic
g.Slice(x, g.AxisRange().Spacer(), g.AxisRangeFromStart(d/2))  // x[..., :d/2]
g.Slice(x, g.AxisRange().Spacer(), g.AxisRangeToEnd(d/2))      // x[..., d/2:]
g.Einsum("btnh,bsnh->bnts", q, k)        // batched contraction
g.MatMul(a, b)                           // numpy matmul, batches leading dims
g.Gather(table, idx)                     // idx last dim = #indexed axes; see embedding task
g.Softmax(logits, -1)                    // numerically stable softmax over last axis
g.Const(graph, tensors.FromFlatDataAndDimensions(data, dims...))  // precomputed constant node
x.Shape().Dimensions  // []int ; x.Rank() int
```

Exec (in `*_test.go` only): `exec := g.MustNewExec(be, func(a, b *g.Node) *g.Node {...}); out := exec.MustExec1(aT, bT)` where `be` comes from `gomlx.New().Compute()` and `aT,bT` are `*tensors.Tensor`.

---

### Task 1: Revise the boundary test to the runtime boundary (ADR-0009)

The current `app/boundary_test.go` fails on ANY `github.com/gomlx` import outside `backend/`. The `model` module will legitimately import `core/graph`, so the gate must be narrowed to the runtime packages first.

**Files:**
- Modify: `app/boundary_test.go`

**Interfaces:**
- Produces: a boundary test that flags only runtime imports (`"github.com/gomlx/compute"`, `gomlx/backends`, `go-xla`, `pjrt`) outside `backend/gomlx/`.

- [ ] **Step 1: Replace the grep pattern and message**

Replace the body of `TestBackendBoundary` in `app/boundary_test.go` so the matched pattern is the runtime packages only (the op-vocabulary packages `core/graph`, `core/tensors`, `compute/dtypes`, `compute/shapes` must NOT match):

```go
func TestBackendBoundary(t *testing.T) {
	root, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		t.Fatalf("git root: %v", err)
	}
	// Runtime packages (ADR-0009): backend construction, plugin/device, execution.
	// The op-vocabulary (core/graph, core/tensors, compute/dtypes, compute/shapes)
	// is intentionally NOT matched — model/ and train/ may use it.
	const runtime = `"github\.com/gomlx/compute"|github\.com/gomlx/gomlx/backends|github\.com/gomlx/go-xla|pjrt`
	cmd := exec.Command("git", "grep", "-lE", runtime, "--", "*.go", ":!app/boundary_test.go")
	cmd.Dir = strings.TrimSpace(string(root))
	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); !ok || ee.ExitCode() != 1 {
			t.Fatalf("git grep failed: %v", err)
		}
		// exit 1 == no matches == clean
	}
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "backend/gomlx/") {
			t.Errorf("runtime import outside backend/gomlx/: %s", line)
		}
	}
}
```

- [ ] **Step 2: Run on the clean tree — expect PASS**

Run: `cd "$(git rev-parse --show-toplevel)" && go test -tags noxla ./app/ -run TestBackendBoundary -v`
Expected: PASS (only `backend/gomlx/gomlx.go` imports the runtime packages).

- [ ] **Step 3: Prove it still catches a runtime leak AND allows op-vocabulary**

```bash
cd "$(git rev-parse --show-toplevel)"
# (a) runtime leak outside backend -> must FAIL
printf 'package app\n\nimport _ "github.com/gomlx/go-xla"\n' > app/leak_tmp.go
git add app/leak_tmp.go
go test -tags noxla ./app/ -run TestBackendBoundary 2>&1 | grep -q "runtime import outside" && echo "CAUGHT leak (good)"
git rm -f app/leak_tmp.go
# (b) op-vocabulary outside backend -> must PASS
printf 'package app\n\nimport _ "github.com/gomlx/gomlx/core/graph"\n' > app/opvocab_tmp.go
git add app/opvocab_tmp.go
go test -tags noxla ./app/ -run TestBackendBoundary && echo "op-vocab allowed (good)"
git rm -f app/opvocab_tmp.go
```
Expected: (a) prints "CAUGHT leak (good)"; (b) PASS + "op-vocab allowed (good)". Both temp files removed.

- [ ] **Step 4: Commit**

```bash
git add app/boundary_test.go
git commit -m "app: narrow boundary test to runtime packages (ADR-0009)

model/ and train/ may use GoMLX core/graph ops; only the runtime
(compute root, backends, go-xla, pjrt) must stay in backend/gomlx/."
```

---

### Task 2: model module + parity harness + RMSNorm (end-to-end)

Stands up the `model` module, the JSON parity harness, the `Compute()` accessor, and the simplest block (RMSNorm) green against a generated golden. Proves the whole harness.

**Files:**
- Create: `model/go.mod`
- Modify: `go.work` (add `./model`)
- Modify: `backend/gomlx/gomlx.go` (add `Compute()` accessor)
- Create: `model/config.go`
- Create: `model/internal/paritytest/paritytest.go`
- Create: `model/testdata/gen_goldens.py`
- Create: `model/testdata/rmsnorm.json` (generated)
- Create: `model/rmsnorm.go`
- Create: `model/rmsnorm_test.go`
- Modify: `scripts/check.sh` (add the `model` module)

**Interfaces:**
- Produces: `model.Config`; `model.RMSNorm(x, scale *g.Node, eps float32) *g.Node`; `paritytest.Load`, `paritytest.Tensor.ToTensor`, `paritytest.AssertClose`; `(*gomlx.Backend).Compute() compute.Backend`.

- [ ] **Step 1: Create the module and add it to the workspace**

`model/go.mod`:
```
module github.com/guygrigsby/lmkit-go/model

go 1.26
```
`go.work` — add `./model` to the `use` block:
```
use (
	./backend
	./app
	./model
)
```

- [ ] **Step 2: Add the `Compute()` accessor to the adapter**

In `backend/gomlx/gomlx.go`, add (the test escape hatch for running raw graphs; `b.be` already exists):
```go
// Compute returns the underlying compute backend for tests and one-shot graph
// evaluation. Callers that take it never import the runtime package by name
// (the type is inferred), keeping the ADR-0009 boundary intact.
func (b *Backend) Compute() compute.Backend { return b.be }
```

- [ ] **Step 3: Write the Config**

`model/config.go`:
```go
// Package model implements Llama building blocks in GoMLX core/graph ops
// (ADR-0009: model may use the op vocabulary; the runtime stays in backend/).
package model

// Config holds Llama block hyperparameters. Only the fields a given block needs
// are read by that block.
type Config struct {
	Hidden    int
	NHeads    int
	NKVHeads  int
	HeadDim   int
	FFNHidden int
	VocabSize int
	RopeBase  float64
	RMSEps    float64
}
```

- [ ] **Step 4: Write the parity harness helper**

`model/internal/paritytest/paritytest.go`:
```go
// Package paritytest loads JSON golden fixtures and compares tensors for the
// model block parity tests. It does not import the runtime (ADR-0009): callers
// obtain a backend via gomlx.New().Compute() and run the graph themselves.
package paritytest

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/gomlx/gomlx/core/tensors"
)

// Tensor is a fixture tensor: row-major float32 data plus its shape.
type Tensor struct {
	Shape []int     `json:"shape"`
	Data  []float32 `json:"data"`
}

// ToTensor builds a GoMLX tensor from the fixture tensor.
func (tn Tensor) ToTensor() *tensors.Tensor {
	return tensors.FromFlatDataAndDimensions(tn.Data, tn.Shape...)
}

// Fixture is one block's golden: config scalars, named inputs and weights, and
// the expected output.
type Fixture struct {
	Config   map[string]float64 `json:"config"`
	Inputs   map[string]Tensor  `json:"inputs"`
	Weights  map[string]Tensor  `json:"weights"`
	Expected Tensor             `json:"expected"`
}

// Load reads and decodes a fixture JSON file.
func Load(t *testing.T, path string) Fixture {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture %s: %v", path, err)
	}
	var f Fixture
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatalf("decode fixture %s: %v", path, err)
	}
	return f
}

// AssertClose fails if got and want differ by more than tol elementwise.
func AssertClose(t *testing.T, got []float32, want Tensor, tol float32) {
	t.Helper()
	if len(got) != len(want.Data) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want.Data))
	}
	var maxDiff float32
	for i := range got {
		d := float32(math.Abs(float64(got[i] - want.Data[i])))
		if d > maxDiff {
			maxDiff = d
		}
		if d > tol {
			t.Errorf("element %d: got %v, want %v (diff %v > tol %v)", i, got[i], want.Data[i], d, tol)
		}
	}
	t.Logf("max abs diff = %g (tol %g)", maxDiff, tol)
}
```

- [ ] **Step 5: Write the golden generator (RMSNorm entry) and generate the fixture**

`model/testdata/gen_goldens.py`:
```python
#!/usr/bin/env python3
"""Generate JSON golden fixtures for lmkit-go model block parity tests.

Run from the model/testdata/ directory: `python3 gen_goldens.py`.
Requires torch (CPU is fine). Deterministic (seeded). Each block writes
<block>.json with {config, inputs, weights, expected} as float32 row-major.
"""
import json, math, torch

torch.manual_seed(0)

def t2j(x):
    x = x.detach().to(torch.float32).contiguous()
    return {"shape": list(x.shape), "data": x.flatten().tolist()}

def write(name, config, inputs, weights, expected):
    obj = {"config": config,
           "inputs": {k: t2j(v) for k, v in inputs.items()},
           "weights": {k: t2j(v) for k, v in weights.items()},
           "expected": t2j(expected)}
    with open(f"{name}.json", "w") as f:
        json.dump(obj, f)
    print(f"wrote {name}.json")

def gen_rmsnorm():
    B, T, H, eps = 2, 3, 8, 1e-5
    x = torch.randn(B, T, H)
    scale = torch.randn(H)
    ms = x.pow(2).mean(-1, keepdim=True)
    y = x * torch.rsqrt(ms + eps) * scale
    write("rmsnorm", {"hidden": H, "rms_eps": eps},
          {"x": x}, {"scale": scale}, y)

if __name__ == "__main__":
    gen_rmsnorm()
```
Run: `cd model/testdata && python3 -c "import torch; print(torch.__version__)" && python3 gen_goldens.py`
Expected: prints a torch version and `wrote rmsnorm.json`. If `import torch` fails, STOP and report — the generator environment must be resolved (per the spec's open question).

- [ ] **Step 6: Write the failing RMSNorm parity test**

`model/rmsnorm_test.go`:
```go
package model_test

import (
	"testing"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/model"
	"github.com/guygrigsby/lmkit-go/model/internal/paritytest"
)

func TestRMSNormParity(t *testing.T) {
	f := paritytest.Load(t, "testdata/rmsnorm.json")
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	eps := float32(f.Config["rms_eps"])
	exec := g.MustNewExec(be.Compute(), func(x, scale *g.Node) *g.Node {
		return model.RMSNorm(x, scale, eps)
	})
	out := exec.MustExec1(f.Inputs["x"].ToTensor(), f.Weights["scale"].ToTensor())
	paritytest.AssertClose(t, tensors.MustCopyFlatData[float32](out), f.Expected, 1e-5)
}
```

- [ ] **Step 7: Run the test — expect FAIL**

Run: `cd model && GOMLX_BACKEND=go go test -tags noxla . -run TestRMSNormParity -v`
Expected: FAIL — `model.RMSNorm` undefined.

- [ ] **Step 8: Implement RMSNorm**

`model/rmsnorm.go`:
```go
package model

import g "github.com/gomlx/gomlx/core/graph"

// RMSNorm normalizes x [B,T,H] over its last axis and scales by scale [H].
// y = x * rsqrt(mean(x^2, lastAxis) + eps) * scale. The mean/rsqrt run in the
// graph's dtype; callers wanting fp32-internal accumulation pass fp32 x.
func RMSNorm(x, scale *g.Node, eps float32) *g.Node {
	ms := g.ReduceAndKeep(g.Mul(x, x), g.ReduceMean, -1) // [B,T,1]
	inv := g.Rsqrt(g.AddScalar(ms, eps))                 // [B,T,1]
	normed := g.Mul(x, inv)                              // [B,T,H]
	s := g.Reshape(scale, 1, 1, -1)                      // [1,1,H]
	return g.Mul(normed, s)
}
```

- [ ] **Step 9: Run the test — expect PASS**

Run: `cd model && GOMLX_BACKEND=go go test -tags noxla . -run TestRMSNormParity -v`
Expected: PASS (max abs diff logged well under 1e-5).

- [ ] **Step 10: Extend the gate to the model module**

In `scripts/check.sh`, change the module loop to include `model`:
```bash
for mod in backend model app; do
```
Run: `make check`
Expected: backend, model, app all build+test, boundary clean, prints `OK`.

- [ ] **Step 11: Commit**

```bash
git add model/ go.work backend/gomlx/gomlx.go scripts/check.sh
git commit -m "model: RMSNorm with parity harness

New model module + JSON golden harness + gomlx.Compute() test accessor.
RMSNorm matches the PyTorch golden on SimpleGo. check.sh covers model."
```

---

### Task 3: RoPE

Rotary position embedding (rotate-half), applied to Q and K. cos/sin are precomputed on the host from positions + base and injected as constants.

**Files:**
- Modify: `model/testdata/gen_goldens.py` (add `gen_rope`)
- Create: `model/testdata/rope.json` (generated)
- Create: `model/rope.go`
- Create: `model/rope_test.go`

**Interfaces:**
- Consumes: `paritytest`, `gomlx.New().Compute()`.
- Produces: `model.RoPE(x *g.Node, positions []int, base float64, headDim int) *g.Node` — applies rotary embedding to a `[B,T,nHeads,headDim]` tensor over its last axis.

- [ ] **Step 1: Add the RoPE generator and regenerate**

Append to `model/testdata/gen_goldens.py` (before `__main__`):
```python
def gen_rope():
    B, T, nH, hd, base = 2, 4, 2, 8, 10000.0
    x = torch.randn(B, T, nH, hd)
    positions = torch.arange(T)
    inv_freq = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))   # [hd/2]
    freqs = positions[:, None].float() * inv_freq[None, :]            # [T, hd/2]
    emb = torch.cat([freqs, freqs], dim=-1)                          # [T, hd]
    cos = emb.cos()[None, :, None, :]                                # [1,T,1,hd]
    sin = emb.sin()[None, :, None, :]
    def rotate_half(t):
        a, b = t[..., : hd // 2], t[..., hd // 2 :]
        return torch.cat([-b, a], dim=-1)
    y = x * cos + rotate_half(x) * sin
    write("rope", {"head_dim": hd, "rope_base": base, "seq_len": T},
          {"x": x}, {}, y)
```
Add `gen_rope()` to the `__main__` block. Run: `cd model/testdata && python3 gen_goldens.py` → `wrote rope.json`.

- [ ] **Step 2: Write the failing RoPE parity test**

`model/rope_test.go`:
```go
package model_test

import (
	"testing"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/model"
	"github.com/guygrigsby/lmkit-go/model/internal/paritytest"
)

func TestRoPEParity(t *testing.T) {
	f := paritytest.Load(t, "testdata/rope.json")
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	headDim := int(f.Config["head_dim"])
	base := f.Config["rope_base"]
	seqLen := int(f.Config["seq_len"])
	positions := make([]int, seqLen)
	for i := range positions {
		positions[i] = i
	}
	exec := g.MustNewExec(be.Compute(), func(x *g.Node) *g.Node {
		return model.RoPE(x, positions, base, headDim)
	})
	out := exec.MustExec1(f.Inputs["x"].ToTensor())
	paritytest.AssertClose(t, tensors.MustCopyFlatData[float32](out), f.Expected, 1e-5)
}
```
Run: `cd model && GOMLX_BACKEND=go go test -tags noxla . -run TestRoPEParity -v` → FAIL (`model.RoPE` undefined).

- [ ] **Step 3: Implement RoPE**

`model/rope.go`:
```go
package model

import (
	"math"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
)

// RoPE applies rotary position embedding (rotate-half convention) to x, shaped
// [B,T,nHeads,headDim], over its last axis. positions has length T. cos/sin are
// computed on the host (matching the HF Llama formula) and injected as constants
// shaped [1,T,1,headDim] so they broadcast across batch and heads.
func RoPE(x *g.Node, positions []int, base float64, headDim int) *g.Node {
	half := headDim / 2
	t := len(positions)
	cos := make([]float32, t*headDim)
	sin := make([]float32, t*headDim)
	for ti, pos := range positions {
		for i := 0; i < half; i++ {
			invFreq := math.Pow(base, -float64(2*i)/float64(headDim))
			ang := float64(pos) * invFreq
			c, s := float32(math.Cos(ang)), float32(math.Sin(ang))
			// emb = cat([freqs, freqs]) -> the value at i and i+half is the same angle.
			cos[ti*headDim+i], cos[ti*headDim+i+half] = c, c
			sin[ti*headDim+i], sin[ti*headDim+i+half] = s, s
		}
	}
	gr := x.Graph()
	cosN := g.Const(gr, tensors.FromFlatDataAndDimensions(cos, 1, t, 1, headDim))
	sinN := g.Const(gr, tensors.FromFlatDataAndDimensions(sin, 1, t, 1, headDim))
	// rotate_half(x) = cat([-x2, x1]) over last axis.
	x1 := g.Slice(x, g.AxisRange().Spacer(), g.AxisRangeFromStart(half))
	x2 := g.Slice(x, g.AxisRange().Spacer(), g.AxisRangeToEnd(half))
	rot := g.Concatenate([]*g.Node{g.Neg(x2), x1}, -1)
	return g.Add(g.Mul(x, cosN), g.Mul(rot, sinN))
}
```
(Note: `x.Graph()` returns the owning `*g.Graph`; verify the accessor name during build — if it is not `Graph()`, get the graph from an input node per the gomlx source. The const dtype is float32; inputs are float32 in these fixtures.)

- [ ] **Step 4: Run the test — expect PASS**

Run: `cd model && GOMLX_BACKEND=go go test -tags noxla . -run TestRoPEParity -v`
Expected: PASS. If diffs slightly exceed 1e-5 from host-vs-torch trig rounding, relax this block's tol to 5e-5 and note why in the test.

- [ ] **Step 5: Commit**

```bash
git add model/rope.go model/rope_test.go model/testdata/gen_goldens.py model/testdata/rope.json
git commit -m "model: RoPE (rotate-half) with parity"
```

---

### Task 4: SwiGLU FFN

**Files:**
- Modify: `model/testdata/gen_goldens.py` (add `gen_swiglu`)
- Create: `model/testdata/swiglu.json` (generated)
- Create: `model/swiglu.go`
- Create: `model/swiglu_test.go`

**Interfaces:**
- Produces: `model.SwiGLU(x, wGate, wUp, wDown *g.Node) *g.Node` — `(SiLU(x·wGate) * (x·wUp)) · wDown`.

- [ ] **Step 1: Add the generator and regenerate**

Append to `gen_goldens.py` and add to `__main__`:
```python
def gen_swiglu():
    import torch.nn.functional as F
    B, T, H, F_ = 2, 3, 8, 16
    x = torch.randn(B, T, H)
    Wg = torch.randn(H, F_); Wu = torch.randn(H, F_); Wd = torch.randn(F_, H)
    y = (F.silu(x @ Wg) * (x @ Wu)) @ Wd
    write("swiglu", {"hidden": H, "ffn_hidden": F_},
          {"x": x}, {"Wg": Wg, "Wu": Wu, "Wd": Wd}, y)
```
Run: `cd model/testdata && python3 gen_goldens.py` → `wrote swiglu.json`.

- [ ] **Step 2: Write the failing test**

`model/swiglu_test.go`:
```go
package model_test

import (
	"testing"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/model"
	"github.com/guygrigsby/lmkit-go/model/internal/paritytest"
)

func TestSwiGLUParity(t *testing.T) {
	f := paritytest.Load(t, "testdata/swiglu.json")
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	exec := g.MustNewExec(be.Compute(), func(x, wg, wu, wd *g.Node) *g.Node {
		return model.SwiGLU(x, wg, wu, wd)
	})
	out := exec.MustExec1(
		f.Inputs["x"].ToTensor(),
		f.Weights["Wg"].ToTensor(), f.Weights["Wu"].ToTensor(), f.Weights["Wd"].ToTensor())
	paritytest.AssertClose(t, tensors.MustCopyFlatData[float32](out), f.Expected, 1e-5)
}
```
Run → FAIL (`model.SwiGLU` undefined).

- [ ] **Step 3: Implement SwiGLU**

`model/swiglu.go`:
```go
package model

import g "github.com/gomlx/gomlx/core/graph"

// SwiGLU is the Llama gated FFN: (SiLU(x·wGate) * (x·wUp)) · wDown.
// x is [B,T,H]; wGate,wUp are [H,F]; wDown is [F,H]. MatMul batches the leading
// [B,T] dims. SiLU(z) = z * sigmoid(z).
func SwiGLU(x, wGate, wUp, wDown *g.Node) *g.Node {
	gate := g.MatMul(x, wGate)              // [B,T,F]
	up := g.MatMul(x, wUp)                  // [B,T,F]
	silu := g.Mul(gate, g.Sigmoid(gate))   // SiLU
	return g.MatMul(g.Mul(silu, up), wDown) // [B,T,H]
}
```
Run → PASS.

- [ ] **Step 4: Commit**

```bash
git add model/swiglu.go model/swiglu_test.go model/testdata/gen_goldens.py model/testdata/swiglu.json
git commit -m "model: SwiGLU FFN with parity"
```

---

### Task 5: Embeddings (lookup + tied logits)

**Files:**
- Modify: `model/testdata/gen_goldens.py` (add `gen_embedding`)
- Create: `model/testdata/embedding.json` (generated)
- Create: `model/embedding.go`
- Create: `model/embedding_test.go`

**Interfaces:**
- Produces: `model.EmbedLookup(table, tokenIDs *g.Node) *g.Node` (`[V,H]`, `[B,T]` int32 → `[B,T,H]`); `model.TiedLogits(h, table *g.Node) *g.Node` (`[B,T,H]`, `[V,H]` → `[B,T,V]`).

- [ ] **Step 1: Add the generator and regenerate**

Append and add to `__main__`:
```python
def gen_embedding():
    V, H, B, T = 10, 8, 2, 3
    table = torch.randn(V, H)
    ids = torch.randint(0, V, (B, T), dtype=torch.int32)
    embed = table[ids.long()]            # [B,T,H]
    h = torch.randn(B, T, H)
    logits = h @ table.t()               # [B,T,V]
    obj_inputs = {"ids": ids, "h": h}
    write("embedding", {"vocab": V, "hidden": H},
          obj_inputs, {"table": table},
          {"embed": embed, "logits": logits})
```
Note this block has TWO outputs; the generic `write` expects one `expected`. Replace its call with an inline write that stores both under `expected` as a dict — OR (simpler, chosen here) emit two separate `expected_*` via a small variant. Use this exact variant for the embedding block:
```python
def gen_embedding():
    V, H, B, T = 10, 8, 2, 3
    table = torch.randn(V, H)
    ids = torch.randint(0, V, (B, T), dtype=torch.int32)
    embed = table[ids.long()]
    h = torch.randn(B, T, H)
    logits = h @ table.t()
    obj = {"config": {"vocab": V, "hidden": H},
           "inputs": {"ids": t2j(ids), "h": t2j(h)},
           "weights": {"table": t2j(table)},
           "expected_embed": t2j(embed),
           "expected_logits": t2j(logits)}
    with open("embedding.json", "w") as f:
        json.dump(obj, f)
    print("wrote embedding.json")
```
Run → `wrote embedding.json`. (`ids` are int32; `t2j` casts to float32 for JSON — the Go test converts back to int32 when building the index tensor; see Step 3.)

- [ ] **Step 2: Write the failing test**

`model/embedding_test.go`:
```go
package model_test

import (
	"encoding/json"
	"os"
	"testing"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/model"
	"github.com/guygrigsby/lmkit-go/model/internal/paritytest"
)

func TestEmbeddingParity(t *testing.T) {
	// embedding.json has two expected outputs; load the raw shape here.
	raw, err := os.ReadFile("testdata/embedding.json")
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	var fx struct {
		Inputs         map[string]paritytest.Tensor `json:"inputs"`
		Weights        map[string]paritytest.Tensor `json:"weights"`
		ExpectedEmbed  paritytest.Tensor            `json:"expected_embed"`
		ExpectedLogits paritytest.Tensor            `json:"expected_logits"`
	}
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatalf("decode: %v", err)
	}
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}

	// ids were serialized as float32; build an int32 tensor for Gather.
	idsF := fx.Inputs["ids"]
	idsI := make([]int32, len(idsF.Data))
	for i, v := range idsF.Data {
		idsI[i] = int32(v)
	}
	idsT := tensors.FromFlatDataAndDimensions(idsI, idsF.Shape...)

	// Lookup.
	execL := g.MustNewExec(be.Compute(), func(table, ids *g.Node) *g.Node {
		return model.EmbedLookup(table, ids)
	})
	gotEmbed := execL.MustExec1(fx.Weights["table"].ToTensor(), idsT)
	paritytest.AssertClose(t, tensors.MustCopyFlatData[float32](gotEmbed), fx.ExpectedEmbed, 1e-5)

	// Tied logits.
	execT := g.MustNewExec(be.Compute(), func(h, table *g.Node) *g.Node {
		return model.TiedLogits(h, table)
	})
	gotLogits := execT.MustExec1(fx.Inputs["h"].ToTensor(), fx.Weights["table"].ToTensor())
	paritytest.AssertClose(t, tensors.MustCopyFlatData[float32](gotLogits), fx.ExpectedLogits, 1e-5)
}
```
Run → FAIL (`model.EmbedLookup`/`model.TiedLogits` undefined).

- [ ] **Step 3: Implement embeddings**

`model/embedding.go`:
```go
package model

import g "github.com/gomlx/gomlx/core/graph"

// EmbedLookup gathers rows of table [V,H] by tokenIDs [B,T] (int) into [B,T,H].
// Gather indexes leading axes of params by the last axis of indices, so the ids
// are reshaped to [B,T,1] (one indexed axis = axis 0 of the table).
func EmbedLookup(table, tokenIDs *g.Node) *g.Node {
	d := tokenIDs.Shape().Dimensions // [B,T]
	idx := g.Reshape(tokenIDs, append(append([]int{}, d...), 1)...) // [B,T,1]
	return g.Gather(table, idx)                                     // [B,T,H]
}

// TiedLogits projects hidden states h [B,T,H] onto the tied embedding table
// [V,H]: logits[...,v] = h · table[v]. Contract H: einsum bth,vh->btv.
func TiedLogits(h, table *g.Node) *g.Node {
	return g.Einsum("bth,vh->btv", h, table)
}
```
Run → PASS. (If `Gather`'s index convention differs from the doc example, adjust the `idx` shaping at this single call site per the gomlx `ops_sparse.go` source; the parity test is the check.)

- [ ] **Step 4: Commit**

```bash
git add model/embedding.go model/embedding_test.go model/testdata/gen_goldens.py model/testdata/embedding.json
git commit -m "model: tied embeddings (lookup + logits) with parity"
```

---

### Task 6: GQA attention

The full attention block: QKV projections, RoPE on Q/K, KV-head grouping, scaled scores, causal mask, softmax, value aggregation, output projection.

**Files:**
- Modify: `model/testdata/gen_goldens.py` (add `gen_attention`)
- Create: `model/testdata/attention.json` (generated)
- Create: `model/attention.go`
- Create: `model/attention_test.go`

**Interfaces:**
- Consumes: `model.RoPE`.
- Produces: `model.Attention(cfg Config, x, wQ, wK, wV, wO *g.Node, positions []int) *g.Node` — input `[B,T,Hidden]` → output `[B,T,Hidden]`.

- [ ] **Step 1: Add the generator and regenerate**

Append and add to `__main__`:
```python
def gen_attention():
    B, T, H = 2, 4, 8
    nH, nKV, hd, base = 4, 2, 2, 10000.0   # H == nH*hd; nH % nKV == 0
    x = torch.randn(B, T, H)
    Wq = torch.randn(H, nH * hd); Wk = torch.randn(H, nKV * hd)
    Wv = torch.randn(H, nKV * hd); Wo = torch.randn(nH * hd, H)
    positions = torch.arange(T)
    inv_freq = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    emb = torch.cat([positions[:, None].float() * inv_freq[None, :]] * 2, dim=-1)
    cos = emb.cos()[None, :, None, :]; sin = emb.sin()[None, :, None, :]
    def rot(t):
        a, b = t[..., : hd // 2], t[..., hd // 2 :]
        return torch.cat([-b, a], dim=-1)
    q = (x @ Wq).view(B, T, nH, hd)
    k = (x @ Wk).view(B, T, nKV, hd)
    v = (x @ Wv).view(B, T, nKV, hd)
    q = q * cos + rot(q) * sin
    k = k * cos + rot(k) * sin
    rep = nH // nKV
    k = k.repeat_interleave(rep, dim=2)   # [B,T,nH,hd]
    v = v.repeat_interleave(rep, dim=2)
    q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)  # [B,nH,T,hd]
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(hd)                   # [B,nH,T,T]
    mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    probs = (scores + mask).softmax(-1)
    out = (probs @ v).transpose(1, 2).reshape(B, T, nH * hd)             # [B,T,nH*hd]
    y = out @ Wo                                                         # [B,T,H]
    write("attention",
          {"hidden": H, "n_heads": nH, "n_kv_heads": nKV, "head_dim": hd,
           "rope_base": base, "seq_len": T},
          {"x": x}, {"Wq": Wq, "Wk": Wk, "Wv": Wv, "Wo": Wo}, y)
```
Run → `wrote attention.json`.

- [ ] **Step 2: Write the failing test**

`model/attention_test.go`:
```go
package model_test

import (
	"testing"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/model"
	"github.com/guygrigsby/lmkit-go/model/internal/paritytest"
)

func TestAttentionParity(t *testing.T) {
	f := paritytest.Load(t, "testdata/attention.json")
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	cfg := model.Config{
		Hidden:   int(f.Config["hidden"]),
		NHeads:   int(f.Config["n_heads"]),
		NKVHeads: int(f.Config["n_kv_heads"]),
		HeadDim:  int(f.Config["head_dim"]),
		RopeBase: f.Config["rope_base"],
	}
	seqLen := int(f.Config["seq_len"])
	positions := make([]int, seqLen)
	for i := range positions {
		positions[i] = i
	}
	exec := g.MustNewExec(be.Compute(), func(x, wq, wk, wv, wo *g.Node) *g.Node {
		return model.Attention(cfg, x, wq, wk, wv, wo, positions)
	})
	out := exec.MustExec1(
		f.Inputs["x"].ToTensor(),
		f.Weights["Wq"].ToTensor(), f.Weights["Wk"].ToTensor(),
		f.Weights["Wv"].ToTensor(), f.Weights["Wo"].ToTensor())
	paritytest.AssertClose(t, tensors.MustCopyFlatData[float32](out), f.Expected, 5e-5)
}
```
(Attention chains matmul+softmax+matmul, so it uses a 5e-5 tolerance — document this in a comment.) Run → FAIL (`model.Attention` undefined).

- [ ] **Step 3: Implement attention**

`model/attention.go`:
```go
package model

import (
	"math"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
)

// Attention is a Llama GQA attention block. x is [B,T,Hidden]; weights project to
// nHeads/nKVHeads of HeadDim. RoPE is applied to Q and K; KV heads are repeated to
// nHeads; scores are scaled by 1/sqrt(HeadDim) with a causal mask; softmax over
// keys; then value aggregation, head merge, and the output projection.
func Attention(cfg Config, x, wQ, wK, wV, wO *g.Node, positions []int) *g.Node {
	d := x.Shape().Dimensions // [B,T,Hidden]
	b, tt := d[0], d[1]
	hd, nH, nKV := cfg.HeadDim, cfg.NHeads, cfg.NKVHeads

	q := g.Reshape(g.MatMul(x, wQ), b, tt, nH, hd)  // [B,T,nH,hd]
	k := g.Reshape(g.MatMul(x, wK), b, tt, nKV, hd) // [B,T,nKV,hd]
	v := g.Reshape(g.MatMul(x, wV), b, tt, nKV, hd)

	q = RoPE(q, positions, cfg.RopeBase, hd)
	k = RoPE(k, positions, cfg.RopeBase, hd)

	// Repeat KV heads to nH (group size rep) by broadcasting then reshaping.
	rep := nH / nKV
	k = repeatKV(k, b, tt, nKV, rep, hd)
	v = repeatKV(v, b, tt, nKV, rep, hd)

	// scores[B,nH,T,S] = q·kᵀ / sqrt(hd), q,k as [B,T,nH,hd].
	scores := g.Einsum("btnh,bsnh->bnts", q, k)
	scores = g.MulScalar(scores, float32(1.0/math.Sqrt(float64(hd))))
	scores = g.Add(scores, causalMask(x.Graph(), tt)) // [1,1,T,T] broadcasts
	probs := g.Softmax(scores, -1)                     // over S
	// out[B,T,nH,hd] = probs·v.
	out := g.Einsum("bnts,bsnh->btnh", probs, v)
	merged := g.Reshape(out, b, tt, nH*hd) // [B,T,nH*hd]
	return g.MatMul(merged, wO)            // [B,T,Hidden]
}

// repeatKV expands [B,T,nKV,hd] to [B,T,nKV*rep,hd] by repeating each KV head rep
// times (matching torch.repeat_interleave on the head axis).
func repeatKV(x *g.Node, b, tt, nKV, rep, hd int) *g.Node {
	x = g.Reshape(x, b, tt, nKV, 1, hd)
	x = g.BroadcastToDims(x, b, tt, nKV, rep, hd)
	return g.Reshape(x, b, tt, nKV*rep, hd)
}

// causalMask returns a [1,1,T,T] additive mask: 0 on/below the diagonal, -inf above.
func causalMask(gr *g.Graph, t int) *g.Node {
	m := make([]float32, t*t)
	neg := float32(math.Inf(-1))
	for i := 0; i < t; i++ {
		for j := 0; j < t; j++ {
			if j > i {
				m[i*t+j] = neg
			}
		}
	}
	return g.Const(gr, tensors.FromFlatDataAndDimensions(m, 1, 1, t, t))
}
```
(Verify during build: `x.Graph()` accessor name; `g.BroadcastToDims` repeat semantics match `repeat_interleave` — the reshape-broadcast-reshape pattern interleaves correctly because the `rep` axis is inserted directly after the head axis. If `BroadcastToDims` won't broadcast a size-1 interior axis, use `g.Concatenate` of `rep` copies instead.)

- [ ] **Step 4: Run the test — expect PASS**

Run: `cd model && GOMLX_BACKEND=go go test -tags noxla . -run TestAttentionParity -v`
Expected: PASS at 5e-5. If it fails, the likely culprits in order: RoPE head-axis layout (q is [B,T,nH,hd] when RoPE is applied — matches the generator), KV-repeat interleave order, or the einsum index letters. Compare intermediate shapes against the generator; do not loosen tolerance to mask a real mismatch.

- [ ] **Step 5: Run the full model suite + gate**

Run: `cd model && GOMLX_BACKEND=go go test -tags noxla ./... -v` then `make check`
Expected: all five parity tests pass; `make check` prints `OK`.

- [ ] **Step 6: Commit**

```bash
git add model/attention.go model/attention_test.go model/testdata/gen_goldens.py model/testdata/attention.json
git commit -m "model: GQA attention with parity"
```

---

## Self-Review

**Spec coverage** (against `2026-06-19-milestone-2-llama-parity-design.md`):
- model module + go.work + Config → Task 2.
- RMSNorm / RoPE / GQA attention / SwiGLU / embeddings, each parity-tested → Tasks 2/3/6/4/5.
- committed PyTorch generator + JSON goldens → Tasks 2–6 (generator grows per block).
- runtime boundary test covers model (ADR-0009) → Task 1 (revised, runtime-specific) + the model code stays op-vocabulary-only.
- backend executor for tests (one-method add) → `gomlx.Compute()` in Task 2; tests use it without naming the runtime type.
- gate extended to model → Task 2 (check.sh) + Task 6 (full run).
- fp32 parity, documented tolerances (1e-5, attention 5e-5) → per-task asserts.

**Out of scope, correctly absent:** full decoder assembly, training/optimizer-on-model, KV cache, FlashAttention, real lm-100m-en config/weights, bf16 (the optional smoke is dropped — YAGNI; add in M3 if wanted).

**Placeholder scan:** every block ships complete Go + Python code. Three call sites carry an explicit "verify against source if it doesn't resolve" note (`x.Graph()` accessor, `Gather` index convention, `BroadcastToDims` repeat) — these are precise, named, single-line verifications with the fallback stated, not vague placeholders, and are inherent to building on a tracked-commit API.

**Type consistency:** `model.Config` fields, `paritytest.{Tensor,Fixture,Load,AssertClose}`, `(*gomlx.Backend).Compute()`, and the block signatures (`RMSNorm`, `RoPE`, `SwiGLU`, `EmbedLookup`, `TiedLogits`, `Attention`) are used identically across the generator, blocks, and tests. `RoPE` is consumed by `Attention` with the same signature defined in Task 3.
</content>
