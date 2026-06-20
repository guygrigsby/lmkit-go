# Milestone 1 — backend bring-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that GoMLX/XLA, driven only through the lmkit-go `backend` boundary, can do a correct matmul, a correct gradient, and an AdamW optimizer step — on CPU locally (SimpleGo) and on CUDA on a CUDA GPU host.

**Architecture:** A per-package-module monorepo. The `backend` module defines lmkit-go's own tensor/device types and a tiny `Backend` interface; the only package allowed to import GoMLX is the `backend/gomlx` adapter. An `app` module ships a `lmkit quickstart` that runs the three proofs. Contract tests pin the numerics so a GoMLX `main` bump breaks here, in one place.

**Tech Stack:** Go 1.26, GoMLX `main` (`github.com/gomlx/gomlx`) on the `github.com/gomlx/compute` + `github.com/gomlx/go-xla` modules. Pure-Go `SimpleGo` backend for local CPU; XLA CUDA on a CUDA GPU host.

## Global Constraints

- **Go 1.26+ toolchain** (GoMLX `main` requires `go 1.26`).
- **Pin GoMLX to a fixed `main` commit** — `backend/go.mod` pins commit `516689cbe91329bf9aeac7750495a48bb85f9287` (the most recent `main` verified for this plan, 2026-06-19); `go mod tidy` resolves matching `compute`/`go-xla`. Bump to latest `@main` only *after* bring-up is green (ADR-0001, ADR-0006) — pin first, move the pin once something works.
- **Backend boundary (ADR-0002):** only files under `backend/**` may import `gomlx`, `go-xla`, `compute`, or `pjrt`. Enforced by a grep test in the gate. `model`/`train`/`app` never import the vendor.
- **Per-package modules + `go.work` (ADR-0005):** each context is its own module; `internal/` exists only in `app`. Milestone 1 creates only the `backend` and `app` modules.
- **Public API is additive (ADR-0008):** the `Backend` interface stays minimal; under-expose now, grow later. Method names are specific to what they prove.
- **Test rigor (ADR-0007):** contract tests pin numerical output; the green gate is the merge bar for every change including dep bumps.
- **Commits:** terse, verb-first, no dashes, no Claude/Anthropic attribution (no `Co-Authored-By: Claude`, no "Generated with").
- **Platforms:** Local dev runs SimpleGo with `-tags noxla` + `GOMLX_BACKEND=go` (no XLA/PJRT libraries needed). XLA-CPU/CUDA build *without* `noxla` and need the XLA libs/PJRT plugin. CUDA runs on a CUDA GPU host (`$GPU_HOST`); check the GPU is idle with `nvidia-smi` first; never edit source on the box — it pulls.

## Verified GoMLX `main` API reference (read against commit ~`516689c`, 2026-06-19)

Use these exact symbols. pkg.go.dev is stale.

```go
// backend construction (separate module github.com/gomlx/compute)
import "github.com/gomlx/compute"
import _ "github.com/gomlx/gomlx/backends/default" // registers gobackend (SimpleGo) + xla
be, err := compute.New()        // honors GOMLX_BACKEND ("go", "xla:cpu", "xla:cuda"); be is compute.Backend
be.Name() // e.g. "go" or "xla"

// graph + tensors (moved to core/)
import g "github.com/gomlx/gomlx/core/graph"
import "github.com/gomlx/gomlx/core/tensors"
exec := g.MustNewExec(be, func(a, b *g.Node) *g.Node { return g.MatMul(a, b) })
out  := exec.MustExec1(t1, t2)                     // *tensors.Tensor (one output)
t    := tensors.FromFlatDataAndDimensions(data, dims...) // data []float32, dims ...int
flat := tensors.MustCopyFlatData[float32](out)     // []float32
// ops: g.MatMul, g.Mul, g.ReduceAllSum(x), g.Gradient(scalarOut, nodes...) []*g.Node

// optimizer + model state + training (ml/...)
import "github.com/gomlx/gomlx/ml/model"
import "github.com/gomlx/gomlx/ml/dataset"
import "github.com/gomlx/gomlx/ml/layers"
import "github.com/gomlx/gomlx/ml/train"
import "github.com/gomlx/gomlx/ml/train/loss"
import "github.com/gomlx/gomlx/ml/train/optimizer"
store := model.NewStore()
store.SetParam(optimizer.ParamLearningRate, 1e-2)
opt := optimizer.Adam().WeightDecay(0.0).LearningRate(1e-2).Done() // AdamW; Interface
modelFn := func(scope *model.Scope, spec any, inputs []*g.Node) []*g.Node {
    return []*g.Node{layers.Dense(scope, inputs[0], false /*useBias*/, 1 /*outDim*/)}
}
ds, _ := dataset.InMemoryFromData(be, "name", []any{inputsTensor}, []any{labelsTensor})
ds = ds.Infinite(true).Shuffle().BatchSize(n, false)
trainer := train.NewTrainer(be, store, modelFn, loss.MeanSquaredError, opt, nil, nil) // 7 args
loop := train.NewLoop(trainer)
metrics, err := loop.RunSteps(ds, steps)           // []*tensors.Tensor, error
w := tensors.MustCopyFlatData[float32](store.GetVariable("/dense/weights").MustValue())
// loss.MeanSquaredError signature: func(labels, predictions []*g.Node) *g.Node  (labels FIRST)
```

**Three points to confirm during Task 1's first real build** (cite source if they drift; fix at the single call site):
1. `store.GetVariable("/dense/weights")` path — taken from `examples/linear/linear.go`; confirm `layers.Dense` uses the `/dense/weights` scope name.
2. The dataset chain returns `*dataset.InMemoryDataset` (assign the result; don't rely on in-place mutation).
3. `compute.New()` returns `(compute.Backend, error)`; `MustNewExec`, `InMemoryFromData`, `NewTrainer` all accept `compute.Backend`.

---

### Task 1: Monorepo skeleton + backend boundary + MatMul on SimpleGo

Stands up the modules, wires GoMLX `main`, and proves the bet builds and runs: a correct matmul through the lmkit-go boundary on the pure-Go SimpleGo backend.

**Files:**
- Create: `go.work`
- Create: `backend/go.mod`
- Create: `backend/backend.go`
- Create: `backend/gomlx/gomlx.go`
- Test: `backend/gomlx/gomlx_test.go`

**Interfaces:**
- Produces: `backend.DType`, `backend.Device{Kind,Config string}`, `backend.Tensor{Shape []int; Data []float32}`, `backend.Backend` interface with `Device() Device` and `MatMul(a, b Tensor) (Tensor, error)`. Adapter: `gomlx.New() (*gomlx.Backend, error)` implementing `backend.Backend`.

- [ ] **Step 1: Create the workspace and backend module**

`go.work`:
```
go 1.26

use (
	./backend
	./app
)
```

`backend/go.mod`:
```
module github.com/guygrigsby/lmkit-go/backend

go 1.26
```

Then pin GoMLX to the fixed `main` commit (resolves the canonical pseudo-version + `compute`/`go-xla`):
```bash
cd backend && go get github.com/gomlx/gomlx@516689cbe91329bf9aeac7750495a48bb85f9287 && cd ..
```
Expected: `backend/go.mod` gains `require github.com/gomlx/gomlx v0.0.0-20260619...-516689cbe913` (and `compute`/`go-xla` once code imports them). Do **not** use `@main` — pin this commit until bring-up is green, then bump.

- [ ] **Step 2: Write the backend domain types and interface**

`backend/backend.go`:
```go
// Package backend is lmkit-go's anti-corruption boundary over an XLA-capable
// compute stack (ADR-0002). Nothing in this package imports gomlx/go-xla/compute;
// the adapter in backend/gomlx does. The interface is grown additively (ADR-0008):
// add methods later, never remove or re-sign them within a major version.
package backend

// DType is the element type of a tensor.
type DType int

const (
	Float32 DType = iota
	BFloat16
)

// Device identifies a compute target, e.g. {"go",""} (SimpleGo, pure Go),
// {"xla","cpu"}, or {"xla","cuda"}.
type Device struct {
	Kind   string
	Config string
}

// Tensor is a dense row-major float32 tensor in lmkit-go's own terms.
type Tensor struct {
	Shape []int
	Data  []float32
}

// Backend drives the underlying compute stack through lmkit-go types only.
type Backend interface {
	// Device reports the selected compute target.
	Device() Device
	// MatMul returns a@b for 2-D matrices (a: m×k, b: k×n -> m×n).
	MatMul(a, b Tensor) (Tensor, error)
}
```

- [ ] **Step 3: Write the failing matmul contract test**

`backend/gomlx/gomlx_test.go`:
```go
package gomlx

import (
	"testing"

	"github.com/guygrigsby/lmkit-go/backend"
)

func TestMatMul(t *testing.T) {
	be, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	// [2x3] @ [3x2] = [2x2]; known product [58 64 139 154].
	a := backend.Tensor{Shape: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}}
	b := backend.Tensor{Shape: []int{3, 2}, Data: []float32{7, 8, 9, 10, 11, 12}}
	got, err := be.MatMul(a, b)
	if err != nil {
		t.Fatalf("MatMul: %v", err)
	}
	want := []float32{58, 64, 139, 154}
	if len(got.Data) != len(want) {
		t.Fatalf("len = %d, want %d", len(got.Data), len(want))
	}
	for i, w := range want {
		if got.Data[i] != w {
			t.Errorf("Data[%d] = %v, want %v", i, got.Data[i], w)
		}
	}
	if got.Shape[0] != 2 || got.Shape[1] != 2 {
		t.Errorf("Shape = %v, want [2 2]", got.Shape)
	}
}
```

- [ ] **Step 4: Run the test to verify it fails**

Run: `cd backend && GOMLX_BACKEND=go go test -tags noxla ./gomlx/ -run TestMatMul -v`
Expected: FAIL — `New` / package `gomlx` undefined (adapter not written yet).

- [ ] **Step 5: Write the gomlx adapter (MatMul)**

`backend/gomlx/gomlx.go`:
```go
// Package gomlx adapts GoMLX/XLA to backend.Backend. This is the ONLY package in
// lmkit-go permitted to import gomlx/go-xla/compute/pjrt (ADR-0002).
package gomlx

import (
	"fmt"

	"github.com/gomlx/compute"
	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"

	_ "github.com/gomlx/gomlx/backends/default" // registers gobackend (SimpleGo) + xla

	"github.com/guygrigsby/lmkit-go/backend"
)

// Backend implements backend.Backend over GoMLX/XLA.
type Backend struct {
	be compute.Backend
}

// New constructs a Backend. Device selection honors the GOMLX_BACKEND env var
// ("go" = SimpleGo, "xla:cpu", "cuda"); compute.New applies it.
func New() (*Backend, error) {
	be, err := compute.New()
	if err != nil {
		return nil, fmt.Errorf("backend/gomlx: new: %w", err)
	}
	return &Backend{be: be}, nil
}

func (b *Backend) Device() backend.Device {
	return backend.Device{Kind: b.be.Name(), Config: b.be.Description()}
}

func (b *Backend) MatMul(a, c backend.Tensor) (backend.Tensor, error) {
	if len(a.Shape) != 2 || len(c.Shape) != 2 {
		return backend.Tensor{}, fmt.Errorf("backend/gomlx: MatMul wants 2-D, got %v and %v", a.Shape, c.Shape)
	}
	at := tensors.FromFlatDataAndDimensions(a.Data, a.Shape...)
	ct := tensors.FromFlatDataAndDimensions(c.Data, c.Shape...)
	exec := g.MustNewExec(b.be, func(x, y *g.Node) *g.Node { return g.MatMul(x, y) })
	out := exec.MustExec1(at, ct)
	return backend.Tensor{
		Shape: []int{a.Shape[0], c.Shape[1]},
		Data:  tensors.MustCopyFlatData[float32](out),
	}, nil
}

// compile-time check that the adapter satisfies the boundary.
var _ backend.Backend = (*Backend)(nil)
```

- [ ] **Step 6: Tidy and run the test to verify it passes**

Run: `cd backend && go mod tidy && GOMLX_BACKEND=go go test -tags noxla ./gomlx/ -run TestMatMul -v`
Expected: PASS. (`go mod tidy` pulls `compute`/`go-xla`. If a signature drifted, fix at the single cited call site per the API reference notes.)

- [ ] **Step 7: Commit**

```bash
git add go.work backend/
git commit -m "backend: matmul through the boundary on SimpleGo

Stand up the per-package-module monorepo (go.work + backend module pinned
to gomlx main). Define the backend domain types and the minimal Backend
interface; the gomlx adapter is the only package importing the vendor.
Contract test pins the 2x3@3x2 product."
```

---

### Task 2: Gradient through the boundary

Proves reverse-mode autodiff: d/dx of sum(x²) is 2x.

**Files:**
- Modify: `backend/backend.go` (add `GradSumSquares` to the interface)
- Modify: `backend/gomlx/gomlx.go` (implement it)
- Test: `backend/gomlx/gomlx_test.go` (add `TestGradSumSquares`)

**Interfaces:**
- Produces: `Backend.GradSumSquares(x Tensor) (Tensor, error)` returning 2x with x's shape.

- [ ] **Step 1: Write the failing gradient test**

Add to `backend/gomlx/gomlx_test.go`:
```go
func TestGradSumSquares(t *testing.T) {
	be, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	x := backend.Tensor{Shape: []int{3}, Data: []float32{1, 2, 3}}
	got, err := be.GradSumSquares(x)
	if err != nil {
		t.Fatalf("GradSumSquares: %v", err)
	}
	want := []float32{2, 4, 6} // d/dx sum(x^2) = 2x
	for i, w := range want {
		if got.Data[i] != w {
			t.Errorf("Data[%d] = %v, want %v", i, got.Data[i], w)
		}
	}
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && GOMLX_BACKEND=go go test -tags noxla ./gomlx/ -run TestGradSumSquares -v`
Expected: FAIL — `GradSumSquares` not in the interface / not implemented.

- [ ] **Step 3: Add the method to the interface**

In `backend/backend.go`, add to the `Backend` interface (additive, ADR-0008):
```go
	// GradSumSquares returns d/dx of sum(x²), i.e. 2x. Proves reverse-mode autodiff.
	GradSumSquares(x Tensor) (Tensor, error)
```

- [ ] **Step 4: Implement it in the adapter**

Add to `backend/gomlx/gomlx.go`:
```go
func (b *Backend) GradSumSquares(x backend.Tensor) (backend.Tensor, error) {
	xt := tensors.FromFlatDataAndDimensions(x.Data, x.Shape...)
	exec := g.MustNewExec(b.be, func(n *g.Node) *g.Node {
		sumSq := g.ReduceAllSum(g.Mul(n, n)) // scalar loss
		return g.Gradient(sumSq, n)[0]       // d(sumSq)/dn = 2n
	})
	out := exec.MustExec1(xt)
	return backend.Tensor{Shape: x.Shape, Data: tensors.MustCopyFlatData[float32](out)}, nil
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd backend && GOMLX_BACKEND=go go test -tags noxla ./gomlx/ -run TestGradSumSquares -v`
Expected: PASS. (If `ReduceAllSum` is unresolved, use `g.ReduceSum(g.Mul(n, n))` — both sum all axes; confirm in `core/graph/ops.go`.)

- [ ] **Step 6: Commit**

```bash
git add backend/
git commit -m "backend: gradient through the boundary

Add GradSumSquares; adapter wires gomlx Gradient. Contract test pins
d/dx sum(x^2) = 2x, proving reverse-mode autodiff behind the boundary."
```

---

### Task 3: AdamW optimizer step through the boundary

Proves the optimizer plumbing: AdamW drives a single weight to a target.

**Files:**
- Modify: `backend/backend.go` (add `FitConstant`)
- Modify: `backend/gomlx/gomlx.go` (implement it)
- Test: `backend/gomlx/gomlx_test.go` (add `TestFitConstant`)

**Interfaces:**
- Produces: `Backend.FitConstant(target float32, steps int) (w, loss float32, err error)`.

- [ ] **Step 1: Write the failing optimizer test**

Add to `backend/gomlx/gomlx_test.go`:
```go
import "math" // add to the test file's import block

func TestFitConstant(t *testing.T) {
	be, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	const target = 3.0
	w, loss, err := be.FitConstant(target, 800)
	if err != nil {
		t.Fatalf("FitConstant: %v", err)
	}
	if math.Abs(float64(w-target)) > 0.05 {
		t.Errorf("w = %v, want ~%v", w, target)
	}
	if loss > 1e-2 {
		t.Errorf("loss = %v, want < 1e-2", loss)
	}
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && GOMLX_BACKEND=go go test -tags noxla ./gomlx/ -run TestFitConstant -v`
Expected: FAIL — `FitConstant` not in the interface / not implemented.

- [ ] **Step 3: Add the method to the interface**

In `backend/backend.go`, add to `Backend`:
```go
	// FitConstant fits a single weight w (no bias) to target via AdamW for the
	// given steps, returning the learned w and the final MSE loss. Proves the
	// optimizer path.
	FitConstant(target float32, steps int) (w, loss float32, err error)
```

- [ ] **Step 4: Implement it in the adapter**

Add the imports to `backend/gomlx/gomlx.go`:
```go
	"github.com/gomlx/gomlx/ml/dataset"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/optimizer"
```
Add the method:
```go
func (b *Backend) FitConstant(target float32, steps int) (float32, float32, error) {
	// Inputs are all ones, labels all `target`; with a no-bias 1-unit Dense,
	// y = w*1, so MSE drives the single weight w -> target.
	const n = 64
	in := make([]float32, n)
	lab := make([]float32, n)
	for i := range in {
		in[i] = 1
		lab[i] = target
	}
	inputs := tensors.FromFlatDataAndDimensions(in, n, 1)
	labels := tensors.FromFlatDataAndDimensions(lab, n, 1)

	ds, err := dataset.InMemoryFromData(b.be, "fit-constant", []any{inputs}, []any{labels})
	if err != nil {
		return 0, 0, fmt.Errorf("backend/gomlx: dataset: %w", err)
	}
	ds = ds.Infinite(true).Shuffle().BatchSize(n, false)

	store := model.NewStore()
	store.SetParam(optimizer.ParamLearningRate, 1e-2)

	modelFn := func(scope *model.Scope, spec any, inputs []*g.Node) []*g.Node {
		return []*g.Node{layers.Dense(scope, inputs[0], false /*useBias*/, 1)}
	}
	opt := optimizer.Adam().WeightDecay(0.0).LearningRate(1e-2).Done() // AdamW path
	trainer := train.NewTrainer(b.be, store, modelFn, loss.MeanSquaredError, opt, nil, nil)
	loop := train.NewLoop(trainer)
	if _, err := loop.RunSteps(ds, steps); err != nil {
		return 0, 0, fmt.Errorf("backend/gomlx: train: %w", err)
	}

	w := tensors.MustCopyFlatData[float32](store.GetVariable("/dense/weights").MustValue())[0]
	loss := (w - target) * (w - target) // MSE of y=w*1 vs target, computed in Go
	return w, loss, nil
}
```

- [ ] **Step 5: Run to verify it passes**

Run: `cd backend && go mod tidy && GOMLX_BACKEND=go go test -tags noxla ./gomlx/ -run TestFitConstant -v`
Expected: PASS — `w ≈ 3.0`, `loss < 1e-2`. (If the weight reads back as ~0, confirm the `/dense/weights` scope path against `examples/linear/linear.go` and adjust the one string.)

- [ ] **Step 6: Run the full contract suite**

Run: `cd backend && GOMLX_BACKEND=go go test -tags noxla ./... -v`
Expected: PASS — TestMatMul, TestGradSumSquares, TestFitConstant.

- [ ] **Step 7: Commit**

```bash
git add backend/
git commit -m "backend: AdamW step through the boundary

Add FitConstant; adapter wires gomlx Trainer + Adam().WeightDecay (AdamW).
Contract test drives a single weight to target, proving the optimizer
plumbing. Backend bring-up is numerically green on SimpleGo."
```

---

### Task 4: Quickstart binary + boundary grep test + the gate

Ships the `lmkit quickstart` that runs all three proofs and prints the device, plus the structural boundary test and the build/test gate.

**Files:**
- Create: `app/go.mod`
- Create: `app/cmd/lmkit/main.go`
- Create: `app/boundary_test.go`
- Create: `scripts/check.sh`
- Create: `Makefile`

**Interfaces:**
- Consumes: `gomlx.New`, `backend.Backend` (MatMul, GradSumSquares, FitConstant, Device).

- [ ] **Step 1: Create the app module**

`app/go.mod`:
```
module github.com/guygrigsby/lmkit-go/app

go 1.26
```
Then wire the local backend module via the workspace (already in `go.work`); resolve the require:
```bash
cd app && go get github.com/guygrigsby/lmkit-go/backend && cd ..
```
(With `go.work` present, the local `./backend` is used; the `require` is recorded for non-workspace builds.)

- [ ] **Step 2: Write the quickstart**

`app/cmd/lmkit/main.go`:
```go
// Command lmkit is the lmkit-go CLI. For Milestone 1 it offers `quickstart`,
// which runs the backend bring-up proofs and prints the selected device.
package main

import (
	"fmt"
	"math"
	"os"

	"github.com/guygrigsby/lmkit-go/backend"
	"github.com/guygrigsby/lmkit-go/backend/gomlx"
)

func main() {
	if len(os.Args) < 2 || os.Args[1] != "quickstart" {
		fmt.Fprintln(os.Stderr, "usage: lmkit quickstart")
		os.Exit(2)
	}
	if err := quickstart(); err != nil {
		fmt.Fprintf(os.Stderr, "FAIL: %v\n", err)
		os.Exit(1)
	}
}

func quickstart() error {
	be, err := gomlx.New()
	if err != nil {
		return err
	}
	dev := be.Device()
	fmt.Printf("device: kind=%q config=%q\n", dev.Kind, dev.Config)

	mm, err := be.MatMul(
		backend.Tensor{Shape: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}},
		backend.Tensor{Shape: []int{3, 2}, Data: []float32{7, 8, 9, 10, 11, 12}},
	)
	if err != nil {
		return err
	}
	if !eq(mm.Data, []float32{58, 64, 139, 154}) {
		return fmt.Errorf("matmul = %v, want [58 64 139 154]", mm.Data)
	}
	fmt.Printf("matmul   OK  %v\n", mm.Data)

	gr, err := be.GradSumSquares(backend.Tensor{Shape: []int{3}, Data: []float32{1, 2, 3}})
	if err != nil {
		return err
	}
	if !eq(gr.Data, []float32{2, 4, 6}) {
		return fmt.Errorf("grad = %v, want [2 4 6]", gr.Data)
	}
	fmt.Printf("gradient OK  %v\n", gr.Data)

	w, loss, err := be.FitConstant(3.0, 800)
	if err != nil {
		return err
	}
	if math.Abs(float64(w-3.0)) > 0.05 || loss > 1e-2 {
		return fmt.Errorf("adamw w=%v loss=%v, want w~3 loss<1e-2", w, loss)
	}
	fmt.Printf("adamw    OK  w=%.4f loss=%.2e\n", w, loss)
	return nil
}

func eq(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
```

- [ ] **Step 3: Run the quickstart (the real path)**

Run: `GOMLX_BACKEND=go go run -tags noxla ./app/cmd/lmkit quickstart`
Expected output (device line, then three OK lines):
```
device: kind="go" config=...
matmul   OK  [58 64 139 154]
gradient OK  [2 4 6]
adamw    OK  w=3.0000 loss=...e-...
```

- [ ] **Step 4: Write the boundary grep test**

`app/boundary_test.go`:
```go
package app_test

import (
	"os/exec"
	"strings"
	"testing"
)

// TestBackendBoundary enforces ADR-0002: only files under backend/ may import the
// vendor stack (everything ships under github.com/gomlx). It greps the whole repo
// from the root and excludes itself (this file names the vendor path in a string).
func TestBackendBoundary(t *testing.T) {
	root, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		t.Fatalf("git root: %v", err)
	}
	cmd := exec.Command("git", "grep", "-lE", `github\.com/gomlx`,
		"--", "*.go", ":!app/boundary_test.go")
	cmd.Dir = strings.TrimSpace(string(root))
	out, _ := cmd.Output() // exit 1 (no matches) yields empty output; that's a pass
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "backend/") {
			t.Errorf("vendor import outside backend/: %s", line)
		}
	}
}
```

- [ ] **Step 5: Run the boundary test to verify it passes**

Run: `cd "$(git rev-parse --show-toplevel)" && go test -tags noxla ./app/ -run TestBackendBoundary -v`
Expected: PASS — every vendor reference is under `backend/`.

- [ ] **Step 6: Write the gate script and Makefile**

`scripts/check.sh`:
```bash
#!/usr/bin/env bash
# Build and test every module on the pure-Go SimpleGo backend (no XLA libs).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
for mod in backend app; do
  echo "== $mod =="
  ( cd "$mod" && go build -tags noxla ./... && GOMLX_BACKEND=go go test -tags noxla ./... )
done
echo "== boundary =="
go test -tags noxla ./app/ -run TestBackendBoundary
echo "OK"
```
`Makefile`:
```make
.PHONY: check
check:
	@bash scripts/check.sh
```
Then: `chmod +x scripts/check.sh`

- [ ] **Step 7: Run the full gate**

Run: `make check`
Expected: each module builds, all tests pass, boundary clean, prints `OK`.

- [ ] **Step 8: Commit**

```bash
git add app/ scripts/ Makefile
git commit -m "app: lmkit quickstart + boundary test + gate

quickstart runs the three bring-up proofs and prints the device. Boundary
test enforces ADR-0002 (vendor imports only under backend/). check.sh +
Makefile build and test every module on SimpleGo."
```

---

### Task 5: Validate on CUDA (GPU host) — DONE 2026-06-19

Confirms the same quickstart runs on the NVIDIA path. No new code — environment
validation. A green SimpleGo run is not a green CUDA run; both are required.

**Deploy method (actual):** cross-compile a binary locally and rsync it to
`"$GPU_HOST":/tmp`; run it there; clean up. No work repo on the box (artifact-only
deploy — a `/tmp` binary is fine, a source checkout is not). `$GPU_HOST` = your CUDA host.
The GPU was confirmed idle via `nvidia-smi` before the (~1s) run.

- [x] **Step 1: Cross-compile for linux/amd64 locally**

SimpleGo (pure Go, no cgo) — proves portability:
```bash
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -tags noxla -o /tmp/lmkit-linux-amd64 ./app/cmd/lmkit
```
CUDA-capable (the `xla` backend needs cgo; cross-compile the C with `zig cc`):
```bash
CC="zig cc -target x86_64-linux-gnu" CXX="zig c++ -target x86_64-linux-gnu" \
CGO_ENABLED=1 GOOS=linux GOARCH=amd64 go build -o /tmp/lmkit-cuda-amd64 ./app/cmd/lmkit
```

- [x] **Step 2: Run SimpleGo on the GPU host**

```bash
rsync -az /tmp/lmkit-linux-amd64 "$GPU_HOST":/tmp/
ssh "$GPU_HOST" 'GOMLX_BACKEND=go /tmp/lmkit-linux-amd64 quickstart'
```
Result: green (`device: "Go Backend"`, three OK lines).

- [x] **Step 3: Run CUDA on the GPU host**

```bash
rsync -az /tmp/lmkit-cuda-amd64 "$GPU_HOST":/tmp/
ssh "$GPU_HOST" 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader'  # confirm idle
ssh "$GPU_HOST" 'GOMLX_BACKEND=xla:cuda /tmp/lmkit-cuda-amd64 quickstart'
```
Note the selector is **`xla:cuda`**, not `cuda` (the backend is `xla`; CUDA is its
PJRT plugin). go-xla auto-installs the `cuda` PJRT plugin to `~/.local/lib` on
first run. Result: green on the NVIDIA GPU (device `xla:cuda … v0.112 [1
device(s)]`, three OK lines, `loss=1.42e-12`).

- [x] **Step 4: Clean up and record**

```bash
ssh "$GPU_HOST" 'rm -f /tmp/lmkit-cuda-amd64 /tmp/lmkit-linux-amd64'
rm -f /tmp/lmkit-cuda-amd64 /tmp/lmkit-linux-amd64
```
Results recorded in the Milestone 1 spec's "Validation results" section.

---

## Self-Review

**Spec coverage** (against `2026-06-19-milestone-1-bringup-design.md`):
- go.work + backend & app modules, backend pinned to gomlx main → Task 1.
- backend domain types + minimal interface → Tasks 1–3 (grown additively).
- single gomlx adapter, only vendor importer → Tasks 1–3; enforced Task 4.
- quickstart printing checks + device → Task 4.
- matmul / gradient / AdamW correctness tests → Tasks 1/2/3.
- boundary grep test wired into the gate → Task 4.
- boundary contract tests (ADR-0007) → the three numerical tests are exactly these.
- device selection SimpleGo/CPU/CUDA via one knob (`GOMLX_BACKEND`) → Tasks 1–5.
- run locally (SimpleGo) and on the CUDA GPU host → Tasks 4 and 5.
- Done criteria → covered by Tasks 1–5.

**Out of scope, correctly absent:** Llama blocks, training loop proper, tokenizer/data, Metal/ROCm, full bf16 validation. A bf16 smoke (spec open question) is deferred — note it as the first follow-up in Milestone 2 rather than expanding M1.

**Placeholder scan:** code blocks are complete and use only source-verified symbols; the three API-confirmation points are precise (named symbol + source file), not vague.

**Type consistency:** `backend.Tensor{Shape,Data}`, `backend.Device{Kind,Config}`, `gomlx.New() (*Backend,error)`, and the interface methods `MatMul`/`GradSumSquares`/`FitConstant`/`Device` are used identically across `backend.go`, the adapter, the tests, and the quickstart.
</content>
