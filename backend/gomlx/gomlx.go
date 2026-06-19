// Package gomlx adapts GoMLX/XLA to backend.Backend. This is the ONLY package in
// lmkit-go permitted to import gomlx/go-xla/compute/pjrt (ADR-0002).
package gomlx

import (
	"fmt"

	"github.com/gomlx/compute"
	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/dataset"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/optimizer"

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

func (b *Backend) GradSumSquares(x backend.Tensor) (backend.Tensor, error) {
	xt := tensors.FromFlatDataAndDimensions(x.Data, x.Shape...)
	exec := g.MustNewExec(b.be, func(n *g.Node) *g.Node {
		sumSq := g.ReduceAllSum(g.Mul(n, n)) // scalar loss
		return g.Gradient(sumSq, n)[0]       // d(sumSq)/dn = 2n
	})
	out := exec.MustExec1(xt)
	return backend.Tensor{Shape: x.Shape, Data: tensors.MustCopyFlatData[float32](out)}, nil
}

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
	opt := optimizer.Adam().WeightDecay(0.0).LearningRate(5e-2).Done() // AdamW path; lr=5e-2 ensures convergence from random init
	trainer := train.NewTrainer(b.be, store, modelFn, loss.MeanSquaredError, opt, nil, nil)
	loop := train.NewLoop(trainer)
	if _, err := loop.RunSteps(ds, steps); err != nil {
		return 0, 0, fmt.Errorf("backend/gomlx: train: %w", err)
	}

	w := tensors.MustCopyFlatData[float32](store.GetVariable("/dense/weights").MustValue())[0]
	finalLoss := (w - target) * (w - target) // MSE of y=w*1 vs target, computed in Go
	return w, finalLoss, nil
}

// compile-time check that the adapter satisfies the boundary.
var _ backend.Backend = (*Backend)(nil)
