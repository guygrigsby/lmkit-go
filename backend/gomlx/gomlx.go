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

func (b *Backend) GradSumSquares(x backend.Tensor) (backend.Tensor, error) {
	xt := tensors.FromFlatDataAndDimensions(x.Data, x.Shape...)
	exec := g.MustNewExec(b.be, func(n *g.Node) *g.Node {
		sumSq := g.ReduceAllSum(g.Mul(n, n)) // scalar loss
		return g.Gradient(sumSq, n)[0]       // d(sumSq)/dn = 2n
	})
	out := exec.MustExec1(xt)
	return backend.Tensor{Shape: x.Shape, Data: tensors.MustCopyFlatData[float32](out)}, nil
}

// compile-time check that the adapter satisfies the boundary.
var _ backend.Backend = (*Backend)(nil)
