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
	// GradSumSquares returns d/dx of sum(x²), i.e. 2x. Proves reverse-mode autodiff.
	GradSumSquares(x Tensor) (Tensor, error)
}
