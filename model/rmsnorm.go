package model

import (
	"github.com/gomlx/compute/dtypes"
	g "github.com/gomlx/gomlx/core/graph"
)

// RMSNorm normalizes x [B,T,H] over its last axis and scales by scale [H].
// The mean/rsqrt run in fp32 (upcast) even for bf16 inputs, matching the baseline's
// fp32-internal norm; the normalized result is returned in x's dtype.
// Uses Reciprocal(Sqrt(...)) instead of Rsqrt: Rsqrt has no gradient defined
// in GoMLX autodiff, so it cannot be used in a training graph.
func RMSNorm(x, scale *g.Node, eps float32) *g.Node {
	inDT := x.DType()
	xf := x
	if inDT != dtypes.Float32 {
		xf = g.ConvertDType(x, dtypes.Float32)
	}
	ms := g.ReduceAndKeep(g.Mul(xf, xf), g.ReduceMean, -1) // [B,T,1] fp32
	inv := g.Reciprocal(g.Sqrt(g.AddScalar(ms, eps)))      // Rsqrt has no VJP (M4)
	normed := g.Mul(xf, inv)                               // fp32
	if inDT != dtypes.Float32 {
		normed = g.ConvertDType(normed, inDT)
	}
	return g.Mul(normed, g.Reshape(scale, 1, 1, -1))
}
