package model

import g "github.com/gomlx/gomlx/core/graph"

// RMSNorm normalizes x [B,T,H] over its last axis and scales by scale [H].
// y = x * rsqrt(mean(x^2, lastAxis) + eps) * scale. The mean/rsqrt run in the
// graph's dtype; callers wanting fp32-internal accumulation pass fp32 x.
// Uses Reciprocal(Sqrt(...)) instead of Rsqrt: Rsqrt has no gradient defined
// in GoMLX autodiff, so it cannot be used in a training graph.
func RMSNorm(x, scale *g.Node, eps float32) *g.Node {
	ms := g.ReduceAndKeep(g.Mul(x, x), g.ReduceMean, -1) // [B,T,1]
	inv := g.Reciprocal(g.Sqrt(g.AddScalar(ms, eps)))    // [B,T,1]
	normed := g.Mul(x, inv)                              // [B,T,H]
	s := g.Reshape(scale, 1, 1, -1)                      // [1,1,H]
	return g.Mul(normed, s)
}
