package train

import g "github.com/gomlx/gomlx/core/graph"

// clipByGlobalNorm scales grads so their global L2 norm <= clip (no-op if already
// under). Mirrors torch.nn.utils.clip_grad_norm_. GoMLX has no built-in for this.
func clipByGlobalNorm(grads []*g.Node, clip float64) []*g.Node {
	if len(grads) == 0 {
		return grads
	}
	gr := grads[0].Graph()
	total := g.ReduceAllSum(g.Mul(grads[0], grads[0]))
	for _, gd := range grads[1:] {
		total = g.Add(total, g.ReduceAllSum(g.Mul(gd, gd)))
	}
	norm := g.Sqrt(total)
	// scale = min(1, clip/(norm+eps)) — clamp to 1 so we never amplify.
	scaleNode := g.MinScalar(g.Div(g.Const(gr, float32(clip)), g.AddScalar(norm, 1e-6)), 1.0)
	out := make([]*g.Node, len(grads))
	for i, gd := range grads {
		out[i] = g.Mul(gd, scaleNode)
	}
	return out
}
