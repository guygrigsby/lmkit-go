package train

import g "github.com/gomlx/gomlx/core/graph"

// clipByGlobalNorm scales grads so their global L2 norm <= clip (no-op if already
// under). Mirrors torch.nn.utils.clip_grad_norm_. GoMLX has no built-in for this.
func clipByGlobalNorm(grads []*g.Node, clip float64) []*g.Node {
	clipped, _ := clipByGlobalNormWithNorm(grads, clip)
	return clipped
}

// clipByGlobalNormWithNorm is like clipByGlobalNorm but also returns the pre-clip
// global L2 norm node (for surfacing as a metric).
func clipByGlobalNormWithNorm(grads []*g.Node, clip float64) (clipped []*g.Node, normNode *g.Node) {
	if len(grads) == 0 {
		return grads, nil
	}
	gr := grads[0].Graph()
	total := g.ReduceAllSum(g.Mul(grads[0], grads[0]))
	for _, gd := range grads[1:] {
		total = g.Add(total, g.ReduceAllSum(g.Mul(gd, gd)))
	}
	norm := g.Sqrt(total)
	// scale = min(1, clip/(norm+eps)) — clamp to 1 so we never amplify. clip const
	// matches the grad/norm dtype (fp32 today; defensive if grads ever arrive non-fp32).
	clipNode := g.ConvertDType(g.Const(gr, float32(clip)), norm.DType())
	scaleNode := g.MinScalar(g.Div(clipNode, g.AddScalar(norm, 1e-6)), 1.0)
	out := make([]*g.Node, len(grads))
	for i, gd := range grads {
		out[i] = g.Mul(gd, scaleNode)
	}
	return out, norm
}
