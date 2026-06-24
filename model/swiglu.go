package model

import g "github.com/gomlx/gomlx/core/graph"

// SwiGLU is the Llama gated FFN: (SiLU(x·wGate) * (x·wUp)) · wDown.
// x is [B,T,H]; wGate,wUp are [H,F]; wDown is [F,H]. SiLU(z) = z * sigmoid(z).
// Tokens are flattened to 2-D ([B*T,H]) for the matmuls: a 3-D-lhs MatMul lowers its
// BACKWARD as a batched DotGeneral whose gradient is ~50x slower than the equivalent
// 2-D gemm, which dominated the training step.
func SwiGLU(x, wGate, wUp, wDown *g.Node) *g.Node {
	d := x.Shape().Dimensions // [B,T,H]
	b, t, h := d[0], d[1], d[2]
	x2 := g.Reshape(x, b*t, h)               // [B*T,H]
	gate := g.MatMul(x2, wGate)              // [B*T,F]
	up := g.MatMul(x2, wUp)                  // [B*T,F]
	silu := g.Mul(gate, g.Sigmoid(gate))     // SiLU
	out := g.MatMul(g.Mul(silu, up), wDown)  // [B*T,H]
	return g.Reshape(out, b, t, h)
}
