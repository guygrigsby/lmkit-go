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
