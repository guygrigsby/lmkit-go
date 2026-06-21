package model

import (
	"math"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
)

// RoPE applies rotary position embedding (rotate-half convention) to x, shaped
// [B,T,nHeads,headDim], over its last axis. positions has length T. cos/sin are
// computed on the host (matching the HF Llama formula) and injected as constants
// shaped [1,T,1,headDim] so they broadcast across batch and heads.
func RoPE(x *g.Node, positions []int, base float64, headDim int) *g.Node {
	half := headDim / 2
	t := len(positions)
	cos := make([]float32, t*headDim)
	sin := make([]float32, t*headDim)
	for ti, pos := range positions {
		for i := 0; i < half; i++ {
			invFreq := math.Pow(base, -float64(2*i)/float64(headDim))
			ang := float64(pos) * invFreq
			c, s := float32(math.Cos(ang)), float32(math.Sin(ang))
			// emb = cat([freqs, freqs]) -> the value at i and i+half is the same angle.
			cos[ti*headDim+i], cos[ti*headDim+i+half] = c, c
			sin[ti*headDim+i], sin[ti*headDim+i+half] = s, s
		}
	}
	gr := x.Graph()
	// cos/sin are built fp32 on the host; cast to x's dtype so the bf16 compute
	// path (CUDA) does not mix dtypes in the Mul below. No-op when x is fp32.
	cosN := g.ConvertDType(g.Const(gr, tensors.FromFlatDataAndDimensions(cos, 1, t, 1, headDim)), x.DType())
	sinN := g.ConvertDType(g.Const(gr, tensors.FromFlatDataAndDimensions(sin, 1, t, 1, headDim)), x.DType())
	// rotate_half(x) = cat([-x2, x1]) over last axis.
	x1 := g.Slice(x, g.AxisRange().Spacer(), g.AxisRangeFromStart(half))
	x2 := g.Slice(x, g.AxisRange().Spacer(), g.AxisRangeToEnd(half))
	rot := g.Concatenate([]*g.Node{g.Neg(x2), x1}, -1)
	return g.Add(g.Mul(x, cosN), g.Mul(rot, sinN))
}
