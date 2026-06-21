package model

import (
	"math"

	"github.com/gomlx/compute/dtypes"
	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
)

// AttentionChunked is Attention computed with online (streaming) softmax over the
// key axis, with the chunks unrolled at graph-build time (a Go for-loop, not a
// runtime While, so the result is differentiable). It is numerically equal to
// Attention; the point is forward memory: it never materializes the full
// [B,nH,T,T] score table, only one [B,nH,T,chunk] block at a time, carrying a
// running max / denominator / output accumulator across chunks.
//
// chunkSize is the key-block size and need not divide T (the last block is
// shorter). The online-softmax math runs in fp32 for fidelity, matching
// Attention's fp32 softmax; the output is cast back to the input dtype.
func AttentionChunked(cfg Config, x, wQ, wK, wV, wO *g.Node, positions []int, chunkSize int) *g.Node {
	d := x.Shape().Dimensions // [B,T,Hidden]
	b, tt := d[0], d[1]
	hd, nH, nKV := cfg.HeadDim, cfg.NHeads, cfg.NKVHeads

	q := g.Reshape(g.MatMul(x, wQ), b, tt, nH, hd)
	k := g.Reshape(g.MatMul(x, wK), b, tt, nKV, hd)
	v := g.Reshape(g.MatMul(x, wV), b, tt, nKV, hd)
	q = RoPE(q, positions, cfg.RopeBase, hd)
	k = RoPE(k, positions, cfg.RopeBase, hd)
	rep := nH / nKV
	k = repeatKV(k, b, tt, nKV, rep, hd) // [B,T,nH,hd]
	v = repeatKV(v, b, tt, nKV, rep, hd)

	gr := x.Graph()
	scale := float32(1.0 / math.Sqrt(float64(hd)))
	const negInf = float32(-1e30) // finite so exp() underflows to 0 without NaN

	// Running state, fp32: m = max [B,nH,T,1], l = denom [B,nH,T,1], acc = out [B,nH,T,hd].
	m := g.Const(gr, tensors.FromFlatDataAndDimensions(fill(b*nH*tt, negInf), b, nH, tt, 1))
	l := g.Const(gr, tensors.FromFlatDataAndDimensions(make([]float32, b*nH*tt), b, nH, tt, 1))
	acc := g.Const(gr, tensors.FromFlatDataAndDimensions(make([]float32, b*nH*tt*hd), b, nH, tt, hd))

	for c0 := 0; c0 < tt; c0 += chunkSize {
		c1 := c0 + chunkSize
		if c1 > tt {
			c1 = tt
		}
		cw := c1 - c0
		kc := g.SliceAxis(k, 1, g.AxisRange(c0, c1)) // [B,cw,nH,hd]
		vc := g.SliceAxis(v, 1, g.AxisRange(c0, c1))

		// scores_c [B,nH,T,cw], upcast to fp32, scaled and causally masked.
		sc := g.ConvertDType(g.Einsum("btnh,bcnh->bntc", q, kc), dtypes.Float32)
		sc = g.MulScalar(sc, scale)
		sc = g.Add(sc, chunkCausalMask(gr, tt, c0, cw, negInf))

		// Online softmax update.
		mNew := g.Max(m, g.ReduceAndKeep(sc, g.ReduceMax, -1)) // [B,nH,T,1]
		corr := g.Exp(g.Sub(m, mNew))                          // rescale old state to new max
		p := g.Exp(g.Sub(sc, mNew))                            // [B,nH,T,cw]
		l = g.Add(g.Mul(l, corr), g.ReduceAndKeep(p, g.ReduceSum, -1))
		contrib := g.Einsum("bntc,bcnh->bnth", p, g.ConvertDType(vc, dtypes.Float32))
		acc = g.Add(g.Mul(acc, corr), contrib)
		m = mNew
	}

	out := g.Mul(acc, g.Reciprocal(l)) // [B,nH,T,hd], l>0 (every query attends itself)
	out = g.Transpose(out, 1, 2)       // [B,T,nH,hd]
	out = g.ConvertDType(out, q.DType())
	merged := g.Reshape(out, b, tt, nH*hd)
	return g.MatMul(merged, wO)
}

// fill returns an n-length slice set to val.
func fill(n int, val float32) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = val
	}
	return s
}

// chunkCausalMask returns a [1,1,T,cw] additive mask for the key chunk starting at
// c0: 0 where query t may attend key c0+j (c0+j <= t), negInf otherwise.
func chunkCausalMask(gr *g.Graph, tt, c0, cw int, negInf float32) *g.Node {
	mask := make([]float32, tt*cw)
	for t := 0; t < tt; t++ {
		for j := 0; j < cw; j++ {
			if c0+j > t {
				mask[t*cw+j] = negInf
			}
		}
	}
	return g.Const(gr, tensors.FromFlatDataAndDimensions(mask, 1, 1, tt, cw))
}
