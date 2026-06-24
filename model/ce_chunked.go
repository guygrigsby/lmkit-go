package model

import (
	"github.com/gomlx/compute/dtypes"
	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
)

// TiedLogitsCrossEntropyChunked computes the mean next-token cross-entropy directly from
// the hidden states h [B,T,H] and the tied embedding table [V,H], WITHOUT materializing the
// full [B,T,V] logits or their split-K backward. It streams the vocabulary in chunks of
// chunkSize (unrolled at graph-build time, a Go for-loop, so the result is differentiable),
// carrying an online softmax (running max m, denominator l) plus the accumulated
// correct-class logit. Each chunk's projection contracts only K=chunkSize, so XLA never
// builds the [vocab_tile,H,B*T] split-K partial-sum buffer the single full V=vocab
// contraction produces (~1.5 GB at lm-100m), which otherwise pins the micro-batch at B=2.
//
// Numerically equal to SparseCategoricalCrossEntropyLogits over the full logits (online
// softmax is exact; the correct-class logit is picked by a per-chunk equality mask). The
// softmax math runs in fp32, matching the non-chunked path. labels is [B,T,1] int (the
// next-token ids), as for SparseCategoricalCrossEntropyLogits.
func TiedLogitsCrossEntropyChunked(h, table, labels *g.Node, chunkSize int) *g.Node {
	gr := h.Graph()
	d := h.Shape().Dimensions // [B,T,H]
	b, t, hidden := d[0], d[1], d[2]
	vocab := table.Shape().Dimensions[0]
	bt := b * t

	h2d := g.Reshape(h, bt, hidden)                              // [BT,H]
	labels2d := g.ConvertDType(g.Reshape(labels, bt, 1), dtypes.Int32) // [BT,1]
	tableT := g.Transpose(table, 0, 1)                          // [H,V]

	const negInf = float32(-1e30) // finite, so exp() underflows to 0 without NaN
	m := g.Const(gr, tensors.FromFlatDataAndDimensions(fill(bt, negInf), bt, 1))          // running max
	l := g.Const(gr, tensors.FromFlatDataAndDimensions(make([]float32, bt), bt, 1))       // running sum exp
	correct := g.Const(gr, tensors.FromFlatDataAndDimensions(make([]float32, bt), bt, 1)) // correct-class logit

	for c0 := 0; c0 < vocab; c0 += chunkSize {
		c1 := c0 + chunkSize
		if c1 > vocab {
			c1 = vocab
		}
		cw := c1 - c0
		tcT := g.SliceAxis(tableT, 1, g.AxisRange(c0, c1))            // [H,cw]
		logitsC := g.ConvertDType(g.MatMul(h2d, tcT), dtypes.Float32) // [BT,cw] fp32

		// Online-softmax update over this vocab chunk.
		mNew := g.Max(m, g.ReduceAndKeep(logitsC, g.ReduceMax, -1)) // [BT,1]
		corr := g.Exp(g.Sub(m, mNew))                              // rescale old state to new max
		l = g.Add(g.Mul(l, corr), g.ReduceAndKeep(g.Exp(g.Sub(logitsC, mNew)), g.ReduceSum, -1))
		m = mNew

		// Accumulate the correct-class logit: the mask is 1 only where the label id equals a
		// vocab index in this chunk, so it is all-zero in chunks that don't hold the label.
		idxData := make([]int32, cw)
		for j := range idxData {
			idxData[j] = int32(c0 + j)
		}
		idxC := g.Const(gr, tensors.FromFlatDataAndDimensions(idxData, 1, cw))  // [1,cw]
		mask := g.ConvertDType(g.Equal(labels2d, idxC), dtypes.Float32)         // [BT,cw]
		correct = g.Add(correct, g.ReduceAndKeep(g.Mul(logitsC, mask), g.ReduceSum, -1))
	}

	lse := g.Add(m, g.Log(l))         // [BT,1] = log sum_v exp(logit_v)
	lossPerTok := g.Sub(lse, correct) // [BT,1] = -log softmax at the correct class
	return g.ReduceAllMean(lossPerTok)
}
