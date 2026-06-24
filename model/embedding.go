package model

import g "github.com/gomlx/gomlx/core/graph"

// EmbedLookup gathers rows of table [V,H] by tokenIDs [B,T] (int) into [B,T,H].
// Gather indexes leading axes of params by the last axis of indices, so the ids
// are reshaped to [B,T,1] (one indexed axis = axis 0 of the table).
func EmbedLookup(table, tokenIDs *g.Node) *g.Node {
	d := tokenIDs.Shape().Dimensions // [B,T]
	idx := g.Reshape(tokenIDs, append(append([]int{}, d...), 1)...) // [B,T,1]
	return g.Gather(table, idx)                                     // [B,T,H]
}

// TiedLogits projects hidden states h [B,T,H] onto the tied embedding table
// [V,H]: logits[...,v] = h · table[v]. Expressed as an explicit 2-D matmul over the
// flattened tokens ([B*T,H]·[H,V]) rather than a 3-D einsum so the backward
// (dH = dLogits·table, contracting V=vocab) lowers as a clean gemm. The 3-D einsum's
// backward presents as a generalized dot that XLA reduce-lowers into a huge
// [vocab_tile,H,B,T] partial-sum buffer (~1.5 GB at lm-100m), pinning the micro-batch.
func TiedLogits(h, table *g.Node) *g.Node {
	d := h.Shape().Dimensions // [B,T,H]
	b, t, hidden := d[0], d[1], d[2]
	vocab := table.Shape().Dimensions[0]
	h2d := g.Reshape(h, b*t, hidden)                    // [B*T,H]
	logits2d := g.MatMul(h2d, g.Transpose(table, 0, 1)) // [B*T,H]·[H,V] -> [B*T,V]
	return g.Reshape(logits2d, b, t, vocab)
}
