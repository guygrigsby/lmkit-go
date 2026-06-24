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
// [V,H]: logits[...,v] = h · table[v]. Contract H: einsum bth,vh->btv.
func TiedLogits(h, table *g.Node) *g.Node {
	return g.Einsum("bth,vh->btv", h, table)
}
