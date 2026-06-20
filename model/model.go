package model

import g "github.com/gomlx/gomlx/core/graph"

// Forward runs the full Llama decoder: token embedding -> NLayers pre-norm decoder
// layers -> final RMSNorm -> tied logits. tokenIDs is [B,T] int32; positions is
// [0..T-1]. Returns logits [B,T,VocabSize]. Pure over the weight Nodes in w; no
// variables or RNG.
func Forward(cfg Config, w Weights, tokenIDs *g.Node, positions []int) *g.Node {
	h := EmbedLookup(w.Embed, tokenIDs) // [B,T,H]
	for i := range w.Layers {
		h = DecoderLayer(cfg, h, w.Layers[i], positions)
	}
	h = RMSNorm(h, w.FinalNorm, float32(cfg.RMSEps))
	return TiedLogits(h, w.Embed) // [B,T,V]
}
