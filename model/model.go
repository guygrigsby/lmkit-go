package model

import (
	"fmt"

	g "github.com/gomlx/gomlx/core/graph"
)

// Forward runs the full Llama decoder: token embedding -> NLayers pre-norm decoder
// layers -> final RMSNorm -> tied logits. tokenIDs is [B,T] int32; positions is
// [0..T-1]. Returns logits [B,T,VocabSize]. Pure over the weight Nodes in w; no
// variables or RNG.
//
// Panics if len(w.Layers) != cfg.NLayers: a weight set whose depth disagrees with the
// config is corrupt (e.g. a partial checkpoint load), and silently running a
// shallower/deeper model would produce wrong results with no error.
func Forward(cfg Config, w Weights, tokenIDs *g.Node, positions []int) *g.Node {
	if len(w.Layers) != cfg.NLayers {
		panic(fmt.Sprintf("model.Forward: weights have %d layers, config NLayers=%d", len(w.Layers), cfg.NLayers))
	}
	h := EmbedLookup(w.Embed, tokenIDs) // [B,T,H]
	for i := range w.Layers {
		h = DecoderLayer(cfg, h, w.Layers[i], positions)
	}
	h = RMSNorm(h, w.FinalNorm, float32(cfg.RMSEps))
	return TiedLogits(h, w.Embed) // [B,T,V]
}
