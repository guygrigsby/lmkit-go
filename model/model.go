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
	return TiedLogits(ForwardHidden(cfg, w, tokenIDs, positions), w.Embed) // [B,T,V]
}

// ForwardHidden runs the decoder up to (and including) the final RMSNorm, returning the
// hidden states h [B,T,H] BEFORE the tied output projection. Training builds the loss
// straight from h and the embedding table (chunked cross-entropy) so the full [B,T,V]
// logits and their split-K backward never materialize; inference uses Forward, which
// applies TiedLogits to this.
func ForwardHidden(cfg Config, w Weights, tokenIDs *g.Node, positions []int) *g.Node {
	if len(w.Layers) != cfg.NLayers {
		panic(fmt.Sprintf("model.ForwardHidden: weights have %d layers, config NLayers=%d", len(w.Layers), cfg.NLayers))
	}
	h := EmbedLookup(w.Embed, tokenIDs) // [B,T,H]
	for i := range w.Layers {
		h = DecoderLayer(cfg, h, w.Layers[i], positions)
	}
	return RMSNorm(h, w.FinalNorm, float32(cfg.RMSEps))
}

// ForwardCheckpointed is Forward with per-layer gradient checkpointing
// (rematerialization): each DecoderLayer's forward activations are recomputed in
// the backward pass instead of being held, so only one layer's [B,T,*] attention
// activations are live at a time. Trades one extra forward per layer for a large
// drop in training peak memory. Numerically identical to Forward; only
// memory/compute differ.
//
// Uses gomlx's Node.Checkpoint() (gomlx #425): marking each layer's input as a
// checkpoint makes autodiff rematerialize that layer in the backward pass, with
// the optimization + scheduling barriers inserted for it. StopCheckpoint caps the
// rematerialization before the final norm and logits.
func ForwardCheckpointed(cfg Config, w Weights, tokenIDs *g.Node, positions []int) *g.Node {
	return TiedLogits(ForwardHiddenCheckpointed(cfg, w, tokenIDs, positions), w.Embed)
}

// ForwardHiddenCheckpointed is ForwardHidden with per-layer gradient checkpointing.
// See ForwardHidden and ForwardCheckpointed.
func ForwardHiddenCheckpointed(cfg Config, w Weights, tokenIDs *g.Node, positions []int) *g.Node {
	if len(w.Layers) != cfg.NLayers {
		panic(fmt.Sprintf("model.ForwardHiddenCheckpointed: weights have %d layers, config NLayers=%d", len(w.Layers), cfg.NLayers))
	}
	h := EmbedLookup(w.Embed, tokenIDs)
	for i := range w.Layers {
		h = h.Checkpoint() // recompute this layer in the backward instead of storing it
		h = DecoderLayer(cfg, h, w.Layers[i], positions)
	}
	h = h.StopCheckpoint()
	return RMSNorm(h, w.FinalNorm, float32(cfg.RMSEps))
}
