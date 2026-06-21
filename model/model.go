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

// ForwardCheckpointed is Forward with per-layer gradient checkpointing
// (rematerialization): each DecoderLayer's forward activations are recomputed in
// the backward pass instead of being held, so only one layer's [B,T,*] attention
// activations are live at a time. Trades one extra forward per layer for a large
// drop in training peak memory. Numerically identical to Forward; only
// memory/compute differ. Inference (no gradient) is unaffected: the recompute copy
// is dead-code-eliminated.
//
// Mechanism: g.InternalFusedOpCaller(fwd, recompute) returns fwd's value for the
// forward but routes the backward VJP through the separately-built recompute copy
// (gomlx rev_autodiff vjpAlternateOutputs), so fwd's internals are not needed in
// the backward. The recompute copy reads its inputs through g.OptimizationBarrier
// so XLA cannot common-subexpression-eliminate it back into the forward copy (which
// would defeat the remat); with the copies kept distinct, XLA frees the forward
// layer's activations after its output and recomputes them only in the backward.
func ForwardCheckpointed(cfg Config, w Weights, tokenIDs *g.Node, positions []int) *g.Node {
	if len(w.Layers) != cfg.NLayers {
		panic(fmt.Sprintf("model.ForwardCheckpointed: weights have %d layers, config NLayers=%d", len(w.Layers), cfg.NLayers))
	}
	h := EmbedLookup(w.Embed, tokenIDs)
	for i := range w.Layers {
		lw := w.Layers[i]
		hin := h
		// Barrier the FORWARD copy's inputs (not the recompute copy's): in
		// InternalFusedOpCaller the fused/forward side is never differentiated (its
		// VJP is redirected to the decomposed/recompute side), so the barrier needs
		// no gradient, which the upstream OptimizationBarrier op does not define. It
		// still keeps the two copies distinct so XLA won't CSE them, and the recompute
		// reads raw inputs so gradients flow straight back.
		fwdFn := func() *g.Node {
			b := g.OptimizationBarriers(hin, lw.AttnNorm, lw.Wq, lw.Wk, lw.Wv, lw.Wo, lw.FFNNorm, lw.Wgate, lw.Wup, lw.Wdown)
			blw := LayerWeights{
				AttnNorm: b[1], Wq: b[2], Wk: b[3], Wv: b[4], Wo: b[5],
				FFNNorm: b[6], Wgate: b[7], Wup: b[8], Wdown: b[9],
			}
			return DecoderLayer(cfg, b[0], blw, positions)
		}
		recomputeFn := func() *g.Node { return DecoderLayer(cfg, hin, lw, positions) }
		h = g.InternalFusedOpCaller(fwdFn, recomputeFn)
	}
	h = RMSNorm(h, w.FinalNorm, float32(cfg.RMSEps))
	return TiedLogits(h, w.Embed)
}
