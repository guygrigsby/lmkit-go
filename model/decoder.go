package model

import g "github.com/gomlx/gomlx/core/graph"

// SkipAttn / SkipFFN drop a sublayer from DecoderLayer for profiling (isolating where the
// backward time goes in-context). Never set in real training.
var SkipAttn, SkipFFN bool

// DecoderLayer is one pre-norm Llama decoder layer with residual connections:
//
//	h = h + Attention(RMSNorm(h, attnNorm))
//	h = h + SwiGLU(RMSNorm(h, ffnNorm))
//
// h is [B,T,Hidden]; positions is [0..T-1].
func DecoderLayer(cfg Config, h *g.Node, lw LayerWeights, positions []int) *g.Node {
	eps := float32(cfg.RMSEps)
	if !SkipAttn {
		normed := RMSNorm(h, lw.AttnNorm, eps)
		var attn *g.Node
		if cfg.AttnChunk > 0 {
			attn = AttentionChunked(cfg, normed, lw.Wq, lw.Wk, lw.Wv, lw.Wo, positions, cfg.AttnChunk)
		} else {
			attn = Attention(cfg, normed, lw.Wq, lw.Wk, lw.Wv, lw.Wo, positions)
		}
		h = g.Add(h, attn)
	}
	if !SkipFFN {
		ffn := SwiGLU(RMSNorm(h, lw.FFNNorm, eps), lw.Wgate, lw.Wup, lw.Wdown)
		h = g.Add(h, ffn)
	}
	return h
}
