package model

import g "github.com/gomlx/gomlx/core/graph"

// DecoderLayer is one pre-norm Llama decoder layer with residual connections:
//
//	h = h + Attention(RMSNorm(h, attnNorm))
//	h = h + SwiGLU(RMSNorm(h, ffnNorm))
//
// h is [B,T,Hidden]; positions is [0..T-1].
func DecoderLayer(cfg Config, h *g.Node, lw LayerWeights, positions []int) *g.Node {
	eps := float32(cfg.RMSEps)
	attn := Attention(cfg, RMSNorm(h, lw.AttnNorm, eps), lw.Wq, lw.Wk, lw.Wv, lw.Wo, positions)
	h = g.Add(h, attn)
	ffn := SwiGLU(RMSNorm(h, lw.FFNNorm, eps), lw.Wgate, lw.Wup, lw.Wdown)
	return g.Add(h, ffn)
}
