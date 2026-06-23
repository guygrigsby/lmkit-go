package model

import (
	"math"

	"github.com/gomlx/compute/dtypes"
	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers/attention"
)

// UseFlashAttention selects the cuDNN flash-attention path for bf16 (CUDA) compute. When false,
// the decomposed path is used for every dtype. Flash and decomposed agree to bf16 tolerance; the
// toggle exists for A/B measurement and as an escape hatch.
var UseFlashAttention = true

// Attention is a Llama GQA attention block. x is [B,T,Hidden]; weights project to
// nHeads/nKVHeads of HeadDim. RoPE is applied to Q and K; KV heads are repeated to
// nHeads; scores are scaled by 1/sqrt(HeadDim) with a causal mask; softmax over
// keys; then value aggregation, head merge, and the output projection.
func Attention(cfg Config, x, wQ, wK, wV, wO *g.Node, positions []int) *g.Node {
	d := x.Shape().Dimensions // [B,T,Hidden]
	b, tt := d[0], d[1]
	hd, nH, nKV := cfg.HeadDim, cfg.NHeads, cfg.NKVHeads

	q := g.Reshape(g.MatMul(x, wQ), b, tt, nH, hd)  // [B,T,nH,hd]
	k := g.Reshape(g.MatMul(x, wK), b, tt, nKV, hd) // [B,T,nKV,hd]
	v := g.Reshape(g.MatMul(x, wV), b, tt, nKV, hd)

	q = RoPE(q, positions, cfg.RopeBase, hd)
	k = RoPE(k, positions, cfg.RopeBase, hd)

	// Repeat KV heads to nH (group size rep) by broadcasting then reshaping.
	// This matches torch.repeat_interleave(rep, dim=2): inserting the rep axis
	// directly after the head axis interleaves correctly.
	rep := nH / nKV
	k = repeatKV(k, b, tt, nKV, rep, hd)
	v = repeatKV(v, b, tt, nKV, rep, hd)

	scale := 1.0 / math.Sqrt(float64(hd))

	var out *g.Node // [B,T,nH,hd]
	if UseFlashAttention && q.DType() == dtypes.BFloat16 {
		// bf16 (CUDA) compute path: cuDNN flash attention, scores never materialized.
		// Off-GPU it transparently falls back to a decomposed attention. q,k,v are
		// [B,T,nH,hd] (BSHD); kv already repeated to nH above.
		out = attention.FlashAttention(q, k, v, scale)
	} else {
		// Decomposed path (PyTorch-parity tested in fp32): scaled q·kᵀ, causal mask, fp32
		// softmax, value aggregation. scores[B,nH,T,S] = q·kᵀ / sqrt(hd), q,k as [B,T,nH,hd].
		scores := g.Einsum("btnh,bsnh->bnts", q, k)
		scores = g.MulScalar(scores, float32(scale))
		// causalMask is a host fp32 constant; cast to scores' dtype so the bf16 compute path
		// does not mix dtypes in the Add. No-op when scores is fp32.
		scores = g.Add(scores, g.ConvertDType(causalMask(x.Graph(), tt), scores.DType())) // [1,1,T,T] broadcasts
		// softmax in fp32 for stability/fidelity; downcast back to scores' dtype.
		sdt := scores.DType()
		sf := scores
		if sdt != dtypes.Float32 {
			sf = g.ConvertDType(scores, dtypes.Float32)
		}
		probs := g.Softmax(sf, -1) // softmax over S (key) axis
		if sdt != dtypes.Float32 {
			probs = g.ConvertDType(probs, sdt)
		}
		out = g.Einsum("bnts,bsnh->btnh", probs, v)
	}
	merged := g.Reshape(out, b, tt, nH*hd) // [B,T,nH*hd]
	return g.MatMul(merged, wO)            // [B,T,Hidden]
}

// repeatKV expands [B,T,nKV,hd] to [B,T,nKV*rep,hd] by repeating each KV head rep
// times (matching torch.repeat_interleave on the head axis).
// Pattern: reshape to [B,T,nKV,1,hd], broadcast to [B,T,nKV,rep,hd], reshape to
// [B,T,nKV*rep,hd]. The rep axis is inserted right after the head axis so the
// broadcast interleaves: head0 rep times, then head1 rep times, etc.
func repeatKV(x *g.Node, b, tt, nKV, rep, hd int) *g.Node {
	x = g.Reshape(x, b, tt, nKV, 1, hd)
	x = g.BroadcastToDims(x, b, tt, nKV, rep, hd)
	return g.Reshape(x, b, tt, nKV*rep, hd)
}

// causalMask returns a [1,1,T,T] additive mask: 0 on/below the diagonal, -inf above.
// Adding this to scores before softmax zeros out future positions.
func causalMask(gr *g.Graph, t int) *g.Node {
	m := make([]float32, t*t)
	neg := float32(math.Inf(-1))
	for i := 0; i < t; i++ {
		for j := 0; j < t; j++ {
			if j > i {
				m[i*t+j] = neg
			}
		}
	}
	return g.Const(gr, tensors.FromFlatDataAndDimensions(m, 1, 1, t, t))
}
