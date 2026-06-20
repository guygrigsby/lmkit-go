package model_test

import (
	"testing"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/model"
	"github.com/guygrigsby/lmkit-go/model/internal/paritytest"
)

func TestDecoderLayerParity(t *testing.T) {
	f := paritytest.Load(t, "testdata/decoder.json")
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	cfg := model.Config{
		Hidden: int(f.Config["hidden"]), NHeads: int(f.Config["n_heads"]),
		NKVHeads: int(f.Config["n_kv_heads"]), HeadDim: int(f.Config["head_dim"]),
		FFNHidden: int(f.Config["ffn_hidden"]), RopeBase: f.Config["rope_base"],
		RMSEps: f.Config["rms_eps"],
	}
	seqLen := int(f.Config["seq_len"])
	positions := make([]int, seqLen)
	for i := range positions {
		positions[i] = i
	}
	exec := g.MustNewExec(be.Compute(), func(gr *g.Graph) *g.Node {
		lw := model.LayerWeights{
			AttnNorm: g.Const(gr, f.Weights["attn_norm"].ToTensor()),
			Wq:       g.Const(gr, f.Weights["Wq"].ToTensor()),
			Wk:       g.Const(gr, f.Weights["Wk"].ToTensor()),
			Wv:       g.Const(gr, f.Weights["Wv"].ToTensor()),
			Wo:       g.Const(gr, f.Weights["Wo"].ToTensor()),
			FFNNorm:  g.Const(gr, f.Weights["ffn_norm"].ToTensor()),
			Wgate:    g.Const(gr, f.Weights["Wgate"].ToTensor()),
			Wup:      g.Const(gr, f.Weights["Wup"].ToTensor()),
			Wdown:    g.Const(gr, f.Weights["Wdown"].ToTensor()),
		}
		h := g.Const(gr, f.Inputs["h"].ToTensor())
		return model.DecoderLayer(cfg, h, lw, positions)
	})
	out := exec.MustCall1()
	// 2e-4: a full decoder layer chains ~10 matmuls + softmax + residuals in fp32;
	// the absolute parity diff varies with weight scale across seeds (a correct impl
	// spanned 5.7e-6..6.1e-5 here). 2e-4 exceeds that spread so any seed passes iff
	// the wiring is correct, rather than depending on a lucky seed.
	paritytest.AssertClose(t, tensors.MustCopyFlatData[float32](out), f.Expected, 2e-4)
}
