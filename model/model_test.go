package model_test

import (
	"fmt"
	"testing"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/model"
	"github.com/guygrigsby/lmkit-go/model/internal/paritytest"
)

func TestForwardParity(t *testing.T) {
	f := paritytest.Load(t, "testdata/model.json")
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	cfg := model.Config{
		Hidden: int(f.Config["hidden"]), NLayers: int(f.Config["n_layers"]),
		NHeads: int(f.Config["n_heads"]), NKVHeads: int(f.Config["n_kv_heads"]),
		HeadDim: int(f.Config["head_dim"]), FFNHidden: int(f.Config["ffn_hidden"]),
		VocabSize: int(f.Config["vocab"]), RopeBase: f.Config["rope_base"], RMSEps: f.Config["rms_eps"],
	}
	seqLen := int(f.Config["seq_len"])
	positions := make([]int, seqLen)
	for i := range positions {
		positions[i] = i
	}
	exec := g.MustNewExec(be.Compute(), func(gr *g.Graph) *g.Node {
		c := func(name string) *g.Node { return g.Const(gr, f.Weights[name].ToTensor()) }
		w := model.Weights{Embed: c("embed"), FinalNorm: c("final_norm")}
		for i := 0; i < cfg.NLayers; i++ {
			p := fmt.Sprintf("layer%d_", i)
			w.Layers = append(w.Layers, model.LayerWeights{
				AttnNorm: c(p + "attn_norm"), Wq: c(p + "Wq"), Wk: c(p + "Wk"), Wv: c(p + "Wv"),
				Wo: c(p + "Wo"), FFNNorm: c(p + "ffn_norm"),
				Wgate: c(p + "Wgate"), Wup: c(p + "Wup"), Wdown: c(p + "Wdown"),
			})
		}
		ids := g.Const(gr, f.Inputs["ids"].ToTensor())
		return model.Forward(cfg, w, ids, positions)
	})
	out := exec.MustCall1()
	// 5e-4: the full forward stacks N decoder layers (each ~10 matmuls + softmax) +
	// final norm + tied projection in fp32, so absolute accumulation is deeper than a
	// single layer. This bound exceeds the per-seed spread — a correct impl passes for
	// ANY seed. Do NOT seed-shop the generator to fit a tighter number; pick the
	// tolerance for the computation depth, not for a lucky weight set.
	paritytest.AssertClose(t, tensors.MustCopyFlatData[float32](out), f.Expected, 5e-4)
}
