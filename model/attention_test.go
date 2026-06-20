package model_test

import (
	"testing"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/model"
	"github.com/guygrigsby/lmkit-go/model/internal/paritytest"
)

// TestAttentionParity checks GQA attention against a PyTorch golden.
// Tolerance is 5e-5 (not 1e-5) because chained matmul+softmax+matmul accumulates
// more fp32 rounding error than single-op blocks.
func TestAttentionParity(t *testing.T) {
	f := paritytest.Load(t, "testdata/attention.json")
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	cfg := model.Config{
		Hidden:   int(f.Config["hidden"]),
		NHeads:   int(f.Config["n_heads"]),
		NKVHeads: int(f.Config["n_kv_heads"]),
		HeadDim:  int(f.Config["head_dim"]),
		RopeBase: f.Config["rope_base"],
	}
	seqLen := int(f.Config["seq_len"])
	positions := make([]int, seqLen)
	for i := range positions {
		positions[i] = i
	}
	exec := g.MustNewExec(be.Compute(), func(x, wq, wk, wv, wo *g.Node) *g.Node {
		return model.Attention(cfg, x, wq, wk, wv, wo, positions)
	})
	out := exec.MustCall1(
		f.Inputs["x"].ToTensor(),
		f.Weights["Wq"].ToTensor(), f.Weights["Wk"].ToTensor(),
		f.Weights["Wv"].ToTensor(), f.Weights["Wo"].ToTensor())
	paritytest.AssertClose(t, tensors.MustCopyFlatData[float32](out), f.Expected, 5e-5)
}
