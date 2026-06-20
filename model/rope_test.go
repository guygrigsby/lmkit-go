package model_test

import (
	"testing"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/model"
	"github.com/guygrigsby/lmkit-go/model/internal/paritytest"
)

func TestRoPEParity(t *testing.T) {
	f := paritytest.Load(t, "testdata/rope.json")
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	headDim := int(f.Config["head_dim"])
	base := f.Config["rope_base"]
	seqLen := int(f.Config["seq_len"])
	positions := make([]int, seqLen)
	for i := range positions {
		positions[i] = i
	}
	exec := g.MustNewExec(be.Compute(), func(x *g.Node) *g.Node {
		return model.RoPE(x, positions, base, headDim)
	})
	out := exec.MustCall1(f.Inputs["x"].ToTensor())
	paritytest.AssertClose(t, tensors.MustCopyFlatData[float32](out), f.Expected, 1e-5)
}
