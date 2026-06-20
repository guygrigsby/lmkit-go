package model_test

import (
	"testing"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/model"
	"github.com/guygrigsby/lmkit-go/model/internal/paritytest"
)

func TestRMSNormParity(t *testing.T) {
	f := paritytest.Load(t, "testdata/rmsnorm.json")
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	eps := float32(f.Config["rms_eps"])
	exec := g.MustNewExec(be.Compute(), func(x, scale *g.Node) *g.Node {
		return model.RMSNorm(x, scale, eps)
	})
	out := exec.MustCall1(f.Inputs["x"].ToTensor(), f.Weights["scale"].ToTensor())
	paritytest.AssertClose(t, tensors.MustCopyFlatData[float32](out), f.Expected, 1e-5)
}
