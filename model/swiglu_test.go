package model_test

import (
	"testing"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/model"
	"github.com/guygrigsby/lmkit-go/model/internal/paritytest"
)

func TestSwiGLUParity(t *testing.T) {
	f := paritytest.Load(t, "testdata/swiglu.json")
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	exec := g.MustNewExec(be.Compute(), func(x, wg, wu, wd *g.Node) *g.Node {
		return model.SwiGLU(x, wg, wu, wd)
	})
	out := exec.MustExec1(
		f.Inputs["x"].ToTensor(),
		f.Weights["Wg"].ToTensor(), f.Weights["Wu"].ToTensor(), f.Weights["Wd"].ToTensor())
	paritytest.AssertClose(t, tensors.MustCopyFlatData[float32](out), f.Expected, 1e-5)
}
