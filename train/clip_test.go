package train

import (
	"math"
	"testing"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
)

func TestClipByGlobalNorm(t *testing.T) {
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	// grad = [3,4] (norm 5); clip 1 -> scaled to norm 1 -> [0.6,0.8].
	exec := g.MustNewExec(be.Compute(), func(in *g.Node) *g.Node {
		return clipByGlobalNorm([]*g.Node{in}, 1.0)[0]
	})
	out := exec.MustCall1(tensors.FromFlatDataAndDimensions([]float32{3, 4}, 2))
	got := tensors.MustCopyFlatData[float32](out)
	if math.Abs(float64(got[0]-0.6)) > 1e-4 || math.Abs(float64(got[1]-0.8)) > 1e-4 {
		t.Errorf("clipped = %v want ~[0.6 0.8]", got)
	}
	// under the clip: unchanged
	exec2 := g.MustNewExec(be.Compute(), func(in *g.Node) *g.Node {
		return clipByGlobalNorm([]*g.Node{in}, 100.0)[0]
	})
	out2 := tensors.MustCopyFlatData[float32](exec2.MustCall1(tensors.FromFlatDataAndDimensions([]float32{3, 4}, 2)))
	if math.Abs(float64(out2[0]-3)) > 1e-4 {
		t.Errorf("under-clip changed: %v", out2)
	}
}
