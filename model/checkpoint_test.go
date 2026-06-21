package model_test

import (
	"math"
	"testing"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/model"
)

// ramp returns deterministic small values so the test is seed-free.
func ramp(n int) []float32 {
	d := make([]float32, n)
	for i := range d {
		d[i] = float32(i%9)*0.05 - 0.2
	}
	return d
}

// TestForwardCheckpointedGradientMatchesForward asserts that per-layer gradient
// checkpointing changes only memory/compute, not numerics: the gradient of the
// loss w.r.t. the (tied) embedding must match Forward to fp32 tolerance.
func TestForwardCheckpointedGradientMatchesForward(t *testing.T) {
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	cfg := model.Config{
		VocabSize: 16, Hidden: 8, NLayers: 3, NHeads: 2, NKVHeads: 1,
		HeadDim: 4, FFNHidden: 16, SeqLen: 4, RopeBase: 10000, RMSEps: 1e-5,
	}
	const B, T = 2, 4
	positions := []int{0, 1, 2, 3}
	tokIDs := []int32{1, 2, 3, 4, 5, 6, 7, 8}

	// gradFn builds the model with `fwd`, returns d(sum(logits))/d(Embed).
	gradFn := func(fwd func(model.Config, model.Weights, *g.Node, []int) *g.Node) []float32 {
		exec := g.MustNewExec(be.Compute(), func(gr *g.Graph) *g.Node {
			embed := g.Const(gr, tensors.FromFlatDataAndDimensions(ramp(cfg.VocabSize*cfg.Hidden), cfg.VocabSize, cfg.Hidden))
			w := model.Weights{Embed: embed, FinalNorm: g.Const(gr, tensors.FromFlatDataAndDimensions(ramp(cfg.Hidden), cfg.Hidden))}
			for l := 0; l < cfg.NLayers; l++ {
				_ = l
				w.Layers = append(w.Layers, model.LayerWeights{
					AttnNorm: g.Const(gr, tensors.FromFlatDataAndDimensions(ramp(cfg.Hidden), cfg.Hidden)),
					Wq:       g.Const(gr, tensors.FromFlatDataAndDimensions(ramp(cfg.Hidden*cfg.NHeads*cfg.HeadDim), cfg.Hidden, cfg.NHeads*cfg.HeadDim)),
					Wk:       g.Const(gr, tensors.FromFlatDataAndDimensions(ramp(cfg.Hidden*cfg.NKVHeads*cfg.HeadDim), cfg.Hidden, cfg.NKVHeads*cfg.HeadDim)),
					Wv:       g.Const(gr, tensors.FromFlatDataAndDimensions(ramp(cfg.Hidden*cfg.NKVHeads*cfg.HeadDim), cfg.Hidden, cfg.NKVHeads*cfg.HeadDim)),
					Wo:       g.Const(gr, tensors.FromFlatDataAndDimensions(ramp(cfg.NHeads*cfg.HeadDim*cfg.Hidden), cfg.NHeads*cfg.HeadDim, cfg.Hidden)),
					FFNNorm:  g.Const(gr, tensors.FromFlatDataAndDimensions(ramp(cfg.Hidden), cfg.Hidden)),
					Wgate:    g.Const(gr, tensors.FromFlatDataAndDimensions(ramp(cfg.Hidden*cfg.FFNHidden), cfg.Hidden, cfg.FFNHidden)),
					Wup:      g.Const(gr, tensors.FromFlatDataAndDimensions(ramp(cfg.Hidden*cfg.FFNHidden), cfg.Hidden, cfg.FFNHidden)),
					Wdown:    g.Const(gr, tensors.FromFlatDataAndDimensions(ramp(cfg.FFNHidden*cfg.Hidden), cfg.FFNHidden, cfg.Hidden)),
				})
			}
			ids := g.Const(gr, tensors.FromFlatDataAndDimensions(tokIDs, B, T))
			logits := fwd(cfg, w, ids, positions)
			loss := g.ReduceAllSum(logits)
			return g.Gradient(loss, embed)[0]
		})
		return tensors.MustCopyFlatData[float32](exec.MustCall1())
	}

	plain := gradFn(model.Forward)
	ckpt := gradFn(model.ForwardCheckpointed)
	if len(plain) != len(ckpt) {
		t.Fatalf("grad len: plain %d vs ckpt %d", len(plain), len(ckpt))
	}
	for i := range plain {
		if d := math.Abs(float64(plain[i] - ckpt[i])); d > 1e-4 {
			t.Errorf("grad[%d]: plain %v vs ckpt %v (diff %v > 1e-4)", i, plain[i], ckpt[i], d)
		}
	}
}
