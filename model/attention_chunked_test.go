package model_test

import (
	"math"
	"testing"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/model"
)

// TestAttentionChunkedMatchesNaive pins that the chunked online-softmax attention
// is numerically equal to the naive full-table attention, in both the forward
// output and the gradient, for chunk sizes that do and don't divide T.
func TestAttentionChunkedMatchesNaive(t *testing.T) {
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	cfg := model.Config{Hidden: 8, NHeads: 2, NKVHeads: 1, HeadDim: 4, RopeBase: 10000}
	const B, T = 2, 6
	positions := []int{0, 1, 2, 3, 4, 5}

	// build returns (sum(output), d sum(output)/d x) flattened, for the given
	// attention closure. Same constant inputs every call so results are comparable.
	build := func(attn func(gr *g.Graph) *g.Node) (out, grad []float32) {
		exec := g.MustNewExec(be.Compute(), func(gr *g.Graph) []*g.Node {
			y := attn(gr)
			loss := g.ReduceAllSum(y)
			return []*g.Node{y, g.Gradient(loss, xConst(gr, B, T, cfg.Hidden))[0]}
		})
		res := exec.MustCall()
		return tensors.MustCopyFlatData[float32](res[0]), tensors.MustCopyFlatData[float32](res[1])
	}

	mkWeights := func(gr *g.Graph) (x, wq, wk, wv, wo *g.Node) {
		x = xConst(gr, B, T, cfg.Hidden)
		wq = wConst(gr, cfg.Hidden, cfg.NHeads*cfg.HeadDim, 1)
		wk = wConst(gr, cfg.Hidden, cfg.NKVHeads*cfg.HeadDim, 2)
		wv = wConst(gr, cfg.Hidden, cfg.NKVHeads*cfg.HeadDim, 3)
		wo = wConst(gr, cfg.NHeads*cfg.HeadDim, cfg.Hidden, 4)
		return
	}

	naiveOut, naiveGrad := build(func(gr *g.Graph) *g.Node {
		x, wq, wk, wv, wo := mkWeights(gr)
		return model.Attention(cfg, x, wq, wk, wv, wo, positions)
	})

	for _, chunk := range []int{2, 4} { // 2 divides T=6; 4 does not (blocks 4 and 2)
		cOut, cGrad := build(func(gr *g.Graph) *g.Node {
			x, wq, wk, wv, wo := mkWeights(gr)
			return model.AttentionChunked(cfg, x, wq, wk, wv, wo, positions, chunk)
		})
		assertClose(t, "output chunk="+itoa(chunk), naiveOut, cOut, 2e-4)
		assertClose(t, "grad chunk="+itoa(chunk), naiveGrad, cGrad, 2e-4)
	}
}

// xConst / wConst produce the same deterministic constants on every call (so the
// gradient node references the same x the forward used).
func xConst(gr *g.Graph, b, t, h int) *g.Node {
	return g.Const(gr, tensors.FromFlatDataAndDimensions(rampSeed(b*t*h, 7), b, t, h))
}

func wConst(gr *g.Graph, in, out, seed int) *g.Node {
	return g.Const(gr, tensors.FromFlatDataAndDimensions(rampSeed(in*out, seed), in, out))
}

// rampSeed is a deterministic small-value ramp, varied by seed.
func rampSeed(n, seed int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32((i+seed)%11)*0.05 - 0.25
	}
	return s
}

func assertClose(t *testing.T, name string, want, got []float32, tol float64) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s: len %d vs %d", name, len(want), len(got))
	}
	for i := range want {
		if d := math.Abs(float64(want[i] - got[i])); d > tol {
			t.Errorf("%s: elem %d: naive %v vs chunked %v (diff %v > %v)", name, i, want[i], got[i], d, tol)
		}
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}
