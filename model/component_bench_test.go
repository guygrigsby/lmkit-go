package model_test

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/gomlx/compute/dtypes"
	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/model"
)

// TestComponentBackwardBench isolates the forward+backward cost of each layer component at the
// real lm-100m shape (B=2, T=2048, bf16) to find which op's BACKWARD dominates the ~215 ms/layer
// backward. Gated on GOMLX_COMPONENT_BENCH=true and a cuda backend.
func TestComponentBackwardBench(t *testing.T) {
	if os.Getenv("GOMLX_COMPONENT_BENCH") != "true" {
		t.Skip("set GOMLX_COMPONENT_BENCH=true")
	}
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	if !strings.Contains(strings.ToLower(be.Compute().Name()+" "+be.Compute().Description()), "cuda") {
		t.Skipf("needs cuda; got %q", be.Compute().Name())
	}
	model.UseFlashAttention = true
	const B, T, H, FFN, nH, nKV, hd = 2, 2048, 768, 2048, 12, 4, 64
	cfg := model.Config{Hidden: H, NHeads: nH, NKVHeads: nKV, HeadDim: hd, FFNHidden: FFN, RopeBase: 10000, RMSEps: 1e-5}
	positions := make([]int, T)
	for i := range positions {
		positions[i] = i
	}

	bf := func(gr *g.Graph, seed int64, dims ...int) *g.Node {
		n := 1
		for _, d := range dims {
			n *= d
		}
		return g.ConvertDType(g.Const(gr, tensors.FromFlatDataAndDimensions(randF32(n, seed), dims...)), dtypes.BFloat16)
	}

	bench := func(name string, build func(gr *g.Graph) (out *g.Node, inputs []*g.Node)) {
		exec := g.MustNewExec(be.Compute(), func(gr *g.Graph) []*g.Node {
			out, inputs := build(gr)
			loss := g.ReduceAllSum(g.ConvertDType(out, dtypes.Float32))
			return append([]*g.Node{loss}, g.Gradient(loss, inputs...)...)
		})
		exec.MustCall() // compile + warmup
		const iters = 10
		start := time.Now()
		for i := 0; i < iters; i++ {
			exec.MustCall()
		}
		t.Logf("%-14s fwd+bwd: %v", name, time.Since(start)/iters)
	}

	bench("MatMul3D", func(gr *g.Graph) (*g.Node, []*g.Node) {
		x, w := bf(gr, 1, B, T, H), bf(gr, 2, H, FFN) // [B,T,H] @ [H,F]
		return g.MatMul(x, w), []*g.Node{x, w}
	})
	bench("MatMul2D", func(gr *g.Graph) (*g.Node, []*g.Node) {
		x, w := bf(gr, 1, B*T, H), bf(gr, 2, H, FFN) // [B*T,H] @ [H,F]
		return g.MatMul(x, w), []*g.Node{x, w}
	})
	bench("SwiGLU-noSiLU", func(gr *g.Graph) (*g.Node, []*g.Node) {
		x, wg, wu, wd := bf(gr, 1, B*T, H), bf(gr, 2, H, FFN), bf(gr, 3, H, FFN), bf(gr, 4, FFN, H)
		return g.MatMul(g.Mul(g.MatMul(x, wg), g.MatMul(x, wu)), wd), []*g.Node{x, wg, wu, wd}
	})
	bench("RMSNorm", func(gr *g.Graph) (*g.Node, []*g.Node) {
		x, s := bf(gr, 1, B, T, H), bf(gr, 2, H)
		return model.RMSNorm(x, s, 1e-5), []*g.Node{x, s}
	})
	bench("RoPE", func(gr *g.Graph) (*g.Node, []*g.Node) {
		x := bf(gr, 1, B, T, nH, hd)
		return model.RoPE(x, positions, 10000, hd), []*g.Node{x}
	})
	bench("SwiGLU-FFN", func(gr *g.Graph) (*g.Node, []*g.Node) {
		x, wg, wu, wd := bf(gr, 1, B, T, H), bf(gr, 2, H, FFN), bf(gr, 3, H, FFN), bf(gr, 4, FFN, H)
		return model.SwiGLU(x, wg, wu, wd), []*g.Node{x, wg, wu, wd}
	})
	bench("Attention", func(gr *g.Graph) (*g.Node, []*g.Node) {
		x := bf(gr, 1, B, T, H)
		wq, wk, wv, wo := bf(gr, 2, H, nH*hd), bf(gr, 3, H, nKV*hd), bf(gr, 4, H, nKV*hd), bf(gr, 5, nH*hd, H)
		return model.Attention(cfg, x, wq, wk, wv, wo, positions), []*g.Node{x, wq, wk, wv, wo}
	})
	bench("DecoderLayer", func(gr *g.Graph) (*g.Node, []*g.Node) {
		x := bf(gr, 1, B, T, H)
		lw := model.LayerWeights{
			AttnNorm: bf(gr, 2, H), Wq: bf(gr, 3, H, nH*hd), Wk: bf(gr, 4, H, nKV*hd), Wv: bf(gr, 5, H, nKV*hd), Wo: bf(gr, 6, nH*hd, H),
			FFNNorm: bf(gr, 7, H), Wgate: bf(gr, 8, H, FFN), Wup: bf(gr, 9, H, FFN), Wdown: bf(gr, 10, FFN, H),
		}
		return model.DecoderLayer(cfg, x, lw, positions), []*g.Node{x, lw.Wq, lw.Wgate}
	})
}
