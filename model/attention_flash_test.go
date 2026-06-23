package model_test

import (
	"math"
	"math/rand"
	"testing"

	"github.com/gomlx/compute/dtypes"
	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/model"
)

func randF32(n int, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed))
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(r.NormFloat64() * 0.25)
	}
	return out
}

// TestAttentionFlashMatchesDecomposed checks the flash wiring: the bf16 Attention block produces
// the same output through the flash path (UseFlashAttention=true) as through the decomposed path
// (false), given identical bf16 inputs. On a non-cuda backend flash falls back to a decomposed
// attention, so this still exercises the wiring (shapes, scale, head merge) in CI; on cuda it
// validates the kernel itself.
func TestAttentionFlashMatchesDecomposed(t *testing.T) {
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	cfg := model.Config{Hidden: 256, NHeads: 4, NKVHeads: 2, HeadDim: 64, RopeBase: 10000}
	const B, T = 1, 128
	hidden, qkOut, kvOut := cfg.Hidden, cfg.NHeads*cfg.HeadDim, cfg.NKVHeads*cfg.HeadDim
	positions := make([]int, T)
	for i := range positions {
		positions[i] = i
	}

	run := func(flash bool) []float32 {
		model.UseFlashAttention = flash
		exec := g.MustNewExec(be.Compute(), func(gr *g.Graph) *g.Node {
			bf := func(data []float32, dims ...int) *g.Node {
				return g.ConvertDType(g.Const(gr, tensors.FromFlatDataAndDimensions(data, dims...)), dtypes.BFloat16)
			}
			x := bf(randF32(B*T*hidden, 1), B, T, hidden)
			wQ := bf(randF32(hidden*qkOut, 2), hidden, qkOut)
			wK := bf(randF32(hidden*kvOut, 3), hidden, kvOut)
			wV := bf(randF32(hidden*kvOut, 4), hidden, kvOut)
			wO := bf(randF32(qkOut*hidden, 5), qkOut, hidden)
			return g.ConvertDType(model.Attention(cfg, x, wQ, wK, wV, wO, positions), dtypes.Float32)
		})
		return tensors.MustCopyFlatData[float32](exec.MustCall1())
	}
	defer func() { model.UseFlashAttention = true }()

	flashOut, decompOut := run(true), run(false)

	var maxDiff, maxAbs float64
	for i := range flashOut {
		if d := math.Abs(float64(flashOut[i] - decompOut[i])); d > maxDiff {
			maxDiff = d
		}
		if a := math.Abs(float64(decompOut[i])); a > maxAbs {
			maxAbs = a
		}
	}
	rel := maxDiff / (maxAbs + 1e-6)
	t.Logf("flash vs decomposed: relative max error %.5f (backend %s)", rel, be.Compute().Name())
	if rel > 0.05 {
		t.Fatalf("flash attention output diverges from decomposed: relative max error %.4f", rel)
	}
}
