package model_test

import (
	"math"
	"math/rand"
	"strings"
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

// TestAttentionFlashGradMatchesDecomposed checks the flash BACKWARD: gradients of a fixed
// non-uniform loss w.r.t. every attention input (x, wQ, wK, wV, wO) match between the flash
// path (cuDNN fmhaSoftmaxBackward + the custom VJP) and the decomposed path, given identical
// bf16 inputs. The backward kernel and its hand-written gradient are the part most likely to
// be wrong, and the forward test cannot catch a bad VJP. On a non-cuda backend flash falls
// back to decomposed so this is a trivial wiring check; on cuda it validates the real kernel.
func TestAttentionFlashGradMatchesDecomposed(t *testing.T) {
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	if !strings.Contains(strings.ToLower(be.Compute().Name()), "cuda") {
		t.Logf("WARNING: backend %q is not cuda; flash falls back to decomposed so this parity is trivial", be.Compute().Name())
	}
	cfg := model.Config{Hidden: 256, NHeads: 4, NKVHeads: 2, HeadDim: 64, RopeBase: 10000}
	const B, T = 2, 128
	hidden, qkOut, kvOut := cfg.Hidden, cfg.NHeads*cfg.HeadDim, cfg.NKVHeads*cfg.HeadDim
	positions := make([]int, T)
	for i := range positions {
		positions[i] = i
	}
	cot := randF32(B*T*hidden, 99) // fixed random cotangent: a non-uniform loss exercises the full VJP

	// run returns gradients w.r.t. [x, wQ, wK, wV, wO], each flattened, for the given flash setting.
	run := func(flash bool) [][]float32 {
		model.UseFlashAttention = flash
		exec := g.MustNewExec(be.Compute(), func(gr *g.Graph) []*g.Node {
			bf := func(data []float32, dims ...int) *g.Node {
				return g.ConvertDType(g.Const(gr, tensors.FromFlatDataAndDimensions(data, dims...)), dtypes.BFloat16)
			}
			x := bf(randF32(B*T*hidden, 1), B, T, hidden)
			wQ := bf(randF32(hidden*qkOut, 2), hidden, qkOut)
			wK := bf(randF32(hidden*kvOut, 3), hidden, kvOut)
			wV := bf(randF32(hidden*kvOut, 4), hidden, kvOut)
			wO := bf(randF32(qkOut*hidden, 5), qkOut, hidden)
			out := model.Attention(cfg, x, wQ, wK, wV, wO, positions)
			cotN := bf(cot, B, T, hidden)
			loss := g.ReduceAllSum(g.Mul(out, cotN))
			grads := g.Gradient(loss, x, wQ, wK, wV, wO)
			outs := make([]*g.Node, len(grads))
			for i, gn := range grads {
				outs[i] = g.ConvertDType(gn, dtypes.Float32)
			}
			return outs
		})
		res := exec.MustCall()
		flat := make([][]float32, len(res))
		for i := range res {
			flat[i] = tensors.MustCopyFlatData[float32](res[i])
		}
		return flat
	}
	defer func() { model.UseFlashAttention = true }()

	flashG, decompG := run(true), run(false)
	names := []string{"dX", "dWq", "dWk", "dWv", "dWo"}
	for i := range names {
		var maxDiff, maxAbs float64
		for j := range flashG[i] {
			if d := math.Abs(float64(flashG[i][j] - decompG[i][j])); d > maxDiff {
				maxDiff = d
			}
			if a := math.Abs(float64(decompG[i][j])); a > maxAbs {
				maxAbs = a
			}
		}
		rel := maxDiff / (maxAbs + 1e-6)
		t.Logf("%s flash vs decomposed: relative max error %.5f (backend %s)", names[i], rel, be.Compute().Name())
		if rel > 0.06 {
			t.Errorf("%s: flash gradient diverges from decomposed: relative max error %.4f", names[i], rel)
		}
	}
}
