package model_test

import (
	"math"
	"testing"

	"github.com/gomlx/compute/dtypes"
	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/model"
)

// RMSNorm on bf16 input must stay close to the fp32 result (fp32-internal reduction).
func TestRMSNormBF16CloseToFP32(t *testing.T) {
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	h := 16
	xData := make([]float32, h)
	sc := make([]float32, h)
	for i := range xData {
		xData[i] = float32(i%7) - 3.0
		sc[i] = 1.0
	}
	run := func(dt dtypes.DType) []float32 {
		exec := g.MustNewExec(be.Compute(), func(x, scale *g.Node) *g.Node {
			if dt == dtypes.BFloat16 {
				x = g.ConvertDType(x, dtypes.BFloat16)
				scale = g.ConvertDType(scale, dtypes.BFloat16)
			}
			out := model.RMSNorm(x, scale, 1e-5)
			return g.ConvertDType(out, dtypes.Float32)
		})
		out := exec.MustCall1(
			tensors.FromFlatDataAndDimensions(xData, 1, 1, h),
			tensors.FromFlatDataAndDimensions(sc, h))
		return tensors.MustCopyFlatData[float32](out)
	}
	f32, bf16 := run(dtypes.Float32), run(dtypes.BFloat16)
	for i := range f32 {
		if d := math.Abs(float64(f32[i] - bf16[i])); d > 5e-2 {
			t.Errorf("elem %d: fp32 %v vs bf16 %v (diff %v > 5e-2)", i, f32[i], bf16[i], d)
		}
	}
}

// RoPE on a bf16 input must build and run: its host-built cos/sin tables have to
// match the input dtype, else the graph rejects Mul(bf16, fp32). Guards the bf16
// compute path the CPU/fp32 parity tests never exercise.
func TestRoPEBF16(t *testing.T) {
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	const b, tt, nH, hd = 1, 3, 2, 4
	xData := make([]float32, b*tt*nH*hd)
	for i := range xData {
		xData[i] = float32(i%5) - 2.0
	}
	positions := []int{0, 1, 2}
	exec := g.MustNewExec(be.Compute(), func(x *g.Node) *g.Node {
		x = g.ConvertDType(x, dtypes.BFloat16)
		out := model.RoPE(x, positions, 10000.0, hd)
		if out.DType() != dtypes.BFloat16 {
			t.Errorf("RoPE bf16 in -> %v out, want BFloat16", out.DType())
		}
		return g.ConvertDType(out, dtypes.Float32)
	})
	out := exec.MustCall1(tensors.FromFlatDataAndDimensions(xData, b, tt, nH, hd))
	for i, v := range tensors.MustCopyFlatData[float32](out) {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("RoPE bf16 out[%d] = %v, want finite", i, v)
		}
	}
}

// Attention on bf16 inputs must build and run: the additive causal mask (a host
// fp32 constant) has to match scores' dtype, else Add(bf16, fp32) is rejected.
func TestAttentionBF16(t *testing.T) {
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	cfg := model.Config{Hidden: 8, NHeads: 2, NKVHeads: 1, HeadDim: 4, RopeBase: 10000.0}
	const b, tt = 1, 3
	ramp := func(n int) []float32 {
		d := make([]float32, n)
		for i := range d {
			d[i] = float32(i%5)*0.1 - 0.2
		}
		return d
	}
	positions := []int{0, 1, 2}
	exec := g.MustNewExec(be.Compute(), func(x, wq, wk, wv, wo *g.Node) *g.Node {
		bf := func(n *g.Node) *g.Node { return g.ConvertDType(n, dtypes.BFloat16) }
		out := model.Attention(cfg, bf(x), bf(wq), bf(wk), bf(wv), bf(wo), positions)
		return g.ConvertDType(out, dtypes.Float32)
	})
	out := exec.MustCall1(
		tensors.FromFlatDataAndDimensions(ramp(b*tt*cfg.Hidden), b, tt, cfg.Hidden),
		tensors.FromFlatDataAndDimensions(ramp(cfg.Hidden*cfg.NHeads*cfg.HeadDim), cfg.Hidden, cfg.NHeads*cfg.HeadDim),
		tensors.FromFlatDataAndDimensions(ramp(cfg.Hidden*cfg.NKVHeads*cfg.HeadDim), cfg.Hidden, cfg.NKVHeads*cfg.HeadDim),
		tensors.FromFlatDataAndDimensions(ramp(cfg.Hidden*cfg.NKVHeads*cfg.HeadDim), cfg.Hidden, cfg.NKVHeads*cfg.HeadDim),
		tensors.FromFlatDataAndDimensions(ramp(cfg.NHeads*cfg.HeadDim*cfg.Hidden), cfg.NHeads*cfg.HeadDim, cfg.Hidden))
	for i, v := range tensors.MustCopyFlatData[float32](out) {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("Attention bf16 out[%d] = %v, want finite", i, v)
		}
	}
}
