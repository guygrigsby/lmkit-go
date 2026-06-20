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
