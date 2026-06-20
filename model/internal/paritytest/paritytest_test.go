package paritytest

import (
	"testing"

	"github.com/gomlx/gomlx/core/tensors"
)

// A token id well above 2^24 must round-trip exactly through the fixture path
// (float32 would have truncated it).
func TestIntTensorRoundTrip(t *testing.T) {
	const big = 16_777_217 // 2^24 + 1: smallest int not exactly representable as float32
	tn := Tensor{Shape: []int{1, 2}, DType: "i32", Data: []float64{big, 7}}
	got := tensors.MustCopyFlatData[int32](tn.ToTensor())
	if got[0] != big || got[1] != 7 {
		t.Errorf("ids = %v, want [%d 7]", got, big)
	}
}

func TestFloatTensorDefault(t *testing.T) {
	tn := Tensor{Shape: []int{2}, Data: []float64{1.5, -2.25}} // no DType -> float32
	got := tensors.MustCopyFlatData[float32](tn.ToTensor())
	if got[0] != 1.5 || got[1] != -2.25 {
		t.Errorf("data = %v, want [1.5 -2.25]", got)
	}
}
