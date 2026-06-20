package gomlx

import (
	"math"
	"testing"

	"github.com/guygrigsby/lmkit-go/backend"
)

func TestMatMul(t *testing.T) {
	be, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	// [2x3] @ [3x2] = [2x2]; known product [58 64 139 154].
	a := backend.Tensor{Shape: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}}
	b := backend.Tensor{Shape: []int{3, 2}, Data: []float32{7, 8, 9, 10, 11, 12}}
	got, err := be.MatMul(a, b)
	if err != nil {
		t.Fatalf("MatMul: %v", err)
	}
	want := []float32{58, 64, 139, 154}
	if len(got.Data) != len(want) {
		t.Fatalf("len = %d, want %d", len(got.Data), len(want))
	}
	for i, w := range want {
		if got.Data[i] != w {
			t.Errorf("Data[%d] = %v, want %v", i, got.Data[i], w)
		}
	}
	if got.Shape[0] != 2 || got.Shape[1] != 2 {
		t.Errorf("Shape = %v, want [2 2]", got.Shape)
	}
}

// TestMatMulShapeMismatch pins that a caller-supplied inner-dim mismatch returns
// an error rather than panicking out of GoMLX's Must* path.
func TestMatMulShapeMismatch(t *testing.T) {
	be, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	a := backend.Tensor{Shape: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}}
	b := backend.Tensor{Shape: []int{4, 2}, Data: []float32{1, 2, 3, 4, 5, 6, 7, 8}} // inner 3 != 4
	if _, err := be.MatMul(a, b); err == nil {
		t.Fatal("MatMul with mismatched inner dims: want error, got nil")
	}
}

func TestGradSumSquares(t *testing.T) {
	be, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	x := backend.Tensor{Shape: []int{3}, Data: []float32{1, 2, 3}}
	got, err := be.GradSumSquares(x)
	if err != nil {
		t.Fatalf("GradSumSquares: %v", err)
	}
	want := []float32{2, 4, 6} // d/dx sum(x^2) = 2x
	for i, w := range want {
		if got.Data[i] != w {
			t.Errorf("Data[%d] = %v, want %v", i, got.Data[i], w)
		}
	}
}

func TestFitConstant(t *testing.T) {
	be, err := New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	const target = 3.0

	// No decay: AdamW (decay=0) drives w -> target.
	w, loss, err := be.FitConstant(target, 800, 0.0)
	if err != nil {
		t.Fatalf("FitConstant(decay=0): %v", err)
	}
	if math.Abs(float64(w-target)) > 0.05 {
		t.Errorf("w = %v, want ~%v", w, target)
	}
	if loss > 1e-2 {
		t.Errorf("loss = %v, want < 1e-2", loss)
	}

	// Decoupled weight decay: with decay>0 AdamW also pulls w toward 0, so it
	// settles measurably below target. This is the behavior that distinguishes
	// AdamW from plain Adam — a zero-decay run could not show it.
	wDecay, _, err := be.FitConstant(target, 800, 1.0)
	if err != nil {
		t.Fatalf("FitConstant(decay=1): %v", err)
	}
	if wDecay >= w-0.1 {
		t.Errorf("weight decay had no measurable effect: wDecay=%v, want < %v (w-0.1)", wDecay, w-0.1)
	}
}
