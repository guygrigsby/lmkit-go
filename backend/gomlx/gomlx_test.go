package gomlx

import (
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
