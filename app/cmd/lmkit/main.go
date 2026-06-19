// Command lmkit is the lmkit-go CLI. For Milestone 1 it offers `quickstart`,
// which runs the backend bring-up proofs and prints the selected device.
package main

import (
	"fmt"
	"math"
	"os"

	"github.com/guygrigsby/lmkit-go/backend"
	"github.com/guygrigsby/lmkit-go/backend/gomlx"
)

func main() {
	if len(os.Args) < 2 || os.Args[1] != "quickstart" {
		fmt.Fprintln(os.Stderr, "usage: lmkit quickstart")
		os.Exit(2)
	}
	if err := quickstart(); err != nil {
		fmt.Fprintf(os.Stderr, "FAIL: %v\n", err)
		os.Exit(1)
	}
}

func quickstart() error {
	be, err := gomlx.New()
	if err != nil {
		return err
	}
	dev := be.Device()
	fmt.Printf("device: kind=%q config=%q\n", dev.Kind, dev.Config)

	mm, err := be.MatMul(
		backend.Tensor{Shape: []int{2, 3}, Data: []float32{1, 2, 3, 4, 5, 6}},
		backend.Tensor{Shape: []int{3, 2}, Data: []float32{7, 8, 9, 10, 11, 12}},
	)
	if err != nil {
		return err
	}
	if !eq(mm.Data, []float32{58, 64, 139, 154}) {
		return fmt.Errorf("matmul = %v, want [58 64 139 154]", mm.Data)
	}
	fmt.Printf("matmul   OK  %v\n", mm.Data)

	gr, err := be.GradSumSquares(backend.Tensor{Shape: []int{3}, Data: []float32{1, 2, 3}})
	if err != nil {
		return err
	}
	if !eq(gr.Data, []float32{2, 4, 6}) {
		return fmt.Errorf("grad = %v, want [2 4 6]", gr.Data)
	}
	fmt.Printf("gradient OK  %v\n", gr.Data)

	w, loss, err := be.FitConstant(3.0, 800)
	if err != nil {
		return err
	}
	if math.Abs(float64(w-3.0)) > 0.05 || loss > 1e-2 {
		return fmt.Errorf("adamw w=%v loss=%v, want w~3 loss<1e-2", w, loss)
	}
	fmt.Printf("adamw    OK  w=%.4f loss=%.2e\n", w, loss)
	return nil
}

func eq(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
