// Package paritytest loads JSON golden fixtures and compares tensors for the
// model block parity tests. It does not import the runtime (ADR-0009): callers
// obtain a backend via gomlx.New().Compute() and run the graph themselves.
package paritytest

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/gomlx/gomlx/core/tensors"
)

// Tensor is a fixture tensor: row-major data plus its shape. Data is JSON-native
// float64; ToTensor casts to the tensor dtype. DType "" or "f32" => float32;
// "i32" => int32 (for token ids / indices, which must round-trip exactly — float32
// would truncate ids above 2^24).
type Tensor struct {
	Shape []int     `json:"shape"`
	DType string    `json:"dtype,omitempty"`
	Data  []float64 `json:"data"`
}

// ToTensor builds a GoMLX tensor of the fixture's dtype.
func (tn Tensor) ToTensor() *tensors.Tensor {
	if tn.DType == "i32" {
		d := make([]int32, len(tn.Data))
		for i, v := range tn.Data {
			d[i] = int32(v)
		}
		return tensors.FromFlatDataAndDimensions(d, tn.Shape...)
	}
	d := make([]float32, len(tn.Data))
	for i, v := range tn.Data {
		d[i] = float32(v)
	}
	return tensors.FromFlatDataAndDimensions(d, tn.Shape...)
}

// Fixture is one block's golden: config scalars, named inputs and weights, and
// the expected output.
type Fixture struct {
	Config   map[string]float64 `json:"config"`
	Inputs   map[string]Tensor  `json:"inputs"`
	Weights  map[string]Tensor  `json:"weights"`
	Expected Tensor             `json:"expected"`
}

// Load reads and decodes a fixture JSON file.
func Load(t *testing.T, path string) Fixture {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read fixture %s: %v", path, err)
	}
	var f Fixture
	if err := json.Unmarshal(b, &f); err != nil {
		t.Fatalf("decode fixture %s: %v", path, err)
	}
	return f
}

// AssertClose fails if got and want differ by more than tol elementwise.
func AssertClose(t *testing.T, got []float32, want Tensor, tol float32) {
	t.Helper()
	if len(got) != len(want.Data) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want.Data))
	}
	var maxDiff float32
	for i := range got {
		d := float32(math.Abs(float64(got[i]) - want.Data[i]))
		if d > maxDiff {
			maxDiff = d
		}
		if d > tol {
			t.Errorf("element %d: got %v, want %v (diff %v > tol %v)", i, got[i], want.Data[i], d, tol)
		}
	}
	t.Logf("max abs diff = %g (tol %g)", maxDiff, tol)
}
