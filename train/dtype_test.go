package train

import (
	"testing"

	"github.com/guygrigsby/lmkit-go/backend"
)

// TestWantBF16 pins the M6 rule: bf16 compute only on CUDA, fp32 everywhere else.
func TestWantBF16(t *testing.T) {
	cuda := backend.Device{Kind: "xla", Config: "xla:cuda - PJRT [1 device(s)]"}
	xlaCPU := backend.Device{Kind: "xla", Config: "xla:cpu - PJRT [1 device(s)]"}
	simpleGo := backend.Device{Kind: "Go Backend", Config: "Go Portable Compute Backend (parallelism=8)"}

	cases := []struct {
		dtype string
		dev   backend.Device
		want  bool
	}{
		{"bfloat16", cuda, true},
		{"bfloat16", xlaCPU, false},
		{"bfloat16", simpleGo, false},
		{"float32", cuda, false},
		{"float32", simpleGo, false},
	}
	for _, c := range cases {
		if got := wantBF16(c.dtype, c.dev); got != c.want {
			t.Errorf("wantBF16(%q, %+v) = %v, want %v", c.dtype, c.dev, got, c.want)
		}
	}
}
