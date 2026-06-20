package lm100men_test

import (
	"path/filepath"
	"runtime"
	"testing"

	"github.com/guygrigsby/lmkit-go/model"
	"github.com/guygrigsby/lmkit-go/train"
)

// pkgDir returns the directory of this source file, so config path lookups work
// regardless of where go test is invoked from.
func pkgDir() string {
	_, f, _, _ := runtime.Caller(0)
	return filepath.Dir(f)
}

// TestModelJSONNonZero verifies that model.json loads with non-zero fields,
// catching any JSON tag drift that would silently zero a field.
func TestModelJSONNonZero(t *testing.T) {
	c, err := model.LoadConfig(filepath.Join(pkgDir(), "model.json"))
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	cases := []struct {
		name string
		got  int
		want int
	}{
		{"NLayers", c.NLayers, 12},
		{"NHeads", c.NHeads, 12},
		{"NKVHeads", c.NKVHeads, 4},
		{"FFNHidden", c.FFNHidden, 2048},
		{"VocabSize", c.VocabSize, 32000},
		{"SeqLen", c.SeqLen, 2048},
		{"Hidden", c.Hidden, 768},
		{"HeadDim", c.HeadDim, 64},
	}
	for _, tc := range cases {
		if tc.got != tc.want {
			t.Errorf("model.json %s = %d, want %d", tc.name, tc.got, tc.want)
		}
	}
	if c.RopeBase != 10000 {
		t.Errorf("RopeBase = %v, want 10000", c.RopeBase)
	}
	if c.RMSEps != 1e-5 {
		t.Errorf("RMSEps = %v, want 1e-5", c.RMSEps)
	}
}

// TestTrainConfigKeys verifies that config.json parses with the key fields present
// (guards against JSON tag drift in train.Config).
func TestTrainConfigKeys(t *testing.T) {
	cfg, err := train.LoadConfig(filepath.Join(pkgDir(), "config.json"))
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.LR != 4e-4 {
		t.Errorf("lr = %v, want 4e-4", cfg.LR)
	}
	if cfg.MaxSteps != 200000 {
		t.Errorf("max_steps = %d, want 200000", cfg.MaxSteps)
	}
	if cfg.Dtype != "bfloat16" {
		t.Errorf("dtype = %q, want bfloat16", cfg.Dtype)
	}
	if cfg.GradAccum != 32 {
		t.Errorf("grad_accum = %d, want 32", cfg.GradAccum)
	}
	if cfg.BatchSize != 2 {
		t.Errorf("batch_size = %d, want 2", cfg.BatchSize)
	}
}
