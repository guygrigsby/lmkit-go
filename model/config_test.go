package model

import (
	"os"
	"path/filepath"
	"testing"
)

// TestLoadConfig checks the snake_case JSON tags map onto the struct fields. A wrong
// tag would silently leave a field at its zero value, so assert each one round-trips.
func TestLoadConfig(t *testing.T) {
	const j = `{
		"hidden": 768, "n_layer": 12, "n_head": 12, "n_kv_heads": 4,
		"head_dim": 64, "intermediate": 2048, "vocab_size": 32000,
		"block": 2048, "rope_base": 10000, "rms_eps": 1e-5
	}`
	p := filepath.Join(t.TempDir(), "model.json")
	if err := os.WriteFile(p, []byte(j), 0o644); err != nil {
		t.Fatal(err)
	}
	c, err := LoadConfig(p)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	want := Config{
		Hidden: 768, NLayers: 12, NHeads: 12, NKVHeads: 4, HeadDim: 64,
		FFNHidden: 2048, VocabSize: 32000, Block: 2048, RopeBase: 10000, RMSEps: 1e-5,
	}
	if c != want {
		t.Errorf("LoadConfig =\n %+v\nwant\n %+v", c, want)
	}
}
