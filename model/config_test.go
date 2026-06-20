package model

import (
	"os"
	"path/filepath"
	"testing"
)

// TestLoadConfig gates the JSON tags against the Python-stack convention used by the
// golden fixtures (model/testdata/model.json): vocab, n_layers, n_heads, ffn_hidden,
// seq_len. A tag that diverges from that convention would silently leave its field at
// zero, so this fixture uses those exact keys with lm-100m values and asserts each one.
func TestLoadConfig(t *testing.T) {
	const j = `{
		"vocab": 32000, "hidden": 768, "n_layers": 12, "n_heads": 12,
		"n_kv_heads": 4, "head_dim": 64, "ffn_hidden": 2048,
		"seq_len": 2048, "rope_base": 10000, "rms_eps": 1e-5
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
		VocabSize: 32000, Hidden: 768, NLayers: 12, NHeads: 12, NKVHeads: 4,
		HeadDim: 64, FFNHidden: 2048, SeqLen: 2048, RopeBase: 10000, RMSEps: 1e-5,
	}
	if c != want {
		t.Errorf("LoadConfig =\n %+v\nwant\n %+v", c, want)
	}
}
