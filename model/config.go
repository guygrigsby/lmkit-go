// Package model implements Llama building blocks in GoMLX core/graph ops
// (ADR-0009: model may use the op vocabulary; the runtime stays in backend/).
package model

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config holds Llama block hyperparameters. Only the fields a given block needs
// are read by that block. JSON tags match the Python lmkit convention used by the
// golden fixtures (model/testdata/model.json) so a model.json round-trips between
// the two stacks. SeqLen is the context length (max sequence / RoPE positions);
// the data loader sizes its blocks from it.
type Config struct {
	VocabSize int     `json:"vocab"`
	Hidden    int     `json:"hidden"`
	NLayers   int     `json:"n_layers"`
	NHeads    int     `json:"n_heads"`
	NKVHeads  int     `json:"n_kv_heads"`
	HeadDim   int     `json:"head_dim"`
	FFNHidden int     `json:"ffn_hidden"`
	SeqLen    int     `json:"seq_len"`
	RopeBase  float64 `json:"rope_base"`
	RMSEps    float64 `json:"rms_eps"`
}

// LoadConfig reads a JSON model config.
func LoadConfig(path string) (Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Config{}, fmt.Errorf("read model config %q: %w", path, err)
	}
	var c Config
	if err := json.Unmarshal(b, &c); err != nil {
		return Config{}, fmt.Errorf("parse model config %q: %w", path, err)
	}
	return c, nil
}
