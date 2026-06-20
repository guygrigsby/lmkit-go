// Package model implements Llama building blocks in GoMLX core/graph ops
// (ADR-0009: model may use the op vocabulary; the runtime stays in backend/).
package model

// Config holds Llama block hyperparameters. Only the fields a given block needs
// are read by that block.
type Config struct {
	Hidden    int
	NLayers   int
	NHeads    int
	NKVHeads  int
	HeadDim   int
	FFNHidden int
	VocabSize int
	RopeBase  float64
	RMSEps    float64
}
