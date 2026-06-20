package model

import g "github.com/gomlx/gomlx/core/graph"

// LayerWeights are the weight Nodes for one decoder layer (no biases).
type LayerWeights struct {
	AttnNorm          *g.Node // [H]   RMSNorm scale before attention
	Wq, Wk, Wv, Wo   *g.Node // attention projections
	FFNNorm           *g.Node // [H]   RMSNorm scale before the FFN
	Wgate, Wup, Wdown *g.Node // SwiGLU projections
}

// Weights are the weight Nodes for the whole model. Embed is tied: it is both the
// token embedding table and (transposed, via TiedLogits) the output projection.
type Weights struct {
	Embed     *g.Node        // [V,H]
	Layers    []LayerWeights // len == Config.NLayers
	FinalNorm *g.Node        // [H]
}
