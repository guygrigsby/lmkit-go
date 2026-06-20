// Package train wires the lmkit-go model to GoMLX variables, an optimizer, and a
// loss for training. It uses the GoMLX training vocabulary (ml/model, ml/train,
// ...) but not the runtime (ADR-0009): the backend comes from gomlx.New().Compute().
package train

import (
	"fmt"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/initializer"

	lmodel "github.com/guygrigsby/lmkit-go/model"
)

// ModelVars declares one trainable variable per model weight in scope and returns
// them as a model.Weights whose Nodes are the variables' graph values. Norm scales
// initialize to 1; projections and the embedding to a small random normal. Called
// inside the training modelFn (gr := inputs[0].Graph()). Variable names mirror the
// weight structure (embed, final_norm, layer{i}_Wq, ...).
func ModelVars(scope *model.Scope, gr *g.Graph, cfg lmodel.Config) lmodel.Weights {
	norm := scope.WithInitializer(initializer.One)
	proj := scope.WithInitializer(initializer.RandomNormalFn(scope, 0.02))
	mk := func(s *model.Scope, name string, dims ...int) *g.Node {
		return s.VariableWithShape(name, shapes.Make(dtypes.Float32, dims...)).NodeValue(gr)
	}
	h, nh, nkv, hd, ffn, vocab := cfg.Hidden, cfg.NHeads, cfg.NKVHeads, cfg.HeadDim, cfg.FFNHidden, cfg.VocabSize
	w := lmodel.Weights{
		Embed:     mk(proj, "embed", vocab, h),
		FinalNorm: mk(norm, "final_norm", h),
	}
	for i := 0; i < cfg.NLayers; i++ {
		p := fmt.Sprintf("layer%d_", i)
		w.Layers = append(w.Layers, lmodel.LayerWeights{
			AttnNorm: mk(norm, p+"attn_norm", h),
			Wq:       mk(proj, p+"Wq", h, nh*hd),
			Wk:       mk(proj, p+"Wk", h, nkv*hd),
			Wv:       mk(proj, p+"Wv", h, nkv*hd),
			Wo:       mk(proj, p+"Wo", nh*hd, h),
			FFNNorm:  mk(norm, p+"ffn_norm", h),
			Wgate:    mk(proj, p+"Wgate", h, ffn),
			Wup:      mk(proj, p+"Wup", h, ffn),
			Wdown:    mk(proj, p+"Wdown", ffn, h),
		})
	}
	return w
}
