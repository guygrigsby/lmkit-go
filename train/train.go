package train

import (
	"fmt"

	"github.com/gomlx/compute/shapes"
	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/dataset"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/optimizer"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	lmodel "github.com/guygrigsby/lmkit-go/model"
)

// Overfit trains a fresh model on one fixed batch for `steps` steps with AdamW and
// next-token cross-entropy, returning the loss at the first step and after `steps`.
// inputs is [B,T] int token ids; labels is [B,T,1] int next-token ids (the trailing
// 1 is required by SparseCategoricalCrossEntropyLogits). seed seeds variable init
// (for reproducible debugging; the loop converges for any seed).
func Overfit(cfg lmodel.Config, inputs, labels *tensors.Tensor, steps int, lr, weightDecay float64, seed int64) (start, end float64, err error) {
	bk, err := gomlx.New()
	if err != nil {
		return 0, 0, fmt.Errorf("train: backend: %w", err)
	}
	be := bk.Compute()

	store := model.NewStore()
	store.RootScope().SetParam(model.ParamInitialSeed, seed)

	tt := inputs.Shape().Dimensions[1] // T
	positions := make([]int, tt)
	for i := range positions {
		positions[i] = i
	}
	modelFn := func(scope *model.Scope, spec any, in []*g.Node) []*g.Node {
		w := ModelVars(scope, in[0].Graph(), cfg)
		return []*g.Node{lmodel.Forward(cfg, w, in[0], positions)}
	}

	ds, err := dataset.InMemoryFromData(be, "overfit", []any{inputs}, []any{labels})
	if err != nil {
		return 0, 0, fmt.Errorf("train: dataset: %w", err)
	}
	bsz := inputs.Shape().Dimensions[0] // B
	ds = ds.Infinite(true).BatchSize(bsz, false)

	opt := optimizer.Adam().WeightDecay(weightDecay).LearningRate(lr).Done()
	trainer := train.NewTrainer(be, store, modelFn, loss.SparseCategoricalCrossEntropyLogits, opt, nil, nil)
	loop := train.NewLoop(trainer)

	m0, err := loop.RunSteps(ds, 1) // first step: loss at (near) init
	if err != nil {
		return 0, 0, fmt.Errorf("train: initial step: %w", err)
	}
	start = shapes.ConvertTo[float64](m0[0].Value())
	mN, err := loop.RunSteps(ds, steps) // `steps` more
	if err != nil {
		return 0, 0, fmt.Errorf("train: steps: %w", err)
	}
	end = shapes.ConvertTo[float64](mN[0].Value())
	return start, end, nil
}
