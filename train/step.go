package train

import (
	"path"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/initializer"
	gomlxtrain "github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/optimizer"

	gomlxbackend "github.com/guygrigsby/lmkit-go/backend/gomlx"
	lmodel "github.com/guygrigsby/lmkit-go/model"
)

// modelLoss builds the fp32 next-token CE loss for one micro-batch. Variables are
// fp32 (master); on CUDA the weights are cast to computeDT (bf16) for the matmuls,
// logits upcast to fp32 for the loss.
func modelLoss(scope *model.Scope, gr *g.Graph, mcfg lmodel.Config, x, y *g.Node, computeDT dtypes.DType, positions []int) *g.Node {
	w := ModelVars(scope, gr, mcfg) // fp32 vars
	if computeDT != dtypes.Float32 {
		w = castWeights(w, computeDT) // ConvertDType each weight node
	}
	logits := lmodel.Forward(mcfg, w, x, positions) // computeDT on CUDA
	logits = g.ConvertDType(logits, dtypes.Float32)  // loss in fp32
	return loss.SparseCategoricalCrossEntropyLogits([]*g.Node{y}, []*g.Node{logits})
}

// castWeights returns a copy of w with every weight node cast to dt (for bf16 compute).
func castWeights(w lmodel.Weights, dt dtypes.DType) lmodel.Weights {
	c := func(n *g.Node) *g.Node { return g.ConvertDType(n, dt) }
	out := lmodel.Weights{Embed: c(w.Embed), FinalNorm: c(w.FinalNorm)}
	for _, lw := range w.Layers {
		out.Layers = append(out.Layers, lmodel.LayerWeights{
			AttnNorm: c(lw.AttnNorm), Wq: c(lw.Wq), Wk: c(lw.Wk), Wv: c(lw.Wv), Wo: c(lw.Wo),
			FFNNorm: c(lw.FFNNorm), Wgate: c(lw.Wgate), Wup: c(lw.Wup), Wdown: c(lw.Wdown),
		})
	}
	return out
}

// accumulatorVar returns (or creates) the accumulator variable for a trainable variable.
// Mirrors GoMLX accgradients.go iterTrainableAndAccumulatorVariables: variables live under
// path.Join("/", AccumulatedGradientsScope, originalScope) with a zero initializer.
func accumulatorVar(scope *model.Scope, v *model.Variable) *model.Variable {
	scopePath := path.Join("/", gomlxtrain.AccumulatedGradientsScope, v.Scope())
	return scope.Store().Scope(scopePath).
		WithInitializer(initializer.Zero).
		VariableWithShape(v.Name(), v.Shape().Clone()).
		SetTrainable(false)
}

// Stepper drives one optimizer step over GradAccum micro-batches via two compiled
// model.Execs: an accumulate-only graph and an accumulate-then-apply graph.
type Stepper struct {
	accOnly   *model.Exec
	accApply  *model.Exec
	gradAccum int
}

// NewStepper builds the accumulate/apply Execs and returns a Stepper.
// bk is the backend wrapper (the runtime stays behind the anti-corruption seam;
// .Compute() is the only runtime reference and is passed straight to model.NewExec).
// gradClip is the global-norm clip threshold (lm-100m uses 1.0). store holds all variables.
// opt must implement gomlxtrain.OptimizeWithGradients (both Adam and SGD do).
func NewStepper(
	bk *gomlxbackend.Backend,
	store *model.Store,
	mcfg lmodel.Config,
	positions []int,
	gradAccum int,
	gradClip float64,
	opt optimizer.Interface,
	computeDT dtypes.DType,
) *Stepper {
	owg, ok := opt.(gomlxtrain.OptimizeWithGradients)
	if !ok {
		panic("optimizer does not implement OptimizeWithGradients; use Adam or SGD")
	}

	// buildAccGraph is the shared graph-building core: computes loss, grads, and
	// adds each grad into its accumulator variable. Returns accumulated grads and loss.
	// If applyGradients is true it also: averages, clips, applies, and zeros accumulators.
	buildGraph := func(applyGradients bool) func(*model.Scope, *g.Node, *g.Node) *g.Node {
		return func(scope *model.Scope, x, y *g.Node) *g.Node {
			gr := x.Graph()
			lossNode := modelLoss(scope, gr, mcfg, x, y, computeDT, positions)

			// Compute per-variable gradients in trainable-variable order.
			grads := scope.BuildTrainableVariablesGradientsGraph(lossNode)

			// Accumulate: for each trainable variable (in the same order as grads),
			// update its accumulator. This mirrors accgradients.go lines 111-125.
			varIdx := 0
			for v := range scope.IterVariables() {
				if !v.Trainable || !v.InUseByGraph(gr) {
					continue
				}
				acc := accumulatorVar(scope, v)
				// accumulated = acc_prev + grad; store both back as in accgradients.go:
				// grads[varIdx] carries the new accumulated value for the apply step.
				grads[varIdx] = g.Add(acc.NodeValue(gr), grads[varIdx])
				acc.SetNodeValue(grads[varIdx])
				varIdx++
			}

			if !applyGradients {
				return lossNode
			}

			// Mean over GradAccum micro-batches.
			ratio := 1.0 / float64(gradAccum)
			for ii := range grads {
				grads[ii] = g.MulScalar(grads[ii], ratio)
			}

			// Global-norm clip then apply with AdamW.
			clipped := clipByGlobalNorm(grads, gradClip)
			owg.UpdateGraphWithGradients(scope, clipped, dtypes.Float32)

			// Zero all accumulators.
			for v := range scope.IterVariables() {
				if !v.Trainable || !v.InUseByGraph(gr) {
					continue
				}
				acc := accumulatorVar(scope, v)
				acc.SetNodeValue(g.Zeros(gr, acc.Shape()))
			}

			return lossNode
		}
	}

	be := bk.Compute() // sole runtime reference; handed straight to model.NewExec
	accOnly := model.MustNewExec(be, store, buildGraph(false))
	accApply := model.MustNewExec(be, store, buildGraph(true))

	return &Stepper{
		accOnly:   accOnly,
		accApply:  accApply,
		gradAccum: gradAccum,
	}
}

// Step runs one optimizer step: (GradAccum-1) accumulate calls then one apply call.
// x is [B,T] int32 token ids; y is [B,T,1] int32 next-token ids.
// Returns the loss from the apply micro-batch.
func (s *Stepper) Step(x, y *tensors.Tensor) float64 {
	for i := 0; i < s.gradAccum-1; i++ {
		s.accOnly.MustCall(x, y)
	}
	out := s.accApply.MustCall(x, y)
	return shapes.ConvertTo[float64](out[0].Value())
}
