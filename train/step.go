package train

import (
	"os"
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
func modelLoss(scope *model.Scope, gr *g.Graph, mcfg lmodel.Config, x, y *g.Node, computeDT dtypes.DType, positions []int, checkpoint bool) *g.Node {
	w := ModelVars(scope, gr, mcfg) // fp32 vars
	if computeDT != dtypes.Float32 {
		w = castWeights(w, computeDT) // ConvertDType each weight node
	}
	// Per-layer gradient checkpointing recomputes layer activations in the backward
	// pass instead of holding them — the only way a deep model fits training on a
	// small GPU. Off for eval (no backward, so nothing to checkpoint).
	// Profiling ablation: skip the LM head (TiedLogits + cross-entropy) and build a trivial
	// loss from the hidden states. The backward still flows through embed+layers+norm, so
	// (full step - this) isolates the LM-head fwd+bwd cost (the 32k-vocab projection + CE).
	if os.Getenv("GOMLX_NOLMHEAD") == "true" {
		fh := lmodel.ForwardHidden
		if checkpoint {
			fh = lmodel.ForwardHiddenCheckpointed
		}
		return g.ReduceAllMean(fh(mcfg, w, x, positions))
	}
	fwd := lmodel.Forward
	if checkpoint {
		fwd = lmodel.ForwardCheckpointed
	}
	logits := fwd(mcfg, w, x, positions)            // computeDT on CUDA
	logits = g.ConvertDType(logits, dtypes.Float32) // loss in fp32
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
	store     *model.Store
	gradAccum int
	lrVar     *model.Variable // cached after first Step; nil until then
}

// NewStepper builds the accumulate/apply Execs and returns a Stepper.
// bk is the backend wrapper (the runtime stays behind the anti-corruption seam;
// .Compute() is the only runtime reference and is passed straight to model.NewExec).
// gradClip is the global-norm clip threshold (lm-100m uses 1.0). store holds all variables.
// weightDecay is the decoupled AdamW weight decay applied to rank>=2 params ONLY (norm
// scales and other 1-D params are excluded, matching the baseline's param-group split).
// The optimizer itself must be built WITHOUT WeightDecay (GoMLX would apply it to every
// variable). opt must implement gomlxtrain.OptimizeWithGradients (both Adam and SGD do).
func NewStepper(
	bk *gomlxbackend.Backend,
	store *model.Store,
	mcfg lmodel.Config,
	positions []int,
	gradAccum int,
	gradClip float64,
	weightDecay float64,
	opt optimizer.Interface,
	computeDT dtypes.DType,
	checkpoint bool,
) *Stepper {
	owg, ok := opt.(gomlxtrain.OptimizeWithGradients)
	if !ok {
		panic("optimizer does not implement OptimizeWithGradients; use Adam or SGD")
	}

	// accOnlyFn: accumulate gradients, return loss only (no apply).
	accOnlyFn := func(scope *model.Scope, x, y *g.Node) *g.Node {
		gr := x.Graph()
		lossNode := modelLoss(scope, gr, mcfg, x, y, computeDT, positions, checkpoint)
		grads := scope.BuildTrainableVariablesGradientsGraph(lossNode)
		varIdx := 0
		for v := range scope.IterVariables() {
			if !v.Trainable || !v.InUseByGraph(gr) {
				continue
			}
			acc := accumulatorVar(scope, v)
			grads[varIdx] = g.Add(acc.NodeValue(gr), grads[varIdx])
			acc.SetNodeValue(grads[varIdx])
			varIdx++
		}
		return lossNode
	}

	// accApplyFn: accumulate + apply; returns (loss, gradNorm) for the metric.
	accApplyFn := func(scope *model.Scope, x, y *g.Node) (*g.Node, *g.Node) {
		gr := x.Graph()
		lossNode := modelLoss(scope, gr, mcfg, x, y, computeDT, positions, checkpoint)
		grads := scope.BuildTrainableVariablesGradientsGraph(lossNode)
		varIdx := 0
		for v := range scope.IterVariables() {
			if !v.Trainable || !v.InUseByGraph(gr) {
				continue
			}
			acc := accumulatorVar(scope, v)
			grads[varIdx] = g.Add(acc.NodeValue(gr), grads[varIdx])
			acc.SetNodeValue(grads[varIdx])
			varIdx++
		}

		// Mean over GradAccum micro-batches.
		ratio := 1.0 / float64(gradAccum)
		for ii := range grads {
			grads[ii] = g.MulScalar(grads[ii], ratio)
		}

		// Global-norm clip, capturing the pre-clip norm for the metric.
		clipped, normNode := clipByGlobalNormWithNorm(grads, gradClip)
		owg.UpdateGraphWithGradients(scope, clipped, dtypes.Float32)

		// Decoupled AdamW weight decay on rank>=2 params only (norm scales / 1-D
		// params excluded, matching the baseline). Applied to the post-Adam weights:
		// w *= (1 - lr*wd). The per-step lr is read from the optimizer's learning_rate
		// variable (created by UpdateGraphWithGradients above, so it exists here). The
		// lr*lr*wd difference from decaying w_old first is O(lr^2) and negligible.
		if weightDecay > 0 {
			lrNode := optimizer.LearningRateVar(scope, dtypes.Float32, 0).NodeValue(gr)
			factor := g.Sub(g.Const(gr, float32(1)), g.MulScalar(lrNode, weightDecay))
			for v := range scope.IterVariables() {
				if !v.Trainable || !v.InUseByGraph(gr) || v.Shape().Rank() < 2 {
					continue
				}
				v.SetNodeValue(g.Mul(v.NodeValue(gr), factor))
			}
		}

		// Zero all accumulators.
		for v := range scope.IterVariables() {
			if !v.Trainable || !v.InUseByGraph(gr) {
				continue
			}
			acc := accumulatorVar(scope, v)
			acc.SetNodeValue(g.Zeros(gr, acc.Shape()))
		}

		return lossNode, normNode
	}

	be := bk.Compute() // sole runtime reference; handed straight to model.NewExec
	accOnly := model.MustNewExec(be, store, accOnlyFn)
	accApply := model.MustNewExec(be, store, accApplyFn)

	return &Stepper{
		accOnly:   accOnly,
		accApply:  accApply,
		store:     store,
		gradAccum: gradAccum,
	}
}

// SetLR updates the learning-rate variable for the next optimizer step. It locates
// the variable lazily (the apply graph builds on first Step; the lr var doesn't exist
// until then) and caches the handle. Safe to call from step 1 onward; step 0 uses
// the initial lr baked into the Adam config. Returns an error only if the lr var
// cannot be found after the first step has run.
func (s *Stepper) SetLR(lr float64) error {
	if s.lrVar == nil {
		for v := range s.store.IterVariables() {
			if v.Name() == optimizer.ParamLearningRate {
				s.lrVar = v
				break
			}
		}
		if s.lrVar == nil {
			// Graph hasn't been built yet (before first step); no-op.
			return nil
		}
	}
	return s.lrVar.SetValue(tensors.FromScalar(float32(lr)))
}

// Step runs one optimizer step: it pulls GradAccum micro-batches from next, accumulating
// gradients over the first GradAccum-1 and accumulate-then-applying on the last. Each call
// to next must return a distinct micro-batch — x is [B,T] int32 token ids, y is [B,T,1]
// int32 next-token ids. Returning the same batch every call is valid (e.g. to overfit a
// fixed batch) but then accumulation averages identical gradients; a real effective-batch
// benefit requires next to yield different data each call.
// Returns the apply micro-batch loss and the pre-clip global gradient norm.
func (s *Stepper) Step(next func() (x, y *tensors.Tensor)) (loss, gradNorm float64) {
	for i := 0; i < s.gradAccum-1; i++ {
		x, y := next()
		s.accOnly.MustCall(x, y)
	}
	x, y := next()
	out := s.accApply.MustCall(x, y)
	loss = shapes.ConvertTo[float64](out[0].Value())
	if len(out) > 1 && out[1] != nil {
		gradNorm = shapes.ConvertTo[float64](out[1].Value())
	}
	return loss, gradNorm
}
