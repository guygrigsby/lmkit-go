package train_test

import (
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train/optimizer"

	gomlxbackend "github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/train"
)

// TestStepDescends is the Task-3 gate: the production step path (grad-accum + clip +
// AdamW) descends the loss on a tiny fixed batch. The init seed is fixed so the gate is
// reproducible; correct gradient wiring converges, while a broken loop (wrong order, dead
// grads, clip bug) will not reach endLoss < 0.3*startLoss in 200 steps.
func TestStepDescends(t *testing.T) {
	cfg := tinyCfg()
	inputs, labels := fixedBatch(2, 8, cfg.VocabSize)
	batch := func() (*tensors.Tensor, *tensors.Tensor) { return inputs, labels } // fixed batch (overfit)

	bk, err := gomlxbackend.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}

	store := model.NewStore()
	store.RootScope().SetParam(model.ParamInitialSeed, int64(1337))

	tt := inputs.Shape().Dimensions[1] // T
	positions := make([]int, tt)
	for i := range positions {
		positions[i] = i
	}

	const (
		gradAccum   = 2
		lr          = 3e-3
		weightDecay = 0.0
		steps       = 200
	)

	opt := optimizer.Adam().WeightDecay(weightDecay).LearningRate(lr).Done()
	s := train.NewStepper(bk, store, cfg, positions, gradAccum, 1.0, 0.0, opt, dtypes.Float32, false)

	var startLoss, endLoss float64
	for i := 0; i < steps; i++ {
		loss, _ := s.Step(batch)
		if i == 0 {
			startLoss = loss
		}
		endLoss = loss
	}

	t.Logf("loss start=%.4f end=%.6f", startLoss, endLoss)
	if endLoss >= 0.3*startLoss {
		t.Errorf("loss did not descend: start=%.4f end=%.4f (want end < 0.3*start)", startLoss, endLoss)
	}
}

type varSnap struct {
	rank int
	data []float32
}

// snapshotTrainables reads every trainable variable's host value, keyed by scope+name.
func snapshotTrainables(t *testing.T, store *model.Store) map[string]varSnap {
	t.Helper()
	out := map[string]varSnap{}
	for v := range store.IterVariables() {
		if !v.Trainable {
			continue
		}
		val, err := v.Value()
		if err != nil {
			t.Fatalf("var %q Value: %v", v.Name(), err)
		}
		out[v.Scope()+"/"+v.Name()] = varSnap{
			rank: v.Shape().Rank(),
			data: tensors.MustCopyFlatData[float32](val),
		}
	}
	return out
}

func sameFloats(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestWeightDecayTwoDimOnly is the C1 gate: decoupled AdamW weight decay must apply to
// rank>=2 params ONLY (matrices/embeddings), never to rank-1 params (norm scales). Two
// runs identical except for weight decay: with the same seed/grad/lr the Adam step is
// identical, so rank-1 vars must stay byte-identical between wd=0 and wd>0, while rank>=2
// vars must differ (shrunk by 1-lr*wd). A regression that decays all params (e.g. via
// Adam's own WeightDecay) would change the rank-1 vars and fail here.
func TestWeightDecayTwoDimOnly(t *testing.T) {
	cfg := tinyCfg()
	inputs, labels := fixedBatch(2, 8, cfg.VocabSize)
	batch := func() (*tensors.Tensor, *tensors.Tensor) { return inputs, labels }

	bk, err := gomlxbackend.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}

	tt := inputs.Shape().Dimensions[1]
	positions := make([]int, tt)
	for i := range positions {
		positions[i] = i
	}

	runOnce := func(wd float64) map[string]varSnap {
		store := model.NewStore()
		store.RootScope().SetParam(model.ParamInitialSeed, int64(1337))
		opt := optimizer.Adam().LearningRate(0.1).Done() // no WeightDecay on Adam
		s := train.NewStepper(bk, store, cfg, positions, 1, 1.0, wd, opt, dtypes.Float32, false)
		s.Step(batch)
		return snapshotTrainables(t, store)
	}

	noWD := runOnce(0.0)
	withWD := runOnce(2.0) // lr*wd = 0.2 -> rank>=2 vars shrink by 0.8

	var checked1D, checked2D int
	for key, a := range noWD {
		b, ok := withWD[key]
		if !ok {
			t.Fatalf("var %q present without WD but missing with WD", key)
		}
		if a.rank >= 2 {
			if sameFloats(a.data, b.data) {
				t.Errorf("rank>=2 var %q was NOT decayed (identical to wd=0)", key)
			}
			checked2D++
		} else {
			if !sameFloats(a.data, b.data) {
				t.Errorf("rank-1 var %q WAS decayed (must be excluded from weight decay)", key)
			}
			checked1D++
		}
	}
	if checked1D == 0 || checked2D == 0 {
		t.Fatalf("ineffective test: rank-1 checked=%d rank>=2 checked=%d (need both > 0)", checked1D, checked2D)
	}
	t.Logf("weight-decay grouping: %d rank-1 vars excluded, %d rank>=2 vars decayed", checked1D, checked2D)
}
