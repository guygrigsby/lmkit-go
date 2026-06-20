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
	s := train.NewStepper(bk, store, cfg, positions, gradAccum, 1.0, opt, dtypes.Float32)

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
