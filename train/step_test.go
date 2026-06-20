package train_test

import (
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train/optimizer"

	gomlxbackend "github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/train"
)

// TestStepDescends is the Task-3 gate: the production step path (grad-accum + clip +
// AdamW) descends the loss on a tiny fixed batch. Seed-independent — correct gradient
// wiring converges for any init; a broken loop (wrong order, dead grads, clip bug)
// will not reach endLoss < 0.3*startLoss in 200 steps.
func TestStepDescends(t *testing.T) {
	cfg := tinyCfg()
	inputs, labels := fixedBatch(2, 8, cfg.VocabSize)

	bk, err := gomlxbackend.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	be := bk.Compute()

	store := model.NewStore()

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
	s := train.NewStepper(be, store, cfg, positions, gradAccum, opt, dtypes.Float32)

	var startLoss, endLoss float64
	for i := 0; i < steps; i++ {
		loss := s.Step(inputs, labels)
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
