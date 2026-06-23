package train_test

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train/optimizer"

	gomlxbackend "github.com/guygrigsby/lmkit-go/backend/gomlx"
	lmodel "github.com/guygrigsby/lmkit-go/model"
	"github.com/guygrigsby/lmkit-go/train"
)

func lm100mCfg() lmodel.Config {
	return lmodel.Config{
		Hidden: 768, NLayers: 12, NHeads: 12, NKVHeads: 4, HeadDim: 64,
		FFNHidden: 2048, VocabSize: 32000, RopeBase: 10000, RMSEps: 1e-5,
	}
}

// TestFlashFullStep2048 measures one lm-100m training step (forward + backward + optimizer) at the
// real shape on the GPU, in bf16 with no gradient checkpointing (the configuration that OOM'd the
// decomposed path). It runs one attention variant: set GOMLX_STEP_FLASH=false for the decomposed
// baseline (default is flash), and run the two in separate processes so an OOM in one is isolated.
// Requires a cuda backend.
func TestFlashFullStep2048(t *testing.T) {
	bk, err := gomlxbackend.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	name := strings.ToLower(bk.Compute().Name() + " " + bk.Compute().Description())
	if !strings.Contains(name, "cuda") {
		t.Skipf("full-model 2048 measurement needs a cuda backend; got %q", bk.Compute().Name())
	}
	lmodel.UseFlashAttention = os.Getenv("GOMLX_STEP_FLASH") != "false"

	const B, T = 2, 2048
	cfg := lm100mCfg()
	inputs, labels := fixedBatch(B, T, cfg.VocabSize)
	batch := func() (*tensors.Tensor, *tensors.Tensor) { return inputs, labels }
	positions := make([]int, T)
	for i := range positions {
		positions[i] = i
	}

	store := model.NewStore()
	store.RootScope().SetParam(model.ParamInitialSeed, int64(1337))
	opt := optimizer.Adam().LearningRate(1e-3).Done()

	var per time.Duration
	fit := func() (ok bool) {
		defer func() {
			if r := recover(); r != nil {
				ok = false
			}
		}()
		s := train.NewStepper(bk, store, cfg, positions, 1, 1.0, 0.0, opt, dtypes.BFloat16, false)
		s.Step(batch) // compile + warmup
		const iters = 5
		start := time.Now()
		for i := 0; i < iters; i++ {
			s.Step(batch)
		}
		per = time.Since(start) / iters
		return true
	}()

	if fit {
		t.Logf("lm-100m step (fwd+bwd+opt) B=%d T=%d flash=%v: %v", B, T, lmodel.UseFlashAttention, per)
	} else {
		t.Fatalf("lm-100m step (B=%d T=%d flash=%v) did NOT fit on this GPU", B, T, lmodel.UseFlashAttention)
	}
}
