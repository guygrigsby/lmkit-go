package train_test

import (
	"os"
	"strconv"
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

	B, T := 2, 2048 // micro-batch size; GOMLX_B overrides to probe throughput vs underfed-GPU
	if v := os.Getenv("GOMLX_B"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			B = n
		}
	}
	cfg := lm100mCfg()
	if v := os.Getenv("GOMLX_NLAYERS"); v != "" { // probe backward scaling: O(N) vs O(N^2) in depth
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.NLayers = n
		}
	}
	inputs, labels := fixedBatch(B, T, cfg.VocabSize)
	batch := func() (*tensors.Tensor, *tensors.Tensor) { return inputs, labels }
	positions := make([]int, T)
	for i := range positions {
		positions[i] = i
	}

	store := model.NewStore()
	store.RootScope().SetParam(model.ParamInitialSeed, int64(1337))
	opt := optimizer.Adam().LearningRate(1e-3).Done()

	ga := 1 // gradient-accumulation micro-batches per optimizer step; real lm-100m config is 32.
	if v := os.Getenv("GOMLX_GA"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			ga = n
		}
	}
	ckpt := os.Getenv("GOMLX_CKPT") == "true" // per-layer gradient checkpointing (real config: on)

	var per time.Duration
	fit := func() (ok bool) {
		defer func() {
			if r := recover(); r != nil {
				ok = false
			}
		}()
		s := train.NewStepper(bk, store, cfg, positions, ga, 1.0, 0.0, opt, dtypes.BFloat16, ckpt)
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
		tokPerSec := float64(ga*B*T) / per.Seconds()
		t.Logf("lm-100m step B=%d T=%d ga=%d flash=%v ckpt=%v: %v/step, %.0f tok/s (%d tokens/step)",
			B, T, ga, lmodel.UseFlashAttention, ckpt, per, tokPerSec, ga*B*T)
	} else {
		t.Fatalf("lm-100m step (B=%d T=%d flash=%v) did NOT fit on this GPU", B, T, lmodel.UseFlashAttention)
	}
}
