package train

import (
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/gomlx/compute/dtypes"
	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"

	gomlxbackend "github.com/guygrigsby/lmkit-go/backend/gomlx"
	lmodel "github.com/guygrigsby/lmkit-go/model"
)

// TestForwardOnly2048 times JUST the forward pass (no backward, no optimizer, no LM head) to
// isolate how much of the training step is forward compute. Compared against the full step and
// the NOLMHEAD ablation, this localizes the broad throughput gap. Gated on GOMLX_FWDONLY=true
// and a cuda backend.
func TestForwardOnly2048(t *testing.T) {
	if os.Getenv("GOMLX_FWDONLY") != "true" {
		t.Skip("set GOMLX_FWDONLY=true to run the forward-only benchmark")
	}
	bk, err := gomlxbackend.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	if !strings.Contains(strings.ToLower(bk.Compute().Name()+" "+bk.Compute().Description()), "cuda") {
		t.Skipf("needs a cuda backend; got %q", bk.Compute().Name())
	}
	lmodel.UseFlashAttention = os.Getenv("GOMLX_STEP_FLASH") != "false"

	B, T := 2, 2048
	if v := os.Getenv("GOMLX_B"); v != "" {
		if n, e := strconv.Atoi(v); e == nil && n > 0 {
			B = n
		}
	}
	cfg := lmodel.Config{
		Hidden: 768, NLayers: 12, NHeads: 12, NKVHeads: 4, HeadDim: 64,
		FFNHidden: 2048, VocabSize: 32000, RopeBase: 10000, RMSEps: 1e-5,
	}
	positions := make([]int, T)
	for i := range positions {
		positions[i] = i
	}
	inputs := tensors.FromFlatDataAndDimensions(make([]int32, B*T), B, T) // token ids [B,T]

	store := model.NewStore()
	store.RootScope().SetParam(model.ParamInitialSeed, int64(1337))

	exec := model.MustNewExec(bk.Compute(), store, func(scope *model.Scope, x *g.Node) *g.Node {
		gr := x.Graph()
		w := castWeights(ModelVars(scope, gr, cfg), dtypes.BFloat16)
		return g.ReduceAllMean(lmodel.ForwardHidden(cfg, w, x, positions)) // forward to hidden, no LM head
	})
	exec.MustCall(inputs) // compile + warmup
	const iters = 5
	start := time.Now()
	for i := 0; i < iters; i++ {
		exec.MustCall(inputs)
	}
	per := time.Since(start) / iters
	t.Logf("lm-100m FORWARD-ONLY B=%d T=%d flash=%v: %v/fwd, %.0f tok/s",
		B, T, lmodel.UseFlashAttention, per, float64(B*T)/per.Seconds())
}
