package train_test

import (
	"testing"

	"github.com/gomlx/gomlx/core/tensors"

	lmodel "github.com/guygrigsby/lmkit-go/model"
	"github.com/guygrigsby/lmkit-go/train"
)

// tinyCfg + a fixed next-token batch shared by the train tests.
func tinyCfg() lmodel.Config {
	return lmodel.Config{
		Hidden: 16, NLayers: 2, NHeads: 4, NKVHeads: 2, HeadDim: 4,
		FFNHidden: 32, VocabSize: 32, RopeBase: 10000, RMSEps: 1e-5,
	}
}

// fixedBatch builds one deterministic batch: a base sequence of B*(T+1) ids, with
// inputs = seq[:, :T] and labels = seq[:, 1:] (next token), labels shaped [B,T,1].
func fixedBatch(b, t, vocab int) (inputs, labels *tensors.Tensor) {
	in := make([]int32, b*t)
	lab := make([]int32, b*t)
	for bi := 0; bi < b; bi++ {
		seq := make([]int32, t+1)
		for k := range seq {
			seq[k] = int32((bi*(t+1)+k)*13+7) % int32(vocab)
		}
		for ti := 0; ti < t; ti++ {
			in[bi*t+ti] = seq[ti]
			lab[bi*t+ti] = seq[ti+1]
		}
	}
	return tensors.FromFlatDataAndDimensions(in, b, t),
		tensors.FromFlatDataAndDimensions(lab, b, t, 1)
}

func TestOverfitRuns(t *testing.T) {
	cfg := tinyCfg()
	inputs, labels := fixedBatch(2, 8, cfg.VocabSize)
	start, end, err := train.Overfit(cfg, inputs, labels, 50, 3e-3, 0.0, 42)
	if err != nil {
		t.Fatalf("Overfit: %v", err)
	}
	t.Logf("loss start=%.4f end=%.4f", start, end)
	if start <= 0 || end <= 0 {
		t.Errorf("losses should be positive, got start=%v end=%v", start, end)
	}
	if end >= start {
		t.Errorf("loss did not decrease over 50 steps: start=%v end=%v", start, end)
	}
}
