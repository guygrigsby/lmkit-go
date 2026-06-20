package train_test

import (
	"testing"

	"github.com/guygrigsby/lmkit-go/train"
)

// TestOverfitBatch is the Milestone-4 gate: a working loop drives the loss on one
// tiny fixed batch to ~0 (memorization). The assertion is seed-independent — it
// checks the loss fell substantially AND reached near zero, both of which a broken
// loop (no gradient flow, wrong loss, dead optimizer) cannot achieve, and which a
// correct loop achieves for any initialization. The absolute threshold below is set
// from the observed converged loss with margin (NOT tuned to a lucky seed/step).
func TestOverfitBatch(t *testing.T) {
	cfg := tinyCfg()
	inputs, labels := fixedBatch(2, 8, cfg.VocabSize)
	start, end, err := train.Overfit(cfg, inputs, labels, 2000, 3e-3, 0.0, 42)
	if err != nil {
		t.Fatalf("Overfit: %v", err)
	}
	t.Logf("loss start=%.4f end=%.6f", start, end)
	// Substantial fall: a correct loop reduces a memorizable batch far below init.
	if end >= 0.1*start {
		t.Errorf("loss did not fall substantially: end=%v, start=%v (want end < 0.1*start)", end, start)
	}
	// Near zero: the batch is memorized. Threshold set from observed convergence
	// (seed=42: end~0.0003, seed=7: end~0.0002) with ~100x margin.
	if end >= 0.05 {
		t.Errorf("batch not memorized: end loss=%v (want < 0.05)", end)
	}
}
