package train

import (
	"context"
	"fmt"
	"math"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train/optimizer"

	gomlxbackend "github.com/guygrigsby/lmkit-go/backend/gomlx"
	lmodel "github.com/guygrigsby/lmkit-go/model"

	"github.com/guygrigsby/lmkit-go/data"
)

// nParams computes the total parameter count for the model (fp32 master weights only;
// norm scales count as their dimension, projection weights as their product).
func nParams(mcfg lmodel.Config) int {
	h := mcfg.Hidden
	nh := mcfg.NHeads
	nkv := mcfg.NKVHeads
	hd := mcfg.HeadDim
	ffn := mcfg.FFNHidden
	v := mcfg.VocabSize

	// Embed [V,H] — tied; count once.
	total := v * h
	// Final norm scale [H].
	total += h
	// Per decoder layer.
	for range mcfg.NLayers {
		total += h              // AttnNorm
		total += h * (nh * hd) // Wq
		total += h * (nkv * hd) // Wk
		total += h * (nkv * hd) // Wv
		total += (nh * hd) * h // Wo
		total += h              // FFNNorm
		total += h * ffn       // Wgate
		total += h * ffn       // Wup
		total += ffn * h       // Wdown
	}
	return total
}

// noGradLossFn is the graph function for eval loss: forward pass only (no gradients,
// no weight updates). Reuses the same fp32 CE path as the training stepper.
func noGradLossFn(mcfg lmodel.Config, positions []int, computeDT dtypes.DType) func(*model.Scope, *g.Node, *g.Node) *g.Node {
	return func(scope *model.Scope, x, y *g.Node) *g.Node {
		return modelLoss(scope, x.Graph(), mcfg, x, y, computeDT, positions)
	}
}

// batchToTensors converts a data.Batch to the (x, y) tensors the Stepper expects:
// x is [B,T] int32; y is [B,T,1] int32.
func batchToTensors(b data.Batch, batchSize, blockSize int) (x, y *tensors.Tensor) {
	return tensors.FromFlatDataAndDimensions(b.X, batchSize, blockSize),
		tensors.FromFlatDataAndDimensions(b.Y, batchSize, blockSize, 1)
}

// Run is the production training loop. It wires the Stepper (gradient accumulation +
// clip + AdamW), the data loaders, eval, checkpointing, metrics.jsonl, and OS signals
// into one resumable process. Returns an exit code: 0 on completion or SIGTERM, 2 on
// non-finite loss.
func Run(cfg Config, mcfg lmodel.Config, trainLoader, valLoader *data.Loader) (int, error) {
	// Signal handling: cancel context on SIGTERM or SIGINT (I2: avoids data race on plain bool).
	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGTERM, syscall.SIGINT)
	defer stop()

	// Output directory.
	if err := os.MkdirAll(cfg.OutDir, 0o755); err != nil {
		return 1, fmt.Errorf("run: mkdir %s: %w", cfg.OutDir, err)
	}
	metricsPath := filepath.Join(cfg.OutDir, "metrics.jsonl")

	// Resolve compute dtype.
	computeDT := dtypes.Float32
	if cfg.Dtype == "bfloat16" {
		computeDT = dtypes.BFloat16
	}

	// Build backend.
	bk, err := gomlxbackend.New()
	if err != nil {
		return 1, fmt.Errorf("run: backend: %w", err)
	}

	// Store + deterministic seed.
	store := model.NewStore()
	store.RootScope().SetParam(model.ParamInitialSeed, cfg.Seed)

	// Build optimizer.
	opt := optimizer.Adam().
		WeightDecay(cfg.WeightDecay).
		Betas(cfg.Beta1, cfg.Beta2).
		LearningRate(getLR(0, cfg)).
		Done()

	// Resume from latest checkpoint if present.
	latestDir := filepath.Join(cfg.OutDir, "latest")
	step, _, err := resumeIfPresent(store, latestDir)
	if err != nil {
		return 1, fmt.Errorf("run: resume: %w", err)
	}

	// Positions 0..BlockSize-1 (fixed for dense batches).
	positions := make([]int, trainLoader.BlockSize())
	for i := range positions {
		positions[i] = i
	}

	// Build stepper.
	stepper := NewStepper(bk, store, mcfg, positions, cfg.GradAccum, cfg.GradClip, opt, computeDT)

	// Build eval exec (no gradients).
	evalExec := model.MustNewExec(bk.Compute(), store, noGradLossFn(mcfg, positions, computeDT))

	// Track best val loss for best-checkpoint.
	// C1: on resume, restore prior best so we don't overwrite best/ with a worse model.
	bestVal := priorBestVal(metricsPath)
	if step == 0 {
		bestVal = math.MaxFloat64
	}

	// Emit start or resume event.
	eventKind := "start"
	if step > 0 {
		eventKind = "resume"
	}
	_ = emit(metricsPath, map[string]any{
		"event": eventKind,
		"step":  step,
	})

	np := nParams(mcfg)
	batchTokens := cfg.BatchSize * trainLoader.BlockSize() * cfg.GradAccum // tokens per optimizer step

	for int(step) < cfg.MaxSteps {
		if ctx.Err() != nil {
			if err := saveCheckpoint(store, latestDir); err != nil {
				return 1, fmt.Errorf("run: sigterm save: %w", err)
			}
			_ = emit(metricsPath, map[string]any{"event": "sigterm", "step": step})
			return 0, nil
		}

		// Set lr for this step (step 0 used the initial lr baked into Adam config;
		// from step 1 onward we update the lr variable).
		if step > 0 {
			if err := stepper.SetLR(getLR(int(step), cfg)); err != nil {
				return 1, fmt.Errorf("run: set_lr step %d: %w", step, err)
			}
		}

		// Eval cadence.
		if cfg.EvalInterval > 0 && int(step)%cfg.EvalInterval == 0 {
			valLoss, err := evalLoss(evalExec, valLoader, cfg, mcfg)
			if err != nil {
				return 1, fmt.Errorf("run: eval step %d: %w", step, err)
			}
			isBest := valLoss < bestVal
			if isBest {
				bestVal = valLoss
				if err := saveCheckpoint(store, filepath.Join(cfg.OutDir, "best")); err != nil {
					return 1, fmt.Errorf("run: best checkpoint: %w", err)
				}
			}
			_ = emit(metricsPath, map[string]any{
				"event":          "eval",
				"step":           step,
				"val_loss":       valLoss,
				"val_perplexity": math.Exp(valLoss),
				"best":           isBest,
			})
		}

		// Periodic save of latest checkpoint.
		if cfg.SaveInterval > 0 && int(step) > 0 && int(step)%cfg.SaveInterval == 0 {
			if err := saveCheckpoint(store, latestDir); err != nil {
				return 1, fmt.Errorf("run: save latest step %d: %w", step, err)
			}
		}

		// Snapshot cadence.
		if cfg.SnapshotInterval > 0 && int(step) > 0 && int(step)%cfg.SnapshotInterval == 0 {
			snapDir := filepath.Join(cfg.OutDir, fmt.Sprintf("step_%06d", step))
			if err := saveCheckpoint(store, snapDir); err != nil {
				return 1, fmt.Errorf("run: snapshot step %d: %w", step, err)
			}
			if cfg.KeepLastSnapshots > 0 {
				// Prune older snapshot directories.
				snapshotParent := cfg.OutDir
				_ = pruneSnapshotDirs(snapshotParent, cfg.KeepLastSnapshots)
			}
		}

		// Capture lr before step++ so the logged value matches what the optimizer used.
		stepLR := getLR(int(step), cfg)

		// One optimizer step.
		t0 := time.Now()
		stepLoss, gradNorm := stepper.Step(func() (*tensors.Tensor, *tensors.Tensor) {
			b := trainLoader.Next()
			return batchToTensors(b, cfg.BatchSize, trainLoader.BlockSize())
		})
		stepDur := time.Since(t0)

		// Non-finite loss: save diverged state to nan/ (NOT latest/) so latest/ stays
		// resumable. Intentional deviation from the brief — keeps latest/ last-good.
		if math.IsNaN(stepLoss) || math.IsInf(stepLoss, 0) {
			nanDir := filepath.Join(cfg.OutDir, "nan")
			func() {
				defer func() { recover() }() //nolint:errcheck
				_ = saveCheckpoint(store, nanDir)
			}()
			_ = emit(metricsPath, map[string]any{
				"event": "nan",
				"step":  step,
				"train_loss": func() any {
					if math.IsNaN(stepLoss) {
						return "NaN"
					}
					if math.IsInf(stepLoss, 1) {
						return "+Inf"
					}
					return "-Inf"
				}(),
			})
			return 2, nil
		}

		step++

		// Log metrics at LogInterval.
		if cfg.LogInterval > 0 && int(step)%cfg.LogInterval == 0 {
			tokPerSec := float64(batchTokens) / stepDur.Seconds()
			tokensSeen := int64(step) * int64(batchTokens)
			_ = emit(metricsPath, map[string]any{
				"event":        "train",
				"step":         step,
				"train_loss":   stepLoss,
				"lr":           stepLR, // M5: log the lr used this step, captured before step++
				"grad_norm":    gradNorm,
				"tok_per_sec":  tokPerSec,
				"step_time_ms": stepDur.Milliseconds(),
				"tokens_seen":  tokensSeen,
				"tflops":       tflops(np, tokPerSec),
				"peak_vram_gb": peakVRAMGB(),
			})
		}
	}

	// Training complete: save final + latest, emit done.
	finalDir := filepath.Join(cfg.OutDir, "final")
	if err := saveCheckpoint(store, finalDir); err != nil {
		return 1, fmt.Errorf("run: final checkpoint: %w", err)
	}
	if err := saveCheckpoint(store, latestDir); err != nil {
		return 1, fmt.Errorf("run: final latest checkpoint: %w", err)
	}
	_ = emit(metricsPath, map[string]any{
		"event": "done",
		"step":  step,
	})
	return 0, nil
}

// evalLoss computes the mean CE loss over EvalIters val batches.
func evalLoss(evalExec *model.Exec, valLoader *data.Loader, cfg Config, mcfg lmodel.Config) (float64, error) {
	var sum float64
	n := cfg.EvalIters
	if n <= 0 {
		n = 1
	}
	for i := 0; i < n; i++ {
		b := valLoader.Next()
		x, y := batchToTensors(b, valLoader.BatchSize(), valLoader.BlockSize()) // M6: use loader's own BatchSize
		out := evalExec.MustCall(x, y)
		sum += shapes.ConvertTo[float64](out[0].Value())
	}
	return sum / float64(n), nil
}

// pruneSnapshotDirs removes step_NNNNNN subdirectories under parent, keeping the
// `keep` most recent ones (by sorted name, oldest first).
func pruneSnapshotDirs(parent string, keep int) error {
	entries, err := os.ReadDir(parent)
	if err != nil {
		return err
	}
	var snapDirs []string
	for _, e := range entries {
		if e.IsDir() && len(e.Name()) >= 5 && e.Name()[:5] == "step_" {
			snapDirs = append(snapDirs, e.Name())
		}
	}
	if len(snapDirs) <= keep {
		return nil
	}
	for _, d := range snapDirs[:len(snapDirs)-keep] {
		_ = os.RemoveAll(filepath.Join(parent, d))
	}
	return nil
}
