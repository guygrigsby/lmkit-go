package train

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
)

// Config mirrors the Python lmkit TrainConfig. Loaded from JSON; no lm-100m
// constants live here (those are in examples/lm-100m-en).
type Config struct {
	OutDir            string  `json:"out_dir"`
	DataDir           string  `json:"data_dir"`
	LR                float64 `json:"lr"`
	MinLR             float64 `json:"min_lr"`
	WarmupSteps       int     `json:"warmup_steps"`
	MaxSteps          int     `json:"max_steps"`
	DecayFrac         float64 `json:"decay_frac"`
	BatchSize         int     `json:"batch_size"`
	GradAccum         int     `json:"grad_accum"`
	GradClip          float64 `json:"grad_clip"`
	WeightDecay       float64 `json:"weight_decay"`
	Beta1             float64 `json:"beta1"`
	Beta2             float64 `json:"beta2"`
	Dtype             string  `json:"dtype"` // "bfloat16" | "float32"
	Seed              int64   `json:"seed"`
	EvalInterval      int     `json:"eval_interval"`
	EvalIters         int     `json:"eval_iters"`
	LogInterval       int     `json:"log_interval"`
	SaveInterval      int     `json:"save_interval"`
	SnapshotInterval  int     `json:"snapshot_interval"`
	KeepLastSnapshots int     `json:"keep_last_snapshots"`
}

// LoadConfig reads a JSON train config.
func LoadConfig(path string) (Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return Config{}, err
	}
	var c Config
	if err := json.Unmarshal(b, &c); err != nil {
		return Config{}, fmt.Errorf("train: config %s: %w", path, err)
	}
	return c, nil
}

// getLR is the WSD schedule (exact port of lmkit.training.get_lr): linear warmup →
// constant lr → cosine decay to min_lr over the final decay_frac of max_steps.
func getLR(step int, c Config) float64 {
	if step < c.WarmupSteps {
		return c.LR * float64(step) / math.Max(1, float64(c.WarmupSteps))
	}
	decaySteps := int(c.DecayFrac * float64(c.MaxSteps))
	decayStart := c.MaxSteps - decaySteps
	if step < decayStart {
		return c.LR
	}
	if step >= c.MaxSteps {
		return c.MinLR
	}
	progress := float64(step-decayStart) / math.Max(1, float64(decaySteps))
	coeff := 0.5 * (1.0 + math.Cos(math.Pi*progress))
	return c.MinLR + coeff*(c.LR-c.MinLR)
}

var _ = json.Marshal // keep encoding/json imported if unused elsewhere
