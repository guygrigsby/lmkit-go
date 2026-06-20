package train

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadConfig(t *testing.T) {
	p := filepath.Join(t.TempDir(), "c.json")
	os.WriteFile(p, []byte(`{"lr":4e-4,"min_lr":4e-5,"warmup_steps":1000,"max_steps":200000,"decay_frac":0.0,"batch_size":2,"grad_accum":32,"grad_clip":1.0,"weight_decay":0.1,"beta1":0.9,"beta2":0.95,"dtype":"bfloat16","seed":1337,"eval_interval":2000,"eval_iters":100,"log_interval":20,"save_interval":200,"snapshot_interval":25000,"keep_last_snapshots":3,"out_dir":"ckpt","data_dir":"data"}`), 0o644)
	c, err := LoadConfig(p)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if c.LR != 4e-4 || c.GradAccum != 32 || c.Dtype != "bfloat16" || c.MaxSteps != 200000 {
		t.Errorf("parsed wrong: %+v", c)
	}
}

func TestGetLR_WSD(t *testing.T) {
	c := Config{LR: 4e-4, MinLR: 4e-5, WarmupSteps: 1000, MaxSteps: 200000, DecayFrac: 0.0}
	// warmup: linear from 0
	if got := getLR(500, c); math.Abs(got-2e-4) > 1e-12 {
		t.Errorf("warmup mid = %v want 2e-4", got)
	}
	if got := getLR(0, c); got != 0 {
		t.Errorf("step0 = %v want 0", got)
	}
	// stable (decay_frac=0): constant lr after warmup, forever
	if got := getLR(1000, c); got != 4e-4 {
		t.Errorf("stable start = %v want 4e-4", got)
	}
	if got := getLR(199999, c); got != 4e-4 {
		t.Errorf("stable late = %v want 4e-4 (decay_frac=0)", got)
	}
	// with a decay tail: ends at min_lr
	c2 := c
	c2.DecayFrac = 0.5
	if got := getLR(c2.MaxSteps, c2); math.Abs(got-c2.MinLR) > 1e-12 {
		t.Errorf("decay end = %v want min_lr", got)
	}
}
