//go:build integration

package lm100men_test

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/guygrigsby/lmkit-go/data"
	"github.com/guygrigsby/lmkit-go/model"
	"github.com/guygrigsby/lmkit-go/train"
)

// packageDir returns the directory of this test file (the example package root),
// so JSON config paths resolve correctly regardless of how the test is invoked.
func packageDir(t *testing.T) string {
	t.Helper()
	// The test binary is run from the module root; configs live next to this file.
	// Use the path relative to go.mod (examples/lm-100m-en/).
	wd, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	return wd
}

// TestShortBF16 runs a short bfloat16 training loop over real shards (env LMKIT_DATA)
// and asserts that the run descends, writes expected events to metrics.jsonl, saves
// at least one checkpoint directory, and — when running on CUDA — reports non-zero
// peak_vram_gb.
func TestShortBF16(t *testing.T) {
	dataDir := os.Getenv("LMKIT_DATA")
	if dataDir == "" {
		t.Skip("set LMKIT_DATA to run")
	}

	pkgDir := packageDir(t)

	// Load configs from the package directory.
	mcfg, err := model.LoadConfig(filepath.Join(pkgDir, "model.json"))
	if err != nil {
		t.Fatalf("model.LoadConfig: %v", err)
	}

	// Verify model.json loaded non-zero (guard against stale tag drift).
	if mcfg.NLayers != 12 {
		t.Fatalf("model.json: NLayers = %d, want 12", mcfg.NLayers)
	}
	if mcfg.NHeads != 12 {
		t.Fatalf("model.json: NHeads = %d, want 12", mcfg.NHeads)
	}
	if mcfg.FFNHidden != 2048 {
		t.Fatalf("model.json: FFNHidden = %d, want 2048", mcfg.FFNHidden)
	}
	if mcfg.VocabSize != 32000 {
		t.Fatalf("model.json: VocabSize = %d, want 32000", mcfg.VocabSize)
	}
	if mcfg.SeqLen != 2048 {
		t.Fatalf("model.json: SeqLen = %d, want 2048", mcfg.SeqLen)
	}

	cfg, err := train.LoadConfig(filepath.Join(pkgDir, "config.json"))
	if err != nil {
		t.Fatalf("train.LoadConfig: %v", err)
	}

	// Override for a short smoke run.
	cfg.MaxSteps = 200
	cfg.EvalInterval = 100
	cfg.EvalIters = 10
	cfg.SaveInterval = 100
	cfg.LogInterval = 20
	cfg.OutDir = t.TempDir()

	// Build loaders: glob train_*.bin and val_*.bin.
	trainShards, err := filepath.Glob(filepath.Join(dataDir, "train_*.bin"))
	if err != nil {
		t.Fatalf("glob train shards: %v", err)
	}
	if len(trainShards) == 0 {
		t.Fatalf("no train_*.bin shards in %q", dataDir)
	}
	valShards, err := filepath.Glob(filepath.Join(dataDir, "val_*.bin"))
	if err != nil {
		t.Fatalf("glob val shards: %v", err)
	}
	if len(valShards) == 0 {
		t.Fatalf("no val_*.bin shards in %q", dataDir)
	}

	trainLoader, err := data.New(data.Config{
		Shards:    trainShards,
		BlockSize: mcfg.SeqLen,
		BatchSize: cfg.BatchSize,
		Seed:      cfg.Seed,
	})
	if err != nil {
		t.Fatalf("data.New (train): %v", err)
	}
	defer trainLoader.Close()

	valLoader, err := data.New(data.Config{
		Shards:    valShards,
		BlockSize: mcfg.SeqLen,
		BatchSize: cfg.BatchSize,
		Seed:      cfg.Seed + 1,
	})
	if err != nil {
		t.Fatalf("data.New (val): %v", err)
	}
	defer valLoader.Close()

	// Run.
	code, err := train.Run(cfg, mcfg, trainLoader, valLoader)
	if err != nil {
		t.Fatalf("train.Run: %v", err)
	}
	if code != 0 {
		t.Fatalf("train.Run exit code = %d, want 0", code)
	}

	// Assert metrics.jsonl exists and contains expected events.
	metricsPath := filepath.Join(cfg.OutDir, "metrics.jsonl")
	if _, err := os.Stat(metricsPath); err != nil {
		t.Fatalf("metrics.jsonl missing: %v", err)
	}

	events, err := parseMetricsJSONL(metricsPath)
	if err != nil {
		t.Fatalf("parse metrics.jsonl: %v", err)
	}

	checkEventPresent(t, events, "start")
	checkEventPresent(t, events, "eval")
	checkEventPresent(t, events, "train")
	checkEventPresent(t, events, "done")

	// Loss should descend: first train_loss > last train_loss.
	trainEvents := filterEvents(events, "train")
	if len(trainEvents) >= 2 {
		first := trainEvents[0]["train_loss"]
		last := trainEvents[len(trainEvents)-1]["train_loss"]
		firstF, firstOK := toFloat64(first)
		lastF, lastOK := toFloat64(last)
		if firstOK && lastOK && lastF >= firstF {
			t.Errorf("train_loss did not descend: first=%v last=%v", firstF, lastF)
		}
	}

	// A checkpoint directory (latest/) should exist.
	latestDir := filepath.Join(cfg.OutDir, "latest")
	if fi, err := os.Stat(latestDir); err != nil || !fi.IsDir() {
		t.Errorf("checkpoint dir %q not written", latestDir)
	}

	// On CUDA: peak_vram_gb must be > 0.
	backendEnv := os.Getenv("GOMLX_BACKEND")
	if strings.Contains(backendEnv, "cuda") {
		for _, ev := range trainEvents {
			if v, ok := ev["peak_vram_gb"]; ok {
				if gb, ok := toFloat64(v); ok && gb > 0 {
					break
				}
				t.Errorf("peak_vram_gb not positive on CUDA: %v", v)
				break
			}
		}
	}
}

// parseMetricsJSONL reads all JSON objects from a .jsonl file.
func parseMetricsJSONL(path string) ([]map[string]any, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var out []map[string]any
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var m map[string]any
		if err := json.Unmarshal([]byte(line), &m); err != nil {
			return nil, err
		}
		out = append(out, m)
	}
	return out, sc.Err()
}

func filterEvents(events []map[string]any, kind string) []map[string]any {
	var out []map[string]any
	for _, e := range events {
		if e["event"] == kind {
			out = append(out, e)
		}
	}
	return out
}

func checkEventPresent(t *testing.T, events []map[string]any, kind string) {
	t.Helper()
	for _, e := range events {
		if e["event"] == kind {
			return
		}
	}
	t.Errorf("metrics.jsonl: no %q event", kind)
}

func toFloat64(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case json.Number:
		f, err := x.Float64()
		return f, err == nil
	}
	return 0, false
}
