package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/guygrigsby/lmkit-go/data"
	"github.com/guygrigsby/lmkit-go/model"
	"github.com/guygrigsby/lmkit-go/train"
)

// trainCmd implements `lmkit train`: load a train + model config, build train/val
// loaders over the shards, and run the loop. It is generic (no lm-100m constants);
// the example package supplies the configs. Returns the process exit code.
func trainCmd(args []string) int {
	fs := flag.NewFlagSet("train", flag.ExitOnError)
	configPath := fs.String("config", "", "path to the train config JSON")
	modelPath := fs.String("model", "", "path to the model config JSON")
	dataDir := fs.String("data", "", "shard directory (overrides the config's data_dir)")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *configPath == "" || *modelPath == "" {
		fmt.Fprintln(os.Stderr, "usage: lmkit train --config <train.json> --model <model.json> [--data <dir>]")
		return 2
	}

	cfg, err := train.LoadConfig(*configPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "FAIL: %v\n", err)
		return 1
	}
	mcfg, err := model.LoadConfig(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "FAIL: %v\n", err)
		return 1
	}

	dir := cfg.DataDir
	if *dataDir != "" {
		dir = *dataDir
	}
	trainLoader, err := newLoader(dir, "train_*.bin", mcfg.Block, cfg.BatchSize, cfg.Seed)
	if err != nil {
		fmt.Fprintf(os.Stderr, "FAIL: %v\n", err)
		return 1
	}
	defer trainLoader.Close()
	valLoader, err := newLoader(dir, "val_*.bin", mcfg.Block, cfg.BatchSize, cfg.Seed+1)
	if err != nil {
		fmt.Fprintf(os.Stderr, "FAIL: %v\n", err)
		return 1
	}
	defer valLoader.Close()

	code, err := train.Run(cfg, mcfg, trainLoader, valLoader)
	if err != nil {
		fmt.Fprintf(os.Stderr, "FAIL: %v\n", err)
	}
	return code
}

// newLoader globs dir/pattern and builds a data.Loader over the matched shards.
func newLoader(dir, pattern string, block, batch int, seed int64) (*data.Loader, error) {
	shards, err := filepath.Glob(filepath.Join(dir, pattern))
	if err != nil {
		return nil, fmt.Errorf("glob %q: %w", pattern, err)
	}
	if len(shards) == 0 {
		return nil, fmt.Errorf("no shards matching %q in %q", pattern, dir)
	}
	sort.Strings(shards) // deterministic shard order
	return data.New(data.Config{
		Shards:    shards,
		BlockSize: block,
		BatchSize: batch,
		Seed:      seed,
	})
}
