package train

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
	"github.com/gomlx/gomlx/ml/train/optimizer"
)

// saveCheckpoint saves the store to dir atomically. GoMLX checkpoint.Save() writes
// the .json and .bin files in place without an atomic rename, so we save into a
// temporary sibling directory first, then rename the pair of files into dir. A crash
// mid-rename leaves the temp directory behind (it is cleaned on the next call) but
// never corrupts the previous checkpoint files in dir.
func saveCheckpoint(store *model.Store, dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("checkpoint: mkdir %s: %w", dir, err)
	}

	// Write into a temp sibling directory.
	parent := filepath.Dir(dir)
	tmp, err := os.MkdirTemp(parent, ".ckpt-tmp-")
	if err != nil {
		return fmt.Errorf("checkpoint: mkdirtemp: %w", err)
	}
	defer func() { _ = os.RemoveAll(tmp) }() // clean up on any path

	h, err := checkpoint.Build(store).Dir(tmp).Keep(-1).Done()
	if err != nil {
		return fmt.Errorf("checkpoint: build in tmp: %w", err)
	}
	if err := h.Save(); err != nil {
		return fmt.Errorf("checkpoint: save in tmp: %w", err)
	}

	// Move the pair of files (.json + .bin) from tmp into dir.
	entries, err := os.ReadDir(tmp)
	if err != nil {
		return fmt.Errorf("checkpoint: readdir tmp: %w", err)
	}
	for _, e := range entries {
		src := filepath.Join(tmp, e.Name())
		dst := filepath.Join(dir, e.Name())
		if err := os.Rename(src, dst); err != nil {
			// Cross-device rename (tmp and dir on different filesystems) — copy+remove.
			if copyErr := copyFile(src, dst); copyErr != nil {
				return fmt.Errorf("checkpoint: rename %s: %w (copy also failed: %v)", e.Name(), err, copyErr)
			}
			_ = os.Remove(src)
		}
	}
	return nil
}

// copyFile copies src to dst, creating dst if needed.
func copyFile(src, dst string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	return os.WriteFile(dst, data, 0o644)
}

// resumeIfPresent builds a checkpoint handler for dir (auto-loading the most recent
// checkpoint if one exists) and returns the restored step. Returns (0, nil) if dir is
// empty or does not exist — a fresh start.
func resumeIfPresent(store *model.Store, dir string) (int64, *checkpoint.Handler, error) {
	if _, err := os.Stat(dir); os.IsNotExist(err) {
		return 0, nil, nil
	}
	h, err := checkpoint.Build(store).Dir(dir).Keep(1).Done()
	if err != nil {
		return 0, nil, fmt.Errorf("checkpoint: resume %s: %w", dir, err)
	}
	step := optimizer.GetGlobalStep(store)
	return step, h, nil
}

