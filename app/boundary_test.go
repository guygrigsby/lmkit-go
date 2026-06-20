package app_test

import (
	"os/exec"
	"strings"
	"testing"
)

// TestBackendBoundary enforces ADR-0009: only files under backend/gomlx/ may import
// the GoMLX runtime packages (backend construction, plugin/device, execution).
// The op-vocabulary (core/graph, core/tensors, compute/dtypes, compute/shapes)
// is intentionally NOT matched — model/ and train/ may use it.
func TestBackendBoundary(t *testing.T) {
	root, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		t.Fatalf("git root: %v", err)
	}
	// Runtime packages (ADR-0009): backend construction, plugin/device, execution.
	// The op-vocabulary (core/graph, core/tensors, compute/dtypes, compute/shapes)
	// is intentionally NOT matched — model/ and train/ may use it.
	const runtime = `"github\.com/gomlx/compute"|github\.com/gomlx/gomlx/backends|github\.com/gomlx/go-xla|github\.com/gomlx/gopjrt`
	cmd := exec.Command("git", "grep", "-lE", runtime, "--", "*.go", ":!app/boundary_test.go")
	cmd.Dir = strings.TrimSpace(string(root))
	out, err := cmd.Output()
	if err != nil {
		if ee, ok := err.(*exec.ExitError); !ok || ee.ExitCode() != 1 {
			t.Fatalf("git grep failed: %v", err)
		}
		// exit 1 == no matches == clean
	}
	for line := range strings.SplitSeq(strings.TrimSpace(string(out)), "\n") {
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "backend/gomlx/") {
			t.Errorf("runtime import outside backend/gomlx/: %s", line)
		}
	}
}
