package app_test

import (
	"os/exec"
	"strings"
	"testing"
)

// TestBackendBoundary enforces ADR-0002: only files under backend/ may import the
// vendor stack (everything ships under github.com/gomlx). It greps the whole repo
// from the root and excludes itself (this file names the vendor path in a string).
func TestBackendBoundary(t *testing.T) {
	root, err := exec.Command("git", "rev-parse", "--show-toplevel").Output()
	if err != nil {
		t.Fatalf("git root: %v", err)
	}
	cmd := exec.Command("git", "grep", "-lE", `github\.com/gomlx`,
		"--", "*.go", ":!app/boundary_test.go")
	cmd.Dir = strings.TrimSpace(string(root))
	out, _ := cmd.Output() // exit 1 (no matches) yields empty output; that's a pass
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line == "" {
			continue
		}
		if !strings.HasPrefix(line, "backend/") {
			t.Errorf("vendor import outside backend/: %s", line)
		}
	}
}
