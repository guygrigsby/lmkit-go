package data

import (
	"os"
	"testing"
)

func seq(n int) []uint16 { // 0,1,2,...,n-1
	s := make([]uint16, n)
	for i := range s {
		s[i] = uint16(i)
	}
	return s
}

func TestLoaderNextToken(t *testing.T) {
	// one shard [0..99], block 8, batch 4
	p := writeShard(t, seq(100))
	l, err := New(Config{Shards: []string{p}, BlockSize: 8, BatchSize: 4, Seed: 1})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer l.Close()
	b := l.Next()
	if len(b.X) != 32 || len(b.Y) != 32 {
		t.Fatalf("batch len X=%d Y=%d want 32", len(b.X), len(b.Y))
	}
	// next-token: within each row, Y[k] == X[k]+1 (since shard is 0,1,2,...),
	// and Y[k] == X[k+1] across the row.
	for r := 0; r < 4; r++ {
		for k := 0; k < 8; k++ {
			if b.Y[r*8+k] != b.X[r*8+k]+1 {
				t.Errorf("row %d k %d: Y=%d X=%d (want Y==X+1)", r, k, b.Y[r*8+k], b.X[r*8+k])
			}
			if k < 7 && b.Y[r*8+k] != b.X[r*8+k+1] {
				t.Errorf("row %d k %d: Y=%d next-X=%d", r, k, b.Y[r*8+k], b.X[r*8+k+1])
			}
		}
	}
}

func TestLoaderDeterministic(t *testing.T) {
	p := writeShard(t, seq(100))
	mk := func() Batch {
		l, _ := New(Config{Shards: []string{p}, BlockSize: 8, BatchSize: 4, Seed: 42})
		defer l.Close()
		return l.Next()
	}
	a, b := mk(), mk()
	for i := range a.X {
		if a.X[i] != b.X[i] {
			t.Fatalf("same seed differs at %d: %d vs %d", i, a.X[i], b.X[i])
		}
	}
	// different seed should (very likely) differ
	l2, _ := New(Config{Shards: []string{p}, BlockSize: 8, BatchSize: 4, Seed: 7})
	defer l2.Close()
	c := l2.Next()
	same := true
	for i := range a.X {
		if a.X[i] != c.X[i] {
			same = false
			break
		}
	}
	if same {
		t.Errorf("different seeds produced identical batch")
	}
}

// real-shard smoke: set LMKIT_SHARD to a real train_*.bin (run on a host with a real shard). Skipped
// when unset, so it never runs in normal CI.
func TestRealShardSmoke(t *testing.T) {
	p := os.Getenv("LMKIT_SHARD")
	if p == "" {
		t.Skip("set LMKIT_SHARD to a real .bin to run")
	}
	l, err := New(Config{Shards: []string{p}, BlockSize: 2048, BatchSize: 2, Seed: 1})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer l.Close()
	b := l.Next()
	if len(b.X) != 2*2048 {
		t.Fatalf("X len %d want %d", len(b.X), 2*2048)
	}
	for i, v := range b.X {
		if v < 0 || v >= 32000 {
			t.Fatalf("token %d out of vocab range: %d", i, v)
		}
	}
	t.Logf("real shard OK: %d tokens, first block starts %v", l.shards[0].Len(), b.X[:8])
}
