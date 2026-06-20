package data

import (
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

// writeShard writes toks as raw little-endian uint16 to a temp .bin file.
func writeShard(t *testing.T, toks []uint16) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "s.bin")
	buf := make([]byte, len(toks)*2)
	for i, v := range toks {
		binary.LittleEndian.PutUint16(buf[i*2:], v)
	}
	if err := os.WriteFile(p, buf, 0o644); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestShardReadsTokens(t *testing.T) {
	want := []uint16{0, 1, 2, 31999, 12345, 7}
	s, err := OpenShard(writeShard(t, want))
	if err != nil {
		t.Fatalf("OpenShard: %v", err)
	}
	defer s.Close()
	if s.Len() != len(want) {
		t.Fatalf("Len=%d want %d", s.Len(), len(want))
	}
	got := s.Tokens()
	for i := range want {
		if got[i] != want[i] {
			t.Errorf("tok[%d]=%d want %d", i, got[i], want[i])
		}
	}
}
