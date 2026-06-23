package main

import (
	"compress/gzip"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"

	"github.com/guygrigsby/lmkit-go/tokenizer"
)

// TestShardWriter checks the tokenize->shard path: documents become uint16 tokens with an EOT
// separator, the first valTokens go to val_00000.bin, and the rest fill train shards of
// shardTokens each, with the totals conserved.
func TestShardWriter(t *testing.T) {
	tok, err := tokenizer.Load("../../../tokenizer/testdata/tokenizer.json")
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	dir := t.TempDir()
	src := filepath.Join(dir, "docs.jsonl.gz")
	f, err := os.Create(src)
	if err != nil {
		t.Fatal(err)
	}
	gz := gzip.NewWriter(f)
	docs := `{"text":"Hello world, this is a test of the sharder."}
{"text":"Another document with several more words to tokenize."}
{"other":"ignored"}
not json at all
`
	if _, err := gz.Write([]byte(docs)); err != nil {
		t.Fatal(err)
	}
	gz.Close()
	f.Close()

	out := filepath.Join(dir, "out")
	if err := os.MkdirAll(out, 0o755); err != nil {
		t.Fatal(err)
	}
	w := &shardWriter{dir: out, shardTokens: 10, valTokens: 6, eot: 0}
	if err := w.tokenizeFile(src, tok, "text"); err != nil {
		t.Fatalf("tokenizeFile: %v", err)
	}
	if err := w.flush(); err != nil {
		t.Fatalf("flush: %v", err)
	}

	if w.docs != 2 { // the two valid "text" docs; the bad lines are skipped
		t.Errorf("docs = %d, want 2", w.docs)
	}

	// val shard holds exactly valTokens; every token across all shards sums to w.total.
	val := readShard(t, filepath.Join(out, "val_00000.bin"))
	if len(val) != 6 {
		t.Errorf("val tokens = %d, want 6", len(val))
	}
	total := len(val)
	for i := 0; ; i++ {
		p := filepath.Join(out, "train_"+pad5(i)+".bin")
		if _, err := os.Stat(p); err != nil {
			break
		}
		total += len(readShard(t, p))
	}
	if int64(total) != w.total {
		t.Errorf("tokens across shards = %d, want total %d", total, w.total)
	}

	// the EOT (0) must appear: documents are separated.
	if !contains(val, 0) && total == int(w.total) {
		all := val
		for i := 0; ; i++ {
			p := filepath.Join(out, "train_"+pad5(i)+".bin")
			if _, err := os.Stat(p); err != nil {
				break
			}
			all = append(all, readShard(t, p)...)
		}
		if !contains(all, 0) {
			t.Error("no EOT (0) token found; document separators missing")
		}
	}
}

func readShard(t *testing.T, path string) []uint16 {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	out := make([]uint16, len(b)/2)
	for i := range out {
		out[i] = binary.LittleEndian.Uint16(b[2*i:])
	}
	return out
}

func pad5(i int) string {
	s := []byte("00000")
	for k := 4; k >= 0 && i > 0; k-- {
		s[k] = byte('0' + i%10)
		i /= 10
	}
	return string(s)
}

func contains(a []uint16, v uint16) bool {
	for _, x := range a {
		if x == v {
			return true
		}
	}
	return false
}
