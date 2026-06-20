package tokenizer

import "testing"

const tjson = "testdata/tokenizer.json"

func TestLoad(t *testing.T) {
	tk, err := Load(tjson)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if n := len(tk.Vocab()); n != 32000 {
		t.Errorf("vocab size %d want 32000", n)
	}
}

func TestByteRuneRoundTrip(t *testing.T) {
	b2r, r2b := gpt2ByteToRune()
	for b := 0; b < 256; b++ {
		if r2b[b2r[byte(b)]] != byte(b) {
			t.Fatalf("byte %d did not round-trip", b)
		}
	}
}
