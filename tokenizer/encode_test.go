package tokenizer

import (
	"encoding/json"
	"os"
	"testing"
)

func TestEncodeMatchesPython(t *testing.T) {
	raw, err := os.ReadFile("testdata/encodings.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cases []struct {
		Text string `json:"text"`
		IDs  []int  `json:"ids"`
	}
	if err := json.Unmarshal(raw, &cases); err != nil {
		t.Fatalf("decode fixture: %v", err)
	}
	tk, err := Load(tjson)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	for _, c := range cases {
		got := tk.Encode(c.Text)
		if len(got) != len(c.IDs) {
			t.Errorf("len mismatch for %q: got %d want %d\n got=%v\nwant=%v", c.Text, len(got), len(c.IDs), got, c.IDs)
			continue
		}
		for i := range got {
			if got[i] != c.IDs[i] {
				t.Errorf("id mismatch for %q at %d: got %d want %d", c.Text, i, got[i], c.IDs[i])
				break
			}
		}
		if dec := tk.Decode(c.IDs); dec != c.Text {
			// byte-level BPE is lossless; decode of the reference ids must equal text.
			t.Errorf("round-trip for %q: Decode=%q", c.Text, dec)
		}
	}
}
