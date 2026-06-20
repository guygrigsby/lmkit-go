// Package tokenizer is a pure-Go byte-level BPE that loads an HF tokenizer.json
// and encodes/decodes byte-exact to the Python `tokenizers` reference (gated).
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

type pair struct{ a, b string }

type addedToken struct {
	ID      int    `json:"id"`
	Content string `json:"content"`
	Special bool   `json:"special"`
}

// Tokenizer is a loaded byte-level BPE.
type Tokenizer struct {
	vocab      map[string]int
	idToTok    map[int]string
	merges     map[pair]int // pair -> rank (lower = earlier)
	added      []addedToken
	addPrefix  bool
	byteToRune [256]rune
	runeToByte map[rune]byte
}

// Load parses an HF tokenizer.json (model.type BPE, ByteLevel pre-tokenizer).
func Load(path string) (*Tokenizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var doc struct {
		AddedTokens []addedToken `json:"added_tokens"`
		PreTok      struct {
			Type         string `json:"type"`
			AddPrefixSpc bool   `json:"add_prefix_space"`
		} `json:"pre_tokenizer"`
		Model struct {
			Type   string          `json:"type"`
			Vocab  map[string]int  `json:"vocab"`
			Merges json.RawMessage `json:"merges"`
		} `json:"model"`
	}
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, fmt.Errorf("tokenizer: parse: %w", err)
	}
	merges, err := parseMerges(doc.Model.Merges)
	if err != nil {
		return nil, err
	}
	b2r, r2b := gpt2ByteToRune()
	idToTok := make(map[int]string, len(doc.Model.Vocab))
	for tok, id := range doc.Model.Vocab {
		idToTok[id] = tok
	}
	for _, a := range doc.AddedTokens {
		idToTok[a.ID] = a.Content
	}
	return &Tokenizer{
		vocab: doc.Model.Vocab, idToTok: idToTok, merges: merges,
		added: doc.AddedTokens, addPrefix: doc.PreTok.AddPrefixSpc,
		byteToRune: b2r, runeToByte: r2b,
	}, nil
}

// parseMerges accepts either ["a b", ...] or [["a","b"], ...]; rank = index.
func parseMerges(raw json.RawMessage) (map[pair]int, error) {
	m := map[pair]int{}
	var asStrings []string
	if err := json.Unmarshal(raw, &asStrings); err == nil {
		for i, s := range asStrings {
			parts := strings.SplitN(s, " ", 2)
			if len(parts) != 2 {
				return nil, fmt.Errorf("tokenizer: bad merge %q", s)
			}
			m[pair{parts[0], parts[1]}] = i
		}
		return m, nil
	}
	var asPairs [][]string
	if err := json.Unmarshal(raw, &asPairs); err != nil {
		return nil, fmt.Errorf("tokenizer: merges format: %w", err)
	}
	for i, p := range asPairs {
		if len(p) != 2 {
			return nil, fmt.Errorf("tokenizer: bad merge pair %v", p)
		}
		m[pair{p[0], p[1]}] = i
	}
	return m, nil
}

func (t *Tokenizer) Vocab() map[string]int { return t.vocab }

// Decode maps ids -> token strings -> the byte-level unicode string -> bytes.
func (t *Tokenizer) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		sb.WriteString(t.idToTok[id])
	}
	s := sb.String()
	out := make([]byte, 0, len(s))
	for _, r := range s {
		if b, ok := t.runeToByte[r]; ok {
			out = append(out, b)
		}
	}
	return string(out)
}
