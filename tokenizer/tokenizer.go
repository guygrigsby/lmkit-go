// Package tokenizer is a pure-Go byte-level BPE that loads an HF tokenizer.json
// and encodes/decodes byte-exact to the Python `tokenizers` reference (gated).
package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/dlclark/regexp2"
)

// gpt2Pattern is the byte-level pre-tokenization regex HF ByteLevel uses (needs
// lookahead, so RE2/stdlib regexp can't run it — hence regexp2).
var gpt2Pattern = regexp2.MustCompile(
	`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`,
	regexp2.None)

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

// Encode tokenizes text to ids via byte-level BPE: optional prefix space, GPT-2
// regex pre-tokenization, byte->unicode mapping, ranked merges, vocab lookup.
// Special/added tokens are matched first (longest content wins) and emitted as
// their single id; the gaps between matches go through the normal BPE path.
func (t *Tokenizer) Encode(text string) []int {
	if t.addPrefix && (len(text) == 0 || text[0] != ' ') {
		text = " " + text
	}
	// Build a longest-first ordering of added token contents so a longer match
	// always shadows a prefix (e.g. "<|im_start|>" before "<|").
	type specEntry struct {
		content string
		id      int
	}
	specs := make([]specEntry, 0, len(t.added))
	for _, a := range t.added {
		if a.Content != "" {
			specs = append(specs, specEntry{a.Content, a.ID})
		}
	}
	sort.Slice(specs, func(i, j int) bool {
		return len(specs[i].content) > len(specs[j].content)
	})

	var ids []int
	// encodePlain runs the GPT-2 regex + BPE path on a plain (non-special) segment.
	encodePlain := func(seg string) {
		for _, piece := range splitGPT2(seg) {
			symbols := make([]string, 0, len(piece))
			for _, b := range []byte(piece) {
				symbols = append(symbols, string(t.byteToRune[b]))
			}
			for _, tok := range t.bpe(symbols) {
				if id, ok := t.vocab[tok]; ok {
					ids = append(ids, id)
				}
			}
		}
	}

	if len(specs) == 0 {
		encodePlain(text)
		return ids
	}

	// Scan left-to-right; at each position try to match a special token.
	pos := 0
	plainStart := 0
	for pos < len(text) {
		matched := false
		for _, sp := range specs {
			if strings.HasPrefix(text[pos:], sp.content) {
				// Flush any plain text before this match.
				if plainStart < pos {
					encodePlain(text[plainStart:pos])
				}
				ids = append(ids, sp.id)
				pos += len(sp.content)
				plainStart = pos
				matched = true
				break
			}
		}
		if !matched {
			pos++
		}
	}
	// Flush trailing plain text.
	if plainStart < len(text) {
		encodePlain(text[plainStart:])
	}
	return ids
}

func splitGPT2(text string) []string {
	var out []string
	m, _ := gpt2Pattern.FindStringMatch(text)
	for m != nil {
		out = append(out, m.String())
		m, _ = gpt2Pattern.FindNextMatch(m)
	}
	return out
}

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
