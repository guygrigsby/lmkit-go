package tokenizer

import "math"

// bpe applies ranked merges greedily (lowest rank first) to a symbol sequence
// (each symbol a byte-level unicode string), matching HF's BPE.
func (t *Tokenizer) bpe(symbols []string) []string {
	for len(symbols) >= 2 {
		bestRank, bestI := math.MaxInt, -1
		for i := 0; i+1 < len(symbols); i++ {
			if r, ok := t.merges[pair{symbols[i], symbols[i+1]}]; ok && r < bestRank {
				bestRank, bestI = r, i
			}
		}
		if bestI < 0 {
			break
		}
		merged := symbols[bestI] + symbols[bestI+1]
		next := make([]string, 0, len(symbols)-1)
		next = append(next, symbols[:bestI]...)
		next = append(next, merged)
		next = append(next, symbols[bestI+2:]...)
		symbols = next
	}
	return symbols
}
