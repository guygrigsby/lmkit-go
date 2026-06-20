package tokenizer

// gpt2ByteToRune builds GPT-2's reversible byte<->unicode map: printable byte
// ranges map to themselves; the rest map to 256+n. Used so BPE operates over a
// unicode alphabet with no control chars.
func gpt2ByteToRune() ([256]rune, map[rune]byte) {
	var printable []int
	add := func(lo, hi int) {
		for c := lo; c <= hi; c++ {
			printable = append(printable, c)
		}
	}
	add('!', '~')   // 33..126
	add(0xA1, 0xAC) // ¡..¬
	add(0xAE, 0xFF) // ®..ÿ
	inPrintable := map[int]bool{}
	for _, c := range printable {
		inPrintable[c] = true
	}
	var b2r [256]rune
	r2b := map[rune]byte{}
	n := 0
	for b := 0; b < 256; b++ {
		var r rune
		if inPrintable[b] {
			r = rune(b)
		} else {
			r = rune(256 + n)
			n++
		}
		b2r[b] = r
		r2b[r] = byte(b)
	}
	return b2r, r2b
}
