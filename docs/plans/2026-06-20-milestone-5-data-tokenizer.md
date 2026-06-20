# Milestone 5 — data + tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build two pure-Go modules — `data` (mmap `uint16` shard DataLoader → next-token `(x,y)` blocks) and `tokenizer` (byte-level BPE loading the 32k `tokenizer.json`, encode/decode) — both validated against the real lm-100m-en artifacts.

**Architecture:** Both are dependency-light, GoMLX-free libraries (ADR-0003/0005). `data` mmaps headerless little-endian `uint16` shards and yields raw `int32` batches via a prefetch goroutine. `tokenizer` is a clean-room HF byte-level BPE, gated byte-exact against Python `tokenizers`.

**Tech Stack:** Go 1.26; `github.com/edsrzf/mmap-go` (data); `github.com/dlclark/regexp2` (tokenizer, GPT-2 split regex with lookahead); Python `tokenizers` on a host with Python tokenizers for the equivalence fixture.

## Global Constraints

- **Pure-Go, GoMLX-free (ADR-0005):** neither module imports any `github.com/gomlx/*`. They trivially pass the runtime boundary test (`-count=1`).
- **Read existing shards (chosen):** the DataLoader reads the lm-100m-en `.bin` shards; no Go sharder, no BPE training, no corpus pull.
- **Shard format:** raw, headerless, **little-endian `uint16`**, 1 token = 1 uint16. Valid on LE hosts (x86-64 linux, arm64 darwin); document the LE assumption.
- **Tokenizer correctness = byte-exact vs Python (ADR-0003):** the committed equivalence fixture is the arbiter — match the *fixture*, never loosen it.
- **PyTorch/HF on the GPU host:** the fixture generator runs on a host with a Python venv with `tokenizers`; `tokenizer.json` + fixtures are copied back and committed.
- SimpleGo not relevant (no GoMLX). Commits: terse, verb-first, no dashes, no Claude/Anthropic attribution.

---

### Task 1: `data` — mmap uint16 Shard

**Files:**
- Create: `data/go.mod`
- Modify: `go.work` (add `./data`)
- Create: `data/shard.go`
- Create: `data/shard_test.go`

**Interfaces:**
- Produces: `data.OpenShard(path string) (*Shard, error)`; `(*Shard).Tokens() []uint16`; `(*Shard).Len() int`; `(*Shard).Close() error`.

- [ ] **Step 1: Create the module and add to the workspace**

`data/go.mod`:
```
module github.com/guygrigsby/lmkit-go/data

go 1.26
```
`go.work` — add `./data` to the `use` block. Then add the mmap dep:
```bash
cd data && go get github.com/edsrzf/mmap-go@latest && cd ..
```

- [ ] **Step 2: Write the failing Shard test**

`data/shard_test.go`:
```go
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
```
Run: `cd data && go test ./... -run TestShardReadsTokens -v` → FAIL (`OpenShard` undefined).

- [ ] **Step 3: Implement Shard**

`data/shard.go`:
```go
// Package data reads tokenized uint16 shards and serves next-token training
// batches. Pure Go, no GoMLX (ADR-0005): it yields raw int32 slices; the train
// layer wraps them in tensors.
package data

import (
	"fmt"
	"os"
	"unsafe"

	"github.com/edsrzf/mmap-go"
)

// Shard is a memory-mapped uint16 token shard: raw, headerless, little-endian
// (nanoGPT-style). Tokens() aliases the mapping and is valid only until Close.
type Shard struct {
	m    mmap.MMap
	toks []uint16
}

// OpenShard memory-maps a .bin shard read-only. The host must be little-endian
// (x86-64, arm64); the uint16 view reinterprets the mapped bytes in host order.
func OpenShard(path string) (*Shard, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	m, err := mmap.Map(f, mmap.RDONLY, 0)
	if err != nil {
		return nil, fmt.Errorf("data: mmap %s: %w", path, err)
	}
	if len(m) == 0 || len(m)%2 != 0 {
		_ = m.Unmap()
		return nil, fmt.Errorf("data: %s: bad length %d (want positive, even)", path, len(m))
	}
	toks := unsafe.Slice((*uint16)(unsafe.Pointer(&m[0])), len(m)/2)
	return &Shard{m: m, toks: toks}, nil
}

func (s *Shard) Tokens() []uint16 { return s.toks }
func (s *Shard) Len() int         { return len(s.toks) }

// Close unmaps the shard. Tokens() must not be used after Close.
func (s *Shard) Close() error {
	s.toks = nil
	return s.m.Unmap()
}
```
Run: `cd data && go test ./... -run TestShardReadsTokens -v` → PASS.

- [ ] **Step 4: Commit**

```bash
git add data/ go.work
git commit -m "data: mmap uint16 shard reader"
```

---

### Task 2: `data` — DataLoader (next-token blocks)

**Files:**
- Create: `data/loader.go`
- Create: `data/loader_test.go`
- Modify: `scripts/check.sh` (add `data` to the module loop)

**Interfaces:**
- Consumes: `data.OpenShard`, `*Shard`.
- Produces: `data.Config{Shards []string; BlockSize, BatchSize int; Seed int64}`; `data.Batch{X, Y []int32}`; `data.New(Config) (*Loader, error)`; `(*Loader).Next() Batch`; `(*Loader).Close() error`.

- [ ] **Step 1: Write the failing loader tests**

`data/loader_test.go`:
```go
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
```
Run: `cd data && go test ./... -run TestLoader -v` → FAIL (`New` undefined).

- [ ] **Step 2: Implement the DataLoader**

`data/loader.go`:
```go
package data

import (
	"fmt"
	"math/rand"
)

// Config configures a Loader. Shards are .bin paths; the train/val split is the
// caller's choice of which files to pass (the lm-100m shards are pre-split).
type Config struct {
	Shards    []string
	BlockSize int
	BatchSize int
	Seed      int64
}

// Batch holds one batch of next-token training data, row-major [BatchSize,BlockSize].
type Batch struct {
	X []int32 // input token ids
	Y []int32 // next-token target ids (Y[k] == the token after X[k])
}

// Loader serves random next-token block batches from mmap'd shards via a prefetch
// goroutine. Deterministic given Seed.
type Loader struct {
	cfg     Config
	shards  []*Shard
	rng     *rand.Rand
	ch      chan Batch
	done    chan struct{}
	stopped chan struct{}
}

func New(cfg Config) (*Loader, error) {
	if cfg.BlockSize < 1 || cfg.BatchSize < 1 {
		return nil, fmt.Errorf("data: BlockSize and BatchSize must be >= 1")
	}
	var shards []*Shard
	for _, p := range cfg.Shards {
		s, err := OpenShard(p)
		if err != nil {
			for _, o := range shards {
				_ = o.Close()
			}
			return nil, err
		}
		if s.Len() < cfg.BlockSize+1 { // need room for x[0:block] and y=x shifted by 1
			_ = s.Close()
			continue
		}
		shards = append(shards, s)
	}
	if len(shards) == 0 {
		return nil, fmt.Errorf("data: no usable shards (need >= BlockSize+1 tokens)")
	}
	l := &Loader{
		cfg: cfg, shards: shards,
		rng:     rand.New(rand.NewSource(cfg.Seed)),
		ch:      make(chan Batch, 4),
		done:    make(chan struct{}),
		stopped: make(chan struct{}),
	}
	go l.fill()
	return l, nil
}

func (l *Loader) fill() {
	defer close(l.stopped)
	for {
		select {
		case <-l.done:
			return
		default:
		}
		b := l.makeBatch()
		select {
		case l.ch <- b:
		case <-l.done:
			return
		}
	}
}

func (l *Loader) makeBatch() Batch {
	bs, blk := l.cfg.BatchSize, l.cfg.BlockSize
	x := make([]int32, bs*blk)
	y := make([]int32, bs*blk)
	for r := 0; r < bs; r++ {
		toks := l.shards[l.rng.Intn(len(l.shards))].Tokens()
		i := l.rng.Intn(len(toks) - blk) // start in [0, len-blk-1]; i+blk <= len-1
		for k := 0; k < blk; k++ {
			x[r*blk+k] = int32(toks[i+k])
			y[r*blk+k] = int32(toks[i+k+1])
		}
	}
	return Batch{X: x, Y: y}
}

// Next returns the next prefetched batch.
func (l *Loader) Next() Batch { return <-l.ch }

// Close stops prefetching and unmaps shards. It waits for the fill goroutine to
// exit before unmapping, so no batch is built from an unmapped shard.
func (l *Loader) Close() error {
	close(l.done)
	<-l.stopped
	var err error
	for _, s := range l.shards {
		if e := s.Close(); e != nil {
			err = e
		}
	}
	return err
}
```
Run: `cd data && go test ./... -run TestLoader -v` → PASS (TestLoaderNextToken, TestLoaderDeterministic; TestRealShardSmoke skipped).

- [ ] **Step 3: Real-shard smoke on the GPU host** (pure-Go, cross-compile the test binary)

`$GPU_HOST` = your CUDA host. `$DATA_DIR` = the lm-100m-en data dir on that host.

```bash
GOOS=linux GOARCH=amd64 CGO_ENABLED=0 go test -c -o /tmp/data.test ./data/
scp /tmp/data.test "$GPU_HOST":/tmp/data.test
ssh "$GPU_HOST" 'LMKIT_SHARD=$DATA_DIR/train_00000.bin /tmp/data.test -test.run TestRealShardSmoke -test.v'
ssh "$GPU_HOST" 'rm -f /tmp/data.test'; rm -f /tmp/data.test
```
Expected: PASS — logs the shard token count (~500M) and a first block of valid `< 32000` ids.

- [ ] **Step 4: Add `data` to the gate**

`scripts/check.sh` — add `data` to the loop: `for mod in backend model train data app; do`. (data has no GoMLX, but the loop builds+tests it.)
Run: `make check` → all modules + boundary green, `OK`.

- [ ] **Step 5: Commit**

```bash
git add data/ scripts/check.sh
git commit -m "data: next-token block DataLoader over mmap shards

Prefetch goroutine yields random (x,y) blocks as int32; deterministic by seed;
Close waits for the filler before unmapping. Real-shard smoke verified on the GPU host."
```

---

### Task 3: `tokenizer` — Load + byte<->unicode + Decode

**Files:**
- Create: `tokenizer/go.mod`
- Modify: `go.work` (add `./tokenizer`)
- Create: `tokenizer/testdata/tokenizer.json` (copied from the GPU host)
- Create: `tokenizer/bytelevel.go`
- Create: `tokenizer/tokenizer.go`
- Create: `tokenizer/tokenizer_test.go`
- Modify: `scripts/check.sh` (add `tokenizer`)

**Interfaces:**
- Produces: `tokenizer.Load(path string) (*Tokenizer, error)`; `(*Tokenizer).Decode(ids []int) string`; `(*Tokenizer).Vocab() map[string]int`; internal `byteToRune [256]rune` / `runeToByte map[rune]byte`.

- [ ] **Step 1: Create the module + copy the real tokenizer.json**

`tokenizer/go.mod`:
```
module github.com/guygrigsby/lmkit-go/tokenizer

go 1.26
```
Add `./tokenizer` to `go.work`. Copy the real tokenizer from the GPU host (the artifact we load + test against):
```bash
mkdir -p tokenizer/testdata
scp "$GPU_HOST":'$DATA_DIR/tokenizer.json' tokenizer/testdata/tokenizer.json
```
Then `cd tokenizer && go get github.com/dlclark/regexp2@latest && cd ..`

- [ ] **Step 2: Implement the byte<->unicode map**

`tokenizer/bytelevel.go`:
```go
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
```

- [ ] **Step 3: Write the failing Load/Decode test**

`tokenizer/tokenizer_test.go`:
```go
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
```
Run: `cd tokenizer && go test ./... -run 'TestLoad|TestByteRune' -v` → FAIL (`Load` undefined).

- [ ] **Step 4: Implement Load + Decode**

`tokenizer/tokenizer.go`:
```go
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
```
Run: `cd tokenizer && go test ./... -run 'TestLoad|TestByteRune' -v` → PASS. (If `len(Vocab()) != 32000`, the JSON path for vocab differs — inspect tokenizer.json and fix the struct path.)

- [ ] **Step 5: Add `tokenizer` to the gate, commit**

`scripts/check.sh`: `for mod in backend model train data tokenizer app; do`.
Run: `make check` → OK.
```bash
git add tokenizer/ go.work scripts/check.sh
git commit -m "tokenizer: load HF byte-level BPE tokenizer.json + decode"
```

---

### Task 4: `tokenizer` — Encode + Python equivalence gate

**Files:**
- Create: `tokenizer/bpe.go`
- Modify: `tokenizer/tokenizer.go` (add `Encode`)
- Create: `tokenizer/testdata/gen_encodings.py`
- Create: `tokenizer/testdata/encodings.json` (generated on the GPU host)
- Create: `tokenizer/encode_test.go`

**Interfaces:**
- Consumes: `Load`, `Decode`, the byte map, `merges`, `vocab`.
- Produces: `(*Tokenizer).Encode(text string) []int`.

- [ ] **Step 1: Implement BPE merge application**

`tokenizer/bpe.go`:
```go
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
```

- [ ] **Step 2: Implement Encode** (add to `tokenizer/tokenizer.go`)

Add the regexp2 import and the GPT-2 split pattern + `Encode`:
```go
import (
	// ... existing ...
	"github.com/dlclark/regexp2"
)

// gpt2Pattern is the byte-level pre-tokenization regex HF ByteLevel uses (needs
// lookahead, so RE2/stdlib regexp can't run it — hence regexp2).
var gpt2Pattern = regexp2.MustCompile(
	`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`,
	regexp2.None)

// Encode tokenizes text to ids via byte-level BPE: optional prefix space, GPT-2
// regex pre-tokenization, byte->unicode mapping, ranked merges, vocab lookup.
func (t *Tokenizer) Encode(text string) []int {
	if t.addPrefix && (len(text) == 0 || text[0] != ' ') {
		text = " " + text
	}
	var ids []int
	for _, piece := range splitGPT2(text) {
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
```
(Special/added-token handling: the equivalence fixture's plain-prose + corpus samples won't contain the 4 specials, so literal special-splitting isn't needed to pass the gate. If the fixture later includes a special token and fails, add a pre-split on `added` special contents — driven by the fixture, not speculatively.)

- [ ] **Step 3: Write the Python fixture generator**

`tokenizer/testdata/gen_encodings.py`:
```python
#!/usr/bin/env python3
"""Generate the byte-level BPE equivalence fixture from the reference Python
tokenizer. Run on a host with a venv with `tokenizers`:
    python gen_encodings.py  # reads tokenizer.json, writes encodings.json
"""
import json
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")
SAMPLES = [
    "Hello, world!",
    "  leading and   multiple   spaces ",
    "trailing space ",
    "no_prefix",
    "Punctuation?! (parens) — em-dash, and 'quotes'.",
    "Numbers 12345 and 6.78 mixed with words2024.",
    "Unicode: café, naïve, résumé, 日本語, emoji 🙂.",
    "Newlines\nand\ttabs\tin text.",
    "A longer sentence of clean English prose, the kind the corpus is full of, "
    "to exercise common merges over ordinary words and spacing.",
    "ALLCAPS and MixedCase and snake_case and kebab-case.",
]
out = [{"text": s, "ids": tok.encode(s).ids} for s in SAMPLES]
with open("encodings.json", "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=0)
print(f"wrote encodings.json: {len(out)} cases")
```
Generate on the GPU host (it has `tokenizer.json` at `$DATA_DIR`; use a Python venv with `tokenizers`, else `pip install tokenizers` in a throwaway):
`$GPU_HOST` = your CUDA host. `$DATA_DIR` = the lm-100m-en data dir on that host.

```bash
scp tokenizer/testdata/gen_encodings.py "$GPU_HOST":'$DATA_DIR/gen_encodings.py'
ssh "$GPU_HOST" 'cd $DATA_DIR && (python -c "import tokenizers" 2>/dev/null && python gen_encodings.py || python3 gen_encodings.py)'
scp "$GPU_HOST":'$DATA_DIR/encodings.json' tokenizer/testdata/encodings.json
ssh "$GPU_HOST" 'cd $DATA_DIR && rm -f gen_encodings.py encodings.json'
```
Expected: `encodings.json` with 10 `{text, ids}` cases lands in `tokenizer/testdata/`. (If `import tokenizers` fails in every venv, STOP and report — the gate needs the reference. Confirm which Python on the GPU host has `tokenizers`.)

- [ ] **Step 4: Write the failing equivalence + round-trip test**

`tokenizer/encode_test.go`:
```go
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
```
Run: `cd tokenizer && go test ./... -run TestEncodeMatchesPython -v` → FAIL (`Encode` undefined, then mismatches if any convention is off).

- [ ] **Step 5: Make it pass — match the fixture, do not loosen it**

Run: `cd tokenizer && go test ./... -run TestEncodeMatchesPython -v`
Expected: PASS — every case's ids equal Python's, and Decode round-trips. If a case mismatches, the bug is a convention difference; debug in this order (do NOT delete cases or loosen the assert):
1. **add_prefix_space** — does the reference prepend a leading space? Compare the first id of `"no_prefix"`.
2. **regex** — whitespace-run handling (`"multiple   spaces"`, trailing space, `\n`/`\t`) points at the split pattern / lookahead.
3. **merges format / rank** — wrong order in `parseMerges` flips merges; check a multi-merge word like `"sentence"`.
4. **byte map** — non-ASCII (`café`, `🙂`) failing points at `gpt2ByteToRune`.
Fix the implementation at the cause; re-run until all 10 pass.

- [ ] **Step 6: Full gate + commit**

Run: `cd tokenizer && go test ./... -v` then `make check`
Expected: Load, byte-roundtrip, encode-equivalence + round-trip all pass; `make check` OK.
```bash
git add tokenizer/bpe.go tokenizer/tokenizer.go tokenizer/encode_test.go tokenizer/testdata/gen_encodings.py tokenizer/testdata/encodings.json
git commit -m "tokenizer: byte-level BPE encode + Python equivalence gate

Encode (ByteLevel pre-tokenization via regexp2 + ranked merges) matches the
Python tokenizers reference byte-exact on a committed fixture; Decode round-trips."
```

---

## Self-Review

**Spec coverage** (against `2026-06-20-milestone-5-data-tokenizer-design.md`):
- `data` module + go.work + gate → Tasks 1/2.
- Shard mmap uint16 (LE, zero-copy, Close lifetime) → Task 1.
- DataLoader next-token (x,y) int32 blocks, batch, prefetch, determinism, multi-shard, train/val by shard list → Task 2.
- real-shard smoke on the GPU host → Task 2 Step 3.
- `tokenizer` module + go.work + gate → Tasks 3/4.
- Load tokenizer.json (vocab/merges/added/ByteLevel), byte<->unicode, Decode → Task 3.
- Encode (ByteLevel pretok via regexp2 + BPE), equivalence gate vs Python on the GPU host, round-trip → Task 4.
- both GoMLX-free / boundary clean → no gomlx imports anywhere in data/ or tokenizer/.

**Out of scope, correctly absent:** BPE training, corpus pull, Go sharder, the training loop's use, chat templates/offsets/padding.

**Placeholder scan:** complete Go + Python throughout; the one judgment point (Encode convention mismatches) is a precise ordered debug procedure against the fixture, not a vague placeholder; special-token handling is explicitly deferred-unless-fixture-demands (not speculative).

**Type consistency:** `data.{Shard,Config,Batch,Loader}` and `OpenShard/New/Next/Close`; `tokenizer.{Tokenizer,pair,addedToken}` and `Load/Encode/Decode/Vocab/bpe/gpt2ByteToRune/splitGPT2` are used identically across files. `Batch.X/Y []int32` row-major `[BatchSize,BlockSize]` consistent in loader + tests. The mmap lifetime contract (Close waits for the filler) is honored in `Loader.Close`.
</content>
