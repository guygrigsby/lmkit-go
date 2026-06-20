# Milestone 5 — data + tokenizer (spec)

**Status:** ready for plan. Sub-projects 4–5 of the context map. Decisions:
ADR-0001..0009. Two new dependency-light, pure-Go modules (no GoMLX) — the reusable
OSS pieces (ADR-0003/0005).

## Goal

Build the two contexts a replica train needs to read its inputs, faithful to the
lm-100m-en pipeline:
- **`data`** — a memory-mapped `DataLoader` that reads the existing `uint16` `.bin`
  shards and yields `(x, y)` next-token block batches.
- **`tokenizer`** — a pure-Go byte-level BPE that loads the existing 32k
  `tokenizer.json`, encodes/decodes, byte-exact vs Python (equivalence-gated).

The production training loop's *use* of these is the next milestone; this milestone
builds and validates the inputs.

## Ground truth (lm-100m-en, on `trig` at `~/projects/training/lm-100m-en/data/`)
- Shards: 23 `train_*.bin` + 1 `val_*.bin`, each ~954 MB, **raw headerless
  little-endian `uint16`** (nanoGPT-style; 1 token = 1 uint16). `block_size = 2048`.
- Tokenizer: `tokenizer.json` — HF **byte-level BPE** (`model.type: BPE`,
  `pre_tokenizer: ByteLevel`, **no normalizer**, vocab 32000, 31785 merges, 4 added
  tokens).

---

## `data` module

`github.com/guygrigsby/lmkit-go/data` (own `go.mod`; dep: `github.com/edsrzf/mmap-go`).
**GoMLX-free** — yields raw `int32` slices; the train milestone wraps them in tensors.

### Components
- **Shard** — mmap a `.bin` file, expose it as a `[]uint16` view (LE, `unsafe.Slice`,
  zero-copy). Length = fileSize/2 tokens. `Close()` unmaps. Lifetime: the `[]uint16`
  aliases the mapping — must not outlive `Close()` (documented; the loader owns it).
- **DataLoader** — config `{Shards []string, BlockSize, BatchSize int, Seed int64}`.
  `Next() (x, y []int32, err error)`: for each of `BatchSize`, pick a random shard +
  random start `i ∈ [0, len-BlockSize-1]`, set `x = tokens[i:i+BlockSize]`,
  `y = tokens[i+1:i+BlockSize+1]` (next-token), cast `uint16→int32`. Returns flat
  `x`,`y` each of logical shape `[BatchSize, BlockSize]`. A goroutine fills a buffered
  channel (prefetch depth = buffer; full channel = backpressure); `Close()` stops it
  and unmaps shards. Deterministic given `Seed`.
- Construct a **train** loader over the 23 `train_*.bin` and a **val** loader over
  `val_*.bin`; same type, different shard list (the train/val split is already on
  disk — no content-hash split at our layer).

### Testing
- Unit (Mac, synthetic): write tiny `.bin` files of known `uint16`, assert: token
  count, block extraction values, next-token offset (`y[k] == x[k+1]`), batch shape,
  determinism (same seed → same batches; different seed → different), boundary (no
  read past `len-BlockSize-1`), multi-shard sampling covers all shards.
- Smoke (`trig`, real shard): open one `train_*.bin`, pull a batch, assert every
  token `< 32000` and blocks are full-length. (Run on `trig`; the 24 GB of shards
  stay there.)

---

## `tokenizer` module

`github.com/guygrigsby/lmkit-go/tokenizer` (own `go.mod`; dep:
`github.com/dlclark/regexp2` — pure-Go regex with lookahead, what tiktoken-go uses).
**GoMLX-free.**

### Components
- **Load(`tokenizer.json`)** — parse: `model.vocab` (`token→id`), `model.merges`
  (ranked pairs → rank), `added_tokens` (the 4 specials: id/content/special), and the
  `pre_tokenizer` ByteLevel config (`add_prefix_space`, `trim_offsets`). Build the
  `bytes↔unicode` table (GPT-2's 256-entry reversible map) and the inverse vocab for
  decode.
- **Encode(text) []int** — byte-level BPE:
  1. ByteLevel pre-tokenization: GPT-2 split regex (via `regexp2`, needs the
     `\s+(?!\S)` lookahead RE2 lacks) + `add_prefix_space` handling; map each
     piece's UTF-8 bytes through `bytes→unicode`.
  2. BPE: apply ranked merges greedily to each piece's char sequence.
  3. Map merged tokens → ids via vocab; handle added/special tokens.
- **Decode([]int) string** — ids → token strings (inverse vocab) → concat → map
  `unicode→bytes` → UTF-8 string.

### Equivalence gate (ADR-0003, non-negotiable)
Byte-level BPE has many edge cases; "pure-Go" is only trustworthy if proven
identical to the reference. A committed fixture
`tokenizer/testdata/encodings.json` of `{text → ids}` is **generated on `trig`** by
Python HF `tokenizers` over a representative sample (varied prose, whitespace,
punctuation, unicode, leading/trailing spaces, and a slice of the real corpus). A Go
test asserts `Encode(text) == ids` for every case, and `Decode(Encode(text)) == text`
round-trips. CI runs the Go test against committed fixtures (no Python in CI).

### Testing
- The equivalence gate above (the primary gate).
- Unit: `bytes↔unicode` round-trips all 256 byte values; a hand-checked merge
  sequence; added/special-token handling; decode of known ids.

---

## Boundary / structure (ADR-0005, ADR-0009)
Both modules are pure-Go libs that import **no GoMLX** — they trivially satisfy the
runtime boundary (the `-count=1` boundary test covers them). Added to `go.work`.

```
data/
  go.mod            (+ edsrzf/mmap-go)
  shard.go   shard_test.go
  loader.go  loader_test.go
  testdata/         (tiny synthetic .bin for unit tests)
tokenizer/
  go.mod            (+ dlclark/regexp2)
  tokenizer.go      (Load, Encode, Decode)
  bytelevel.go      (bytes<->unicode, GPT-2 split regex)
  bpe.go            (ranked-merge application)
  *_test.go
  testdata/
    gen_encodings.py   (Python HF reference generator, run on trig)
    encodings.json     (committed fixture)
```

## Out of scope (later / chosen out)
- The production training loop's *use* of these (next milestone).
- BPE **training** (we load an existing vocab).
- The corpus **pull** (`data.py`) and a Go **sharder** (`shard.py`) — chose
  read-existing-shards.
- Chat templates, offsets/alignment, truncation/padding strategies (not needed to
  read pre-tokenized shards / tokenize plain text).

## Validation philosophy
- **Tokenizer:** byte-exact vs Python is the gate (the worst failure is silent
  mis-tokenization; the equivalence test makes pure-Go trustworthy).
- **Data:** next-token correctness + determinism on synthetic shards; a real-shard
  smoke confirms the format reads.

## Done criteria
- [ ] `data` + `tokenizer` modules in `go.work`; build clean; boundary clean.
- [ ] `data` DataLoader yields correct next-token `(x,y)` blocks; deterministic;
      synthetic unit tests green + a real-shard smoke on `trig`.
- [ ] `tokenizer` loads `tokenizer.json`, encode/decode; the Python equivalence gate
      passes (byte-exact) + round-trip.
- [ ] `make check` (both modules) ends OK.

## Open questions to resolve during the plan
- The exact GPT-2/HF ByteLevel split regex string + `add_prefix_space` semantics to
  match the reference (the equivalence gate is the arbiter — match the *fixture*,
  not docs).
- Mmap lifetime ergonomics: who owns `Close()` and how the loader guarantees the
  `[]uint16` view doesn't outlive the mapping (no use-after-unmap).
- Fixture breadth for the equivalence gate — enough edge cases (whitespace runs,
  combining unicode, byte-fallback paths) that passing it genuinely implies
  correctness.
</content>
