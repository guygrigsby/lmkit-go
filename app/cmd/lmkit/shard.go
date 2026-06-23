package main

import (
	"bufio"
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/guygrigsby/lmkit-go/tokenizer"
)

// shardCmd implements `lmkit shard`: tokenize gzipped-JSONL documents into raw little-endian
// uint16 token shards (the format data.Loader memory-maps). Each document's tokens are followed
// by the end-of-text token so the model sees document boundaries. The first --val-tokens tokens
// go to val_00000.bin; the rest fill train_NNNNN.bin shards of --shard-tokens each.
//
// Usage: lmkit shard --tokenizer tok.json --out dir [flags] file1.jsonl.gz file2.jsonl.gz ...
func shardCmd(args []string) int {
	fs := flag.NewFlagSet("shard", flag.ContinueOnError)
	tokPath := fs.String("tokenizer", "", "path to tokenizer.json (required)")
	outDir := fs.String("out", "", "output shard directory (required)")
	textField := fs.String("text-field", "text", "JSON field holding the document text")
	shardTokens := fs.Int("shard-tokens", 100_000_000, "tokens per train shard")
	valTokens := fs.Int("val-tokens", 10_000_000, "tokens held out to val_00000.bin")
	eotID := fs.Int("eot", 0, "end-of-text token id appended after each document")
	if err := fs.Parse(args); err != nil {
		return 2
	}
	if *tokPath == "" || *outDir == "" || fs.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "usage: lmkit shard --tokenizer tok.json --out dir file.jsonl.gz ...")
		return 2
	}

	tok, err := tokenizer.Load(*tokPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "FAIL: load tokenizer: %v\n", err)
		return 1
	}
	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "FAIL: mkdir: %v\n", err)
		return 1
	}

	w := &shardWriter{dir: *outDir, shardTokens: *shardTokens, valTokens: *valTokens, eot: uint16(*eotID)}
	start := time.Now()
	for fi, path := range fs.Args() {
		if err := w.tokenizeFile(path, tok, *textField); err != nil {
			fmt.Fprintf(os.Stderr, "FAIL: %s: %v\n", path, err)
			return 1
		}
		fmt.Printf("[%d/%d] %s done | docs=%d tokens=%d shards=%d elapsed=%s\n",
			fi+1, fs.NArg(), filepath.Base(path), w.docs, w.total, w.trainShards, time.Since(start).Round(time.Second))
	}
	if err := w.flush(); err != nil {
		fmt.Fprintf(os.Stderr, "FAIL: flush: %v\n", err)
		return 1
	}
	fmt.Printf("done: %d docs, %d tokens, %d train shards + 1 val shard in %s\n",
		w.docs, w.total, w.trainShards, time.Since(start).Round(time.Second))
	return 0
}

// shardWriter accumulates uint16 tokens and flushes fixed-size shards. The first valTokens go to
// a val shard; the rest fill numbered train shards. Shards are written incrementally so a crash
// loses only the in-progress shard.
type shardWriter struct {
	dir         string
	shardTokens int
	valTokens   int
	eot         uint16

	buf         []uint16
	wroteVal    bool
	trainShards int
	docs        int64
	total       int64
}

func (w *shardWriter) tokenizeFile(path string, tok *tokenizer.Tokenizer, field string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return err
	}
	defer gz.Close()

	sc := bufio.NewScanner(gz)
	sc.Buffer(make([]byte, 1<<20), 64<<20) // documents can be long
	for sc.Scan() {
		var rec map[string]json.RawMessage
		if err := json.Unmarshal(sc.Bytes(), &rec); err != nil {
			continue // skip malformed lines rather than abort a multi-hour run
		}
		raw, ok := rec[field]
		if !ok {
			continue
		}
		var text string
		if err := json.Unmarshal(raw, &text); err != nil || text == "" {
			continue
		}
		ids := tok.Encode(text)
		for _, id := range ids {
			w.buf = append(w.buf, uint16(id))
		}
		w.buf = append(w.buf, w.eot)
		w.docs++
		w.total += int64(len(ids)) + 1
		if err := w.maybeFlush(); err != nil {
			return err
		}
	}
	return sc.Err()
}

// maybeFlush writes the val shard once valTokens have accumulated, then full train shards.
func (w *shardWriter) maybeFlush() error {
	if !w.wroteVal {
		if len(w.buf) >= w.valTokens {
			if err := w.writeShard("val_00000.bin", w.buf[:w.valTokens]); err != nil {
				return err
			}
			w.buf = append(w.buf[:0], w.buf[w.valTokens:]...)
			w.wroteVal = true
		}
		return nil
	}
	for len(w.buf) >= w.shardTokens {
		name := fmt.Sprintf("train_%05d.bin", w.trainShards)
		if err := w.writeShard(name, w.buf[:w.shardTokens]); err != nil {
			return err
		}
		w.trainShards++
		w.buf = append(w.buf[:0], w.buf[w.shardTokens:]...)
	}
	return nil
}

// flush writes whatever remains: the val shard if it never filled, plus a final train shard.
func (w *shardWriter) flush() error {
	if !w.wroteVal {
		if err := w.writeShard("val_00000.bin", w.buf); err != nil {
			return err
		}
		return nil
	}
	if len(w.buf) > 0 {
		name := fmt.Sprintf("train_%05d.bin", w.trainShards)
		if err := w.writeShard(name, w.buf); err != nil {
			return err
		}
		w.trainShards++
	}
	return nil
}

func (w *shardWriter) writeShard(name string, toks []uint16) error {
	path := filepath.Join(w.dir, name)
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	bw := bufio.NewWriterSize(f, 1<<20)
	if err := binary.Write(bw, binary.LittleEndian, toks); err != nil {
		f.Close()
		return err
	}
	if err := bw.Flush(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}
