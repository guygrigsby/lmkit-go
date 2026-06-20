// Package data reads tokenized uint16 shards and serves next-token training
// batches. Pure Go, no GoMLX (ADR-0005): it yields raw uint16 slices; the train
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
