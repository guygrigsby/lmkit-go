package data

import (
	"fmt"
	"math/rand"
	"sync"
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
	cfg        Config
	shards     []*Shard
	rng        *rand.Rand
	ch         chan Batch
	done       chan struct{}
	stopped    chan struct{}
	closeOnce  sync.Once
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

// Next returns the next prefetched batch. Must not be called after Close.
func (l *Loader) Next() Batch { return <-l.ch }

// Close stops prefetching and unmaps shards. It waits for the fill goroutine to
// exit before unmapping, so no batch is built from an unmapped shard.
// Safe to call more than once; subsequent calls are no-ops and return nil.
// Do not call Next after Close.
func (l *Loader) Close() error {
	var err error
	l.closeOnce.Do(func() {
		close(l.done)
		<-l.stopped
		for _, s := range l.shards {
			if e := s.Close(); e != nil {
				err = e
			}
		}
	})
	return err
}
