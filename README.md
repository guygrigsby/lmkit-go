# lmkit-go

> The Go sibling to [lmkit](https://github.com/guygrigsby/lmkit). Aeryx/`guygrigsby` (research), MIT.

From-scratch LLM training in pure Go, on an XLA backend. The ergonomic Go training
toolkit the Go ecosystem is missing, built *on top of* a real compiler (XLA via
[GoMLX](https://github.com/gomlx/gomlx)) instead of reinventing kernels or autograd.

The core bet: don't write the math. Models, the training loop, data, tokenization,
checkpointing, and eval live in Go; XLA owns the kernels, fusion, autodiff, and the
hardware (CPU, CUDA, and whatever PJRT plugins exist). Reimplementing PyTorch in Go
loses to torch/JAX/Candle; the unfilled contribution is the Go layer above the
compiler. Rationale in [`DESIGN.md`](./DESIGN.md) and [`docs/adr/`](./docs/adr).

**Not** a replacement for the Python/PyTorch stack ([lmkit](https://github.com/guygrigsby/lmkit));
that keeps doing the real research training. This is a parallel, community-facing project.

## Status

Working end to end. The from-scratch Llama is parity-tested against PyTorch goldens,
the training loop trains, and the target is reproducing a published tiny-Llama
baseline (`Aeryx-ai/lm-100m-en-overtrained`, val loss 1.7337) in pure Go-on-XLA.
See [`examples/lm-100m-en`](./examples/lm-100m-en).

## The stack

Each package is its own Go module so the reusable pieces are testable and consumable
on their own ([ADR-0005](./docs/adr/0005-per-package-modules.md)).

- **`backend`**: anti-corruption layer over GoMLX/XLA. Nothing else in the repo
  imports the compute stack directly; the boundary is the only seam to the vendor.
- **`model`**: Llama building blocks: RMSNorm, RoPE, grouped-query attention,
  SwiGLU, tied embeddings. fp32 parity-tested against PyTorch, with an fp32-internal
  norm/softmax and a bf16 compute path for CUDA.
- **`data`**: memory-mapped uint16 token shards with a prefetching, deterministic
  batch loader.
- **`tokenizer`**: ByteLevel BPE, loads HuggingFace `tokenizer.json` and round-trips
  against a committed Python fixture.
- **`train`**: the loop: WSD learning-rate schedule, gradient accumulation,
  global-norm clipping, decoupled AdamW (weight decay on rank>=2 params), bf16 on
  CUDA / fp32 on CPU, periodic eval, checkpoint + resume, JSONL metrics, signal
  handling, and optional gradient checkpointing to fit deep models on small GPUs.
- **`app`**: the `lmkit` CLI.

## Quick start

Build the CLI:

```bash
go build -o lmkit ./app/cmd/lmkit
```

Prove the backend (matmul, autodiff, an AdamW fit) and print the selected device:

```bash
./lmkit quickstart
```

Train, pointing at a config, a model definition, and a shard directory:

```bash
./lmkit train \
  --config examples/lm-100m-en/config.json \
  --model  examples/lm-100m-en/model.json \
  --data   <shard-dir>
```

`train.Run` resumes from `latest/` under `out_dir` on restart, exits 0 on
completion or SIGTERM, and 2 on non-finite loss. Wrap it in a process supervisor on
a CUDA host for a durable run. Progress streams to `metrics.jsonl`, one event per
line (`start`/`resume`, `train`, `eval`, `done`).

## Backend selection

The device is chosen at runtime via `GOMLX_BACKEND`:

- `go`: SimpleGo, pure-Go CPU backend (no XLA C dependency; what `make check` uses).
- `xla:cpu`: XLA on CPU.
- `xla:cuda`: XLA on an NVIDIA GPU; go-xla fetches the PJRT CUDA plugin on first run.

## Docs

- [`DESIGN.md`](./DESIGN.md): the brief and the architectural bet.
- [`docs/architecture.md`](./docs/architecture.md): system overview.
- [`docs/adr/`](./docs/adr): decision records (build on GoMLX/XLA, the backend
  boundary, per-package modules, the runtime seam).
- [`docs/specs/`](./docs/specs): per-milestone designs.

MIT.
