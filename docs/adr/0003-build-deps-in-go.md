# 3. Build dependencies in Go, not shell-outs

Status: accepted (2026-06-19)

## Context

Ecosystem research (mid-2026) found that several pieces have no mature Go library
and an easy shell-out: train the BPE in Python HF `tokenizers`; export GGUF via
llama.cpp; push to the HF Hub via the `hf` CLI. That is the pragmatic path for a
consumer project.

lmkit-go is not a consumer project. The whole point is the Go-native toolkit — the
gap in the ecosystem *is* these missing dependencies. Shelling out to Python /
llama.cpp / the `hf` CLI reintroduces the runtimes we exist to remove and makes the
"pure Go" story false.

## Decision

Build the missing dependencies in Go:
- pure-Go **BPE trainer** (and `tokenizer.json` loader), not train-in-Python.
- **safetensors** read/write (the format is a JSON header + raw tensor bytes;
  ~200 lines).
- **GGUF** writer for F16/F32 export.
- **HF Hub uploader** over `net/http` + the Git-LFS protocol.
- the **DataLoader** (mmap shards + `(x,y)` block batches + prefetch channel).
- **FlashAttention** as an XLA op-graph (tiled online-softmax), not a custom
  kernel (kernels: ADR-0004).

Use mature off-the-shelf Go libraries where they exist and are sound (e.g.
`edsrzf/mmap-go` for mmap). Own everything central.

Two pragmatic carve-outs, documented not hidden:
- GGUF **quantization** (Q4_K_M etc.) is llama.cpp kernel territory; export F16
  from Go and shell to `llama-quantize` only for quantized variants.
- Byte-identical tokenization: the pure-Go BPE must pass an equivalence gate
  against HF `tokenizers` on the real corpus before it is trusted; until it does,
  the cgo `daulet/tokenizers` loader is the verified reference for that check.

## Consequences

- A larger build surface than a consumer project — but that surface *is* the
  product, and most pieces are small (safetensors, dataloader, hashing-split).
- Single static binary, no Python/CUDA-Python runtime, predictable memory.
- The tokenizer equivalence gate is mandatory; a vocab that silently diverges from
  HF semantics is the worst failure mode and must fail CI.
- Each owned dependency is its own sub-project with its own spec and tests.
</content>
