# Milestone 2 — Llama block parity (spec)

**Status:** ready for plan. Sub-project 2 of the context map
(`lmkit-go-context-map.md`). Decisions: ADR-0001..0009 (esp. ADR-0009, the runtime
boundary).

## Goal

Implement the Llama building blocks in Go (GoMLX ops) and prove each matches a
PyTorch reference within tight tolerance. No full-model assembly, no training.

The blocks: **RMSNorm, RoPE, GQA attention, SwiGLU FFN, and embeddings (tied)**.
Each is config-driven, takes explicit weights (no internal RNG so parity is
deterministic), and passes a layer-parity test against committed golden fixtures.

## In scope

- New **`model` module** (own `go.mod`, ADR-0005), depending on `backend` and
  GoMLX `core/graph` (pragmatic runtime boundary, ADR-0009). Added to `go.work`.
- `model.Config` — block hyperparameters: `Hidden, NHeads, NKVHeads, HeadDim,
  FFNHidden, RopeBase float64, RMSEps float64, VocabSize`.
- One file per block, each a pure function from (config, weights, input) to output,
  built in `core/graph` ops:
  - `rmsnorm.go`, `rope.go`, `attention.go`, `swiglu.go`, `embedding.go`.
- A parity harness: a committed PyTorch generator + committed JSON golden fixtures
  + a Go parity test per block.
- The updated boundary test (ADR-0009) extended to cover the `model` module: the
  runtime packages appear only under `backend/gomlx/`.

## Out of scope (later milestones)

- Assembling the full decoder stack + whole-model forward parity (Milestone 3).
- Training, optimizer use on the model, KV cache, generation.
- **FlashAttention** — attention is the explicit `softmax(QKᵀ/√d)·V`; the XLA-graph
  flash variant (ADR-0004) is a later perf milestone.
- The real lm-100m-en config and real weights (safetensors loading is Milestone 6).
- bf16 training validation — fp32 parity here; a bf16 tolerance smoke is optional.

## The blocks (clean-room; gemma + the Llama paper as correctness references)

Conventions match the PyTorch reference that generates the goldens (below); where
Llama has a convention choice, the reference uses the canonical HF Llama one.

- **RMSNorm** — `y = x * rsqrt(mean(x², lastAxis) + eps) * scale`. The mean/rsqrt
  accumulate in **fp32** even when activations are bf16 (DESIGN requirement);
  `scale` is a learned `[hidden]` vector. Config: `Hidden, RMSEps`.
- **RoPE** — standard Llama rotary embedding, **rotate-half** convention (split the
  head dim in half, not interleaved). `inv_freq[i] = RopeBase^(-2i/HeadDim)`;
  `cos`/`sin` from integer positions; applied to Q and K. Config: `HeadDim,
  RopeBase`.
- **GQA attention** — input `x [B,T,Hidden]` and projection weights
  `Wq [Hidden, NHeads*HeadDim]`, `Wk,Wv [Hidden, NKVHeads*HeadDim]`,
  `Wo [NHeads*HeadDim, Hidden]`. Project, reshape to heads, apply RoPE to Q/K,
  repeat KV heads to `NHeads` (group size `NHeads/NKVHeads`), scores
  `QKᵀ/√HeadDim`, additive causal mask (−inf above the diagonal), softmax (fp32),
  `·V`, merge heads, output projection. Config: `Hidden, NHeads, NKVHeads, HeadDim`.
- **SwiGLU FFN** — weights `Wgate, Wup [Hidden, FFNHidden]`, `Wdown [FFNHidden,
  Hidden]`; `y = (SiLU(x·Wgate) * (x·Wup)) · Wdown`. Config: `Hidden, FFNHidden`.
- **Embeddings (tied)** — table `E [VocabSize, Hidden]`. Lookup: gather rows by token
  id. Tied logits: `logits = h · Eᵀ`. Two small exported functions
  (`EmbedLookup`, `TiedLogits`). Config: `VocabSize, Hidden`.

Each block is an exported Go function taking `*graph.Graph` nodes for input +
weights and returning an output node — no variable creation, no RNG. Weight
*management* (variables, init) is a training concern, deferred to M3/train.

## Parity harness

1. **Generator** — `model/testdata/gen_goldens.py` (PyTorch, committed). For each
   block: fix a small config and a seed, build the block in torch with explicit
   random weights + input, run the forward, and write a JSON fixture
   `model/testdata/<block>.json` containing `{config, inputs{name:{shape,data}},
   weights{name:{shape,data}}, expected{shape,data}}` (float32, row-major). The
   torch reference defines the canonical convention each Go block must match. The
   script is deterministic (seeded) and re-run only when a block changes.
2. **Go parity test** — `<block>_test.go` loads the JSON, builds `core/graph` input
   + weight nodes from the fixture, runs the Go block through a `backend` executor
   on **SimpleGo** (`-tags noxla`, fp32), reads the output, and asserts elementwise
   `|got - expected| <= 1e-5` (absolute; `5e-5` fallback if a block's accumulation
   needs it — pick the tightest that's stable, documented per block). An optional
   bf16 run asserts `~1e-3`.
3. JSON loading is a tiny shared test helper in the `model` test code (stdlib
   `encoding/json`); fixtures are small (tiny configs), so committing them is cheap.

The `backend` interface gains whatever minimal executor entry the parity tests need
to run an arbitrary `core/graph` function and read results (additive, ADR-0008) —
or the test uses GoMLX exec directly *within the model module's test files*, which
is allowed (model may use the op vocabulary; tests are not the runtime boundary).
The plan resolves which; prefer reusing `backend` if it's a one-method add.

## Validation philosophy (DESIGN, non-negotiable)

- **Layer parity:** every block matches the torch golden within tolerance — this
  milestone IS that gate. Fixtures committed so CI needs no Python.
- Deterministic: explicit weights + seeded generator → reproducible, no flakiness.

## Structure

```
model/
  go.mod            (module .../model; requires backend, gomlx core/graph)
  config.go         (model.Config)
  rmsnorm.go  rmsnorm_test.go
  rope.go     rope_test.go
  attention.go attention_test.go
  swiglu.go   swiglu_test.go
  embedding.go embedding_test.go
  internal/paritytest/   (tiny JSON-fixture loader shared by the block tests)
  testdata/
    gen_goldens.py
    rmsnorm.json rope.json attention.json swiglu.json embedding.json
```

Each block file has one responsibility and is testable in isolation.
`model/internal/` is fine — it is internal to the `model` module, not the
lmkit-specific `app` internal (ADR-0005 reserves `internal/` in `app`; a module's
own `internal/` for shared test helpers is normal Go and not lmkit glue).

## Done criteria

- [ ] `model` module in `go.work`; builds clean.
- [ ] `gen_goldens.py` produces committed JSON fixtures for all five blocks.
- [ ] Each block has a parity test green on SimpleGo at the documented tolerance.
- [ ] Boundary test (ADR-0009) covers `model`: runtime packages only in
      `backend/gomlx/`; `model` uses only the op vocabulary.
- [ ] `make check` (extended to the `model` module) ends OK.

## Open questions to resolve during the plan

- Confirm PyTorch is available where the generator runs (`python -c "import torch"`)
  — the plan's first task verifies this and records the environment.
- Exact RoPE convention parity (rotate-half vs interleaved) — the generator pins it;
  the Go block must match the *generator*, verified by the fixture, not by external
  docs.
- Whether the parity tests run the graph via a small additive `backend` executor
  method or via GoMLX exec inside the model test files (both allowed under ADR-0009;
  pick the lighter).
- Attention numerical tolerance — softmax+matmul accumulation may need `5e-5` rather
  than `1e-5`; document the chosen per-block tolerance and why.
</content>
