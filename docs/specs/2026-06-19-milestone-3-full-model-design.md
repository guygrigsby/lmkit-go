# Milestone 3 — full Llama decoder + forward parity (spec)

**Status:** ready for plan. Sub-project 3 of the context map. Decisions:
ADR-0001..0009.

## Goal

Assemble the Milestone-2 blocks into a complete, config-driven Llama decoder and
prove the whole forward pass matches a PyTorch reference within tolerance. No
training, no KV cache, no generation.

## In scope

In the **existing `model` module** (new files, not a new module). The model is a
**pure forward over explicit weights** — blocks stay as written in M2; no GoMLX
variables or `model.Scope`/`Store` (the training milestone introduces those).

- **`Weights` / `LayerWeights`** — named weight Nodes for the whole model.
- **`model.Config`** gains `NLayers`. `Forward` reads the full Config — this is where
  `RMSEps`, `FFNHidden`, `VocabSize` (declared in M2, previously unread) become used
  at the model level. The M2 block signatures (loose params) are unchanged; `Forward`
  and `DecoderLayer` pass the Config fields through.
- **`DecoderLayer`** — one pre-norm Llama decoder layer.
- **`Forward`** — embeddings → N decoder layers → final norm → tied logits.
- **Integer-tensor path** in `paritytest` so token ids round-trip exactly (resolves
  the deferred M2 finding I1; no more float32-cast workaround).
- A hand-written PyTorch full-model generator (same conventions as the M2 block
  references) + committed JSON goldens + Go parity tests.

## Out of scope (later milestones)

- Training, optimizer, GoMLX variables/`Store` (training milestone).
- KV cache, sampling/generation, autoregressive decode.
- Real lm-100m-en config and real weights; safetensors loading (Milestone 6).
- FlashAttention; bf16 training validation.

## Components

```go
// LayerWeights are the weight Nodes for one decoder layer.
type LayerWeights struct {
	AttnNorm          *graph.Node // [H]   RMSNorm scale before attention
	Wq, Wk, Wv, Wo    *graph.Node // attention projections
	FFNNorm           *graph.Node // [H]   RMSNorm scale before the FFN
	Wgate, Wup, Wdown *graph.Node // SwiGLU projections
}

// Weights are the weight Nodes for the whole model. Embed is tied: it is both the
// token embedding table and (transposed) the output projection.
type Weights struct {
	Embed     *graph.Node    // [V,H]
	Layers    []LayerWeights // len == Config.NLayers
	FinalNorm *graph.Node    // [H]
}
```

- **`DecoderLayer(cfg Config, h *graph.Node, lw LayerWeights, positions []int) *graph.Node`**
  — pre-norm with residuals:
  - `h = h + Attention(cfg, RMSNorm(h, lw.AttnNorm, cfg.RMSEps), lw.Wq, lw.Wk, lw.Wv, lw.Wo, positions)`
  - `h = h + SwiGLU(RMSNorm(h, lw.FFNNorm, cfg.RMSEps), lw.Wgate, lw.Wup, lw.Wdown)`
- **`Forward(cfg Config, w Weights, tokenIDs *graph.Node, positions []int) *graph.Node`**:
  - `h = EmbedLookup(w.Embed, tokenIDs)`            `[B,T,H]`
  - `for i := range w.Layers { h = DecoderLayer(cfg, h, w.Layers[i], positions) }`
  - `h = RMSNorm(h, w.FinalNorm, cfg.RMSEps)`
  - `return TiedLogits(h, w.Embed)`                 `[B,T,V]`

Standard Llama: pre-norm, residual connections, tied input/output embeddings, no
biases. `positions` is `[0..T-1]` for the dense forward.

## Parity harness extension (integer tensors)

`paritytest` currently stores all fixture tensors as `[]float32`. Add an exact
integer path so token ids (and any index tensors) survive the JSON round-trip
without float32 truncation:

- carry integer fixture tensors as `int32` (or `int64`) end to end — e.g. a dtype
  tag on `Tensor` (`"f32"` vs `"i32"`) or a sibling `IntTensor` type with
  `ToTensor()` producing an int32 GoMLX tensor;
- the generator writes token ids via this integer path;
- `gen_embedding` (which currently bypasses the shared `write()` helper for its
  bespoke schema) can rejoin a uniform path once integers are first-class — fold
  that in if cheap, otherwise leave it.

The plan picks the exact shape; the requirement is that an id of any realistic
vocab size (well past 2^24) round-trips exactly.

## Tests

- **`DecoderLayer` parity** — one layer, given `h` + one `LayerWeights` + positions,
  matches a torch golden. Intermediate check that the residual/pre-norm wiring is
  right before stacking.
- **`Forward` parity (the milestone gate)** — a small multi-layer model (e.g.
  `NLayers=2`, tiny dims) with integer token-id input matches a full torch
  reference forward, output `[B,T,V]`.

Both fp32 on SimpleGo (`-tags noxla`, `GOMLX_BACKEND=go`). Tolerance documented per
test; expect ~5e-5 (deepest accumulation chain so far — embeddings through N
attention+FFN layers, final norm, tied projection). Goldens generated on `trig`
(PyTorch, per the M2 procedure), committed so CI needs no Python.

## Structure

```
model/
  config.go         (+ NLayers)
  weights.go        (Weights, LayerWeights)
  decoder.go   decoder_test.go     (DecoderLayer)
  model.go     model_test.go       (Forward)
  internal/paritytest/paritytest.go (+ integer-tensor path)
  testdata/
    gen_goldens.py  (+ gen_decoder_layer, gen_model)
    decoder.json model.json
```

## Validation philosophy (DESIGN, non-negotiable)

- Whole-forward parity vs a torch reference is this milestone's gate.
- Deterministic: explicit weights, seeded generator, no RNG in the model.

## Done criteria

- [ ] `model.Config` has `NLayers`; `Forward` reads the full Config.
- [ ] `Weights`/`LayerWeights`, `DecoderLayer`, `Forward` implemented.
- [ ] Integer-tensor path in `paritytest`; token-id goldens round-trip exactly.
- [ ] `DecoderLayer` parity test green.
- [ ] `Forward` parity test green (full multi-layer model vs torch).
- [ ] `make check` (model included) ends OK; boundary still clean (ADR-0009).

## Open questions to resolve during the plan

- Exact integer-tensor representation in `paritytest` (dtype tag vs sibling type) —
  pick the lighter; must round-trip large ids exactly.
- Per-test tolerance for the full forward (start 5e-5; tighten if the actual diff is
  far under, document the choice; do not loosen to mask a wiring bug).
- Whether the torch full-model reference shares helper functions with the per-block
  references in `gen_goldens.py` (preferred — one RoPE/attention definition reused)
  to guarantee the full-model golden uses the same conventions the blocks were
  parity-checked against.
</content>
