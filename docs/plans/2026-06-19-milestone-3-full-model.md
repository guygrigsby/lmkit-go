# Milestone 3 — full Llama decoder + forward parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Assemble the Milestone-2 blocks into a complete config-driven Llama decoder and prove the whole forward pass matches a PyTorch reference within tolerance.

**Architecture:** A pure forward function over explicit weight Nodes (no GoMLX variables) in the existing `model` module: embeddings → N pre-norm decoder layers with residuals → final RMSNorm → tied logits. Parity tests build all weights + token ids as graph constants from a committed PyTorch golden and compare on SimpleGo. The `paritytest` harness gains an exact integer-tensor path for token ids.

**Tech Stack:** Go 1.26, GoMLX `core/graph` (pinned `516689cbe913`); reuses the M2 blocks; PyTorch (on `trig`) for goldens; SimpleGo (`-tags noxla`, `GOMLX_BACKEND=go`).

## Global Constraints

- **Runtime boundary (ADR-0009):** `model/` (incl. `*_test.go`) imports only the op vocabulary (`core/graph`, `core/tensors`, `compute/dtypes`, `compute/shapes`) + `backend/gomlx`; never the runtime (`"github.com/gomlx/compute"` root, `gomlx/backends`, `go-xla`, `pjrt`). Tests get a backend via `gomlx.New().Compute()` (type inferred — never write `compute.Backend`).
- **Pure forward, explicit weights:** no `model.Scope`/`Store`/variables; the model threads weight Nodes through the M2 blocks (unchanged).
- **Determinism:** explicit weights, seeded generator, no RNG in the model.
- **Tolerance:** pick per test for the computation depth in fp32 (decoder layer 2e-4, full forward 5e-4), set *before* running — exceed the per-seed spread so a correct impl passes for any seed. Do not loosen to mask a wiring bug, and do not seed-shop the generator to fit a tighter number (that hides the real tolerance floor).
- **PyTorch is on `trig`.** Generate goldens via the procedure below, copy JSON back, no repo on the box.
- **Commits:** terse, verb-first, no dashes, no Claude/Anthropic attribution.

## Existing M2 block signatures (reuse verbatim — all in package `model`, import `g "github.com/gomlx/gomlx/core/graph"`)

```go
func RMSNorm(x, scale *g.Node, eps float32) *g.Node          // x [B,T,H], scale [H]
func RoPE(x *g.Node, positions []int, base float64, headDim int) *g.Node
func Attention(cfg Config, x, wQ, wK, wV, wO *g.Node, positions []int) *g.Node  // x [B,T,Hidden] -> [B,T,Hidden]
func SwiGLU(x, wGate, wUp, wDown *g.Node) *g.Node
func EmbedLookup(table, tokenIDs *g.Node) *g.Node            // table [V,H], ids [B,T] int -> [B,T,H]
func TiedLogits(h, table *g.Node) *g.Node                    // h [B,T,H], table [V,H] -> [B,T,V]
```
Residual add: `g.Add(a, b)`. Constant from a tensor: `g.Const(gr, t)` where `gr` is the `*g.Graph` the exec fn receives and `t` is a `*tensors.Tensor`.

## Golden generation procedure (PyTorch on `trig`)

```bash
scp model/testdata/gen_goldens.py trig:/tmp/gen_goldens.py
ssh trig 'cd /tmp && ~/venvs/cuda/bin/python gen_goldens.py'
scp trig:/tmp/<block>.json model/testdata/
ssh trig 'rm -f /tmp/gen_goldens.py /tmp/*.json'
```
Deterministic (seeded). Copy back only the fixture(s) the current task adds.

---

### Task 1: paritytest integer path + Config.NLayers + Weights types

Scaffolding for the model: an exact integer-tensor path in the harness (resolves the M2 float32-id-truncation finding), the `NLayers` config field, and the `Weights`/`LayerWeights` structs.

**Files:**
- Modify: `model/internal/paritytest/paritytest.go`
- Test: `model/internal/paritytest/paritytest_test.go`
- Modify: `model/config.go`
- Create: `model/weights.go`

**Interfaces:**
- Produces: `paritytest.Tensor` with a `DType` tag and `[]float64` data; `Tensor.ToTensor()` returns an int32 tensor when `DType=="i32"`, else float32. `Config.NLayers int`. `model.Weights`, `model.LayerWeights`.

- [ ] **Step 1: Write the failing integer round-trip test**

`model/internal/paritytest/paritytest_test.go`:
```go
package paritytest

import (
	"testing"

	"github.com/gomlx/gomlx/core/tensors"
)

// A token id well above 2^24 must round-trip exactly through the fixture path
// (float32 would have truncated it).
func TestIntTensorRoundTrip(t *testing.T) {
	const big = 5_000_001 // > 2^22, not float32-exact in general
	tn := Tensor{Shape: []int{1, 2}, DType: "i32", Data: []float64{big, 7}}
	got := tensors.MustCopyFlatData[int32](tn.ToTensor())
	if got[0] != big || got[1] != 7 {
		t.Errorf("ids = %v, want [%d 7]", got, big)
	}
}

func TestFloatTensorDefault(t *testing.T) {
	tn := Tensor{Shape: []int{2}, Data: []float64{1.5, -2.25}} // no DType -> float32
	got := tensors.MustCopyFlatData[float32](tn.ToTensor())
	if got[0] != 1.5 || got[1] != -2.25 {
		t.Errorf("data = %v, want [1.5 -2.25]", got)
	}
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd model && go test -tags noxla ./internal/paritytest/ -run TestIntTensorRoundTrip -v`
Expected: FAIL — `Tensor` has no `DType` field / `ToTensor` returns float32 (compile error or wrong type).

- [ ] **Step 3: Update `paritytest.Tensor` for the dtype path**

In `model/internal/paritytest/paritytest.go`, replace the `Tensor` type and its `ToTensor`, and update `AssertClose` to read `[]float64` data. Keep `Fixture`, `Load` as-is except the data-type change flows through.

```go
// Tensor is a fixture tensor: row-major data plus its shape. Data is JSON-native
// float64; ToTensor casts to the tensor dtype. DType "" or "f32" => float32;
// "i32" => int32 (for token ids / indices, which must round-trip exactly — float32
// would truncate ids above 2^24).
type Tensor struct {
	Shape []int     `json:"shape"`
	DType string    `json:"dtype,omitempty"`
	Data  []float64 `json:"data"`
}

// ToTensor builds a GoMLX tensor of the fixture's dtype.
func (tn Tensor) ToTensor() *tensors.Tensor {
	if tn.DType == "i32" {
		d := make([]int32, len(tn.Data))
		for i, v := range tn.Data {
			d[i] = int32(v)
		}
		return tensors.FromFlatDataAndDimensions(d, tn.Shape...)
	}
	d := make([]float32, len(tn.Data))
	for i, v := range tn.Data {
		d[i] = float32(v)
	}
	return tensors.FromFlatDataAndDimensions(d, tn.Shape...)
}
```
Update `AssertClose` to compare against `[]float64` want data (cast each to float32 for the diff, since `got` is `[]float32`):
```go
func AssertClose(t *testing.T, got []float32, want Tensor, tol float32) {
	t.Helper()
	if len(got) != len(want.Data) {
		t.Fatalf("length mismatch: got %d, want %d", len(got), len(want.Data))
	}
	var maxDiff float32
	for i := range got {
		d := float32(math.Abs(float64(got[i]) - want.Data[i]))
		if d > maxDiff {
			maxDiff = d
		}
		if d > tol {
			t.Errorf("element %d: got %v, want %v (diff %v > tol %v)", i, got[i], want.Data[i], d, tol)
		}
	}
	t.Logf("max abs diff = %g (tol %g)", maxDiff, tol)
}
```
(The existing block fixtures store data as plain JSON numbers, which decode into `[]float64` unchanged; `ToTensor` casts back to the same float32 values, so the M2 tests stay green without regenerating any fixture.)

- [ ] **Step 4: Run the int + float tests, then the full model suite**

Run: `cd model && go test -tags noxla ./internal/paritytest/ -v`
Expected: `TestIntTensorRoundTrip`, `TestFloatTensorDefault` PASS.
Run: `cd model && GOMLX_BACKEND=go go test -tags noxla ./... -v 2>&1 | grep -E "--- (PASS|FAIL)"`
Expected: all five existing block parity tests still PASS (no fixture regen needed).

- [ ] **Step 5: Add `NLayers` to Config**

In `model/config.go`, add the field to the `Config` struct (additive):
```go
	NLayers   int
```
(Place it after `Hidden` for readability.)

- [ ] **Step 6: Add the Weights types**

`model/weights.go`:
```go
package model

import g "github.com/gomlx/gomlx/core/graph"

// LayerWeights are the weight Nodes for one decoder layer (no biases).
type LayerWeights struct {
	AttnNorm          *g.Node // [H]   RMSNorm scale before attention
	Wq, Wk, Wv, Wo    *g.Node // attention projections
	FFNNorm           *g.Node // [H]   RMSNorm scale before the FFN
	Wgate, Wup, Wdown *g.Node // SwiGLU projections
}

// Weights are the weight Nodes for the whole model. Embed is tied: it is both the
// token embedding table and (transposed, via TiedLogits) the output projection.
type Weights struct {
	Embed     *g.Node        // [V,H]
	Layers    []LayerWeights // len == Config.NLayers
	FinalNorm *g.Node        // [H]
}
```

- [ ] **Step 7: Build the model module to confirm it compiles, then commit**

Run: `cd model && go build -tags noxla ./... && cd .. && make check 2>&1 | tail -3`
Expected: builds clean; `make check` prints OK.
```bash
git add model/internal/paritytest/ model/config.go model/weights.go
git commit -m "model: integer fixture path, Config.NLayers, Weights types

paritytest carries data as float64 with a dtype tag so token ids round-trip
exactly (i32) instead of truncating through float32. Add NLayers and the
Weights/LayerWeights types for the full-model forward."
```

---

### Task 2: DecoderLayer

One pre-norm Llama decoder layer with residuals, parity-checked against a torch golden.

**Files:**
- Modify: `model/testdata/gen_goldens.py` (add shared torch helpers + `gen_decoder_layer`)
- Create: `model/testdata/decoder.json` (generated on trig)
- Create: `model/decoder.go`
- Create: `model/decoder_test.go`

**Interfaces:**
- Consumes: `RMSNorm`, `Attention`, `SwiGLU`, `Config`, `LayerWeights`, `g.Add`.
- Produces: `model.DecoderLayer(cfg Config, h *g.Node, lw LayerWeights, positions []int) *g.Node`.

- [ ] **Step 1: Add reusable torch helpers + `gen_decoder_layer` to the generator**

Append to `model/testdata/gen_goldens.py` (these module-level helpers replicate the M2 block conventions exactly and are reused by the decoder and model goldens):
```python
import torch.nn.functional as F

def _rmsnorm(x, scale, eps):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * scale

def _rope(x, hd, base):                       # x [B,T,nheads,hd]
    T = x.shape[1]
    inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    emb = torch.cat([torch.arange(T)[:, None].float() * inv[None, :]] * 2, dim=-1)
    cos = emb.cos()[None, :, None, :]; sin = emb.sin()[None, :, None, :]
    a, b = x[..., : hd // 2], x[..., hd // 2 :]
    return x * cos + torch.cat([-b, a], dim=-1) * sin

def _attention(x, w, nH, nKV, hd, base):
    B, T, _ = x.shape
    q = (x @ w["Wq"]).view(B, T, nH, hd); k = (x @ w["Wk"]).view(B, T, nKV, hd); v = (x @ w["Wv"]).view(B, T, nKV, hd)
    q = _rope(q, hd, base); k = _rope(k, hd, base)
    rep = nH // nKV
    k = k.repeat_interleave(rep, dim=2); v = v.repeat_interleave(rep, dim=2)
    q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)
    s = (q @ k.transpose(-1, -2)) / math.sqrt(hd)
    s = s + torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    o = (s.softmax(-1) @ v).transpose(1, 2).reshape(B, T, nH * hd)
    return o @ w["Wo"]

def _swiglu(x, w):
    return (F.silu(x @ w["Wg"]) * (x @ w["Wu"])) @ w["Wd"]

def _decoder_layer(h, w, cfg):
    h = h + _attention(_rmsnorm(h, w["attn_norm"], cfg["eps"]), w, cfg["nH"], cfg["nKV"], cfg["hd"], cfg["base"])
    h = h + _swiglu(_rmsnorm(h, w["ffn_norm"], cfg["eps"]), w)
    return h

def _layer_weights(H, nH, nKV, hd, ffn):
    return {"attn_norm": torch.randn(H), "Wq": torch.randn(H, nH*hd), "Wk": torch.randn(H, nKV*hd),
            "Wv": torch.randn(H, nKV*hd), "Wo": torch.randn(nH*hd, H), "ffn_norm": torch.randn(H),
            "Wg": torch.randn(H, ffn), "Wu": torch.randn(H, ffn), "Wd": torch.randn(ffn, H)}

def gen_decoder_layer():
    B, T, H, nH, nKV, hd, ffn = 2, 4, 8, 4, 2, 2, 16
    cfg = {"nH": nH, "nKV": nKV, "hd": hd, "base": 10000.0, "eps": 1e-5}
    h = torch.randn(B, T, H)
    w = _layer_weights(H, nH, nKV, hd, ffn)
    y = _decoder_layer(h, w, cfg)
    weights = {"attn_norm": w["attn_norm"], "Wq": w["Wq"], "Wk": w["Wk"], "Wv": w["Wv"],
               "Wo": w["Wo"], "ffn_norm": w["ffn_norm"], "Wgate": w["Wg"], "Wup": w["Wu"], "Wdown": w["Wd"]}
    write("decoder", {"hidden": H, "n_heads": nH, "n_kv_heads": nKV, "head_dim": hd,
                      "ffn_hidden": ffn, "rope_base": 10000.0, "rms_eps": 1e-5, "seq_len": T},
          {"h": h}, weights, y)
```
Add `gen_decoder_layer()` to the `__main__` block. Regenerate on trig (procedure above), copying back `decoder.json`:
```bash
scp model/testdata/gen_goldens.py trig:/tmp/gen_goldens.py
ssh trig 'cd /tmp && ~/venvs/cuda/bin/python gen_goldens.py'
scp trig:/tmp/decoder.json model/testdata/
ssh trig 'rm -f /tmp/gen_goldens.py /tmp/*.json'
```

- [ ] **Step 2: Write the failing DecoderLayer parity test**

`model/decoder_test.go`:
```go
package model_test

import (
	"testing"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/model"
	"github.com/guygrigsby/lmkit-go/model/internal/paritytest"
)

func TestDecoderLayerParity(t *testing.T) {
	f := paritytest.Load(t, "testdata/decoder.json")
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	cfg := model.Config{
		Hidden: int(f.Config["hidden"]), NHeads: int(f.Config["n_heads"]),
		NKVHeads: int(f.Config["n_kv_heads"]), HeadDim: int(f.Config["head_dim"]),
		FFNHidden: int(f.Config["ffn_hidden"]), RopeBase: f.Config["rope_base"],
		RMSEps: f.Config["rms_eps"],
	}
	seqLen := int(f.Config["seq_len"])
	positions := make([]int, seqLen)
	for i := range positions {
		positions[i] = i
	}
	exec := g.MustNewExec(be.Compute(), func(gr *g.Graph) *g.Node {
		lw := model.LayerWeights{
			AttnNorm: g.Const(gr, f.Weights["attn_norm"].ToTensor()),
			Wq:       g.Const(gr, f.Weights["Wq"].ToTensor()),
			Wk:       g.Const(gr, f.Weights["Wk"].ToTensor()),
			Wv:       g.Const(gr, f.Weights["Wv"].ToTensor()),
			Wo:       g.Const(gr, f.Weights["Wo"].ToTensor()),
			FFNNorm:  g.Const(gr, f.Weights["ffn_norm"].ToTensor()),
			Wgate:    g.Const(gr, f.Weights["Wgate"].ToTensor()),
			Wup:      g.Const(gr, f.Weights["Wup"].ToTensor()),
			Wdown:    g.Const(gr, f.Weights["Wdown"].ToTensor()),
		}
		h := g.Const(gr, f.Inputs["h"].ToTensor())
		return model.DecoderLayer(cfg, h, lw, positions)
	})
	out := exec.MustExec1()
	// 2e-4: a full decoder layer is ~10 matmuls + softmax + residuals in fp32; the
	// absolute parity diff varies with weight scale (5.7e-6..6.1e-5 across seeds), so
	// the bound exceeds that spread rather than fitting a lucky seed.
	paritytest.AssertClose(t, tensors.MustCopyFlatData[float32](out), f.Expected, 2e-4)
}
```
Run: `cd model && GOMLX_BACKEND=go go test -tags noxla . -run TestDecoderLayerParity -v` → FAIL (`model.DecoderLayer` undefined).

- [ ] **Step 3: Implement DecoderLayer**

`model/decoder.go`:
```go
package model

import g "github.com/gomlx/gomlx/core/graph"

// DecoderLayer is one pre-norm Llama decoder layer with residual connections:
//
//	h = h + Attention(RMSNorm(h, attnNorm))
//	h = h + SwiGLU(RMSNorm(h, ffnNorm))
//
// h is [B,T,Hidden]; positions is [0..T-1].
func DecoderLayer(cfg Config, h *g.Node, lw LayerWeights, positions []int) *g.Node {
	eps := float32(cfg.RMSEps)
	attn := Attention(cfg, RMSNorm(h, lw.AttnNorm, eps), lw.Wq, lw.Wk, lw.Wv, lw.Wo, positions)
	h = g.Add(h, attn)
	ffn := SwiGLU(RMSNorm(h, lw.FFNNorm, eps), lw.Wgate, lw.Wup, lw.Wdown)
	return g.Add(h, ffn)
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd model && GOMLX_BACKEND=go go test -tags noxla . -run TestDecoderLayerParity -v`
Expected: PASS (max abs diff well under 5e-5). If it fails, compare against the generator: the residual order (attn before ffn), the two separate RMSNorms, and the pre-norm (norm the input to each sublayer, add to the un-normed residual).

- [ ] **Step 5: Commit**

```bash
git add model/decoder.go model/decoder_test.go model/testdata/gen_goldens.py model/testdata/decoder.json
git commit -m "model: DecoderLayer (pre-norm + residual) with parity"
```

---

### Task 3: Forward (full model)

The whole forward pass: embeddings → N decoder layers → final norm → tied logits, with integer token-id input, parity-checked against a full torch reference.

**Files:**
- Modify: `model/testdata/gen_goldens.py` (add `gen_model`)
- Create: `model/testdata/model.json` (generated on trig)
- Create: `model/model.go`
- Create: `model/model_test.go`

**Interfaces:**
- Consumes: `EmbedLookup`, `DecoderLayer`, `RMSNorm`, `TiedLogits`, `Config`, `Weights`.
- Produces: `model.Forward(cfg Config, w Weights, tokenIDs *g.Node, positions []int) *g.Node`.

- [ ] **Step 1: Add `gen_model` to the generator**

Append to `model/testdata/gen_goldens.py` (reuses the helpers from Task 2):
```python
def gen_model():
    torch.manual_seed(3)  # fixed isolation seed (chosen for reproducibility, NOT to fit the tolerance)
    B, T, V, H, nL, nH, nKV, hd, ffn = 2, 5, 16, 8, 2, 4, 2, 2, 16
    cfg = {"nH": nH, "nKV": nKV, "hd": hd, "base": 10000.0, "eps": 1e-5}
    ids = torch.randint(0, V, (B, T), dtype=torch.int64)
    embed = torch.randn(V, H)
    layers = [_layer_weights(H, nH, nKV, hd, ffn) for _ in range(nL)]
    final_norm = torch.randn(H)
    h = embed[ids]
    for w in layers:
        h = _decoder_layer(h, w, cfg)
    h = _rmsnorm(h, final_norm, cfg["eps"])
    logits = h @ embed.t()
    weights = {"embed": embed, "final_norm": final_norm}
    for i, w in enumerate(layers):
        for src, dst in [("attn_norm", "attn_norm"), ("Wq", "Wq"), ("Wk", "Wk"), ("Wv", "Wv"),
                         ("Wo", "Wo"), ("ffn_norm", "ffn_norm"), ("Wg", "Wgate"), ("Wu", "Wup"), ("Wd", "Wdown")]:
            weights[f"layer{i}_{dst}"] = w[src]
    obj = {"config": {"vocab": V, "hidden": H, "n_layers": nL, "n_heads": nH, "n_kv_heads": nKV,
                      "head_dim": hd, "ffn_hidden": ffn, "rope_base": 10000.0, "rms_eps": 1e-5, "seq_len": T},
           "inputs": {"ids": {"shape": list(ids.shape), "dtype": "i32", "data": ids.flatten().tolist()}},
           "weights": {k: t2j(v) for k, v in weights.items()},
           "expected": t2j(logits)}
    with open("model.json", "w") as fp:
        json.dump(obj, fp)
    print("wrote model.json")
```
Add `gen_model()` to `__main__`. (Token ids are written with `"dtype": "i32"` and raw integer data — the integer fixture path from Task 1.) Regenerate on trig, copying back `model.json`:
```bash
scp model/testdata/gen_goldens.py trig:/tmp/gen_goldens.py
ssh trig 'cd /tmp && ~/venvs/cuda/bin/python gen_goldens.py'
scp trig:/tmp/model.json model/testdata/
ssh trig 'rm -f /tmp/gen_goldens.py /tmp/*.json'
```

- [ ] **Step 2: Write the failing Forward parity test**

`model/model_test.go`:
```go
package model_test

import (
	"fmt"
	"testing"

	g "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"

	"github.com/guygrigsby/lmkit-go/backend/gomlx"
	"github.com/guygrigsby/lmkit-go/model"
	"github.com/guygrigsby/lmkit-go/model/internal/paritytest"
)

func TestForwardParity(t *testing.T) {
	f := paritytest.Load(t, "testdata/model.json")
	be, err := gomlx.New()
	if err != nil {
		t.Fatalf("backend: %v", err)
	}
	cfg := model.Config{
		Hidden: int(f.Config["hidden"]), NLayers: int(f.Config["n_layers"]),
		NHeads: int(f.Config["n_heads"]), NKVHeads: int(f.Config["n_kv_heads"]),
		HeadDim: int(f.Config["head_dim"]), FFNHidden: int(f.Config["ffn_hidden"]),
		VocabSize: int(f.Config["vocab"]), RopeBase: f.Config["rope_base"], RMSEps: f.Config["rms_eps"],
	}
	seqLen := int(f.Config["seq_len"])
	positions := make([]int, seqLen)
	for i := range positions {
		positions[i] = i
	}
	exec := g.MustNewExec(be.Compute(), func(gr *g.Graph) *g.Node {
		c := func(name string) *g.Node { return g.Const(gr, f.Weights[name].ToTensor()) }
		w := model.Weights{Embed: c("embed"), FinalNorm: c("final_norm")}
		for i := 0; i < cfg.NLayers; i++ {
			p := fmt.Sprintf("layer%d_", i)
			w.Layers = append(w.Layers, model.LayerWeights{
				AttnNorm: c(p + "attn_norm"), Wq: c(p + "Wq"), Wk: c(p + "Wk"), Wv: c(p + "Wv"),
				Wo: c(p + "Wo"), FFNNorm: c(p + "ffn_norm"),
				Wgate: c(p + "Wgate"), Wup: c(p + "Wup"), Wdown: c(p + "Wdown"),
			})
		}
		ids := g.Const(gr, f.Inputs["ids"].ToTensor())
		return model.Forward(cfg, w, ids, positions)
	})
	out := exec.MustExec1()
	// 5e-4: the full forward stacks N decoder layers (each ~10 matmuls + softmax) +
	// final norm + tied projection in fp32, so absolute accumulation is deeper than a
	// single layer. This bound exceeds the per-seed spread — a correct impl passes for
	// ANY seed. Do NOT seed-shop the generator to fit a tighter number; pick the
	// tolerance for the computation depth, not for a lucky weight set.
	paritytest.AssertClose(t, tensors.MustCopyFlatData[float32](out), f.Expected, 5e-4)
}
```
Run: `cd model && GOMLX_BACKEND=go go test -tags noxla . -run TestForwardParity -v` → FAIL (`model.Forward` undefined).

- [ ] **Step 3: Implement Forward**

`model/model.go`:
```go
package model

import g "github.com/gomlx/gomlx/core/graph"

// Forward runs the full Llama decoder: token embedding -> NLayers pre-norm decoder
// layers -> final RMSNorm -> tied logits. tokenIDs is [B,T] int32; positions is
// [0..T-1]. Returns logits [B,T,VocabSize]. Pure over the weight Nodes in w; no
// variables or RNG.
func Forward(cfg Config, w Weights, tokenIDs *g.Node, positions []int) *g.Node {
	h := EmbedLookup(w.Embed, tokenIDs) // [B,T,H]
	for i := range w.Layers {
		h = DecoderLayer(cfg, h, w.Layers[i], positions)
	}
	h = RMSNorm(h, w.FinalNorm, float32(cfg.RMSEps))
	return TiedLogits(h, w.Embed) // [B,T,V]
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd model && GOMLX_BACKEND=go go test -tags noxla . -run TestForwardParity -v`
Expected: PASS (max abs diff under 5e-5). If it fails: the tied-logits step uses the same `Embed` table; the final norm is applied before logits; layer order matches the generator. Debug by shape/intermediate comparison, not by loosening tolerance.

- [ ] **Step 5: Run the full model suite + gate**

Run: `cd model && GOMLX_BACKEND=go go test -tags noxla ./... -v 2>&1 | grep -E "--- (PASS|FAIL)"` then `make check`
Expected: all parity tests (RMSNorm, RoPE, SwiGLU, embedding, attention, decoder-layer, forward) + the paritytest unit tests pass; `make check` prints OK.

- [ ] **Step 6: Commit**

```bash
git add model/model.go model/model_test.go model/testdata/gen_goldens.py model/testdata/model.json
git commit -m "model: full Forward (decoder stack) with whole-forward parity"
```

---

## Self-Review

**Spec coverage** (against `2026-06-19-milestone-3-full-model-design.md`):
- `Config.NLayers` + Forward reads full Config → Task 1 (NLayers) + Task 3 (Forward uses cfg fields).
- `Weights`/`LayerWeights` → Task 1.
- `DecoderLayer` (pre-norm + residual) → Task 2.
- `Forward` (embed → N layers → norm → tied logits) → Task 3.
- integer-tensor path in paritytest → Task 1 (dtype tag, round-trip test).
- DecoderLayer parity test → Task 2; Forward parity test → Task 3.
- hand-written torch reference reusing shared helpers → Task 2 helpers reused in Task 3 (`_decoder_layer`, `_rmsnorm`, etc.), guaranteeing the full-model golden uses the conventions the blocks were parity-checked against.
- make check / boundary clean → Tasks 1 and 3.

**Out of scope, correctly absent:** training/variables/Store, KV cache, generation, real config/weights, FlashAttention.

**Placeholder scan:** all Go and Python is complete; no vague steps. The integer path is concrete (dtype tag + float64 storage); the parity tests build all weights as graph constants (no variadic-exec issue).

**Type consistency:** `model.Config` (Hidden/NLayers/NHeads/NKVHeads/HeadDim/FFNHidden/VocabSize/RopeBase/RMSEps), `Weights{Embed, Layers, FinalNorm}`, `LayerWeights{AttnNorm, Wq, Wk, Wv, Wo, FFNNorm, Wgate, Wup, Wdown}`, `DecoderLayer`/`Forward` signatures, and the fixture weight key names (`layer{i}_Wq`, `attn_norm`, `Wgate`…) match across the generator, the Go types, and both tests. `paritytest.Tensor.DType`/`ToTensor` are used identically in Tasks 2 and 3.
</content>
