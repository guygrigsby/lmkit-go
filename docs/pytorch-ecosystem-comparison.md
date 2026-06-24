# lmkit-go vs the PyTorch + Python ML ecosystem — a parity panorama

A wide, honest comparison against the *whole* ecosystem (not just the Python
`lmkit` stack — that's the narrower view in `python-parity-gaps.md`). Grounded in
a current (2025-2026) inventory of torch ~2.9, the HF stack (transformers v5, TRL
v1, PEFT, datasets v4, tokenizers, hub v1/Xet), the kernel/quant world (FlashAttn
2/3/4, torchao, bitsandbytes, TE), the serving world (vLLM V1, SGLang, TGI), and
the data/eval ecosystems.

## How to read this — three tiers

The PyTorch ecosystem is decades of breadth. lmkit-go is a narrow vertical:
*train transformer LMs on XLA, in Go* — a composable architecture kit (arbitrary
architectures intended), with a Llama (`lm-100m-en`) as the first reproduction
target, not the ceiling. Most of the surface below is **not a gap to close**
— it falls into one of two "not ours to build" buckets:

- **⚙️ Inherited from XLA/GoMLX** — the kernel/compiler/runtime machinery (autograd
  engine, op fusion, codegen, device runtime, RNG, sharding primitives, fp8 GEMMs).
  PyTorch *builds* this; lmkit-go *rides* it. DESIGN.md's core bet: don't write
  kernels or autograd. We get it from the backend or we don't get it at all.
- **🚫 Out of scope by design** — lmkit-go is not a general framework, not a serving
  stack, not the HF hub, not a PEFT/RLHF lab, not a quantization toolkit. Huge
  swaths here (serving runtimes, the PEFT zoo, the eval ecosystem, the model hub,
  distributed frameworks) are whole products we are not trying to be.

Parity only meaningfully applies to the **third tier**:

- **⬜ Genuine user-facing gap** — provider-agnostic features a user wires directly
  that lmkit-go *could and may* add (optimizer schedules, grad-accum/clip, bf16
  wiring, checkpoint/resume, the data pipeline, tokenizer load, generation, eval).

**Status legend:** ✅ built · 🟡 partial · ⬜ gap (could add) · ⚙️ inherited from
XLA/GoMLX · 🚫 out of scope by design · 🧊 ADR-deferred (intentional).

---

## 1. Tensor core · autograd · nn · compilation

| Area | PyTorch (representative) | lmkit-go |
|---|---|---|
| Dtypes | fp32/64, fp16, **bf16**, fp8 (e4m3/e5m2), int4–64, bool, complex, quantized | ✅ fp32/bf16 (via XLA); ⚙️ others as XLA supports; 🚫 complex/quantized dtypes |
| Devices | cuda, cpu, **mps**, xpu, **xla**, meta | ✅ cuda + cpu (SimpleGo) via XLA/PJRT; 🧊 metal/rocm (ADR-0004) |
| Op surface (~2000 ATen ops), broadcasting, indexing, linalg, fft | full | ⚙️ whatever GoMLX `core/graph` exposes (we use a focused subset) |
| `torch.func` (vmap/grad/jacrev/jvp/hessian) | full | ⚙️ XLA has the machinery; 🚫 not surfaced |
| **Autograd** (reverse-mode, custom VJP, double-backward, grad-checkpoint, forward-mode) | full | ⚙️ reverse-mode **inherited from GoMLX** (this is the core bet); 🚫 we don't own the engine. *Note: a missing VJP in the backend is our problem — we hit `Rsqrt` having none and worked around it.* |
| `nn.Module` system, layer/loss zoo, `nn.init`, parametrizations, pruning | vast | 🟡 we built the **Llama** blocks (RMSNorm/RoPE/GQA/SwiGLU/embeddings) + cross-entropy; 🚫 the general module system + layer zoo is not a goal |
| `nn.utils.clip_grad_norm_` | yes | ⬜ **gap** — hand-roll global-norm clip (GoMLX has clip-by-value only) |
| `torch.compile` / Inductor / Triton / FX / export / AOTInductor / CUDA graphs | the whole compiler stack | ⚙️ **XLA *is* our compiler** — fusion/codegen/graph-opt come from it; 🚫 we don't build Dynamo/Inductor/export |

**Read:** the entire core/autograd/compile column is either ✅ (the slice we use) or
⚙️ (inherited) or 🚫 (general-framework breadth). The one real ⬜ here is
**global-norm gradient clipping**.

## 2. Optimizers · schedules · mixed precision · distributed · profiling

| Area | PyTorch / ecosystem | lmkit-go |
|---|---|---|
| Optimizer **update rules** (SGD/Adam/**AdamW**/Adafactor/Lion/Muon/…) | full zoo | ✅ **AdamW** (decoupled, the lm-100m optimizer); ⬜ others if needed (rules are small math) |
| AdamW full config (β1/β2/wd) | yes | ⬜ **gap** — wire `β=0.9/0.95, wd=0.1` through (M4 used defaults) |
| `foreach`/`fused`/`capturable` optimizer kernels | yes | ⚙️ inherited (XLA fuses the step) |
| **LR schedules** (cosine/linear/OneCycle/**WSD**/warmup/Sequential) | full zoo (WSD via HF) | ⬜ **gap** — hand-roll **WSD** (warmup→stable→decay); Python hand-rolls it too |
| Param groups / no-decay-on-norm-bias / LLRD | yes | ⬜ gap (wiring) |
| **Gradient accumulation** | manual / `accelerate` | ⬜ **gap** — `grad_accum=32`; GoMLX has `AccumulateGradients` to wire |
| EMA of weights | yes | ⬜ gap (small) |
| **AMP** (autocast + GradScaler), **bf16 training**, fp8 (TE/torchao) | full | ⬜ **bf16 training** is a gap (M4 was fp32); ⚙️ fp8 GEMMs inherited if XLA supports; 🚫 the autocast policy engine |
| Loss scaling, deterministic mode, TF32 toggles | yes | ⬜ bf16 needs no scaler (a plus); 🟡 seeding partial |
| **Distributed** (DDP, FSDP/FSDP2, TP/DTensor, PP, CP, ZeRO/DeepSpeed, Megatron) | the whole scaling stack | 🧊 single-GPU for the repro; ⚙️ XLA/GSPMD provides sharding primitives; 🚫 we don't build FSDP/DeepSpeed |
| Training frameworks (Accelerate, HF Trainer, Lightning, torchtitan, DeepSpeed) | many | 🚫 not a framework — we build *the lm-100m loop*, not a general Trainer |
| **Checkpoint / resume** (`torch.save`, DCP, async, resharding) | full | ⬜ **gap** — GoMLX `checkpoint` saves all vars incl. optimizer state (a strength); add the latest/snapshot/best rotation + resume |
| Profiling (torch.profiler/Kineto, TensorBoard, W&B, FLOP/MFU, memory snapshot) | the whole observability stack | ⬜ **metrics.jsonl** (step/loss/lr/tok_per_sec/peak_vram) for the existing Grafana; **peak_vram** via go-nvml; 🚫 the profiler internals |

**Read:** this is **the milestone**. The lm-100m replica needs the ⬜ items —
**grad-accum, global-norm clip, WSD schedule, bf16 training, AdamW-config,
checkpoint/resume, metrics.jsonl** — bundled into the production training loop.
Everything tagged ⚙️/🚫 (fused kernels, FSDP/DeepSpeed, the framework/profiler
machinery) is inherited or not-our-product.

## 3. Attention kernels · kernel authoring · quantization · memory

| Area | PyTorch / ecosystem | lmkit-go |
|---|---|---|
| SDPA + **FlashAttention 2/3/4**, xFormers, FlexAttention, paged attn | hand-written CUDA/Triton/CuTe | ✅ **cuDNN flash** via an explicit `__cudnn$fmha` custom-call (gomlx/compute/go-xla forks, 2026-06-23), parity-verified on sm_86 — the *same* kernel PyTorch's SDPA uses. Earned ~1.4x; the residual ~26x gap to PyTorch is NOT attention (it's broad — see Read below) |
| GQA/MQA/sliding-window/ALiBi/sink | kernel configs | ✅ GQA (built); ⬜ others if a model needs them |
| Kernel authoring (Triton, CUTLASS, CuTe, custom CUDA, Liger fused CE/RMSNorm/SwiGLU) | the kernel world | 🚫/🧊 we **ride XLA**; hand-written kernels deferred unless profiling forces it (ADR-0004). XLA's escape hatch (analogous to Pallas) is the someday-option |
| Quantization (torchao, bitsandbytes, GPTQ, AWQ, FP8/FP4, KV-cache quant, QAT) | enormous, kernel-heavy | 🚫 out of scope (post-training, kernel-bound); maybe GGUF *export* later for inference interop |
| Memory (activation/grad checkpointing, CPU/NVMe offload, flash-decoding, fused/chunked loss) | full | ⚙️ XLA rematerialization ≈ recompute; 🚫 offload/flash-decoding stacks |

**Read:** almost entirely ⚙️ (XLA gives fusion) / 🚫 (kernel + quantization world)
/ 🧊 (ADR-0004 deferrals).

**Measured 2026-06-23 — the "ride XLA" bet has a real cost at this scale.** Flash is
now done (same cuDNN kernel as PyTorch), yet lmkit-go still trains lm-100m **~26x
slower** than PyTorch on the same 3070 Ti (3313 ms/step vs 127 ms; ~1300 XLA kernels &
7.6 GB vs ~128 kernels & 4 GB). Ablation shows the gap is **broad**, not one op: the LM
head is 36x slower but only 22% of the step; the rest of the model is ~24x. So this row's
"⚙️ XLA gives fusion" is optimistic — XLA's compiled gemms/elementwise underperform
PyTorch's eager cuDNN/cuBLAS/ATen at 100M/B=2, and the `fused CE/RMSNorm/SwiGLU` /
kernel-authoring rows (marked 🚫) are where the real throughput lives. Closing it likely
needs op-by-op fused custom-calls in go-xla (the flash pattern) or is partly an XLA
small-scale ceiling. See `docs/specs/2026-06-22-flash-attention-design.md` Outcome.

## 4. Data · tokenization · serialization · hub · model libraries

| Area | PyTorch / ecosystem | lmkit-go |
|---|---|---|
| DataLoader (Dataset/IterableDataset/samplers/collate/workers/prefetch), HF `datasets` (Arrow/streaming), webdataset, Mosaic Streaming, LitData | rich | ⬜ **gap → next milestone** — mmap `uint16` shard DataLoader → `(x,y)` blocks (we read the existing lm-100m shards); 🚫 the Arrow/streaming/elastic-resume infra |
| LM data patterns (packing, doc-masking, deterministic/elastic mid-epoch resume) | mature | 🟡 we do next-token block extraction; 🚫 elastic-resume infra (only Mosaic/LitData do it fully) |
| **Tokenization** (HF `tokenizers`: BPE/byte-level/train/load `tokenizer.json`, normalizers, post-proc, offsets, chat templates) | full Rust lib | ⬜ **gap → next milestone** — pure-Go **byte-level BPE load + encode/decode** for the 32k `tokenizer.json`, equivalence-gated vs Python (ADR-0003); ⬜ BPE *training*, chat templates later |
| Serialization (**safetensors**, `torch.save`, GGUF, ONNX, sharded checkpoints) | full | ⬜ **safetensors** r/w (io milestone), ⬜ GGUF export; 🚫 ONNX |
| **Hub** (huggingface_hub, Xet/LFS, cache layout, model cards, `hf` CLI) | full | ⬜ **HF push** (io milestone, net/http+LFS or shell `hf`); 🚫 Xet/CAS infra, hub client breadth |
| Model libraries (`transformers` 400+ archs + generation, `timm`) | vast | 🟡 **composable transformer blocks** (arbitrary architectures intended; a new arch = a Go `Forward` + the odd block, Config-dispatched). Llama is the first/reproduction target. 🚫 no general module *registry* / `from_pretrained` hub of pretrained archs |

**Read:** **tokenizer + data are the immediate next milestone** (the two ⬜ in
bold). safetensors/GGUF/hub are the later **io** milestone. The rest (Arrow,
transformers' 400 architectures, the hub infra) is 🚫.

## 5. Generation · serving · PEFT/RLHF · evaluation

| Area | PyTorch / ecosystem | lmkit-go |
|---|---|---|
| **Generation** (greedy/beam/sampling, temp/top-k/top-p/min-p, logits processors, constrained/grammar, streaming, KV cache classes) | full `generate` | ⬜ **gap (post-repro)** — autoregressive sampling + **KV cache**; the rest (beam, grammar, the processor zoo) optional |
| Serving (vLLM, SGLang, TGI, TensorRT-LLM, paged/radix attn, continuous batching, speculative decoding, P/D, multi-LoRA) | the whole serving industry | 🚫 not a serving stack. (Trained weights export to GGUF → llama.cpp/ollama for serving) |
| PEFT (LoRA/QLoRA/DoRA + ~45 tuners) & RLHF (TRL: SFT/DPO/GRPO/PPO/reward) | huge | 🚫 mostly out of scope; 🟡 a basic **SFT** loop is plausible later (lmkit has one); the PEFT/RLHF zoo is not a goal |
| Evaluation (lm-eval-harness, lighteval, MTEB, perplexity, LLM-judge, MMLU-Pro/GPQA/SWE-bench/…) | the whole eval ecosystem | ⬜ **val-loss / perplexity** eval is the in-scope piece (it's how we hit the baseline); 🚫 the benchmark + judge ecosystem |

**Read:** generation+KV-cache is a real post-reproduction ⬜; **val-loss/perplexity
eval** is in-scope (the reproduction metric). Serving, the PEFT/RLHF zoo, and the
benchmark ecosystem are 🚫 whole-product territory.

---

## The honest scale

The PyTorch + Python ecosystem here is **hundreds of features across ~15
subsystems**, much of it kernel-deep or whole products. Mapped to lmkit-go:

- **⚙️ Inherited from XLA / not ours to build:** the autograd engine, op fusion +
  codegen, the compiler stack, fused optimizer/attention kernels, sharding
  primitives, fp8 GEMMs, RNG/dispatch internals. *This is the majority of PyTorch's
  engineering and the entire point of the core bet — we ride it.*
- **🚫 Out of scope by design:** the serving industry (vLLM/SGLang/TGI/TRT-LLM),
  the PEFT/RLHF zoo (PEFT 45 tuners, TRL), the quantization world (torchao/bnb/
  GPTQ/AWQ/FP8 kernels), the model hub + Xet, the distributed frameworks
  (FSDP/DeepSpeed/Megatron), the eval/benchmark ecosystem, `transformers`' 400
  architectures, the data-infra (Arrow/streaming). *Whole products lmkit-go is not.*
- **⬜ The actual, finite parity target** (provider-agnostic, user-facing, in-scope):
  1. **Production training loop** — grad-accum, global-norm clip, WSD schedule, bf16,
     AdamW-config, checkpoint/resume, metrics.jsonl. *(the milestone after data)*
  2. **Tokenizer** — load the 32k byte-level BPE, encode/decode (+ later: train). *(next)*
  3. **DataLoader** — mmap shard loader → blocks. *(next)*
  4. **IO** — safetensors r/w, GGUF export, HF push. *(later milestone)*
  5. **Eval + generation** — val-loss/perplexity (the baseline metric) + sampling/KV
     cache. *(reproduction + post-repro)*
  6. **CLI** — `shard`/`train`/`eval` subcommands. *(ergonomics)*

**Bottom line:** lmkit-go will never approach PyTorch-ecosystem breadth, and
shouldn't try — most of that breadth is inherited from XLA or is a separate product.
The realistic, *achievable* parity is the ~6 user-facing areas above: a
load → tokenize → shard → **train (full loop)** → eval → export → push pipeline for
transformer LMs (Llama first; the blocks compose to other architectures in Go). Of
those, the model layer is done; **training-loop + tokenizer + data** are the
near road, and **io + eval/gen** are the far road. That's the whole comparison: a
small, finite target inside an enormous ecosystem.
</content>
