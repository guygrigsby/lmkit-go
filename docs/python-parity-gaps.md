# lmkit-go — feature parity gaps vs the Python lmkit / PyTorch ecosystem

What lmkit-go must add to mirror the Python `lmkit` surface (its stated parity
target) and, beyond that, the broader PyTorch/HF ecosystem. Grounded in the
`lmkit` package layout (`pretrain`, `training`, `shard`, `tokenizer`, `sft`,
`eval/{gen,judge}`, `observability`, `push`, `cli`), the `lm-100m-en` config, and
the mid-2026 Go-vs-Python ecosystem research.

**Status legend:** ✅ built · 🟡 partial · ⬜ missing · 🧊 deferred by ADR (intentional)

## Built so far (Milestones 1–4)
- ✅ **backend** — XLA via GoMLX, CUDA-validated; autodiff; AdamW; bf16 dtype.
- ✅ **model** — Llama blocks (RMSNorm, RoPE, GQA, SwiGLU, tied emb) + full decoder
  forward, all parity-verified vs PyTorch. Config matches lm-100m-en exactly.
- 🟡 **train** — overfit-a-batch loop only (AdamW + cross-entropy over one batch).
  The production training loop is the big gap below.

---

## A. Critical path to the lm-100m-en reproduction (Chinchilla-budget baseline)
These gate the replica train. lm-100m-en config values in parens.

- ⬜ **DataLoader** — mmap `uint16` shards → `(x,y)` block batches (`block_size=2048`).
  *(next milestone — data module)*
- ⬜ **Tokenizer load** — load the 32k byte-level BPE `tokenizer.json`, encode/decode.
  *(next milestone — tokenizer module; needed for eval, not for training on
  pre-tokenized shards)*
- ⬜ **Gradient accumulation** (`grad_accum=32`, effective batch 64, 131k tok/step).
  GoMLX has `Trainer.AccumulateGradients` — wire it.
- ⬜ **Gradient clipping by global norm** (`grad_clip=1.0`). GoMLX ships clip-by-value
  only; hand-roll global-L2-norm clip (~30 lines).
- ⬜ **WSD LR schedule** — warmup → stable (constant) → optional decay
  (`warmup_steps=1000`, `lr=4e-4`, `min_lr=4e-5`, `decay_frac=0`). Hand-rolled
  piecewise (Python hand-rolls it too); GoMLX has cosine only.
- ⬜ **AdamW full config** — `beta1=0.9 beta2=0.95 weight_decay=0.1`. Wire the betas
  + decay through (M4 used defaults + wd=0).
- ⬜ **bf16 training** (`dtype=bfloat16`) — train in bf16 with fp32-internal norms
  (the model already does fp32 norm accumulation). Validate numerics vs fp32.
- ⬜ **Periodic eval** (`eval_interval=2000`, `eval_iters=100`) — val loss over the
  held-out `val_*.bin` shard; this is the val-loss signal compared against the baseline.
- ⬜ **Checkpointing + resume** — rolling `latest` (`save_interval=200`), permanent
  snapshots (`snapshot_interval=25000`, `keep_last_snapshots=3`), resume from
  latest. GoMLX `ml/model/checkpoint` saves all variables (incl. optimizer state)
  automatically — a strength; add the best/snapshot rotation policy on top.
- ⬜ **metrics.jsonl** — emit `step/train_loss/val_loss/lr/tok_per_sec/peak_vram`
  (mirror lmkit's `observability`) so the existing Grafana dashboard + ops CLI work
  unchanged. JSONL is stdlib-trivial; **peak_vram** needs `go-nvml` (CUDA-only).
- ⬜ **The pretrain driver** — the real `lmkit.pretrain` equivalent: the step loop
  tying accumulation, clip, schedule, eval, checkpoint, metrics, resume together.
  M4's `Overfit` is the seed; this is the production loop.
- ⬜ **TrainConfig + config loading** — a Go `TrainConfig` mirroring the dataclass
  (lr/schedule/batch/intervals/optimizer/seed/dtype); load from file.
- ⬜ **Run on the GPU host** — the actual long run toward the Chinchilla-budget val curve (single
  CUDA GPU ~8GB, bf16, ~days). Durable/resumable per the WSD config.

## B. Tokenizer (lmkit/tokenizer.py, lmkit/shard.py)
- ⬜ **Load + encode/decode** byte-level BPE (next milestone).
- ⬜ **Train a fresh BPE** from a corpus (`train_tokenizer.py --vocab-size 32000`).
  We load an existing vocab; training one in Go is a later/optional add.
- 🧊 **Byte-exact without cgo** — pure-Go loader + Python equivalence gate (ADR-0003);
  cgo `daulet/tokenizers` is the exact-but-cgo alternative we rejected.
- ⬜ **Corpus pull** (`data.py`: stream SmolLM-corpus + wikipedia → jsonl.gz). Exists
  in Python; out of scope for Go (one-time data prep).
- ⬜ **Go corpus sharder** (`lmkit/shard.py`: jsonl → tokenize → `uint16` shards,
  content-hash split). Chose read-existing-shards; a Go sharder is optional
  completeness (needs the Go tokenizer exact first).

## C. Eval / generation (lmkit/eval/gen.py, judge.py)
- ⬜ **Autoregressive generation / sampling** — greedy/temperature/top-k/top-p decode.
- ⬜ **KV cache** for efficient inference/generation (the model is forward-only now).
- ⬜ **LLM-judge eval** (`eval/judge.py`) — quality eval via a judge model.
- ⬜ **Eval harness / metrics** beyond val loss.

## D. SFT — supervised fine-tuning (lmkit/sft.py, sft_data.py)
- ⬜ **SFT loop** — fine-tune off a pretrained base (`SFTConfig`, `init_from` weights-
  only + fresh optimizer).
- ⬜ **Instruct data** handling (`instruct.jsonl.gz`, chat templating, loss masking on
  prompt tokens).

## E. IO / hub (lmkit/push.py)
- ⬜ **safetensors** read/write (~200-line hand-roll; the format is a JSON header +
  raw bytes).
- ⬜ **GGUF export** (F16/F32 hand-roll; quantize via llama.cpp `llama-quantize`).
- ⬜ **HF Hub push** (`lmkit.push`) — upload model + tokenizer (net/http + LFS, or
  shell the `hf` CLI first).
- ⬜ **Checkpoint format** interop (read Python `.pt`?) — likely N/A; we use GoMLX
  checkpoints. Cross-stack weight loading would go via safetensors.

## F. Backend / kernels / hardware (mostly ADR-0004, intentional)
- 🧊 **FlashAttention** as an XLA op-graph — explicit `softmax(QKᵀ/√d)V` today;
  O(seq²) memory. Fine at this scale; a perf milestone.
- 🧊 **ROCm** (an AMD/ROCm GPU) training path — PJRT-ROCm validation pending.
- 🧊 **Metal** (Apple GPU) training — revive the PJRT-Metal bridge (jax-metal is
  abandoned). Workstream, not on the repro critical path.
- ⬜ **peak-VRAM** query — `go-nvml` poll (CUDA-only), for metrics.
- 🧊 **Hand-written kernels** (Triton/CUTLASS-style) — deferred unless profiling
  forces it.
- ⬜ **Multi-GPU / distributed** (FSDP/DDP) — single-GPU for the repro; later.
- 🧊 **gotch fallback** — second `Backend` impl, kept real by the boundary, unbuilt.

## G. CLI / ergonomics (lmkit/cli.py, quickstart.py)
- 🟡 **CLI** — `lmkit quickstart` demo exists; need `shard`, `train`, `eval`
  subcommands (`cmd/lmkit`).
- 🟡 **quickstart** — a bring-up demo exists; the lmkit `quickstart` is a fuller
  end-to-end demo.
- ⬜ **observability** (`lmkit/observability.py`) — structured run logging/metrics.

## Net read
The model layer is at parity (forward, parity-verified). The dominant gap is the
**production training loop** (section A) — accumulation, clip, WSD, bf16, eval,
checkpoint/resume, metrics — which is exactly what stands between M4's
overfit-a-batch and the lm-100m-en replica run. Tokenizer/data are the immediate
next milestone; eval/SFT/io/hub are post-reproduction; FlashAttention/ROCm/Metal/
kernels are intentional ADR-0004 deferrals.
