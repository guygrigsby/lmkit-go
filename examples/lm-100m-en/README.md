# lm-100m-en baseline

~100M parameter Llama (GQA, bfloat16) trained on English text, to a Chinchilla token budget.
The original pretraining corpus is no longer available, so the baseline is re-established on a
freshly assembled English corpus rather than the original run's exact validation loss.

## Architecture

| field       | value  |
|-------------|--------|
| vocab       | 32000  |
| hidden      | 768    |
| n_layers    | 12     |
| n_heads     | 12     |
| n_kv_heads  | 4      |
| head_dim    | 64     |
| ffn_hidden  | 2048   |
| seq_len     | 2048   |

## Launch

Build the binary and run under a process supervisor on a CUDA host. The run resumes automatically from `latest/` on restart.

```bash
# On $GPU_HOST, with shards at $LMKIT_DATA:
lmkit train \
  --config examples/lm-100m-en/config.json \
  --model  examples/lm-100m-en/model.json \
  --data   <shard-dir>
```

Process supervisor example (systemd, runit, s6, etc.) — wrap the above command so it restarts on non-zero exit codes. `train.Run` exits 0 on SIGTERM or completion, 2 on NaN loss; a supervisor should restart on 1 and leave code-2 for manual inspection.

## Resume

On restart, `train.Run` detects `latest/` under `out_dir` and resumes from that checkpoint. No flags needed.

## Monitoring

Progress is written to `./out/lm-100m-en/metrics.jsonl`, one JSON object per line. Events:

- `start` / `resume` — run beginning
- `eval` — val loss + perplexity at each eval interval
- `train` — per-step metrics (loss, lr, grad_norm, tok_per_sec, peak_vram_gb, ...)
- `done` — training complete

Target: val loss descending steadily over the run, toward a tiny-Llama baseline at the
Chinchilla token budget on the freshly assembled corpus.

## On-demand integration test

Requires real shards and a CUDA backend:

```bash
GOMLX_BACKEND=xla:cuda \
LMKIT_DATA=$LMKIT_DATA \
go test -tags integration ./examples/lm-100m-en/ -run TestShortBF16 -v
```

The test runs 200 steps (a smoke check, not the full run). It skips if `LMKIT_DATA` is unset.

## Run log

### 2026-06-24 — launch (trig, RTX 3070 Ti, CUDA/XLA)

First lmkit-go pretraining run. Unblocked by the gomlx autodiff fix (matmul weight
gradients now lower as tensor-core gemms instead of CUDA-core reductions, gomlx PR #428),
which took the training step from ~3.0s to ~0.16s (~18x), reaching ~92% of eager PyTorch
on the same GPU.

- config: B=2 x grad_accum 32 (131,072 tok/step), lr 4e-4, WSD stable trunk (decay_frac=0), seed 1337
- throughput: ~29,600 tok/s, step ~4.4s, ~17.8 TFLOP/s, peak VRAM 6.5 GB
- step-0 eval: val_loss 10.556, perplexity 38,410 (random init)
- early train_loss: step 20 = 10.43 -> step 60 = 8.7; grad_norm healthy, no NaN
- ETA to Chinchilla floor (~2B tokens / 15k steps): ~18h; then a stable trunk until val plateaus
- supervision: run under a process supervisor (systemd --user, runit, etc.), auto-resume from `latest/`, no restart on NaN (exit 2)
- metrics: `<out_dir>/metrics.jsonl`
