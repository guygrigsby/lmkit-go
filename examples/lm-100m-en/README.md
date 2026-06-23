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

<!-- Record launch date, step-0 metrics, and subsequent checkpoints here. -->
