package train

import (
	"context"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/guygrigsby/mlflow"
	lmodel "github.com/guygrigsby/lmkit-go/model"
)

// mlflowLogger mirrors the metrics.jsonl stream to an MLflow tracking server when
// MLFLOW_TRACKING_URI is set (gputex injects it into the job env). It is
// best-effort: a nil logger is a no-op so call sites stay unconditional, and every
// tracking error is swallowed — instrumentation never fails or stalls training.
type mlflowLogger struct {
	c     *mlflow.Client
	a     *mlflow.AsyncLogger
	runID string
}

// newMLflowLogger starts an MLflow run for this training job, logging the model and
// training config as params. Returns nil (a working no-op) when MLFLOW_TRACKING_URI
// is unset or the tracking server can't be reached, so training proceeds regardless.
func newMLflowLogger(cfg Config, mcfg lmodel.Config, nParams int) *mlflowLogger {
	uri := os.Getenv("MLFLOW_TRACKING_URI")
	if uri == "" {
		return nil
	}
	c, err := mlflow.NewClient(uri)
	if err != nil {
		return nil
	}
	ctx, cancel := mlCtx()
	defer cancel()
	exp, err := c.GetOrCreateExperiment(ctx, experimentName(cfg))
	if err != nil {
		return nil
	}
	run, err := c.CreateRun(ctx, exp, mlflow.WithRunName("lmkit-go"))
	if err != nil {
		return nil
	}
	l := &mlflowLogger{c: c, a: c.NewAsyncLogger(), runID: run.Info.RunID}
	params := []mlflow.Param{
		{Key: "framework", Value: "lmkit-go"},
		{Key: "params", Value: strconv.Itoa(nParams)},
		{Key: "lr", Value: f(cfg.LR)}, {Key: "min_lr", Value: f(cfg.MinLR)},
		{Key: "warmup_steps", Value: strconv.Itoa(cfg.WarmupSteps)},
		{Key: "max_steps", Value: strconv.Itoa(cfg.MaxSteps)},
		{Key: "decay_frac", Value: f(cfg.DecayFrac)},
		{Key: "batch_size", Value: strconv.Itoa(cfg.BatchSize)},
		{Key: "grad_accum", Value: strconv.Itoa(cfg.GradAccum)},
		{Key: "grad_clip", Value: f(cfg.GradClip)},
		{Key: "weight_decay", Value: f(cfg.WeightDecay)},
		{Key: "beta1", Value: f(cfg.Beta1)}, {Key: "beta2", Value: f(cfg.Beta2)},
		{Key: "dtype", Value: cfg.Dtype}, {Key: "seed", Value: strconv.FormatInt(cfg.Seed, 10)},
		{Key: "gradient_checkpoint", Value: strconv.FormatBool(cfg.GradientCheckpoint)},
		{Key: "vocab", Value: strconv.Itoa(mcfg.VocabSize)},
		{Key: "hidden", Value: strconv.Itoa(mcfg.Hidden)},
		{Key: "n_layers", Value: strconv.Itoa(mcfg.NLayers)},
		{Key: "n_heads", Value: strconv.Itoa(mcfg.NHeads)},
		{Key: "n_kv_heads", Value: strconv.Itoa(mcfg.NKVHeads)},
		{Key: "head_dim", Value: strconv.Itoa(mcfg.HeadDim)},
		{Key: "ffn_hidden", Value: strconv.Itoa(mcfg.FFNHidden)},
		{Key: "seq_len", Value: strconv.Itoa(mcfg.SeqLen)},
		{Key: "rope_base", Value: f(mcfg.RopeBase)},
	}
	_ = c.LogBatch(ctx, run.Info.RunID, nil, params, []mlflow.RunTag{{Key: "framework", Value: "lmkit-go"}})
	return l
}

// log mirrors a metrics.jsonl event map to MLflow: every numeric field of a train
// or eval event becomes a metric at the event's step. Other events (start, done,
// nan, sigterm) carry no per-step metrics and are ignored here.
func (l *mlflowLogger) log(fields map[string]any) {
	if l == nil {
		return
	}
	ms, ok := fieldMetrics(fields, time.Now().UnixMilli())
	if !ok || len(ms) == 0 {
		return
	}
	for _, m := range ms {
		// Fire-and-forget; the worker batches and flushes. Blocks only if the
		// buffer is full, which on a down server self-limits the backlog.
		_ = l.a.LogMetric(context.Background(), l.runID, m.Key, m.Value, m.Timestamp, m.Step)
	}
}

// fieldMetrics turns a metrics.jsonl event map into MLflow metrics: every numeric
// field of a train or eval event becomes a metric at the event's step. ok is false
// for events that carry no per-step metrics (start, done, nan, sigterm). Bookkeeping
// keys (event, step, ts, improved) and non-numeric values (e.g. nan's string
// train_loss) are skipped.
func fieldMetrics(fields map[string]any, now int64) (ms []mlflow.Metric, ok bool) {
	switch fields["event"] {
	case "train", "eval":
	default:
		return nil, false
	}
	step := toInt64(fields["step"])
	for k, v := range fields {
		switch k {
		case "event", "step", "ts", "improved":
			continue
		}
		if x, okf := toFloat(v); okf {
			ms = append(ms, mlflow.Metric{Key: k, Value: x, Timestamp: now, Step: step})
		}
	}
	return ms, true
}

// finishDone/finishFailed/finishKilled terminate the run with the matching status
// (training complete / non-finite loss / SIGTERM). No-op on a nil logger.
func (l *mlflowLogger) finishDone()   { l.finish(mlflow.StatusFinished) }
func (l *mlflowLogger) finishFailed() { l.finish(mlflow.StatusFailed) }
func (l *mlflowLogger) finishKilled() { l.finish(mlflow.StatusKilled) }

// finish marks the MLflow run terminated with the given status.
func (l *mlflowLogger) finish(status mlflow.RunStatus) {
	if l == nil {
		return
	}
	_ = l.a.Close() // flush buffered metrics before the run goes terminal
	ctx, cancel := mlCtx()
	defer cancel()
	_ = l.c.UpdateRun(ctx, l.runID, status, time.Now().UnixMilli())
}

// mlCtx is a short-deadline context off the background context (NOT the training
// cancel context, so logging keeps working through a SIGTERM shutdown) — a slow or
// down tracking server can never stall the training loop.
func mlCtx() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), 5*time.Second)
}

// experimentName is the project the run belongs to, derived from the out dir
// (e.g. ~/runs/lm-100m-en/out -> "lm-100m-en"); falls back to "lmkit-go".
func experimentName(cfg Config) string {
	if cfg.OutDir == "" {
		return "lmkit-go"
	}
	if p := filepath.Base(filepath.Dir(cfg.OutDir)); p != "" && p != "." && p != "/" {
		return p
	}
	return "lmkit-go"
}

func f(v float64) string { return strconv.FormatFloat(v, 'g', -1, 64) }

func toInt64(v any) int64 {
	switch x := v.(type) {
	case int64:
		return x
	case int:
		return int64(x)
	case float64:
		return int64(x)
	}
	return 0
}

func toFloat(v any) (float64, bool) {
	switch x := v.(type) {
	case float64:
		return x, true
	case int64:
		return float64(x), true
	case int:
		return float64(x), true
	}
	return 0, false
}
