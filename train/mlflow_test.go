package train

import (
	"sort"
	"testing"

	lmodel "github.com/guygrigsby/lmkit-go/model"
)

func TestFieldMetrics(t *testing.T) {
	// A train event: numeric fields become metrics at the event's step; the
	// bookkeeping keys (event, step, ts) are excluded.
	train := map[string]any{
		"event": "train", "step": int64(40), "ts": int64(123),
		"train_loss": 9.2, "lr": 1.6e-5, "step_time_ms": int64(4500), "tokens_seen": int64(5242880),
	}
	ms, ok := fieldMetrics(train, 999)
	if !ok {
		t.Fatal("train event should produce metrics")
	}
	got := map[string]float64{}
	for _, m := range ms {
		got[m.Key] = m.Value
		if m.Step != 40 || m.Timestamp != 999 {
			t.Errorf("metric %s: step=%d ts=%d, want step=40 ts=999", m.Key, m.Step, m.Timestamp)
		}
	}
	for _, bad := range []string{"event", "step", "ts"} {
		if _, present := got[bad]; present {
			t.Errorf("bookkeeping key %q leaked into metrics", bad)
		}
	}
	if got["train_loss"] != 9.2 || got["lr"] != 1.6e-5 || got["step_time_ms"] != 4500 {
		t.Errorf("numeric extraction wrong: %v", got)
	}

	// An eval event: the bool "improved" is excluded, numerics kept.
	eval := map[string]any{"event": "eval", "step": int64(2000), "val_loss": 3.1, "improved": true}
	ms, ok = fieldMetrics(eval, 1)
	if !ok {
		t.Fatal("eval event should produce metrics")
	}
	keys := []string{}
	for _, m := range ms {
		keys = append(keys, m.Key)
	}
	sort.Strings(keys)
	if len(keys) != 1 || keys[0] != "val_loss" {
		t.Errorf("eval metrics = %v, want [val_loss] (improved bool excluded)", keys)
	}

	// Non-metric events produce nothing (nan carries a string train_loss, which
	// must not be logged as a metric, and the event itself is not train/eval).
	for _, ev := range []string{"start", "done", "sigterm"} {
		if _, ok := fieldMetrics(map[string]any{"event": ev, "step": int64(1)}, 1); ok {
			t.Errorf("event %q should not produce metrics", ev)
		}
	}
	if _, ok := fieldMetrics(map[string]any{"event": "nan", "step": int64(1), "train_loss": "NaN"}, 1); ok {
		t.Error("nan event should not produce metrics")
	}
}

func TestMLflowDisabledWithoutEnv(t *testing.T) {
	t.Setenv("MLFLOW_TRACKING_URI", "")
	l := newMLflowLogger(Config{OutDir: "/x/lm-100m-en/out"}, lmodel.Config{}, 1)
	if l != nil {
		t.Fatal("logger should be nil (disabled) without MLFLOW_TRACKING_URI")
	}
	// nil logger methods are safe no-ops.
	l.log(map[string]any{"event": "train", "step": int64(1), "train_loss": 1.0})
	l.finishDone()
	l.finishFailed()
	l.finishKilled()
}

func TestExperimentName(t *testing.T) {
	for out, want := range map[string]string{
		"/home/u/runs/lm-100m-en/out":            "lm-100m-en",
		"/home/u/projects/training/moe/ckpt-16e": "moe",
		"":                                       "lmkit-go",
	} {
		if got := experimentName(Config{OutDir: out}); got != want {
			t.Errorf("experimentName(%q)=%q want %q", out, got, want)
		}
	}
}
