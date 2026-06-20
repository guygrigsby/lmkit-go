package train

import (
	"encoding/json"
	"os"
	"time"

	"github.com/NVIDIA/go-nvml/pkg/nvml"
)

// emit appends one JSON line to path, adding a "ts" field with the current Unix
// timestamp in seconds. Creates the file if it does not exist.
func emit(path string, fields map[string]any) error {
	fields["ts"] = time.Now().Unix()
	line, err := json.Marshal(fields)
	if err != nil {
		return err
	}
	line = append(line, '\n')
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write(line)
	return err
}

// peakVRAMGB returns the peak VRAM usage in GB across all NVML devices. Returns 0
// if NVML is unavailable (non-CUDA host, driver not installed, etc.).
func peakVRAMGB() float64 {
	if ret := nvml.Init(); ret != nvml.SUCCESS {
		return 0
	}
	defer nvml.Shutdown() //nolint:errcheck
	count, ret := nvml.DeviceGetCount()
	if ret != nvml.SUCCESS || count == 0 {
		return 0
	}
	var peak uint64
	for i := 0; i < count; i++ {
		dev, ret := nvml.DeviceGetHandleByIndex(i)
		if ret != nvml.SUCCESS {
			continue
		}
		mem, ret := dev.GetMemoryInfo()
		if ret != nvml.SUCCESS {
			continue
		}
		if mem.Used > peak {
			peak = mem.Used
		}
	}
	return float64(peak) / (1 << 30)
}

// tflops estimates the compute throughput from parameter count and token throughput.
// Formula: 6 * nParams * tokPerSec / 1e12 (the standard 6-FLOP-per-param estimate).
func tflops(nParams int, tokPerSec float64) float64 {
	return 6.0 * float64(nParams) * tokPerSec / 1e12
}
