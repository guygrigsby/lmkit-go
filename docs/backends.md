# Backends

lmkit-go runs on whatever PJRT plugin GoMLX / go-xla can load. The device is chosen
at runtime via `GOMLX_BACKEND`. Nothing in the model or training code is backend
specific; the same binary runs on all of these.

| `GOMLX_BACKEND` | device | setup |
| --- | --- | --- |
| `go` | pure-Go CPU (SimpleGo) | none. No XLA C dependency. What `make check` uses. |
| `xla:cpu` | XLA on CPU | go-xla fetches the CPU PJRT plugin on first run. |
| `xla:cuda` | NVIDIA GPU | go-xla fetches the CUDA PJRT plugin on first run. cuDNN flash attention on the bf16 path. |
| `xla:rocm` | AMD GPU (ROCm) | manual plugin setup, see below. Decomposed attention (no cuDNN flash). |

## CUDA (NVIDIA)

Set `GOMLX_BACKEND=xla:cuda`. go-xla downloads the matching PJRT CUDA plugin on
first run. bf16 compute, cuDNN flash attention on the attention path. A single
~8 GB card trains the 100M model at seq 2048.

## ROCm (AMD)

Validated on RDNA4 (gfx1201). XLA reaches consumer RDNA through the JAX ROCm PJRT
plugin, but the plugin is not self-contained: it dynamically links the full ROCm
userspace, which you supply as pip wheels. No system ROCm install is required, and
this does not interfere with a separate PyTorch+ROCm setup.

1. Fresh venv. Install the JAX ROCm PJRT plugin at the version matching go-xla's
   PJRT line (currently `0.10.2`, the same line as the CUDA plugin):

   ```
   pip install jax==0.10.2 jaxlib==0.10.2 jax-rocm7-pjrt==0.10.2 jax-rocm7-plugin==0.10.2
   ```

2. Install the ROCm 7 runtime wheels for your gfx target from AMD's TheRock index.
   These supply the libraries the plugin links (HIP, rocBLAS, MIOpen, rocprofiler,
   RCCL, ...):

   ```
   pip install --index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ "rocm[libraries]"
   ```

   `gfx120X-all` covers RDNA4; use the index matching your architecture.

3. go-xla discovers plugins by globbing `pjrt_c_api_*_plugin.so`. The JAX plugin
   ships as `xla_rocm_plugin.so`, so expose it under the name go-xla expects:

   ```
   mkdir -p ~/.local/lib/go-xla/rocm
   ln -s .../site-packages/jax_plugins/xla_rocm7/xla_rocm_plugin.so \
         ~/.local/lib/go-xla/rocm/pjrt_c_api_rocm_plugin.so
   ```

4. Run with the plugin path set and the ROCm runtime libs visible:

   ```
   GOMLX_BACKEND=xla:rocm \
   PJRT_PLUGIN_LIBRARY_PATH=~/.local/lib/go-xla/rocm \
   LD_LIBRARY_PATH=.../_rocm_sdk_core/lib:.../_rocm_sdk_libraries_gfx120X_all/lib \
   lmkit quickstart
   ```

   `quickstart` should report `xla:rocm` and pass matmul, gradient and AdamW.

Flash attention does not carry over: the cuDNN custom-call is NVIDIA-only, so on
ROCm the attention path falls back to the decomposed implementation automatically
(`UseFlashAttention` returns `ErrNotImplemented` off-CUDA). Decomposed attention at
seq 2048 needs more memory than flash, but RDNA4 cards have the VRAM headroom; set
`gradient_checkpoint: true` in the train config if tight. An AMD fused-attention
path (MIOpen) is a later optimization.

## Metal (Apple)

Deferred. See [ADR-0004](./adr/0004-multi-gpu-via-pjrt-plugins.md): needs a revived
StableHLO→MPSGraph PJRT bridge.
