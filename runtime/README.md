# Yeoubi native runtime

A C runtime on [ggml](https://github.com/ggml-org/ggml). It runs the same
forward pass and full-canvas sampler as `sample.py`, with no Python or torch.
One binary can hold several backends at once; the device is picked when you
run it, and any op a backend lacks runs on the CPU instead.

| Hardware | ggml backend | CMake flag |
|---|---|---|
| NVIDIA | CUDA | `-DGGML_CUDA=ON` |
| AMD | HIP/ROCm | `-DGGML_HIP=ON` |
| Any GPU (AMD, NVIDIA, Intel, Adreno, Mali) | Vulkan | `-DGGML_VULKAN=ON` |
| Apple GPU | Metal | `-DGGML_METAL=ON` (default on macOS) |
| Intel CPU, GPU and NPU | OpenVINO | `-DGGML_OPENVINO=ON` |
| Intel GPU | SYCL | `-DGGML_SYCL=ON` |
| Qualcomm Hexagon NPU | Hexagon | `-DGGML_HEXAGON=ON` |
| Qualcomm Adreno GPU | OpenCL | `-DGGML_OPENCL=ON` |
| CPU (x86, ARM, RISC-V) | CPU | always on |

Each backend needs its own SDK installed (CUDA toolkit, ROCm, Vulkan SDK,
OpenVINO, Hexagon SDK, and so on). See ggml's build docs for each one.

## Build

```bash
python export.py --checkpoint checkpoints/final.pt --output_dir export/ --format gguf
```

On a machine without torch, the numpy-only exporter writes the same file. It
reads the checkpoint zip directly and resolves no pickled code, the same
safety rule as `weights_only=True`:

```bash
python -m yeoubi.gguf_export checkpoints/final.pt export/model.gguf
```

Prebuilt executables (Windows and Linux with Vulkan, macOS with Metal) come
out of every CI run of `.github/workflows/runtime.yml`, as `yeoubi-<OS>`
artifacts.

```bash
cmake -S runtime -B runtime/build -DGGML_VULKAN=ON -DGGML_CUDA=ON
```

```bash
cmake --build runtime/build --config Release -j
```

CMake fetches ggml at the tag in `YEOUBI_GGML_TAG` (default `v1.0.0`). To
build offline or against a patched ggml, put a checkout at `runtime/ggml/`
and CMake uses that instead.

## Run

```bash
runtime/build/yeoubi --list-devices
```

```bash
runtime/build/yeoubi -m export/model.gguf -p "Hello" -n 256 --steps 20
```

The flags match `sample.py`: `--temp`, `--top-p`, `--seed`, `--loops`,
`--remask random|confidence|remdm`, `--eta`, `--revise`, `--revise-t-max`.
`-d NAME` picks a device by the name `--list-devices` prints. On Windows,
pass non-ASCII prompts with `-f prompt.txt`, because the console code page
can corrupt `-p`.

Block diffusion: the prompt and each finished block are committed to a 4-bit
KV cache, and each new block is denoised against it, the same as
`sample.py --block`. Here `-n` is the number of bytes to generate, and output
is printed one block at a time:

```bash
runtime/build/yeoubi -m export/model.gguf -p "Hello" -n 1024 --block 256 --auto-length
```

## Check against PyTorch

Run this once for each new build or backend:

```bash
python scripts/check_runtime_parity.py --checkpoint checkpoints/final.pt --gguf export/model.gguf --runtime runtime/build/yeoubi
```

`--act none` is the exact check. It turns quantization off on both sides
and must reach a cosine of at least 0.9999. `--block 8` checks the KV-cached
path.

## How it works and what is missing

- Ternary weights are stored on disk as ggml `TQ1_0`, which packs five
  base-3 digits per byte, or 1.69 bits per weight. The step-25000
  checkpoint's GGUF is 35 MB, against 837 MB for the `.pt`. At load time the
  runtime repacks the weights into the most compact type the device can
  multiply natively: `TQ2_0` if the backend has kernels for it, otherwise
  `Q4_0`, which every backend supports. The repack is exact, because every
  value is 0 or ± the scale. To force a type, for example for benchmarking,
  pass `--weight-type q4|tq2|tq1`.
- Activations are quantized the same way as in `sample.py`: INT4 into
  q/k/v/up/gate, TopK followed by INT8 into o/down, and INT8 into the other
  layers. `--act a8` matches the A8 training stage (INT8 everywhere), and
  `--act none` turns quantization off.
- MoE is supported. Routing matches PyTorch's eager path: a softmax router
  with INT8 weights, top-k, renormalised probabilities, and Switch capacity
  dropping in the same order. Experts run through `ggml_mul_mat_id`.
- Not supported: legacy thinking tokens. Attention is plain softmax
  attention, not flash attention. The KV cache stores its 4-bit values
  dequantized, as F32.
