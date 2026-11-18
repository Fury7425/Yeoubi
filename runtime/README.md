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

## Check against PyTorch

Run this once for each new build or backend:

```bash
python scripts/check_runtime_parity.py --checkpoint checkpoints/final.pt --gguf export/model.gguf --runtime runtime/build/yeoubi
```

## How it works and what is missing

- Ternary weights are stored as ggml `Q4_0`, with codes {7, 8, 9} and the
  absmean scale in each block. This is exact, and every backend has a Q4_0
  matmul. The cost is 4.5 bits per weight instead of 2.
- Activations go through the trained INT4 (absmean) quantizer and TopK
  sparsification, as `sample.py` does. `--act a8` turns both off. The INT8
  rounding steps are skipped because their error is below ggml's own
  activation quantization.
- Not supported yet: MoE checkpoints, the block sampler with its quantized KV
  cache (`--block`), and legacy thinking tokens. Attention is plain softmax
  attention, not flash attention.
