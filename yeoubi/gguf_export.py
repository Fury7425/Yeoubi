# SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
# Model weights derived from this code are subject to the BigCode OpenRAIL-M license.
# Source code is licensed under Apache 2.0. See LICENSE for details.
"""
Write a Yeoubi model as GGUF for the native ggml runtime in ``runtime/``.

The file is plain GGUF v3, but the tensor graph is Yeoubi's own: generic GGUF
consumers (llama.cpp etc.) still cannot run it. Only ``runtime/yeoubi`` can.

Ternary BitLinear weights are stored as ggml ``Q4_0`` with every block scale
set to the per-matrix absmean scale and codes {7, 8, 9} = {-1, 0, +1}. That is
exact (up to fp16 rounding of the scale) and, unlike ggml's 2-bit ternary
types, Q4_0 has a matmul kernel on every ggml backend (CUDA, HIP, Vulkan,
Metal, SYCL, OpenCL/Adreno, Hexagon, OpenVINO, CPU), so no backend falls back
to the CPU for the hot path.
"""

from __future__ import annotations

import math
import struct
from dataclasses import asdict

import numpy as np
import torch

from .quantization import absmean_quantize

_ALIGN = 32
# gguf_type values (gguf.h). Floats go out as F64: the runtime computes
# int(n * topk_ratio) and an f32 ratio can land one below Python's result.
_GGUF_I32, _GGUF_BOOL, _GGUF_STR, _GGUF_F64 = 5, 7, 8, 12
# ggml_type values (ggml.h)
_GGML_F32, _GGML_Q4_0 = 0, 2


def _str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, value) -> bytes | None:
    if isinstance(value, bool):  # before int: bool is an int subclass
        return _str(key) + struct.pack("<I?", _GGUF_BOOL, value)
    if isinstance(value, int):
        return _str(key) + struct.pack("<Ii", _GGUF_I32, value)
    if isinstance(value, float):
        return _str(key) + struct.pack("<Id", _GGUF_F64, value)
    if isinstance(value, str):
        return _str(key) + struct.pack("<I", _GGUF_STR) + _str(value)
    return None


def _q4_0_ternary(latent: torch.Tensor) -> bytes:
    """Absmean-ternarize exactly like ``BitLinear`` and encode as Q4_0 blocks."""
    w_q, scale = absmean_quantize(latent.detach().float())
    q = (w_q.to(torch.int8).numpy() + 8).astype(np.uint8)  # {7, 8, 9}
    out_f, in_f = q.shape
    q = q.reshape(out_f, in_f // 32, 32)
    # quantize_row_q4_0_ref: qs[j] = x[j] | x[j + 16] << 4
    qs = q[..., :16] | (q[..., 16:] << 4)
    d = np.full(qs.shape[:-1] + (1,), float(scale), dtype=np.float16).view(np.uint8)
    return np.concatenate([d, qs], axis=-1).tobytes()


def _collect_tensors(model: torch.nn.Module, config) -> list[tuple[str, tuple, int, bytes]]:
    """Return ``(name, torch_shape, ggml_type, data)`` for every runtime tensor."""
    tensors = []
    tie = getattr(config, "tie_embeddings", False)
    for name, value in model.state_dict().items():
        if tie and name == "unembed.weight":
            continue  # the runtime reuses embed.weight
        value = value.detach().cpu()
        if name.endswith(".latent_weight"):
            base = name[: -len("latent_weight")] + "weight"
            if value.shape[1] % 32 == 0:
                tensors.append((base, tuple(value.shape), _GGML_Q4_0, _q4_0_ternary(value)))
            else:
                # ponytail: odd widths stay dequantized F32; none exist at the
                # shipped shapes, so a packed path for them is not worth it.
                w_q, scale = absmean_quantize(value.float())
                tensors.append((base, tuple(value.shape), _GGML_F32,
                                (w_q * scale).float().numpy().tobytes()))
            continue
        value = value.float()
        if name == "recurrent.injection.A_raw":
            # Precompute the LTI diagonal; the runtime never needs A_raw.
            name, value = "recurrent.injection.A", 0.99 * torch.tanh(value)
        elif name.endswith("_local.weight"):
            # Depthwise conv (D, 1, K) -> (K, D): ggml then sees one contiguous
            # [D] row per tap, which the runtime multiplies into shifted views.
            value = value.squeeze(1).t().contiguous()
        tensors.append((name, tuple(value.shape), _GGML_F32, value.numpy().tobytes()))
    return tensors


def write_gguf(model: torch.nn.Module, config, path: str, *, is_rdt: bool) -> None:
    kvs = [_kv("general.architecture", "yeoubi"),
           _kv("yeoubi.model_type", "rdt" if is_rdt else "standard")]
    for key, value in asdict(config).items():
        if isinstance(value, float) and not math.isfinite(value):
            continue
        entry = _kv(f"yeoubi.{key}", value)
        if entry is not None:
            kvs.append(entry)

    tensors = _collect_tensors(model, config)
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)))
    for entry in kvs:
        header += entry
    offset = 0
    for name, shape, ggml_type, data in tensors:
        # ggml dims run innermost-first: reverse the torch shape.
        header += _str(name) + struct.pack("<I", len(shape))
        header += struct.pack(f"<{len(shape)}Q", *reversed(shape))
        header += struct.pack("<IQ", ggml_type, offset)
        offset += len(data) + (-len(data)) % _ALIGN

    with open(path, "wb") as f:
        f.write(header)
        f.write(b"\0" * ((-len(header)) % _ALIGN))
        for _, _, _, data in tensors:
            f.write(data)
            f.write(b"\0" * ((-len(data)) % _ALIGN))
