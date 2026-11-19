# SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
# Model weights derived from this code are subject to the BigCode OpenRAIL-M license.
# Source code is licensed under Apache 2.0. See LICENSE for details.
"""
Write a Yeoubi model as GGUF for the native ggml runtime in ``runtime/``.

The file is plain GGUF v3, but the tensor graph is Yeoubi's own: generic GGUF
consumers (llama.cpp etc.) still cannot run it. Only ``runtime/yeoubi`` can.

Ternary BitLinear weights are stored as ggml ``TQ1_0`` (1.6875 bits/weight,
five base-3 digits per byte), with every block scale set to the per-matrix
absmean scale. That is exact (up to fp16 rounding of the scale). The runtime
repacks them at load time into a type the device multiplies natively (TQ2_0,
or Q4_0, which every ggml backend has), so the file is as small as ggml allows
and still runs on any GPU. Rows not a multiple of 256 wide (the RDT LoRA
up-projection) use Q4_0 with codes {7, 8, 9} instead.

This module needs only numpy, so a machine without torch can export a
training checkpoint directly:

    python -m yeoubi.gguf_export checkpoints/step_25000.pt export/model.gguf
"""

from __future__ import annotations

import collections
import math
import pickle
import re
import struct
import sys
import zipfile

import numpy as np

_ALIGN = 32
# gguf_type values (gguf.h). Floats go out as F64: the runtime computes
# int(n * topk_ratio) and an f32 ratio can land one below Python's result.
_GGUF_I32, _GGUF_BOOL, _GGUF_STR, _GGUF_F64 = 5, 7, 8, 12
# ggml_type values (ggml.h)
_GGML_F32, _GGML_Q4_0, _GGML_TQ1_0 = 0, 2, 34


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


def _absmean_ternary(w: np.ndarray) -> tuple[np.ndarray, np.float32]:
    """``quantization.absmean_quantize`` in float32 numpy (round half to even, as torch)."""
    scale = np.float32(max(np.abs(w).mean(dtype=np.float32), np.float32(1e-8)))
    return np.clip(np.round(w / scale), -1, 1).astype(np.int8), scale


def _q4_0_ternary(w: np.ndarray) -> bytes:
    """Absmean-ternarize exactly like ``BitLinear`` and encode as Q4_0 blocks."""
    w_q, scale = _absmean_ternary(w)
    q = (w_q + 8).astype(np.uint8)  # {7, 8, 9}
    out_f, in_f = q.shape
    q = q.reshape(out_f, in_f // 32, 32)
    # quantize_row_q4_0_ref: qs[j] = x[j] | x[j + 16] << 4
    qs = q[..., :16] | (q[..., 16:] << 4)
    d = np.full(qs.shape[:-1] + (1,), scale, dtype=np.float16).view(np.uint8)
    return np.concatenate([d, qs], axis=-1).tobytes()


def _tq1_0_ternary(w: np.ndarray) -> bytes:
    """Absmean-ternarize and pack as ggml TQ1_0 (1.6875 bits/weight).

    Byte-for-byte gguf-py's ``TQ1_0.quantize_blocks``: per 256 weights, 48
    bytes of five base-3 digits each (the first 160 weights in 32 columns, the
    next 80 in 16), 4 bytes of four digits for the last 16, then the fp16
    scale. Each byte holds ceil(value * 256 / 243) so a decoder can pull the
    digits out with a multiply and shift. The block scale is the matrix's
    absmean scale, so the ternary values are exact.
    """
    w_q, scale = _absmean_ternary(w)
    codes = (w_q + 1).astype(np.uint16).reshape(-1, 256)  # {0, 1, 2}
    nb = codes.shape[0]
    p5 = np.array([81, 27, 9, 3, 1], dtype=np.uint16)
    qs0 = (codes[:, :160].reshape(nb, 5, 32) * p5[None, :, None]).sum(axis=1)
    qs1 = (codes[:, 160:240].reshape(nb, 5, 16) * p5[None, :, None]).sum(axis=1)
    qh = (codes[:, 240:].reshape(nb, 4, 4) * p5[None, :4, None]).sum(axis=1)
    q = (np.concatenate([qs0, qs1, qh], axis=-1) * 256 + 242) // 243
    d = np.full((nb, 1), scale, dtype=np.float16).view(np.uint8)
    return np.concatenate([q.astype(np.uint8), d], axis=-1).tobytes()


def _ternary(name: str, value: np.ndarray) -> tuple[str, tuple, int, bytes]:
    # Smallest exact format ggml has. The runtime repacks TQ1_0 at load time
    # into whatever the device multiplies natively (TQ2_0 or Q4_0), so the
    # file stays minimal without tying it to one backend's kernels.
    if value.shape[-1] % 256 == 0:
        return name, value.shape, _GGML_TQ1_0, _tq1_0_ternary(value)
    if value.shape[-1] % 32 == 0:
        return name, value.shape, _GGML_Q4_0, _q4_0_ternary(value)
    # ponytail: odd widths stay dequantized F32; none exist at the shipped
    # shapes, so a packed path for them is not worth it.
    w_q, scale = _absmean_ternary(value)
    return name, value.shape, _GGML_F32, (w_q.astype(np.float32) * scale).tobytes()


def _int8_absmax(w: np.ndarray) -> np.ndarray:
    """``Int8Linear``'s forward weight (per-tensor absmax INT8), dequantized."""
    # 127.0 / amax on a tensor is reciprocal(amax) * 127 in torch; same here.
    scale = (np.float32(1.0) / np.float32(max(np.abs(w).max(), np.float32(1e-8)))) * np.float32(127.0)
    return (np.clip(np.round(w * scale), -127, 127) / scale).astype(np.float32)


_EXPERT = re.compile(r"^(.+\.experts)\.(\d+)\.(.+)$")


def _collect_tensors(state: dict, config: dict) -> list[tuple[str, tuple, int, bytes]]:
    """Return ``(name, numpy_shape, ggml_type, data)`` for every runtime tensor."""
    tensors = []
    experts: dict[str, dict[int, np.ndarray]] = collections.defaultdict(dict)
    tie = config.get("tie_embeddings", False)
    for name, value in state.items():
        if tie and name == "unembed.weight":
            continue  # the runtime reuses embed.weight
        value = np.ascontiguousarray(value, dtype=np.float32)
        hit = _EXPERT.match(name)
        if hit:  # MoE: gathered and stacked below
            experts[f"{hit.group(1)}.{hit.group(3)}"][int(hit.group(2))] = value
            continue
        if name.endswith(".latent_weight"):
            tensors.append(_ternary(name[: -len("latent_weight")] + "weight", value))
            continue
        if name.endswith(".router.weight"):
            value = _int8_absmax(value)
        elif name == "recurrent.injection.A_raw":
            # Precompute the LTI diagonal; the runtime never needs A_raw.
            name, value = "recurrent.injection.A", (np.float32(0.99) * np.tanh(value)).astype(np.float32)
        elif name.endswith("_local.weight"):
            # Depthwise conv (D, 1, K) -> (D, K): ggml's [K, D], the kernel
            # layout ggml_ssm_conv takes.
            value = np.ascontiguousarray(value[:, 0, :])
        tensors.append((name, value.shape, _GGML_F32, value.tobytes()))

    # Experts become one [n_experts, out, in] tensor per projection (ggml's
    # [in, out, n_experts] for ggml_mul_mat_id). Each expert keeps its own
    # absmean scale, which TQ1_0/Q4_0 carry per block anyway.
    for name, by_index in experts.items():
        stack = [by_index[i] for i in range(len(by_index))]
        if name.endswith(".latent_weight"):
            parts = [_ternary(name, w) for w in stack]
            shape = (len(stack),) + stack[0].shape
            tensors.append((name[: -len("latent_weight")] + "weight", shape, parts[0][2],
                            b"".join(p[3] for p in parts)))
        else:  # per-expert ffn_sub_norm gain
            value = np.ascontiguousarray(np.stack(stack))
            tensors.append((name, value.shape, _GGML_F32, value.tobytes()))
    return tensors


def write_gguf(state: dict, config: dict, path: str, *, is_rdt: bool) -> None:
    """``state``: name -> float numpy array (a model state dict); ``config``: asdict(config)."""
    kvs = [_kv("general.architecture", "yeoubi"),
           _kv("yeoubi.model_type", "rdt" if is_rdt else "standard")]
    for key, value in config.items():
        if isinstance(value, float) and not math.isfinite(value):
            continue
        entry = _kv(f"yeoubi.{key}", value)
        if entry is not None:
            kvs.append(entry)

    tensors = _collect_tensors(state, config)
    header = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)))
    for entry in kvs:
        header += entry
    offset = 0
    for name, shape, ggml_type, data in tensors:
        # ggml dims run innermost-first: reverse the numpy shape.
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


# ---------------------------------------------------------------------------
# Torch-free checkpoint reader
# ---------------------------------------------------------------------------

_STORAGE_DTYPES = {
    "FloatStorage": "<f4", "DoubleStorage": "<f8", "HalfStorage": "<f2",
    "BFloat16Storage": "bf16", "LongStorage": "<i8", "IntStorage": "<i4",
    "ShortStorage": "<i2", "CharStorage": "i1", "ByteStorage": "u1",
    "BoolStorage": "?", "UntypedStorage": "u1",
}

# Mirrors utils._LEGACY_MODEL_CONFIG_DEFAULTS + rdt._LEGACY_RDT_CONFIG_DEFAULTS
# (copied: both modules import torch). Keep in sync when a field is added.
_LEGACY_DEFAULTS = {
    "use_sub_norm": False,
    "int4_absmean": False,
    "timestep_scale": 1.0,
}
_LEGACY_RDT_DEFAULTS = {
    "detach_recurrent_anchor": True,
    "recurrent_state_init_std": 0.0,
}


class _Stub:
    """Stand-in for every global the checkpoint names that we do not need.

    Resolving nothing real is what keeps this as safe as torch's
    ``weights_only=True``: no constructor from the file ever runs.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _Lazy:
    """A tensor view, read from the zip only if asked for (optimizer state never is)."""

    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, tuple(size), tuple(stride)

    def numpy(self) -> np.ndarray:
        zf, member, dtype = self.storage
        raw = zf.read(member)
        if dtype == "bf16":
            flat = (np.frombuffer(raw, "<u2").astype(np.uint32) << 16).view(np.float32)
        else:
            flat = np.frombuffer(raw, dtype)
        view = np.lib.stride_tricks.as_strided(
            flat[self.offset:], self.size, tuple(s * flat.itemsize for s in self.stride),
        )
        return np.array(view)


def _rebuild_tensor(storage, offset, size, stride, *_):
    return _Lazy(storage, offset, size, stride)


def read_checkpoint_numpy(path: str) -> dict:
    """Load a ``torch.save`` zip checkpoint with numpy; tensors come back as ``_Lazy``."""
    zf = zipfile.ZipFile(path)
    pkl = next(n for n in zf.namelist() if n.endswith("/data.pkl"))
    root = pkl[: -len("/data.pkl")]

    class Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if (module, name) == ("collections", "OrderedDict"):
                return collections.OrderedDict
            if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
                return _rebuild_tensor
            if (module, name) == ("torch._utils", "_rebuild_parameter"):
                return lambda data, *_: data
            if module == "torch" and name in _STORAGE_DTYPES:
                return _STORAGE_DTYPES[name]
            return _Stub

        def persistent_load(self, pid):
            _, dtype, key, _location, _numel = pid
            return (zf, f"{root}/data/{key}", dtype if isinstance(dtype, str) else "u1")

    with zf.open(pkl) as f:
        return Unpickler(f).load()


def export_checkpoint_file(ckpt_path: str, out_path: str) -> None:
    ckpt = read_checkpoint_numpy(ckpt_path)
    is_rdt = bool(ckpt.get("rdt_config")) or any(
        k.startswith(("prelude.", "recurrent.", "coda.")) for k in ckpt["model_state_dict"]
    )
    config = dict(ckpt.get("rdt_config") or ckpt.get("model_config") or {})
    if not config:
        raise ValueError("checkpoint has no embedded config; use export.py with fallback flags")
    for key, legacy in {**_LEGACY_DEFAULTS, **(_LEGACY_RDT_DEFAULTS if is_rdt else {})}.items():
        config.setdefault(key, legacy)
    if config.get("vocab_size", 258) != 258:
        raise ValueError("not a byte-level checkpoint")
    state = {k: v.numpy() for k, v in ckpt["model_state_dict"].items() if isinstance(v, _Lazy)}
    write_gguf(state, config, out_path, is_rdt=is_rdt)
    print(f"wrote {out_path} ({'rdt' if is_rdt else 'standard'}, step {ckpt.get('step', '?')}, "
          f"{len(state)} tensors)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python -m yeoubi.gguf_export CHECKPOINT.pt OUT.gguf")
    export_checkpoint_file(sys.argv[1], sys.argv[2])
