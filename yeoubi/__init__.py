# SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
# Model weights derived from this code are subject to the BigCode OpenRAIL-M license.
# Source code is licensed under Apache 2.0. See LICENSE for details.
"""Yeoubi language-model package."""

from importlib import import_module

MODEL_NAME = "Yeoubi"

from .byte_codec import (
    BYTE_VOCAB_SIZE, EOS_TOKEN_ID, MASK_TOKEN_ID, PAD_TOKEN_ID, VOCAB_SIZE,
)

# Resolved on first use so torch-free tooling (prepare_hf_jsonl.py -> data_pipeline)
# can import this package on a machine without torch installed.
_LAZY_EXPORTS = {
    "BitDiffusionTransformer": "model",
    "ModelConfig": "model",
    "BitRDTTransformer": "rdt",
    "RDTConfig": "rdt",
    "resolve_device": "device",
    "detect_best_device": "device",
}

__all__ = [
    "MODEL_NAME",
    "BitDiffusionTransformer", "ModelConfig",
    "BitRDTTransformer", "RDTConfig",
    "resolve_device", "detect_best_device",
    "BYTE_VOCAB_SIZE", "VOCAB_SIZE", "PAD_TOKEN_ID", "EOS_TOKEN_ID", "MASK_TOKEN_ID",
]


def __getattr__(name: str):
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module}", __name__), name)
    globals()[name] = value
    return value
