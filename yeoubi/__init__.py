# SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
# Model weights derived from this code are subject to the BigCode OpenRAIL-M license.
# Source code is licensed under Apache 2.0. See LICENSE for details.
"""Yeoubi language-model package."""

MODEL_NAME = "Yeoubi"

from .model import BitDiffusionTransformer, ModelConfig
from .rdt import BitRDTTransformer, RDTConfig
from .device import resolve_device, detect_best_device
from .byte_codec import (
    BYTE_VOCAB_SIZE, EOS_TOKEN_ID, MASK_TOKEN_ID, PAD_TOKEN_ID, VOCAB_SIZE,
)

__all__ = [
    "MODEL_NAME",
    "BitDiffusionTransformer", "ModelConfig",
    "BitRDTTransformer", "RDTConfig",
    "resolve_device", "detect_best_device",
    "BYTE_VOCAB_SIZE", "VOCAB_SIZE", "PAD_TOKEN_ID", "EOS_TOKEN_ID", "MASK_TOKEN_ID",
]
