# SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
"""fp16 activations with a quiet token must not quantize to NaN.

Regression for a T4 run: with torch.compile off, fp16 absmax quantization
divided by an amax that had underflowed, and the loss went NaN.
"""

import torch

from yeoubi.quantization import HybridQuantizer


def test_fp16_quiet_token_stays_finite():
    x = torch.randn(2, 3, 64)
    x[0, 0] = 0.0      # all-zero token (amax clamp rounded to 0 in fp16)
    x[0, 1] *= 1e-4    # tiny token (127 / amax overflowed fp16)
    x = x.half()
    for mode in ("int8", "int4", "topk_int8"):
        quantizer = HybridQuantizer(mode=mode)
        out = quantizer(x)
        assert out.dtype == torch.float16
        assert torch.isfinite(out).all(), mode
