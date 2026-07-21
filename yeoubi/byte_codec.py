# SPDX-License-Identifier: Apache-2.0
"""Tokenizer-free UTF-8 byte encoding and the shared special-token layout."""

from __future__ import annotations

from collections.abc import Iterable


# IDs 0..255 are literal byte values. The predictable vocabulary additionally
# reserves PAD=256 and EOS=257. MASK=258 is the diffusion absorbing state and is
# not a clean-data target. No BOS token is used by the current pipeline. When
# thinking is enabled, ModelConfig assigns THINK=259 (MASK + 1).
BYTE_VOCAB_SIZE = 256
PAD_TOKEN_ID = 256
EOS_TOKEN_ID = 257
VOCAB_SIZE = 258
MASK_TOKEN_ID = 258


def encode_text(text: str, *, add_eos: bool = False) -> list[int]:
    """Encode text as UTF-8 bytes, optionally terminating the document."""
    ids = list(text.encode("utf-8"))
    if add_eos:
        ids.append(EOS_TOKEN_ID)
    return ids


def decode_byte_ids(ids: Iterable[int], *, errors: str = "replace") -> str:
    """Decode byte IDs, omitting PAD/EOS/MASK/THINK and other non-byte IDs.

    ``errors="replace"`` is intentional: diffusion may expose an incomplete or
    otherwise invalid UTF-8 sequence at an intermediate step, and previews
    should remain printable rather than fail or silently hide the corruption.
    """
    byte_values = bytearray()
    for token_id in ids:
        value = int(token_id)
        if 0 <= value < BYTE_VOCAB_SIZE:
            byte_values.append(value)
    return byte_values.decode("utf-8", errors=errors)
