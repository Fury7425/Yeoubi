# SPDX-License-Identifier: Apache-2.0
"""Materialize JSONL UTF-8 bytes plus EOS into ``.pt`` shards for fast training.

Training with text-only ``.jsonl`` re-encodes every document on every epoch.
Running this once converts the assembled JSONL into byte-ID shards;
point ``train.py --train_data`` at the resulting ``*.pt`` glob and the loader
auto-selects the tokenizer-free fast path (3-5x higher throughput).

Example:
    python scripts/pretokenize.py \
        --jsonl "data/train/*.jsonl" \
        --output_dir data/train_pt

    python train.py --train_data "data/train_pt/*.pt"
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yeoubi.data import pretokenize_dataset
from yeoubi.utils import force_utf8_console, setup_logging


def main() -> None:
    force_utf8_console()
    setup_logging()

    parser = argparse.ArgumentParser(description="Materialize JSONL UTF-8 bytes into .pt shards")
    parser.add_argument("--jsonl", required=True,
                        help="Glob for input .jsonl files (e.g. 'data/train/*.jsonl')")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to write shard_NNNNNN.pt files")
    parser.add_argument("--shard_size", type=int, default=100_000,
                        help="Documents per shard (tune to fit in RAM)")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.jsonl))
    if not paths:
        raise FileNotFoundError(f"No files matched: {args.jsonl}")

    pretokenize_dataset(paths, args.output_dir, shard_size=args.shard_size)


if __name__ == "__main__":
    main()
