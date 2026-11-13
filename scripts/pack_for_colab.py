# SPDX-License-Identifier: Apache-2.0
"""Split prepared JSONL into small gzip parts for repeated Colab sessions.

Each byte is stored as a JSON number, so plain JSONL is roughly 3.5x the text
size; gzip recovers most of that. Parts stay small because a resumed run has no
saved loader position: the Colab cell picks a part from the checkpoint step and
replays at most one part.

Example:
    python scripts/pack_for_colab.py --data-root data --output data/colab
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
from pathlib import Path

PART_BYTES = 180 * 1024**2
VAL_LINES = 3000  # train.py reads only --val_max_batches batches of this
CHUNKS_PER_STEP = 16  # batch_size 4 * grad_accum_steps 4 in the Colab cell


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("data/colab"))
    args = parser.parse_args()

    parts_dir = args.output / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    # Binary mode: Windows text mode would rewrite newlines and cost a decode.
    parts = size = lines = 0
    out = None
    with open(args.data_root / "train" / "hf_mix_train.jsonl", "rb") as source:
        for line in source:
            if out is None:
                out = gzip.open(parts_dir / f"part_{parts:05d}.jsonl.gz", "wb")
            out.write(line)
            size += len(line)
            lines += 1
            if size >= PART_BYTES:
                out.close()
                out = None
                parts += 1
                size = 0
    if out is not None:
        out.close()
        parts += 1
    if not parts:
        raise RuntimeError("training JSONL is empty; run prepare_hf_jsonl.py first")

    with open(args.data_root / "val" / "hf_mix_val.jsonl", "rb") as source, gzip.open(
        args.output / "val.jsonl.gz", "wb"
    ) as target:
        target.writelines(itertools.islice(source, VAL_LINES))

    meta = {"parts": parts, "steps_per_part": lines // parts // CHUNKS_PER_STEP}
    (args.output / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"{parts} parts, {lines} chunks, {meta['steps_per_part']} steps per part")


if __name__ == "__main__":
    main()
