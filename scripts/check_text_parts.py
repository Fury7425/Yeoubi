# SPDX-License-Identifier: Apache-2.0
"""Verify a directory of gzip text parts written by prepare_text_parts.py.

A copied or downloaded build can be short a file or hold a truncated one, and
train.py's loader skips lines that fail to parse rather than stopping, so damage
would otherwise only show up as a model trained on less data than intended.

By default five parts (first, last, three random) and val.jsonl.gz are read in full.
--all reads every part and compares per-source byte totals with dataset_report.json,
which must match exactly.

Example:
    python scripts/check_text_parts.py data/40b
    python scripts/check_text_parts.py data/40b --all
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path


def read_part(path: Path) -> tuple[int, Counter]:
    """Return (documents, UTF-8 text bytes per source). Raises on any damage."""
    documents = 0
    source_bytes: Counter[str] = Counter()
    with gzip.open(path, "rb") as handle:   # a truncated or corrupt stream raises here
        for raw in handle:
            record = json.loads(raw)
            text = record["text"]
            if not isinstance(text, str) or not text:
                raise ValueError(f"record {documents} has no text")
            documents += 1
            source_bytes[str(record.get("source"))] += len(text.encode("utf-8"))
    return documents, source_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path)
    parser.add_argument("--all", action="store_true", help="Read every part and reconcile with the report")
    args = parser.parse_args()

    meta = json.loads((args.root / "meta.json").read_text(encoding="utf-8"))
    names = [f"part_{index:05d}.jsonl.gz" for index in range(meta["parts"])]
    missing = [name for name in names if not (args.root / "parts" / name).is_file()]
    print(f"meta.json expects {len(names)} parts, {len(missing)} missing")
    failed = bool(missing)
    for name in missing[:10]:
        print(f"  missing {name}")

    present = [name for name in names if name not in set(missing)]
    if args.all:
        chosen = present
    else:
        middle = random.Random(0).sample(present[1:-1], min(3, max(0, len(present) - 2)))
        chosen = sorted({*present[:1], *present[-1:], *middle})

    documents = 0
    totals: Counter[str] = Counter()
    for position, name in enumerate(chosen, 1):
        try:
            count, source_bytes = read_part(args.root / "parts" / name)
        except Exception as error:
            failed = True
            print(f"  BAD {name}: {type(error).__name__}: {error}")
            continue
        documents += count
        totals.update(source_bytes)
        if not args.all or position % 50 == 0 or position == len(chosen):
            print(f"  ok {name}: {count:,} documents, {sum(source_bytes.values()) / 1e6:.1f} MB", flush=True)

    try:
        val_documents, _ = read_part(args.root / "val.jsonl.gz")
        print(f"  ok val.jsonl.gz: {val_documents:,} documents")
    except Exception as error:
        failed = True
        print(f"  BAD val.jsonl.gz: {type(error).__name__}: {error}")

    print(f"read {len(chosen)} parts: {documents:,} documents, {sum(totals.values()) / 1e9:.3f} GB of text")
    for source, size in sorted(totals.items()):
        print(f"  {source}: {100 * size / max(1, sum(totals.values())):.1f}%")

    report_path = args.root / "dataset_report.json"
    if args.all and report_path.is_file():
        expected = json.loads(report_path.read_text(encoding="utf-8"))["unique_bytes"]
        for source, size in sorted(expected.items()):
            if totals.get(source) != size:
                failed = True
                print(f"  MISMATCH {source}: parts hold {totals.get(source, 0):,} bytes, report says {size:,}")
        if not failed:
            print("per-source byte totals match dataset_report.json exactly")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
