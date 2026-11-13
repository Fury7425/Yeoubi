# SPDX-License-Identifier: Apache-2.0
"""Single-pass, multi-process build of the byte mixture as gzip text parts.

prepare_hf_jsonl.py spends most of its time on work this layout does not need: it
stores every byte as a JSON number (about 3.8 characters per byte), compresses that
three times (shard, shuffle bucket, part) and runs on one core. Here each document is
written once, as a {"text": ...} record that train.py's loader encodes itself, and
the sources are read by parallel processes.

Kept: PII redaction and the quality filter from yeoubi.data_pipeline, exact
deduplication across all sources, and a document-level validation split.
Dropped: SimHash near-duplicate detection, chunk overlap and randomized chunk
lengths. The loader cuts a document into consecutive max_seq_len pieces with no
overlap and appends EOS at the document end.

A run always starts from zero; there is no resume. Needs pyarrow and huggingface_hub
(both come with `datasets`); torch is not needed.

notebooks/colab_prepare_data.ipynb still embeds an older copy that streams through the
`datasets` library. Point that notebook at this file once it is pushed.

Example (about 1 GB, enough to read the MB/s line):
    python scripts/prepare_text_parts.py --scale 0.1 --out data/tmp/speedtest
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue as queue_module
import random
import re
import shutil
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

# Reader processes are the unit of parallelism, so the numeric libraries must not each
# start a thread pool as well: on a 22-thread laptop OpenBLAS reserved per-thread buffers
# in every one of 12 readers and the run died with "Memory allocation still failed".
# Set before anything imports numpy or pyarrow; spawned readers re-run this module.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yeoubi import data_pipeline  # noqa: E402
from yeoubi.data_pipeline import assess_quality, handle_pii  # noqa: E402

# slug, dataset repo, parquet files inside it, unique bytes at --scale 1, reader
# processes at --reader-scale 1. Mirrors configs/colab_mix.json. Verify every dataset
# card and license before training. The largest source gets two readers so that all
# cores stay busy until the end.
SOURCES = [
    ("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample/100BT/*.parquet", 5_000_000_000, 2),
    ("cosmopedia", "HuggingFaceTB/cosmopedia", "data/web_samples_v1/*.parquet", 3_000_000_000, 1),
    ("wikipedia_en", "wikimedia/wikipedia", "20231101.en/*.parquet", 2_000_000_000, 1),
]
# A resumed training session replays about one part, so parts stay small.
PART_BYTES = 64 * 1024**2
# Every part is an open gzip stream for the whole run, so this is bounded by the
# process's open-file limit: under Linux's usual 1024, and 900 open at once was checked
# on Windows 11 with Python 3.14. Past this many, parts grow instead.
MAX_PARTS = 900
# Must match --max_seq_len and --batch_size * --grad_accum_steps in colab_train.ipynb:
# meta.json's step counts are derived from them.
MAX_SEQ_LEN = 4096
CHUNKS_PER_STEP = 16
VAL_EVERY = 200   # one document in 200 is held out of training
VAL_DOCS = 3000   # how many of those are stored; train.py reads only a few batches
BATCH_BYTES = 1024**2   # reader-to-writer hand-off size; amortizes the pickling
# Data is compressed exactly once, by the single writer process, so a middling level
# keeps the writer from becoming the bottleneck for little extra size.
GZIP_LEVEL = 4
# Reader memory is bounded by these two, not by the parquet layout. Streaming the same
# files through `datasets` read ahead whole decoded row groups: four readers held 4.7 GB
# after 50 seconds and were still climbing on a 16 GB laptop.
READ_BLOCK_BYTES = 8 * 1024**2
ROWS_PER_BATCH = 256

# The first measured build ran at 1.5 MB/s, nearly all of it inside the four PII regexes
# and the long-run regex: Python's re costs a few hundred ns per character for patterns
# that open with a lookbehind or an alternation, and each one scanned every document.
# Each hint below is a NECESSARY condition for its regex to match and is far cheaper to
# test, so the full regex only runs on documents that could contain a match. The output
# is unchanged; --check asserts that against the original functions. Measured after:
# about 3 MB/s on the same 2-vCPU runtime.
_IP_HINT = re.compile(r"\d\.\d{1,3}\.\d{1,3}\.\d")   # the inner 3 dots of any dotted quad
# Every phone alternative starts with + ( or a digit, then has at least 6 characters
# from this class before its final digit.
_PHONE_HINT = re.compile(r"[+(\d][\d ()-]{6,}\d")
# The value half of the secret pattern, verbatim; the keyword half is the slow part.
_SECRET_HINT = re.compile(r"(?i)[:=]\s*['\"]?[A-Za-z0-9_./+\-=]{12,}")
# A run of 128 equal characters covers 8 consecutive multiples of 16, so it shows up as
# 8 equal neighbours in every-16th-character sampling, which is 16x less text to scan.
_RUN_HINT = re.compile(r"(.)\1{7}", re.DOTALL)
_LONG_RUN = data_pipeline._LONG_RUN_RE


def redact(text):
    """Same result as handle_pii(text, "redact")[0], skipping scans that cannot match."""
    if "@" in text:
        text = data_pipeline._EMAIL_RE.sub("<EMAIL>", text)
    if _IP_HINT.search(text):
        text = data_pipeline._IPV4_RE.sub("<IP_ADDRESS>", text)
    if _PHONE_HINT.search(text):
        text = data_pipeline._PHONE_RE.sub("<PHONE>", text)
    if _SECRET_HINT.search(text):
        text = data_pipeline._SECRET_RE.sub("<SECRET>", text)
    return text


class _GuardedLongRun:
    """Stands in for data_pipeline._LONG_RUN_RE inside assess_quality."""

    def finditer(self, text):
        if _RUN_HINT.search(text[::16]) is None:
            return iter(())
        return _LONG_RUN.finditer(text)


# ponytail: patches a private name of the package. If data_pipeline changes how it
# finds long runs this silently stops helping; move the guard into assess_quality
# itself, and the hints into handle_pii, so prepare_hf_jsonl.py gets them too.
# Runs at import, so spawned reader processes (Windows) apply it as well.
data_pipeline._LONG_RUN_RE = _GuardedLongRun()


def read_source(slug, files, out_queue, stop, check):
    """Read one reader's share of a source's parquet files and hand records to the writer."""
    stats = Counter()
    try:
        # Imported here so the writer process never loads pyarrow.
        import pyarrow.parquet as pq
        from huggingface_hub import HfFileSystem

        filesystem = HfFileSystem()
        batch, size = [], 0
        for path in files:
            if stop.is_set():
                break
            with filesystem.open(path, "rb", block_size=READ_BLOCK_BYTES) as raw:
                # buffer_size makes pyarrow pull a column chunk in pieces. Without it the
                # whole compressed chunk of a row group is requested in one read and held.
                # Only the text column is fetched and decoded at all.
                parquet = pq.ParquetFile(raw, buffer_size=READ_BLOCK_BYTES)
                for rows in parquet.iter_batches(batch_size=ROWS_PER_BATCH, columns=["text"]):
                    if stop.is_set():
                        break
                    for text in rows.column(0).to_pylist():
                        if not isinstance(text, str):
                            continue
                        text = text.strip()
                        if not text:
                            continue
                        if check:
                            # A rehearsal run proves the shortcuts on real documents before
                            # a full build trusts them. A failure here stops the build.
                            assert redact(text) == handle_pii(text, "redact")[0], "redact() differs from handle_pii"
                            assert _RUN_HINT.search(text[::16]) or not _LONG_RUN.search(text), "long-run guard missed a run"
                        text = redact(text)
                        quality = assess_quality(text)
                        if not quality.accepted:
                            stats[f"rejected_{quality.reason}"] += 1
                            continue
                        try:
                            data = text.encode("utf-8")
                        except UnicodeEncodeError:
                            # A lone surrogate can be neither written here nor encoded by the loader.
                            stats["rejected_bad_unicode"] += 1
                            continue
                        record = json.dumps({"text": text, "source": slug}, ensure_ascii=False)
                        line = record.encode("utf-8") + b"\n"
                        batch.append((hashlib.blake2b(data, digest_size=8).digest(), line, len(data)))
                        size += len(line)
                        if size >= BATCH_BYTES:
                            out_queue.put((slug, batch, None))
                            batch, size = [], 0
        if batch:
            out_queue.put((slug, batch, None))
    except Exception:
        traceback.print_exc()
        stats["reader_failed"] += 1
    finally:
        out_queue.put((slug, None, dict(stats)))   # end marker; the writer counts these


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--reader-scale",
        type=int,
        default=1,
        help="Multiply every source's reader count (4 readers at 1); raise it until cores are busy",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Assert the regex shortcuts against the original functions (slow; rehearsal only)",
    )
    args = parser.parse_args()

    targets = {slug: int(target * args.scale) for slug, _, _, target, _ in SOURCES}
    parts = max(1, min(MAX_PARTS, math.ceil(sum(targets.values()) / PART_BYTES)))
    shutil.rmtree(args.out, ignore_errors=True)   # never mix parts from two runs
    parts_dir = args.out / "parts"
    parts_dir.mkdir(parents=True)
    handles = [
        gzip.open(parts_dir / f"part_{index:05d}.jsonl.gz", "wb", compresslevel=GZIP_LEVEL)
        for index in range(parts)
    ]

    from huggingface_hub import HfFileSystem

    filesystem = HfFileSystem()
    out_queue = mp.Queue(maxsize=16)   # readers block while the writer is behind
    stops = {slug: mp.Event() for slug in targets}
    readers = []
    for slug, repo, pattern, _, base in SOURCES:
        files = sorted(filesystem.glob(f"datasets/{repo}/{pattern}"))
        if not files:
            sys.exit(f"{slug}: no files match {pattern} in {repo}")
        # File order follows how the source was built (crawl dump, article id), and a
        # build only gets through the first few files. A fixed shuffle samples across
        # the whole source and stays reproducible.
        random.Random(slug).shuffle(files)
        count = base * args.reader_scale
        for shard in range(count):
            share = files[shard::count]
            if share:
                readers.append(
                    mp.Process(
                        target=read_source,
                        args=(slug, share, out_queue, stops[slug], args.check),
                        daemon=True,
                    )
                )
    for reader in readers:
        reader.start()

    rng = random.Random(42)
    seen = set()   # 8-byte digests: about 1 GB of RAM per 10M documents
    kept, docs, stats = Counter(), Counter(), Counter()
    val_lines = []
    chunks = 0
    running = len(readers)
    started = last_report = time.time()
    while running:
        try:
            slug, batch, reader_stats = out_queue.get(timeout=60)
        except queue_module.Empty:
            if not any(reader.is_alive() for reader in readers):
                # A reader died without its end marker, for example killed for memory.
                stats["reader_failed"] += 1
                break
            continue
        if batch is None:
            running -= 1
            stats.update(reader_stats)
            continue
        for digest, line, size in batch:
            if digest in seen:
                stats["rejected_duplicate"] += 1
                continue
            seen.add(digest)
            if int.from_bytes(digest, "big") % VAL_EVERY == 0:
                # Held out whether or not it is stored, so validation never overlaps training.
                if len(val_lines) < VAL_DOCS:
                    val_lines.append(line)
                continue
            if kept[slug] >= targets[slug]:
                continue   # stragglers from a reader that has not seen the stop yet
            # A random part per document mixes the sources without a shuffle pass.
            handles[rng.randrange(parts)].write(line)
            kept[slug] += size
            docs[slug] += 1
            # The loader appends EOS and cuts every MAX_SEQ_LEN positions.
            chunks += size // MAX_SEQ_LEN + 1
        if kept[slug] >= targets[slug]:
            stops[slug].set()
        now = time.time()
        if now - last_report >= 30:
            last_report = now
            total = sum(kept.values())
            shares = "  ".join(f"{name} {100 * kept[name] // targets[name]}%" for name in targets)
            print(f"{total / 1e9:7.2f} GB  {total / 1e6 / (now - started):5.1f} MB/s  {shares}", flush=True)

    for handle in handles:
        handle.close()
    if stats["reader_failed"]:
        # No meta.json, so nothing downstream mistakes a partial mixture for a finished one.
        sys.exit("a reader failed (see the traceback above); nothing was finalized")

    rng.shuffle(val_lines)   # stored in arrival order, which is one source at a time
    with gzip.open(args.out / "val.jsonl.gz", "wb", compresslevel=GZIP_LEVEL) as handle:
        handle.writelines(val_lines)
    meta = {
        "parts": parts,
        "steps_per_part": max(1, chunks // parts // CHUNKS_PER_STEP),
        "total_steps": chunks // CHUNKS_PER_STEP,
    }
    (args.out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    report = {
        "seconds": round(time.time() - started),
        "target_bytes": targets,
        "unique_bytes": dict(kept),
        "documents": dict(docs),
        "validation_documents": len(val_lines),
        "rejections": dict(stats),
        "meta": meta,
    }
    (args.out / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, target in targets.items():
        if kept[name] < target:
            print(f"note: {name} ran out at {kept[name] / 1e9:.2f} of {target / 1e9:.2f} GB")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
