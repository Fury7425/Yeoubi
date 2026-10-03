# SPDX-License-Identifier: Apache-2.0
"""Prepare a leakage-safe, curated byte-level pretraining mixture.

The pipeline streams Hugging Face datasets, filters and deduplicates complete
documents, assigns each document to a stable split, and only then creates byte
chunks. Quotas are measured in unique accepted UTF-8 document bytes; overlap,
EOS positions, and emitted training positions are tracked separately.

See ``docs/DATASET_GUIDE.md`` before starting a large run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterator, Mapping

from yeoubi.data_pipeline import (
    ContaminationIndex,
    DedupIndex,
    assess_quality,
    assign_split,
    chunk_document,
    document_id,
    handle_pii,
    iter_jsonl_texts,
    source_metadata,
)


DATA_FORMAT_VERSION = 3
DEFAULT_MANIFEST = Path(__file__).parent / "configs" / "data_mix.json"
VAL_RATIO = 0.005
SEED = 42
MIN_BYTES = 64
MAX_DOCUMENT_BYTES = 8_000_000
MAX_CHUNK_LENGTH = 16384
MAX_CHUNK_OVERLAP = 128
# Bucket sizing is a BYTE budget, not a line count. A line here is one byte-chunk record
# reaching 16 KB (SEQ_LENGTH_DIST tops out at 16384), so 500k lines could mean an 8 GB
# bucket -- and assemble_shuffled reads each bucket fully into RAM to shuffle it.
SHUFFLE_BUCKET_BYTES = 256 * 1024**2
# Every bucket is held open for the whole scatter pass. Stay well under the stdio
# descriptor ceiling (512 on Windows) rather than discovering it on a big corpus.
MAX_OPEN_BUCKETS = 256
SEQ_LENGTH_DIST = [
    (128, 3),
    (256, 5),
    (512, 7),
    (1024, 10),
    (2048, 15),
    (4096, 25),
    (8192, 20),
    (16384, 15),
]
SEQ_LENGTHS = [length for length, _ in SEQ_LENGTH_DIST]
SEQ_WEIGHTS = [weight for _, weight in SEQ_LENGTH_DIST]


class PipelinePaths:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.train_dir = self.root / "train"
        self.val_dir = self.root / "val"
        self.shard_dir = self.root / "hf_shards"
        self.state = self.shard_dir / "progress.json"
        self.dedup = self.shard_dir / "dedup.sqlite3"
        self.final_train = self.train_dir / "hf_mix_train.jsonl"
        self.final_val = self.val_dir / "hf_mix_val.jsonl"
        self.report = self.root / "dataset_report.json"

    def ensure(self) -> None:
        for directory in (self.train_dir, self.val_dir, self.shard_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def shard_paths(self, slug: str) -> tuple[Path, Path]:
        return (
            self.shard_dir / f"{slug}.train.jsonl",
            self.shard_dir / f"{slug}.val.jsonl",
        )


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("manifest must contain a non-empty 'sources' list")
    slugs: set[str] = set()
    for spec in sources:
        required = {"slug", "path", "split", "text_field", "target_unique_bytes"}
        missing = required - set(spec)
        if missing:
            raise ValueError(f"source is missing required fields: {sorted(missing)}")
        slug = str(spec["slug"])
        if slug in slugs:
            raise ValueError(f"duplicate source slug in manifest: {slug}")
        slugs.add(slug)
        if int(spec["target_unique_bytes"]) <= 0:
            raise ValueError(f"{slug}: target_unique_bytes must be positive")
    return manifest


def enabled_sources(manifest: Mapping[str, object]) -> list[dict]:
    return [dict(spec) for spec in manifest["sources"] if spec.get("enabled", True)]


def load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: Mapping[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_stream(spec: Mapping[str, object]):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install the data dependency with: pip install datasets>=2.19") from exc

    kwargs: dict[str, object] = {
        "path": spec["path"],
        "split": spec.get("split", "train"),
        "streaming": True,
    }
    for key in ("name", "data_dir", "revision"):
        if spec.get(key) is not None:
            kwargs[key] = spec[key]
    return load_dataset(**kwargs)


def iter_rows(spec: Mapping[str, object]) -> Iterator[tuple[str, Mapping[str, object]]]:
    requested = str(spec.get("text_field", "text"))
    candidates = list(dict.fromkeys([requested, "text", "content", "raw_content"]))
    detected: str | None = None
    for row in load_stream(spec):
        field = detected
        if field is None:
            field = next((name for name in candidates if isinstance(row.get(name), str)), None)
            if field is not None:
                detected = field
                if field != requested:
                    print(f"  {spec['slug']}: auto-detected text field '{field}'")
        text = row.get(field) if field else None
        if isinstance(text, str) and text.strip():
            yield text.strip(), row


def source_needs_rebuild(
    spec: Mapping[str, object], state: Mapping[str, object], paths: PipelinePaths
) -> bool:
    source_state = state.get(str(spec["slug"]), {})
    train_path, val_path = paths.shard_paths(str(spec["slug"]))
    return not (
        source_state.get("complete")
        and source_state.get("format_version") == DATA_FORMAT_VERSION
        and source_state.get("spec_hash") == _canonical_hash(spec)
        and train_path.exists()
        and val_path.exists()
    )


def build_source(
    spec: Mapping[str, object],
    *,
    state: dict,
    paths: PipelinePaths,
    dedup: DedupIndex,
    contamination: ContaminationIndex | None,
    val_ratio: float,
    seed: int,
    pii_policy: str,
) -> dict:
    slug = str(spec["slug"])
    train_path, val_path = paths.shard_paths(slug)
    if not source_needs_rebuild(spec, state, paths):
        result = state[slug]
        print(
            f"  skip {slug:<24} "
            f"{result.get('unique_bytes', 0) / 1e9:.2f}B unique bytes already prepared"
        )
        return result

    dedup.reset_source(slug)
    rng = random.Random(f"{seed}:{slug}:chunks")
    target = int(spec["target_unique_bytes"])
    source_pii_policy = str(spec.get("pii_policy", pii_policy))
    min_quality_score = float(spec.get("min_quality_score", 0.20))
    counters: Counter[str] = Counter()
    unique_bytes = 0
    emitted_positions = 0
    train_positions = 0
    val_positions = 0
    train_unique_bytes = 0
    val_unique_bytes = 0
    pii_totals: Counter[str] = Counter()

    train_tmp = train_path.with_suffix(".tmp")
    val_tmp = val_path.with_suffix(".tmp")
    print(f"\n  building {slug} (target {target / 1e9:.2f}B unique UTF-8 bytes)")

    try:
        with train_tmp.open("w", encoding="utf-8") as train_handle, val_tmp.open(
            "w", encoding="utf-8"
        ) as val_handle:
            for text, row in iter_rows(spec):
                counters["documents_seen"] += 1
                text, pii_counts = handle_pii(text, source_pii_policy)
                pii_totals.update(pii_counts)
                if text is None:
                    counters["rejected_pii"] += 1
                    continue

                quality = assess_quality(
                    text,
                    min_bytes=MIN_BYTES,
                    max_bytes=int(spec.get("max_document_bytes", MAX_DOCUMENT_BYTES)),
                    min_score=min_quality_score,
                )
                if not quality.accepted:
                    counters[f"rejected_{quality.reason}"] += 1
                    continue
                if contamination is not None and contamination.contaminated(text):
                    counters["rejected_benchmark_overlap"] += 1
                    continue

                doc_hash = document_id(text)
                duplicate_reason = dedup.check_and_add(text, source=slug, doc_hash=doc_hash)
                if duplicate_reason:
                    counters[f"rejected_{duplicate_reason}"] += 1
                    continue

                byte_ids = list(text.encode("utf-8"))
                split = assign_split(doc_hash, val_ratio=val_ratio, seed=seed)
                handle = val_handle if split == "val" else train_handle
                metadata = source_metadata(row, spec)
                document_chunks = 0
                for chunk_index, chunk in enumerate(
                    chunk_document(
                        byte_ids,
                        rng=rng,
                        lengths=SEQ_LENGTHS,
                        weights=SEQ_WEIGHTS,
                        min_length=MIN_BYTES,
                        max_length=MAX_CHUNK_LENGTH,
                        max_overlap=MAX_CHUNK_OVERLAP,
                    )
                ):
                    record = {
                        "input_ids": chunk.input_ids,
                        "doc_id": doc_hash,
                        "chunk_index": chunk_index,
                        "byte_start": chunk.byte_start,
                        "byte_end": chunk.byte_end,
                        "document_end": chunk.document_end,
                        "quality_score": round(quality.score, 4),
                        "pii_redactions": sum(pii_counts.values()),
                        **metadata,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    positions = len(chunk.input_ids)
                    emitted_positions += positions
                    if split == "val":
                        val_positions += positions
                    else:
                        train_positions += positions
                    document_chunks += 1

                if not document_chunks:
                    counters["rejected_no_chunks"] += 1
                    continue
                counters["documents_accepted"] += 1
                counters["chunks_emitted"] += document_chunks
                unique_bytes += len(byte_ids)
                if split == "val":
                    val_unique_bytes += len(byte_ids)
                else:
                    train_unique_bytes += len(byte_ids)

                if counters["documents_accepted"] % 1000 == 0:
                    dedup.commit()
                if counters["documents_accepted"] % 10_000 == 0:
                    print(
                        f"    {slug}: {counters['documents_accepted']:>9,} accepted docs, "
                        f"{unique_bytes / 1e9:.3f}B / {target / 1e9:.2f}B unique bytes"
                    )
                if unique_bytes >= target:
                    break

        dedup.commit()
        os.replace(train_tmp, train_path)
        os.replace(val_tmp, val_path)
    except BaseException:
        train_tmp.unlink(missing_ok=True)
        val_tmp.unlink(missing_ok=True)
        raise

    result = {
        "complete": unique_bytes >= target,
        "format_version": DATA_FORMAT_VERSION,
        "spec_hash": _canonical_hash(spec),
        "target_unique_bytes": target,
        "unique_bytes": unique_bytes,
        "emitted_positions": emitted_positions,
        "overlap_and_eos_positions": max(0, emitted_positions - unique_bytes),
        "train_unique_bytes": train_unique_bytes,
        "val_unique_bytes": val_unique_bytes,
        "train_positions": train_positions,
        "val_positions": val_positions,
        "pii_matches": dict(pii_totals),
        **dict(counters),
    }
    state[slug] = result
    save_state(paths.state, state)
    status = "done" if result["complete"] else "source exhausted"
    print(
        f"  {status}: {slug} -> {unique_bytes / 1e9:.3f}B unique bytes, "
        f"{emitted_positions / 1e9:.3f}B emitted positions"
    )
    return result


def assemble_shuffled(
    specs: list[Mapping[str, object]], paths: PipelinePaths, *, seed: int
) -> None:
    """Shuffle source shards without loading the complete corpus in memory."""
    for split, final_path in (("train", paths.final_train), ("val", paths.final_val)):
        shards: list[Path] = []
        total_lines = 0
        total_bytes = 0
        print(f"\nassembling {split} mixture")
        for spec in specs:
            shard = paths.shard_paths(str(spec["slug"]))[0 if split == "train" else 1]
            if not shard.exists():
                print(f"  missing {shard.name}; skipping")
                continue
            line_count = sum(1 for line in shard.open("r", encoding="utf-8") if line.strip())
            total_lines += line_count
            total_bytes += shard.stat().st_size
            shards.append(shard)
            print(f"  {spec['slug']:<24} {line_count:>10,} chunks")
        if total_lines == 0:
            print(f"  no {split} chunks found")
            continue

        bucket_count = max(1, math.ceil(total_bytes / SHUFFLE_BUCKET_BYTES))
        if bucket_count > MAX_OPEN_BUCKETS:
            bucket_count = MAX_OPEN_BUCKETS
            print(
                f"  note: {total_bytes / 1e9:.1f} GB over {bucket_count} buckets "
                f"(~{total_bytes / bucket_count / 1e9:.1f} GB each); each is loaded "
                f"into RAM to shuffle. Raise SHUFFLE_BUCKET_BYTES only with the RAM "
                f"to match, or shuffle in more passes."
            )
        rng = random.Random(f"{seed}:{split}:assembly")
        with tempfile.TemporaryDirectory(prefix="yeoubi_shuffle_") as temporary_dir:
            temp_root = Path(temporary_dir)
            bucket_paths = [temp_root / f"bucket_{index:06d}.jsonl" for index in range(bucket_count)]
            bucket_handles = [path.open("w", encoding="utf-8") for path in bucket_paths]
            try:
                for shard in shards:
                    with shard.open("r", encoding="utf-8") as source_handle:
                        for line in source_handle:
                            if line.strip():
                                bucket_handles[rng.randrange(bucket_count)].write(line)
            finally:
                for handle in bucket_handles:
                    handle.close()

            final_tmp = final_path.with_suffix(".tmp")
            with final_tmp.open("w", encoding="utf-8") as output:
                for bucket_path in bucket_paths:
                    lines = bucket_path.read_text(encoding="utf-8").splitlines(keepends=True)
                    rng.shuffle(lines)
                    output.writelines(lines)
            os.replace(final_tmp, final_path)
        print(
            f"  wrote {final_path} ({final_path.stat().st_size / 1e9:.2f} GB, "
            f"{total_lines:,} chunks)"
        )


def write_report(
    manifest: Mapping[str, object], specs: list[Mapping[str, object]], state: Mapping[str, object], paths: PipelinePaths
) -> None:
    totals: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    for spec in specs:
        source_state = state.get(str(spec["slug"]), {})
        for key in (
            "unique_bytes",
            "emitted_positions",
            "overlap_and_eos_positions",
            "documents_seen",
            "documents_accepted",
            "chunks_emitted",
        ):
            totals[key] += int(source_state.get(key, 0))
        unique = int(source_state.get("unique_bytes", 0))
        categories[str(spec.get("category", "uncategorized"))] += unique
        languages[str(spec.get("language", "unknown"))] += unique
    report = {
        "format_version": DATA_FORMAT_VERSION,
        "manifest_description": manifest.get("description"),
        "totals": dict(totals),
        "unique_bytes_by_category": dict(categories),
        "unique_bytes_by_manifest_language": dict(languages),
        "sources": {str(spec["slug"]): state.get(str(spec["slug"]), {}) for spec in specs},
        "notes": [
            "unique_bytes counts each accepted document once",
            "emitted_positions includes chunk overlap and EOS",
            "language totals use manifest labels; row-level language remains in JSONL metadata",
            "license metadata is descriptive and is not legal advice",
        ],
    }
    paths.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def print_summary(specs: list[Mapping[str, object]], state: Mapping[str, object]) -> None:
    print("\nDataset preparation summary")
    print(f"  {'source':<24} {'unique GB':>10} {'emitted GB':>12} {'accepted docs':>14}")
    for spec in specs:
        result = state.get(str(spec["slug"]), {})
        print(
            f"  {spec['slug']:<24} "
            f"{result.get('unique_bytes', 0) / 1e9:>10.3f} "
            f"{result.get('emitted_positions', 0) / 1e9:>12.3f} "
            f"{result.get('documents_accepted', 0):>14,}"
        )


def reset_outputs(paths: PipelinePaths) -> None:
    """Remove only known generated pipeline files after explicit --rebuild."""
    known_files = [paths.state, paths.dedup, paths.final_train, paths.final_val, paths.report]
    known_files.extend(paths.shard_dir.glob("*.train.jsonl"))
    known_files.extend(paths.shard_dir.glob("*.val.jsonl"))
    known_files.extend(
        [
            paths.dedup.with_name(paths.dedup.name + "-wal"),
            paths.dedup.with_name(paths.dedup.name + "-shm"),
        ]
    )
    for path in known_files:
        if path.is_file():
            path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--val-ratio", type=float, default=VAL_RATIO)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--pii-policy", choices=("keep", "redact", "drop"), default="redact")
    parser.add_argument("--no-fuzzy-dedup", action="store_true")
    parser.add_argument(
        "--decontam",
        type=Path,
        action="append",
        default=[],
        help="Plain-text or JSONL benchmark file; repeat for multiple files",
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Prepare only the named source slug; repeat as needed",
    )
    parser.add_argument("--require-reviewed-licenses", action="store_true")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Discard known generated shards/state and rebuild using this manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val-ratio must be in [0, 1)")
    manifest = load_manifest(args.manifest)
    all_specs = enabled_sources(manifest)
    if args.source:
        requested = set(args.source)
        known = {str(spec["slug"]) for spec in all_specs}
        unknown = requested - known
        if unknown:
            raise ValueError(f"unknown source slug(s): {sorted(unknown)}")
        build_specs = [spec for spec in all_specs if spec["slug"] in requested]
    else:
        build_specs = all_specs

    if args.require_reviewed_licenses:
        unreviewed = [
            spec["slug"] for spec in all_specs
            if "review" in str(spec.get("license", "review-required")).lower()
        ]
        if unreviewed:
            raise RuntimeError(
                "license review is still required for: " + ", ".join(map(str, unreviewed))
            )

    paths = PipelinePaths(args.data_root)
    paths.ensure()
    if args.rebuild:
        reset_outputs(paths)
    state = load_state(paths.state)
    manifest_hash = _canonical_hash(manifest)
    previous_meta = state.get("_meta", {})
    if state and previous_meta.get("manifest_hash") != manifest_hash:
        raise RuntimeError(
            "The manifest or pipeline format changed. Re-run with --rebuild after reviewing "
            "the new mixture; existing generated data will not be silently mixed with it."
        )
    if state and not paths.dedup.exists():
        raise RuntimeError("Dedup index is missing for an existing run; use --rebuild")
    state["_meta"] = {
        "format_version": DATA_FORMAT_VERSION,
        "manifest_hash": manifest_hash,
        "manifest_path": str(args.manifest.resolve()),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "pii_policy": args.pii_policy,
        "fuzzy_dedup": not args.no_fuzzy_dedup,
    }
    save_state(paths.state, state)

    contamination = None
    if args.decontam:
        contamination = ContaminationIndex(iter_jsonl_texts(args.decontam))
        print(f"loaded {len(contamination.hashes):,} benchmark n-grams for decontamination")

    with DedupIndex(paths.dedup, fuzzy=not args.no_fuzzy_dedup) as dedup:
        for spec in build_specs:
            build_source(
                spec,
                state=state,
                paths=paths,
                dedup=dedup,
                contamination=contamination,
                val_ratio=args.val_ratio,
                seed=args.seed,
                pii_policy=args.pii_policy,
            )

    print_summary(all_specs, state)
    assemble_shuffled(all_specs, paths, seed=args.seed)
    write_report(manifest, all_specs, state, paths)
    print(f"\nAudit report: {paths.report}")


if __name__ == "__main__":
    main()
