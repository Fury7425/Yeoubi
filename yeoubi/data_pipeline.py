# SPDX-License-Identifier: Apache-2.0
"""Document-level data curation primitives used by the preparation pipeline.

The functions in this module deliberately have no Hugging Face dependency so
they can be tested on small local corpora before a multi-terabyte streaming run.
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

from .byte_codec import EOS_TOKEN_ID


_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")
_EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
_IPV4_RE = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+\d[\d ()-]{7,}\d|\(\d{2,4}\)[\d -]{5,}\d|\d{2,4}[- ]\d{3,4}[- ]\d{3,4})(?!\w)"
)
_SECRET_RE = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*['\"]?[A-Za-z0-9_./+\-=]{12,}"
)


def normalize_document(text: str) -> str:
    """Return a stable normalization for exact cross-source deduplication."""
    text = unicodedata.normalize("NFKC", text).replace("\x00", "")
    return _WHITESPACE_RE.sub(" ", text).strip()


def document_id(text: str) -> str:
    """Content-derived ID: identical normalized documents share a split."""
    return hashlib.sha256(normalize_document(text).encode("utf-8")).hexdigest()


def assign_split(doc_id: str, *, val_ratio: float, seed: int) -> str:
    """Deterministically assign a whole document to train or validation."""
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    digest = hashlib.blake2b(
        f"{seed}:{doc_id}".encode("ascii"), digest_size=8
    ).digest()
    fraction = int.from_bytes(digest, "big") / 2**64
    return "val" if fraction < val_ratio else "train"


@dataclass(frozen=True)
class QualityAssessment:
    accepted: bool
    score: float
    reason: str | None
    metrics: Mapping[str, float]


def assess_quality(
    text: str,
    *,
    min_bytes: int = 64,
    max_bytes: int = 8_000_000,
    min_score: float = 0.20,
) -> QualityAssessment:
    """Apply conservative, language- and code-tolerant document filters.

    This intentionally rejects only obvious extraction failures. Dataset-native
    quality classifiers remain valuable and should be recorded in the manifest.
    """
    encoded = text.encode("utf-8", errors="replace")
    byte_count = len(encoded)
    if byte_count < min_bytes:
        return QualityAssessment(False, 0.0, "too_short", {"bytes": float(byte_count)})
    if byte_count > max_bytes:
        return QualityAssessment(False, 0.0, "too_large", {"bytes": float(byte_count)})

    char_count = max(1, len(text))
    controls = sum(
        1 for ch in text
        if unicodedata.category(ch) == "Cc" and ch not in "\n\r\t"
    )
    replacements = text.count("\ufffd")
    printable = sum(ch.isprintable() or ch in "\n\r\t" for ch in text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    unique_line_ratio = len(set(lines)) / max(1, len(lines))

    longest_run = 1
    current_run = 1
    previous = None
    for ch in text:
        if ch == previous:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 1
            previous = ch

    control_ratio = controls / char_count
    replacement_ratio = replacements / char_count
    printable_ratio = printable / char_count
    run_ratio = longest_run / char_count

    metrics = {
        "bytes": float(byte_count),
        "control_ratio": control_ratio,
        "replacement_ratio": replacement_ratio,
        "printable_ratio": printable_ratio,
        "unique_line_ratio": unique_line_ratio,
        "longest_run_ratio": run_ratio,
    }
    if control_ratio > 0.01:
        return QualityAssessment(False, 0.0, "excess_control_chars", metrics)
    if replacement_ratio > 0.01:
        return QualityAssessment(False, 0.0, "encoding_damage", metrics)
    if printable_ratio < 0.85:
        return QualityAssessment(False, 0.0, "mostly_non_printable", metrics)
    if longest_run >= 128 and run_ratio > 0.02:
        return QualityAssessment(False, 0.0, "repeated_character_run", metrics)
    if len(lines) >= 10 and unique_line_ratio < 0.20:
        return QualityAssessment(False, 0.0, "repeated_lines", metrics)

    score = 1.0
    score -= min(0.35, control_ratio * 20.0)
    score -= min(0.35, replacement_ratio * 20.0)
    score -= max(0.0, 0.95 - printable_ratio)
    if len(lines) >= 10:
        score -= max(0.0, 0.50 - unique_line_ratio) * 0.5
    score = max(0.0, min(1.0, score))
    return QualityAssessment(score >= min_score, score, None if score >= min_score else "low_score", metrics)


def handle_pii(text: str, policy: str) -> tuple[str | None, dict[str, int]]:
    """Keep, redact, or drop documents containing basic high-risk PII patterns."""
    if policy not in {"keep", "redact", "drop"}:
        raise ValueError("pii policy must be one of: keep, redact, drop")
    patterns = (
        ("email", _EMAIL_RE, "<EMAIL>"),
        ("ipv4", _IPV4_RE, "<IP_ADDRESS>"),
        ("phone", _PHONE_RE, "<PHONE>"),
        ("secret", _SECRET_RE, "<SECRET>"),
    )
    counts = {name: len(regex.findall(text)) for name, regex, _ in patterns}
    counts = {name: count for name, count in counts.items() if count}
    if not counts or policy == "keep":
        return text, counts
    if policy == "drop":
        return None, counts
    for _, regex, replacement in patterns:
        text = regex.sub(replacement, text)
    return text, counts


def _simhash(text: str, *, max_shingles: int = 512) -> int | None:
    # A bounded sample keeps near-deduplication practical for very large PDFs.
    tokens = _TOKEN_RE.findall(normalize_document(text[:500_000]).casefold())
    if len(tokens) < 40:
        return None
    shingle_count = len(tokens) - 4
    if shingle_count <= max_shingles:
        starts = range(shingle_count)
    else:
        stride = shingle_count / max_shingles
        starts = [math.floor(index * stride) for index in range(max_shingles)]
    values = []
    for start in starts:
        shingle = "\x1f".join(tokens[start : start + 5])
        values.append(
            int.from_bytes(hashlib.blake2b(shingle.encode("utf-8"), digest_size=8).digest(), "big")
        )
    weights = [0] * 64
    for value in values:
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


class DedupIndex:
    """Disk-backed exact and conservative SimHash near-deduplication index."""

    def __init__(
        self,
        path: str | Path,
        *,
        fuzzy: bool = True,
        hamming_threshold: int = 3,
        min_fuzzy_bytes: int = 512,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fuzzy = fuzzy
        self.hamming_threshold = hamming_threshold
        self.min_fuzzy_bytes = min_fuzzy_bytes
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_hash TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                simhash TEXT,
                byte_count INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS bands (
                band INTEGER NOT NULL,
                band_value INTEGER NOT NULL,
                doc_hash TEXT NOT NULL REFERENCES documents(doc_hash) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_bands_lookup ON bands(band, band_value);
            CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
            """
        )

    def reset_source(self, source: str) -> None:
        self.connection.execute("DELETE FROM documents WHERE source = ?", (source,))
        self.connection.commit()

    def check_and_add(self, text: str, *, source: str, doc_hash: str | None = None) -> str | None:
        """Return duplicate reason, or add the document and return ``None``."""
        doc_hash = doc_hash or document_id(text)
        if self.connection.execute(
            "SELECT 1 FROM documents WHERE doc_hash = ?", (doc_hash,)
        ).fetchone():
            return "exact_duplicate"

        encoded_bytes = len(text.encode("utf-8"))
        signature = _simhash(text) if self.fuzzy and encoded_bytes >= self.min_fuzzy_bytes else None
        if signature is not None:
            candidates: dict[str, str] = {}
            for band in range(4):
                value = (signature >> (band * 16)) & 0xFFFF
                rows = self.connection.execute(
                    """SELECT d.doc_hash, d.simhash
                       FROM bands b JOIN documents d ON d.doc_hash = b.doc_hash
                       WHERE b.band = ? AND b.band_value = ? LIMIT 128""",
                    (band, value),
                )
                for candidate_hash, candidate_signature in rows:
                    if candidate_signature is not None:
                        candidates[candidate_hash] = candidate_signature
            if any(
                (signature ^ int(candidate_signature, 16)).bit_count() <= self.hamming_threshold
                for candidate_signature in candidates.values()
            ):
                return "near_duplicate"

        signature_hex = f"{signature:016x}" if signature is not None else None
        self.connection.execute(
            "INSERT INTO documents(doc_hash, source, simhash, byte_count) VALUES (?, ?, ?, ?)",
            (doc_hash, source, signature_hex, encoded_bytes),
        )
        if signature is not None:
            self.connection.executemany(
                "INSERT INTO bands(band, band_value, doc_hash) VALUES (?, ?, ?)",
                [
                    (band, (signature >> (band * 16)) & 0xFFFF, doc_hash)
                    for band in range(4)
                ],
            )
        return None

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def __enter__(self) -> "DedupIndex":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


@dataclass(frozen=True)
class ByteChunk:
    input_ids: list[int]
    byte_start: int
    byte_end: int
    document_end: bool


def chunk_document(
    byte_ids: Sequence[int],
    *,
    rng,
    lengths: Sequence[int],
    weights: Sequence[float],
    min_length: int,
    max_length: int,
    max_overlap: int,
) -> Iterator[ByteChunk]:
    """Chunk one document, append EOS once, and never cross its boundary."""
    if not byte_ids:
        return
    if any(not 0 <= value <= 255 for value in byte_ids):
        raise ValueError("byte_ids must contain values in [0, 255]")
    position = 0
    total = len(byte_ids)
    while position < total:
        sampled = min(max_length, int(rng.choices(lengths, weights=weights, k=1)[0]))
        sampled = max(min_length, sampled)
        remaining = total - position
        is_final = remaining <= max(1, sampled - 1)
        content_length = min(remaining, sampled - 1 if is_final else sampled)
        if content_length <= 0:
            break
        chunk = list(byte_ids[position : position + content_length])
        byte_end = position + content_length
        if is_final:
            chunk.append(EOS_TOKEN_ID)
        if len(chunk) >= min_length:
            yield ByteChunk(chunk, position, byte_end, is_final)
        if is_final:
            break
        overlap = min(max_overlap, max(0, content_length // 8))
        next_position = position + max(1, content_length - overlap)
        # Ensure a final EOS-bearing chunk is never discarded as a tiny tail.
        if 0 < total - next_position < min_length - 1:
            next_position = max(position + 1, total - (min_length - 1))
        position = next_position


def source_metadata(row: Mapping[str, object], spec: Mapping[str, object]) -> dict[str, object]:
    """Select compact provenance metadata without copying arbitrary source rows."""
    language_field = str(spec.get("language_field", "language"))
    license_field = str(spec.get("license_field", "license"))
    language = row.get(language_field) or row.get("lang") or spec.get("language", "unknown")
    license_name = row.get(license_field) or spec.get("license", "review-required")
    return {
        "source": spec["slug"],
        "category": spec.get("category", "uncategorized"),
        "language": language,
        "license": license_name,
        "quality_tier": spec.get("quality_tier", "unrated"),
        "synthetic": spec.get("synthetic", False),
    }


class ContaminationIndex:
    """In-memory benchmark n-gram index for conservative decontamination."""

    def __init__(self, texts: Iterable[str], *, ngram_size: int = 13) -> None:
        if ngram_size < 5:
            raise ValueError("ngram_size must be at least 5")
        self.ngram_size = ngram_size
        self.hashes: set[bytes] = set()
        for text in texts:
            for ngram in self._ngrams(text):
                self.hashes.add(hashlib.blake2b(ngram.encode("utf-8"), digest_size=8).digest())

    def _ngrams(self, text: str) -> Iterator[str]:
        tokens = _TOKEN_RE.findall(normalize_document(text).casefold())
        for index in range(len(tokens) - self.ngram_size + 1):
            yield "\x1f".join(tokens[index : index + self.ngram_size])

    def contaminated(self, text: str) -> bool:
        if not self.hashes:
            return False
        return any(
            hashlib.blake2b(ngram.encode("utf-8"), digest_size=8).digest() in self.hashes
            for ngram in self._ngrams(text)
        )


def iter_jsonl_texts(paths: Iterable[str | Path]) -> Iterator[str]:
    """Yield benchmark or local-corpus text from plain-text/JSONL files."""
    import json

    for path_value in paths:
        path = Path(path_value)
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if path.suffix.lower() == ".jsonl":
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = row.get("text") or row.get("prompt") or row.get("content")
                    if isinstance(text, str) and text:
                        yield text
                else:
                    yield line
