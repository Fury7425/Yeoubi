# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

from yeoubi.byte_codec import EOS_TOKEN_ID
from yeoubi.data import StreamingJsonlDataset
from yeoubi.data_pipeline import (
    ContaminationIndex,
    DedupIndex,
    assess_quality,
    assign_split,
    chunk_document,
    document_id,
    handle_pii,
)


class TestDocumentSplitting(unittest.TestCase):
    def test_split_is_content_stable(self):
        doc_hash = document_id("A complete document with stable content.")
        split = assign_split(doc_hash, val_ratio=0.2, seed=42)
        self.assertEqual(split, assign_split(doc_hash, val_ratio=0.2, seed=42))

    def test_chunks_have_one_final_eos(self):
        byte_ids = list(("0123456789 abcdef " * 80).encode("utf-8"))
        chunks = list(
            chunk_document(
                byte_ids,
                rng=random.Random(7),
                lengths=[128],
                weights=[1],
                min_length=64,
                max_length=128,
                max_overlap=128,
            )
        )
        self.assertGreater(len(chunks), 1)
        self.assertTrue(chunks[-1].document_end)
        self.assertEqual(chunks[-1].input_ids[-1], EOS_TOKEN_ID)
        self.assertEqual(sum(EOS_TOKEN_ID in chunk.input_ids for chunk in chunks), 1)
        self.assertTrue(all(64 <= len(chunk.input_ids) <= 128 for chunk in chunks))
        self.assertTrue(all(not chunk.document_end for chunk in chunks[:-1]))

    def test_all_chunks_from_document_share_split(self):
        doc_hash = document_id("one document " * 1000)
        expected = assign_split(doc_hash, val_ratio=0.5, seed=9)
        self.assertTrue(
            all(assign_split(doc_hash, val_ratio=0.5, seed=9) == expected for _ in range(20))
        )


class TestCurationFilters(unittest.TestCase):
    def test_exact_cross_source_dedup(self):
        with tempfile.TemporaryDirectory() as directory:
            with DedupIndex(Path(directory) / "dedup.sqlite3", fuzzy=False) as index:
                text = "This sufficiently long document is shared between two data sources. " * 4
                self.assertIsNone(index.check_and_add(text, source="first"))
                self.assertEqual(index.check_and_add(text, source="second"), "exact_duplicate")

    def test_near_dedup(self):
        with tempfile.TemporaryDirectory() as directory:
            with DedupIndex(Path(directory) / "dedup.sqlite3", fuzzy=True) as index:
                base = " ".join(f"token{number}" for number in range(300))
                changed = base.replace("token150", "replacement150")
                self.assertIsNone(index.check_and_add(base, source="first"))
                self.assertEqual(index.check_and_add(changed, source="second"), "near_duplicate")

    def test_quality_rejects_repetition(self):
        text = ("same boilerplate line\n" * 20) + "x" * 200
        result = assess_quality(text)
        self.assertFalse(result.accepted)
        self.assertIn(result.reason, {"repeated_character_run", "repeated_lines"})

    def test_pii_redaction(self):
        text, counts = handle_pii("Contact person@example.com or 192.168.1.2 for access.", "redact")
        self.assertEqual(counts["email"], 1)
        self.assertEqual(counts["ipv4"], 1)
        self.assertNotIn("person@example.com", text)

    def test_benchmark_ngram_decontamination(self):
        benchmark = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi"
        index = ContaminationIndex([benchmark], ngram_size=13)
        self.assertTrue(index.contaminated("prefix " + benchmark + " suffix"))
        self.assertFalse(index.contaminated("an unrelated document with completely different words"))


class TestLoaderEOSCompatibility(unittest.TestCase):
    def test_text_fallback_adds_eos(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(json.dumps({"text": "hello"}) + "\n", encoding="utf-8")
            dataset = StreamingJsonlDataset(
                [str(path)], max_length=32, min_chunk_size=1, shuffle_buffer_size=1
            )
            example = next(iter(dataset))
            self.assertEqual(example["input_ids"][-1].item(), EOS_TOKEN_ID)

    def test_prepared_input_ids_accept_eos(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text(
                json.dumps({"input_ids": [104, 105, EOS_TOKEN_ID]}) + "\n",
                encoding="utf-8",
            )
            dataset = StreamingJsonlDataset(
                [str(path)], max_length=32, min_chunk_size=1, shuffle_buffer_size=1
            )
            example = next(iter(dataset))
            self.assertEqual(example["input_ids"].tolist(), [104, 105, EOS_TOKEN_ID])


if __name__ == "__main__":
    unittest.main()
