# SPDX-License-Identifier: Apache-2.0
"""Torch-free check of the Colab gzip part writer: python tests/test_gzip_parts.py"""

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import prepare_hf_jsonl as pipeline  # noqa: E402


class GzipPartsTest(unittest.TestCase):
    def test_parts_round_trip_and_chunk_count(self) -> None:
        short = json.dumps({"input_ids": [1] * 100}, separators=(",", ":")) + "\n"
        long = json.dumps({"input_ids": [1] * 8192}, separators=(",", ":")) + "\n"
        lines = [short, long, short]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "parts").mkdir()
            with mock.patch.object(pipeline, "COLAB_PART_CHARS", len(short) + 1):
                with mock.patch.object(pipeline, "COLAB_CHUNKS_PER_STEP", 1):
                    meta = pipeline.write_colab_parts(lines, root)

            files = sorted((root / "parts").glob("part_*.jsonl.gz"))
            restored = [
                line for file in files for line in gzip.open(file, "rt", encoding="utf-8", newline="\n")
            ]
            self.assertEqual(restored, lines)
            # short=1 chunk; 8192 ids split at 4096 gives 2 full chunks.
            self.assertEqual(meta["total_steps"], 4)
            self.assertEqual(meta["parts"], len(files))
            self.assertEqual(json.loads((root / "meta.json").read_text()), meta)

    def test_free_space_guard_stops_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = pipeline.PipelinePaths(Path(temporary), min_free_bytes=10**18)
            with self.assertRaises(RuntimeError):
                paths.check_free_space()

    def test_tmp_suffix_stays_gzipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "a.train.jsonl.gz.tmp"
            with pipeline._open_text(path, "w") as handle:
                handle.write("x\n")
            self.assertEqual(path.read_bytes()[:2], b"\x1f\x8b")


if __name__ == "__main__":
    unittest.main()
