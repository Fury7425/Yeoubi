# Dataset preparation guide

This pipeline is designed to make data experiments auditable before changing
the model architecture. It treats the **document** as the unit of filtering,
deduplication, provenance, and train/validation assignment. Chunks are created
only after those decisions have been made.

## What changed

- Quotas count unique accepted UTF-8 document bytes, not overlapping chunks.
- Stable content hashes assign every chunk from a document to one split.
- The final document chunk contains `EOS=257`.
- Exact cross-source deduplication is always enabled through a disk-backed
  SQLite index. Conservative SimHash near-deduplication is enabled by default.
- Obvious extraction failures are rejected with language- and code-tolerant
  filters.
- Email, IP address, phone, and obvious secret patterns are redacted by default.
- Optional benchmark n-gram decontamination happens before splitting/chunking.
- Every JSONL record carries document ID, source, category, language, license,
  quality tier, synthetic-data status, byte offsets, and quality score.
- `data/dataset_report.json` separates unique bytes from emitted positions and
  reports category/language composition.

## Review the mixture first

The default manifest is [`configs/data_mix.json`](../configs/data_mix.json). It
is a **research starting point**, not a legal approval. Review every source's
current dataset card, terms, opt-out process, and intended use before download.
Rows marked `review-*` deliberately fail when
`--require-reviewed-licenses` is used.

The 20B-byte example mixture is approximately:

| Area | Unique-byte allocation |
|---|---:|
| High-quality/general English web | 35% |
| Mathematics and math/code | 20% |
| Multilingual web | 25% |
| Synthetic educational text | 5% |
| Reference, documents, papers, and public-domain culture | 11.25% |
| Code-related web | 3.75% |

Byte quotas are not linguistically neutral: many scripts require more UTF-8
bytes per character than English. Use row-level language metadata and downstream
evaluations to adjust the mixture rather than assuming equal bytes means equal
semantic exposure.

For a different mixture, copy the manifest and change source quotas or metadata:

```bash
cp configs/data_mix.json configs/data_mix.local.json
python prepare_hf_jsonl.py --manifest configs/data_mix.local.json
```

On PowerShell, use `Copy-Item` in place of `cp`.

## Recommended rollout

Do not begin with a 20B-byte download. Use three stages:

1. **Pipeline smoke test:** set each manifest quota to 1-10MB. Inspect records,
   rejection counters, PII redactions, and source schemas.
2. **Mixture pilot:** collect 1-5B unique bytes and train small 50M-150M models.
   Compare held-out bits per byte by source, language, and category.
3. **Scaling corpus:** scale only mixtures that win the pilot. A 20B-byte corpus
   is still small for a 1B-class model; plan hundreds of billions of unique bytes
   before treating a 1B run as a capability result.

The source order is intentional. When identical material occurs in multiple
sources, the first accepted source owns it. Put preferred, better-documented
sources earlier in the manifest.

The FineWeb2 entries are pinned to the published `v2.1.1` revision. Before a
production run, pin every remaining source to a reviewed Hub commit SHA through
its optional `revision` field. Floating `main` revisions are useful for pilots
but do not define a reproducible corpus.

## Build commands

Basic run:

```bash
python prepare_hf_jsonl.py
```

Build or resume selected sources:

```bash
python prepare_hf_jsonl.py --source finemath --source fineweb2_korean
```

Include benchmark decontamination. Each file may be plain text (one example per
line) or JSONL with a `text`, `prompt`, or `content` field:

```bash
python prepare_hf_jsonl.py \
  --decontam eval_data/humaneval.jsonl \
  --decontam eval_data/math_benchmarks.jsonl
```

Require the manifest to contain no unresolved license-review labels:

```bash
python prepare_hf_jsonl.py --require-reviewed-licenses
```

The manifest and format are fingerprinted. If either changes, the program stops
instead of mixing incompatible old and new shards. After reviewing the change,
rebuild only the known generated outputs:

```bash
python prepare_hf_jsonl.py --rebuild
```

`--rebuild` removes generated progress, source shards, the dedup index, final
mixtures, and the audit report. It does not recursively remove the data root or
unrelated files.

## Chunking policy

The current mixed length distribution spans 128-16384 model positions; positions
are bytes plus EOS. Short windows remain in the mixture to protect local quality,
while 8192- and 16384-byte examples teach extended retrieval. Each long-document
chunk samples its target length. Overlap
is capped at both 128 bytes and one eighth of the chunk, preventing the old
128-byte chunk / 128-byte overlap failure mode. If a tiny tail would otherwise
lose the EOS-bearing final chunk, the final overlap grows just enough to retain
at least 64 positions.

Do not concatenate unrelated documents into one bidirectional diffusion sequence
until the model supports segment-level attention masks. EOS is a boundary signal,
not a guarantee that bidirectional attention cannot leak across packed documents.

## Output contract

Each chunk resembles:

```json
{
  "input_ids": [72, 101, 108, 108, 111, 257],
  "doc_id": "sha256...",
  "chunk_index": 0,
  "byte_start": 0,
  "byte_end": 5,
  "document_end": true,
  "quality_score": 1.0,
  "pii_redactions": 0,
  "source": "fineweb_edu",
  "category": "educational_web",
  "language": "en",
  "license": "ODC-By-1.0",
  "quality_tier": "dataset_filtered",
  "synthetic": false
}
```

Generated files:

- `data/hf_shards/<source>.train.jsonl`
- `data/hf_shards/<source>.val.jsonl`
- `data/hf_shards/progress.json`
- `data/hf_shards/dedup.sqlite3`
- `data/train/hf_mix_train.jsonl`
- `data/val/hf_mix_val.jsonl`
- `data/dataset_report.json`

The existing JSONL, `.pt`, and litdata loaders remain usable. Pretokenized shards
now allow literal byte IDs `0..255` and the clean `EOS=257` ID.

## Validation policy

The pipeline validation split measures pretraining generalization, not benchmark
ability. Maintain separate immutable evaluation suites for:

- every important source and category;
- every supported language;
- code, mathematics, long documents, and noisy OCR;
- valid UTF-8 reconstruction and EOS/length behavior;
- public benchmarks kept entirely outside the training mixture.

Report bits per byte and unique held-out bytes. A leakage-safe validation loss may
look worse than the previous post-chunk random split; that is expected and should
not be "fixed" by weakening the split.

## What this pipeline does not claim

- Regex PII handling is a first pass, not complete privacy protection.
- SimHash is a scalable conservative near-duplicate screen, not semantic dedup.
- Manifest license labels are metadata, not legal advice.
- Dataset-native quality labels are not automatically comparable across sources.
- More diversity is not automatically better. Mixture weights must be selected
  with small-model ablations and source-specific validation.
