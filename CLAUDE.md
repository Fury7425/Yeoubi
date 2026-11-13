# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Yeoubi(여우비) is a tokenizer-free **masked-diffusion** language model with **ternary
(BitNet b1.58 / a4.8) weights**. There is no autoregressive decoding and no tokenizer:
the model denoises a canvas of UTF-8 bytes. Read `README.md` for the research framing
and `docs/LONG_CONTEXT.md` / `docs/DATASET_GUIDE.md` for the long-context and data
designs.

## Commands

```bash
python -m pip install -r requirements.txt   # torch, safetensors, wandb, datasets
python -m pip install pytest                # tests are not in requirements.txt
```

Optional extras, both with graceful fallbacks: `bitsandbytes` (enables
`--use_8bit_adam`), `triton` (packed-inference GPU kernels; otherwise a correctness-
oriented CPU path), `litdata` (for `--use_litdata` / `scripts/optimize_litdata.py`).

```bash
python -m pytest -q                                  # full suite
python -m pytest -q tests/test_long_context.py       # one file
python -m pytest -q tests/test_rdt.py::test_depth_extrapolation   # one pytest-style test
python -m pytest -q tests/test_kernels.py -k pack_unpack          # unittest-style: select by -k
```

Tests mix `unittest.TestCase` classes and bare pytest functions; run everything through
pytest. Kernel/MoE tests self-skip when the required backend is unavailable.
`tests/test_gzip_parts.py` is torch-free and can run as `python tests/test_gzip_parts.py`
on a machine without torch.

```bash
# Data: build the curated byte mixture (see docs/DATASET_GUIDE.md first)
python prepare_hf_jsonl.py --manifest configs/data_mix.json --data-root data --pii-policy redact
python scripts/pretokenize.py --jsonl "data/train/*.jsonl" --output_dir data/pt   # 3-5x loader speed
python scripts/optimize_litdata.py --jsonl "data/train/*.jsonl" --output_dir data/litdata

# Train (every TrainConfig field is a CLI flag — see below)
python train.py --train_data "data/train/*.jsonl" --val_data "data/val/*.jsonl"

# Sample / export / benchmark
python sample.py --checkpoint checkpoints/final.pt --prompt "Hello" --length 256 --steps 20
python sample.py --checkpoint checkpoints/final.pt --block --block_size 256 --auto_length
python export.py --checkpoint checkpoints/final.pt --output_dir export/ --pack
python scripts/bench_packed_linear.py --shapes 768,1024,2048,4096
```

A CPU smoke run before any real training: `python train.py --train_data ... --device cpu
--max_steps 5 --batch_size 1 --grad_accum_steps 1 --max_seq_len 512 --hidden_dim 128
--n_heads 4 --ffn_dim 512 --wandb_project ""` (empty `--wandb_project` disables W&B).

## Architecture

### Two model families, one config lineage

- `yeoubi/model.py` — `ModelConfig`, `BitLinear`, `BitAttention`, `BitFFN`/`BitMoEFFN`,
  `BytePatchCodec`, and the flat `BitDiffusionTransformer` (`--model_type standard`).
- `yeoubi/rdt.py` — `RDTConfig` (**subclasses `ModelConfig`**) and `BitRDTTransformer`,
  the **default** (`--model_type rdt`): prelude → pre-recurrent bridge → recurrent block
  looped 1–3× → post-recurrent bridge → coda. The timestep is reinjected each loop; loop
  count is randomized in training and overridable at inference with `--n_loops`.

Because `RDTConfig` is a superset, checkpoints store **both** `model_config` and
`rdt_config`. `utils.resolve_checkpoint_model_config()` filters the stored dict down to
`ModelConfig`'s own fields, and the RDT loader rebuilds the full `RDTConfig` from the
separate key. Adding a config field means touching `ModelConfig`, `TrainConfig`
(forwarding block in `train.train()`), and — if it changes topology — the
`_MODEL_TOPOLOGY_FIELDS` list in `yeoubi/utils.py`.

### Byte-level contract (do not break)

`yeoubi/byte_codec.py` fixes the IDs: **0–255 = literal bytes, PAD=256, EOS=257**
(`VOCAB_SIZE = 258` predictable IDs), **MASK=258** is the absorbing state and never a
clean-data target, legacy THINK=259. `train()` hard-rejects any `vocab_size != 258`.
There is no BOS. Diffusion can expose a split multi-byte character, so decoding always
uses `errors="replace"` — never "fix" that by raising.

### Byte patches: lengths are always in bytes

With `use_byte_patches` (default on, `byte_patch_size=4`), the local encoder compresses
4 byte positions into 1 latent before the global stack and the decoder expands back.
`max_seq_len`, `--length`, and mask tensors are **byte** counts everywhere outside
attention; `patch_attention_mask()` / `block_causal_attention_mask()` convert them to
patch positions. Corruption, loss, and sampling all stay byte-level.

### Quantization: simulation during training, packing at export

- `BitLinear` keeps a full-precision `latent_weight` and ternarizes it with a
  straight-through estimator on every forward. Training is a *simulation* — no speedup.
- Activation precision is a global stage, not a per-layer setting:
  `ActivationSchedule` (first 90% A8 → last 10% A4) calls `set_activation_mode()` down
  the module tree; each layer restores its configured mode from `_original_act_mode`.
- `pack_for_inference()` is **one-way**: it deletes `latent_weight`, installs `w_packed`
  (2-bit, 4 weights/byte) plus a per-matrix `scale_w`, and routes through
  `yeoubi/kernels.py` (Triton → `_int_mm` → CPU fallback). `BitLinear._load_from_state_dict`
  auto-detects packed checkpoints and reshapes the module before loading.
- Any new ternary layer needs *all three*: `pack_for_inference`, the `_load_from_state_dict`
  packed-detection hook, and `set_activation_mode`. Layers that must stay non-ternary set
  `exclude_from_ternary` (see `Int8Linear`) and are skipped by packing.
- `yeoubi/quantization.py` also owns the 3-bit KV cache (first position 4-bit), used only
  by the block sampler.

### Training loop (`yeoubi/train.py`)

`TrainConfig` is the single source of truth: `main()` reflects over its fields and emits
one `--<field>` per field. **Booleans are parsed as strings** (`--bf16 false`,
`--gradient_checkpointing true`) — there are no `--no-x` forms. Adding a field to
`TrainConfig` automatically adds the flag; nothing else registers arguments.

Three stages via `--training_stage`:
- `pretrain` — full-text masked diffusion.
- `reasoning_sft` — prompt frozen and excluded from loss; only response bytes are
  corrupted and supervised. Requires JSONL (not litdata / not `.pt`).
- `correction` — reasoning SFT plus paired `corrupted_response` (must be **byte-length
  identical** to the clean response after truncation) or synthetic corruption, with `t`
  scaled by `correction_t_max`.

`--resume_from` restores optimizer/scheduler state; `--finetune_from` loads weights only
and the two are mutually exclusive. Optimizer is hybrid by default: **Muon** for 2D
transformer-body matrices, **AdamW** for embeddings/norms/biases/head, wrapped by
`_HybridOptimizer`/`_HybridScheduler` whose state dicts use a `{"hybrid": [...]}` envelope
(legacy single-optimizer dicts are still accepted).

### Data (`yeoubi/data.py`, `prepare_hf_jsonl.py`)

`_build_dataloader()` dispatches on the glob's suffixes: all `.pt` → pretokenized loader,
all `.jsonl` → streaming loader, mixed → hard error; `--use_litdata` treats the path as a
litdata directory instead. JSONL records carry `input_ids` (byte values plus optional
EOS=257 — validated) or a `text` fallback; post-training records need `prompt` plus
`response` (or `reasoning`/`rationale` + `answer`). Documents never bleed across chunk
boundaries. `prepare_hf_jsonl.py` deduplicates, splits, and assigns provenance at
**document** level before chunking, so quotas are unique accepted bytes, not emitted
positions.

Colab path (disk-constrained): `prepare_hf_jsonl.py --gzip-parts` gzips every
intermediate, deletes source shards as they are consumed, and writes
`<data-root>/colab/parts/part_*.jsonl.gz` + `val.jsonl.gz` + `meta.json` (parts,
steps_per_part, total_steps) instead of the single train/val JSONL. Parts are small
because a resumed run has no saved loader position — the Colab cell picks a part from the
checkpoint step. `scripts/pack_for_colab.py` produces the same layout from an
already-built plain JSONL. `configs/colab_mix.json` is the matching ~10B-byte English
mixture. `train.py`'s loader does **not** read `.gz` itself (suffix dispatch only knows
`.pt`/`.jsonl`); parts must be decompressed before training. Related flags:
`--min-free-gb` (abort on low disk), `--hf-token-file` (keeps the token out of shell
history).

### Devices

Never hardcode `.cuda()`. `yeoubi/device.py` resolves `auto | cuda[:n] | xpu[:n] | rocm |
cpu` and centralizes AMP settings, TF32, cache emptying, seeding, and `torch.compile`.
Training substitutes a no-op GradScaler off CUDA-fp16.

### Checkpoint loading is security-sensitive

`utils.read_checkpoint()` defaults to `weights_only=True`; the full-pickle path is opt-in
via `--trust_checkpoint`. Keep new load paths on that helper rather than calling
`torch.load` directly. `save_checkpoint(async_io=True)` clones state to CPU synchronously
and writes on a background thread — call `drain_checkpoint_writer()` before any code that
must observe the file (the loop already does so on SIGTERM and at exit).

### Sampling (`yeoubi/sample.py`)

`denoise()` is the full-canvas sampler (no KV cache — every step reprocesses the canvas).
`BlockDiffusionSampler` (`--block`) is the semi-autoregressive path that commits blocks
into the quantized KV cache and supports `--auto_length` EOS stopping.
`ThinkingDiffusionSampler` serves the legacy thinking-token ablation (`N_think`, disabled
by default). `sample.py`/`export.py` model-shape flags (`--hidden_dim 768` etc.) are
**legacy fallbacks** used only when a checkpoint has no embedded `model_config`; a modern
checkpoint's config wins.

## Conventions

- Every source file starts with an SPDX header. Modules that can produce or transform
  weights use `BigCode-OpenRAIL-M-1.0`; pure tooling (`byte_codec`, `data_pipeline`,
  `device`, `scripts/`, `prepare_hf_jsonl.py`) uses `Apache-2.0`. Match the neighbouring
  file when adding one.
- Optional dependencies are imported in a `try/except` with a logged fallback, never made
  hard requirements (see the bitsandbytes and Triton patterns).
- Comments in this codebase explain *why* a value or fallback exists (memory budgets,
  checkpoint compatibility, numerical stability). Keep that density.
