# Yeoubi(여우비)

Yeoubi(여우비) is an experimental, tokenizer-free diffusion language model with
ternary Transformer weights. It combines BitNet a4.8-style quantization with an
absorbing-mask diffusion objective: instead of predicting one token at a time from
left to right, the model repeatedly denoises a canvas of masked UTF-8 bytes.

The default configuration is a recurrent-depth Transformer (RDT) with roughly 1B
stored parameters. Four input bytes are compressed into one latent patch for the
expensive global stack, while corruption, prediction, and loss remain byte-level.

The included 20B-unique-byte data mixture and 57,500-step configuration are research
starting points, not claims of a fully trained or benchmarked foundation model. Under
the planning assumptions below, a 20B-byte pilot represents about 30–53 H100
GPU-hours, or approximately **$120–$210 at $4/GPU-hour**, before storage, CPU, and
network costs.

> [!WARNING]
> This repository is research code. The large-scale context, capability, throughput,
> and cost figures in this document are targets or planning estimates, not measured
> results from released checkpoints.

## Highlights

- **Tokenizer-free:** inputs and outputs are UTF-8 bytes; no tokenizer files, BPE
  merges, or vocabulary download is required.
- **Ternary weights:** quantized Transformer matrices use `{-1, 0, +1}` weights.
- **Hybrid activations:** training transitions from W1.58A8 to a W1.58A4 path with
  INT4 inputs, TopK sparsification, and INT8 intermediates.
- **Parallel denoising:** multiple masked positions can be updated during each
  diffusion step.
- **Recurrent depth:** the default RDT executes one to three recurrent refinement
  passes and supports an inference-time loop override.
- **Long-context path:** 16,384-byte training sequences become 4,096 global latent
  positions through four-byte patches.
- **Real packed inference:** inference-only export can replace float quantization
  simulation with packed 2-bit ternary weights and Triton/CPU kernels.

## Yeoubi compared with a standard LLM

| Property | Yeoubi(여우비) | Typical autoregressive LLM |
|---|---|---|
| Generation | Bidirectional masked diffusion | Left-to-right next-token prediction |
| Text units | Raw UTF-8 bytes | Subword tokens |
| Noise state | Absorbing `[MASK]` byte ID | None |
| Quantized weights | Ternary `{-1, 0, +1}` | Commonly FP16/BF16 or integer quantized |
| Activations | A8, then hybrid A4/TopK/INT8 | Commonly FP16/BF16 |
| KV cache | 3-bit quantized cache; first position at 4-bit | Commonly FP16/BF16 |
| Default context | 16,384 bytes | Model-dependent |
| Post-training | Response-only reasoning SFT and optional correction tuning | Instruction/chat SFT and preference tuning |
| Runtime format | Custom diffusion runtime | Commonly Transformers, vLLM, llama.cpp, etc. |

## Requirements and installation

The code requires Python 3.10 or newer and PyTorch 2.2 or newer. Accelerator training
needs a PyTorch build compatible with the installed CUDA, ROCm, or Intel XPU runtime.
The default ~1B configuration is intended for a high-memory accelerator; its reference
memory target is approximately 40 GB with gradient checkpointing. Start with the smoke
test in the training guide before committing to a full run.

```bash
git clone https://github.com/Fury7425/bitDiffusion-a4.8.git Yeoubi
cd Yeoubi

python -m venv .venv
source .venv/bin/activate              # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Weights produced with this code are covered by the BigCode OpenRAIL-M license. Source
code is Apache 2.0. See [LICENSE](LICENSE).

## Step-by-step workflows

<details>
<summary><strong>Train a model from scratch</strong></summary>

### 1. Review and prepare the data mixture

The default manifest, [`configs/data_mix.json`](configs/data_mix.json), describes a
multilingual 20B-unique-byte research mixture. Dataset licenses and access conditions
can change, so review every active dataset card before downloading or training.

If a source requires Hugging Face authentication, set a token in your shell:

```bash
export HF_TOKEN=hf_your_token_here
# Windows PowerShell: $env:HF_TOKEN = "hf_your_token_here"
```

Build the configured dataset:

```bash
python prepare_hf_jsonl.py --manifest configs/data_mix.json
```

This writes the combined training and validation files to `data/train/` and
`data/val/`, keeps resumable source shards in `data/hf_shards/`, and writes composition
and rejection statistics to `data/dataset_report.json`.

Do not start with the full 20B-byte download. Clone the manifest, reduce each source's
`target_unique_bytes`, and validate a small mixture first. The detailed rollout,
decontamination, rebuild, and audit procedure is in
[`docs/DATASET_GUIDE.md`](docs/DATASET_GUIDE.md).

### 2. Optionally pre-tokenize for faster loading

Because tokenization is only UTF-8 byte encoding, the helper materializes JSONL records
into `.pt` shards:

```bash
python scripts/pretokenize.py \
  --jsonl "data/train/*.jsonl" \
  --output_dir data/train_pt
```

Use either all JSONL files or all `.pt` shards for a run; do not mix both formats in
one glob.

### 3. Run a small end-to-end smoke test

This validates data loading, forward/backward passes, checkpointing, and sampling with
a deliberately small standard Transformer:

```bash
python train.py \
  --model_type standard \
  --hidden_dim 256 --n_layers 4 --n_heads 4 --ffn_dim 1024 \
  --max_seq_len 512 --batch_size 2 --grad_accum_steps 1 \
  --max_steps 20 --warmup_steps 2 --save_every 10 \
  --val_every 10 --val_max_batches 2 --num_workers 0 \
  --train_data "data/train/*.jsonl" \
  --val_data "data/val/*.jsonl" \
  --output_dir checkpoints/smoke \
  --wandb_project ""
```

On a CPU, add `--device cpu --bf16 false`. This is only a correctness test and will be
slow; it is not a useful training configuration.

### 4. Launch the default ~1B RDT run

The model and optimizer defaults are already defined in `TrainConfig`:

```bash
wandb login                              # optional; omit if WandB is disabled
python train.py \
  --train_data "data/train_pt/*.pt" \
  --val_data "data/val/*.jsonl" \
  --output_dir checkpoints/pretrain
```

If you did not create `.pt` shards, set `--train_data "data/train/*.jsonl"` instead.
To disable Weights & Biases, pass `--wandb_project ""`.

The default run performs:

```text
57,500 optimizer steps
× 2 examples/device
× 16 gradient-accumulation batches
× 16,384 byte positions
= 30.1B padded byte positions
```

Padded positions are not the same as unique dataset bytes. Use
`data/dataset_report.json` to distinguish accepted bytes, overlap, padding, and EOS
positions.

### 5. Resume safely after interruption

```bash
python train.py \
  --resume_from checkpoints/pretrain/step_25000.pt \
  --train_data "data/train_pt/*.pt" \
  --val_data "data/val/*.jsonl" \
  --output_dir checkpoints/pretrain
```

`--resume_from` restores model, optimizer, scheduler, step, and activation stage. Only
load checkpoints you trust; use `--trust_checkpoint true` solely for a trusted legacy
checkpoint that requires full pickle loading.

### 6. Monitor the A8 → A4 transition

With the default `--a4_warmup_fraction 0.10`, steps 0–51,749 use W1.58A8 and steps
51,750–57,500 use the target hybrid W1.58A4 path. A checkpoint is saved at the stage
transition. Packed weights are not differentiable, so never call
`pack_for_inference()` during training.

</details>

<details>
<summary><strong>Fine-tune or post-train a checkpoint</strong></summary>

### 1. Create response-supervision JSONL

Each line needs a `prompt` and either a complete `response` or separate `reasoning` and
`answer` fields:

```json
{"prompt":"If 3x + 2 = 11, find x.","reasoning":"Subtract 2, then divide by 3.","answer":"3"}
```

Store training and validation files separately, for example:

```text
data/post_train/reasoning/train/examples.jsonl
data/post_train/reasoning/val/examples.jsonl
```

During post-training, prompt bytes are frozen and diffusion loss is applied only to the
assistant response region. Long examples are truncated to `max_seq_len`, so inspect the
rendered prompt/response format before a large run.

### 2. Run reasoning SFT

```bash
python train.py \
  --finetune_from checkpoints/pretrain/final.pt \
  --training_stage reasoning_sft \
  --train_data "data/post_train/reasoning/train/*.jsonl" \
  --val_data "data/post_train/reasoning/val/*.jsonl" \
  --output_dir checkpoints/reasoning_sft \
  --lr 2e-5 --max_steps 5000 --warmup_steps 200
```

`--finetune_from` loads model weights but starts a new optimizer, scheduler, and step
counter. It is mutually exclusive with `--resume_from`.

### 3. Optionally run correction tuning

Correction training can synthesize mistakes in the assistant region:

```bash
python train.py \
  --finetune_from checkpoints/reasoning_sft/final.pt \
  --training_stage correction \
  --use_self_cond true --self_condition_prob 0.5 \
  --correction_noise_prob 0.15 --correction_t_max 0.5 \
  --train_data "data/post_train/correction/train/*.jsonl" \
  --val_data "data/post_train/correction/val/*.jsonl" \
  --output_dir checkpoints/correction \
  --lr 1e-5 --max_steps 3000 --warmup_steps 100
```

Clean records are sufficient. To supply an explicit erroneous draft, add
`corrupted_response`, or add `corrupted_reasoning` and `corrupted_answer`. The clean
and corrupted response must have the same UTF-8 byte length after truncation so target
positions align.

### 4. Resume a fine-tuning run

After the new run has started, resume its own checkpoint with `--resume_from` rather
than returning to `--finetune_from`:

```bash
python train.py \
  --resume_from checkpoints/reasoning_sft/step_2500.pt \
  --training_stage reasoning_sft \
  --train_data "data/post_train/reasoning/train/*.jsonl" \
  --val_data "data/post_train/reasoning/val/*.jsonl" \
  --output_dir checkpoints/reasoning_sft \
  --lr 2e-5 --max_steps 5000 --warmup_steps 200
```

For legacy checkpoints that contain THINK embeddings, preserve the checkpoint's
original `N_think` topology and set `--think_prob 0`. New runs keep legacy thinking
tokens disabled.

</details>

<details>
<summary><strong>Use, launch, and export the model</strong></summary>

### 1. Generate from a training checkpoint

```bash
python sample.py \
  --checkpoint checkpoints/reasoning_sft/final.pt \
  --prompt "The theory of relativity states that" \
  --length 512 --steps 20
```

`--length` is the total byte length of the prompt plus generated canvas, not a count of
new subword tokens. The checkpoint's saved `model_config` is detected automatically.

Useful controls:

| Flag | Default | Purpose |
|---|---:|---|
| `--steps` | 20 | More denoising steps can improve quality but increase latency |
| `--temperature` | 0.9 | Sampling temperature |
| `--top_p` | 0.95 | Nucleus sampling cutoff |
| `--num_samples` | 1 | Independent samples to generate |
| `--device` | `auto` | `auto`, `cuda`, `xpu`, `cpu`, or `rocm` |
| `--n_loops` | checkpoint default | Override RDT recurrence depth |
| `--verbose` | off | Print intermediate denoising states |

### 2. Use EOS-controlled or block generation

Let the model stop at EOS, up to a hard cap:

```bash
python sample.py \
  --checkpoint checkpoints/reasoning_sft/final.pt \
  --block --auto_length --max_length 2048 \
  --prompt "What is the mitochondrion?"
```

Generate a longer answer one block at a time while reusing committed KV state:

```bash
python sample.py \
  --checkpoint checkpoints/reasoning_sft/final.pt \
  --block --block_size 256 \
  --prompt "Write a detailed explanation of" \
  --length 2048 --steps 20
```

The repository currently provides a command-line sampler, not an HTTP inference server.
Wrap `yeoubi.sample` in your serving framework if you need a web API.

### 3. Use legacy thinking only when the checkpoint supports it

```bash
python sample.py \
  --checkpoint checkpoints/legacy_think.pt \
  --thinking --adaptive_think --max_think_steps 128 \
  --prompt "Explain how neural networks learn" \
  --length 512 --answer_steps 20 --verbose
```

Thinking is disabled by default and exists for compatible legacy checkpoints and
ablations. `--adaptive_think` by itself does not add THINK positions to a checkpoint
that was not trained with them.

### 4. Export portable weights

```bash
python export.py \
  --checkpoint checkpoints/reasoning_sft/final.pt \
  --output_dir exports/yeoubi-1b \
  --format safetensors
```

This writes `model.safetensors`, `model_config.json`, and `export_metadata.json`. No
tokenizer files are created because byte IDs define the text interface.

### 5. Export packed, inference-only weights

```bash
python export.py \
  --checkpoint checkpoints/reasoning_sft/final.pt \
  --output_dir exports/yeoubi-1b-packed \
  --format safetensors --pack
```

`--pack` replaces each trainable `latent_weight` with `w_packed` and `scale_w`. The
result is much smaller but cannot be fine-tuned. Keep the original checkpoint.

GGUF runtimes such as llama.cpp cannot execute this architecture: it is a
bidirectional diffusion model, not an autoregressive decoder. Use the included runtime
and safetensors or PyTorch weights.

</details>

## Architecture

### Default recurrent-depth topology

The default `model_type="rdt"` topology is:

```text
UTF-8 bytes
  → local byte encoder (4 bytes → 1 latent patch)
  → 4 prelude blocks
  → 2 pre-recurrent bridge blocks
  → 2 recurrent blocks × 1–3 refinement loops
  → 2 post-recurrent bridge blocks
  → 4 coda blocks
  → local byte decoder (1 latent patch → 4 byte predictions)
```

The global stack is bidirectional. The diffusion timestep is reinjected at each
recurrence, loop dropout makes shorter loop prefixes useful, and inference may override
the recurrence count with `sample.py --n_loops`. Set `--model_type standard` for the
flat `BitDiffusionTransformer`.

### Default model configuration

| Setting | Default |
|---|---:|
| Stored parameters | Approximately 1B; exact count is logged at startup |
| Hidden size | 2,048 |
| Physical blocks | 14 |
| Attention heads | 16 |
| Head dimension | 128 |
| FFN size | 8,192 |
| Training context | 16,384 bytes |
| Global latent sequence | Up to 4,096 four-byte patches |
| RoPE | YaRN factor 4 beyond the original 4,096-byte window |
| Predictable IDs | 258: 256 bytes, PAD, and EOS |
| Absorbing mask ID | 258 |
| Embeddings | Tied by default |
| KV-cache precision | 3-bit; first position 4-bit |
| RDT recurrence | 1–3 loops |

`PAD=256`, `EOS=257`, and `MASK=258`. No BOS token is inserted. A mask may split a
multi-byte UTF-8 character, so intermediate and final samples can contain invalid byte
sequences; the sampler decodes these with replacement characters instead of failing.

### Training objective and quantization schedule

- The cosine absorbing-state process replaces selected byte positions with MASK.
- MDLM NELBO weighting is the default; uniform masked cross-entropy is available with
  `--loss_weighting uniform`.
- Full-precision latent weights are retained during training and ternarized with a
  straight-through estimator on each forward pass.
- The first 90% of training uses INT8 activations (W1.58A8).
- The last 10% uses hybrid INT4 inputs, TopK(55%) sparsification, and INT8
  intermediates (W1.58A4).
- Block-causal examples are mixed into training to support cached block diffusion.

### Reference training hyperparameters

| Parameter | Default | Notes |
|---|---:|---|
| Steps | 57,500 | 30.1B padded byte positions |
| Batch size | 2 | Per device |
| Gradient accumulation | 16 | 524,288 padded positions per optimizer step |
| Maximum sequence | 16,384 bytes | 4,096 global patches |
| AdamW peak LR | `2e-4` | Embeddings, norms, biases, and output head |
| Muon peak LR | `0.02` | 2D Transformer body matrices |
| Warmup | 4,000 steps | Linear warmup, then cosine decay |
| Minimum LR ratio | 0.1 | Cosine floor |
| Weight decay | 0.05 | AdamW group |
| Gradient clipping | 1.0 | Global norm |
| Mixed precision | BF16 | Enabled by default |
| Gradient checkpointing | Enabled | Reference estimate: ~29.5 GB on A100 40 GB |
| A4 fraction | 10% | Final 5,750 steps |
| Loss | MDLM NELBO | Continuous-time weighted masked loss |

## Data pipeline

`prepare_hf_jsonl.py` streams documents, applies filtering, exact and optional fuzzy
deduplication, handles PII according to the selected policy, and assigns complete
documents to stable train or validation splits before chunking. Its default byte-length
distribution is:

```text
128: 3%    256: 5%    512: 7%    1,024: 10%
2,048: 15%    4,096: 25%    8,192: 20%    16,384: 15%
```

Overlap is capped at 128 bytes and at one eighth of a chunk. The final chunk of each
document receives EOS. Read [`docs/DATASET_GUIDE.md`](docs/DATASET_GUIDE.md) before a
large build; the manifest records provenance but is not legal approval for every source.

## Packed low-bit inference

Training checkpoints contain full-precision latent weights and simulate quantization.
That path preserves trainability but does not provide a low-bit speedup. Calling
`pack_for_inference()` or exporting with `--pack` is a one-way conversion that:

1. ternarizes and packs four weights per byte;
2. stores one scale per packed matrix;
3. removes the trainable latent matrix; and
4. dispatches to a Triton kernel when supported, otherwise to a correctness-oriented
   CPU fallback.

| | Training / float simulation | Packed inference |
|---|---|---|
| Transformer weight storage | FP32 latent matrices | 2-bit ternary payload plus scales |
| Activation computation | Floating-point simulation of rounding | Integer dot-product path |
| Trainable | Yes | No |
| Expected speedup | None from packing | Hardware, shape, and batching dependent |

The theoretical ternary payload is `parameters / 4` bytes, but real exports are larger
because embeddings, normalization parameters, metadata, scales, padding, and any
non-ternary tensors remain. Run the included benchmark on the target device rather than
assuming an ideal speedup:

```bash
python scripts/bench_packed_linear.py --shapes 768,1024,2048,4096
python scripts/bench_packed_linear.py --batch 1 --seq 1024
```

## Scaling, context, and capability targets

All context values below are **bytes**, not subword tokens. For ordinary English text,
four bytes are a rough approximation of one conventional token, but the ratio varies
substantially for code and multilingual text.

The base window is a proposed training curriculum target. The extended window requires
continued long-context training, block-causal diffusion, cache-aware or sparse
attention, and retrieval evaluation. Configuration support alone does not demonstrate
effective context length.

| Stored model | Active per position | Type | Base train window | Extended target | Rough text-token equivalent at target |
|---:|---:|---|---:|---:|---:|
| 500M | 500M | Dense | 16K bytes | 32K bytes | ~8K |
| 750M | 750M | Dense | 16K bytes | 64K bytes | ~16K |
| 1B | 1B | Dense/RDT | 16K bytes | 64K bytes | ~16K |
| 3B | 3B | Dense | 32K bytes | 128K bytes | ~32K |
| 5B | 5B | Dense | 64K bytes | 256K bytes | ~64K |
| 7B | 7B | Dense | 64K bytes | 256K bytes | ~64K |
| 12B | 12B | Dense | 128K bytes | 512K bytes | ~128K |
| 32B | ~8B | MoE | 128K bytes | 1M bytes | ~256K |
| 70B | ~12B | MoE | 256K bytes | 2M bytes | ~512K |
| 500B | ~30B | MoE | 512K bytes | 4M bytes | ~1M |
| 1T | ~50B | MoE | 1M bytes | 8M bytes | ~2M |

### Expected capability by scale

These are hypotheses for successful pretraining and post-training, not benchmark claims.
Data quality, training bytes, active parameters, optimization, and evaluation matter more
than stored parameter count alone.

| Model | Research expectation |
|---:|---|
| 500M | Compact completion, extraction, classification, shallow QA, and local code completion; not dependable for reasoning |
| 750M | Better instruction following, multilingual fluency, summarization, and structured extraction than the 500M tier |
| 1B | Small assistant, basic coding/math, grounded summarization, and short verified multistep tasks |
| 3B | Strong compact assistant, document synthesis, RAG, code completion, and moderate reasoning after post-training |
| 5B | More robust instruction following, coding, tool use, mathematics, and long-document retrieval |
| 7B | Capable general assistant with stronger code/RAG and useful agent behavior after trajectory training |
| 12B | High-quality single-node assistant with stronger coding, mathematics, synthesis, and reasoning reliability |
| 32B MoE | Roughly 8B-active path with a broader expert bank; routing quality is critical |
| 70B MoE | Strong general and coding target if routing, data scale, and post-training succeed |
| 500B MoE | Frontier research candidate whose stored size is meaningful only with multi-trillion-byte training and evaluation |
| 1T MoE | Experimental frontier system dominated by data, routing, distributed-training, and post-training risk |

## Estimated training and inference cost

> [!IMPORTANT]
> These are order-of-magnitude planning ranges, not measured bills. The estimates were
> repriced on **2026-07-22** at a rounded **$4 per H100 GPU-hour**. At that time,
> [Runpod listed H100 SXM pods at about $2.99/hour](https://www.runpod.io/pricing),
> while [Lambda listed H100 instances around $3.99–$4.29 per GPU-hour](https://lambda.ai/pricing).
> Prices, availability, taxes, storage, networking, orchestration, and reserved/spot
> discounts vary. Recalculate `GPU-hours × your actual rate` before budgeting.

### Assumptions

- Four-byte latent patches and the improved 1–3-loop RDT.
- Mixed sequence lengths rather than padding every example to the maximum window.
- Approximately 40% training model-FLOP utilization.
- A conservative 2–3.5× effective dense-model training reduction relative to an
  otherwise similar full-resolution design; MoE dispatch is budgeted more cautiously.
- No assumed sample-efficiency gain from data cleaning.
- GPU compute only; dataset preparation, CPU RAM, storage, egress, failed runs,
  evaluation, and post-training are excluded.

Ternary inference savings do not transfer directly to training: training keeps latent
weights, optimizer state, and straight-through quantization. Twenty bytes are only
roughly five conventional English tokens, so the old `20 tokens/parameter` heuristic
cannot be copied directly. Dense capability budgets below use about 80 accepted bytes
per parameter. MoE budgets use the larger of `80 × active parameters` and
`20 × stored parameters`.

### Training budget

| Model | Pilot bytes | Capability bytes | Capability H100 GPU-hours | Compute rental at $4/GPU-hour | Base train window |
|---:|---:|---:|---:|---:|---:|
| 500M | 10B | 40B | 28–48 | **$112–$192** | 16K bytes |
| 750M | 15B | 60B | 72–120 | **$288–$480** | 16K bytes |
| 1B | 20B | 80B | 120–212 | **$480–$848** | 16K bytes |
| 3B | 60B | 240B | 1,100–1,900 | **$4.4K–$7.6K** | 32K bytes |
| 5B | 100B | 400B | 3,200–5,600 | **$12.8K–$22.4K** | 64K bytes |
| 7B | 140B | 560B | 6,200–11,000 | **$24.8K–$44K** | 64K bytes |
| 12B | 240B | 960B | 18,000–32,000 | **$72K–$128K** | 128K bytes |
| 32B MoE (~8B active) | 300B | 640B | 10,700–19,200 | **$42.8K–$76.8K** | 128K bytes |
| 70B MoE (~12B active) | 500B | 1.4T | 36,400–64,400 | **$145.6K–$257.6K** | 256K bytes |
| 500B MoE (~30B active) | 3T | 10T | 683K–1.17M | **$2.73M–$4.68M** | 512K bytes |
| 1T MoE (~50B active) | 6T | 20T | 2.25M–3.83M | **$9.0M–$15.32M** | 1M bytes |

The 1B/20B-byte reference run is a pilot. Scaling the 1B capability range linearly to
20B bytes gives roughly 30–53 H100 GPU-hours, or **$120–$212** at the stated rate.
Real pilot cost may be higher because short runs compile kernels, load data, validate,
and checkpoint for a larger fraction of wall time.

### Packed inference footprint

The packed-weight column is an approximate two-bit ternary payload, not total runtime
memory. FP16 is shown as a comparison point. Activations, non-ternary tensors, expert
workspace, and KV cache must also fit.

At 20 denoising steps, the arithmetic floor below targets about 80 generated bytes per
second—roughly 20 conventional English tokens per second. It excludes prompt prefill,
kernel launches, cache decompression, and distributed communication.

| Model | Approx. packed ternary payload | FP16 weight equivalent | Arithmetic floor at ~80 B/s | Suggested minimum serving class |
|---:|---:|---:|---:|---|
| 500M | 0.13 GB | 1 GB | ~0.5 TFLOPS | CPU, modern iGPU, or edge accelerator |
| 750M | 0.19 GB | 1.5 GB | ~0.8 TFLOPS | CPU/iGPU or small discrete GPU |
| 1B | 0.25 GB | 2 GB | ~1.0 TFLOPS | CPU/iGPU or 8 GB GPU |
| 3B | 0.75 GB | 6 GB | ~3.1 TFLOPS | 8–12 GB GPU |
| 5B | 1.25 GB | 10 GB | ~5.2 TFLOPS | 12–16 GB GPU |
| 7B | 1.75 GB | 14 GB | ~7.3 TFLOPS | 16–24 GB GPU |
| 12B | 3.0 GB | 24 GB | ~12.5 TFLOPS | 24–48 GB GPU |
| 32B MoE | 8.0 GB | 64 GB | ~8.3 TFLOPS active | 24–48 GB GPU; more for long context |
| 70B MoE | 17.5 GB | 140 GB | ~12.5 TFLOPS active | 48–80 GB GPU |
| 500B MoE | 125 GB | 1,000 GB | ~31 TFLOPS active | At least 2 × 80 GB GPUs |
| 1T MoE | 250 GB | 2,000 GB | ~52 TFLOPS active | At least 4 × 80 GB GPUs |

Full-sequence diffusion reprocesses the canvas at every denoising step and becomes
uneconomical at long context. The extended targets assume block diffusion with a
committed KV cache. Even then, cached decode grows roughly linearly with committed
context and prompt prefill remains expensive.

Inference can be weight-bandwidth-bound. Treat **1.2–2×** single-request speedup,
**1.5–3×** with useful batching, and anything higher as a benchmark question rather
than a promise.

## Repository layout

```text
yeoubi/
  byte_codec.py       UTF-8 byte encoding and fixed special IDs
  data.py             JSONL and pre-tokenized streaming loaders
  data_pipeline.py    dataset processing utilities
  diffusion.py        absorbing-mask schedule and loss
  model.py            BitLinear, attention, FFN/MoE, flat Transformer
  rdt.py              recurrent-depth Transformer (default)
  quantization.py     activation and KV-cache quantization
  kernels.py          packed ternary kernels and fallbacks
  muon.py             Muon optimizer
  train.py            training and post-training loop
  sample.py           full-canvas and block samplers
  export.py           safetensors/PyTorch export and packing

configs/data_mix.json dataset mixture and source quotas
docs/DATASET_GUIDE.md curation, chunking, and audit guide
docs/LONG_CONTEXT.md  long-context design and validation notes
scripts/pretokenize.py
scripts/bench_packed_linear.py
prepare_hf_jsonl.py
train.py / sample.py / export.py
```

## Validation

Install pytest if it is not already available, then run:

```bash
python -m pip install pytest
python -m pytest -q
```

Long-context configuration support should not be advertised as effective retrieval
without passing the long-context tests and a continued-training evaluation:

```bash
python -m pytest -q tests/test_long_context.py
```

## References

- Ma et al., [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)
- Wang et al., [BitNet a4.8: 4-bit Activations for 1-bit LLMs](https://arxiv.org/abs/2411.04965)
- Sahoo et al., [Simple and Effective Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)
- Lou et al., [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834)
- Peng et al., [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)
- Arriola et al., [Block Diffusion: Interpolating Between Autoregressive and Diffusion Language Models](https://arxiv.org/abs/2503.09573)
- Dao, [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- Zhang and Sennrich, [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)

## License and data notice

- **Model weights:** BigCode OpenRAIL-M v1.0; use restrictions apply.
- **Source code:** Apache License 2.0.
- **Training data:** mixed and source-specific licenses; review every dataset card.

The data manifest is provenance metadata, not legal approval. Reconfirm source terms,
PII handling, benchmark decontamination, and the intended license for resulting weights
before training or redistribution.
