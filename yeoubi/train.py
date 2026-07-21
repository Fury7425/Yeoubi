# SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
# Model weights derived from this code are subject to the BigCode OpenRAIL-M license.
# Source code is licensed under Apache 2.0. See LICENSE for details.
"""
Training loop for Yeoubi.

Implements:
- TrainConfig with all hyperparameters
- ActivationSchedule for two-stage (A8 → A4) training
- Full training loop with mixed precision, gradient accumulation,
  checkpoint saving/loading, validation, and WandB logging
"""

from __future__ import annotations

import glob
import logging
import math
import os
import signal
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from .device import (
    compile_model as _compile_model,
    configure_tf32,
    empty_cache,
    get_amp_settings,
    log_device_info,
    resolve_device,
    seed_device,
)


# ---------------------------------------------------------------------------
# GradScaler no-op — used on XPU (bf16 doesn't overflow) and CPU
# ---------------------------------------------------------------------------

class _NoOpScaler:
    """Drop-in for GradScaler when loss scaling is not needed."""
    def is_enabled(self) -> bool: return False
    def scale(self, loss: torch.Tensor) -> torch.Tensor: return loss
    def unscale_(self, optimizer) -> None: pass
    def step(self, optimizer) -> None: optimizer.step()
    def update(self) -> None: pass

from .data import make_dataloader, make_litdata_dataloader, make_pretokenized_dataloader
from .byte_codec import (
    BYTE_VOCAB_SIZE, MASK_TOKEN_ID, PAD_TOKEN_ID, VOCAB_SIZE, decode_byte_ids,
)
from .diffusion import (
    CosineSchedule, MaskDiffusionLoss, ThinkingMaskSchedule, apply_mask,
)
from .model import BitDiffusionTransformer, ModelConfig, self_cond_vector
from . import MODEL_NAME
from .muon import Muon, split_params_for_muon
from .quantization import KVCache
from .utils import (
    BitStats, ExpertStats, WandBLogger, count_parameters, drain_checkpoint_writer,
    force_utf8_console, load_checkpoint, log_expert_utilization, read_checkpoint,
    resolve_checkpoint_model_config, save_checkpoint, setup_logging,
    validate_model_config_topology,
)

# Optional: bitsandbytes 8-bit AdamW (block-wise 8-bit optimizer state).
# Cuts AdamW optimizer-state memory ~4x (fp32 m+v = 8 bytes/param -> ~2 bytes/
# param) for the embedding / unembedding-head / RMSNorm group, freeing VRAM for
# a larger batch. The Muon group (2D transformer-body matrices) is untouched and
# keeps its single fp32 momentum buffer. Falls back to fp32 torch.optim.AdamW
# when not installed — `pip install bitsandbytes` (CUDA) to enable.
try:
    import bitsandbytes as bnb  # type: ignore
    _HAS_BNB = True
except Exception:  # noqa: BLE001 - optional dependency, absent by default
    bnb = None  # type: ignore[assignment]
    _HAS_BNB = False

logger = logging.getLogger("yeoubi")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training hyperparameters.

    Args:
        train_data: Glob pattern or list of paths to training JSONL files.
        val_data: Glob pattern or path to validation JSONL file.
        output_dir: Directory for checkpoints and logs.
        max_steps: Total training steps.
        batch_size: Per-device batch size.
        max_seq_len: Maximum sequence length.
        lr: Peak learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Linear warmup steps.
        grad_accum_steps: Gradient accumulation steps.
        grad_clip_norm: Maximum gradient norm for clipping.
        a4_warmup_fraction: Fraction of training for the A4 activation stage.
                            0.10 means the last 10% of steps use A4 mode.
        save_every: Save checkpoint every N steps.
        val_every: Run validation every N steps.
        bitstats_every: Log BitStats every N steps.
        num_workers: DataLoader workers.
        seed: Random seed.
        resume_from: Path to checkpoint to resume from.
        wandb_project: WandB project name (empty to disable).
        bf16: Use bf16 mixed precision.
        device: Device string.
    """
    train_data: str = "data/train/*.jsonl"
    val_data: str = "data/val/*.jsonl"
    output_dir: str = "checkpoints"
    # 57 500 steps × (8 batch × 16 accum × 4096 seq) ≈ 30.1B tokens.
    # For a 40B-token run set max_steps=76_294 (40e9 / 524_288).
    # batch_size=8 keeps 524K tok/step on a 40 GB A100 (~29.5 GB total).
    # 4096 context covers all standard benchmarks and beats every published
    # masked diffusion LM. Scale to 8192+ with a multi-GPU cluster.
    max_steps: int = 57500
    # Four-byte latent patches make 16K bytes a 4K-position global sequence.
    # Batch 2 preserves the previous per-step byte budget of batch 8 at 4K.
    batch_size: int = 2
    max_seq_len: int = 16384
    lr: float = 2e-4
    weight_decay: float = 0.05
    warmup_steps: int = 4000
    grad_accum_steps: int = 16
    grad_clip_norm: float = 1.0
    min_lr_ratio: float = 0.1       # cosine floor: decays to 10% of peak LR, not 0
    a4_warmup_fraction: float = 0.10
    # Loss weighting: "mdlm" applies the continuous-time MDLM NELBO weight
    # w(t) per masked token (better sample-efficiency); "uniform" is plain
    # mean cross-entropy over masked positions (legacy behaviour).
    loss_weighting: str = "mdlm"
    loss_t_min: float = 1e-3        # clamp t away from 0 to bound the MDLM weight

    # Training stages:
    # - pretrain: ordinary full-text masked diffusion
    # - reasoning_sft: clean/frozen prompt, response-only corruption and loss
    # - correction: reasoning SFT plus paired or synthetic response mistakes
    training_stage: str = "pretrain"
    finetune_from: str = ""          # initialize model weights only; do not restore optimizer
    assistant_separator: str = "\nAssistant:\n"

    # Fine-tuning self-conditioning. The first prediction is no-grad and its
    # soft distribution is supplied to the supervised second prediction.
    use_self_cond: bool = False
    self_condition_prob: float = 0.5

    # Correction-stage controls. Paired ``corrupted_response`` data takes
    # precedence; otherwise real response bytes are replaced synthetically.
    correction_noise_prob: float = 0.15
    correction_t_max: float = 0.5
    save_every: int = 2500
    val_every: int = 500
    bitstats_every: int = 250
    num_workers: int = 4
    # Read training data from a litdata-optimized directory (resumable
    # mid-epoch, deterministic shuffle, balanced shards). When True,
    # --train_data / --val_data are treated as litdata dataset directories
    # produced by scripts/optimize_litdata.py, not .jsonl / .pt globs.
    use_litdata: bool = False
    seed: int = 42
    resume_from: str = ""
    trust_checkpoint: bool = False
    wandb_project: str = "yeoubi"
    bf16: bool = True
    device: str = "auto"  # "auto" | "cuda[:<n>]" | "xpu[:<n>]" | "cpu" | "rocm"

    # Model config fields (forwarded to ModelConfig)
    # Defaults match the 1B recommended config (~30B tokens)
    # Predictable byte vocabulary: 0..255 bytes, PAD=256, EOS=257. MASK=258.
    vocab_size: int = VOCAB_SIZE
    hidden_dim: int = 2048
    n_layers: int = 16
    n_heads: int = 16
    ffn_dim: int = 8192
    topk_ratio: float = 0.55
    dropout: float = 0.0
    qk_norm: bool = True
    # The scaling path is dormant inside the original window and activates
    # only when a batch crosses ``rope_original_max_seq_len``.
    rope_scaling_type: str = "yarn"  # "none" | "linear" | "yarn"
    rope_scaling_factor: float = 4.0
    rope_original_max_seq_len: int = 4096
    t_embed_dim: int = 256
    kv_cache_bits: int = 3
    kv_cache_bos_bits: int = 4
    # Tie input embedding and unembedding head (see ModelConfig.tie_embeddings).
    tie_embeddings: bool = True
    # Neural byte compression: the global model sees one latent per four byte
    # positions while the loss and sampler remain byte-level diffusion.
    use_byte_patches: bool = True
    byte_patch_size: int = 4
    byte_local_kernel: int = 15

    # Legacy thinking tokens are retained for ablations but disabled for new
    # runs. Reasoning is taught through the supervised response canvas instead.
    N_think: int = 0
    think_prob: float = 0.0

    # Mixture of Experts
    use_moe: bool = False
    n_experts: int = 8
    top_k_experts: int = 2
    moe_layers: str = "alternate"
    aux_loss_weight: float = 0.01
    expert_capacity_factor: float = 1.25

    # Training efficiency
    gradient_checkpointing: bool = True
    # Checkpoint every Nth block (1=all, 2=every other).  Set to 2 on
    # memory-abundant GPUs to cut recompute overhead ~15 %.
    gc_every_n_layers: int = 1
    # Compile the model with torch.compile for GPU kernel fusion (~20-40 %
    # throughput gain on Ampere/Hopper).  Adds a one-time compilation cost
    # of ~2-5 min on first step.
    compile_model: bool = False
    # Offload periodic checkpoint I/O to a background thread so training
    # resumes immediately after the state snapshot (GPU→CPU clone).
    async_checkpoint: bool = True
    # Maximum validation batches per evaluation run (reduce to speed up).
    val_max_batches: int = 50

    # --- Muon optimizer (DeepSeek V4 style hybrid) ---
    # Muon orthogonalizes the SGD-momentum update of every 2D weight matrix
    # before applying it. We use Muon for the transformer body's BitLinear /
    # Int8Linear weights and AdamW for embeddings, RMSNorm gains, biases, and
    # the unembedding head. Set use_muon=False to fall back to AdamW for all.
    use_muon: bool = True
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_weight_decay: float = 0.0

    # Use bitsandbytes 8-bit AdamW for the AdamW parameter group (embeddings,
    # unembedding head, RMSNorm gains, biases). ~4x smaller optimizer state on
    # that group -> frees VRAM for a larger batch. Muon group is unaffected.
    # Requires `pip install bitsandbytes` (CUDA); no-op fallback to fp32 AdamW
    # if the library is absent.
    use_8bit_adam: bool = False

    # --- Recurrent-Depth Transformer (OpenMythos integration) ---
    # RDT is the default model. Set model_type="standard" to train the flat
    # BitDiffusionTransformer instead.
    model_type: str = "rdt"  # "rdt" (default) or "standard"
    rdt_prelude_layers: int = 4
    rdt_pre_recurrent_layers: int = 2
    rdt_recurrent_layers: int = 2
    rdt_post_recurrent_layers: int = 2
    rdt_coda_layers: int = 4
    rdt_max_loop_iters: int = 3
    rdt_min_loop_iters: int = 1
    rdt_lora_rank: int = 32
    rdt_loop_dim: int = 64
    rdt_use_act: bool = False
    rdt_act_ponder_weight: float = 0.01
    rdt_randomize_loops: bool = True
    rdt_adaptive_loops: bool = True
    rdt_loop_specific_norms: bool = True
    rdt_context_adaptive_loops: bool = True
    rdt_context_loop_threshold: int = 4096

    # Match cached block-diffusion inference during a configurable fraction of
    # training. Zero preserves legacy fully bidirectional training.
    block_causal_prob: float = 0.5
    block_causal_min_size: int = 128
    block_causal_max_size: int = 512


# ---------------------------------------------------------------------------
# Activation schedule
# ---------------------------------------------------------------------------

class ActivationSchedule:
    """Manages the two-stage activation training schedule.

    Stage 1 (first ~90% of training): A8 mode — all activations are INT8.
    Stage 2 (final ~10% of training): A4 mode — hybrid INT4 + TopK-INT8.

    The transition point is ``max_steps * (1 - a4_warmup_fraction)``.

    Args:
        max_steps: Total number of training steps.
        a4_warmup_fraction: Fraction of total steps allocated to A4 mode.
    """

    def __init__(self, max_steps: int, a4_warmup_fraction: float = 0.10):
        self.max_steps = max_steps
        self.a4_warmup_fraction = a4_warmup_fraction
        self.transition_step = int(max_steps * (1.0 - a4_warmup_fraction))

    def get_mode(self, step: int) -> str:
        """Return the activation mode for the given step.

        Args:
            step: Current training step.

        Returns:
            ``"A8"`` or ``"A4"``.
        """
        return "A4" if step >= self.transition_step else "A8"

    def __repr__(self) -> str:
        return (f"ActivationSchedule(transition_step={self.transition_step}, "
                f"max_steps={self.max_steps}, a4_frac={self.a4_warmup_fraction})")


# ---------------------------------------------------------------------------
# LR scheduler helpers
# ---------------------------------------------------------------------------

def _cosine_with_warmup(
    step: int, warmup_steps: int, max_steps: int, min_lr_ratio: float = 0.1
) -> float:
    """Compute LR multiplier for cosine schedule with linear warmup.

    Decays to ``min_lr_ratio`` of peak LR rather than 0, preventing
    destabilisation in the final training steps.
    """
    if step < warmup_steps:
        return max(step, 0) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


# ---------------------------------------------------------------------------
# Hybrid optimizer / scheduler wrappers (Muon for matrices + AdamW for the rest)
# ---------------------------------------------------------------------------

class _HybridOptimizer:
    """Holds one or more torch optimizers and exposes a single state_dict API.

    Used to combine Muon (2D weight matrices) and AdamW (embeddings, norms,
    biases, head). Duck-types the small subset of the ``Optimizer`` API that
    the training loop and ``save_checkpoint`` rely on.
    """

    def __init__(self, *optimizers: torch.optim.Optimizer):
        if not optimizers:
            raise ValueError("_HybridOptimizer needs at least one optimizer")
        self.optimizers = list(optimizers)

    @property
    def param_groups(self):
        groups = []
        for o in self.optimizers:
            groups.extend(o.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for o in self.optimizers:
            o.step()

    def state_dict(self):
        return {"hybrid": [o.state_dict() for o in self.optimizers]}

    def load_state_dict(self, state) -> None:
        # New format: {"hybrid": [sd_per_optimizer, ...]}
        if isinstance(state, dict) and "hybrid" in state:
            payload = state["hybrid"]
            if len(payload) != len(self.optimizers):
                logger.warning(
                    "Hybrid optimizer count mismatch (checkpoint=%d, current=%d); "
                    "skipping optimizer state restoration.",
                    len(payload), len(self.optimizers),
                )
                return
            for o, sd in zip(self.optimizers, payload):
                o.load_state_dict(sd)
            return
        # Legacy format: a single optimizer's state_dict (pre-Muon checkpoints).
        if isinstance(state, dict) and "param_groups" in state and "state" in state:
            if len(self.optimizers) == 1:
                self.optimizers[0].load_state_dict(state)
            else:
                logger.warning(
                    "Loading legacy single-optimizer checkpoint into hybrid setup; "
                    "skipping optimizer state restoration."
                )
            return
        raise ValueError("Unrecognized optimizer state_dict format")


class _HybridScheduler:
    """Holds one or more LR schedulers, mirroring ``_HybridOptimizer``."""

    def __init__(self, *schedulers):
        if not schedulers:
            raise ValueError("_HybridScheduler needs at least one scheduler")
        self.schedulers = list(schedulers)

    def step(self) -> None:
        for s in self.schedulers:
            s.step()

    def state_dict(self):
        return {"hybrid": [s.state_dict() for s in self.schedulers]}

    def load_state_dict(self, state) -> None:
        if isinstance(state, dict) and "hybrid" in state:
            payload = state["hybrid"]
            if len(payload) != len(self.schedulers):
                logger.warning(
                    "Hybrid scheduler count mismatch (checkpoint=%d, current=%d); "
                    "skipping scheduler state restoration.",
                    len(payload), len(self.schedulers),
                )
                return
            for s, sd in zip(self.schedulers, payload):
                s.load_state_dict(sd)
            return
        if isinstance(state, dict) and len(self.schedulers) == 1:
            self.schedulers[0].load_state_dict(state)
            return
        raise ValueError("Unrecognized scheduler state_dict format")


def _make_adamw(params, cfg: "TrainConfig") -> torch.optim.Optimizer:
    """Build the AdamW-group optimizer.

    Returns bitsandbytes' block-wise 8-bit ``AdamW8bit`` when
    ``cfg.use_8bit_adam`` is set and the library imports, otherwise the stdlib
    ``torch.optim.AdamW``. Only the AdamW parameter group (embeddings,
    unembedding head, RMSNorm gains, biases) is affected — the Muon group is
    built separately and keeps full-precision momentum. Same hyperparameters
    either way, so checkpoints and LR schedules are unchanged.
    """
    if cfg.use_8bit_adam:
        if _HAS_BNB:
            logger.info("AdamW group: bitsandbytes AdamW8bit (8-bit optimizer state).")
            return bnb.optim.AdamW8bit(
                params, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay,
            )
        logger.warning(
            "use_8bit_adam=True but bitsandbytes is not installed; using fp32 "
            "torch.optim.AdamW. `pip install bitsandbytes` (CUDA) to enable."
        )
    return torch.optim.AdamW(
        params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )


def _build_optimizer_and_scheduler(
    model: nn.Module, cfg: "TrainConfig"
) -> tuple[_HybridOptimizer, _HybridScheduler]:
    """Construct the Muon+AdamW hybrid (or pure AdamW if ``use_muon=False``)
    along with matching cosine-with-warmup schedulers.

    Args:
        model: The model whose parameters will be optimized.
        cfg: Training config.

    Returns:
        ``(optimizer, scheduler)`` — both wrappers around one or two underlying
        torch optimizers/schedulers.
    """
    lr_lambda = lambda step: _cosine_with_warmup(  # noqa: E731
        step, cfg.warmup_steps, cfg.max_steps, cfg.min_lr_ratio
    )

    if cfg.use_muon:
        muon_params, adamw_params = split_params_for_muon(model)
        if not muon_params:
            logger.warning(
                "Muon enabled but no 2D matrix parameters were found; falling back to AdamW only."
            )
            adamw_opt = _make_adamw(model.parameters(), cfg)
            return (
                _HybridOptimizer(adamw_opt),
                _HybridScheduler(torch.optim.lr_scheduler.LambdaLR(adamw_opt, lr_lambda)),
            )

        adamw_opt = _make_adamw(adamw_params, cfg)
        muon_opt = Muon(
            muon_params,
            lr=cfg.muon_lr,
            momentum=cfg.muon_momentum,
            nesterov=cfg.muon_nesterov,
            ns_steps=cfg.muon_ns_steps,
            weight_decay=cfg.muon_weight_decay,
        )
        logger.info(
            "Hybrid optimizer: Muon (%d tensors / %.2fM params, lr=%.4f, momentum=%.2f) + "
            "AdamW (%d tensors / %.2fM params, lr=%.2e)",
            len(muon_params), sum(p.numel() for p in muon_params) / 1e6,
            cfg.muon_lr, cfg.muon_momentum,
            len(adamw_params), sum(p.numel() for p in adamw_params) / 1e6,
            cfg.lr,
        )
        return (
            _HybridOptimizer(adamw_opt, muon_opt),
            _HybridScheduler(
                torch.optim.lr_scheduler.LambdaLR(adamw_opt, lr_lambda),
                torch.optim.lr_scheduler.LambdaLR(muon_opt, lr_lambda),
            ),
        )

    adamw_opt = _make_adamw(model.parameters(), cfg)
    logger.info("Optimizer: AdamW (lr=%.2e, weight_decay=%.3f)", cfg.lr, cfg.weight_decay)
    return (
        _HybridOptimizer(adamw_opt),
        _HybridScheduler(torch.optim.lr_scheduler.LambdaLR(adamw_opt, lr_lambda)),
    )


# ---------------------------------------------------------------------------
# Post-training corruption and validation
# ---------------------------------------------------------------------------

def _prepare_training_state(
    batch: dict,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    t: torch.Tensor,
    cfg: TrainConfig,
    mask_token_id: int,
    schedule: CosineSchedule,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a noisy input and the exact positions supervised by the loss."""
    device = input_ids.device
    loss_mask = batch.get("loss_mask")
    if loss_mask is None:
        loss_mask = attention_mask
    else:
        loss_mask = loss_mask.to(device=device, dtype=torch.bool)

    frozen_mask = batch.get("frozen_mask")
    if frozen_mask is None:
        frozen_mask = ~attention_mask
    else:
        frozen_mask = frozen_mask.to(device=device, dtype=torch.bool) | ~attention_mask

    source_ids = input_ids
    if cfg.training_stage == "correction":
        paired = batch.get("correction_input_ids")
        source_ids = (
            paired.to(device) if paired is not None else input_ids.clone()
        )

        # Paired, model-generated mistakes are preferred. Positions still
        # equal to the clean target receive synthetic byte replacements so
        # correction training remains useful with clean-only SFT records.
        synthetic_candidates = (
            loss_mask & attention_mask & ~frozen_mask & source_ids.eq(input_ids)
        )
        replace = (
            torch.rand(source_ids.shape, device=device) < cfg.correction_noise_prob
        ) & synthetic_candidates
        random_bytes = torch.randint(
            0, BYTE_VOCAB_SIZE, source_ids.shape,
            device=device, dtype=source_ids.dtype,
        )
        source_ids = torch.where(replace, random_bytes, source_ids)

    noisy_ids, is_masked = apply_mask(
        source_ids, t, mask_token_id, schedule, frozen_mask=frozen_mask,
    )
    supervised = (is_masked | source_ids.ne(input_ids)) & loss_mask & attention_mask
    return noisy_ids, supervised

@torch.no_grad()
def validate(
    model: BitDiffusionTransformer,
    val_loader,
    loss_fn: MaskDiffusionLoss,
    schedule: CosineSchedule,
    mask_token_id: int,
    device: torch.device,
    max_batches: int = 50,
    amp_dtype: torch.dtype = torch.bfloat16,
    cfg: TrainConfig | None = None,
) -> float:
    """Run validation and return average loss.

    Args:
        model: The model in eval mode.
        val_loader: Validation DataLoader.
        loss_fn: MaskDiffusionLoss instance.
        schedule: CosineSchedule instance.
        mask_token_id: Mask token ID.
        device: Device.
        max_batches: Max batches to evaluate.

    Returns:
        Average validation loss.
    """
    was_training = model.training
    model.eval()
    model_core = getattr(model, "_orig_mod", model)
    total_loss = 0.0
    count = 0
    use_amp = device.type in ("cuda", "xpu")

    try:
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)  # (B, T) bool
            B, T = input_ids.shape

            t = torch.rand(B, device=device)
            if cfg is not None and cfg.training_stage == "correction":
                t = t * cfg.correction_t_max
            if cfg is None:
                masked_ids, is_masked = apply_mask(
                    input_ids, t, mask_token_id, schedule,
                    frozen_mask=~attention_mask,
                )
            else:
                masked_ids, is_masked = _prepare_training_state(
                    batch, input_ids, attention_mask, t, cfg,
                    mask_token_id, schedule,
                )

            with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                self_cond = None
                if getattr(model_core.config, "use_self_cond", False):
                    previous_logits, _ = model(
                        masked_ids, t, attention_mask=attention_mask
                    )
                    self_cond = self_cond_vector(
                        previous_logits,
                        model_core.embed.weight,
                        model_core.config.vocab_size,
                    )
                logits, _ = model(
                    masked_ids, t, attention_mask=attention_mask,
                    self_cond=self_cond,
                )
                loss = loss_fn(logits, input_ids, is_masked, t=t)

            total_loss += loss.item()
            count += 1
    finally:
        if was_training:
            model.train()

    return total_loss / max(count, 1)


# ---------------------------------------------------------------------------
# Sample generation for qualitative monitoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_sample(
    model: BitDiffusionTransformer,
    mask_token_id: int,
    device: torch.device,
    seq_len: int = 64,
    steps: int = 10,
    temperature: float = 0.9,
    seed: int = 42,
) -> str:
    """Generate a sample using diffusion denoising for qualitative monitoring.

    Args:
        model: The model.
        mask_token_id: Mask token ID.
        device: Device.
        seq_len: Length of sequence to generate.
        steps: Number of denoising steps.
        temperature: Sampling temperature.
        seed: Random seed for reproducibility.

    Returns:
        Decoded string.
    """
    was_training = model.training
    model.eval()
    try:
        model_core = getattr(model, "_orig_mod", model)
        gen = torch.Generator(device=device).manual_seed(seed)
        schedule = CosineSchedule()
        vocab_size = model_core.config.vocab_size

        ids = torch.full((1, seq_len), mask_token_id, dtype=torch.long, device=device)

        t_steps = torch.linspace(1.0, 0.0, steps + 1, device=device)

        self_cond = None
        for i in range(steps):
            t_curr = t_steps[i]
            t_next = t_steps[i + 1]

            t_input = t_curr.unsqueeze(0)  # (1,)
            logits, _ = model(ids, t_input, self_cond=self_cond)  # (1, T, V)
            if getattr(model_core.config, "use_self_cond", False):
                self_cond = self_cond_vector(
                    logits, model_core.embed.weight, model_core.config.vocab_size,
                )

            # Slice to normal vocab only so special tokens (mask, think) cannot
            # be sampled and silently clamped.
            probs = torch.softmax(logits[:, :, :vocab_size] / temperature, dim=-1)
            flat_probs = probs.view(-1, probs.shape[-1])
            sampled = torch.multinomial(flat_probs, 1, generator=gen).view(1, seq_len)

            # Unmask sampled tokens, then re-mask positions that should remain
            # uncertain at the next noise level.
            is_masked = (ids == mask_token_id)
            ids = torch.where(is_masked, sampled, ids)
            if i + 1 < steps:
                mask_prob_next = schedule.mask_prob(t_next)
                rand = torch.rand(1, seq_len, device=device, generator=gen)
                should_remask = rand < mask_prob_next
                ids = torch.where(should_remask & is_masked, mask_token_id, ids)

        # Invalid/incomplete UTF-8 is rendered as U+FFFD instead of raising.
        text = decode_byte_ids(ids[0].tolist(), errors="replace")
    finally:
        if was_training:
            model.train()
    return text


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def _build_dataloader(
    paths: list[str],
    cfg: "TrainConfig",
    mask_token_id: int,
    pad_token_id: int,
):
    """Construct a DataLoader, auto-selecting the fast pre-tokenized path.

    If every file ends in ``.pt`` the corpus is treated as pre-tokenized
    shards (produced by ``data.pretokenize_dataset`` / ``scripts/pretokenize.py``)
    and the tokenizer-free :func:`make_pretokenized_dataloader` is used —
    typically 3-5x higher throughput because documents are not re-tokenized
    every epoch. Otherwise the JSONL streaming loader is used. A mix of
    ``.pt`` and ``.jsonl`` is rejected rather than silently mis-handled.

    When ``cfg.use_litdata`` is set, ``paths`` is treated as a single
    litdata-optimized dataset directory and the resumable streaming loader is
    used instead.
    """
    post_training = cfg.training_stage in {"reasoning_sft", "correction"}
    if post_training and cfg.use_litdata:
        raise ValueError(
            "reasoning_sft/correction require JSONL prompt/response masks; "
            "litdata pretraining shards do not contain them"
        )
    if cfg.use_litdata:
        input_dir = paths[0]
        logger.info("Using litdata StreamingDataset (%s) — resumable, sharded.", input_dir)
        return make_litdata_dataloader(
            input_dir, cfg.max_seq_len, cfg.batch_size, cfg.num_workers,
            pad_token_id=pad_token_id, shuffle=True,
        )

    suffixes = {os.path.splitext(p)[1].lower() for p in paths}
    if post_training and suffixes != {".jsonl"}:
        raise ValueError(
            "reasoning_sft/correction require JSONL records with prompt plus "
            "response or reasoning/answer fields"
        )
    if suffixes == {".pt"}:
        logger.info("Detected pre-tokenized .pt shards (%d) — using fast loader.", len(paths))
        return make_pretokenized_dataloader(
            paths, cfg.max_seq_len, cfg.batch_size, cfg.num_workers,
            pad_token_id=pad_token_id,
        )
    if ".pt" in suffixes and len(suffixes) > 1:
        raise ValueError(
            f"Mixed .pt and non-.pt files in data glob ({sorted(suffixes)}). "
            "Point --train_data at either all-.pt pre-tokenized shards or all-.jsonl files."
        )
    logger.info(
        "Using JSONL streaming loader (%d files). Tip: pre-tokenize once with "
        "scripts/pretokenize.py and pass the .pt shards for 3-5x loader throughput.",
        len(paths),
    )
    return make_dataloader(
        paths, cfg.max_seq_len, cfg.batch_size, cfg.num_workers,
        mask_token_id=mask_token_id, pad_token_id=pad_token_id,
        post_training=post_training,
        assistant_separator=cfg.assistant_separator,
    )


def train(cfg: TrainConfig) -> None:
    """Run the full training loop.

    Args:
        cfg: Training configuration.
    """
    setup_logging()

    valid_training_stages = {"pretrain", "reasoning_sft", "correction"}
    if cfg.training_stage not in valid_training_stages:
        raise ValueError(
            f"training_stage must be one of {sorted(valid_training_stages)}, "
            f"got {cfg.training_stage!r}"
        )
    if cfg.resume_from and cfg.finetune_from:
        raise ValueError("resume_from and finetune_from are mutually exclusive")
    if not 0.0 <= cfg.self_condition_prob <= 1.0:
        raise ValueError("self_condition_prob must be between 0 and 1")
    if not 0.0 <= cfg.correction_noise_prob <= 1.0:
        raise ValueError("correction_noise_prob must be between 0 and 1")
    if not 0.0 < cfg.correction_t_max <= 1.0:
        raise ValueError("correction_t_max must be in (0, 1]")

    # Device
    device = resolve_device(cfg.device)
    configure_tf32(device)   # no-op on non-CUDA; enables TF32 on Ampere+
    log_device_info(device)
    logger.info("Using device: %s", device)

    # Seed
    seed_device(device, cfg.seed)

    if cfg.vocab_size != VOCAB_SIZE:
        raise ValueError(
            f"Byte-level training requires vocab_size={VOCAB_SIZE}; got {cfg.vocab_size}."
        )
    vocab_size = VOCAB_SIZE
    mask_token_id = MASK_TOKEN_ID
    pad_token_id = PAD_TOKEN_ID
    logger.info(
        "UTF-8 byte input: vocab_size=%d, pad_token_id=%d, mask_token_id=%d",
        vocab_size, pad_token_id, mask_token_id,
    )

    # Model config
    model_cfg = ModelConfig(
        vocab_size=vocab_size,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        ffn_dim=cfg.ffn_dim,
        max_seq_len=cfg.max_seq_len,
        rope_scaling_type=cfg.rope_scaling_type,
        rope_scaling_factor=cfg.rope_scaling_factor,
        rope_original_max_seq_len=cfg.rope_original_max_seq_len,
        qk_norm=cfg.qk_norm,
        block_causal_prob=cfg.block_causal_prob,
        block_causal_min_size=cfg.block_causal_min_size,
        block_causal_max_size=cfg.block_causal_max_size,
        mask_token_id=mask_token_id,
        topk_ratio=cfg.topk_ratio,
        dropout=cfg.dropout,
        t_embed_dim=cfg.t_embed_dim,
        kv_cache_bits=cfg.kv_cache_bits,
        kv_cache_bos_bits=cfg.kv_cache_bos_bits,
        tie_embeddings=cfg.tie_embeddings,
        use_self_cond=cfg.use_self_cond,
        use_byte_patches=cfg.use_byte_patches,
        byte_patch_size=cfg.byte_patch_size,
        byte_local_kernel=cfg.byte_local_kernel,
        N_think=cfg.N_think,
        think_prob=cfg.think_prob,
        use_moe=cfg.use_moe,
        n_experts=cfg.n_experts,
        top_k_experts=cfg.top_k_experts,
        moe_layers=cfg.moe_layers,
        aux_loss_weight=cfg.aux_loss_weight,
        expert_capacity_factor=cfg.expert_capacity_factor,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gc_every_n_layers=cfg.gc_every_n_layers,
    )

    # Model
    if cfg.model_type == "rdt":
        from .rdt import BitRDTTransformer, RDTConfig
        base = asdict(model_cfg)
        base.update(
            use_rdt=True,
            prelude_layers=cfg.rdt_prelude_layers,
            pre_recurrent_layers=cfg.rdt_pre_recurrent_layers,
            recurrent_layers=cfg.rdt_recurrent_layers,
            post_recurrent_layers=cfg.rdt_post_recurrent_layers,
            coda_layers=cfg.rdt_coda_layers,
            max_loop_iters=cfg.rdt_max_loop_iters,
            min_loop_iters=cfg.rdt_min_loop_iters,
            lora_rank=cfg.rdt_lora_rank,
            loop_dim=cfg.rdt_loop_dim,
            use_act=cfg.rdt_use_act,
            act_ponder_weight=cfg.rdt_act_ponder_weight,
            randomize_loops=cfg.rdt_randomize_loops,
            adaptive_loops=cfg.rdt_adaptive_loops,
            loop_specific_norms=cfg.rdt_loop_specific_norms,
        )
        base["context_adaptive_loops"] = cfg.rdt_context_adaptive_loops
        base["context_loop_threshold"] = cfg.rdt_context_loop_threshold
        rdt_cfg = RDTConfig(**base)
        model = BitRDTTransformer(rdt_cfg).to(device)
        # Keep the plain model_cfg in sync so resume topology validation
        # (which compares use_rdt) matches RDT checkpoints instead of always
        # mismatching requested=False vs checkpoint=True.
        model_cfg.use_rdt = True
        logger.info(
            "BitRDTTransformer: prelude=%d, pre_bridge=%d, recurrent=%d, "
            "post_bridge=%d, coda=%d, loops=%d-%d, byte_patch=%d",
            cfg.rdt_prelude_layers, cfg.rdt_pre_recurrent_layers,
            cfg.rdt_recurrent_layers, cfg.rdt_post_recurrent_layers,
            cfg.rdt_coda_layers, cfg.rdt_min_loop_iters,
            cfg.rdt_max_loop_iters,
            cfg.byte_patch_size if cfg.use_byte_patches else 1,
        )
    else:
        model = BitDiffusionTransformer(model_cfg).to(device)

    if cfg.finetune_from:
        checkpoint = read_checkpoint(
            cfg.finetune_from,
            device="cpu",
            trust_checkpoint=cfg.trust_checkpoint,
        )
        incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        allowed_new_keys = {"self_cond_proj.weight"}
        missing = set(incompatible.missing_keys) - allowed_new_keys
        unexpected = set(incompatible.unexpected_keys) - allowed_new_keys
        if missing or unexpected:
            raise RuntimeError(
                "Fine-tuning checkpoint is not compatible with this model. "
                f"Missing keys: {sorted(missing)}; unexpected keys: {sorted(unexpected)}"
            )
        if incompatible.missing_keys:
            logger.info(
                "Initialized new fine-tuning parameters: %s",
                ", ".join(incompatible.missing_keys),
            )
        logger.info("Loaded fine-tuning weights from %s", cfg.finetune_from)

    param_info = count_parameters(model, model_cfg)
    logger.info("Parameter breakdown: %s", param_info)

    # Optional torch.compile — large one-time cost, significant steady-state gain
    if cfg.compile_model:
        model = _compile_model(model, device, dynamic=True)

    # Data
    train_paths = sorted(glob.glob(cfg.train_data))
    val_paths = sorted(glob.glob(cfg.val_data))
    if not train_paths:
        raise FileNotFoundError(f"No training files found matching: {cfg.train_data}")
    logger.info("Training files: %d, Validation files: %d", len(train_paths), len(val_paths))

    train_loader = _build_dataloader(
        train_paths, cfg, mask_token_id, pad_token_id
    )
    val_loader = None
    if val_paths:
        val_loader = _build_dataloader(
            val_paths, cfg, mask_token_id, pad_token_id
        )

    # Loss, optimizer, scheduler
    loss_fn = MaskDiffusionLoss(weighting=cfg.loss_weighting, t_min=cfg.loss_t_min)
    logger.info("Loss weighting: %s (t_min=%.1e)", cfg.loss_weighting, cfg.loss_t_min)
    schedule = CosineSchedule()
    act_schedule = ActivationSchedule(cfg.max_steps, cfg.a4_warmup_fraction)
    logger.info("Activation schedule: %s", act_schedule)

    optimizer, lr_scheduler = _build_optimizer_and_scheduler(model, cfg)

    # Mixed precision
    use_amp, use_bf16, amp_dtype = get_amp_settings(device, cfg.bf16)
    # GradScaler is only useful for CUDA fp16; bf16 and non-CUDA never need it.
    use_scaler = (not use_bf16) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_scaler) if use_scaler else _NoOpScaler()
    logger.info("Mixed precision: device=%s, dtype=%s, amp=%s, scaler=%s",
                device.type, amp_dtype, use_amp, scaler.is_enabled())

    # WandB
    wandb_logger = WandBLogger(
        project=cfg.wandb_project,
        config={**vars(cfg), **vars(model_cfg)},
        enabled=bool(cfg.wandb_project),
    )

    # BitStats and ExpertStats
    bit_stats = BitStats(model)
    expert_stats = ExpertStats(model) if model_cfg.use_moe else None

    # Thinking schedule
    think_schedule = None
    if cfg.training_stage == "pretrain" and model_cfg.N_think > 0:
        think_schedule = ThinkingMaskSchedule(model_cfg.N_think, model_cfg.think_token_id)

    # Resume
    global_step = 0
    current_mode = "A8"
    if cfg.resume_from and os.path.isfile(cfg.resume_from):
        resume_ckpt = read_checkpoint(
            cfg.resume_from,
            device="cpu",
            trust_checkpoint=cfg.trust_checkpoint,
        )
        try:
            resume_model_cfg, _ = resolve_checkpoint_model_config(resume_ckpt)
        except ValueError:
            resume_model_cfg = None
            logger.warning(
                "Resume checkpoint %s has no serialized model_config; skipping topology validation.",
                cfg.resume_from,
            )
        if resume_model_cfg is not None:
            validate_model_config_topology(
                model_cfg,
                resume_model_cfg,
                context=f"Resume checkpoint {cfg.resume_from}",
            )
        info = load_checkpoint(
            cfg.resume_from,
            model,
            optimizer,
            lr_scheduler,
            device,
            trust_checkpoint=cfg.trust_checkpoint,
        )
        global_step = info["step"]
        current_mode = info["activation_mode"]
        logger.info("Resumed from step %d, activation mode=%s", global_step, current_mode)

    # Set initial activation mode
    model.set_activation_mode(current_mode)
    if cfg.gradient_checkpointing:
        logger.info("Gradient checkpointing enabled — trading compute for memory")
    model.train()

    # --- SIGTERM handler (GCP preemptible VMs send SIGTERM 30s before kill) ---
    _stop_requested = False

    def _handle_sigterm(signum, frame):  # noqa: ANN001
        nonlocal _stop_requested
        logger.warning("SIGTERM received — saving checkpoint and exiting cleanly.")
        _stop_requested = True

    signal.signal(signal.SIGTERM, _handle_sigterm)

    # --- Training loop ---
    logger.info("Starting training for %d steps", cfg.max_steps)
    logger.info(
        "Training stage: %s (self-conditioning=%s, legacy thinking probability=%.3f)",
        cfg.training_stage,
        cfg.use_self_cond,
        cfg.think_prob if cfg.training_stage == "pretrain" else 0.0,
    )
    model_core = getattr(model, "_orig_mod", model)
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    running_grad_norm = 0.0
    running_masked_frac = 0.0
    step_in_accum = 0
    epoch = 0

    while global_step < cfg.max_steps:
        epoch += 1
        for batch in train_loader:
            if global_step >= cfg.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)  # (B, T) bool
            B, T = input_ids.shape

            # Correction fine-tuning concentrates on partially corrupted states;
            # ordinary pre-training and reasoning SFT retain the full noise range.
            t = torch.rand(B, device=device)
            if cfg.training_stage == "correction":
                t = t * cfg.correction_t_max

            # --- Thinking token augmentation ---
            is_think = None
            use_thinking = (
                cfg.training_stage == "pretrain"
                and
                think_schedule is not None
                and torch.rand(1).item() < model_cfg.think_prob
            )
            if use_thinking:
                # Prepend N_think think tokens to the sequence
                think_prefix = think_schedule.make_think_prefix(B, device)
                input_ids = torch.cat([think_prefix, input_ids], dim=1)  # (B, N_think + T)
                T_full = input_ids.shape[1]
                is_think = think_schedule.is_think_position(T_full, device)  # (T_full,)
                # Extend attention_mask: think positions are always real tokens
                think_attn = torch.ones(B, model_cfg.N_think, dtype=torch.bool, device=device)
                attention_mask = torch.cat([think_attn, attention_mask], dim=1)

            if use_thinking:
                # Legacy augmentation remains available for checkpoint-compatible
                # experiments, but is bypassed by all post-training stages.
                model_input_ids, supervised_mask = apply_mask(
                    input_ids, t, mask_token_id, schedule, frozen_mask=~attention_mask,
                )
            else:
                model_input_ids, supervised_mask = _prepare_training_state(
                    batch=batch,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    t=t,
                    cfg=cfg,
                    mask_token_id=mask_token_id,
                    schedule=schedule,
                )
            running_masked_frac += (
                supervised_mask.float().mean().item() / cfg.grad_accum_steps
            )

            # Forward
            with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                self_cond = None
                if cfg.use_self_cond and torch.rand(1).item() < cfg.self_condition_prob:
                    with torch.no_grad():
                        previous_logits, _ = model(
                            model_input_ids,
                            t,
                            attention_mask=attention_mask,
                        )
                        self_cond = self_cond_vector(
                            previous_logits,
                            model_core.embed.weight,
                            model_cfg.vocab_size,
                        )

                logits, aux_loss = model(
                    model_input_ids,
                    t,
                    self_cond=self_cond,
                    attention_mask=attention_mask,
                )
                diffusion_loss = loss_fn(
                    logits,
                    input_ids,
                    supervised_mask,
                    is_think=is_think,
                    t=t,
                )
                # Total loss: diffusion + MoE load balance
                moe_loss = aux_loss * model_cfg.aux_loss_weight if model_cfg.use_moe else 0.0
                loss = (diffusion_loss + moe_loss) / cfg.grad_accum_steps

            # Backward
            scaler.scale(loss).backward()
            running_loss += loss.item()
            step_in_accum += 1

            if step_in_accum < cfg.grad_accum_steps:
                continue

            # Optimizer step — iterate underlying optimizers so GradScaler
            # tracks per-optimizer state correctly when fp16 is in use. With
            # bf16 (the default) the scaler is disabled and these calls are
            # no-ops aside from optimizer.step() itself.
            for inner_opt in optimizer.optimizers:
                scaler.unscale_(inner_opt)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            running_grad_norm += float(grad_norm)
            for inner_opt in optimizer.optimizers:
                scaler.step(inner_opt)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1
            step_in_accum = 0

            # --- Activation mode transition ---
            new_mode = act_schedule.get_mode(global_step)
            if new_mode != current_mode:
                # Save a checkpoint right at the A8→A4 boundary so you can
                # ablate whether degradation came from before or after the switch.
                pre_a4_path = os.path.join(cfg.output_dir, f"step_{global_step}_pre_a4.pt")
                save_checkpoint(pre_a4_path, model, optimizer, lr_scheduler,
                                global_step, current_mode, async_io=cfg.async_checkpoint)
                logger.info("Pre-A4 ablation checkpoint saved to %s", pre_a4_path)
                current_mode = new_mode
                model.set_activation_mode(current_mode)
                logger.info("Step %d: activation mode → %s", global_step, current_mode)

            # --- Logging ---
            if global_step % 50 == 0:
                avg_loss = running_loss / 50
                avg_grad_norm = running_grad_norm / 50
                avg_masked_frac = running_masked_frac / 50
                # Per-optimizer LR (one value when AdamW-only, two with Muon).
                opt_lrs = [opt.param_groups[0]["lr"] for opt in optimizer.optimizers]
                lr = opt_lrs[0]  # primary (AdamW) LR for the headline log line.
                logger.info(
                    "step=%d  loss=%.4f  lr=%.2e  grad_norm=%.3f  masked_frac=%.3f  mode=%s",
                    global_step, avg_loss, lr, avg_grad_norm, avg_masked_frac, current_mode,
                )
                log_data = {
                    "train/loss": avg_loss,
                    "train/lr": lr,
                    "train/grad_norm": avg_grad_norm,
                    "train/masked_frac": avg_masked_frac,
                    "train/activation_mode": 0 if current_mode == "A8" else 1,
                    "train/epoch": epoch,
                }
                if len(opt_lrs) > 1:
                    log_data["train/lr_muon"] = opt_lrs[1]
                wandb_logger.log(log_data, step=global_step)
                running_loss = 0.0
                running_grad_norm = 0.0
                running_masked_frac = 0.0

            # --- MoE load balance loss logging (every 100 steps) ---
            if model_cfg.use_moe and global_step % 100 == 0:
                moe_log = log_expert_utilization(model, global_step)
                wandb_logger.log(moe_log, step=global_step)

            # --- BitStats ---
            if global_step % cfg.bitstats_every == 0:
                stats = bit_stats.log_to_console(global_step)
                wandb_logger.log(stats, step=global_step)
                if expert_stats is not None:
                    expert_data = expert_stats.compute()
                    wandb_logger.log(expert_data, step=global_step)

            # --- Validation ---
            if val_loader is not None and global_step % cfg.val_every == 0:
                val_loss = validate(
                    model, val_loader, loss_fn, schedule, mask_token_id, device,
                    max_batches=cfg.val_max_batches, amp_dtype=amp_dtype,
                    cfg=cfg,
                )
                logger.info("step=%d  val_loss=%.4f", global_step, val_loss)
                wandb_logger.log({"val/loss": val_loss}, step=global_step)

                # Qualitative sample
                sample_text = generate_sample(
                    model, mask_token_id, device,
                    seq_len=min(64, cfg.max_seq_len), steps=10, seed=cfg.seed,
                )
                logger.info("Sample generation:\n%s", sample_text)
                wandb_logger.log({"val/sample": sample_text}, step=global_step)

                empty_cache(device)
                model.train()

            # --- Preemption / SIGTERM ---
            if _stop_requested:
                drain_checkpoint_writer()  # flush any pending async write first
                ckpt_path = os.path.join(cfg.output_dir, f"step_{global_step}_preempted.pt")
                save_checkpoint(ckpt_path, model, optimizer, lr_scheduler,
                                global_step, current_mode)  # always synchronous on preemption
                logger.info("Preemption checkpoint saved to %s — exiting.", ckpt_path)
                wandb_logger.finish()
                return

            # --- Checkpoint ---
            if global_step % cfg.save_every == 0:
                ckpt_path = os.path.join(cfg.output_dir, f"step_{global_step}.pt")
                save_checkpoint(ckpt_path, model, optimizer, lr_scheduler,
                                global_step, current_mode, async_io=cfg.async_checkpoint)

    # Final checkpoint — drain async writer first so the file exists on exit
    drain_checkpoint_writer()
    ckpt_path = os.path.join(cfg.output_dir, "final.pt")
    save_checkpoint(ckpt_path, model, optimizer, lr_scheduler, global_step, current_mode)
    wandb_logger.finish()
    logger.info("Training complete at step %d", global_step)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments and launch training."""
    import argparse

    force_utf8_console()
    parser = argparse.ArgumentParser(description=f"Train {MODEL_NAME}")
    # Expose all TrainConfig fields as CLI arguments
    defaults = TrainConfig()
    for field_name, field_val in vars(defaults).items():
        ftype = type(field_val)
        if ftype is bool:
            parser.add_argument(f"--{field_name}", type=lambda x: x.lower() in ("true", "1", "yes"),
                                default=field_val, help=f"Default: {field_val}")
        else:
            parser.add_argument(f"--{field_name}", type=ftype, default=field_val,
                                help=f"Default: {field_val}")

    args = parser.parse_args()
    cfg = TrainConfig(**vars(args))
    train(cfg)


if __name__ == "__main__":
    main()
