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
from .sample import _remask_step
from .utils import (
    BitStats, ExpertStats, WandBLogger, count_parameters, drain_checkpoint_writer,
    force_utf8_console, load_checkpoint, log_expert_utilization, read_checkpoint,
    resolve_checkpoint_model_config, save_checkpoint, setup_logging,
    validate_config_dict_topology, validate_model_config_topology,
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
    # 57 500 steps × (2 batch × 16 accum × 16384 seq) ≈ 30.1B BYTES -- roughly
    # 7.5B subword-token equivalents at ~4 bytes/token, not 30B tokens.
    # For a 40B-byte run set max_steps=76_294 (40e9 / 524_288).
    # 524K bytes/step fits a 40 GB A100 (~29.5 GB total).
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
    # BitNet SubLN before o_proj / down_proj (see ModelConfig.use_sub_norm).
    use_sub_norm: bool = True
    # INT4 activation scale: a4.8 absmean (True) or legacy per-token absmax.
    # See ModelConfig.int4_absmean.
    int4_absmean: bool = True
    # Scaling applies at every position once enabled: gating it on batch length
    # would give one absolute position two different rotations.
    rope_scaling_type: str = "yarn"  # "none" | "linear" | "yarn"
    rope_scaling_factor: float = 4.0
    rope_original_max_seq_len: int = 4096
    t_embed_dim: int = 256
    # See ModelConfig.timestep_scale. Pass 1.0 to resume or fine-tune a
    # checkpoint trained before the scaled noise embedding.
    timestep_scale: float = 1000.0
    kv_cache_bits: int = 4
    kv_cache_bos_bits: int = 4
    # Tie input embedding and unembedding head (see ModelConfig.tie_embeddings).
    tie_embeddings: bool = True
    # Neural byte compression: the global model sees one latent per four byte
    # positions while the loss and sampler remain byte-level diffusion.
    use_byte_patches: bool = True
    byte_patch_size: int = 4
    byte_local_kernel: int = 15
    # Byte-resolution SwiGLU in the patch decoder, which is otherwise linear
    # (see BytePatchCodec). Costs about 5% per step. Pass false to resume or
    # fine-tune a checkpoint trained without it.
    byte_decoder_ffn: bool = True

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
    # of ~2-5 min on first step -- paid again by every resumed session, so
    # turn it off for short ones. Skipped on CPU; frames that fail to compile
    # run eagerly (see device.compile_model).
    compile_model: bool = True
    # Offload periodic checkpoint I/O to a background thread so training
    # resumes immediately after the state snapshot (GPU→CPU clone).
    async_checkpoint: bool = True
    # Maximum validation batches per evaluation run (reduce to speed up).
    val_max_batches: int = 50
    # Batch training examples of similar length together: sort this many
    # consecutive examples by length before cutting batches, so almost no
    # position is padding (about a third were). Each batch carries a loss
    # weight that keeps every document's share of the loss where random
    # batching had it (see data.LengthBucketedBatches). 0 restores random
    # batching. JSONL / .pt loaders only; validation is never bucketed.
    length_bucket_window: int = 128
    # Fraction of training samples masked in whole byte-patch groups instead
    # of byte by byte (see diffusion.apply_mask). 0 disables. Validation
    # always uses plain byte masking so its loss stays comparable.
    span_mask_prob: float = 0.5

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
    rdt_randomize_loops: bool = True
    rdt_adaptive_loops: bool = True
    rdt_loop_specific_norms: bool = True
    # Learned loop-index table instead of the near-constant sinusoid (see
    # RDTConfig.learned_loop_embedding). Pass false to resume older RDT runs.
    rdt_learned_loop_embedding: bool = True
    rdt_context_adaptive_loops: bool = True
    rdt_context_loop_threshold: int = 4096
    # See RDTConfig.detach_recurrent_anchor / .recurrent_state_init_std.
    # Default False lets the recurrent core's per-iteration reads of the
    # prelude anchor train the prelude, matching the recurrent-depth design.
    rdt_detach_recurrent_anchor: bool = False
    # 0.0 keeps the deterministic h_0. Set ~0.02-1.0 to train the core as a
    # contraction to a fixed point, which is the precondition for the
    # inference-time depth extrapolation --n_loops advertises.
    rdt_recurrent_state_init_std: float = 0.0

    # Match cached block-diffusion inference during a configurable fraction of
    # training. Zero preserves legacy fully bidirectional training.
    block_causal_prob: float = 0.5
    block_causal_min_size: int = 128
    block_causal_max_size: int = 512
    # Fraction of training micro-batches recast as one cached block-generation
    # step: a random block noised against a clean prefix conditioned at t=0.
    # That is the state BlockDiffusionSampler presents, and block-causal
    # masking alone never produces it (see _clean_prefix_batch). Each such
    # step supervises one block only, so keep it small; 0 disables.
    clean_prefix_prob: float = 0.1


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

    @staticmethod
    def _restore(o: torch.optim.Optimizer, sd) -> None:
        # Per-parameter state is keyed by position across param groups, so a
        # checkpoint written under a different group layout (e.g. before the
        # AdamW decay / no-decay split) cannot be mapped back; torch raises.
        # Start that optimizer's moments fresh instead of failing the resume.
        if len(sd["param_groups"]) != len(o.param_groups):
            logger.warning(
                "%s param-group layout changed (checkpoint=%d, current=%d); "
                "skipping its optimizer state restoration.",
                type(o).__name__, len(sd["param_groups"]), len(o.param_groups),
            )
            return
        o.load_state_dict(sd)

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
                self._restore(o, sd)
            return
        # Legacy format: a single optimizer's state_dict (pre-Muon checkpoints).
        if isinstance(state, dict) and "param_groups" in state and "state" in state:
            if len(self.optimizers) == 1:
                self._restore(self.optimizers[0], state)
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

    @staticmethod
    def _restore(s, sd) -> None:
        s.load_state_dict(sd)
        groups = s.optimizer.param_groups
        if len(s.base_lrs) != len(groups):
            # Checkpoint predates a param-group layout change. base_lrs came
            # back with the old length and LambdaLR zips against it, so the
            # extra groups would silently sit at their warmup-start lr of 0
            # for the rest of the run. Every group shares one base LR, so
            # rebuild from the optimizer and re-apply the restored step.
            s.base_lrs = [g["initial_lr"] for g in groups]
            for g, lam, base in zip(groups, s.lr_lambdas, s.base_lrs):
                g["lr"] = base * lam(s.last_epoch)

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
                self._restore(s, sd)
            return
        if isinstance(state, dict) and len(self.schedulers) == 1:
            self._restore(self.schedulers[0], state)
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

    Weight decay applies to matrices only. Decaying 1D tensors pulls RMSNorm
    gains (and the RDT ``alpha`` / ``A_raw`` / ``iter_gate`` scalars) toward
    zero rather than toward their identity value, which is never the intent.
    """
    params = list(params)
    groups = [
        {"params": [p for p in params if p.ndim >= 2], "weight_decay": cfg.weight_decay},
        {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0},
    ]
    groups = [g for g in groups if g["params"]]
    if cfg.use_8bit_adam:
        if _HAS_BNB:
            logger.info("AdamW group: bitsandbytes AdamW8bit (8-bit optimizer state).")
            return bnb.optim.AdamW8bit(
                groups, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay,
            )
        logger.warning(
            "use_8bit_adam=True but bitsandbytes is not installed; using fp32 "
            "torch.optim.AdamW. `pip install bitsandbytes` (CUDA) to enable."
        )
    return torch.optim.AdamW(
        groups, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
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

def _clean_prefix_batch(
    batch: dict, t: torch.Tensor, cfg: TrainConfig,
) -> Optional[tuple[dict, torch.Tensor, int]]:
    """Recast a host-side batch as one cached block-generation step.

    Block-causal masking alone noises every block at the same level. The block
    sampler instead denoises block k against *clean* earlier blocks whose K/V
    were computed at t=0. This reproduces that state in a single forward: a
    random block k keeps the sample's t, everything before it is frozen clean
    and conditioned at t=0, and everything after it is cut off (the sampler
    has not generated it, and block k could not attend to it anyway).

    Returns ``(batch, t_per_byte, block_bytes)``, the batch truncated to the
    end of block k with ``frozen_mask`` / ``loss_mask`` narrowed to that block,
    or None when no block holds a supervisable byte.

    ponytail: one block of loss per sequence, so clean_prefix_prob stays
    small. BD3-LM-style clean+noisy concatenation supervises every block at
    twice the length if this proves too thin.
    """
    patch = cfg.byte_patch_size if cfg.use_byte_patches else 1
    block_bytes = int(torch.randint(
        cfg.block_causal_min_size, cfg.block_causal_max_size + 1, (),
    ))
    # Whole patches, so the block grid the model derives matches this one.
    block_bytes = -(-block_bytes // patch) * patch

    attention = batch["attention_mask"].bool()
    maskable = attention
    if batch.get("loss_mask") is not None:
        maskable = maskable & batch["loss_mask"].bool()
    if batch.get("frozen_mask") is not None:
        maskable = maskable & ~batch["frozen_mask"].bool()
    # Only blocks with something to supervise: in SFT most early blocks are
    # all prompt, and a step spent on one would train nothing.
    columns = maskable.any(dim=0).nonzero().squeeze(1)
    if columns.numel() == 0:
        return None
    blocks = torch.unique(columns // block_bytes)
    start = int(blocks[torch.randint(blocks.numel(), ())]) * block_bytes
    T = attention.shape[1]
    end = min(T, start + block_bytes)

    out = {
        key: value[:, :end]
        if torch.is_tensor(value) and value.dim() == 2 and value.shape[1] == T
        else value
        for key, value in batch.items()
    }
    in_block = torch.arange(end, device=attention.device) >= start
    prefix = ~in_block
    frozen = out.get("frozen_mask")
    out["frozen_mask"] = (
        prefix.expand(attention.shape[0], -1) if frozen is None
        else frozen.bool() | prefix
    )
    loss_mask = out.get("loss_mask")
    out["loss_mask"] = (
        out["attention_mask"].bool() if loss_mask is None else loss_mask.bool()
    ) & in_block
    if out.get("correction_input_ids") is not None:
        # A paired mistake left in the prefix would make the "clean" context wrong.
        out["correction_input_ids"] = torch.where(
            prefix, out["input_ids"], out["correction_input_ids"],
        )
    t_per_byte = t.unsqueeze(1) * in_block.to(device=t.device, dtype=t.dtype)
    return out, t_per_byte, block_bytes


def _prepare_training_state(
    batch: dict,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    t: torch.Tensor,
    cfg: TrainConfig,
    mask_token_id: int,
    schedule: CosineSchedule,
    span_prob: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a noisy input and the exact positions supervised by the loss.

    Returns ``(noisy_ids, supervised, maskable)``. ``maskable`` is every
    position the stage *could* supervise and is the NELBO denominator: in
    pretraining it equals ``attention_mask``, in reasoning SFT / correction it
    excludes the frozen prompt. Dividing by all valid positions there made the
    loss scale -- and so the effective learning rate -- track each batch's
    response-to-prompt ratio.
    """
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
        span_size=cfg.byte_patch_size, span_prob=span_prob,
    )
    maskable = loss_mask & attention_mask
    supervised = (is_masked | source_ids.ne(input_ids)) & maskable
    return noisy_ids, supervised, maskable

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
    val_seed: int = 1234,
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
        val_seed: Fixed seed for the noise draw. Validation samples t and the
            mask pattern from a forked RNG re-seeded to this value on every
            call, so successive evaluations score the *same* corruption of the
            same batches. Drawing fresh noise each time made the reported
            val loss dominated by which t happened to come up -- with
            max_batches=50 and a small batch size the run-to-run spread
            swamped the training signal, so the curve could not be read.

    Returns:
        Average validation loss.
    """
    was_training = model.training
    model.eval()
    model_core = getattr(model, "_orig_mod", model)
    total_loss = 0.0
    count = 0
    seen = 0
    use_amp = device.type in ("cuda", "xpu")

    # Fork the RNG so re-seeding does not perturb the training stream. CUDA
    # generators must be listed explicitly; other accelerators fall back to
    # forking the host RNG only, which still fixes t (mask draws follow it).
    fork_devices = [device] if device.type == "cuda" else []

    try:
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(val_seed)
            for i, batch in enumerate(val_loader):
                if i >= max_batches:
                    break
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)  # (B, T) bool
                B, T = input_ids.shape

                # Low-discrepancy t: the additive golden-ratio sequence covers
                # [0, 1) evenly for any number of samples, so the few hundred
                # scored here no longer under- or over-sample the low-t tail
                # where the MDLM weight is largest. A fixed grid would need
                # the batch count in advance and skew whenever the loader ran
                # short of max_batches.
                t = ((seen + torch.arange(B, device=device)) * 0.6180339887498949 + 0.5) % 1.0
                seen += B
                if cfg is not None and cfg.training_stage == "correction":
                    t = t * cfg.correction_t_max
                loss_valid_mask = attention_mask
                if cfg is None:
                    masked_ids, is_masked = apply_mask(
                        input_ids, t, mask_token_id, schedule,
                        frozen_mask=~attention_mask,
                    )
                else:
                    masked_ids, is_masked, loss_valid_mask = _prepare_training_state(
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
                    # Clean-vocab logits only, as in training (see train()).
                    loss = loss_fn(
                        logits[..., :model_core.config.vocab_size], input_ids,
                        is_masked, t=t, valid_mask=loss_valid_mask,
                    )

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

        # Uniform in mask probability rather than in t: equal commit fraction per step.
        # Linear-in-t spacing commits ~23% of the canvas in the first 3 of 20 steps,
        # when the model has no context to condition on.
        p_steps = torch.linspace(1.0, 0.0, steps + 1, device=device)
        t_steps = CosineSchedule.inverse(p_steps).clamp(0.0, 1.0)

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
                # Shared with the real samplers so the monitor cannot drift
                # from them again: it kept re-masking against p(t_next) alone
                # after they moved to p(t_next) / p(t_curr), committing most
                # of the canvas in the first few steps and under-selling the
                # model in every logged sample.
                ids = _remask_step(
                    ids, is_masked, schedule, t_curr, t_next, mask_token_id, gen,
                )

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
    pad_token_id: int,
    reshuffle_each_epoch: bool = False,
    bucket_window: int = 0,
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
            reshuffle_each_epoch=reshuffle_each_epoch,
            bucket_window=bucket_window,
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
        pad_token_id=pad_token_id,
        reshuffle_each_epoch=reshuffle_each_epoch,
        bucket_window=bucket_window,
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
    if cfg.model_type not in {"rdt", "standard"}:
        raise ValueError(f"model_type must be 'rdt' or 'standard', got {cfg.model_type!r}")
    if not 0.0 <= cfg.self_condition_prob <= 1.0:
        raise ValueError("self_condition_prob must be between 0 and 1")
    if not 0.0 <= cfg.correction_noise_prob <= 1.0:
        raise ValueError("correction_noise_prob must be between 0 and 1")
    if not 0.0 < cfg.correction_t_max <= 1.0:
        raise ValueError("correction_t_max must be in (0, 1]")
    if not 0.0 <= cfg.span_mask_prob <= 1.0:
        raise ValueError("span_mask_prob must be between 0 and 1")
    if cfg.length_bucket_window < 0:
        raise ValueError("length_bucket_window must be >= 0")

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
        # use_sub_norm was declared on TrainConfig but never forwarded, so
        # `--use_sub_norm false` was silently ignored.
        use_sub_norm=cfg.use_sub_norm,
        int4_absmean=cfg.int4_absmean,
        block_causal_prob=cfg.block_causal_prob,
        block_causal_min_size=cfg.block_causal_min_size,
        block_causal_max_size=cfg.block_causal_max_size,
        mask_token_id=mask_token_id,
        topk_ratio=cfg.topk_ratio,
        dropout=cfg.dropout,
        t_embed_dim=cfg.t_embed_dim,
        timestep_scale=cfg.timestep_scale,
        kv_cache_bits=cfg.kv_cache_bits,
        kv_cache_bos_bits=cfg.kv_cache_bos_bits,
        tie_embeddings=cfg.tie_embeddings,
        use_self_cond=cfg.use_self_cond,
        use_byte_patches=cfg.use_byte_patches,
        byte_patch_size=cfg.byte_patch_size,
        byte_local_kernel=cfg.byte_local_kernel,
        byte_decoder_ffn=cfg.byte_decoder_ffn,
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
            randomize_loops=cfg.rdt_randomize_loops,
            adaptive_loops=cfg.rdt_adaptive_loops,
            loop_specific_norms=cfg.rdt_loop_specific_norms,
            learned_loop_embedding=cfg.rdt_learned_loop_embedding,
        )
        base["context_adaptive_loops"] = cfg.rdt_context_adaptive_loops
        base["context_loop_threshold"] = cfg.rdt_context_loop_threshold
        base["detach_recurrent_anchor"] = cfg.rdt_detach_recurrent_anchor
        base["recurrent_state_init_std"] = cfg.rdt_recurrent_state_init_std
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
        # No tensor depends on timestep_scale, so load_state_dict cannot catch
        # a mismatch; the noise embedding would just be fed a scale it was
        # never fit to. A stored config without the key predates it (1.0).
        stored_scale = (checkpoint.get("model_config") or {}).get("timestep_scale", 1.0)
        if stored_scale != model_cfg.timestep_scale:
            raise ValueError(
                f"Fine-tuning checkpoint was trained with timestep_scale={stored_scale}, "
                f"but this run uses {model_cfg.timestep_scale}. "
                f"Pass --timestep_scale {stored_scale}."
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
    # On by default. Skipped on CPU: those runs are smoke tests, and the
    # compile would cost far more than the handful of steps it speeds up.
    if cfg.compile_model and device.type != "cpu":
        model = _compile_model(model, device, dynamic=True)

    # Data
    train_paths = sorted(glob.glob(cfg.train_data))
    val_paths = sorted(glob.glob(cfg.val_data))
    if not train_paths:
        raise FileNotFoundError(f"No training files found matching: {cfg.train_data}")
    logger.info("Training files: %d, Validation files: %d", len(train_paths), len(val_paths))

    # Train only: the validation loader must keep one fixed order (see
    # data._epoch_shuffle_seed). litdata reshuffles per epoch on its own.
    train_loader = _build_dataloader(
        train_paths, cfg, pad_token_id, reshuffle_each_epoch=True,
        bucket_window=cfg.length_bucket_window,
    )
    val_loader = None
    if val_paths:
        val_loader = _build_dataloader(
            val_paths, cfg, pad_token_id
        )

    # Loss, optimizer, scheduler
    # Correction also supervises *visible* wrong bytes. The MDLM weight ~2/t is
    # self-bounding only for masked positions, whose count shrinks with t^2;
    # on visible ones it reaches ~2000 at t_min, so the lowest-noise tenth of
    # samples carried about two thirds of the weight. Plain mean CE is the
    # bounded choice there.
    loss_weighting = cfg.loss_weighting
    if cfg.training_stage == "correction" and loss_weighting != "uniform":
        logger.info("Correction stage: overriding loss_weighting=%s with uniform.", loss_weighting)
        loss_weighting = "uniform"
    loss_fn = MaskDiffusionLoss(weighting=loss_weighting, t_min=cfg.loss_t_min)
    logger.info("Loss weighting: %s (t_min=%.1e)", loss_weighting, cfg.loss_t_min)
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
    if cfg.resume_from:
        if not os.path.isfile(cfg.resume_from):
            raise FileNotFoundError(f"resume_from checkpoint not found: {cfg.resume_from}")
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
        # ModelConfig topology alone isn't enough for RDT: RDTConfig (a
        # ModelConfig subclass) carries its own layer/loop/LoRA topology in a
        # separate "rdt_config" checkpoint key, invisible to the check above.
        if cfg.model_type == "rdt":
            if resume_ckpt.get("rdt_config") is None:
                logger.warning(
                    "Resume checkpoint %s has no serialized rdt_config; skipping RDT topology validation.",
                    cfg.resume_from,
                )
            else:
                # Imported here, not at module top, to avoid a train<->rdt
                # circular import (rdt.py has no reason to import train.py,
                # but utils.py — imported by both — must not import rdt).
                from .rdt import resolve_rdt_config
                checkpoint_rdt_cfg = resolve_rdt_config(resume_ckpt)
                validate_config_dict_topology(
                    asdict(rdt_cfg),
                    asdict(checkpoint_rdt_cfg),
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
    # Accumulated on-device and read once per log interval: a per-micro-step
    # .item() forces a host sync that drains the CUDA pipeline for a number
    # nothing reads until step % 50 == 0.
    running_loss = torch.zeros((), device=device)
    running_grad_norm = torch.zeros((), device=device)
    running_masked_frac = torch.zeros((), device=device)
    step_in_accum = 0
    epoch = 0

    while global_step < cfg.max_steps:
        epoch += 1
        for batch in train_loader:
            if global_step >= cfg.max_steps:
                break

            # Read on the host, before the copy: asking the device would cost
            # a sync on every micro-step.
            unpadded = bool(batch["attention_mask"].all())
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)  # (B, T) bool
            B, T = input_ids.shape

            # Low-discrepancy t (as in MDLM): one random offset per optimizer
            # step, with that step's B * grad_accum_steps samples spread
            # evenly over [0, 1). Independent draws left each update's mix of
            # noise levels to chance, and under the ~2/t loss weight a step
            # that happened to draw low t looked nothing like one that did not.
            if step_in_accum == 0:
                t_offset = torch.rand((), device=device)
            t = (
                t_offset
                + (step_in_accum * B + torch.arange(B, device=device))
                / (B * cfg.grad_accum_steps)
            ) % 1.0
            # Correction fine-tuning concentrates on partially corrupted states;
            # ordinary pre-training and reasoning SFT retain the full noise range.
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

            # --- Clean-prefix block step (see _clean_prefix_batch) ---
            t_model = t
            block_bytes = 0
            if (
                cfg.clean_prefix_prob > 0.0
                and not use_thinking
                and torch.rand(1).item() < cfg.clean_prefix_prob
            ):
                recast = _clean_prefix_batch(batch, t, cfg)
                if recast is not None:
                    batch, t_model, block_bytes = recast
                    unpadded = bool(batch["attention_mask"].all())
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    B, T = input_ids.shape

            if use_thinking:
                # Prepend N_think think tokens to the sequence
                think_prefix = think_schedule.make_think_prefix(B, device)
                input_ids = torch.cat([think_prefix, input_ids], dim=1)  # (B, N_think + T)
                T_full = input_ids.shape[1]
                is_think = think_schedule.is_think_position(T_full, device)  # (T_full,)
                # Extend attention_mask: think positions are always real tokens
                think_attn = torch.ones(B, model_cfg.N_think, dtype=torch.bool, device=device)
                attention_mask = torch.cat([think_attn, attention_mask], dim=1)

            loss_valid_mask = attention_mask
            if use_thinking:
                # Legacy augmentation remains available for checkpoint-compatible
                # experiments, but is bypassed by all post-training stages.
                model_input_ids, supervised_mask = apply_mask(
                    input_ids, t, mask_token_id, schedule, frozen_mask=~attention_mask,
                )
            else:
                model_input_ids, supervised_mask, loss_valid_mask = _prepare_training_state(
                    batch=batch,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    t=t,
                    cfg=cfg,
                    mask_token_id=mask_token_id,
                    schedule=schedule,
                    span_prob=cfg.span_mask_prob,
                )
            running_masked_frac += (
                supervised_mask.detach().float().mean() / cfg.grad_accum_steps
            )

            # Forward
            with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                self_cond = None
                if cfg.use_self_cond and torch.rand(1).item() < cfg.self_condition_prob:
                    with torch.no_grad():
                        previous_logits, _ = model(
                            model_input_ids,
                            t_model,
                            attention_mask=attention_mask,
                            unpadded=unpadded,
                            block_causal_block_size=block_bytes,
                        )
                        self_cond = self_cond_vector(
                            previous_logits,
                            model_core.embed.weight,
                            model_cfg.vocab_size,
                        )

                # block_bytes is 0 outside clean-prefix steps, which leaves the
                # model's own block-causal coin flip in charge.
                logits, aux_loss = model(
                    model_input_ids,
                    t_model,
                    self_cond=self_cond,
                    attention_mask=attention_mask,
                    unpadded=unpadded,
                    block_causal_block_size=block_bytes,
                )
                # Normalise over the clean vocabulary only. MASK (and legacy
                # THINK) are never targets and every sampler slices them off
                # before its softmax, so including them here trained a
                # different distribution from the one that gets sampled.
                diffusion_loss = loss_fn(
                    logits[..., :model_cfg.vocab_size],
                    input_ids,
                    supervised_mask,
                    is_think=is_think,
                    t=t,
                    valid_mask=loss_valid_mask,
                )
                # Length-bucketed batches carry the weight that keeps each
                # document's share of the loss where random batching had it.
                batch_weight = batch.get("loss_weight")
                if batch_weight is not None:
                    diffusion_loss = diffusion_loss * batch_weight.to(device)
                # Total loss: diffusion + MoE load balance
                moe_loss = aux_loss * model_cfg.aux_loss_weight if model_cfg.use_moe else 0.0
                loss = (diffusion_loss + moe_loss) / cfg.grad_accum_steps

            # Backward
            scaler.scale(loss).backward()
            running_loss += loss.detach()
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
            running_grad_norm += grad_norm.detach()
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
                avg_loss = running_loss.item() / 50
                avg_grad_norm = running_grad_norm.item() / 50
                avg_masked_frac = running_masked_frac.item() / 50
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
                running_loss.zero_()
                running_grad_norm.zero_()
                running_masked_frac.zero_()

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
