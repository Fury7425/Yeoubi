# SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
# Model weights derived from this code are subject to the BigCode OpenRAIL-M license.
# Source code is licensed under Apache 2.0. See LICENSE for details.
"""
Diffusion denoising sampler for Yeoubi.

Loads a trained checkpoint and generates text by iteratively denoising
from a fully masked sequence. Supports prompt conditioning (frozen prefix),
temperature and top-p nucleus sampling, batch generation, and verbose
step-by-step output.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import asdict
from typing import Optional

import torch
import torch.nn.functional as F

from .byte_codec import EOS_TOKEN_ID, MASK_TOKEN_ID, VOCAB_SIZE, decode_byte_ids
from .diffusion import CosineSchedule, ThinkingMaskSchedule
from .model import BitDiffusionTransformer, ModelConfig, self_cond_vector
from . import MODEL_NAME
from .quantization import KVCache
from .utils import force_utf8_console, read_checkpoint, resolve_checkpoint_model_config, setup_logging


def _model_fwd(
    model, ids, t, *, kv_cache=None, rope_offset=0, n_loops=None,
    self_cond=None,
):
    """Unified model forward that passes n_loops only to BitRDTTransformer."""
    if n_loops is not None and hasattr(model, "rdt_config"):
        return model(
            ids, t, kv_cache=kv_cache, rope_offset=rope_offset,
            n_loops=n_loops, self_cond=self_cond,
        )
    return model(
        ids, t, kv_cache=kv_cache, rope_offset=rope_offset,
        self_cond=self_cond,
    )

logger = logging.getLogger("yeoubi")

_MOE_LAYER_CHOICES = ("all", "alternate", "alternate_even", "top_half")


def _validate_byte_config(config: ModelConfig) -> None:
    """Reject subword checkpoints instead of decoding their IDs as bytes."""
    if config.vocab_size != VOCAB_SIZE or config.mask_token_id != MASK_TOKEN_ID:
        raise ValueError(
            "Checkpoint is not byte-level: expected "
            f"vocab_size={VOCAB_SIZE} and mask_token_id={MASK_TOKEN_ID}, got "
            f"vocab_size={config.vocab_size} and mask_token_id={config.mask_token_id}."
        )


def _utf8_preview(text: str, limit: int = 200) -> str:
    """Truncate to a byte budget without splitting a UTF-8 sequence."""
    return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def _filter_top_p_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply nucleus filtering while keeping the first threshold-crossing token."""
    if not 0.0 < top_p <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if top_p == 1.0:
        return logits

    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    sorted_probs = sorted_logits.softmax(dim=-1)
    remove_mask = sorted_probs.cumsum(dim=-1) - sorted_probs >= top_p
    sorted_logits.masked_fill_(remove_mask, -float("inf"))
    return torch.full_like(logits, -float("inf")).scatter_(
        -1, sorted_idx, sorted_logits,
    )


def _build_model_config(args: argparse.Namespace) -> ModelConfig:
    n_think = args.n_think if args.thinking else 0
    return ModelConfig(
        vocab_size=VOCAB_SIZE,
        # Only reached for checkpoints with no embedded config, all of which
        # predate the absmean INT4 scale.
        int4_absmean=False,
        timestep_scale=1.0,  # likewise: they predate the scaled embedding
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        max_seq_len=args.max_seq_len,
        mask_token_id=MASK_TOKEN_ID,
        topk_ratio=args.topk_ratio,
        t_embed_dim=args.t_embed_dim,
        kv_cache_bits=args.kv_cache_bits,
        kv_cache_bos_bits=args.kv_cache_bos_bits,
        use_self_cond=args.use_self_cond,
        N_think=n_think,
        use_moe=args.use_moe,
        n_experts=args.n_experts,
        top_k_experts=args.top_k_experts,
        moe_layers=args.moe_layers,
    )


def load_model_from_checkpoint(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[BitDiffusionTransformer, dict, ModelConfig]:
    """Load a checkpoint and resolve the runtime ``ModelConfig``.

    Handles both standard ``BitDiffusionTransformer`` and ``BitRDTTransformer``
    checkpoints: when the checkpoint carries ``rdt_config`` (or when
    ``config.use_rdt`` is True), the RDT variant is instantiated automatically.
    """
    ckpt = read_checkpoint(
        args.checkpoint,
        device="cpu",
        trust_checkpoint=getattr(args, "trust_checkpoint", False),
    )
    config, from_checkpoint = resolve_checkpoint_model_config(
        ckpt,
        fallback_factory=lambda: _build_model_config(args),
        moe_layers_override=args.moe_layers_override,
    )
    _validate_byte_config(config)

    if from_checkpoint:
        logger.info("Using model_config embedded in checkpoint metadata")
    else:
        logger.warning("Checkpoint has no embedded model_config; using CLI fallback arguments")

    is_rdt = bool(config.use_rdt or ckpt.get("rdt_config"))
    if not is_rdt:
        # Topology safeguard for legacy checkpoints whose model_config does
        # not carry use_rdt/rdt_config: if the state-dict has RDT-specific
        # parameter names, route to BitRDTTransformer regardless.
        state_keys = ckpt.get("model_state_dict", {}).keys()
        if any(k.startswith(("prelude.", "recurrent.", "coda.")) for k in state_keys):
            logger.warning(
                "Checkpoint state_dict has RDT topology (prelude/recurrent/coda) "
                "but no use_rdt flag; loading as BitRDTTransformer."
            )
            is_rdt = True

    if is_rdt:
        from .rdt import BitRDTTransformer, resolve_rdt_config
        rdt_cfg = resolve_rdt_config(ckpt, fallback=None) if ckpt.get("rdt_config") else None
        if rdt_cfg is None:
            from .rdt import RDTConfig
            base = asdict(config)
            base["use_rdt"] = True
            rdt_cfg = RDTConfig(**base)
        _validate_byte_config(rdt_cfg)
        model = BitRDTTransformer(rdt_cfg).to(device)
        logger.info(
            "Loaded BitRDTTransformer (prelude=%d, bridges=%d+%d, recurrent=%d, "
            "coda=%d, loops=%d-%d, byte_patch=%d)",
            rdt_cfg.prelude_layers, rdt_cfg.pre_recurrent_layers,
            rdt_cfg.post_recurrent_layers, rdt_cfg.recurrent_layers,
            rdt_cfg.coda_layers, rdt_cfg.min_loop_iters,
            rdt_cfg.max_loop_iters,
            rdt_cfg.byte_patch_size if rdt_cfg.use_byte_patches else 1,
        )
    else:
        model = BitDiffusionTransformer(config).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt, config


def nucleus_sample(
    logits: torch.Tensor,
    temperature: float = 0.9,
    top_p: float = 0.95,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample from logits using temperature and optional top-p nucleus filtering.

    Args:
        logits: (B, V) unnormalized logits.
        temperature: Sampling temperature. Values < 1 sharpen, > 1 flatten.
        top_p: Cumulative probability threshold for nucleus sampling.
               Set to 1.0 to disable.
        generator: Optional torch Generator for reproducibility.

    Returns:
        (B,) sampled token IDs.
    """
    logits = logits / max(temperature, 1e-8)

    logits = _filter_top_p_logits(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(-1)


@torch.no_grad()
def _remask_step(
    ids: torch.Tensor,
    is_masked: torch.Tensor,
    schedule: CosineSchedule,
    t_curr: torch.Tensor,
    t_next: torch.Tensor,
    mask_token_id: int,
    gen: torch.Generator,
    frozen: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return *ids* with a schedule-correct subset of positions back at MASK.

    The reverse absorbing-state step is *conditional*. Of the positions masked
    at ``t_curr`` (a fraction ``p(t_curr)`` of the canvas), the schedule wants
    ``p(t_next)`` of the whole canvas still masked afterwards, so each one is
    re-masked with probability ``p(t_next) / p(t_curr)``. Drawing against
    ``p(t_next)`` directly — as all four samplers used to — makes the masked
    fraction the running *product* of the per-step probabilities: with 20 steps
    the canvas is fully committed by roughly step 8, every later step is a
    no-op, and the bytes that were committed went in at high noise.

    Committed and frozen positions are never re-corrupted: absorbing-state
    training never shows the model a re-masked clean byte.

    p(t_curr) is 1.0 on the first step and strictly positive on every step that
    can reach here (``t_next`` hits 0 only after the last re-mask), so the
    clamp is belt-and-braces, not a live branch.
    """
    p_curr = schedule.mask_prob(t_curr).clamp(min=1e-6)
    ratio = schedule.mask_prob(t_next) / p_curr
    rand = torch.rand(ids.shape, device=ids.device, generator=gen)
    should_remask = (rand < ratio) & is_masked
    if frozen is not None:
        should_remask = should_remask & ~frozen
    return torch.where(should_remask, mask_token_id, ids)


@torch.no_grad()
def _revise_step(
    ids: torch.Tensor,
    logits: torch.Tensor,
    is_masked: torch.Tensor,
    vocab_size: int,
    tau: float,
    frozen: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Overwrite visible bytes the model confidently disagrees with.

    Samplers otherwise only write MASK positions and ``_remask_step`` never
    re-masks a committed byte, so an early mistake is permanent. The
    ``correction`` training stage teaches exactly this repair: input byte
    visible but wrong, target the clean byte. Overwriting in place reproduces
    that input/target pair; re-masking would instead ask the absorbing-state
    task, which never shows a re-masked clean byte.

    Only ``~is_masked`` positions qualify: the logits were computed from the
    canvas at the start of this step, so a decision is only meaningful for a
    byte that was visible in that input — bytes sampled this step are not
    revised in the same step. Frozen positions (prompt) are never touched.

    Opt-in only: a checkpoint that never ran the correction stage has
    unsupervised logits at visible positions (the loss covers masked ones
    only), and correction training used ``t <= correction_t_max``, so callers
    gate this on a low noise level. Raw logits, no temperature or top-p: this
    is a confidence test, not a draw.
    """
    p, best = logits[..., :vocab_size].float().softmax(-1).max(-1)
    revise = ~is_masked & (best != ids) & (p > tau)
    if frozen is not None:
        revise = revise & ~frozen
    return torch.where(revise, best, ids)


def denoise(
    model: BitDiffusionTransformer,
    prompt: str = "",
    gen_length: int = 128,
    steps: int = 20,
    temperature: float = 0.9,
    top_p: float = 0.95,
    num_samples: int = 1,
    verbose: bool = False,
    seed: int = 42,
    device: torch.device = torch.device("cpu"),
    n_loops: Optional[int] = None,
    revise_threshold: float = 0.0,
    revise_t_max: float = 0.5,
) -> list[str]:
    """Generate text via iterative diffusion denoising.

    The prompt (if provided) is UTF-8 encoded and treated as a frozen prefix
    that is never masked. The rest of the sequence starts fully masked
    at t=1 and is progressively denoised to t=0.

    **No KV cache:** The full denoiser re-processes the entire sequence from
    scratch at every step because the mask pattern changes at each step.
    A KV cache from step *i* is invalid at step *i+1* and would need to be
    reset anyway, so no cache is allocated here.  KV cache reuse only
    benefits the :class:`BlockDiffusionSampler`, where committed blocks
    never change between steps.

    Args:
        model: Trained BitDiffusionTransformer.
        prompt: Optional text prompt as conditioning prefix.
        gen_length: Total sequence length (prompt + generated tokens).
        steps: Number of denoising steps.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        num_samples: Number of independent samples to generate.
        verbose: Print intermediate sequences after each step.
        seed: Random seed.
        device: Torch device.
        revise_threshold: Confidence above which a visible byte is overwritten
            by the model's top byte (0.0 = off). See :func:`_revise_step`.
        revise_t_max: Only revise on steps with ``t_curr <= revise_t_max``.

    Returns:
        List of generated text strings.
    """
    if not 0.0 <= revise_threshold < 1.0:
        raise ValueError(f"revise_threshold={revise_threshold} must be in [0.0, 1.0)")
    model.eval()
    schedule = CosineSchedule()
    mask_token_id = model.config.mask_token_id
    vocab_size = model.config.vocab_size

    # Prompt IDs are literal UTF-8 bytes; no BOS or other token is inserted.
    if prompt:
        prompt_ids = list(prompt.encode("utf-8"))
    else:
        prompt_ids = []

    prefix_len = len(prompt_ids)
    total_len = max(gen_length, prefix_len + 1)

    # Initialize the generated region with the absorbing mask sentinel.
    ids = torch.full((num_samples, total_len), mask_token_id, dtype=torch.long, device=device)
    if prefix_len > 0:
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        ids[:, :prefix_len] = prompt_tensor

    # Frozen mask: True where tokens should never be corrupted (the prompt)
    frozen = torch.zeros(num_samples, total_len, dtype=torch.bool, device=device)
    if prefix_len > 0:
        frozen[:, :prefix_len] = True

    gen = torch.Generator(device=device).manual_seed(seed)

    # Denoising schedule: uniform in mask probability, not in t. Linear-in-t front-loads
    # the work — the first 3 of 20 steps would commit ~23% of the canvas while the model
    # still has almost no context — so invert the cosine to equalize the commit fraction.
    p_steps = torch.linspace(1.0, 0.0, steps + 1, device=device)
    t_steps = CosineSchedule.inverse(p_steps).clamp(0.0, 1.0)

    self_cond = None
    for step_idx in range(steps):
        t_curr = t_steps[step_idx]
        t_next = t_steps[step_idx + 1]

        # Forward pass — no KV cache; each step reprocesses the full sequence.
        t_input = t_curr.expand(num_samples)
        logits, _ = _model_fwd(
            model, ids, t_input, n_loops=n_loops, self_cond=self_cond,
        )  # (B, T, V)
        if getattr(model.config, "use_self_cond", False):
            self_cond = self_cond_vector(
                logits, model.embed.weight, model.config.vocab_size,
            )

        # Identify currently masked positions (excluding frozen prefix)
        is_masked = (ids == mask_token_id)

        # Sample at all masked positions (batched)
        if is_masked.any():
            masked_logits = logits[:, :, :vocab_size]  # (B, T, V) exclude mask token
            masked_logits = masked_logits / max(temperature, 1e-8)
            masked_logits = _filter_top_p_logits(masked_logits, top_p)
            probs = F.softmax(masked_logits, dim=-1)
            flat_probs = probs.view(-1, probs.shape[-1])
            sampled = torch.multinomial(flat_probs, 1, generator=gen).view(num_samples, total_len)
            ids = torch.where(is_masked, sampled, ids)

        if revise_threshold > 0 and t_curr <= revise_t_max:
            ids = _revise_step(ids, logits, is_masked, vocab_size, revise_threshold, frozen=frozen)

        # Re-mask positions that should still be uncertain at t_next
        if step_idx + 1 < steps:
            ids = _remask_step(
                ids, is_masked, schedule, t_curr, t_next, mask_token_id, gen,
                frozen=frozen,
            )

        if verbose:
            # Print intermediate result
            for b in range(num_samples):
                temp_ids = ids[b].tolist()
                # Byte-wise diffusion can expose invalid UTF-8 mid-denoising;
                # replacement characters make that corruption explicit.
                text = decode_byte_ids(temp_ids, errors="replace")
                n_masked = (ids[b] == mask_token_id).sum().item()
                logger.info(
                    "Step %d/%d (t=%.3f→%.3f, masked=%d): %s",
                    step_idx + 1, steps, t_curr.item(), t_next.item(), n_masked, _utf8_preview(text),
                )

    # Final decode
    results = []
    for b in range(num_samples):
        text = decode_byte_ids(ids[b].tolist(), errors="replace")
        results.append(text)

    return results


# ---------------------------------------------------------------------------
# Thinking Diffusion Sampler — dual-phase denoising
# ---------------------------------------------------------------------------

class ThinkingDiffusionSampler:
    """Two-phase diffusion sampler with latent scratchpad thinking tokens.

    Phase 1 (thinking): Denoise only the [THINK] positions while answer
    positions remain fully masked and frozen. After thinking converges (or
    ``max_think_steps`` is reached) the thinking tokens are committed.

    Adaptive thinking: when ``adaptive_think=True`` the thinking phase runs
    until the fraction of thinking tokens that change between consecutive
    steps drops below ``think_change_threshold`` for ``think_patience``
    steps in a row, indicating the scratchpad has stabilised. This removes
    the need to set a fixed ``think_steps`` budget.

    Phase 2 (answer): Denoise only the answer positions, attending to the
    committed thinking tokens as additional context.

    Args:
        model: Trained BitDiffusionTransformer.
        n_think: Number of thinking token positions.
        think_steps: Fixed denoising steps for thinking (used when
            ``adaptive_think=False``).
        answer_steps: Denoising steps for the answer phase.
        adaptive_think: If True, run thinking until convergence instead of
            a fixed number of steps.
        max_think_steps: Hard cap on thinking steps when adaptive (default
            64 — prevents infinite loops on pathological inputs).
        think_change_threshold: Fraction of thinking tokens that must
            change between steps to be considered "not converged". When the
            change rate drops below this value for ``think_patience``
            consecutive steps, thinking stops early (default 0.02 = 2%).
        think_patience: Number of consecutive below-threshold steps before
            early stopping triggers (default 3).
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        verbose: Print intermediate results.
        seed: Random seed.
        device: Torch device.
    """

    def __init__(
        self,
        model: BitDiffusionTransformer,
        n_think: int = 64,
        think_steps: int = 10,
        answer_steps: int = 20,
        adaptive_think: bool = False,
        max_think_steps: int = 128,
        think_change_threshold: float = 0.02,
        think_patience: int = 3,
        temperature: float = 0.9,
        top_p: float = 0.95,
        verbose: bool = False,
        seed: int = 42,
        device: torch.device = torch.device("cpu"),
        n_loops: Optional[int] = None,
    ):
        self.model = model
        self.n_think = n_think
        self.think_steps = think_steps
        self.answer_steps = answer_steps
        self.adaptive_think = adaptive_think
        self.max_think_steps = max_think_steps
        self.think_change_threshold = think_change_threshold
        self.think_patience = think_patience
        self.temperature = temperature
        self.top_p = top_p
        self.verbose = verbose
        self.seed = seed
        self.device = device
        self.n_loops = n_loops

    @torch.no_grad()
    def generate(
        self,
        prompt: str = "",
        gen_length: int = 128,
        num_samples: int = 1,
    ) -> list[dict[str, str]]:
        """Generate text with dual-phase thinking + answer denoising.

        Args:
            prompt: Optional conditioning prompt prefix.
            gen_length: Length of the answer portion (excluding thinking).
            num_samples: Number of independent samples to generate.

        Returns:
            List of dicts with keys ``"thinking"`` and ``"answer"``.
        """
        self.model.eval()
        config = self.model.config
        schedule = CosineSchedule()
        mask_token_id = config.mask_token_id
        think_token_id = config.think_token_id
        vocab_size = config.vocab_size

        # Prompt IDs are literal UTF-8 bytes.
        if prompt:
            prompt_ids = list(prompt.encode("utf-8"))
        else:
            prompt_ids = []

        prefix_len = len(prompt_ids)
        answer_len = max(gen_length, prefix_len + 1)
        total_len = self.n_think + answer_len

        # Initialize: [prompt] + [THINK]*n_think + [MASK]*answer_len
        # Layout: positions 0..prefix_len-1 = prompt (frozen)
        #         positions prefix_len..prefix_len+n_think-1 = think (initially masked)
        #         positions prefix_len+n_think.. = answer (initially masked)
        ids = torch.full(
            (num_samples, total_len), mask_token_id,
            dtype=torch.long, device=self.device,
        )
        if prefix_len > 0:
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
            ids[:, :prefix_len] = prompt_tensor

        # Position ranges
        think_start = prefix_len
        think_end = prefix_len + self.n_think
        answer_start = think_end

        # Frozen mask: prompt positions are never masked
        frozen = torch.zeros(num_samples, total_len, dtype=torch.bool, device=self.device)
        if prefix_len > 0:
            frozen[:, :prefix_len] = True

        gen = torch.Generator(device=self.device).manual_seed(self.seed)

        # No KV cache here: this sampler re-processes the full sequence with
        # a new mask pattern at every step (like `denoise()`), so a cache
        # would never serve a cached read. Passing one anyway used to cost
        # real quantization noise on every K/V round-trip for zero benefit,
        # and it also disabled padding-mask installation in the model
        # forward (see `install_runtime_attention_mask(..., None)` when
        # `kv_cache is not None` in model.py) — a second, unrelated
        # correctness cost. Compare `BlockDiffusionSampler`, where committed
        # blocks genuinely persist across steps and the cache does real work.

        # --- Phase 1: Thinking ---
        total_think_steps = self.max_think_steps if self.adaptive_think else self.think_steps
        if self.verbose:
            mode = f"adaptive (max={total_think_steps})" if self.adaptive_think else f"fixed ({total_think_steps} steps)"
            logger.info("=== Phase 1: Thinking [%s] ===", mode)

        t_steps_think = torch.linspace(1.0, 0.0, total_think_steps + 1, device=self.device)

        # Adaptive convergence tracking
        _patience_counter = 0
        _prev_think_ids: Optional[torch.Tensor] = None
        self_cond = None

        for step_idx in range(total_think_steps):
            t_curr = t_steps_think[step_idx]
            t_next = t_steps_think[step_idx + 1]

            t_input = t_curr.expand(num_samples)
            logits, _ = _model_fwd(self.model, ids, t_input,
                                   n_loops=self.n_loops,
                                   self_cond=self_cond)
            if getattr(self.model.config, "use_self_cond", False):
                self_cond = self_cond_vector(
                    logits, self.model.embed.weight, self.model.config.vocab_size,
                )

            # Snapshot before sampling for change-rate measurement
            prev_slice = ids[:, think_start:think_end].clone()

            # Sample at thinking positions that are masked (batched)
            think_is_masked = (ids[:, think_start:think_end] == mask_token_id)  # (B, n_think)
            if think_is_masked.any():
                think_logits = logits[:, think_start:think_end, :vocab_size]
                sampled = nucleus_sample(
                    think_logits.reshape(-1, vocab_size), self.temperature, self.top_p, generator=gen,
                ).view(num_samples, self.n_think)
                ids[:, think_start:think_end] = torch.where(
                    think_is_masked, sampled, ids[:, think_start:think_end],
                )

            # Re-mask thinking positions for next step
            if step_idx + 1 < total_think_steps:
                ids[:, think_start:think_end] = _remask_step(
                    ids[:, think_start:think_end], think_is_masked, schedule,
                    t_curr, t_next, mask_token_id, gen,
                )

            if self.verbose:
                for b in range(num_samples):
                    think_ids = ids[b, think_start:think_end].tolist()
                    text = decode_byte_ids(think_ids, errors="replace")
                    n_masked = (ids[b, think_start:think_end] == mask_token_id).sum().item()
                    logger.info(
                        "Think step %d/%d (t=%.3f→%.3f, masked=%d): %s",
                        step_idx + 1, total_think_steps, t_curr.item(), t_next.item(),
                        n_masked, _utf8_preview(text),
                    )

            # Adaptive early stopping: measure token change rate after re-masking
            if self.adaptive_think and _prev_think_ids is not None:
                # Compare unmasked tokens only — masked positions always "differ"
                curr_slice = ids[:, think_start:think_end]
                both_unmasked = (prev_slice != mask_token_id) & (curr_slice != mask_token_id)
                n_unmasked = both_unmasked.sum().item()
                if n_unmasked > 0:
                    n_changed = ((prev_slice != curr_slice) & both_unmasked).sum().item()
                    change_rate = n_changed / n_unmasked
                    if self.verbose:
                        logger.info("  [adaptive] change_rate=%.3f (threshold=%.3f, patience=%d/%d)",
                                    change_rate, self.think_change_threshold,
                                    _patience_counter + 1, self.think_patience)
                    if change_rate < self.think_change_threshold:
                        _patience_counter += 1
                        if _patience_counter >= self.think_patience:
                            if self.verbose:
                                logger.info("  [adaptive] thinking converged at step %d/%d",
                                            step_idx + 1, total_think_steps)
                            break
                    else:
                        _patience_counter = 0

            _prev_think_ids = ids[:, think_start:think_end].clone()

        # Commit thinking tokens — freeze them
        frozen[:, think_start:think_end] = True

        # --- Phase 2: Answer ---
        if self.verbose:
            logger.info("=== Phase 2: Answer (%d steps) ===", self.answer_steps)

        t_steps_answer = torch.linspace(1.0, 0.0, self.answer_steps + 1, device=self.device)
        self_cond = None

        for step_idx in range(self.answer_steps):
            t_curr = t_steps_answer[step_idx]
            t_next = t_steps_answer[step_idx + 1]

            t_input = t_curr.expand(num_samples)
            logits, _ = _model_fwd(self.model, ids, t_input,
                                   n_loops=self.n_loops,
                                   self_cond=self_cond)
            if getattr(self.model.config, "use_self_cond", False):
                self_cond = self_cond_vector(
                    logits, self.model.embed.weight, self.model.config.vocab_size,
                )

            # Sample at answer positions that are masked (batched)
            answer_is_masked = (ids[:, answer_start:] == mask_token_id)  # (B, answer_len)
            if answer_is_masked.any():
                answer_logits = logits[:, answer_start:, :vocab_size]
                n_answer = total_len - answer_start
                sampled = nucleus_sample(
                    answer_logits.reshape(-1, vocab_size), self.temperature, self.top_p, generator=gen,
                ).view(num_samples, n_answer)
                ids[:, answer_start:] = torch.where(
                    answer_is_masked, sampled, ids[:, answer_start:],
                )

            # Re-mask answer positions for next step
            if step_idx + 1 < self.answer_steps:
                ids[:, answer_start:] = _remask_step(
                    ids[:, answer_start:], answer_is_masked, schedule,
                    t_curr, t_next, mask_token_id, gen,
                )

            if self.verbose:
                for b in range(num_samples):
                    answer_ids = ids[b, answer_start:].tolist()
                    text = decode_byte_ids(answer_ids, errors="replace")
                    n_masked = (ids[b, answer_start:] == mask_token_id).sum().item()
                    logger.info(
                        "Answer step %d/%d (t=%.3f→%.3f, masked=%d): %s",
                        step_idx + 1, self.answer_steps, t_curr.item(), t_next.item(),
                        n_masked, _utf8_preview(text),
                    )

        # Final decode
        results = []
        for b in range(num_samples):
            think_ids = ids[b, think_start:think_end].tolist()
            answer_ids = ids[b, answer_start:].tolist()
            think_text = decode_byte_ids(think_ids, errors="replace")
            answer_text = decode_byte_ids(answer_ids, errors="replace")
            results.append({"thinking": think_text, "answer": answer_text})

        return results


# ---------------------------------------------------------------------------
# Block Diffusion Sampler — semi-autoregressive with quantized KV cache
# ---------------------------------------------------------------------------

class BlockDiffusionSampler:
    """Semi-autoregressive block diffusion sampler.

    Generates text in fixed-size blocks, left to right. Each block is
    denoised via the full MDLM diffusion schedule (bidirectional attention
    within the block). Previously committed blocks are read through the
    quantized KV cache, giving correct left-to-right coherence across
    blocks without recomputing their K/V.

    This makes the quantized KV cache effective: committed context
    accumulates in 3-bit storage and is never re-masked or recomputed.

    Optionally prepends per-block thinking tokens as a planning scratchpad
    before each block is generated.

    Args:
        model: Trained BitDiffusionTransformer.
        block_size: Number of byte positions per block.
        steps: Denoising steps per block.
        think_tokens: Number of thinking token positions per block (0=disabled).
        think_steps: Denoising steps for the thinking phase.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        verbose: Print intermediate results.
        seed: Random seed.
        device: Torch device.
    """

    def __init__(
        self,
        model: BitDiffusionTransformer,
        block_size: int = 256,
        steps: int = 20,
        think_tokens: int = 0,
        think_steps: int = 10,
        temperature: float = 0.9,
        top_p: float = 0.95,
        verbose: bool = False,
        seed: int = 42,
        device: torch.device = torch.device("cpu"),
        auto_length: bool = False,
        max_length: int = 2048,
        n_loops: Optional[int] = None,
        revise_threshold: float = 0.0,
        revise_t_max: float = 0.5,
    ):
        if not 0.0 <= revise_threshold < 1.0:
            raise ValueError(f"revise_threshold={revise_threshold} must be in [0.0, 1.0)")
        self.model = model
        self.block_size = block_size
        self.steps = steps
        self.think_tokens = think_tokens
        self.think_steps = think_steps
        self.temperature = temperature
        self.top_p = top_p
        self.verbose = verbose
        self.seed = seed
        self.device = device
        self.auto_length = auto_length
        self.max_length = max_length
        self.n_loops = n_loops
        self.revise_threshold = revise_threshold
        self.revise_t_max = revise_t_max
        self.eos_token_id: Optional[int] = EOS_TOKEN_ID

        # Every committed chunk must start on a patch boundary, or the codec's
        # chunk-relative sub-positions drift from the patch grid seen in training.
        cfg = model.config
        self.patch_size = cfg.byte_patch_size if getattr(cfg, "use_byte_patches", False) else 1
        if block_size % self.patch_size != 0:
            raise ValueError(
                f"block_size={block_size} must be a multiple of byte_patch_size={self.patch_size}"
            )
        if think_tokens > 0 and think_tokens % self.patch_size != 0:
            raise ValueError(
                f"think_tokens={think_tokens} must be a multiple of byte_patch_size={self.patch_size}"
            )

    def _sample_masked(
        self,
        logits: torch.Tensor,
        is_masked: torch.Tensor,
        ids: torch.Tensor,
        gen: torch.Generator,
    ) -> torch.Tensor:
        """Sample at masked positions using nucleus sampling (batched)."""
        vocab_size = self.model.config.vocab_size
        if not is_masked.any():
            return ids
        masked_logits = logits[:, :, :vocab_size] / max(self.temperature, 1e-8)
        masked_logits = _filter_top_p_logits(masked_logits, self.top_p)
        probs = F.softmax(masked_logits, dim=-1)
        B, T, V = probs.shape
        sampled = torch.multinomial(probs.view(-1, V), 1, generator=gen).view(B, T)
        return torch.where(is_masked, sampled, ids)

    def _denoise_block(
        self,
        block_ids: torch.Tensor,
        kv_cache: "KVCache",
        rope_offset: int,
        n_steps: int,
        gen: torch.Generator,
        frozen: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the diffusion denoising schedule on a single block.

        The KV cache is set to ephemeral mode: committed context K/V are
        read from cache, current block K/V are computed fresh each step
        but not stored.

        Args:
            block_ids: (B, block_len) token IDs (initially masked).
            kv_cache: KVCache with committed context.
            rope_offset: Absolute position offset for this block.
            n_steps: Number of denoising steps.
            gen: Random generator.
            frozen: Optional (B, block_len) bool mask of positions that must
                never be revised (carried prompt bytes).

        Returns:
            (B, block_len) denoised token IDs.
        """
        schedule = CosineSchedule()
        mask_token_id = self.model.config.mask_token_id
        B, block_len = block_ids.shape

        # Space steps uniformly in mask probability so each step commits an equal
        # fraction of the block; linear-in-t would commit ~23% of it in the first
        # 3 of 20 steps, before the block has any of its own context.
        p_steps = torch.linspace(1.0, 0.0, n_steps + 1, device=self.device)
        t_steps = CosineSchedule.inverse(p_steps).clamp(0.0, 1.0)

        kv_cache.ephemeral = True
        self_cond = None
        for step_idx in range(n_steps):
            t_curr = t_steps[step_idx]
            t_next = t_steps[step_idx + 1]

            t_input = t_curr.expand(B)
            logits, _ = _model_fwd(self.model, block_ids, t_input,
                                   kv_cache=kv_cache, rope_offset=rope_offset,
                                   n_loops=self.n_loops,
                                   self_cond=self_cond)
            if getattr(self.model.config, "use_self_cond", False):
                self_cond = self_cond_vector(
                    logits, self.model.embed.weight, self.model.config.vocab_size,
                )

            is_masked = (block_ids == mask_token_id)
            block_ids = self._sample_masked(logits, is_masked, block_ids, gen)
            if self.revise_threshold > 0 and t_curr <= self.revise_t_max:
                block_ids = _revise_step(
                    block_ids, logits, is_masked, self.model.config.vocab_size,
                    self.revise_threshold, frozen=frozen,
                )
            if step_idx + 1 < n_steps:
                block_ids = _remask_step(
                    block_ids, is_masked, schedule, t_curr, t_next,
                    mask_token_id, gen,
                )

            if self.verbose:
                n_masked = (block_ids == mask_token_id).sum().item()
                preview = decode_byte_ids(block_ids[0].tolist(), errors="replace")
                logger.info(
                    "  step %d/%d (t=%.3f→%.3f, masked=%d): %s",
                    step_idx + 1, n_steps, t_curr.item(), t_next.item(), n_masked, preview[:120],
                )

        kv_cache.ephemeral = False
        return block_ids

    def _commit_block(
        self,
        block_ids: torch.Tensor,
        kv_cache: "KVCache",
        rope_offset: int,
    ) -> None:
        """Forward a committed block at t=0 and store its K/V in the cache.

        Args:
            block_ids: (B, block_len) fully denoised token IDs.
            kv_cache: KVCache to store into (normal mode).
            rope_offset: Absolute position offset for this block.
        """
        B = block_ids.shape[0]
        t_zero = torch.zeros(B, device=self.device)
        kv_cache.ephemeral = False
        with torch.no_grad():
            _model_fwd(self.model, block_ids, t_zero,
                       kv_cache=kv_cache, rope_offset=rope_offset,
                       n_loops=self.n_loops)

    @torch.no_grad()
    def generate(
        self,
        prompt: str = "",
        gen_length: int = 512,
        num_samples: int = 1,
    ) -> list[dict[str, str]]:
        """Generate text using block-wise diffusion with quantized KV cache.

        When ``auto_length=True`` (set on the sampler), generation stops as
        soon as an EOS token appears in a completed block. The model naturally
        decides how long the response should be — short prompts get short
        answers, complex ones get longer answers. A hard cap of
        ``max_length`` byte positions prevents runaway generation.

        Args:
            prompt: Optional conditioning prompt prefix.
            gen_length: Total byte positions to generate (ignored when
                ``auto_length=True``, which uses ``max_length`` instead).
            num_samples: Number of independent samples.

        Returns:
            List of dicts with ``"text"`` and ``"blocks"`` keys.
        """
        self.model.eval()
        config = self.model.config
        mask_token_id = config.mask_token_id
        vocab_size = config.vocab_size

        # In auto_length mode ignore gen_length, use max_length as the cap
        effective_max = self.max_length if self.auto_length else gen_length

        gen = torch.Generator(device=self.device).manual_seed(self.seed)

        # Prompt IDs are literal UTF-8 bytes.
        if prompt:
            prompt_ids = list(prompt.encode("utf-8"))
        else:
            prompt_ids = []
        prefix_len = len(prompt_ids)

        # Create one cache slot per virtual recurrent depth.  Shared weights
        # still produce different K/V states on each visit.
        cache_layers = (
            self.model.cache_layers_for_loops(self.n_loops)
            if hasattr(self.model, "cache_layers_for_loops")
            else config.n_layers
        )
        kv_cache = KVCache(
            n_layers=cache_layers,
            default_bits=config.kv_cache_bits,
            bos_bits=config.kv_cache_bos_bits,
        )

        # Commit only the patch-aligned part of the prompt. A partial patch
        # committed here would sit mid-sequence (training only ever pads the
        # final patch) and shift the patch grid of every later block, so the
        # 0..patch-1 leftover bytes ride as fixed literals at the head of the
        # first content block instead. The legacy thinking path keeps the
        # old full-prompt commit.
        commit_len = prefix_len
        if self.think_tokens == 0:
            commit_len = prefix_len // self.patch_size * self.patch_size
        carry = prompt_ids[commit_len:]
        if commit_len > 0:
            prompt_tensor = torch.tensor(prompt_ids[:commit_len], dtype=torch.long, device=self.device)
            prompt_batch = prompt_tensor.unsqueeze(0).expand(num_samples, -1)
            # In block_size chunks, not one piece: block-causal training only
            # shows blocks of bounded size whose local convs stop at the block
            # edge, so a long prompt committed whole is a state the model was
            # never trained on. block_size is a patch multiple, so every chunk
            # but the last stays on the patch grid.
            for start in range(0, commit_len, self.block_size):
                self._commit_block(
                    prompt_batch[:, start:start + self.block_size], kv_cache,
                    rope_offset=start,
                )

        # Divide generation into blocks (capped at effective_max); the carry
        # occupies canvas space in the first block, hence the extra bytes.
        n_blocks = max(1, (effective_max + len(carry) + self.block_size - 1) // self.block_size)
        all_generated: list[list[int]] = [[] for _ in range(num_samples)]
        block_texts: list[list[str]] = [[] for _ in range(num_samples)]
        current_offset = commit_len
        eos_hit = [False] * num_samples

        block_times: list[float] = []

        for block_idx in range(n_blocks):
            n_carry = len(carry) if block_idx == 0 else 0
            remaining = effective_max - len(all_generated[0])
            n_new = min(self.block_size - n_carry, remaining)
            if n_new <= 0:
                break
            this_block_size = n_new + n_carry

            block_t0 = time.perf_counter()

            if self.verbose:
                mode = "auto" if self.auto_length else f"{n_blocks} blocks"
                logger.info(
                    "=== Block %d/%s (offset=%d, size=%d) ===",
                    block_idx + 1, mode, current_offset, this_block_size,
                )

            # --- Per-block thinking phase (optional) ---
            if self.think_tokens > 0:
                if self.verbose:
                    logger.info("  [thinking phase: %d tokens, %d steps]", self.think_tokens, self.think_steps)
                think_ids = torch.full(
                    (num_samples, self.think_tokens), mask_token_id,
                    dtype=torch.long, device=self.device,
                )
                think_ids = self._denoise_block(
                    think_ids, kv_cache, rope_offset=current_offset,
                    n_steps=self.think_steps, gen=gen,
                )
                self._commit_block(think_ids, kv_cache, rope_offset=current_offset)
                current_offset += self.think_tokens

            # --- Denoise the content block ---
            if self.verbose:
                logger.info("  [content phase: %d tokens, %d steps]", this_block_size, self.steps)
            block_ids = torch.full(
                (num_samples, this_block_size), mask_token_id,
                dtype=torch.long, device=self.device,
            )
            if n_carry:
                # Non-MASK positions are never resampled, so these stay fixed.
                block_ids[:, :n_carry] = torch.tensor(carry, dtype=torch.long, device=self.device)
            carry_frozen = None
            if n_carry:
                # Carried bytes are prompt, not model output: never revise them.
                carry_frozen = torch.zeros_like(block_ids, dtype=torch.bool)
                carry_frozen[:, :n_carry] = True
            block_ids = self._denoise_block(
                block_ids, kv_cache, rope_offset=current_offset,
                n_steps=self.steps, gen=gen, frozen=carry_frozen,
            )

            # Commit content block to cache
            self._commit_block(block_ids, kv_cache, rope_offset=current_offset)
            current_offset += this_block_size

            # Collect block tokens per sample independently
            for b in range(num_samples):
                if eos_hit[b]:
                    continue
                block_token_list = block_ids[b, n_carry:].tolist()

                # Auto-length: check for EOS in this block and truncate there
                if self.auto_length and self.eos_token_id is not None:
                    for pos, tok in enumerate(block_token_list):
                        if tok == self.eos_token_id:
                            block_token_list = block_token_list[:pos]
                            eos_hit[b] = True
                            if self.verbose:
                                logger.info(
                                    "  [auto_length] EOS at sample %d block %d pos %d — stopping",
                                    b, block_idx + 1, pos,
                                )
                            break

                block_text = decode_byte_ids(block_token_list, errors="replace")
                block_texts[b].append(block_text)
                all_generated[b].extend(block_token_list)

            block_elapsed = time.perf_counter() - block_t0
            block_times.append(block_elapsed)

            if self.verbose:
                # Log sample 0 as representative
                if block_texts[0]:
                    tok_per_s = n_new / max(block_elapsed, 1e-6)
                    logger.info(
                        "  committed[0]: %s  [%.2f s, %.0f tok/s]",
                        block_texts[0][-1][:160], block_elapsed, tok_per_s,
                    )

            if all(eos_hit):
                break

        # Final decode per sample
        total_block_time = sum(block_times)
        results = []
        for b in range(num_samples):
            text = decode_byte_ids(all_generated[b], errors="replace")
            n_gen = len(all_generated[b])
            results.append({
                "text": text,
                "blocks": block_texts[b],
                "n_tokens": n_gen,
                "elapsed_s": total_block_time,
                "tok_per_s": n_gen / max(total_block_time, 1e-6),
            })

        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for sampling."""
    force_utf8_console()
    parser = argparse.ArgumentParser(description=f"{MODEL_NAME} — Denoising Sampler")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--trust_checkpoint",
        action="store_true",
        help="Allow unsafe full-pickle checkpoint loading for trusted legacy checkpoints only.",
    )
    parser.add_argument("--prompt", type=str, default="", help="Conditioning prompt prefix")
    parser.add_argument("--length", type=int, default=128, help="Total sequence length in bytes")
    parser.add_argument("--steps", type=int, default=20, help="Number of denoising steps")
    parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling threshold")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--verbose", action="store_true", help="Print each denoising step")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: auto | cuda[:<n>] | xpu[:<n>] | cpu | rocm")

    # Model config — must match training config
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--ffn_dim", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--topk_ratio", type=float, default=0.55)
    parser.add_argument("--t_embed_dim", type=int, default=256)
    parser.add_argument("--kv_cache_bits", type=int, default=4)
    parser.add_argument("--kv_cache_bos_bits", type=int, default=4)

    # Thinking token args
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable thinking token generation for legacy checkpoints without embedded model_config",
    )
    parser.add_argument("--n_think", type=int, default=0, help="Number of legacy thinking token positions")
    parser.add_argument(
        "--use_self_cond",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable self-conditioning for legacy checkpoints without embedded model_config",
    )
    parser.add_argument("--think_steps", type=int, default=10, help="Denoising steps for thinking phase (fixed mode)")
    parser.add_argument("--answer_steps", type=int, default=20, help="Denoising steps for answer phase")
    parser.add_argument("--adaptive_think", action="store_true", help="Adaptively stop thinking when tokens stop changing")
    parser.add_argument("--max_think_steps", type=int, default=128, help="Hard cap on thinking steps in adaptive mode")
    parser.add_argument("--think_change_threshold", type=float, default=0.02, help="Token change rate below which thinking is considered converged (adaptive mode)")
    parser.add_argument("--think_patience", type=int, default=3, help="Consecutive below-threshold steps before early stopping (adaptive mode)")

    # Block diffusion args
    parser.add_argument(
        "--block",
        action="store_true",
        help="Use block diffusion sampler (semi-AR with quantized KV cache)",
    )
    parser.add_argument("--block_size", type=int, default=256, help="Byte positions per block in block diffusion")
    parser.add_argument(
        "--block_think_tokens", type=int, default=0,
        help="Per-block thinking tokens in block diffusion (0=disabled, try 32-64)",
    )
    parser.add_argument("--auto_length", action="store_true",
        help="Stop generating when EOS token appears — model decides response length automatically")
    parser.add_argument("--max_length", type=int, default=2048,
        help="Hard cap on total byte positions when --auto_length is enabled")
    # RDT args
    parser.add_argument(
        "--n_loops", type=int, default=None,
        help="Recurrence iterations for BitRDTTransformer (None = use checkpoint default). "
             "Set higher than trained max_loop_iters for inference-time depth extrapolation.",
    )
    # Self-correction args (full denoiser and --block)
    parser.add_argument(
        "--revise_threshold", type=float, default=0.0,
        help="Overwrite a visible byte when the model's top byte differs with prob > this "
             "(0.0 = off). Needs a checkpoint trained with --training_stage correction; try 0.9.",
    )
    parser.add_argument(
        "--revise_t_max", type=float, default=0.5,
        help="Only revise on steps with t <= this (match the training correction_t_max).",
    )

    # MoE args
    parser.add_argument("--use_moe", action="store_true", help="Enable MoE FFN layers")
    parser.add_argument("--n_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--top_k_experts", type=int, default=2, help="Top-K experts per token")
    parser.add_argument(
        "--moe_layers",
        type=str,
        default="alternate",
        choices=_MOE_LAYER_CHOICES,
        help="MoE layer pattern for legacy checkpoints without embedded model_config",
    )
    parser.add_argument(
        "--moe_layers_override",
        type=str,
        default=None,
        choices=_MOE_LAYER_CHOICES,
        help="Override the checkpoint MoE pattern to recover alternate-even compatibility-bug checkpoints.",
    )

    args = parser.parse_args()
    setup_logging()

    from .device import resolve_device
    device = resolve_device(args.device)

    model, ckpt, _ = load_model_from_checkpoint(args, device)
    model.eval()

    logger.info("Model loaded from %s (step %d)", args.checkpoint, ckpt.get("step", 0))

    if args.block:
        logger.info(
            "Block diffusion: %d sample(s), block_size=%d, steps=%d, think_tokens=%d"
            "  [KV cache active — committed blocks reused across denoising steps]",
            args.num_samples, args.block_size, args.steps, args.block_think_tokens,
        )
        sampler = BlockDiffusionSampler(
            model=model,
            block_size=args.block_size,
            steps=args.steps,
            think_tokens=args.block_think_tokens,
            think_steps=args.think_steps,
            temperature=args.temperature,
            auto_length=args.auto_length,
            max_length=args.max_length,
            top_p=args.top_p,
            verbose=args.verbose,
            seed=args.seed,
            device=device,
            n_loops=args.n_loops,
            revise_threshold=args.revise_threshold,
            revise_t_max=args.revise_t_max,
        )
        t0 = time.perf_counter()
        results = sampler.generate(
            prompt=args.prompt,
            gen_length=args.length,
            num_samples=args.num_samples,
        )
        wall_s = time.perf_counter() - t0

        print("\n" + "=" * 60)
        for i, result in enumerate(results):
            if args.num_samples > 1:
                print(f"\n--- Sample {i + 1} ---")
            print(result["text"])
        print("=" * 60)
        # Separate the block-sampler story: report per-sample tok/s so it can
        # be compared against the full denoiser on an equal footing.
        avg_tps = sum(r["tok_per_s"] for r in results) / max(len(results), 1)
        logger.info(
            "Block sampler: %.2f s total, %.0f tok/s (avg over %d sample(s), "
            "KV cache reuse across %d block(s) each)",
            wall_s, avg_tps, args.num_samples,
            max(1, (args.length + args.block_size - 1) // args.block_size),
        )

    elif args.thinking:
        if model.config.N_think <= 0:
            raise ValueError("This checkpoint was not trained with thinking tokens enabled.")
        logger.info(
            "Generating %d sample(s) with thinking (%d think + %d answer steps)...",
            args.num_samples, args.think_steps, args.answer_steps,
        )
        sampler = ThinkingDiffusionSampler(
            model=model,
            n_think=model.config.N_think,
            think_steps=args.think_steps,
            answer_steps=args.answer_steps,
            adaptive_think=args.adaptive_think,
            max_think_steps=args.max_think_steps,
            think_change_threshold=args.think_change_threshold,
            think_patience=args.think_patience,
            temperature=args.temperature,
            top_p=args.top_p,
            verbose=args.verbose,
            seed=args.seed,
            device=device,
            n_loops=args.n_loops,
        )
        results = sampler.generate(
            prompt=args.prompt,
            gen_length=args.length,
            num_samples=args.num_samples,
        )

        print("\n" + "=" * 60)
        for i, result in enumerate(results):
            if args.num_samples > 1:
                print(f"\n--- Sample {i + 1} ---")
            if args.verbose:
                print(f"[THINKING]: {result['thinking']}")
                print(f"[ANSWER]:   {result['answer']}")
            else:
                print(result["answer"])
        print("=" * 60)
    else:
        logger.info(
            "Full denoiser: %d sample(s), %d steps"
            "  [no KV cache — full sequence re-processed every step]",
            args.num_samples, args.steps,
        )

        t0 = time.perf_counter()
        results = denoise(
            model=model,
            prompt=args.prompt,
            gen_length=args.length,
            steps=args.steps,
            temperature=args.temperature,
            top_p=args.top_p,
            num_samples=args.num_samples,
            verbose=args.verbose,
            seed=args.seed,
            device=device,
            n_loops=args.n_loops,
            revise_threshold=args.revise_threshold,
            revise_t_max=args.revise_t_max,
        )
        wall_s = time.perf_counter() - t0
        n_tokens = args.num_samples * args.length

        print("\n" + "=" * 60)
        for i, text in enumerate(results):
            if args.num_samples > 1:
                print(f"\n--- Sample {i + 1} ---")
            print(text)
        print("=" * 60)
        # Separate the full-denoiser story: tok/s here is steps × seq_len
        # forward passes, not KV-cache-accelerated — do not conflate with
        # block-sampler tok/s numbers.
        logger.info(
            "Full denoiser: %.2f s, %.0f tok/s  "
            "(%d steps × %d tokens/step × %d sample(s), no KV reuse)",
            wall_s, n_tokens / max(wall_s, 1e-6),
            args.steps, args.length, args.num_samples,
        )


if __name__ == "__main__":
    main()
