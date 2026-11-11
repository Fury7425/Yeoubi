# SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
# Model weights derived from this code are subject to the BigCode OpenRAIL-M license.
# Source code is licensed under Apache 2.0. See LICENSE for details.
"""
OpenMythos Recurrent-Depth Transformer integration for Yeoubi.

Adapts the three-stage Prelude → RecurrentBlock → Coda architecture from
OpenMythos (https://github.com/kyegomez/OpenMythos) for masked diffusion:

Key adaptations vs the original autoregressive OpenMythos design:
- Bidirectional attention (no causal mask) throughout, required for diffusion
- Diffusion timestep t_emb re-injected at every recurrence iteration
- Soft ACT weighting only — no hard per-token halting (preserves uniform refinement)
- Loop-index signal combined with t_emb (multiplicative + additive) so the model
  differentiates "iteration 3 at t=0.9 (noisy)" from "iteration 3 at t=0.1 (clean)"
- LTI A matrix parameterised as 0.99 * tanh(A_raw) — guaranteed spectral radius < 1
- Hidden state h reset fresh every forward() call (no cross-step state persistence)
- Per-loop alpha scaling lets loops specialise (early coarse, late fine refinement)
- Randomised loop count during training (loop dropout) so every loop prefix is useful
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, fields as dc_fields
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.checkpoint import checkpoint as grad_checkpoint

from .model import (
    BitBlock,
    BitLinear,
    BitMoEFFN,
    BytePatchCodec,
    ModelConfig,
    NoiseEmbedding,
    RMSNorm,
)
from .model import (
    apply_int4_scale,
    block_causal_attention_mask,
    install_runtime_attention_mask,
    patch_attention_mask,
)
from .byte_codec import PAD_TOKEN_ID
from .quantization import KVCache


# ---------------------------------------------------------------------------
# RDTConfig — extends ModelConfig with recurrent-depth parameters
# ---------------------------------------------------------------------------

@dataclass
class RDTConfig(ModelConfig):
    """Configuration for the Recurrent-Depth Transformer variant.

    All fields from ``ModelConfig`` are inherited. ``n_layers`` is ignored in
    favour of ``prelude_layers + recurrent_layers + coda_layers``.

    Extra args:
        prelude_layers: Number of standard BitBlock layers run once before
                        the recurrent block.
        recurrent_layers: Number of BitBlock layers in the shared recurrent
                          block (these weights are reused every loop iteration).
        coda_layers: Number of standard BitBlock layers run once after the
                     recurrent block.
        max_loop_iters: Maximum number of recurrence iterations (train default).
        lora_rank: Rank for the per-iteration LoRA depth adaptation.
        loop_dim: Number of hidden channels that receive the loop-index signal.
        use_act: Enable soft ACT output weighting across iterations.
        randomize_loops: If True, sample n_loops ~ Uniform(1, max_loop_iters)
                         during training so every loop prefix is meaningful.
    """
    prelude_layers: int = 4
    recurrent_layers: int = 2
    coda_layers: int = 4
    max_loop_iters: int = 8
    lora_rank: int = 32
    loop_dim: int = 64
    use_act: bool = True
    randomize_loops: bool = True
    # Unique bridge layers surrounding the recurrent core.  Zero defaults keep
    # legacy RDT checkpoint topology unchanged; new runs enable them explicitly.
    pre_recurrent_layers: int = 0
    post_recurrent_layers: int = 0
    adaptive_loops: bool = False
    min_loop_iters: int = 1
    loop_specific_norms: bool = False
    # Learned per-iteration vector instead of the sinusoid of
    # loop_idx / (max_loop_iters - 1). That fraction never exceeds 1, so
    # against frequencies in [1e-4, 1] the sinusoid is close to the same
    # (sin~0, cos~1) vector for every loop; a table with max_loop_iters rows
    # says "which loop" directly. False keeps legacy checkpoint topology; new
    # runs enable it explicitly.
    learned_loop_embedding: bool = False
    # Long inputs receive at least one additional refinement visit per
    # doubling beyond this byte length, even at low diffusion noise.
    context_adaptive_loops: bool = False
    context_loop_threshold: int = 4096
    # The recurrent core reads the prelude output ``e`` through the LTI
    # injection on every iteration. In the recurrent-depth design that is a
    # trained pathway: gradients from all iterations reach the prelude through
    # it, which is how the prelude learns to emit a good *anchor* rather than
    # just a good h_0. Detaching it (the previous behaviour) leaves
    # ``injection.B`` fitting a signal it cannot shape. Set True to restore the
    # old behaviour for ablation.
    detach_recurrent_anchor: bool = False
    # Std of the random initial recurrent state. The source architecture draws
    # s_0 ~ N(0, sigma^2 I) so the core must learn a *contraction to a fixed
    # point* from an arbitrary start -- that property, not the loop count
    # alone, is what makes inference-time depth extrapolation work. 0.0 keeps
    # the deterministic h_0 = prelude(x) start. Non-zero is only meaningful
    # together with detach_recurrent_anchor=False, since the injected anchor
    # then becomes the core's only route to the input.
    recurrent_state_init_std: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        if self.prelude_layers < 0 or self.recurrent_layers <= 0 or self.coda_layers < 0:
            raise ValueError("RDT layer counts must be non-negative and recurrent_layers positive")
        if self.pre_recurrent_layers < 0 or self.post_recurrent_layers < 0:
            raise ValueError("RDT bridge layer counts must be non-negative")
        if self.max_loop_iters < 1:
            raise ValueError("max_loop_iters must be at least 1")
        if not 1 <= self.min_loop_iters <= self.max_loop_iters:
            raise ValueError("min_loop_iters must be in [1, max_loop_iters]")
        if self.context_loop_threshold < 1:
            raise ValueError("context_loop_threshold must be >= 1")
        if self.loop_dim < 2 or self.loop_dim > self.hidden_dim or self.loop_dim % 2:
            raise ValueError("loop_dim must be even and in [2, hidden_dim]")
        if self.recurrent_state_init_std < 0.0:
            raise ValueError(
                "recurrent_state_init_std must be non-negative, got "
                f"{self.recurrent_state_init_std}"
            )


# ---------------------------------------------------------------------------
# BitLTIInjection — stable hidden-state update
# ---------------------------------------------------------------------------

class BitLTIInjection(nn.Module):
    """LTI-stable hidden state update: h_{t+1} = A·h_t + B(e) + block_out.

    A is a diagonal matrix parameterised as ``0.99 * tanh(A_raw)``, which
    guarantees every diagonal entry lies in (-0.99, 0.99) — spectral radius
    strictly less than 1 by construction.

    B is a BitLinear layer (ternary weights, INT8 activations) that projects
    the frozen prelude output e into the hidden dimension.

    Args:
        hidden_dim: Hidden dimension of the transformer.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.A_raw = nn.Parameter(torch.zeros(hidden_dim))
        self.B = BitLinear(hidden_dim, hidden_dim, act_mode="int8")

    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        block_out: torch.Tensor,
    ) -> torch.Tensor:
        """Compute h_{t+1} = A·h_t + B(e) + block_out.

        Args:
            h: (B, T, hidden_dim) current hidden state.
            e: (B, T, hidden_dim) frozen prelude output (constant anchor).
            block_out: (B, T, hidden_dim) output of the current recurrent pass.

        Returns:
            Updated hidden state (B, T, hidden_dim).
        """
        A_eff = 0.99 * torch.tanh(self.A_raw)  # (hidden_dim,) diagonal, |A_eff| < 0.99
        return A_eff * h + self.B(e) + block_out


# ---------------------------------------------------------------------------
# BitLoRAAdapter — per-iteration depth adaptation
# ---------------------------------------------------------------------------

class BitLoRAAdapter(nn.Module):
    """Per-iteration low-rank residual delta.

    A shared pair of BitLinear layers (down then up projection) with a
    learned per-iteration scalar gate applied before the up projection.
    The delta is added to the hidden state after the recurrent block to
    let each iteration refine the representation in a loop-aware way.

    Args:
        hidden_dim: Hidden dimension.
        lora_rank: Rank of the low-rank bottleneck.
        max_loop_iters: Maximum number of loop iterations (for per-iter gate).
        topk_ratio: TopK ratio for the up-projection activation quantizer.
    """

    def __init__(
        self,
        hidden_dim: int,
        lora_rank: int = 32,
        max_loop_iters: int = 8,
        topk_ratio: float = 0.55,
    ):
        super().__init__()
        self.down = BitLinear(hidden_dim, lora_rank, act_mode="int8")
        self.up = BitLinear(lora_rank, hidden_dim, act_mode="topk_int8",
                            topk_ratio=topk_ratio)
        # Learned per-iteration scalar: initialised to small value so delta
        # starts near-zero and the model gradually learns to use it.
        self.iter_gate = nn.Parameter(torch.full((max_loop_iters,), 0.1))

    def forward(self, x: torch.Tensor, loop_iter: int) -> torch.Tensor:
        """Return the low-rank residual delta for the given iteration.

        Args:
            x: (B, T, hidden_dim) hidden state.
            loop_iter: Zero-based current loop index.

        Returns:
            (B, T, hidden_dim) delta to add to the hidden state.
        """
        # Clamp loop_iter to the last trained gate so depth extrapolation
        # (n_loops > max_loop_iters at inference) does not index out of bounds.
        idx = min(loop_iter, self.iter_gate.shape[0] - 1)
        gate = self.iter_gate[idx]
        return gate * self.up(self.down(x))


# ---------------------------------------------------------------------------
# BitACTHalting — soft adaptive computation time
# ---------------------------------------------------------------------------

class BitACTHalting(nn.Module):
    """Soft ACT: produces per-position iteration weights without hard halting.

    All positions always iterate the full n_loops; the output is a weighted
    average of hidden states across iterations (weights normalised across the
    loop dimension after collection). This preserves uniform refinement
    depth across the sequence — important for diffusion where inconsistent
    refinement depth would create artifacts.

    Args:
        hidden_dim: Hidden dimension.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.weight_proj = BitLinear(hidden_dim, 1, act_mode="int8")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict per-position soft weight.

        Args:
            x: (B, T, hidden_dim) hidden state.

        Returns:
            (B, T, 1) weight in [0, 1].
        """
        return torch.sigmoid(self.weight_proj(x))


# ---------------------------------------------------------------------------
# Loop-index injection with timestep conditioning
# ---------------------------------------------------------------------------

def loop_index_inject(
    x: torch.Tensor,
    loop_idx: int,
    max_loop_iters: int,
    t_emb: torch.Tensor,
    loop_proj: nn.Linear,
    loop_proj2: nn.Linear,
    loop_dim: int = 64,
    loop_embed: Optional[nn.Embedding] = None,
) -> torch.Tensor:
    """Compute the loop-index + timestep conditioning delta.

    Uses both multiplicative and additive timestep conditioning:
        delta = loop_signal * loop_proj(t_emb) + loop_proj2(t_emb)
    The multiplicative path encodes how strongly each channel responds to the
    loop position at this noise level; the additive path ensures the signal
    never vanishes even when the multiplicative projection is near zero.

    The delta is zero-padded from loop_dim to hidden_dim before return so it
    can be added directly to the hidden state.

    Args:
        x: (B, T, hidden_dim) — used only for shape/device.
        loop_idx: Current zero-based loop iteration.
        max_loop_iters: Maximum loop iterations (for normalising the signal).
        t_emb: (B, t_embed_dim) noise level embedding.
        loop_proj: nn.Linear(t_embed_dim, loop_dim) — multiplicative scale.
        loop_proj2: nn.Linear(t_embed_dim, loop_dim) — additive bias.
        loop_dim: Number of channels to inject into.

    Returns:
        (B, T, hidden_dim) additive delta (zero for channels >= loop_dim).
    """
    B, T, D = x.shape
    device = x.device
    dtype = x.dtype

    if loop_embed is not None:
        # Learned table (RDTConfig.learned_loop_embedding). Clamp the index for
        # depth extrapolation, like the per-loop alpha and LoRA gate.
        idx = min(loop_idx, loop_embed.num_embeddings - 1)
        loop_signal = loop_embed.weight[idx].to(dtype)  # (loop_dim,)
    else:
        # Legacy sinusoidal encoding of the loop fraction
        half = loop_dim // 2
        frac = loop_idx / max(max_loop_iters - 1, 1)  # [0, 1]
        freqs = torch.arange(half, device=device, dtype=torch.float32) / half
        freqs = torch.exp(-math.log(10000.0) * freqs)
        angle = frac * freqs  # (half,)
        loop_signal = torch.cat([angle.sin(), angle.cos()], dim=0)  # (loop_dim,)
        loop_signal = loop_signal.to(dtype)

    # Timestep-conditioned scale/bias: (B, loop_dim)
    t_scale = loop_proj(t_emb)   # (B, loop_dim)
    t_bias  = loop_proj2(t_emb)  # (B, loop_dim)

    # Combined: (B, loop_dim), or (B, T, loop_dim) with per-position t_emb
    combined = loop_signal * t_scale + t_bias

    # Expand to (B, T, loop_dim) and zero-pad to (B, T, D)
    if combined.dim() == 2:
        combined = combined.unsqueeze(1).expand(B, T, loop_dim)
    delta = torch.zeros(B, T, D, device=device, dtype=dtype)
    delta[..., :loop_dim] = combined
    return delta


# ---------------------------------------------------------------------------
# BitRecurrentBlock — the shared block looped N times
# ---------------------------------------------------------------------------

class BitRecurrentBlock(nn.Module):
    """Shared recurrent transformer block looped N times.

    A single ``BitBlock`` (or stack of ``recurrent_layers`` blocks) is reused
    across all N loop iterations.  Between iterations the hidden state is
    updated via the LTI injection rule and a per-loop LoRA delta.

    The output is a soft-ACT weighted blend across all iteration hidden states,
    normalised so no single iteration dominates. The blending uses incremental
    residual updates: h_out = h_out + w_i * (h_i - h_out) which avoids the
    blurring effect of a direct weighted average, then divides by the achieved
    total weight so the output magnitude does not depend on ``n_loops``.

    Args:
        config: ``RDTConfig`` with all hyper-parameters.
        base_layer_idx: Index offset for ``BitBlock`` layer IDs (for KV cache
                        addressing when prelude blocks precede this block).
    """

    def __init__(self, config: RDTConfig, base_layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.base_layer_idx = base_layer_idx

        self.shared_blocks = nn.ModuleList([
            BitBlock(config, layer_idx=base_layer_idx + i)
            for i in range(config.recurrent_layers)
        ])

        self.injection = BitLTIInjection(config.hidden_dim)
        self.lora = BitLoRAAdapter(
            config.hidden_dim,
            lora_rank=config.lora_rank,
            max_loop_iters=config.max_loop_iters,
            topk_ratio=config.topk_ratio,
        )
        self.act = BitACTHalting(config.hidden_dim)

        # Per-loop learned output scaling — lets loops specialise.
        # Initialised to 1 / recurrent_layers so the first pass is roughly
        # the same magnitude as a standard residual block.
        self.alpha = nn.Parameter(
            torch.ones(config.max_loop_iters) / config.recurrent_layers
        )

        # Loop-index × timestep conditioning projections (standard fp32 Linear
        # for stable conditioning; not ternary).
        self.loop_proj  = nn.Linear(config.t_embed_dim, config.loop_dim, bias=False)
        self.loop_proj2 = nn.Linear(config.t_embed_dim, config.loop_dim, bias=True)
        if config.learned_loop_embedding:
            self.loop_embed = nn.Embedding(config.max_loop_iters, config.loop_dim)

        if config.loop_specific_norms:
            self.loop_norms = nn.ModuleList([
                RMSNorm(config.hidden_dim) for _ in range(config.max_loop_iters)
            ])
        else:
            self.loop_norm = RMSNorm(config.hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        t_emb: torch.Tensor,
        n_loops: int,
        kv_cache: Optional[KVCache] = None,
        rope_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the recurrent block for n_loops iterations.

        h is initialised fresh from x — state is NEVER carried across separate
        model.forward() calls (which would break the diffusion Markov property).

        Args:
            x: (B, T, hidden_dim) input hidden state (output of prelude).
            e: (B, T, hidden_dim) frozen prelude output used as LTI anchor.
            t_emb: (B, t_embed_dim) diffusion noise level embedding.
            n_loops: Number of iterations to run (may be < max_loop_iters at
                     train time when randomize_loops is enabled).
            kv_cache: Optional KVCache for inference (passed through to blocks).
            rope_offset: Position offset for RoPE.

        Returns:
            Tuple of (output, total_aux_loss).
        """
        if n_loops < 1:
            raise ValueError(f"n_loops must be at least 1, got {n_loops}")
        h = x
        std = self.config.recurrent_state_init_std
        if std > 0.0:
            # Random start (see RDTConfig.recurrent_state_init_std). Sampled
            # fresh per forward; never carried across model.forward() calls,
            # which would break the diffusion Markov property.
            h = h + torch.randn_like(h) * std
        total_aux_loss = x.new_zeros(())

        all_h: list[torch.Tensor] = []
        all_w: list[torch.Tensor] = []

        use_ckpt = self.config.gradient_checkpointing and self.training
        # Checkpoint every Nth *virtual* layer (each recurrent visit counts),
        # matching BitDiffusionTransformer. RDT used to ignore this knob.
        gc_n = max(1, self.config.gc_every_n_layers)

        for i in range(n_loops):
            # Inject combined loop-index + timestep signal into h
            # Normalise by max_loop_iters, not n_loops: with randomize_loops a
            # given iteration index must carry the same signal whatever the
            # sampled loop count, or no loop prefix is reusable and depth
            # extrapolation is meaningless.
            delta = loop_index_inject(
                h, i, self.config.max_loop_iters, t_emb,
                self.loop_proj, self.loop_proj2, self.config.loop_dim,
                loop_embed=getattr(self, "loop_embed", None),
            )
            h = h + delta

            # Run shared blocks — t_emb re-injected every iteration so each
            # loop stays anchored to the same diffusion noise level
            if self.config.loop_specific_norms:
                norm_idx = min(i, len(self.loop_norms) - 1)
                block_out = self.loop_norms[norm_idx](h)
            else:
                block_out = self.loop_norm(h)
            for block_offset, block in enumerate(self.shared_blocks):
                cache_idx = (
                    self.base_layer_idx
                    + i * self.config.recurrent_layers
                    + block_offset
                )
                if use_ckpt and cache_idx % gc_n == 0:
                    block_out, aux = grad_checkpoint(
                        block, block_out, t_emb, kv_cache, rope_offset, cache_idx,
                        use_reentrant=False,
                    )
                else:
                    block_out, aux = block(
                        block_out, t_emb, kv_cache, rope_offset, cache_idx
                    )
                total_aux_loss = total_aux_loss + aux

            # Per-iteration LoRA delta
            block_out = block_out + self.lora(block_out, i)

            # Per-loop output scaling (lets loops specialise). Clamp index for
            # depth extrapolation (n_loops > max_loop_iters at inference).
            alpha_idx = min(i, self.alpha.shape[0] - 1)
            block_out = self.alpha[alpha_idx] * block_out

            # LTI hidden state update
            h = self.injection(h, e, block_out)

            # Collect hidden state and soft ACT weight for this iteration
            if self.config.use_act:
                w = self.act(h)  # (B, T, 1)
                all_w.append(w)
            all_h.append(h)

        if self.config.use_act and len(all_w) > 0:
            # Normalise ACT weights across loop dimension to prevent early-loop
            # dominance — stack to (B, T, n_loops) then normalise across loops
            w_stack = torch.cat(all_w, dim=-1)                               # (B, T, n_loops)
            w_norm  = w_stack / (w_stack.sum(dim=-1, keepdim=True) + 1e-6)  # (B, T, n_loops)

            # Incremental residual blend: h_out = h_out + w_i * (h_i - h_out).
            # This is a sequential lerp, not sum(w_i * h_i): the effective
            # weight on h_i is w_i * prod_{j>i}(1 - w_j), so later loops lead.
            # That bias is the intent (progressive refinement), but the
            # effective weights sum to 1 - prod(1 - w_j), not to 1 -- roughly
            # 1.00 at n_loops=1, 0.75 at 2, 0.66 at 8. Since n_loops is
            # randomised in training and overridable at inference, that would
            # swing the output magnitude ~34% along the very axis this block
            # exists to vary, so divide the achieved total back out.
            h_out = torch.zeros_like(h)
            for idx, (w_i, h_i) in enumerate(zip(w_norm.unbind(-1), all_h)):
                h_out = h_out + w_i.unsqueeze(-1) * (h_i - h_out)
            total_w = 1.0 - torch.prod(1.0 - w_norm, dim=-1, keepdim=True)
            h_out = h_out / total_w.clamp_min(1e-6)
        else:
            # ACT disabled — use the final hidden state directly
            h_out = all_h[-1]

        return h_out, total_aux_loss


# ---------------------------------------------------------------------------
# BitRDTTransformer — full three-stage model
# ---------------------------------------------------------------------------

class BitRDTTransformer(nn.Module):
    """Recurrent-Depth Transformer for masked diffusion with BitNet quantization.

    Three-stage architecture:
      1. Prelude: ``prelude_layers`` standard BitBlock layers (run once).
      2. RecurrentBlock: a single shared BitBlock looped N times with LTI
         injection, LoRA adaptation, and soft ACT weighting.
      3. Coda: ``coda_layers`` standard BitBlock layers (run once).

    Follows the same forward contract as ``BitDiffusionTransformer``:
        ``forward(input_ids, t, kv_cache=None, rope_offset=0) -> (logits, aux_loss)``

    with an additional optional ``n_loops`` parameter to override the number
    of recurrence iterations at inference time (depth extrapolation).

    Args:
        config: ``RDTConfig`` with all model and RDT hyper-parameters.
    """

    def __init__(self, config: RDTConfig):
        super().__init__()
        self.config = config
        # Keep a reference under both attribute names for compatibility with
        # code that checks model.config (generic) vs model.rdt_config (RDT-specific).
        self.rdt_config = config

        # Vocabulary: base + mask token + optional think token
        n_special = 1
        if config.N_think > 0:
            n_special = 2
        vocab_total = config.vocab_size + n_special

        self.embed = nn.Embedding(vocab_total, config.hidden_dim)
        self.embed_drop = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.noise_embed = NoiseEmbedding(config.t_embed_dim, config.timestep_scale)

        if config.use_byte_patches:
            self.byte_codec = BytePatchCodec(config)

        if config.use_self_cond:
            self.self_cond_proj = nn.Linear(
                config.hidden_dim, config.hidden_dim, bias=False
            )

        # Prelude blocks (run once)
        self.prelude = nn.ModuleList([
            BitBlock(config, layer_idx=i)
            for i in range(config.prelude_layers)
        ])

        pre_bridge_start = config.prelude_layers
        self.pre_recurrent = nn.ModuleList([
            BitBlock(config, layer_idx=pre_bridge_start + i)
            for i in range(config.pre_recurrent_layers)
        ])

        # Shared recurrent block (looped N times)
        recurrent_start = config.prelude_layers + config.pre_recurrent_layers
        self.recurrent = BitRecurrentBlock(
            config, base_layer_idx=recurrent_start
        )

        # A recurrent weight is shared physically but each visit is a distinct
        # virtual layer for KV caching.
        post_bridge_start = recurrent_start + config.recurrent_layers * config.max_loop_iters
        self.post_recurrent = nn.ModuleList([
            BitBlock(config, layer_idx=post_bridge_start + i)
            for i in range(config.post_recurrent_layers)
        ])

        # Coda blocks (run once)
        coda_start = post_bridge_start + config.post_recurrent_layers
        self.coda = nn.ModuleList([
            BitBlock(config, layer_idx=coda_start + i)
            for i in range(config.coda_layers)
        ])
        self.cache_n_layers = coda_start + config.coda_layers

        self.final_norm = RMSNorm(config.hidden_dim)
        # Unembedding head — fp32, not quantized (same as BitDiffusionTransformer)
        self.unembed = nn.Linear(config.hidden_dim, vocab_total, bias=False)

        self._save_original_modes()
        apply_int4_scale(self, config.int4_absmean)
        self.apply(self._init_weights)
        # Scale for parameter visits rather than physical storage depth.
        n_residual = (
            config.prelude_layers
            + config.pre_recurrent_layers
            + config.recurrent_layers * config.max_loop_iters
            + config.post_recurrent_layers
            + config.coda_layers
        )
        self._scale_residual_init(n_residual)

        if config.use_self_cond:
            nn.init.zeros_(self.self_cond_proj.weight)

        if config.learned_loop_embedding:
            # Unit scale, like the sinusoid it replaces: the generic 0.02
            # embedding init above would start the multiplicative loop x
            # timestep path about 50x weaker.
            nn.init.normal_(self.recurrent.loop_embed.weight, std=1.0)

        # Tie input embedding and unembedding head (see ModelConfig.tie_embeddings).
        # Both `embed.weight` and `unembed.weight` keys are still serialized,
        # pointing at one shared tensor, so checkpoint round-trips are unchanged.
        if config.tie_embeddings:
            self.unembed.weight = self.embed.weight

    def _scale_residual_init(self, n_residual_blocks: int) -> None:
        """GPT-2-style scaled init for residual output projections."""
        scale = 1.0 / math.sqrt(max(2 * n_residual_blocks, 1))
        for name, module in self.named_modules():
            if isinstance(module, BitLinear) and (
                name.endswith(".attn.o_proj")
                or name.endswith(".ffn.down_proj")
                or name.endswith(".down_proj")
            ):
                with torch.no_grad():
                    module.latent_weight.mul_(scale)

    def _init_weights(self, module: nn.Module) -> None:
        from .model import Int8Linear
        if isinstance(module, Int8Linear):
            return
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _save_original_modes(self) -> None:
        for module in self.modules():
            if isinstance(module, BitLinear):
                module._save_original_mode()

    def set_activation_mode(self, mode: str) -> None:
        """Switch all BitLinear layers between A8 and A4 activation modes."""
        for module in self.modules():
            if isinstance(module, BitLinear):
                module.set_activation_mode(mode)

    def pack_for_inference(self) -> "BitRDTTransformer":
        """Pack every ternary projection, including the byte codec, in place."""
        for module in self.modules():
            if isinstance(module, BitLinear) and not getattr(
                module, "exclude_from_ternary", False
            ):
                module.pack_for_inference()
        for module in self.modules():
            if isinstance(module, BitMoEFFN):
                module.pack_for_inference()
        return self

    def cache_layers_for_loops(self, n_loops: Optional[int] = None) -> int:
        """Return the number of virtual layers required by the KV cache."""
        loops = self.config.max_loop_iters if n_loops is None else max(1, n_loops)
        return (
            self.config.prelude_layers
            + self.config.pre_recurrent_layers
            + self.config.recurrent_layers * loops
            + self.config.post_recurrent_layers
            + self.config.coda_layers
        )

    def _select_loop_count(self, t: torch.Tensor, seq_len: int = 0) -> int:
        """Choose depth from corruption level and long-context difficulty."""
        if not self.config.adaptive_loops:
            return self.config.max_loop_iters
        noise = float(t.detach().max().clamp(0, 1))  # single sync, adaptive_loops only
        span = self.config.max_loop_iters - self.config.min_loop_iters
        noise_loops = self.config.min_loop_iters + int(math.ceil(noise * span - 1e-8))
        length_loops = self.config.min_loop_iters
        if self.config.context_adaptive_loops and seq_len > self.config.context_loop_threshold:
            ratio = seq_len / self.config.context_loop_threshold
            length_loops += int(math.ceil(math.log2(ratio)))
        return min(self.config.max_loop_iters, max(noise_loops, length_loops))

    def forward(
        self,
        input_ids: torch.Tensor,
        t: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        rope_offset: int = 0,
        n_loops: Optional[int] = None,
        self_cond: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        block_causal_block_size: int = 0,
        unpadded: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: (B, T) token IDs (may contain mask tokens).
            t: (B,) noise levels in [0, 1], or (B, T) per byte for
               clean-prefix block training (constant inside each patch).
            kv_cache: Optional KVCache for inference (block diffusion).
            rope_offset: Position offset for RoPE (block diffusion).
            n_loops: Override the number of recurrence iterations. If None,
                     uses max_loop_iters (eval) or a random count (train when
                     randomize_loops=True). Values > max_loop_iters enable
                     inference-time depth extrapolation.
            self_cond: Optional detached soft embedding from a prior denoising
                       prediction, used only when configured.
        Returns:
            Tuple of (logits, aux_loss). logits is (B, T, vocab_total);
            aux_loss is the sum of MoE load-balance losses.
        """
        if n_loops is None:
            if self.training and self.config.randomize_loops:
                n_loops = random.randint(
                    self.config.min_loop_iters, self.config.max_loop_iters
                )
            elif kv_cache is not None:
                # Committed context and the current diffusion block must use
                # identical virtual depths.  Supporting a different adaptive
                # depth per denoising step would require one committed cache
                # bank per loop count, so cached block generation uses max
                # depth unless the caller supplies one fixed n_loops value.
                n_loops = self.config.max_loop_iters
            else:
                n_loops = self._select_loop_count(t, input_ids.shape[1])

        x = self.embed(input_ids)
        if attention_mask is None:
            attention_mask = input_ids.ne(PAD_TOKEN_ID)
        else:
            attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)
        x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        if self.config.use_self_cond and self_cond is not None:
            if self_cond.shape != x.shape:
                raise ValueError(
                    f"self_cond shape {tuple(self_cond.shape)} must match "
                    f"embedded input shape {tuple(x.shape)}"
                )
            x = x + self.self_cond_proj(self_cond) * attention_mask.unsqueeze(-1).to(x.dtype)
        x = self.embed_drop(x)
        patch_size = self.config.byte_patch_size if self.config.use_byte_patches else 1
        if t.dim() == 2:
            # Per-byte noise levels (clean-prefix block training); see
            # BitDiffusionTransformer.forward.
            t = t[:, ::patch_size]
        t_emb = self.noise_embed(t)  # (B, t_embed_dim) or (B, P, t_embed_dim)

        if block_causal_block_size <= 0 and self.training and self.config.block_causal_prob > 0.0:
            # Plain host RNG: the old version hashed t on-device and cost two
            # GPU syncs per step to recover one coin flip. This is a training
            # augmentation, not something a checkpoint has to reproduce.
            if random.random() < self.config.block_causal_prob:
                block_causal_block_size = random.randint(
                    self.config.block_causal_min_size, self.config.block_causal_max_size
                )
        # Settled before the codec runs: its local convs have to stop at the
        # same block edges the attention mask enforces.
        global_block_size = (
            max(1, math.ceil(block_causal_block_size / patch_size))
            if block_causal_block_size > 0 else 0
        )
        block_bytes = global_block_size * patch_size

        local_skip = None
        original_length = x.shape[1]
        global_rope_offset = rope_offset
        if self.config.use_byte_patches:
            global_rope_offset = (
                kv_cache.committed_len()
                if kv_cache is not None
                else rope_offset // self.config.byte_patch_size
            )
            x, local_skip, original_length = self.byte_codec.encode(
                x, attention_mask, block_bytes,
            )

        global_attention_mask = patch_attention_mask(attention_mask, patch_size)
        # See BitDiffusionTransformer.forward: no mask at all when the caller
        # vouches for an unpadded batch and the step is plain bidirectional.
        runtime_mask: Optional[torch.Tensor] = None if unpadded else global_attention_mask
        if global_block_size > 0:
            runtime_mask = block_causal_attention_mask(global_attention_mask, global_block_size)
        install_runtime_attention_mask(self, runtime_mask if kv_cache is None else None)

        total_aux_loss = x.new_zeros(())

        use_ckpt = self.config.gradient_checkpointing and self.training
        # Checkpoint every Nth *virtual* layer (each recurrent visit counts),
        # matching BitDiffusionTransformer. RDT used to ignore this knob.
        gc_n = max(1, self.config.gc_every_n_layers)

        # --- Prelude: standard BitBlocks, run once ---
        for block in self.prelude:
            if use_ckpt and block.layer_idx % gc_n == 0:
                x, aux = grad_checkpoint(
                    block, x, t_emb, kv_cache, global_rope_offset, use_reentrant=False
                )
            else:
                x, aux = block(
                    x, t_emb, kv_cache=kv_cache, rope_offset=global_rope_offset
                )
            total_aux_loss = total_aux_loss + aux

        # --- Unique semantic bridge before recurrence ---
        for block in self.pre_recurrent:
            if use_ckpt and block.layer_idx % gc_n == 0:
                x, aux = grad_checkpoint(
                    block, x, t_emb, kv_cache, global_rope_offset, use_reentrant=False
                )
            else:
                x, aux = block(
                    x, t_emb, kv_cache=kv_cache, rope_offset=global_rope_offset
                )
            total_aux_loss = total_aux_loss + aux

        # Prelude output is the LTI injection anchor (e), re-read every loop.
        # Gradients flow through it by default: that second path is how the
        # prelude learns to emit a good anchor, not merely a good h_0. It is
        # not a double-count to be suppressed.
        e = x.detach() if self.config.detach_recurrent_anchor else x

        # --- RecurrentBlock: shared block looped n_loops times ---
        x, rec_aux = self.recurrent(
            x, e, t_emb, n_loops, kv_cache=kv_cache, rope_offset=global_rope_offset
        )
        total_aux_loss = total_aux_loss + rec_aux

        # --- Unique consolidation bridge after recurrence ---
        runtime_post_cache_start = (
            self.config.prelude_layers
            + self.config.pre_recurrent_layers
            + self.config.recurrent_layers * n_loops
        )
        for bridge_idx, block in enumerate(self.post_recurrent):
            cache_idx = runtime_post_cache_start + bridge_idx
            if use_ckpt and cache_idx % gc_n == 0:
                x, aux = grad_checkpoint(
                    block, x, t_emb, kv_cache, global_rope_offset, cache_idx,
                    use_reentrant=False,
                )
            else:
                x, aux = block(
                    x, t_emb, kv_cache=kv_cache, rope_offset=global_rope_offset,
                    cache_layer_idx=cache_idx,
                )
            total_aux_loss = total_aux_loss + aux

        # --- Coda: standard BitBlocks, run once ---
        runtime_coda_cache_start = runtime_post_cache_start + self.config.post_recurrent_layers
        for coda_idx, block in enumerate(self.coda):
            cache_idx = runtime_coda_cache_start + coda_idx
            if use_ckpt and cache_idx % gc_n == 0:
                x, aux = grad_checkpoint(
                    block, x, t_emb, kv_cache, global_rope_offset, cache_idx,
                    use_reentrant=False,
                )
            else:
                x, aux = block(
                    x, t_emb, kv_cache=kv_cache, rope_offset=global_rope_offset,
                    cache_layer_idx=cache_idx,
                )
            total_aux_loss = total_aux_loss + aux

        x = self.final_norm(x)
        if self.config.use_byte_patches:
            x = self.byte_codec.decode(
                x, local_skip, original_length, attention_mask, block_bytes,
            )
        x = x * attention_mask.unsqueeze(-1).to(x.dtype)
        logits = self.unembed(x)
        return logits, total_aux_loss

    def count_parameters(self) -> dict:
        """Count total, trainable, and ternary parameters."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        latent = sum(p.numel() for n, p in self.named_parameters() if "latent_weight" in n)
        return {"total": total, "trainable": trainable, "latent_ternary": latent}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

# Pre-change values for fields added after checkpoints existed (see
# utils._LEGACY_MODEL_CONFIG_DEFAULTS). The recurrent anchor used to be
# unconditionally detached, and h_0 was always deterministic.
_LEGACY_RDT_CONFIG_DEFAULTS = {
    "use_sub_norm": False,
    "int4_absmean": False,
    "timestep_scale": 1.0,
    "detach_recurrent_anchor": True,
    "recurrent_state_init_std": 0.0,
}


def resolve_rdt_config(ckpt: dict, fallback: Optional[RDTConfig] = None) -> RDTConfig:
    """Reconstruct ``RDTConfig`` from a checkpoint dict.

    Args:
        ckpt: Raw checkpoint dict (as returned by ``read_checkpoint``).
        fallback: Optional ``RDTConfig`` to use if the checkpoint has none.

    Returns:
        Resolved ``RDTConfig``.

    Raises:
        ValueError: If the checkpoint has no rdt_config and no fallback given.
    """
    data = ckpt.get("rdt_config")
    if data is not None:
        # Filter out any unknown fields for forward-compatibility
        known = {f.name for f in dc_fields(RDTConfig)}
        filtered = {k: v for k, v in data.items() if k in known}
        # Back-fill fields added after this checkpoint was written, so its
        # stored topology and training semantics are reproduced rather than
        # silently replaced by a newer default. See
        # utils._LEGACY_MODEL_CONFIG_DEFAULTS for the same treatment upstream.
        for key, legacy in _LEGACY_RDT_CONFIG_DEFAULTS.items():
            filtered.setdefault(key, legacy)
        return RDTConfig(**filtered)
    if fallback is not None:
        return fallback
    raise ValueError(
        "Checkpoint has no rdt_config key and no fallback RDTConfig was provided."
    )
