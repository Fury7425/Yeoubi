# SPDX-License-Identifier: Apache-2.0
"""Write two tiny random checkpoints that cover the native runtime's branches.

CI exports them with ``export.py --format gguf`` and runs
``check_runtime_parity.py`` on each, so the runtime is checked against
PyTorch without a trained model.

    rdt.pt: RDT, byte patches + decoder FFN, sub-norms, qk-norm, absmean INT4,
            learned loop embedding, loop-specific norms, bridges, soft ACT.
    std.pt: flat transformer, no byte patches, YaRN RoPE, absmax INT4,
            untied output head.
    moe.pt: RDT with MoE in every block, 4 experts / top-2, capacity factor
            1.0 so tokens are dropped and the capacity logic is exercised;
            use_act=False.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from yeoubi.model import BitDiffusionTransformer, ModelConfig  # noqa: E402
from yeoubi.rdt import BitRDTTransformer, RDTConfig  # noqa: E402

COMMON = dict(hidden_dim=128, n_heads=4, ffn_dim=256, t_embed_dim=64, max_seq_len=256)


def _randomize(model: torch.nn.Module) -> None:
    # Default init makes the tied head ~0.02 wide, so logits are near-uniform
    # and argmax agreement is a coin flip. Unit-scale embeddings give the head
    # real margins; small random offsets exercise the zero-initialised paths.
    with torch.no_grad():
        model.embed.weight.normal_(0.0, 1.0)
        if model.unembed.weight is not model.embed.weight:  # untied head
            model.unembed.weight.normal_(0.0, 1.0)
        for name, p in model.named_parameters():
            if name.endswith("subposition") or name.endswith("A_raw"):
                p.normal_(0.0, 0.5)
            elif name.endswith("router.weight"):
                # Default std 0.01 leaves expert probabilities near-tied, and
                # a tie decided by float noise is not a runtime bug.
                p.normal_(0.0, 1.0)


def _save(model: torch.nn.Module, config, path: str, rdt: bool) -> None:
    ckpt = {"model_state_dict": model.state_dict(), "model_config": asdict(config), "step": 0}
    if rdt:
        ckpt["rdt_config"] = asdict(config)
    torch.save(ckpt, path)
    print(f"wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--bench", action="store_true",
                   help="also write bench.pt: random weights at the step-25000 checkpoint's dimensions, "
                        "for timing the runtime (not for parity)")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)

    if args.bench:
        bench_cfg = RDTConfig(
            hidden_dim=768, n_heads=12, ffn_dim=2048, t_embed_dim=256, max_seq_len=4096,
            rope_scaling_type="yarn", rope_scaling_factor=4.0, qk_norm=True, use_rdt=True,
            use_byte_patches=True, byte_decoder_ffn=True, prelude_layers=4,
            pre_recurrent_layers=2, recurrent_layers=2, post_recurrent_layers=2, coda_layers=4,
            max_loop_iters=3, adaptive_loops=True, use_act=False,
        )
        bench = BitRDTTransformer(bench_cfg).eval()
        _randomize(bench)
        _save(bench, bench_cfg, os.path.join(args.out, "bench.pt"), rdt=True)

    rdt_cfg = RDTConfig(
        **COMMON, use_rdt=True, use_byte_patches=True, byte_decoder_ffn=True,
        qk_norm=True, prelude_layers=1, pre_recurrent_layers=1, recurrent_layers=1,
        post_recurrent_layers=1, coda_layers=1, max_loop_iters=3, lora_rank=32,
        loop_dim=32, learned_loop_embedding=True, loop_specific_norms=True,
    )
    rdt = BitRDTTransformer(rdt_cfg).eval()
    _randomize(rdt)
    _save(rdt, rdt_cfg, os.path.join(args.out, "rdt.pt"), rdt=True)

    std_cfg = ModelConfig(
        **COMMON, n_layers=2, int4_absmean=False, rope_scaling_type="yarn",
        rope_scaling_factor=4.0, tie_embeddings=False,
    )
    std = BitDiffusionTransformer(std_cfg).eval()
    _randomize(std)
    _save(std, std_cfg, os.path.join(args.out, "std.pt"), rdt=False)

    moe_cfg = RDTConfig(
        **COMMON, use_rdt=True, use_byte_patches=True, use_moe=True, n_experts=4,
        top_k_experts=2, moe_layers="all", expert_capacity_factor=1.0,
        prelude_layers=1, recurrent_layers=1, coda_layers=1, max_loop_iters=2,
        lora_rank=32, loop_dim=32,
        # The ACT head exists either way; the runtime must follow the flag
        # (the step-25000 checkpoint is use_act=False).
        use_act=False,
    )
    moe = BitRDTTransformer(moe_cfg).eval()
    _randomize(moe)
    _save(moe, moe_cfg, os.path.join(args.out, "moe.pt"), rdt=True)


if __name__ == "__main__":
    main()
