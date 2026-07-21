"""Static regression coverage for the long-context model path.

These tests intentionally use tiny models. They validate masking, RoPE mode
selection, patch compression, and context-aware recurrence without requiring a
long-context training run.
"""

import unittest

import torch

from yeoubi.byte_codec import PAD_TOKEN_ID
from yeoubi.model import (
    BitDiffusionTransformer,
    ModelConfig,
    RotaryEmbedding,
    block_causal_attention_mask,
    patch_attention_mask,
)
from yeoubi.rdt import BitRDTTransformer, RDTConfig


class LongContextMaskTests(unittest.TestCase):
    def test_patch_mask_marks_only_nonempty_patches(self):
        mask = torch.tensor([[True, True, False, False, False, True, False]])
        patched = patch_attention_mask(mask, 4)
        self.assertEqual(patched.tolist(), [[True, True]])

    def test_block_causal_mask_is_bidir_within_block(self):
        valid = torch.ones(1, 6, dtype=torch.bool)
        mask = block_causal_attention_mask(valid, block_size=2)[0]
        self.assertTrue(mask[0, 1])
        self.assertFalse(mask[0, 2])
        self.assertTrue(mask[4, 0])
        self.assertTrue(mask[4, 5])

    def test_padding_does_not_change_real_prefix(self):
        torch.manual_seed(0)
        cfg = ModelConfig(
            hidden_dim=32,
            n_layers=2,
            n_heads=4,
            ffn_dim=64,
            max_seq_len=32,
            use_byte_patches=True,
            byte_patch_size=4,
            byte_local_kernel=3,
            qk_norm=True,
        )
        model = BitDiffusionTransformer(cfg).eval()
        ids = torch.tensor([[65, 66, 67, 68, 69, 70, 71, 72]])
        padded = torch.cat(
            [ids, torch.full((1, 4), PAD_TOKEN_ID, dtype=torch.long)], dim=1
        )
        t = torch.tensor([0.5])
        with torch.no_grad():
            short, _ = model(ids, t)
            long, _ = model(padded, t)
        torch.testing.assert_close(short, long[:, : ids.shape[1]], atol=1e-5, rtol=1e-5)


class LongContextPositionTests(unittest.TestCase):
    def test_rope_uses_original_mode_inside_training_window(self):
        rope = RotaryEmbedding(
            8,
            max_seq_len=16,
            scaling_type="yarn",
            scaling_factor=4.0,
            original_max_seq_len=16,
        )
        x = torch.randn(1, 2, 8, 8)
        rope(x)
        self.assertFalse(rope._cache_scaled)
        rope(torch.randn(1, 2, 20, 8))
        self.assertTrue(rope._cache_scaled)

    def test_context_length_can_raise_rdt_depth(self):
        cfg = RDTConfig(
            hidden_dim=32,
            n_heads=4,
            ffn_dim=64,
            prelude_layers=1,
            recurrent_layers=1,
            coda_layers=1,
            min_loop_iters=1,
            max_loop_iters=3,
            adaptive_loops=True,
            context_adaptive_loops=True,
            context_loop_threshold=16,
            use_act=False,
        )
        model = BitRDTTransformer(cfg)
        low_noise = torch.tensor([0.0])
        self.assertEqual(model._select_loop_count(low_noise, 16), 1)
        self.assertEqual(model._select_loop_count(low_noise, 32), 2)
        self.assertEqual(model._select_loop_count(low_noise, 64), 3)


if __name__ == "__main__":
    unittest.main()
