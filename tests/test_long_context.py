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

    @staticmethod
    def _padding_model():
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
        return BitDiffusionTransformer(cfg).eval()

    @staticmethod
    def _padded_prefix_delta(model):
        ids = torch.tensor([[65, 66, 67, 68, 69, 70, 71, 72]])
        padded = torch.cat(
            [ids, torch.full((1, 4), PAD_TOKEN_ID, dtype=torch.long)], dim=1
        )
        t = torch.tensor([0.5])
        with torch.no_grad():
            short, _ = model(ids, t)
            long, _ = model(padded, t)
        return (short - long[:, : ids.shape[1]]).abs().max().item()

    def test_padding_does_not_change_real_prefix(self):
        """The attention mask must not let PAD reach real positions.

        Asserted with the a4.8 TopK stage neutralized. TopK is a *hard*
        threshold on the activation vector, so it is discontinuous: it cannot
        preserve a bound like this one, and letting it participate turns this
        into a test of tie-breaking luck rather than of the mask (see
        ``test_topk_amplifies_padding_float_noise``). Everything else on the
        path is continuous, so a genuine mask leak still fails here.
        """
        model = self._padding_model()
        for _, module in model.named_modules():
            if getattr(module, "mode", None) == "topk_int8":
                module.topk_ratio = 1.0
        # Not exactly zero: masked SDPA reduces over a longer axis, and float
        # addition is not associative. ~1e-7 is that, and only that.
        self.assertLess(self._padded_prefix_delta(model), 1e-5)

    def test_topk_amplifies_padding_float_noise(self):
        """Documents a known, accepted property of the a4.8 TopK stage.

        The ~1e-7 float-reassociation gap above straddles the k-th-largest
        boundary whenever two activations are nearly tied there, and TopK then
        keeps a different element in the padded batch than in the unpadded
        one -- an O(1) divergence downstream. Consequence: logits for a real
        token depend slightly on how its batch was padded. This is inherent to
        a hard top-k, not a masking bug, and it is measured here so a future
        change to the quantizer cannot make it quietly worse.
        """
        model = self._padding_model()
        delta = self._padded_prefix_delta(model)
        self.assertGreater(delta, 1e-5, "TopK tie-flip no longer reproduces")
        self.assertLess(delta, 0.1, "padding divergence grew beyond the known band")


class LongContextPositionTests(unittest.TestCase):
    def test_rope_scaling_is_length_invariant(self):
        """A position must get the same rotation in a short and a long batch.

        The scaling mode used to flip once a sequence crossed
        original_max_seq_len, which gave position p two different embeddings
        depending only on how long its batch happened to be.
        """
        rope = RotaryEmbedding(
            8,
            max_seq_len=16,
            scaling_type="yarn",
            scaling_factor=4.0,
            original_max_seq_len=16,
        )
        x = torch.randn(1, 2, 20, 8)
        short = rope(x[:, :, :8])
        long = rope(x)
        torch.testing.assert_close(short, long[:, :, :8])
        self.assertTrue(rope._cache_scaled)

    def test_context_length_can_raise_rdt_depth(self):
        cfg = RDTConfig(
            hidden_dim=32,
            loop_dim=16,  # must be even and <= hidden_dim; the 64 default is not
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
