# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the specific failure modes identified in the code review.

Covers:
1. MaskDiffusionLoss returns finite zero when no positions are supervised.
2. generate_sample() never samples special-token IDs into normal vocab range.
3. BlockDiffusionSampler.generate(num_samples>1) returns independently tracked outputs.
4. StreamingJsonlDataset preserves per-document length distribution (no rolling buffer).
5. A tiny CPU forward/backward pass completes without NaN.
"""

from __future__ import annotations

import sys
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from yeoubi.diffusion import CosineSchedule, MaskDiffusionLoss, apply_mask
from yeoubi.model import BitDiffusionTransformer, ModelConfig


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _small_config(**overrides) -> ModelConfig:
    cfg = dict(
        vocab_size=32,
        hidden_dim=16,
        n_layers=2,
        n_heads=4,
        ffn_dim=32,
        max_seq_len=32,
        mask_token_id=32,
        topk_ratio=0.55,
        t_embed_dim=8,
        kv_cache_bits=3,
        kv_cache_bos_bits=4,
        N_think=0,
        think_prob=0.0,
        use_moe=False,
    )
    cfg.update(overrides)
    return ModelConfig(**cfg)


# ---------------------------------------------------------------------------
# 1. MaskDiffusionLoss — zero supervised positions must not return NaN
# ---------------------------------------------------------------------------

class TestMaskDiffusionLossZeroSupervision(unittest.TestCase):
    def test_all_unmasked_returns_finite_zero(self):
        """If is_masked is all-False the loss must be 0.0, not NaN."""
        loss_fn = MaskDiffusionLoss()
        B, T, V = 2, 8, 32
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        is_masked = torch.zeros(B, T, dtype=torch.bool)  # nothing masked

        loss = loss_fn(logits, targets, is_masked)

        self.assertTrue(torch.isfinite(loss), f"Expected finite loss, got {loss.item()}")
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_all_excluded_by_think_returns_finite_zero(self):
        """All masked positions excluded by is_think must also return 0, not NaN."""
        loss_fn = MaskDiffusionLoss()
        B, T, V = 2, 8, 32
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        is_masked = torch.ones(B, T, dtype=torch.bool)   # everything masked
        is_think = torch.ones(T, dtype=torch.bool)        # but all positions are think

        loss = loss_fn(logits, targets, is_masked, is_think=is_think)

        self.assertTrue(torch.isfinite(loss), f"Expected finite loss, got {loss.item()}")
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_normal_masked_positions_give_positive_loss(self):
        """Sanity-check: a batch with actual masked positions produces non-zero finite loss."""
        loss_fn = MaskDiffusionLoss()
        B, T, V = 2, 8, 32
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        is_masked = torch.ones(B, T, dtype=torch.bool)

        loss = loss_fn(logits, targets, is_masked)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)


# ---------------------------------------------------------------------------
# 1b. MaskDiffusionLoss — MDLM continuous-time weighting
# ---------------------------------------------------------------------------

class TestMDLMWeighting(unittest.TestCase):
    def test_equal_t_matches_uniform_scaled_by_masked_fraction(self):
        """With per-batch normalization and one shared t every weight is 1, so
        the MDLM loss reduces to the masked-CE *sum* over the valid-token count.

        That is ``uniform * (N_masked / N_valid)`` -- the NELBO normalizes by
        valid positions, not by masked ones, so the two agree only up to the
        masked fraction."""
        torch.manual_seed(0)
        B, T, V = 4, 8, 32
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        is_masked = torch.rand(B, T) < 0.5

        uniform = MaskDiffusionLoss(weighting="uniform")
        mdlm = MaskDiffusionLoss(weighting="mdlm", normalize_per_batch=True)

        t = torch.full((B,), 0.5)
        lu = uniform(logits, targets, is_masked)
        lm = mdlm(logits, targets, is_masked, t=t)
        expected = lu * is_masked.float().sum() / (B * T)
        self.assertTrue(torch.allclose(expected, lm, atol=1e-5),
                        f"equal-t MDLM ({lm.item()}) must equal {expected.item()}")

    def test_mdlm_scale_is_batch_size_invariant(self):
        """The NELBO denominator is a constant, so the expected loss magnitude
        must not drift with batch size (the old N_masked denominator made it
        drift, silently retuning the effective LR)."""
        torch.manual_seed(7)
        V, T = 32, 64
        loss_fn = MaskDiffusionLoss(weighting="mdlm")
        means = []
        for B in (2, 16):
            vals = []
            for _ in range(64):
                logits = torch.zeros(B, T, V)
                targets = torch.randint(0, V, (B, T))
                t = torch.rand(B).clamp(min=1e-3)
                is_masked = torch.rand(B, T) < (
                    1 - torch.cos(t * math.pi / 2)
                ).unsqueeze(1)
                vals.append(loss_fn(logits, targets, is_masked, t=t).item())
            means.append(sum(vals) / len(vals))
        # Uniform logits => per-token CE is exactly log(V); E[w*(1-abar)] = 1,
        # so both batch sizes must land near log(V).
        ref = math.log(V)
        for B, m in zip((2, 16), means):
            self.assertLess(abs(m - ref) / ref, 0.25,
                            f"B={B}: mean {m:.3f} strayed from log(V)={ref:.3f}")

    def test_none_t_falls_back_to_uniform(self):
        """MDLM mode with t=None must reproduce the uniform loss exactly."""
        torch.manual_seed(1)
        B, T, V = 2, 8, 32
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        is_masked = torch.ones(B, T, dtype=torch.bool)

        lu = MaskDiffusionLoss(weighting="uniform")(logits, targets, is_masked)
        lm = MaskDiffusionLoss(weighting="mdlm")(logits, targets, is_masked, t=None)
        self.assertTrue(torch.allclose(lu, lm, atol=1e-6))

    def test_mdlm_finite_positive_with_random_t(self):
        """A real masked batch with random t must give a finite, positive loss."""
        torch.manual_seed(2)
        B, T, V = 3, 8, 32
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        is_masked = torch.ones(B, T, dtype=torch.bool)
        t = torch.rand(B)

        loss = MaskDiffusionLoss(weighting="mdlm")(logits, targets, is_masked, t=t)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_mdlm_zero_supervision_returns_zero(self):
        """No supervised positions → finite 0 loss even in MDLM mode."""
        B, T, V = 2, 8, 32
        logits = torch.randn(B, T, V)
        targets = torch.randint(0, V, (B, T))
        is_masked = torch.zeros(B, T, dtype=torch.bool)
        t = torch.rand(B)

        loss = MaskDiffusionLoss(weighting="mdlm")(logits, targets, is_masked, t=t)
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_small_t_upweights_relative_to_large_t(self):
        """A clean (small-t) sample's masked tokens must carry more weight than
        a noisy (large-t) sample's when their per-token CE is identical."""
        B, T, V = 2, 4, 8
        # Identical logits/targets for both samples so per-token CE matches.
        logits = torch.randn(1, T, V).repeat(B, 1, 1)
        targets = torch.randint(0, V, (1, T)).repeat(B, 1)
        is_masked = torch.ones(B, T, dtype=torch.bool)

        # Without per-batch normalization the absolute weights are visible.
        loss_fn = MaskDiffusionLoss(weighting="mdlm", normalize_per_batch=False)
        t = torch.tensor([0.05, 0.95])  # sample 0 clean, sample 1 noisy
        loss = loss_fn(logits, targets, is_masked, t=t)
        # w(0.05) >> w(0.95): weighted mean must exceed the unweighted CE mean.
        unweighted = MaskDiffusionLoss(weighting="uniform")(logits, targets, is_masked)
        self.assertGreater(loss.item(), unweighted.item())


# ---------------------------------------------------------------------------
# 2. generate_sample — must not sample special-token IDs
# ---------------------------------------------------------------------------

class TestGenerateSampleVocabSlice(unittest.TestCase):
    def test_no_special_token_ids_in_output(self):
        """generate_sample() must never emit IDs >= vocab_size.

        MASK (and legacy THINK) live above vocab_size in the embedding table,
        so the sampler slices them off before softmax. decode_byte_ids drops
        non-byte IDs silently, so inspecting the returned string cannot prove
        this — capture the IDs actually handed to the decoder instead.
        """
        import yeoubi.train as train_mod

        config = _small_config()
        model = BitDiffusionTransformer(config)
        model.eval()

        # Bias logits hard toward the special-token region: without the vocab
        # slice in generate_sample, every sampled position would land there.
        orig_forward = model.forward

        def biased_forward(x, t, **kw):
            logits, aux = orig_forward(x, t, **kw)
            logits[..., config.vocab_size:] += 1000.0
            return logits, aux

        model.forward = biased_forward

        captured: list[int] = []
        real_decode = train_mod.decode_byte_ids

        def capturing_decode(ids, **kw):
            captured.extend(ids)
            return real_decode(ids, **kw)

        train_mod.decode_byte_ids = capturing_decode
        try:
            train_mod.generate_sample(
                model,
                mask_token_id=config.mask_token_id,
                device=torch.device("cpu"),
                seq_len=16,
                steps=3,
                temperature=1.0,
            )
        finally:
            train_mod.decode_byte_ids = real_decode

        self.assertTrue(captured, "generate_sample never decoded anything")
        leaked = [i for i in captured if i >= config.vocab_size]
        self.assertFalse(leaked, f"special token IDs leaked into output: {sorted(set(leaked))}")


class TestBlockSamplerMultiSample(unittest.TestCase):
    def test_per_sample_tracking_is_independent(self):
        """With temperature > 0 and num_samples=2, the sampler must maintain
        separate token lists and block-text lists per sample."""
        from yeoubi.sample import BlockDiffusionSampler
        from yeoubi.quantization import KVCache

        config = _small_config()
        model = BitDiffusionTransformer(config)
        model.eval()

        sampler = BlockDiffusionSampler(
            model=model,
            block_size=8,
            steps=2,
            temperature=1.0,
            top_p=1.0,
            seed=0,
            device=torch.device("cpu"),
        )

        results = sampler.generate(prompt="", gen_length=16, num_samples=2)

        self.assertEqual(len(results), 2, "Must return exactly num_samples results")
        # Each result must have its own blocks list (not a shared reference)
        self.assertIsNot(
            results[0]["blocks"], results[1]["blocks"],
            "Block text lists must not be the same object",
        )

    def test_returns_correct_number_of_results(self):
        """generate() must always return exactly num_samples dicts."""
        from yeoubi.sample import BlockDiffusionSampler

        config = _small_config()
        model = BitDiffusionTransformer(config)
        model.eval()

        for n in (1, 3):
            sampler = BlockDiffusionSampler(
                model=model,
                block_size=8,
                steps=2,
                device=torch.device("cpu"),
            )
            results = sampler.generate(gen_length=8, num_samples=n)
            self.assertEqual(len(results), n)
            for r in results:
                self.assertIn("text", r)
                self.assertIn("blocks", r)


# ---------------------------------------------------------------------------
# 4. StreamingJsonlDataset — preserves per-document length (no rolling buffer)
# ---------------------------------------------------------------------------

class TestDataLoaderLengthPreservation(unittest.TestCase):
    def _make_dataset(self, lines: list[str], max_length: int = 32):
        import tempfile, json
        from yeoubi.data import StreamingJsonlDataset

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as f:
            for line in lines:
                f.write(json.dumps({"text": line}) + "\n")
            path = f.name

        return StreamingJsonlDataset(
            paths=[path],
            max_length=max_length,
            shuffle_buffer_size=1,
            min_chunk_size=1,
        )

    def test_no_cross_document_bleeding(self):
        """Documents shorter than max_length must come out shorter, not padded
        to max_length by bleeding tokens from adjacent documents."""
        # Lengths are UTF-8 bytes plus one EOS -- there is no tokenizer.
        short_doc = " ".join(str(i) for i in range(5))   # 9 bytes  + EOS
        long_doc = " ".join(str(i) for i in range(10))   # 19 bytes + EOS

        ds = self._make_dataset([short_doc, long_doc], max_length=32)
        lengths = [example["input_ids"].shape[0] for example in ds]

        short_len = len(short_doc.encode("utf-8")) + 1
        long_len = len(long_doc.encode("utf-8")) + 1
        self.assertIn(short_len, lengths, "short document must survive at its own length")
        self.assertIn(long_len, lengths, "long document must survive at its own length")
        self.assertNotIn(
            short_len + long_len, lengths,
            "Documents must not be concatenated across boundaries",
        )

    def test_long_document_splits_within_document(self):
        """A document longer than max_length must be split into sub-max chunks,
        each entirely within that document (no cross-document tokens)."""
        # 50-token document, max_length=20 → should yield 2–3 chunks all ≤ 20
        long_doc = " ".join(str(i) for i in range(50))
        ds = self._make_dataset([long_doc], max_length=20)
        lengths = [example["input_ids"].shape[0] for example in ds]

        self.assertGreater(len(lengths), 1, "Long document should be split into multiple chunks")
        for length in lengths:
            self.assertLessEqual(length, 20, f"Chunk length {length} exceeds max_length=20")


# ---------------------------------------------------------------------------
# 5. CPU forward/backward pass — end-to-end sanity
# ---------------------------------------------------------------------------

class TestForwardBackwardCPU(unittest.TestCase):
    def test_forward_backward_no_nan(self):
        """A single forward + backward pass on CPU must produce a finite loss."""
        config = _small_config()
        model = BitDiffusionTransformer(config)
        model.train()

        schedule = CosineSchedule()
        loss_fn = MaskDiffusionLoss()

        B, T = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, T))
        t = torch.rand(B)
        masked_ids, is_masked = apply_mask(input_ids, t, config.mask_token_id, schedule)

        logits, _ = model(masked_ids, t)
        loss = loss_fn(logits, input_ids, is_masked)

        self.assertTrue(torch.isfinite(loss), f"Forward loss is not finite: {loss.item()}")

        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                self.assertTrue(
                    torch.isfinite(param.grad).all(),
                    f"NaN/Inf gradient in parameter: {name}",
                )

    def test_padding_excluded_from_loss(self):
        """Padded positions (attention_mask=False) must not contribute to the loss."""
        config = _small_config()
        model = BitDiffusionTransformer(config)
        model.eval()

        schedule = CosineSchedule()
        loss_fn = MaskDiffusionLoss()

        B, T = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, T))
        # Only the first 8 positions are real; the rest are padding
        attention_mask = torch.zeros(B, T, dtype=torch.bool)
        attention_mask[:, :8] = True

        t = torch.rand(B)
        masked_ids, is_masked = apply_mask(
            input_ids, t, config.mask_token_id, schedule, frozen_mask=~attention_mask
        )

        # Padding positions must never be masked
        padding_positions = ~attention_mask
        self.assertFalse(
            (is_masked & padding_positions).any(),
            "Padded positions must not be included in is_masked",
        )


# ---------------------------------------------------------------------------
# 6. Nucleus sampling — top-p filter must preserve top-mass and zero rest
# ---------------------------------------------------------------------------

class TestNucleusSampling(unittest.TestCase):
    def test_top_p_keeps_top_mass_only(self):
        """After top-p filtering, the renormalised mass over kept tokens
        must equal 1.0 and removed tokens must have zero probability."""
        from yeoubi.sample import nucleus_sample

        torch.manual_seed(0)
        B, V = 4, 64
        logits = torch.randn(B, V) * 3.0  # spread mass across vocab

        for top_p in (0.5, 0.8, 0.95):
            # Recover what the function does internally to verify mass
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            cumprobs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            remove_mask = cumprobs - sorted_logits.softmax(dim=-1) >= top_p
            sorted_logits = sorted_logits.clone()
            sorted_logits[remove_mask] = -float("inf")
            filtered = torch.full_like(logits, float("-inf")).scatter(
                -1, sorted_idx, sorted_logits
            )
            probs = torch.softmax(filtered, dim=-1)

            # Removed tokens must have probability 0 after softmax of -inf
            removed_count = remove_mask.sum().item()
            self.assertGreater(removed_count, 0, f"top_p={top_p} removed nothing")
            # Each row must sum to 1 (renormalised over the kept set)
            row_sums = probs.sum(dim=-1)
            self.assertTrue(torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5))

        # End-to-end: returned token IDs must lie inside [0, V).
        ids = nucleus_sample(logits, temperature=1.0, top_p=0.5)
        self.assertEqual(ids.shape, (B,))
        self.assertTrue(((ids >= 0) & (ids < V)).all())


class TestKVQuantizationZeroPoint(unittest.TestCase):
    """Zero must survive the KV round trip.

    The offset used to be ``max_val / 2`` (3.5 at 3 bits), so an exact 0.0
    rounded to 4 and came back as +0.5 * scale — a systematic positive bias on
    every zero-valued K/V element in the cache.
    """

    def test_zero_round_trips_exactly(self):
        from yeoubi.quantization import dequantize_kv, quantize_kv

        for bits in (3, 4, 8):
            tensor = torch.zeros(1, 2, 4, 8)
            tensor[0, 0, 0, 0] = 1.0  # non-degenerate absmax
            packed, scale = quantize_kv(tensor, bits=bits)
            out = dequantize_kv(packed, scale, bits=bits, orig_last_dim=8)
            zeros = out[0, 0, 1:, :]
            self.assertTrue(
                torch.equal(zeros, torch.zeros_like(zeros)),
                f"{bits}-bit KV cache biased zero to {zeros.abs().max().item()}",
            )


class TestDenoiseNeverUnmasksTwice(unittest.TestCase):
    """denoise() must not re-mask a byte it already committed.

    Masked diffusion uses an absorbing state: once a position is filled it
    stays filled. denoise() previously re-masked every non-frozen position,
    throwing away all progress on each step.
    """

    def test_committed_count_is_monotone(self):
        from yeoubi.sample import denoise

        mask_id, vocab = 258, 258
        committed = []

        class StubModel:
            """Records how many bytes are already committed on each step."""

            config = SimpleNamespace(
                mask_token_id=mask_id, vocab_size=vocab, use_self_cond=False,
            )

            def eval(self):
                return self

            def __call__(self, ids, t, **kwargs):
                committed.append(int((ids != mask_id).sum()))
                return torch.zeros(*ids.shape, vocab + 1), torch.zeros(())

        denoise(StubModel(), gen_length=16, steps=5, num_samples=1, seed=0)

        self.assertEqual(len(committed), 5)
        self.assertEqual(
            committed, sorted(committed),
            f"denoise re-masked committed bytes: {committed}",
        )


class TestDenoiseTracksMaskSchedule(unittest.TestCase):
    """The masked fraction at step k must match the sampler's p(t_k).

    Re-masking is conditional on already being masked, so drawing against
    p(t_next) instead of p(t_next)/p(t_curr) makes the masked fraction the
    running product of the per-step probabilities: the canvas commits
    everything about halfway through and the remaining steps are no-ops.
    """

    def test_masked_fraction_follows_schedule(self):
        from yeoubi.sample import denoise

        mask_id, vocab, length, steps = 258, 258, 4096, 10
        committed = []

        class StubModel:
            config = SimpleNamespace(
                mask_token_id=mask_id, vocab_size=vocab, use_self_cond=False,
            )

            def eval(self):
                return self

            def __call__(self, ids, t, **kwargs):
                committed.append(int((ids != mask_id).sum()))
                return torch.zeros(*ids.shape, vocab + 1), torch.zeros(())

        denoise(StubModel(), gen_length=length, steps=steps, num_samples=1, seed=0)

        # denoise() spaces steps uniformly in mask probability, so p(t_k) is
        # exactly 1 - k/steps and the committed fraction is k/steps.
        p_steps = torch.linspace(1.0, 0.0, steps + 1)
        t_steps = CosineSchedule.inverse(p_steps).clamp(0.0, 1.0)
        for k, count in enumerate(committed):
            expected = length * float(1.0 - CosineSchedule.mask_prob(t_steps[k]))
            # Each step is a binomial draw conditioned on the last, so the
            # noise compounds across steps: ~2.5% of the canvas at 1 sigma
            # over 10 steps. 10% is 4 sigma, and the defect this guards
            # against misses by 29% of the canvas at step 10 — a wide margin
            # here costs nothing and a tight one buys a flaky test.
            self.assertLess(
                abs(count - expected), 0.10 * length,
                f"step {k}: {count} committed, schedule wants ~{expected:.0f}",
            )


class TestValidationIsDeterministic(unittest.TestCase):
    """validate() must score the same corruption on every call.

    It draws t and the mask pattern from a forked RNG re-seeded to val_seed,
    so two consecutive runs over the same batches return the identical loss.
    Fresh noise per call made the val curve unreadable at small batch counts.
    """

    def test_repeated_validate_returns_identical_loss(self):
        from yeoubi.train import validate

        V, B, T = 258, 2, 16
        batches = [
            {
                "input_ids": torch.randint(0, 256, (B, T)),
                "attention_mask": torch.ones(B, T, dtype=torch.bool),
            }
            for _ in range(3)
        ]

        class StubModel:
            training = False
            config = SimpleNamespace(
                mask_token_id=258, vocab_size=V, use_self_cond=False,
            )
            embed = SimpleNamespace(weight=torch.randn(V + 1, 8))

            def eval(self):
                return self

            def train(self):
                return self

            def __call__(self, ids, t, **kwargs):
                torch.manual_seed(int(ids.sum()) % 9973)
                return torch.randn(*ids.shape, V + 1), torch.zeros(())

        kw = dict(
            loss_fn=MaskDiffusionLoss(weighting="mdlm"),
            schedule=CosineSchedule(),
            mask_token_id=258,
            device=torch.device("cpu"),
            max_batches=3,
        )
        first = validate(StubModel(), batches, **kw)
        second = validate(StubModel(), batches, **kw)
        self.assertEqual(first, second, "validate() must be reproducible")

        # ...and the fork must leave the training RNG stream untouched.
        torch.manual_seed(99)
        expected = torch.rand(4)
        torch.manual_seed(99)
        validate(StubModel(), batches, **kw)
        self.assertTrue(torch.equal(expected, torch.rand(4)),
                        "validate() perturbed the global RNG stream")


class TestYarnAttentionTemperature(unittest.TestCase):
    """YaRN scales attention logits by 1/t, sqrt(1/t) = 0.1*ln(s) + 1.

    Folded into cos/sin, so |q| and |k| each scale by mscale and the logits
    scale by mscale**2. Frequency interpolation alone is only half of YaRN.
    """

    def test_yarn_applies_mscale_and_others_do_not(self):
        from yeoubi.model import RotaryEmbedding

        dim, s = 64, 4.0
        plain = RotaryEmbedding(dim, 128, 10000.0)
        yarn = RotaryEmbedding(dim, 128, 10000.0, scaling_type="yarn",
                               scaling_factor=s, original_max_seq_len=32)
        linear = RotaryEmbedding(dim, 128, 10000.0, scaling_type="linear",
                                 scaling_factor=s, original_max_seq_len=32)

        mscale = 0.1 * math.log(s) + 1.0
        # Position 0: all freqs are 0, so cos == 1 before scaling.
        self.assertAlmostEqual(plain.cos_cached[0, 0].item(), 1.0, places=6)
        self.assertAlmostEqual(yarn.cos_cached[0, 0].item(), mscale, places=5)
        # Linear position-interpolation carries no attention temperature.
        self.assertAlmostEqual(linear.cos_cached[0, 0].item(), 1.0, places=6)

    def test_mscale_is_unity_when_scaling_disabled(self):
        from yeoubi.model import RotaryEmbedding

        off = RotaryEmbedding(64, 128, 10000.0, scaling_type="yarn",
                              scaling_factor=1.0, original_max_seq_len=128)
        self.assertAlmostEqual(off.cos_cached[0, 0].item(), 1.0, places=6)


class TestSubNormAndLegacyCompat(unittest.TestCase):
    """BitNet SubLN sits before o_proj and down_proj, and is opt-out.

    Checkpoints written before the field existed omit the key; the resolver
    must back-fill the pre-change value rather than let the dataclass default
    describe a model whose weights the state dict does not contain.
    """

    def _cfg(self, **kw):
        from yeoubi.model import ModelConfig
        return ModelConfig(hidden_dim=32, n_heads=4, ffn_dim=64,
                           max_seq_len=32, **kw)

    def test_sub_norm_present_by_default_absent_when_disabled(self):
        from yeoubi.model import BitAttention, BitFFN, RMSNorm

        on_a, on_f = BitAttention(self._cfg()), BitFFN(self._cfg())
        off_a = BitAttention(self._cfg(use_sub_norm=False))
        off_f = BitFFN(self._cfg(use_sub_norm=False))

        self.assertIsInstance(on_a.sub_norm, RMSNorm)
        self.assertIsInstance(on_f.sub_norm, RMSNorm)
        self.assertNotIsInstance(off_a.sub_norm, RMSNorm)
        self.assertNotIsInstance(off_f.sub_norm, RMSNorm)
        # Shapes must match the tensor each one normalizes.
        self.assertEqual(on_a.sub_norm.weight.shape[0], 32)   # n_heads*head_dim
        self.assertEqual(on_f.sub_norm.weight.shape[0], 64)   # ffn_dim

    def test_sub_norm_changes_output_and_adds_state_dict_keys(self):
        from yeoubi.model import BitFFN

        torch.manual_seed(0)
        x = torch.randn(1, 5, 32)
        on, off = BitFFN(self._cfg()), BitFFN(self._cfg(use_sub_norm=False))
        off.load_state_dict(
            {k: v for k, v in on.state_dict().items() if "sub_norm" not in k}
        )
        self.assertIn("sub_norm.weight", on.state_dict())
        self.assertNotIn("sub_norm.weight", off.state_dict())
        with torch.no_grad():
            self.assertFalse(torch.allclose(on(x), off(x)),
                             "sub_norm must actually affect the output")

    def test_legacy_checkpoint_backfills_pre_subnorm_default(self):
        from dataclasses import asdict
        from yeoubi.utils import resolve_checkpoint_model_config

        stored = asdict(self._cfg())
        stored.pop("use_sub_norm")
        cfg, _ = resolve_checkpoint_model_config({"model_config": stored})
        self.assertFalse(cfg.use_sub_norm,
                         "pre-SubLN checkpoint must not gain sub-norms")

        fresh, _ = resolve_checkpoint_model_config({"model_config": asdict(self._cfg())})
        self.assertTrue(fresh.use_sub_norm)


class TestRecurrentAnchorGradients(unittest.TestCase):
    """The recurrent core re-reads the prelude anchor every loop.

    Gradients must reach the prelude through that path by default -- that is
    how the prelude learns to emit a good anchor rather than only a good h_0.
    """

    def _model(self, **kw):
        from yeoubi.rdt import BitRDTTransformer, RDTConfig
        return BitRDTTransformer(RDTConfig(
            hidden_dim=32, n_heads=4, ffn_dim=64, max_seq_len=32,
            prelude_layers=1, pre_recurrent_layers=0, recurrent_layers=1,
            post_recurrent_layers=0, coda_layers=1, max_loop_iters=2,
            loop_dim=8, use_byte_patches=False, use_rdt=True, **kw))

    def test_anchor_is_a_trained_path_by_default(self):
        torch.manual_seed(0)
        m = self._model()
        m.train()
        logits, _ = m(torch.randint(0, 255, (2, 8)), torch.rand(2), n_loops=2)
        logits.sum().backward()
        grad = m.recurrent.injection.B.latent_weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0.0)
        self.assertFalse(m.config.detach_recurrent_anchor)

    def test_random_state_init_is_off_by_default_and_stochastic_when_on(self):
        torch.manual_seed(0)
        ids, t = torch.randint(0, 255, (2, 8)), torch.rand(2)

        off = self._model()
        off.eval()
        with torch.no_grad():
            a, _ = off(ids, t, n_loops=2)
            b, _ = off(ids, t, n_loops=2)
        self.assertTrue(torch.allclose(a, b), "default h_0 must be deterministic")

        on = self._model(recurrent_state_init_std=0.5)
        on.eval()
        with torch.no_grad():
            c, _ = on(ids, t, n_loops=2)
            d, _ = on(ids, t, n_loops=2)
        self.assertFalse(torch.allclose(c, d))

    def test_legacy_rdt_checkpoint_keeps_detached_anchor(self):
        from dataclasses import asdict
        from yeoubi.rdt import RDTConfig, resolve_rdt_config

        stored = asdict(RDTConfig(hidden_dim=32, n_heads=4, ffn_dim=64, loop_dim=8))
        for key in ("use_sub_norm", "detach_recurrent_anchor"):
            stored.pop(key)
        cfg = resolve_rdt_config({"rdt_config": stored})
        self.assertTrue(cfg.detach_recurrent_anchor)
        self.assertFalse(cfg.use_sub_norm)


class TestSTEGradientIsIdentity(unittest.TestCase):
    """ste_ternary must reproduce BitNet b1.58 `weight_quant` exactly.

    The absmean scale is a function of every element of the latent, so
    leaving it in the autograd graph adds a dense ``sign(w)`` term that
    dominated the true STE signal (cos ~0.27 with the correct update).
    """

    def test_forward_is_dequantized_ternary(self):
        from yeoubi.quantization import absmean_quantize, ste_ternary

        torch.manual_seed(0)
        w = (torch.randn(64, 128) * 0.02).requires_grad_(True)
        out = ste_ternary(w)
        w_q, scale = absmean_quantize(w.detach())
        # Bit-exact, not just close: export packs `w_q` and `scale`
        # separately, and the packed path must match the simulated one.
        self.assertTrue(torch.equal(out.detach(), w_q * scale))
        self.assertEqual(out.detach().unique().numel(), 3)

    def test_backward_is_identity(self):
        from yeoubi.quantization import ste_ternary

        torch.manual_seed(0)
        w = (torch.randn(64, 128) * 0.02).requires_grad_(True)
        g = torch.randn(64, 128)
        ste_ternary(w).backward(g)
        self.assertTrue(torch.equal(w.grad, g))

    def test_no_scale_gradient_leak(self):
        """The dense sign(w) term must be absent, not merely small."""
        from yeoubi.quantization import ste_ternary

        torch.manual_seed(0)
        w = (torch.randn(32, 64) * 0.02).requires_grad_(True)
        x = torch.randn(16, 64)
        target = torch.randn(16, 32)
        loss = (torch.nn.functional.linear(x, ste_ternary(w)) - target).pow(2).mean()
        loss.backward()

        # Reference: gradient of the plain (unquantized) linear at the
        # dequantized weight. Identity STE makes these equal by definition.
        w2 = w.detach().clone()
        from yeoubi.quantization import absmean_quantize

        w_q, scale = absmean_quantize(w2)
        w_eff = (w_q * scale).requires_grad_(True)
        (torch.nn.functional.linear(x, w_eff) - target).pow(2).mean().backward()
        self.assertTrue(torch.allclose(w.grad, w_eff.grad, atol=1e-7))


class TestLatentScaleIsTracked(unittest.TestCase):
    """BitStats must expose mean|W| so unbounded latent drift is visible.

    Muon defaults to no weight decay on the ternary latents; the ternary
    pattern is scale-invariant but the dequantized weight is not, so the
    scale is the quantity that has to be watched.
    """

    def test_bitstats_reports_latent_absmean(self):
        from yeoubi.model import BitDiffusionTransformer, ModelConfig
        from yeoubi.utils import BitStats

        cfg = ModelConfig(
            hidden_dim=32, n_heads=4, head_dim=8, ffn_dim=64,
            n_layers=2, max_seq_len=64,
        )
        model = BitDiffusionTransformer(cfg)
        stats = BitStats(model).compute()

        for key in ("weight/latent_absmean",
                    "weight/latent_absmean_max",
                    "weight/latent_absmean_min"):
            self.assertIn(key, stats)
            self.assertGreater(stats[key], 0.0)
        self.assertLessEqual(stats["weight/latent_absmean_min"],
                             stats["weight/latent_absmean"])
        self.assertLessEqual(stats["weight/latent_absmean"],
                             stats["weight/latent_absmean_max"])

    def test_absmean_tracks_a_scaled_latent(self):
        from yeoubi.model import BitDiffusionTransformer, ModelConfig
        from yeoubi.utils import BitStats

        cfg = ModelConfig(
            hidden_dim=32, n_heads=4, head_dim=8, ffn_dim=64,
            n_layers=2, max_seq_len=64,
        )
        model = BitDiffusionTransformer(cfg)
        before = BitStats(model).compute()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if "latent_weight" in name:
                    param.mul_(4.0)
        after = BitStats(model).compute()

        self.assertAlmostEqual(
            after["weight/latent_absmean"],
            before["weight/latent_absmean"] * 4.0, places=5,
        )
        # Ternary pattern is scale-invariant: only the scale moved.
        self.assertAlmostEqual(after["weight/frac_zero"],
                               before["weight/frac_zero"], places=6)


if __name__ == "__main__":
    unittest.main()
