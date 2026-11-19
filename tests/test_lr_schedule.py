# SPDX-License-Identifier: BigCode-OpenRAIL-M-1.0
# Model weights derived from this code are subject to the BigCode OpenRAIL-M license.
# Source code is licensed under Apache 2.0. See LICENSE for details.
"""Warmup-stable-decay LR multiplier and the A4-off activation schedule."""

import pytest
import torch

import yeoubi.train as train_mod


def test_wsd_holds_then_decays_linearly_to_floor():
    wsd = lambda step: train_mod._wsd(step, 100, 1000, 400, 0.1)  # noqa: E731
    assert wsd(50) == pytest.approx(0.5)      # warmup
    assert wsd(100) == pytest.approx(1.0)
    assert wsd(999) == pytest.approx(1.0)     # stable phase
    assert wsd(1200) == pytest.approx(0.55)   # halfway down
    assert wsd(1400) == pytest.approx(0.1)
    assert wsd(5000) == pytest.approx(0.1)    # floor holds past the end


def test_decay_flags_switch_a_resumed_cosine_scheduler():
    # A branch resumes a cosine run's scheduler state; the decay must take over from it.
    model = torch.nn.Linear(2, 2)
    cosine = train_mod.TrainConfig(use_muon=False, warmup_steps=0, max_steps=10_000)
    _, sched = train_mod._build_optimizer_and_scheduler(model, cosine)
    for _ in range(100):
        sched.step()
    state = sched.state_dict()

    decay = train_mod.TrainConfig(
        use_muon=False, warmup_steps=0, max_steps=200, decay_start_step=100, decay_steps=100,
    )
    opt, sched = train_mod._build_optimizer_and_scheduler(model, decay)
    sched.load_state_dict(state)
    for _ in range(50):
        sched.step()
    assert opt.optimizers[0].param_groups[0]["lr"] == pytest.approx(decay.lr * 0.55)


def test_a4_fraction_zero_never_switches():
    schedule = train_mod.ActivationSchedule(1000, 0.0)
    assert schedule.get_mode(1000) == "A8"
    assert train_mod.ActivationSchedule(1000, 0.1).get_mode(900) == "A4"
