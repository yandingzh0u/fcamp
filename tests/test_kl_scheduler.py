from __future__ import annotations

from types import SimpleNamespace

import torch

from algorithms.kl_scheduler import adaptive_lr_from_kl, normalized_kl
from algorithms.sfpo import SFPO


def test_normalized_kl_divides_by_units() -> None:
    assert normalized_kl(0.04, 4) == 0.01
    assert normalized_kl(0.01, 1) == 0.01
    # units clamped to >= 1
    assert normalized_kl(0.01, 0) == 0.01


def test_adaptive_lr_h4_raw_kl_in_band_does_not_drop_lr() -> None:
    # SFPO h=4 first-update scenario from the log: raw KL ~0.033, horizon=4.
    # per-step KL ~0.0083, target=0.01 -> within [0.005, 0.02] band -> hold LR.
    lr = 1e-3
    new_lr, kl_per_step = adaptive_lr_from_kl(
        raw_kl=0.033063, kl_units=4, target_per_step=0.01, lr=lr, min_lr=1e-5, max_lr=1e-2
    )
    assert abs(kl_per_step - 0.033063 / 4) < 1e-9
    assert new_lr == lr  # no change


def test_adaptive_lr_drops_when_per_step_kl_too_high() -> None:
    # raw KL 0.1 over h=4 -> per-step 0.025 > 2*0.01 -> drop.
    lr = 1e-3
    new_lr, kl_per_step = adaptive_lr_from_kl(
        raw_kl=0.1, kl_units=4, target_per_step=0.01, lr=lr, min_lr=1e-5, max_lr=1e-2
    )
    assert kl_per_step == 0.025
    assert new_lr < lr
    assert abs(new_lr - lr / 1.5) < 1e-12


def test_adaptive_lr_raises_when_per_step_kl_too_low() -> None:
    lr = 1e-3
    new_lr, _ = adaptive_lr_from_kl(
        raw_kl=0.001, kl_units=4, target_per_step=0.01, lr=lr, min_lr=1e-5, max_lr=1e-2
    )
    # per-step 0.00025 < 0.5*0.01 -> raise
    assert new_lr > lr
    assert abs(new_lr - lr * 1.5) < 1e-12


def test_adaptive_lr_clamps_to_min_and_max() -> None:
    low, _ = adaptive_lr_from_kl(
        raw_kl=1.0, kl_units=1, target_per_step=0.01, lr=1e-5, min_lr=1e-5, max_lr=1e-2
    )
    assert low == 1e-5
    high, _ = adaptive_lr_from_kl(
        raw_kl=1e-6, kl_units=1, target_per_step=0.01, lr=1e-2, min_lr=1e-5, max_lr=1e-2
    )
    assert high == 1e-2


def test_adaptive_lr_target_zero_or_nonpositive_kl_holds() -> None:
    lr = 1e-3
    assert adaptive_lr_from_kl(0.0, 4, 0.01, lr, 1e-5, 1e-2)[0] == lr
    assert adaptive_lr_from_kl(0.05, 4, 0.0, lr, 1e-5, 1e-2)[0] == lr


def test_sfpo_kl_units_is_per_frame() -> None:
    # SFPO actor loss uses per-frame / per-prefix ratios and the adaptive-LR
    # KL is the masked MEAN per-frame KL, so kl_units=1 (horizon-independent).
    # desired_kl is a per-frame budget compared directly against the per-frame KL.
    for h in (1, 2, 4, 8):
        cfg = SimpleNamespace(horizon=h)
        algo = SFPO(cfg=cfg, env=None, simulation_app=None)
        assert algo.kl_units == 1


def test_sfpo_per_frame_kl_aligned_with_desired_kl() -> None:
    # desired_kl is a per-frame budget; with kl_units=1, raw KL is per-frame
    # and is compared directly against desired_kl.
    cfg = SimpleNamespace(horizon=4, desired_kl=0.01)
    algo = SFPO(cfg=cfg, env=None, simulation_app=None)
    assert algo.kl_units == 1
    # per-frame KL 0.004 < 0.5*0.01 -> raise
    lr = 1e-3
    new_lr, kl_per_step = adaptive_lr_from_kl(
        raw_kl=0.004, kl_units=algo.kl_units, target_per_step=cfg.desired_kl,
        lr=lr, min_lr=1e-5, max_lr=1e-2,
    )
    assert kl_per_step < 0.5 * cfg.desired_kl
    assert new_lr > lr
