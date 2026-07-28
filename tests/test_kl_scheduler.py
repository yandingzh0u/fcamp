from __future__ import annotations

from components.optim.kl_scheduler import adaptive_lr_from_kl


def test_adaptive_lr_h4_raw_kl_in_band_does_not_drop_lr() -> None:
    # A four-unit aggregate KL of 0.033 has per-unit KL ~0.0083, which is
    # within the [0.005, 0.02] controller band.
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
