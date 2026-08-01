from __future__ import annotations

import torch

from engine.restore_tolerance import float_restore_error


def test_restore_check_accepts_two_ulps_at_large_world_coordinates() -> None:
    expected = torch.tensor([64.0, -64.0, 0.25], dtype=torch.float32)
    restored = torch.nextafter(expected, torch.full_like(expected, float("inf")))
    absolute_error, bound_ratio, maximum_tolerance = float_restore_error(
        restored, expected
    )

    assert absolute_error == 7.62939453125e-6
    assert bound_ratio <= 0.5
    assert maximum_tolerance == 2.0 * absolute_error


def test_restore_check_still_rejects_real_large_and_small_state_errors() -> None:
    large_expected = torch.tensor([64.0], dtype=torch.float32)
    large_restored = large_expected + 1.0e-4
    _, large_ratio, _ = float_restore_error(large_restored, large_expected)
    assert large_ratio > 1.0

    small_expected = torch.tensor([0.0], dtype=torch.float32)
    small_restored = torch.tensor([1.1e-6], dtype=torch.float32)
    _, small_ratio, _ = float_restore_error(small_restored, small_expected)
    assert small_ratio > 1.0
