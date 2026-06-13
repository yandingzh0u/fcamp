from __future__ import annotations

import torch

from env.adaptive_sampling import (
    allocate_sampling_counts,
    build_adaptive_phase_probabilities,
    compute_failure_rates,
    stratified_uniform_offsets,
)


def _probabilities(
    rates: torch.Tensor,
) -> torch.Tensor:
    return build_adaptive_phase_probabilities(
        rates,
        torch.arange(912),
        motion_num_frames=960,
    )


def test_failure_rate_is_invariant_to_sampling_volume() -> None:
    failures = torch.tensor([9.0, 90.0, 0.0])
    exposures = torch.tensor([10.0, 100.0, 0.0])
    rates = compute_failure_rates(failures, exposures)

    assert torch.allclose(rates, torch.tensor([0.9, 0.9, 0.9]))


def test_adaptive_probabilities_follow_failure_rate_not_failure_count() -> None:
    rates = torch.ones(20)
    rates[-2] = 4.0
    probabilities = _probabilities(rates)

    assert probabilities[880] > probabilities[100]
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0))


def test_sampling_quotas_are_exact() -> None:
    start, uniform, adaptive = allocate_sampling_counts(
        2048,
        uniform_ratio=0.1,
        start_phase_ratio=0.25,
    )

    assert start == 512
    assert uniform == 205
    assert adaptive == 1331


def test_zero_failure_history_falls_back_to_uniform_sampling() -> None:
    rates = compute_failure_rates(torch.zeros(20), torch.zeros(20))
    probabilities = _probabilities(rates)
    expected = torch.full_like(probabilities, 1.0 / probabilities.numel())

    assert torch.allclose(probabilities, expected)


def test_stratified_uniform_quota_covers_the_full_clip() -> None:
    torch.manual_seed(0)
    offsets = stratified_uniform_offsets(912, 205, device="cpu")
    coarse_bins = torch.clamp((offsets * 19) // 912, 0, 18)

    assert offsets.min() < 5
    assert offsets.max() > 906
    assert torch.unique(coarse_bins).numel() == 19
