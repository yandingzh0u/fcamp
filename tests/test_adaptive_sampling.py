"""Unit tests for the Holosoma death-frame adaptive motion sampler.

Run:  python tests/test_adaptive_sampling.py   (from the repo root)
"""
from __future__ import annotations

import torch

from env.adaptive_sampling import ADAPTIVE_SAMPLER_VERSION, AdaptiveTimestepsSampler


MOTION_FRAMES = 960
ENV_FPS = 50
NUM_BINS = MOTION_FRAMES // ENV_FPS + 1  # ~1-second bins -> 20 bins, 48 frames each


def _sampler(**kw) -> AdaptiveTimestepsSampler:
    params = dict(adaptive_kernel_size=1, adaptive_uniform_ratio=0.1, adaptive_alpha=0.001)
    params.update(kw)
    return AdaptiveTimestepsSampler(
        motion_time_step_total=MOTION_FRAMES,
        device="cpu",
        num_bins=NUM_BINS,
        **params,
    )


def check(name, cond):
    if not cond:
        raise AssertionError(f"FAILED: {name}")
    print(f"  ok: {name}")


def test_death_frame_maps_to_top_bin() -> None:
    sampler = _sampler()
    deaths = torch.full((4096,), 830, dtype=torch.long)
    sampler.update_current_bin_failed_count(deaths)
    sampler.update_bin_failed_count()

    expected_bin = (830 * NUM_BINS) // MOTION_FRAMES  # == 17
    top_bin = int(sampler.sampling_probabilities.argmax().item())
    lo, hi = sampler.bin_frame_bounds(torch.tensor([expected_bin]))
    check("death at 830 -> bin 17", expected_bin == 17)
    check("top sampled bin is the death bin", top_bin == expected_bin)
    check("death bin covers ~815-862", int(lo.item()) <= 830 < int(hi.item()) and 800 <= int(lo.item()) <= 820)


def test_zero_history_is_uniform() -> None:
    sampler = _sampler()
    probs = sampler.sampling_probabilities
    check("no deaths -> uniform probabilities", torch.allclose(probs, torch.full_like(probs, 1.0 / NUM_BINS)))
    check("no deaths -> normalized entropy ~1", abs(sampler.stats()["entropy"] - 1.0) < 1e-4)


def test_ema_folds_and_zeros_accumulator() -> None:
    sampler = _sampler(adaptive_alpha=0.5)
    sampler.update_current_bin_failed_count(torch.full((10,), 830, dtype=torch.long))
    check("accumulator holds raw failures pre-fold", float(sampler.current_bin_failed_count.sum()) == 10.0)
    sampler.update_bin_failed_count()
    check("EMA = alpha * current (first fold)", abs(float(sampler.bin_failed_count.sum()) - 5.0) < 1e-5)
    check("accumulator zeroed after fold", float(sampler.current_bin_failed_count.sum()) == 0.0)
    # A subsequent step with no deaths decays the EMA toward zero.
    sampler.update_bin_failed_count()
    check("EMA decays when no new failures", abs(float(sampler.bin_failed_count.sum()) - 2.5) < 1e-5)


def test_empty_failures_is_noop() -> None:
    sampler = _sampler()
    sampler.update_current_bin_failed_count(torch.empty(0, dtype=torch.long))
    check("empty death tensor adds nothing", float(sampler.current_bin_failed_count.sum()) == 0.0)


def test_sample_frames_respect_bounds() -> None:
    torch.manual_seed(0)
    sampler = _sampler()
    sampler.update_current_bin_failed_count(torch.full((1000,), 830, dtype=torch.long))
    sampler.update_bin_failed_count()
    frames = sampler.sample_frames(5000, min_phase=0, max_phase=900)
    check("sampled frames within [0, 900]", int(frames.min()) >= 0 and int(frames.max()) <= 900)
    # The death bin should dominate: most samples land near phase 830.
    near_death = ((frames >= 816) & (frames < 864)).float().mean().item()
    check("majority of samples near the death frame", near_death > 0.5)


def test_conditional_range_no_boundary_spikes() -> None:
    torch.manual_seed(0)
    sampler = _sampler()
    # Failures concentrated at frame 200 (far below the training window) must NOT leak in or
    # pile onto the window boundaries: sampling is conditioned on [760, 850], not clamped.
    sampler.update_current_bin_failed_count(torch.full((2000,), 200, dtype=torch.long))
    sampler.update_bin_failed_count()
    lo, hi = 760, 850
    frames = sampler.sample_frames(20000, min_phase=lo, max_phase=hi)
    check("all samples inside [760,850]", int(frames.min()) >= lo and int(frames.max()) <= hi)
    # No single boundary frame should absorb the out-of-range mass (clamp bug signature).
    at_lo = (frames == lo).float().mean().item()
    at_hi = (frames == hi).float().mean().item()
    check("no spike at min boundary", at_lo < 0.05)
    check("no spike at max boundary", at_hi < 0.05)
    # With no in-range failure mass it falls back to uniform over the in-range bins -> roughly flat.
    counts = torch.histc(frames.float(), bins=9, min=lo, max=hi)
    check("in-range distribution is roughly uniform", float(counts.max() / counts.min()) < 1.6)


def test_conditional_range_oversamples_in_range_failures() -> None:
    torch.manual_seed(0)
    sampler = _sampler()
    # Failure at frame 830 (inside the window) should dominate within [760, 850].
    sampler.update_current_bin_failed_count(torch.full((2000,), 830, dtype=torch.long))
    sampler.update_bin_failed_count()
    frames = sampler.sample_frames(20000, min_phase=760, max_phase=850)
    near_death = ((frames >= 816) & (frames <= 850)).float().mean().item()
    check("in-range failure bin dominates conditional samples", near_death > 0.6)


def test_state_dict_roundtrip_and_version_guard() -> None:
    sampler = _sampler()
    sampler.update_current_bin_failed_count(torch.full((100,), 830, dtype=torch.long))
    sampler.update_bin_failed_count()
    state = sampler.state_dict()
    check("state carries the sampler version", state["version"] == ADAPTIVE_SAMPLER_VERSION)

    restored = _sampler()
    check("matching version restores", restored.load_state_dict(state) is True)
    check("restored EMA matches", torch.allclose(restored.bin_failed_count, sampler.bin_failed_count))

    stale = _sampler()
    bad_state = dict(state)
    bad_state["version"] = ADAPTIVE_SAMPLER_VERSION - 1
    check("version mismatch is rejected", stale.load_state_dict(bad_state) is False)
    check("rejected state keeps fresh zeros", float(stale.bin_failed_count.sum()) == 0.0)
    check("None state is rejected", _sampler().load_state_dict(None) is False)


if __name__ == "__main__":
    for fn in (
        test_death_frame_maps_to_top_bin,
        test_zero_history_is_uniform,
        test_ema_folds_and_zeros_accumulator,
        test_empty_failures_is_noop,
        test_sample_frames_respect_bounds,
        test_conditional_range_no_boundary_spikes,
        test_conditional_range_oversamples_in_range_failures,
        test_state_dict_roundtrip_and_version_guard,
    ):
        print(f"== {fn.__name__} ==")
        fn()
    print("ALL ADAPTIVE SAMPLER TESTS PASSED")
