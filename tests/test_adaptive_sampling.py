"""Unit tests for the Holosoma adaptive motion sampler (v5: official per-bin failure EMA only).

Run:  python tests/test_adaptive_sampling.py   (from the repo root)
"""
from __future__ import annotations

import torch

from env.adaptive_sampling import ADAPTIVE_SAMPLER_VERSION, AdaptiveTimestepsSampler


MOTION_FRAMES = 959  # frames 0..958
ENV_FPS = 50


def _sampler(**kw) -> AdaptiveTimestepsSampler:
    params = dict(
        num_bins=0,          # auto -> floor(959/50)+1 = 20
        env_fps=ENV_FPS,
        adaptive_kernel_size=1,
        adaptive_lambda=0.8,
        adaptive_uniform_ratio=0.1,
        adaptive_alpha=0.001,
    )
    params.update(kw)
    return AdaptiveTimestepsSampler(
        motion_time_step_total=MOTION_FRAMES,
        device="cpu",
        **params,
    )


def check(name, cond):
    if not cond:
        raise AssertionError(f"FAILED: {name}")
    print(f"  ok: {name}")


def test_auto_num_bins() -> None:
    # ~1 bin per second: floor(959/50)+1 = 20.
    sampler = _sampler(num_bins=0)
    check("auto num_bins == 20", sampler.num_bins == 20)
    explicit = _sampler(num_bins=33)
    check("explicit num_bins honoured", explicit.num_bins == 33)


def test_bin_ema_fold_and_zero() -> None:
    sampler = _sampler(adaptive_alpha=0.5)
    sampler.update_current_failure_count(torch.full((10,), 834, dtype=torch.long))
    check("bin accumulator holds raw deaths pre-fold", float(sampler.current_bin_failed_count.sum()) == 10.0)
    sampler.update_failure_ema()
    # alpha * count: 0.5 * 10 = 5 in the death frame's bin.
    bin834 = sampler.frames_to_bins(torch.tensor([834]))[0].item()
    check("bin EMA = alpha*count (first fold)", abs(float(sampler.bin_failed_count[bin834]) - 5.0) < 1e-6)
    check("bin accumulator zeroed after fold", float(sampler.current_bin_failed_count.sum()) == 0.0)
    sampler.update_failure_ema()
    check("bin EMA decays with no new deaths", abs(float(sampler.bin_failed_count[bin834]) - 2.5) < 1e-6)


def test_empty_failures_is_noop() -> None:
    sampler = _sampler()
    sampler.update_current_failure_count(torch.empty(0, dtype=torch.long))
    check("empty death tensor adds nothing (bin)", float(sampler.current_bin_failed_count.sum()) == 0.0)


def test_official_bin_maps_death_to_bin17() -> None:
    # Official bin map: death 834 -> bin floor(834*20/959) = 17 (range [815,862]).
    sampler = _sampler()
    b = int(sampler.frames_to_bins(torch.tensor([834]))[0].item())
    check("death 834 maps to bin 17", b == 17)
    lo, hi = sampler.bin_frame_bounds(torch.tensor([17]))
    check("bin 17 lower bound is 815", int(lo[0]) == 815)
    check("bin 17 upper bound (exclusive) is 863", int(hi[0]) == 863)


def test_official_sampler_concentrates_in_death_bin() -> None:
    # The official sampler draws from the failure bin (plus the additive uniform floor across all
    # bins); the dominant mode sits inside the death bin [815,862], i.e. on/after the death frame.
    torch.manual_seed(0)
    sampler = _sampler()
    sampler.update_current_failure_count(torch.full((4096,), 834, dtype=torch.long))
    sampler.update_failure_ema()
    frames = sampler.sample_frames(60000, min_phase=0, max_phase=MOTION_FRAMES - 1)
    in_death_bin = ((frames >= 815) & (frames <= 862)).float().mean().item()
    check("official sampler concentrates in the death bin", in_death_bin > 0.5)


def test_official_conditional_no_boundary_spikes() -> None:
    # Deaths far below the range must not leak in nor pile on a boundary (conditional, not clamp).
    torch.manual_seed(0)
    sampler = _sampler()
    sampler.update_current_failure_count(torch.full((4096,), 200, dtype=torch.long))
    sampler.update_failure_ema()
    lo, hi = 760, 850
    frames = sampler.sample_frames(40000, min_phase=lo, max_phase=hi)
    check("all samples inside [760,850]", int(frames.min()) >= lo and int(frames.max()) <= hi)
    at_lo = (frames == lo).float().mean().item()
    at_hi = (frames == hi).float().mean().item()
    check("no spike at min boundary", at_lo < 0.05)
    check("no spike at max boundary", at_hi < 0.05)


def test_zero_history_is_uniform_over_bins() -> None:
    sampler = _sampler()
    probs = sampler.sampling_probabilities
    check("no failures -> uniform bin distribution", torch.allclose(probs, torch.full_like(probs, 1.0 / sampler.num_bins)))


def test_stats_fields() -> None:
    sampler = _sampler()
    sampler.update_current_failure_count(torch.full((4096,), 834, dtype=torch.long))
    sampler.update_failure_ema()
    stats = sampler.stats(0, MOTION_FRAMES - 1)
    for key in ("top_bin", "top_prob", "failed_sum", "entropy", "peak_bin"):
        check(f"stats has {key}", key in stats)
    check("peak bin is 17", int(stats["peak_bin"]) == 17)


def test_state_dict_roundtrip_and_version_guard() -> None:
    sampler = _sampler()
    sampler.update_current_failure_count(torch.full((4096,), 834, dtype=torch.long))
    sampler.update_failure_ema()
    state = sampler.state_dict()
    check("state carries the sampler version", state["version"] == ADAPTIVE_SAMPLER_VERSION)
    check("version is 5 (official bin-only design)", ADAPTIVE_SAMPLER_VERSION == 5)

    restored = _sampler()
    check("matching version restores", restored.load_state_dict(state) is True)
    check("restored bin EMA matches", torch.allclose(restored.bin_failed_count, sampler.bin_failed_count))

    stale = _sampler()
    bad_state = dict(state)
    bad_state["version"] = ADAPTIVE_SAMPLER_VERSION - 1
    check("version mismatch is rejected", stale.load_state_dict(bad_state) is False)
    check("rejected state keeps fresh bin zeros", float(stale.bin_failed_count.sum()) == 0.0)
    check("None state is rejected", _sampler().load_state_dict(None) is False)


if __name__ == "__main__":
    for fn in (
        test_auto_num_bins,
        test_bin_ema_fold_and_zero,
        test_empty_failures_is_noop,
        test_official_bin_maps_death_to_bin17,
        test_official_sampler_concentrates_in_death_bin,
        test_official_conditional_no_boundary_spikes,
        test_zero_history_is_uniform_over_bins,
        test_stats_fields,
        test_state_dict_roundtrip_and_version_guard,
    ):
        print(f"== {fn.__name__} ==")
        fn()
    print("ALL ADAPTIVE SAMPLER TESTS PASSED")
