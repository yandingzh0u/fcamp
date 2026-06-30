"""Unit tests for the Holosoma causal-lookback adaptive motion sampler.

Run:  python tests/test_adaptive_sampling.py   (from the repo root)
"""
from __future__ import annotations

import torch

from env.adaptive_sampling import ADAPTIVE_SAMPLER_VERSION, AdaptiveTimestepsSampler


MOTION_FRAMES = 959  # frames 0..958
LOOKBACK_MIN = 20
LOOKBACK_MAX = 80


def _sampler(**kw) -> AdaptiveTimestepsSampler:
    params = dict(
        num_envs=4096,
        lookback_min=LOOKBACK_MIN,
        lookback_max=LOOKBACK_MAX,
        hard_ratio=0.7,
        uniform_ratio=0.3,
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


def test_hard_starts_strictly_before_failure() -> None:
    # A failure at frame 832 must produce hard starts ONLY in the causal lookback window
    # [832 - 80, 832 - 20] = [752, 812]; never on or after the failure frame.
    torch.manual_seed(0)
    sampler = _sampler(hard_ratio=1.0, uniform_ratio=0.0)
    sampler.update_current_failure_count(torch.full((4096,), 832, dtype=torch.long))
    sampler.update_failure_ema()
    frames = sampler.sample_frames(50000, min_phase=0, max_phase=MOTION_FRAMES - 1)
    lo, hi = 832 - LOOKBACK_MAX, 832 - LOOKBACK_MIN
    check("all hard starts within [752, 812]", int(frames.min()) >= lo and int(frames.max()) <= hi)
    check("no hard start at or after the failure frame 832", int(frames.max()) < 832)


def test_lookback_scores_window_bounds() -> None:
    sampler = _sampler()
    sampler.update_current_failure_count(torch.full((4096,), 832, dtype=torch.long))
    sampler.update_failure_ema()
    scores = sampler._lookback_scores()
    nonzero = torch.nonzero(scores > 0).flatten()
    check("lowest start that sees the failure is 752", int(nonzero.min()) == 832 - LOOKBACK_MAX)
    check("highest start that sees the failure is 812", int(nonzero.max()) == 832 - LOOKBACK_MIN)


def test_mixture_keeps_uniform_floor() -> None:
    torch.manual_seed(0)
    sampler = _sampler(hard_ratio=0.7, uniform_ratio=0.3)
    sampler.update_current_failure_count(torch.full((4096,), 832, dtype=torch.long))
    sampler.update_failure_ema()
    frames = sampler.sample_frames(60000, min_phase=0, max_phase=MOTION_FRAMES - 1)
    in_hard = ((frames >= 752) & (frames <= 812)).float().mean().item()
    # ~0.7 hard (in window) + 0.3 uniform spread over the whole clip -> majority in window but a
    # meaningful uniform tail outside it.
    check("hard window dominates (~0.7+)", in_hard > 0.6)
    check("uniform floor leaks outside the window", in_hard < 0.95)
    check("uniform floor can sample at/after the failure frame", int(frames.max()) >= 832)


def test_normalization_is_env_count_independent() -> None:
    # The same fraction of envs dying at the same frame must yield an identical EMA regardless of
    # num_envs (an 8192-env run is not twice as peaked as a 4096-env run).
    a = _sampler(num_envs=4096)
    a.update_current_failure_count(torch.full((4096,), 832, dtype=torch.long))
    a.update_failure_ema()
    b = _sampler(num_envs=8192)
    b.update_current_failure_count(torch.full((8192,), 832, dtype=torch.long))
    b.update_failure_ema()
    check("4096-env and 8192-env EMA identical", torch.allclose(a.failure_ema, b.failure_ema))
    check("EMA peak normalized to alpha (all envs died)", abs(float(a.failure_ema.max()) - a.adaptive_alpha) < 1e-6)


def test_zero_history_is_uniform_in_range() -> None:
    sampler = _sampler()
    probs = sampler.start_probabilities(760, 850)
    in_range = probs[760:851]
    check("no failures -> uniform over the valid range", torch.allclose(in_range, torch.full_like(in_range, 1.0 / in_range.numel())))
    check("no failures -> zero mass outside the range", float(probs[:760].sum() + probs[851:].sum()) < 1e-6)


def test_conditional_range_no_boundary_spikes() -> None:
    torch.manual_seed(0)
    sampler = _sampler()
    # Failures far below the window (frame 200) must not leak into [760, 850] nor pile on a
    # boundary (sampling is conditioned on the range, not clamped to it).
    sampler.update_current_failure_count(torch.full((4096,), 200, dtype=torch.long))
    sampler.update_failure_ema()
    lo, hi = 760, 850
    frames = sampler.sample_frames(40000, min_phase=lo, max_phase=hi)
    check("all samples inside [760,850]", int(frames.min()) >= lo and int(frames.max()) <= hi)
    at_lo = (frames == lo).float().mean().item()
    at_hi = (frames == hi).float().mean().item()
    check("no spike at min boundary", at_lo < 0.05)
    check("no spike at max boundary", at_hi < 0.05)
    counts = torch.histc(frames.float(), bins=9, min=lo, max=hi)
    check("in-range distribution roughly uniform", float(counts.max() / counts.min()) < 1.6)


def test_ema_folds_and_zeros_accumulator() -> None:
    sampler = _sampler(num_envs=10, adaptive_alpha=0.5)
    sampler.update_current_failure_count(torch.full((10,), 832, dtype=torch.long))
    check("accumulator holds raw failures pre-fold", float(sampler.current_failure_count.sum()) == 10.0)
    sampler.update_failure_ema()
    # alpha * (count / num_envs) = 0.5 * (10/10) = 0.5 at frame 832.
    check("EMA = alpha * normalized count (first fold)", abs(float(sampler.failure_ema[832]) - 0.5) < 1e-6)
    check("accumulator zeroed after fold", float(sampler.current_failure_count.sum()) == 0.0)
    sampler.update_failure_ema()
    check("EMA decays when no new failures", abs(float(sampler.failure_ema[832]) - 0.25) < 1e-6)


def test_empty_failures_is_noop() -> None:
    sampler = _sampler()
    sampler.update_current_failure_count(torch.empty(0, dtype=torch.long))
    check("empty death tensor adds nothing", float(sampler.current_failure_count.sum()) == 0.0)


def test_stats_fields() -> None:
    sampler = _sampler()
    sampler.update_current_failure_count(torch.full((4096,), 832, dtype=torch.long))
    sampler.update_failure_ema()
    stats = sampler.stats(0, MOTION_FRAMES - 1)
    for key in ("start_p10", "start_p50", "start_p90", "peak_fail_frame", "frac_before_peak", "failed_sum", "entropy"):
        check(f"stats has {key}", key in stats)
    check("peak failure frame is 832", int(stats["peak_fail_frame"]) == 832)
    check("median start lands before the wall", stats["start_p50"] < 832)
    # All 0.7 hard mass + the pre-wall part of the 0.3 uniform floor lands before the peak.
    check("most start mass is before the peak failure", stats["frac_before_peak"] > 0.9)


def test_state_dict_roundtrip_and_version_guard() -> None:
    sampler = _sampler()
    sampler.update_current_failure_count(torch.full((4096,), 832, dtype=torch.long))
    sampler.update_failure_ema()
    state = sampler.state_dict()
    check("state carries the sampler version", state["version"] == ADAPTIVE_SAMPLER_VERSION)
    check("version is 3 (per-frame design)", ADAPTIVE_SAMPLER_VERSION == 3)

    restored = _sampler()
    check("matching version restores", restored.load_state_dict(state) is True)
    check("restored EMA matches", torch.allclose(restored.failure_ema, sampler.failure_ema))

    stale = _sampler()
    bad_state = dict(state)
    bad_state["version"] = ADAPTIVE_SAMPLER_VERSION - 1
    check("version mismatch is rejected", stale.load_state_dict(bad_state) is False)
    check("rejected state keeps fresh zeros", float(stale.failure_ema.sum()) == 0.0)
    check("None state is rejected", _sampler().load_state_dict(None) is False)


if __name__ == "__main__":
    for fn in (
        test_hard_starts_strictly_before_failure,
        test_lookback_scores_window_bounds,
        test_mixture_keeps_uniform_floor,
        test_normalization_is_env_count_independent,
        test_zero_history_is_uniform_in_range,
        test_conditional_range_no_boundary_spikes,
        test_ema_folds_and_zeros_accumulator,
        test_empty_failures_is_noop,
        test_stats_fields,
        test_state_dict_roundtrip_and_version_guard,
    ):
        print(f"== {fn.__name__} ==")
        fn()
    print("ALL ADAPTIVE SAMPLER TESTS PASSED")
