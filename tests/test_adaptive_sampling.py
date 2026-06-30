"""Unit tests for the Holosoma adaptive motion sampler (v4: official failure-bin base +
optional causal second stage bound to the rollout credit window).

Run:  python tests/test_adaptive_sampling.py   (from the repo root)
"""
from __future__ import annotations

import torch

from env.adaptive_sampling import ADAPTIVE_SAMPLER_VERSION, AdaptiveTimestepsSampler


MOTION_FRAMES = 959  # frames 0..958
ENV_FPS = 50
CAUSAL_H = 24            # PPO/FPO rollout credit horizon (num_steps_per_env)
CAUSAL_DECAY = 0.9405    # gamma * lambda


def _sampler(**kw) -> AdaptiveTimestepsSampler:
    params = dict(
        num_bins=0,          # auto -> floor(959/50)+1 = 20
        env_fps=ENV_FPS,
        adaptive_kernel_size=1,
        adaptive_lambda=0.8,
        adaptive_uniform_ratio=0.1,
        adaptive_alpha=0.001,
        causal_max_ratio=0.5,
        causal_horizon=0,
        causal_decay=0.0,
        causal_arm_lo=0.5,
        causal_arm_hi=0.8,
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


def test_dual_ema_fold_and_zero() -> None:
    sampler = _sampler(adaptive_alpha=0.5)
    sampler.update_current_failure_count(torch.full((10,), 834, dtype=torch.long))
    check("frame accumulator holds raw deaths pre-fold", float(sampler.current_frame_failed_count.sum()) == 10.0)
    check("bin accumulator holds raw deaths pre-fold", float(sampler.current_bin_failed_count.sum()) == 10.0)
    sampler.update_failure_ema()
    # alpha * count: 0.5 * 10 = 5 at frame 834 and in its bin.
    check("frame EMA = alpha*count (first fold)", abs(float(sampler.frame_failed_count[834]) - 5.0) < 1e-6)
    bin834 = sampler.frames_to_bins(torch.tensor([834]))[0].item()
    check("bin EMA = alpha*count (first fold)", abs(float(sampler.bin_failed_count[bin834]) - 5.0) < 1e-6)
    check("frame accumulator zeroed after fold", float(sampler.current_frame_failed_count.sum()) == 0.0)
    check("bin accumulator zeroed after fold", float(sampler.current_bin_failed_count.sum()) == 0.0)
    sampler.update_failure_ema()
    check("frame EMA decays with no new deaths", abs(float(sampler.frame_failed_count[834]) - 2.5) < 1e-6)


def test_empty_failures_is_noop() -> None:
    sampler = _sampler()
    sampler.update_current_failure_count(torch.empty(0, dtype=torch.long))
    check("empty death tensor adds nothing (frame)", float(sampler.current_frame_failed_count.sum()) == 0.0)
    check("empty death tensor adds nothing (bin)", float(sampler.current_bin_failed_count.sum()) == 0.0)


def test_official_bin_maps_death_to_bin17() -> None:
    # Official bin map: death 834 -> bin floor(834*20/959) = 17 (range [815,862]).
    sampler = _sampler()
    b = int(sampler.frames_to_bins(torch.tensor([834]))[0].item())
    check("death 834 maps to bin 17", b == 17)
    lo, hi = sampler.bin_frame_bounds(torch.tensor([17]))
    check("bin 17 lower bound is 815", int(lo[0]) == 815)
    check("bin 17 upper bound (exclusive) is 863", int(hi[0]) == 863)


def test_causal_max_ratio_zero_is_pure_official() -> None:
    # causal_max_ratio == 0 => byte-for-byte official: no causal mass, beta == 0, all starts come
    # from the official failure-bin sampler (death 834 -> starts land inside its bin).
    torch.manual_seed(0)
    sampler = _sampler(causal_max_ratio=0.0)
    sampler.configure_credit(CAUSAL_H, CAUSAL_DECAY)
    sampler.update_current_failure_count(torch.full((4096,), 834, dtype=torch.long))
    sampler.update_failure_ema()
    check("causal weight is 0 when disabled", sampler.causal_weight() == 0.0)
    frames = sampler.sample_frames(60000, min_phase=0, max_phase=MOTION_FRAMES - 1)
    # Official sampler draws from the failure bin (and the additive uniform floor across all bins);
    # the dominant mode sits inside the death bin [815,862], i.e. on/after the death frame.
    in_death_bin = ((frames >= 815) & (frames <= 862)).float().mean().item()
    check("official sampler concentrates in the death bin", in_death_bin > 0.5)


def test_impulse_causal_support_strictly_before_death() -> None:
    # THE critical guarantee: with a single death at frame 834 and H=24, the causal predecessor
    # score must be > 0 strictly on [834-(H-1), 834-1] = [811, 833] and EXACTLY 0 on [834, ...].
    sampler = _sampler()
    sampler.configure_credit(CAUSAL_H, CAUSAL_DECAY)
    sampler.update_current_failure_count(torch.tensor([834], dtype=torch.long))
    sampler.update_failure_ema()
    scores = sampler._causal_frame_scores()
    nonzero = torch.nonzero(scores > 0).flatten()
    check("lowest causal start is 811 (= f-(H-1))", int(nonzero.min()) == 834 - (CAUSAL_H - 1))
    check("highest causal start is 833 (= f-1)", int(nonzero.max()) == 834 - 1)
    check("zero causal mass AT the death frame 834", float(scores[834]) == 0.0)
    check("zero causal mass AFTER the death frame", float(scores[835:].sum()) == 0.0)
    # Decay: the start nearest the death (833, k=1) carries the most mass; 811 (k=23) the least.
    check("nearest predecessor is most weighted", float(scores[833]) > float(scores[811]) > 0.0)


def test_impulse_causal_sampling_window() -> None:
    # Drawn causal starts (when beta forces the second stage on) also obey [811,833].
    torch.manual_seed(0)
    sampler = _sampler(causal_max_ratio=1.0, causal_arm_lo=0.0, causal_arm_hi=1e-9)
    sampler.configure_credit(CAUSAL_H, CAUSAL_DECAY)
    sampler.update_current_failure_count(torch.full((4096,), 834, dtype=torch.long))
    sampler.update_failure_ema()
    causal = sampler._sample_causal(40000, min_phase=0, max_phase=MOTION_FRAMES - 1)
    check("causal draws are non-empty", causal.numel() > 0)
    check("all causal starts in [811,833]", int(causal.min()) >= 811 and int(causal.max()) <= 833)
    check("no causal start at or after the death frame", int(causal.max()) < 834)


def test_no_causal_without_credit_injection() -> None:
    # Without configure_credit (H stays 0) there is no causal mass even at max ratio.
    sampler = _sampler(causal_max_ratio=0.5)
    sampler.update_current_failure_count(torch.full((4096,), 834, dtype=torch.long))
    sampler.update_failure_ema()
    check("causal scores empty when H==0", float(sampler._causal_frame_scores().sum()) == 0.0)
    check("causal weight 0 when H==0", sampler.causal_weight() == 0.0)


def test_readiness_ramps_zero_to_one() -> None:
    # bottleneck_concentration (readiness) is low when deaths are spread across bins and high when
    # they concentrate in one bin; causal_weight ramps 0 -> causal_max_ratio across [arm_lo,arm_hi].
    spread = _sampler()
    spread.configure_credit(CAUSAL_H, CAUSAL_DECAY)
    # Deaths spread over the whole clip -> low concentration.
    spread.update_current_failure_count(torch.arange(0, MOTION_FRAMES, dtype=torch.long))
    spread.update_failure_ema()
    r_spread = spread.bottleneck_concentration()
    check("spread deaths -> low bottleneck concentration", r_spread < 0.5)
    check("spread deaths -> beta floor (causal weight 0)", spread.causal_weight() == 0.0)

    peaked = _sampler()
    peaked.configure_credit(CAUSAL_H, CAUSAL_DECAY)
    peaked.update_current_failure_count(torch.full((4096,), 834, dtype=torch.long))
    peaked.update_failure_ema()
    r_peak = peaked.bottleneck_concentration()
    check("concentrated deaths -> high bottleneck concentration", r_peak > 0.8)
    check("concentrated deaths -> beta hits max (0.5*causal_max)", abs(peaked.causal_weight() - 0.5) < 1e-6)
    check("readiness increased from spread to peaked", r_peak > r_spread)


def test_official_conditional_no_boundary_spikes() -> None:
    # Deaths far below the range must not leak in nor pile on a boundary (conditional, not clamp).
    torch.manual_seed(0)
    sampler = _sampler(causal_max_ratio=0.0)
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
    sampler.configure_credit(CAUSAL_H, CAUSAL_DECAY)
    sampler.update_current_failure_count(torch.full((4096,), 834, dtype=torch.long))
    sampler.update_failure_ema()
    stats = sampler.stats(0, MOTION_FRAMES - 1)
    for key in ("top_bin", "top_prob", "failed_sum", "entropy", "peak_bin",
                "peak_fail_frame", "bottleneck_concentration", "causal_beta"):
        check(f"stats has {key}", key in stats)
    check("peak failure frame is 834", int(stats["peak_fail_frame"]) == 834)
    check("peak bin is 17", int(stats["peak_bin"]) == 17)
    check("causal_beta hits the cap (0.5)", abs(stats["causal_beta"] - 0.5) < 1e-6)


def test_state_dict_roundtrip_and_version_guard() -> None:
    sampler = _sampler()
    sampler.update_current_failure_count(torch.full((4096,), 834, dtype=torch.long))
    sampler.update_failure_ema()
    state = sampler.state_dict()
    check("state carries the sampler version", state["version"] == ADAPTIVE_SAMPLER_VERSION)
    check("version is 4 (dual-EMA design)", ADAPTIVE_SAMPLER_VERSION == 4)

    restored = _sampler()
    check("matching version restores", restored.load_state_dict(state) is True)
    check("restored bin EMA matches", torch.allclose(restored.bin_failed_count, sampler.bin_failed_count))
    check("restored frame EMA matches", torch.allclose(restored.frame_failed_count, sampler.frame_failed_count))

    stale = _sampler()
    bad_state = dict(state)
    bad_state["version"] = ADAPTIVE_SAMPLER_VERSION - 1
    check("version mismatch is rejected", stale.load_state_dict(bad_state) is False)
    check("rejected state keeps fresh bin zeros", float(stale.bin_failed_count.sum()) == 0.0)
    check("rejected state keeps fresh frame zeros", float(stale.frame_failed_count.sum()) == 0.0)
    check("None state is rejected", _sampler().load_state_dict(None) is False)


if __name__ == "__main__":
    for fn in (
        test_auto_num_bins,
        test_dual_ema_fold_and_zero,
        test_empty_failures_is_noop,
        test_official_bin_maps_death_to_bin17,
        test_causal_max_ratio_zero_is_pure_official,
        test_impulse_causal_support_strictly_before_death,
        test_impulse_causal_sampling_window,
        test_no_causal_without_credit_injection,
        test_readiness_ramps_zero_to_one,
        test_official_conditional_no_boundary_spikes,
        test_zero_history_is_uniform_over_bins,
        test_stats_fields,
        test_state_dict_roundtrip_and_version_guard,
    ):
        print(f"== {fn.__name__} ==")
        fn()
    print("ALL ADAPTIVE SAMPLER TESTS PASSED")
