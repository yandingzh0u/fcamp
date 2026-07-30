from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class AMPChunkCredit:
    """Variable-duration GAE results on the decision grid."""

    advantages: torch.Tensor
    value_targets: torch.Tensor


def _validate_inputs(
    discounted_rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    durations: torch.Tensor,
    bootstrap_mask: torch.Tensor,
    trace_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    if discounted_rewards.ndim != 2 or 0 in discounted_rewards.shape:
        raise ValueError(
            "discounted_rewards must have non-empty shape [decision, env]"
        )
    if not discounted_rewards.is_floating_point():
        raise TypeError("discounted_rewards must be floating point")

    expected = discounted_rewards.shape
    device = discounted_rewards.device
    for name, tensor in (
        ("values", values),
        ("next_values", next_values),
        ("durations", durations),
        ("bootstrap_mask", bootstrap_mask),
        ("trace_mask", trace_mask),
        ("valid_mask", valid_mask),
    ):
        if tensor.shape != expected:
            raise ValueError(
                f"{name} must have shape {tuple(expected)}, got {tuple(tensor.shape)}"
            )
        if tensor.device != device:
            raise ValueError(f"{name} and discounted_rewards must share a device")

    for name, tensor in (("values", values), ("next_values", next_values)):
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        if tensor.dtype != discounted_rewards.dtype:
            raise TypeError(
                f"{name} must have dtype {discounted_rewards.dtype}, got {tensor.dtype}"
            )

    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be finite and in [0, 1], got {gamma}")
    if not math.isfinite(gae_lambda) or not 0.0 <= gae_lambda <= 1.0:
        raise ValueError(
            f"gae_lambda must be finite and in [0, 1], got {gae_lambda}"
        )

    valid = valid_mask.bool()
    for name, mask in (
        ("bootstrap_mask", bootstrap_mask),
        ("trace_mask", trace_mask),
        ("valid_mask", valid_mask),
    ):
        if mask.dtype != torch.bool:
            finite = torch.isfinite(mask)
            binary = (mask == 0) | (mask == 1)
            if not bool((finite & binary).all()):
                raise ValueError(f"{name} must contain only 0/1 or boolean values")

    if durations.dtype == torch.bool:
        raise TypeError("durations must contain integer decision lengths")
    valid_durations = durations[valid]
    if valid_durations.is_floating_point():
        if not bool(torch.isfinite(valid_durations).all()):
            raise ValueError("valid durations must be finite")
        if not bool((valid_durations == valid_durations.round()).all()):
            raise ValueError("valid durations must be integer-valued")
    if not bool((valid_durations >= 1).all()):
        raise ValueError("valid durations must be at least one primitive step")

    for name, tensor in (
        ("discounted_rewards", discounted_rewards),
        ("values", values),
        ("next_values", next_values),
    ):
        if not bool(torch.isfinite(tensor[valid]).all()):
            raise ValueError(f"{name} must be finite at valid decisions")
    return valid


def compute_amp_chunk_gae(
    discounted_rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    durations: torch.Tensor,
    bootstrap_mask: torch.Tensor,
    trace_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> AMPChunkCredit:
    """Compute GAE for variable-duration AMP action chunks.

    Inputs use a rectangular ``[decision, env]`` layout.  A reward at decision
    ``d`` is already discounted within its executed prefix:

    ``R_d = sum_{i=0}^{k_d-1} gamma**i * r_{d,i}``.

    ``bootstrap_mask`` controls the value term in the local TD error.
    ``trace_mask`` independently controls whether the following decision's
    advantage crosses the boundary.  Consequently a timeout can bootstrap
    while still cutting the GAE trace.  Invalid padding is always returned as
    zero and cannot carry a trace.
    """

    valid = _validate_inputs(
        discounted_rewards,
        values,
        next_values,
        durations,
        bootstrap_mask,
        trace_mask,
        valid_mask,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    dtype = discounted_rewards.dtype
    device = discounted_rewards.device

    safe_duration = torch.where(
        valid,
        durations,
        torch.ones_like(durations),
    ).to(dtype=dtype)
    zeros = torch.zeros_like(discounted_rewards)
    rewards = torch.where(valid, discounted_rewards, zeros)
    current_values = torch.where(valid, values, zeros)
    following_values = torch.where(valid, next_values, zeros)
    bootstrap = bootstrap_mask.to(dtype=dtype) * valid.to(dtype=dtype)
    trace = trace_mask.to(dtype=dtype) * valid.to(dtype=dtype)

    gamma_base = torch.tensor(float(gamma), dtype=dtype, device=device)
    trace_base = torch.tensor(
        float(gamma) * float(gae_lambda),
        dtype=dtype,
        device=device,
    )
    bootstrap_discount = gamma_base.pow(safe_duration)
    trace_discount = trace_base.pow(safe_duration)
    deltas = rewards + bootstrap_discount * bootstrap * following_values
    deltas = (deltas - current_values) * valid.to(dtype=dtype)

    advantages = torch.zeros_like(discounted_rewards)
    running = torch.zeros(
        discounted_rewards.shape[1],
        dtype=dtype,
        device=device,
    )
    for decision in range(discounted_rewards.shape[0] - 1, -1, -1):
        current = deltas[decision] + (
            trace_discount[decision] * trace[decision] * running
        )
        running = torch.where(valid[decision], current, torch.zeros_like(current))
        advantages[decision] = running

    value_targets = torch.where(valid, values + advantages, zeros)
    return AMPChunkCredit(
        advantages=advantages,
        value_targets=value_targets,
    )


def normalize_and_clip_amp_advantages(
    advantages: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    clip: float = 4.0,
    epsilon: float = 1.0e-5,
) -> torch.Tensor:
    """Globally normalize valid advantages, clip them, and zero padding."""

    if advantages.ndim != 2 or advantages.shape != valid_mask.shape:
        raise ValueError(
            "advantages and valid_mask must share shape [decision, env]"
        )
    if not advantages.is_floating_point():
        raise TypeError("advantages must be floating point")
    if valid_mask.device != advantages.device:
        raise ValueError("valid_mask and advantages must share a device")
    if valid_mask.dtype != torch.bool:
        finite = torch.isfinite(valid_mask)
        binary = (valid_mask == 0) | (valid_mask == 1)
        if not bool((finite & binary).all()):
            raise ValueError("valid_mask must contain only 0/1 or boolean values")
    if not math.isfinite(clip) or clip <= 0.0:
        raise ValueError(f"clip must be finite and positive, got {clip}")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be finite and positive, got {epsilon}")

    valid = valid_mask.bool()
    if not bool(valid.any()):
        raise ValueError("valid_mask contains no valid decisions")
    selected = advantages[valid]
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("valid advantages must be finite")

    if selected.numel() == 1:
        # Unbiased variance is undefined for a singleton. With no second
        # decision there is no relative policy preference to learn, so this
        # degenerate batch is a safe actor no-op. Normal training uses the
        # official unbiased normalization below over thousands of decisions.
        normalized = torch.zeros_like(advantages)
    else:
        std, mean = torch.std_mean(selected, unbiased=True)
        normalized = (advantages - mean) / std.clamp_min(float(epsilon))
    normalized = normalized.clamp(min=-float(clip), max=float(clip))
    return torch.where(valid, normalized, torch.zeros_like(normalized))
