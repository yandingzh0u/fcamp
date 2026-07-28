from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


CHANNELS = ("task", "amp")


@dataclass(frozen=True)
class DualChannelCredit:
    """Outputs of primitive-step, two-channel causal GAE.

    All channel tensors use shape ``[time, env, 2]`` in ``[task, amp]`` order.
    ``mixed_advantage`` is the raw weighted task/style mixture.  The mixture is
    normalized exactly once for the actor; both components share that same
    normalization factor, so reward weights and physical scales cannot be
    erased by independent channel standardization. ``channel_valid_mask`` is
    the effective ``[time, env, 2]`` critic/credit mask.
    """

    advantages: torch.Tensor
    value_targets: torch.Tensor
    mixed_advantage: torch.Tensor
    actor_advantage: torch.Tensor
    actor_advantage_components: torch.Tensor
    channel_valid_mask: torch.Tensor

def resolve_terminal_masks(
    done: torch.Tensor,
    timeout: torch.Tensor,
    motion_complete: torch.Tensor,
    failure: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve overlapping terminal signals with failure-first precedence."""

    if not (done.shape == timeout.shape == motion_complete.shape == failure.shape):
        raise ValueError("all terminal masks must have the same shape")
    done_b = done.bool()
    failure_b = done_b & failure.bool()
    motion_complete_b = done_b & motion_complete.bool() & ~failure_b
    timeout_b = done_b & timeout.bool() & ~failure_b & ~motion_complete_b
    return failure_b, timeout_b, motion_complete_b


def _validate_inputs(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    bootstrap_mask: torch.Tensor,
    trace_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    channel_valid_mask: torch.Tensor,
    channel_bootstrap_mask: torch.Tensor,
    channel_trace_mask: torch.Tensor,
) -> None:
    if rewards.ndim != 3 or rewards.shape[-1] != len(CHANNELS):
        raise ValueError("rewards must have shape [time, env, 2] in [task, amp] order")
    if values.shape != rewards.shape or next_values.shape != rewards.shape:
        raise ValueError("values and next_values must have the same shape as rewards")
    expected_mask_shape = rewards.shape[:2]
    for name, mask in (
        ("bootstrap_mask", bootstrap_mask),
        ("trace_mask", trace_mask),
        ("valid_mask", valid_mask),
    ):
        if mask.shape != expected_mask_shape:
            raise ValueError(
                f"{name} must have shape {expected_mask_shape}, got {tuple(mask.shape)}"
            )
        if mask.device != rewards.device:
            raise ValueError(f"{name} and rewards must be on the same device")
    if values.device != rewards.device or next_values.device != rewards.device:
        raise ValueError("rewards, values and next_values must be on the same device")
    for name, mask in (
        ("channel_valid_mask", channel_valid_mask),
        ("channel_bootstrap_mask", channel_bootstrap_mask),
        ("channel_trace_mask", channel_trace_mask),
    ):
        if mask.shape != rewards.shape:
            raise ValueError(
                f"{name} must have shape {tuple(rewards.shape)}, got {tuple(mask.shape)}"
            )
        if mask.device != rewards.device:
            raise ValueError(f"{name} and rewards must be on the same device")


def _mix_advantages(
    advantages: torch.Tensor,
    valid_mask: torch.Tensor,
    channel_valid_mask: torch.Tensor,
    actor_weights: Sequence[float] | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = torch.as_tensor(
        actor_weights, device=advantages.device, dtype=advantages.dtype
    )
    if weights.numel() != len(CHANNELS):
        raise ValueError(
            f"actor_weights must contain {len(CHANNELS)} values in [task, amp] order"
        )
    valid_bool = valid_mask.unsqueeze(-1).bool()
    channel_valid_bool = channel_valid_mask.bool() & valid_bool
    safe_advantages = torch.where(
        channel_valid_bool, advantages, torch.zeros_like(advantages)
    )
    raw_weighted = safe_advantages * weights.reshape(1, 1, -1)
    mixed = raw_weighted.sum(dim=-1)
    return mixed, mixed, raw_weighted


def normalize_actor_mixture(
    credit: DualChannelCredit,
    valid_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    epsilon: float = 1e-8,
) -> DualChannelCredit:
    """Normalize one scalar actor objective under its exact sampling measure.

    The same scalar mean and standard deviation are used for both channel
    components, while ``sample_weights`` describes the objective measure (for
    FCAMP, the fixed phase0/curriculum mixture).
    """

    if valid_mask.shape != credit.mixed_advantage.shape:
        raise ValueError("valid_mask must match mixed actor advantage")
    if sample_weights.shape != valid_mask.shape:
        raise ValueError("sample_weights must match valid_mask")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if bool((sample_weights < 0).any()) or not bool(
        torch.isfinite(sample_weights).all()
    ):
        raise ValueError("sample_weights must be finite and non-negative")

    valid = valid_mask.bool()
    weights = torch.where(
        valid,
        sample_weights.to(
            device=credit.mixed_advantage.device,
            dtype=credit.mixed_advantage.dtype,
        ),
        torch.zeros_like(credit.mixed_advantage),
    )
    weight_sum = weights.sum()
    if not bool(weight_sum > 0):
        raise ValueError("sample_weights contain no valid actor mass")

    raw_components = credit.actor_advantage_components
    if credit.channel_valid_mask.shape != raw_components.shape:
        raise ValueError("credit.channel_valid_mask must match channel advantages")
    channel_valid = credit.channel_valid_mask.bool() & valid.unsqueeze(-1)
    channel_weights = weights.unsqueeze(-1) * channel_valid.to(weights.dtype)
    channel_weight_sum = channel_weights.sum(dim=(0, 1))
    component_mean = torch.where(
        channel_weight_sum > 0,
        (channel_weights * raw_components).sum(dim=(0, 1))
        / channel_weight_sum.clamp_min(float(epsilon)),
        torch.zeros_like(channel_weight_sum),
    )
    centered_components = torch.where(
        channel_valid,
        raw_components - component_mean.reshape(1, 1, -1),
        torch.zeros_like(raw_components),
    )
    centered_mixed = centered_components.sum(dim=-1)
    variance = (weights * centered_mixed.square()).sum() / weight_sum
    inv_std = torch.rsqrt(variance + float(epsilon))
    actor_components = torch.where(
        valid.unsqueeze(-1),
        centered_components * inv_std,
        torch.zeros_like(centered_components),
    )
    actor_advantage = actor_components.sum(dim=-1)

    return DualChannelCredit(
        advantages=credit.advantages,
        value_targets=credit.value_targets,
        mixed_advantage=credit.mixed_advantage,
        actor_advantage=actor_advantage,
        actor_advantage_components=actor_components,
        channel_valid_mask=credit.channel_valid_mask,
    )


def compute_dual_channel_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    bootstrap_mask: torch.Tensor,
    trace_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
    actor_weights: Sequence[float] | torch.Tensor = (1.0, 1.0),
    channel_valid_mask: torch.Tensor,
    channel_bootstrap_mask: torch.Tensor,
    channel_trace_mask: torch.Tensor,
) -> DualChannelCredit:
    """Compute primitive-step causal GAE for task and style rewards.

    The leading dimension is chronological primitive time, not chunk time, so
    the backward recursion naturally crosses chunk boundaries. Callers encode
    terminal semantics separately:

    * failure: ``bootstrap_mask=0, trace_mask=0``;
    * timeout: ``bootstrap_mask=1, trace_mask=0``;
    * ordinary transition (including a chunk boundary): both masks are one.

    Channel masks refine those shared masks without weakening them. This lets
    an external intervention cut only AMP bootstrap/trace and lets causally
    contaminated imitation windows be invalid for the AMP critic and actor
    component while the task channel remains fully trainable.

    A reward at time ``t`` can only affect advantages at ``t`` and earlier;
    it never leaks into a later conditional action prefix.
    """

    _validate_inputs(
        rewards,
        values,
        next_values,
        bootstrap_mask,
        trace_mask,
        valid_mask,
        channel_valid_mask,
        channel_bootstrap_mask,
        channel_trace_mask,
    )
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}")

    dtype = rewards.dtype
    base_valid_bool = valid_mask.bool().unsqueeze(-1)
    effective_channel_valid = (
        base_valid_bool.expand_as(rewards) & channel_valid_mask.bool()
    )
    bootstrap = bootstrap_mask.to(dtype).unsqueeze(-1).expand_as(rewards)
    bootstrap = bootstrap * channel_bootstrap_mask.to(dtype)
    trace = trace_mask.to(dtype).unsqueeze(-1).expand_as(rewards)
    trace = trace * channel_trace_mask.to(dtype)
    valid = effective_channel_valid.to(dtype)
    td_errors = (
        rewards + float(gamma) * bootstrap * next_values - values
    ) * valid

    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    trace_coefficient = float(gamma) * float(gae_lambda)
    for time_index in range(rewards.shape[0] - 1, -1, -1):
        current = td_errors[time_index] + (
            trace_coefficient * trace[time_index] * running
        )
        running = current * valid[time_index]
        advantages[time_index] = running

    # Invalid padded frames must never become accidental critic targets if a
    # caller forgets to apply its minibatch mask a second time.
    value_targets = (values + advantages) * valid
    mixed_advantage, actor_advantage, actor_components = _mix_advantages(
        advantages,
        valid_mask.to(torch.bool),
        effective_channel_valid,
        actor_weights,
    )

    return DualChannelCredit(
        advantages=advantages,
        value_targets=value_targets,
        mixed_advantage=mixed_advantage,
        actor_advantage=actor_advantage,
        actor_advantage_components=actor_components,
        channel_valid_mask=effective_channel_valid,
    )
