from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TemporalCredit:
    """Primitive-step AMP credit on one scalar reward channel."""

    advantages: torch.Tensor
    value_targets: torch.Tensor
    actor_advantage: torch.Tensor
    valid_mask: torch.Tensor


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


def _validate_gae_inputs(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    bootstrap_mask: torch.Tensor,
    trace_mask: torch.Tensor,
    valid_mask: torch.Tensor,
) -> None:
    if rewards.ndim != 2:
        raise ValueError("rewards must have shape [time, env]")
    if values.shape != rewards.shape or next_values.shape != rewards.shape:
        raise ValueError("values and next_values must have the same shape as rewards")
    for name, tensor in (
        ("bootstrap_mask", bootstrap_mask),
        ("trace_mask", trace_mask),
        ("valid_mask", valid_mask),
    ):
        if tensor.shape != rewards.shape:
            raise ValueError(
                f"{name} must have shape {tuple(rewards.shape)}, "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.device != rewards.device:
            raise ValueError(f"{name} and rewards must be on the same device")
    if values.device != rewards.device or next_values.device != rewards.device:
        raise ValueError("rewards, values and next_values must be on the same device")


def compute_amp_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    bootstrap_mask: torch.Tensor,
    trace_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> TemporalCredit:
    """Compute causal AMP GAE over chronological primitive steps.

    ``bootstrap_mask`` controls the value after each transition, while
    ``trace_mask`` controls whether later TD errors flow across that transition.
    ``valid_mask`` describes real trainable actions, not whether a
    discriminator window happened to be available at that endpoint. Callers
    gate invalid discriminator endpoints by supplying zero reward, and cut
    temporal credit only at genuine terminal or intervention edges.
    """

    _validate_gae_inputs(
        rewards,
        values,
        next_values,
        bootstrap_mask,
        trace_mask,
        valid_mask,
    )
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}")

    valid_bool = valid_mask.bool()
    valid = valid_bool.to(dtype=rewards.dtype)
    bootstrap = bootstrap_mask.to(dtype=rewards.dtype)
    trace = trace_mask.to(dtype=rewards.dtype)
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

    value_targets = (values + advantages) * valid
    return TemporalCredit(
        advantages=advantages,
        value_targets=value_targets,
        actor_advantage=advantages,
        valid_mask=valid_bool,
    )


def normalize_amp_advantage(
    credit: TemporalCredit,
    valid_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> TemporalCredit:
    """Normalize AMP advantage once under the actor's sampling measure."""

    if valid_mask.shape != credit.advantages.shape:
        raise ValueError("valid_mask must match AMP advantages")
    if sample_weights.shape != valid_mask.shape:
        raise ValueError("sample_weights must match valid_mask")
    if credit.valid_mask.shape != valid_mask.shape:
        raise ValueError("credit.valid_mask must match valid_mask")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if bool((sample_weights < 0).any()) or not bool(
        torch.isfinite(sample_weights).all()
    ):
        raise ValueError("sample_weights must be finite and non-negative")

    valid = valid_mask.bool() & credit.valid_mask.bool()
    weights = torch.where(
        valid,
        sample_weights.to(
            device=credit.advantages.device,
            dtype=credit.advantages.dtype,
        ),
        torch.zeros_like(credit.advantages),
    )
    weight_sum = weights.sum()
    if not bool(weight_sum > 0):
        raise ValueError("sample_weights contain no valid actor mass")

    mean = (weights * credit.advantages).sum() / weight_sum
    centered = torch.where(
        valid,
        credit.advantages - mean,
        torch.zeros_like(credit.advantages),
    )
    variance = (weights * centered.square()).sum() / weight_sum
    actor_advantage = centered * torch.rsqrt(variance + float(epsilon))
    return TemporalCredit(
        advantages=credit.advantages,
        value_targets=credit.value_targets,
        actor_advantage=actor_advantage,
        valid_mask=credit.valid_mask,
    )
