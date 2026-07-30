from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TaskCredit:
    """Scalar primitive-step GAE and its actor-normalized view."""

    advantages: torch.Tensor
    value_targets: torch.Tensor
    actor_advantage: torch.Tensor


def resolve_terminal_masks(
    done: torch.Tensor,
    timeout: torch.Tensor,
    motion_complete: torch.Tensor,
    failure: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resolve overlapping terminal causes with failure-first precedence."""

    if not (done.shape == timeout.shape == motion_complete.shape == failure.shape):
        raise ValueError("all terminal masks must have the same shape")
    done_b = done.bool()
    failure_b = done_b & failure.bool()
    complete_b = done_b & motion_complete.bool() & ~failure_b
    timeout_b = done_b & timeout.bool() & ~failure_b & ~complete_b
    return failure_b, timeout_b, complete_b


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
    for name, value in (
        ("values", values),
        ("next_values", next_values),
        ("bootstrap_mask", bootstrap_mask),
        ("trace_mask", trace_mask),
        ("valid_mask", valid_mask),
    ):
        if value.shape != rewards.shape:
            raise ValueError(
                f"{name} must have shape {tuple(rewards.shape)}, "
                f"got {tuple(value.shape)}"
            )
        if value.device != rewards.device:
            raise ValueError(f"{name} and rewards must be on the same device")
    for name, value in (
        ("rewards", rewards),
        ("values", values),
        ("next_values", next_values),
    ):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name} contains non-finite values")


def compute_task_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    bootstrap_mask: torch.Tensor,
    trace_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> TaskCredit:
    """Compute scalar GAE over chronological primitive transitions.

    A failure or motion-completion transition uses bootstrap=0 and trace=0.
    A timeout uses bootstrap=1 and trace=0. Ordinary transitions, including
    chunk boundaries, use bootstrap=1 and trace=1.
    """

    _validate_gae_inputs(
        rewards,
        values,
        next_values,
        bootstrap_mask,
        trace_mask,
        valid_mask,
    )
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not 0.0 <= float(gae_lambda) <= 1.0:
        raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}")

    dtype = rewards.dtype
    valid = valid_mask.to(dtype=dtype)
    bootstrap = bootstrap_mask.to(dtype=dtype)
    trace = trace_mask.to(dtype=dtype)
    delta = (
        rewards + float(gamma) * bootstrap * next_values - values
    ) * valid

    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    coefficient = float(gamma) * float(gae_lambda)
    for time_index in range(rewards.shape[0] - 1, -1, -1):
        running = (
            delta[time_index]
            + coefficient * trace[time_index] * running
        ) * valid[time_index]
        advantages[time_index] = running

    value_targets = (values + advantages) * valid
    if not bool(torch.isfinite(advantages).all()):
        raise FloatingPointError("advantages contain non-finite values")
    if not bool(torch.isfinite(value_targets).all()):
        raise FloatingPointError("value targets contain non-finite values")
    return TaskCredit(
        advantages=advantages,
        value_targets=value_targets,
        actor_advantage=advantages,
    )


def normalize_actor_advantage(
    credit: TaskCredit,
    valid_mask: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    epsilon: float = 1.0e-8,
) -> TaskCredit:
    """Apply one scalar normalization under the exact actor objective measure."""

    if valid_mask.shape != credit.advantages.shape:
        raise ValueError("valid_mask must match advantages")
    if sample_weights.shape != valid_mask.shape:
        raise ValueError("sample_weights must match valid_mask")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if not bool(torch.isfinite(sample_weights).all()) or bool(
        (sample_weights < 0).any()
    ):
        raise ValueError("sample_weights must be finite and non-negative")

    valid = valid_mask.bool()
    weights = torch.where(
        valid,
        sample_weights.to(
            device=credit.advantages.device,
            dtype=credit.advantages.dtype,
        ),
        torch.zeros_like(credit.advantages),
    )
    mass = weights.sum()
    if not bool(mass > 0):
        raise ValueError("sample_weights contain no valid actor mass")
    mean = (weights * credit.advantages).sum() / mass
    centered = torch.where(
        valid,
        credit.advantages - mean,
        torch.zeros_like(credit.advantages),
    )
    variance = (weights * centered.square()).sum() / mass
    actor_advantage = centered * torch.rsqrt(variance + float(epsilon))
    if not bool(torch.isfinite(actor_advantage).all()):
        raise FloatingPointError("normalized advantages contain non-finite values")
    return TaskCredit(
        advantages=credit.advantages,
        value_targets=credit.value_targets,
        actor_advantage=actor_advantage,
    )
