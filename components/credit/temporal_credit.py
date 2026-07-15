from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch


CHANNELS = ("task", "amp")
NormalizationMode = Literal[
    "per_offset",
    "global",
    "per_channel_per_offset",
    "per_channel_global",
    "none",
]


@dataclass(frozen=True)
class DualChannelCredit:
    """Outputs of primitive-step, two-channel causal GAE.

    All channel tensors use shape ``[time, env, 2]`` in ``[task, amp]`` order.
    ``mixed_advantage`` is the raw weighted task/style mixture. It is normalized
    exactly once into ``actor_advantage`` so style scale/weight changes really
    change actor credit.
    """

    td_errors: torch.Tensor
    advantages: torch.Tensor
    value_targets: torch.Tensor
    mixed_advantage: torch.Tensor
    actor_advantage: torch.Tensor
    actor_advantage_components: torch.Tensor

    @property
    def normalized_advantages(self) -> torch.Tensor:
        return self.actor_advantage_components


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


def _mix_then_normalize_masked(
    advantages: torch.Tensor,
    valid_mask: torch.Tensor,
    actor_weights: Sequence[float] | torch.Tensor,
    mode: NormalizationMode,
    chunk_horizon: int,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    weights = torch.as_tensor(
        actor_weights, device=advantages.device, dtype=advantages.dtype
    )
    if weights.numel() != len(CHANNELS):
        raise ValueError(
            f"actor_weights must contain {len(CHANNELS)} values in [task, amp] order"
        )
    if mode not in (
        "global",
        "per_offset",
        "per_channel_per_offset",
        "per_channel_global",
        "none",
    ):
        raise ValueError(
            "normalization must be one of {'per_offset', 'global', "
            "'per_channel_per_offset', 'per_channel_global', 'none'}, "
            f"got {mode!r}",
        )
    if chunk_horizon < 1:
        raise ValueError(f"chunk_horizon must be >= 1, got {chunk_horizon}")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    valid_bool = valid_mask.unsqueeze(-1).bool()
    safe_advantages = torch.where(valid_bool, advantages, torch.zeros_like(advantages))
    weighted = safe_advantages * weights.reshape(1, 1, -1)
    mixed = weighted.sum(dim=-1)
    if mode == "none":
        return mixed, mixed, weighted

    normalized = torch.zeros_like(mixed)
    components = torch.zeros_like(weighted)
    time_steps = advantages.shape[0]
    per_channel = mode in {"per_channel_per_offset", "per_channel_global"}
    by_offset = mode in {"per_offset", "per_channel_per_offset"}
    groups = range(chunk_horizon) if by_offset else (None,)
    time_indices = torch.arange(time_steps, device=advantages.device)

    for offset in groups:
        if offset is None:
            group_mask = valid_mask
        else:
            group_mask = valid_mask & ((time_indices % chunk_horizon) == offset).unsqueeze(1)
        if not bool(group_mask.any()):
            continue
        selected = safe_advantages[group_mask]
        channel_mean = selected.mean(dim=0)
        centered_components = (selected - channel_mean) * weights.reshape(1, -1)
        if per_channel:
            inv_std = torch.rsqrt(
                centered_components.square().mean(dim=0, keepdim=True) + epsilon
            )
            normalized_components = centered_components * inv_std
            components[group_mask] = normalized_components
            normalized[group_mask] = normalized_components.sum(dim=-1)
        else:
            centered_mixed = centered_components.sum(dim=-1)
            inv_std = torch.rsqrt(centered_mixed.square().mean() + epsilon)
            components[group_mask] = centered_components * inv_std
            normalized[group_mask] = centered_mixed * inv_std
    return mixed, normalized, components


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
    chunk_horizon: int,
    normalization: NormalizationMode = "per_offset",
    actor_weights: Sequence[float] | torch.Tensor = (1.0, 1.0),
    normalization_epsilon: float = 1e-8,
) -> DualChannelCredit:
    """Compute primitive-step causal GAE for task and style rewards.

    The leading dimension is chronological primitive time, not chunk time, so
    the backward recursion naturally crosses chunk boundaries. Callers encode
    terminal semantics separately:

    * failure: ``bootstrap_mask=0, trace_mask=0``;
    * timeout: ``bootstrap_mask=1, trace_mask=0``;
    * ordinary transition (including a chunk boundary): both masks are one.

    A reward at time ``t`` can only affect advantages at ``t`` and earlier;
    it never leaks into a later conditional action prefix.
    """

    _validate_inputs(
        rewards, values, next_values, bootstrap_mask, trace_mask, valid_mask
    )
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not 0.0 <= gae_lambda <= 1.0:
        raise ValueError(f"gae_lambda must be in [0, 1], got {gae_lambda}")

    dtype = rewards.dtype
    bootstrap = bootstrap_mask.to(dtype).unsqueeze(-1)
    trace = trace_mask.to(dtype).unsqueeze(-1)
    valid = valid_mask.to(dtype).unsqueeze(-1)
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
    mixed_advantage, actor_advantage, actor_components = _mix_then_normalize_masked(
        advantages,
        valid_mask.to(torch.bool),
        actor_weights,
        normalization,
        int(chunk_horizon),
        float(normalization_epsilon),
    )

    return DualChannelCredit(
        td_errors=td_errors,
        advantages=advantages,
        value_targets=value_targets,
        mixed_advantage=mixed_advantage,
        actor_advantage=actor_advantage,
        actor_advantage_components=actor_components,
    )


def with_chunk_shared_actor_credit(
    credit: DualChannelCredit,
    valid_mask: torch.Tensor,
    *,
    chunk_horizon: int,
    normalization: NormalizationMode = "per_offset",
    actor_weights: Sequence[float] | torch.Tensor = (1.0, 1.0),
    normalization_epsilon: float = 1e-8,
) -> DualChannelCredit:
    """Replace causal per-frame actor credit with one advantage per chunk.

    This is the explicit ``w/ chunk advantage`` ablation.  The primitive dual
    GAE and value targets remain unchanged for the two Flow critics.  For the
    actor, each valid conditional in chunk ``k`` receives the normalized
    start-of-chunk advantage ``A[k, 0]``.  Since primitive GAE recursively
    expands across all ``H`` steps, that start advantage is exactly the
    discounted chunk reward plus the ``(gamma * lambda) ** H`` continuation.
    """

    if chunk_horizon < 1:
        raise ValueError(f"chunk_horizon must be >= 1, got {chunk_horizon}")
    if credit.advantages.ndim != 3 or credit.advantages.shape[-1] != len(CHANNELS):
        raise ValueError("credit advantages must have shape [time, env, 2]")
    if valid_mask.shape != credit.advantages.shape[:2]:
        raise ValueError("valid_mask must match credit time/env dimensions")
    time_steps, num_envs, _ = credit.advantages.shape
    if time_steps % chunk_horizon != 0:
        raise ValueError(
            f"time dimension {time_steps} is not divisible by chunk_horizon {chunk_horizon}"
        )

    num_chunks = time_steps // chunk_horizon
    chunk_advantages = credit.advantages.view(
        num_chunks, chunk_horizon, num_envs, len(CHANNELS)
    )[:, 0]
    chunk_valid = valid_mask.view(num_chunks, chunk_horizon, num_envs)[:, 0]
    chunk_norm_mode: NormalizationMode = (
        "none" if normalization == "none" else "global"
    )
    mixed_chunks, normalized_chunks, component_chunks = _mix_then_normalize_masked(
        chunk_advantages,
        chunk_valid,
        actor_weights,
        chunk_norm_mode,
        chunk_horizon=1,
        epsilon=float(normalization_epsilon),
    )
    shared_mixed = mixed_chunks[:, None].expand(-1, chunk_horizon, -1).reshape(
        time_steps, num_envs
    )
    actor_advantage = normalized_chunks[:, None].expand(
        -1, chunk_horizon, -1
    ).reshape(time_steps, num_envs)
    shared_components = component_chunks[:, None].expand(
        -1, chunk_horizon, -1, -1
    ).reshape(time_steps, num_envs, len(CHANNELS))
    shared_mixed = torch.where(valid_mask, shared_mixed, torch.zeros_like(shared_mixed))
    actor_advantage = torch.where(
        valid_mask, actor_advantage, torch.zeros_like(actor_advantage)
    )
    shared_components = torch.where(
        valid_mask.unsqueeze(-1),
        shared_components,
        torch.zeros_like(shared_components),
    )

    return DualChannelCredit(
        td_errors=credit.td_errors,
        advantages=credit.advantages,
        value_targets=credit.value_targets,
        mixed_advantage=shared_mixed,
        actor_advantage=actor_advantage,
        actor_advantage_components=shared_components,
    )
