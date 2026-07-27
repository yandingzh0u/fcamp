from __future__ import annotations

import math

import torch


def normalize_command_rate(
    command_rate: torch.Tensor,
    rate_limit: torch.Tensor,
) -> torch.Tensor:
    """Expose carried rate to policies as a dimensionless support fraction."""

    if command_rate.ndim < 1:
        raise ValueError("command_rate must have at least one dimension")
    action_dim = command_rate.shape[-1]
    if rate_limit.shape != (action_dim,):
        raise ValueError(
            f"rate_limit must have shape {(action_dim,)}, "
            f"got {tuple(rate_limit.shape)}"
        )
    if not bool(torch.isfinite(command_rate).all()):
        raise ValueError("command_rate must be finite")
    if not bool(torch.isfinite(rate_limit).all()) or bool(
        (rate_limit <= 0.0).any()
    ):
        raise ValueError("rate_limit must be finite and strictly positive")
    return command_rate / rate_limit


def command_rate_decay(
    control_dt: float,
    half_life_seconds: float,
) -> float:
    """Return the per-control-frame decay for a physical rate half-life."""

    control_dt = float(control_dt)
    half_life_seconds = float(half_life_seconds)
    if not math.isfinite(control_dt) or control_dt <= 0.0:
        raise ValueError("control_dt must be finite and positive")
    if not math.isfinite(half_life_seconds) or half_life_seconds <= 0.0:
        raise ValueError("half_life_seconds must be finite and positive")
    return 2.0 ** (-control_dt / half_life_seconds)


def decode_raw_target_rate(
    raw_target_rate: torch.Tensor,
    previous_action: torch.Tensor,
    previous_rate: torch.Tensor,
    rate_limit: torch.Tensor,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    *,
    control_dt: float,
    decay: float,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pure one-frame decoder from raw target rate to normalized PD command.

    ``previous_rate`` and ``rate_limit`` are expressed in normalized action per
    second.  This function projects exactly once into the environment-owned
    action domain and never mutates the carried state.
    """

    if (
        raw_target_rate.shape != previous_action.shape
        or previous_rate.shape != previous_action.shape
    ):
        raise ValueError(
            "raw_target_rate, previous_action, and previous_rate must have "
            "identical shapes"
        )
    if previous_action.ndim < 1:
        raise ValueError("decoder tensors must have at least one dimension")
    action_dim = previous_action.shape[-1]
    for name, value in (
        ("rate_limit", rate_limit),
        ("action_low", action_low),
        ("action_high", action_high),
    ):
        if value.shape != (action_dim,):
            raise ValueError(f"{name} must have shape {(action_dim,)}")
    for name, value in (
        ("raw_target_rate", raw_target_rate),
        ("previous_action", previous_action),
        ("previous_rate", previous_rate),
        ("rate_limit", rate_limit),
        ("action_low", action_low),
        ("action_high", action_high),
    ):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
    if bool((rate_limit <= 0.0).any()):
        raise ValueError("rate_limit must be strictly positive")
    if bool((action_low >= action_high).any()):
        raise ValueError("action_low must be strictly below action_high")

    control_dt = float(control_dt)
    decay = float(decay)
    if not math.isfinite(control_dt) or control_dt <= 0.0:
        raise ValueError("control_dt must be finite and positive")
    if not math.isfinite(decay) or not 0.0 < decay < 1.0:
        raise ValueError("decay must lie strictly between zero and one")

    desired_rate = rate_limit * torch.tanh(raw_target_rate)
    proposed_rate = (
        decay * previous_rate + (1.0 - decay) * desired_rate
    )
    requested_action = previous_action + control_dt * proposed_rate
    requested_action = torch.maximum(
        torch.minimum(requested_action, action_high),
        action_low,
    )

    if active_mask is not None:
        expected_mask_shape = previous_action.shape[:-1]
        if active_mask.shape != expected_mask_shape:
            raise ValueError(
                f"active_mask must have shape {expected_mask_shape}, "
                f"got {tuple(active_mask.shape)}"
            )
        active_mask = active_mask.to(device=previous_action.device, dtype=torch.bool)
        requested_action = torch.where(
            active_mask.unsqueeze(-1),
            requested_action,
            previous_action,
        )
    return requested_action
