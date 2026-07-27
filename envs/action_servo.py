"""Exact stateful position-servo dynamics."""

from __future__ import annotations

import math

import torch


def advance_position_servo(
    target_action: torch.Tensor,
    action: torch.Tensor,
    rate: torch.Tensor,
    acceleration: torch.Tensor,
    *,
    dt: float,
    omega: float,
    active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance a third-order critically damped position servo exactly.

    ``target_action`` is constant over this interval.  The carried state obeys

    ``action_dot = rate``,
    ``rate_dot = acceleration``,
    ``acceleration_dot = omega**3 * (target_action - action)
                         - 3 * omega**2 * rate
                         - 3 * omega * acceleration``.

    Repeated calls with a physics-substep ``dt`` therefore produce the exact
    substep command trajectory without Euler integration or interpolation.
    """

    state_shape = action.shape
    if action.ndim < 1:
        raise ValueError("servo tensors must have at least one dimension")
    for name, value in (
        ("target_action", target_action),
        ("rate", rate),
        ("acceleration", acceleration),
    ):
        if value.shape != state_shape:
            raise ValueError(
                "target_action, action, rate, and acceleration must have "
                "identical shapes"
            )
    for name, value in (
        ("target_action", target_action),
        ("action", action),
        ("rate", rate),
        ("acceleration", acceleration),
    ):
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")

    dt = float(dt)
    omega = float(omega)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(omega) or omega <= 0.0:
        raise ValueError("omega must be finite and positive")

    scaled_time = omega * dt
    decay = math.exp(-scaled_time)
    position_error = action - target_action
    linear_coefficient = rate + omega * position_error
    quadratic_coefficient = (
        acceleration
        + 2.0 * omega * rate
        + omega**2 * position_error
    )
    next_action = (
        action
        + math.expm1(-scaled_time) * position_error
        + decay
        * (
            linear_coefficient * dt
            + 0.5 * quadratic_coefficient * dt**2
        )
    )
    next_rate = decay * (
        rate
        + (acceleration + omega * rate) * dt
        - 0.5 * omega * quadratic_coefficient * dt**2
    )
    next_acceleration = decay * (
        acceleration
        + (
            -2.0 * omega * quadratic_coefficient
            + omega**2 * linear_coefficient
        )
        * dt
        + 0.5 * omega**2 * quadratic_coefficient * dt**2
    )

    if active_mask is not None:
        expected_shape = state_shape[:-1]
        if active_mask.shape != expected_shape:
            raise ValueError(
                f"active_mask must have shape {expected_shape}, "
                f"got {tuple(active_mask.shape)}"
            )
        active = active_mask.to(device=action.device, dtype=torch.bool).unsqueeze(-1)
        next_action = torch.where(active, next_action, action)
        next_rate = torch.where(active, next_rate, rate)
        next_acceleration = torch.where(
            active, next_acceleration, acceleration
        )

    return next_action, next_rate, next_acceleration
