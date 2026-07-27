from __future__ import annotations

import math

import torch


def advance_rate_servo(
    target_rate: torch.Tensor,
    action: torch.Tensor,
    rate: torch.Tensor,
    acceleration: torch.Tensor,
    *,
    dt: float,
    omega: float,
    active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the critically damped target-rate servo exactly.

    ``target_rate`` is constant over this interval.  The carried state follows

    ``action_dot = rate``,
    ``rate_dot = acceleration``,
    ``acceleration_dot = omega**2 * (target_rate - rate)
                         - 2 * omega * acceleration``.

    Repeated calls with a physics-substep ``dt`` therefore produce the exact
    substep command trajectory without Euler integration or interpolation.
    """

    state_shape = action.shape
    if action.ndim < 1:
        raise ValueError("servo tensors must have at least one dimension")
    for name, value in (
        ("target_rate", target_rate),
        ("rate", rate),
        ("acceleration", acceleration),
    ):
        if value.shape != state_shape:
            raise ValueError(
                "target_rate, action, rate, and acceleration must have "
                "identical shapes"
            )
    for name, value in (
        ("target_rate", target_rate),
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
    rate_error = rate - target_rate
    repeated_root_coefficient = acceleration + omega * rate_error

    next_rate = target_rate + (
        rate_error + repeated_root_coefficient * dt
    ) * decay
    next_acceleration = (
        acceleration
        - omega * repeated_root_coefficient * dt
    ) * decay
    rate_error_integral = (
        rate_error * (-math.expm1(-scaled_time) / omega)
        + repeated_root_coefficient
        * ((-math.expm1(-scaled_time) - scaled_time * decay) / omega**2)
    )
    next_action = action + target_rate * dt + rate_error_integral

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
