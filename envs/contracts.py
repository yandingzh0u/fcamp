from __future__ import annotations

from typing import Literal

import torch


RootVelocityFrame = Literal["com", "link"]


def resolve_root_velocity_frame(
    configured_frame: str,
    explicit_frame: RootVelocityFrame | None,
) -> RootVelocityFrame:
    resolved = str(configured_frame if explicit_frame is None else explicit_frame)
    if resolved not in {"com", "link"}:
        raise ValueError(
            f"root velocity frame must be 'com' or 'link', got {resolved!r}"
        )
    return resolved  # type: ignore[return-value]


def select_imitation_root_domain(
    *,
    strict_fcamp: bool,
    legacy_root_pos: torch.Tensor,
    legacy_root_quat: torch.Tensor,
    legacy_root_velocity: torch.Tensor,
    root_link_pos: torch.Tensor,
    root_link_quat: torch.Tensor,
    root_link_velocity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select FCAMP root-link features without changing other methods."""

    if not isinstance(strict_fcamp, bool):
        raise TypeError("strict_fcamp must be bool")
    legacy = (legacy_root_pos, legacy_root_quat, legacy_root_velocity)
    link = (root_link_pos, root_link_quat, root_link_velocity)
    for legacy_value, link_value in zip(legacy, link, strict=True):
        if legacy_value.shape != link_value.shape:
            raise ValueError("legacy and root-link state tensors must have matching shapes")
    return link if strict_fcamp else legacy


def validate_actions_in_bounds(
    actions: torch.Tensor,
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    tolerance: float = 1.0e-6,
) -> None:
    if actions.shape[-1:] != low.shape[-1:] or low.shape != high.shape:
        raise ValueError("Actions and bounds have incompatible shapes")
    if not bool(torch.isfinite(actions).all()):
        raise ValueError("Actions must be finite")
    below = (low - actions).clamp_min(0.0)
    above = (actions - high).clamp_min(0.0)
    max_violation = torch.maximum(below, above).max()
    if float(max_violation.item()) > float(tolerance):
        raise RuntimeError(
            "Action violates the configured policy command domain; "
            f"max violation={float(max_violation.item()):.6g}"
        )
