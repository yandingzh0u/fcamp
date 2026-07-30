from __future__ import annotations

import torch


def require_finite_tensors(
    tensors: dict[str, torch.Tensor],
    *,
    context: str,
) -> None:
    """Fail with field names using one device synchronization on the healthy path."""
    if not tensors:
        return
    names = tuple(tensors)
    invalid = torch.stack(
        tuple(~torch.isfinite(tensors[name]).all() for name in names)
    )
    if bool(invalid.any()):
        invalid_cpu = invalid.detach().cpu().tolist()
        bad_names = [name for name, is_bad in zip(names, invalid_cpu) if is_bad]
        raise RuntimeError(
            f"{context} contains non-finite values in fields: {bad_names}"
        )


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
