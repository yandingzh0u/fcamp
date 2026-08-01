from __future__ import annotations

import torch


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
