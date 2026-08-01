from __future__ import annotations

import torch


def float_restore_error(
    restored: torch.Tensor,
    expected: torch.Tensor,
    *,
    absolute_floor: float = 1.0e-6,
    ulps: float = 2.0,
) -> tuple[float, float, float]:
    """Measure restore error against an elementwise float-spacing bound."""

    if restored.shape != expected.shape or restored.dtype != expected.dtype:
        raise RuntimeError("restored state tensor layout changed")
    if not restored.is_floating_point():
        difference = (restored != expected).to(torch.float32)
        maximum = float(difference.max().item())
        return maximum, maximum, 0.0
    positive_infinity = torch.full_like(expected, float("inf"))
    negative_infinity = torch.full_like(expected, float("-inf"))
    spacing = torch.maximum(
        (torch.nextafter(expected, positive_infinity) - expected).abs(),
        (expected - torch.nextafter(expected, negative_infinity)).abs(),
    )
    tolerance = torch.maximum(
        torch.full_like(expected, float(absolute_floor)),
        float(ulps) * spacing,
    )
    error = (restored - expected).abs()
    return (
        float(error.max().item()),
        float((error / tolerance).max().item()),
        float(tolerance.max().item()),
    )
