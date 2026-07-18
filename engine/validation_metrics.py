from __future__ import annotations

import torch


def terminal_phase_metrics(
    prefix: str,
    phases: torch.Tensor,
    mask: torch.Tensor,
    *,
    motion_end_phase: float,
) -> dict[str, float]:
    selected = phases[mask.bool()].float()
    metrics = {f"{prefix}/count": float(selected.numel())}
    if selected.numel() == 0:
        for name in ("min", "mean", "p50", "p95", "max", "progress_mean"):
            metrics[f"{prefix}/phase_{name}"] = -1.0
        return metrics
    metrics.update(
        {
            f"{prefix}/phase_min": float(selected.min().item()),
            f"{prefix}/phase_mean": float(selected.mean().item()),
            f"{prefix}/phase_p50": float(torch.quantile(selected, 0.50).item()),
            f"{prefix}/phase_p95": float(torch.quantile(selected, 0.95).item()),
            f"{prefix}/phase_max": float(selected.max().item()),
            f"{prefix}/phase_progress_mean": float(
                (selected / max(float(motion_end_phase), 1.0)).mean().item()
            ),
        }
    )
    return metrics
