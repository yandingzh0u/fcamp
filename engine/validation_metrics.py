from __future__ import annotations

import torch


class ChunkBoundaryDiagnostics:
    """Accumulate executed-action d1/d2 RMS at chunk seams and interiors."""

    _DERIVATIVES = ("delta", "d2")
    _CATEGORIES = ("boundary", "internal")

    def __init__(self, *, horizon: int, initial_action: torch.Tensor) -> None:
        if horizon <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if initial_action.ndim != 2:
            raise ValueError("initial_action must be a rank-2 tensor")
        self.horizon = int(horizon)
        self.num_envs = initial_action.shape[0]
        self.device = initial_action.device
        self._previous_action = initial_action.detach().clone()
        self._previous_delta = torch.zeros_like(initial_action)
        self._episode_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        shape = (len(self._DERIVATIVES), len(self._CATEGORIES))
        self._sum_sq = torch.zeros(shape, dtype=torch.float64, device=self.device)
        self._counts = torch.zeros(shape, dtype=torch.long, device=self.device)

    def update(
        self,
        *,
        active_mask: torch.Tensor,
        chunk_offset: int | torch.Tensor,
        action: torch.Tensor,
    ) -> None:
        active = self._mask(active_mask)
        if action.shape != self._previous_action.shape:
            raise ValueError(
                f"action must have shape {tuple(self._previous_action.shape)}, "
                f"got {tuple(action.shape)}"
            )
        offsets = torch.as_tensor(
            chunk_offset, dtype=torch.long, device=self.device
        ).reshape(-1)
        if offsets.numel() == 1:
            offsets = offsets.expand(self.num_envs)
        if offsets.shape != (self.num_envs,):
            raise ValueError(
                f"chunk_offset must contain {self.num_envs} values, "
                f"got {tuple(offsets.shape)}"
            )
        if bool(((offsets < 0) | (offsets >= self.horizon)).any()):
            raise ValueError(f"chunk offsets must lie in [0, {self.horizon - 1}]")

        delta = action - self._previous_action
        derivatives = (delta, delta - self._previous_delta)
        measured = active & (self._episode_steps > 0)
        categories = (measured & (offsets == 0), measured & (offsets != 0))
        for derivative_idx, values in enumerate(derivatives):
            for category_idx, mask in enumerate(categories):
                selected = values[mask]
                self._sum_sq[derivative_idx, category_idx] += (
                    selected.double().square().sum()
                )
                self._counts[derivative_idx, category_idx] += selected.numel()

        self._previous_action[active] = action[active]
        self._previous_delta[active] = delta[active]
        self._episode_steps[active] += 1

    def metrics(self, prefix: str = "validation") -> dict[str, float]:
        metrics: dict[str, float] = {}
        for derivative_idx, derivative in enumerate(self._DERIVATIVES):
            rms_values: list[float] = []
            for category_idx, category in enumerate(self._CATEGORIES):
                count = int(self._counts[derivative_idx, category_idx].item())
                rms = (
                    float(
                        (
                            self._sum_sq[derivative_idx, category_idx] / count
                        )
                        .sqrt()
                        .item()
                    )
                    if count
                    else -1.0
                )
                base = f"{prefix}/chunk_action_{derivative}_{category}_component"
                metrics[f"{base}_count"] = float(count)
                metrics[f"{base}_rms"] = rms
                rms_values.append(rms)
            boundary_rms, internal_rms = rms_values
            metrics[
                f"{prefix}/chunk_action_{derivative}_"
                "boundary_internal_component_rms_ratio"
            ] = (
                boundary_rms / internal_rms
                if boundary_rms >= 0.0 and internal_rms > 1.0e-12
                else -1.0
            )
        return metrics

    def _mask(self, mask: torch.Tensor) -> torch.Tensor:
        selected = mask.to(device=self.device, dtype=torch.bool).reshape(-1)
        if selected.shape != (self.num_envs,):
            raise ValueError(
                f"mask must contain {self.num_envs} values, "
                f"got {tuple(selected.shape)}"
            )
        return selected


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
