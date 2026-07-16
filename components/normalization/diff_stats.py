"""MimicKit ADD-style normalization for demo-policy discriminator differences."""

from __future__ import annotations

from collections.abc import Iterable
import math

import torch
from torch import nn


class DiffNormalizer(nn.Module):
    """Running mean-absolute difference normalizer with explicit commit."""

    def __init__(
        self,
        shape: int | Iterable[int],
        *,
        device: torch.device | str = "cpu",
        min_diff: float = 1.0e-4,
        clip: float = math.inf,
    ) -> None:
        super().__init__()
        shape_tuple = (shape,) if isinstance(shape, int) else tuple(shape)
        if not shape_tuple or any(int(v) <= 0 for v in shape_tuple):
            raise ValueError(f"invalid diff normalizer shape {shape_tuple}")
        if min_diff <= 0.0 or clip <= 0.0:
            raise ValueError("min_diff and clip must be positive")
        self.min_diff = float(min_diff)
        self.clip = float(clip)
        self.register_buffer("count", torch.zeros((), dtype=torch.float64, device=device))
        self.register_buffer("mean_abs", torch.ones(shape_tuple, dtype=torch.float32, device=device))
        self.register_buffer("pending_count", torch.zeros((), dtype=torch.float64, device=device))
        self.register_buffer("pending_sum_abs", torch.zeros(shape_tuple, dtype=torch.float64, device=device))

    @torch.no_grad()
    def record(self, samples: torch.Tensor) -> None:
        feature_ndim = self.mean_abs.ndim
        if samples.ndim < feature_ndim + 1 or tuple(samples.shape[-feature_ndim:]) != tuple(self.mean_abs.shape):
            raise ValueError(
                f"samples must end in normalizer shape {tuple(self.mean_abs.shape)}, got {tuple(samples.shape)}"
            )
        values = samples.detach().to(device=self.mean_abs.device, dtype=torch.float64)
        values = values.reshape((-1,) + tuple(self.mean_abs.shape))
        self.pending_count.add_(values.shape[0])
        self.pending_sum_abs.add_(values.abs().sum(dim=0))

    @torch.no_grad()
    def clear_pending(self) -> None:
        self.pending_count.zero_()
        self.pending_sum_abs.zero_()

    @torch.no_grad()
    def commit(self) -> bool:
        n = float(self.pending_count.item())
        if n == 0.0:
            return False
        total = float(self.count.item()) + n
        new_mean_abs = self.pending_sum_abs / n
        old_weight = float(self.count.item()) / total
        new_weight = n / total
        self.mean_abs.copy_((old_weight * self.mean_abs.double() + new_weight * new_mean_abs).float())
        self.count.fill_(total)
        self.clear_pending()
        return True

    def normalize(self, samples: torch.Tensor) -> torch.Tensor:
        scale = torch.clamp_min(self.mean_abs.to(dtype=samples.dtype), self.min_diff)
        normalized = samples / scale
        if math.isfinite(self.clip):
            normalized = torch.clamp(normalized, -self.clip, self.clip)
        return normalized

    @torch.no_grad()
    def statistics(self, samples: torch.Tensor | None = None, prefix: str = "disc_diff_norm") -> dict[str, float]:
        scale = torch.clamp_min(self.mean_abs, self.min_diff)
        metrics = {
            f"{prefix}/count": float(self.count.item()),
            f"{prefix}/pending_count": float(self.pending_count.item()),
            f"{prefix}/mean_abs_mean": float(self.mean_abs.mean().item()),
            f"{prefix}/mean_abs_min": float(self.mean_abs.min().item()),
            f"{prefix}/mean_abs_max": float(self.mean_abs.max().item()),
            f"{prefix}/scale_min": float(scale.min().item()),
            f"{prefix}/scale_max": float(scale.max().item()),
        }
        if samples is not None and math.isfinite(self.clip):
            normalized_unclipped = samples / scale.to(samples)
            metrics[f"{prefix}/clip_fraction"] = float(
                (normalized_unclipped.abs() > self.clip).float().mean().item()
            )
        return metrics
