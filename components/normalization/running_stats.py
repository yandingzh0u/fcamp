"""Transactional running normalization dedicated to imitation observations."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Streaming observation normalization used by the actor and critic."""

    def __init__(self, shape: int, device, eps: float = 1e-2):
        super().__init__()
        self.eps = eps
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + self.eps)

    @torch.no_grad()
    def _update(self, x: torch.Tensor) -> None:
        batch_size = x.shape[0]
        batch_mean = torch.mean(x, dim=0, keepdim=True)
        batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)
        new_count = self.count + batch_size
        delta = batch_mean - self._mean
        self._mean.copy_(self._mean + delta * (batch_size / new_count))
        m_a = self._var * self.count
        m_b = batch_var * batch_size
        m2 = m_a + m_b + delta.pow(2) * (self.count * batch_size / new_count)
        self._var.copy_(m2 / new_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(new_count)


class RunningNormalizer(nn.Module):
    """Running moments with explicit pending-record and commit phases.

    ``normalize`` always uses committed statistics, so rollout rewards remain
    stationary within an iteration. ``freeze`` prevents an accidental commit;
    samples may still be recorded and committed after ``unfreeze``.
    """

    def __init__(
        self,
        shape: int | Iterable[int],
        *,
        device: torch.device | str = "cpu",
        clip: float = 10.0,
        min_std: float = 1.0e-4,
    ) -> None:
        super().__init__()
        shape_tuple = (shape,) if isinstance(shape, int) else tuple(shape)
        if not shape_tuple or any(int(v) <= 0 for v in shape_tuple):
            raise ValueError(f"invalid normalizer shape {shape_tuple}")
        if clip <= 0 or min_std <= 0:
            raise ValueError("clip and min_std must be positive")
        self.clip = float(clip)
        self.min_std = float(min_std)
        self.register_buffer("count", torch.zeros((), dtype=torch.float64, device=device))
        self.register_buffer("mean", torch.zeros(shape_tuple, dtype=torch.float32, device=device))
        self.register_buffer("variance", torch.ones(shape_tuple, dtype=torch.float32, device=device))
        self.register_buffer("pending_count", torch.zeros((), dtype=torch.float64, device=device))
        self.register_buffer("pending_sum", torch.zeros(shape_tuple, dtype=torch.float64, device=device))
        self.register_buffer("pending_sum_sq", torch.zeros(shape_tuple, dtype=torch.float64, device=device))
        self.register_buffer("frozen", torch.tensor(False, dtype=torch.bool, device=device))

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(torch.clamp(self.variance, min=self.min_std**2))

    @torch.no_grad()
    def record(self, samples: torch.Tensor) -> None:
        feature_ndim = self.mean.ndim
        if samples.ndim < feature_ndim + 1 or tuple(samples.shape[-feature_ndim:]) != tuple(self.mean.shape):
            raise ValueError(
                f"samples must end in normalizer shape {tuple(self.mean.shape)}, got {tuple(samples.shape)}"
            )
        values = samples.detach().to(device=self.mean.device, dtype=torch.float64)
        values = values.reshape((-1,) + tuple(self.mean.shape))
        self.pending_count.add_(values.shape[0])
        self.pending_sum.add_(values.sum(dim=0))
        self.pending_sum_sq.add_((values * values).sum(dim=0))

    @torch.no_grad()
    def commit(self) -> bool:
        if bool(self.frozen.item()):
            raise RuntimeError("cannot commit frozen normalizer; call unfreeze() first")
        n = float(self.pending_count.item())
        if n == 0:
            return False
        old_n = float(self.count.item())
        total = old_n + n
        old_second = self.variance.double() + self.mean.double().square()
        combined_sum = old_n * self.mean.double() + self.pending_sum
        combined_second_sum = old_n * old_second + self.pending_sum_sq
        new_mean = combined_sum / total
        new_var = combined_second_sum / total - new_mean.square()
        self.mean.copy_(new_mean.float())
        self.variance.copy_(torch.clamp(new_var, min=self.min_std**2).float())
        self.count.fill_(total)
        self.clear_pending()
        return True

    @torch.no_grad()
    def clear_pending(self) -> None:
        self.pending_count.zero_()
        self.pending_sum.zero_()
        self.pending_sum_sq.zero_()

    @torch.no_grad()
    def freeze(self) -> None:
        self.frozen.fill_(True)

    @torch.no_grad()
    def unfreeze(self) -> None:
        self.frozen.fill_(False)

    def normalize(self, samples: torch.Tensor) -> torch.Tensor:
        normalized = (samples - self.mean.to(dtype=samples.dtype)) / self.std.to(dtype=samples.dtype)
        return torch.clamp(normalized, -self.clip, self.clip)

    @torch.no_grad()
    def statistics(self, samples: torch.Tensor | None = None, prefix: str = "disc_norm") -> dict[str, float]:
        metrics = {
            f"{prefix}/count": float(self.count.item()),
            f"{prefix}/pending_count": float(self.pending_count.item()),
            f"{prefix}/frozen": float(self.frozen.item()),
            f"{prefix}/mean_abs": float(self.mean.abs().mean().item()),
            f"{prefix}/std_mean": float(self.std.mean().item()),
            f"{prefix}/std_min": float(self.std.min().item()),
            f"{prefix}/std_max": float(self.std.max().item()),
        }
        if samples is not None:
            normalized_unclipped = (samples - self.mean.to(samples)) / self.std.to(samples)
            metrics[f"{prefix}/clip_fraction"] = float(
                (normalized_unclipped.abs() > self.clip).float().mean().item()
            )
        return metrics
