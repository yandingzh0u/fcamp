"""Streaming normalization for actor and critic observations."""

from __future__ import annotations

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
