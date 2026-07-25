from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    def __init__(self, shape: int, device, eps: float = 1e-2, until: int | None = None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, center: bool = True, update: bool = True) -> torch.Tensor:
        if self.training and update:
            self._update(x)
        if center:
            return (x - self._mean) / (self._std + self.eps)
        return x / (self._std + self.eps)

    @torch.no_grad()
    def _update(self, x: torch.Tensor) -> None:
        if self.until is not None and self.count >= self.until:
            return
        batch_size = x.shape[0]
        batch_mean = torch.mean(x, dim=0, keepdim=True)
        batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)
        new_count = self.count + batch_size

        delta = batch_mean - self._mean
        self._mean.copy_(self._mean + delta * (batch_size / new_count))
        m_a = self._var * self.count
        m_b = batch_var * batch_size
        M2 = m_a + m_b + delta.pow(2) * (self.count * batch_size / new_count)
        self._var.copy_(M2 / new_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(new_count)
