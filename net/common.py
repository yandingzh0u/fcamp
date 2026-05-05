from __future__ import annotations

import math

import torch
from torch import nn


MIN_POLICY_OBS_DIM = 154
LOG_TWO_PI = math.log(2.0 * math.pi)


def sinusoidal_time_embedding(time: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim must be even, got {embedding_dim}")

    half_dim = embedding_dim // 2
    if half_dim == 0:
        return time.unsqueeze(-1)

    frequencies = torch.exp(
        torch.linspace(0.0, -math.log(10_000.0), half_dim, device=time.device, dtype=time.dtype)
    )
    phase = time.unsqueeze(-1) * frequencies.unsqueeze(0)
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden
        hidden = self.norm(hidden)
        hidden = self.act(self.fc1(hidden))
        hidden = self.fc2(hidden)
        return residual + hidden
