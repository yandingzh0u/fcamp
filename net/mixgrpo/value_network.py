from __future__ import annotations

import torch
from torch import nn

from .flow_policy import _activation


class ValueNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
    ):
        super().__init__()
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be positive, got {obs_dim}")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")

        layers: list[nn.Module] = []
        in_dim = int(obs_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_activation(activation))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.value_net = nn.Sequential(*layers)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.value_net(observation).squeeze(-1)
