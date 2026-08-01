from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    try:
        return activations[name.lower()]()
    except KeyError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


class ValueCritic(nn.Module):
    """State-only scalar value function."""

    def __init__(
        self,
        observation_dim: int,
        hidden_dims: Sequence[int],
        activation: str,
    ) -> None:
        super().__init__()
        if int(observation_dim) <= 0:
            raise ValueError("observation_dim must be positive")
        layers: list[nn.Module] = []
        input_dim = int(observation_dim)
        for hidden_dim in hidden_dims:
            width = int(hidden_dim)
            if width <= 0:
                raise ValueError("hidden dimensions must be positive")
            layers.extend((nn.Linear(input_dim, width), _activation(activation)))
            input_dim = width
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
        self.observation_dim = int(observation_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[-1] != self.observation_dim:
            raise ValueError(
                "observation must have shape "
                f"[batch, {self.observation_dim}], got {tuple(observation.shape)}"
            )
        return self.network(observation).squeeze(-1)

    def evaluate(self, observation: torch.Tensor) -> torch.Tensor:
        return self(observation)
