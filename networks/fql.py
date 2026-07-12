from __future__ import annotations

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    activations = {
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    normalized = name.lower()
    if normalized not in activations:
        raise ValueError(f"Unknown FQL activation {name!r}")
    return activations[normalized]()


def _init_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        nn.init.zeros_(module.bias)


class _MLP(nn.Module):
    """Flax FQL MLP ordering: linear, activation, then optional layer norm."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        output_dim: int,
        activation: str,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("FQL networks require at least one hidden layer")
        layers: list[nn.Module] = []
        last_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(_activation(activation))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)
        self.apply(_init_linear)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class FQLVectorField(nn.Module):
    """Vector field used by either FQL's BC flow or its one-step policy."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str,
        layer_norm: bool,
        *,
        time_conditioned: bool,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.time_conditioned = bool(time_conditioned)
        input_dim = self.obs_dim + self.action_dim + int(self.time_conditioned)
        self.mlp = _MLP(
            input_dim,
            tuple(hidden_dims),
            self.action_dim,
            activation,
            bool(layer_norm),
        )

    def forward(
        self,
        obs: torch.Tensor,
        action_or_noise: torch.Tensor,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.time_conditioned:
            if time is None:
                raise ValueError("FQL BC flow requires a time input")
            inputs = torch.cat((obs, action_or_noise, time), dim=-1)
        else:
            if time is not None:
                raise ValueError("FQL one-step actor does not take a time input")
            inputs = torch.cat((obs, action_or_noise), dim=-1)
        return self.mlp(inputs)


class _QBranch(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        self.mlp = _MLP(
            int(obs_dim) + int(action_dim),
            tuple(hidden_dims),
            1,
            activation,
            bool(layer_norm),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat((obs, action), dim=-1))


class FQLTwinQ(nn.Module):
    """Two independently initialized Q functions, matching official FQL."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        self.q1 = _QBranch(obs_dim, action_dim, hidden_dims, activation, layer_norm)
        self.q2 = _QBranch(obs_dim, action_dim, hidden_dims, activation, layer_norm)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)
