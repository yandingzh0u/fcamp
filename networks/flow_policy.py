from __future__ import annotations

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "elu":
        return nn.ELU()
    if normalized == "relu":
        return nn.ReLU()
    if normalized == "silu":
        return nn.SiLU()
    if normalized == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class FlowMatchingPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 29,
        horizon: int = 1,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        action_squash_scale: float = 5.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon


        self.chunk_dim = horizon * action_dim
        self.action_chunk_dim = horizon * action_dim
        self.obs_dim = obs_dim
        self.hidden_dims = tuple(hidden_dims)
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        layers: list[nn.Module] = []
        in_dim = self.obs_dim + self.chunk_dim + 1
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_activation(activation))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.chunk_dim))
        self.velocity_net = nn.Sequential(*layers)
        if action_squash_scale <= 0.0:
            raise ValueError(f"action_squash_scale must be > 0, got {action_squash_scale}")
        self.action_squash_scale = float(action_squash_scale)

    def _prepare_observation(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Expected observation dim {self.obs_dim}, got {observation.shape[-1]}"
            )
        return observation

    def _validate_inputs(self, observation: torch.Tensor, flow_noise: torch.Tensor, steps: int) -> None:
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if flow_noise.shape[-1] != self.chunk_dim:
            raise ValueError(f"Expected noise dim {self.chunk_dim}, got {flow_noise.shape[-1]}")
        if observation.shape[0] != flow_noise.shape[0]:
            raise ValueError(
                f"Observation batch size {observation.shape[0]} must match noise batch size {flow_noise.shape[0]}"
            )

    def velocity_field(self, observation: torch.Tensor, noisy_actions: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        observation = self._prepare_observation(observation)
        if time.ndim != 1 or time.shape[0] != observation.shape[0]:
            raise ValueError(f"time must have shape ({observation.shape[0]},), got {tuple(time.shape)}")
        net_input = torch.cat([observation, noisy_actions, time.unsqueeze(-1)], dim=-1)
        return self.velocity_net(net_input)

    def _action_transform(self, action_value: torch.Tensor) -> torch.Tensor:
        scale = self.action_squash_scale
        leading_shape = action_value.shape[:-1]
        chunk = action_value.view(*leading_shape, self.horizon, self.action_dim)
        squashed = scale * torch.tanh(chunk / scale)
        return squashed.reshape(*leading_shape, self.action_chunk_dim)
