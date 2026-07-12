from __future__ import annotations

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    activations = {"elu": nn.ELU, "relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}
    if normalized not in activations:
        raise ValueError(f"Unknown activation {name!r}")
    return activations[normalized]()


def _init_linear(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1.0)
        nn.init.zeros_(module.bias)


class FlowRLActor(nn.Module):
    """ByteDance FlowRL velocity field with midpoint ODE integration."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str,
        flow_steps: int,
        action_scale: float,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("FlowRL actor requires at least one hidden layer")
        if flow_steps < 1:
            raise ValueError("flow_steps must be positive")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.flow_steps = int(flow_steps)
        self.action_scale = float(action_scale)

        layers: list[nn.Module] = []
        last_dim = self.obs_dim + self.action_dim + 1
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(last_dim, hidden_dim), nn.LayerNorm(hidden_dim), _activation(activation)))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, self.action_dim))
        self.velocity_net = nn.Sequential(*layers)
        self.apply(_init_linear)

    def velocity(self, obs: torch.Tensor, action_t: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return self.velocity_net(torch.cat((obs, action_t, time), dim=-1))

    def _midpoint_step(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        time_start: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        velocity_start = self.velocity(obs, action, time_start)
        midpoint_action = action + 0.5 * dt * velocity_start
        midpoint_time = time_start + 0.5 * dt
        return action + dt * self.velocity(obs, midpoint_action, midpoint_time)

    def sample(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
        base_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = obs.shape[0]
        if base_noise is None:
            if deterministic:
                base_noise = torch.zeros(batch, self.action_dim, device=obs.device, dtype=obs.dtype)
            else:
                base_noise = torch.randn(batch, self.action_dim, device=obs.device, dtype=obs.dtype)
        action = base_noise.clamp(-1.0, 1.0)
        dt = 1.0 / self.flow_steps
        time = torch.zeros(batch, 1, device=obs.device, dtype=obs.dtype)
        for _ in range(self.flow_steps):
            action = self._midpoint_step(obs, action, time, dt)
            time = time + dt
        return self.action_scale * torch.tanh(action), base_noise

    def cfm_loss(
        self,
        obs: torch.Tensor,
        data_action: torch.Tensor,
        base_noise: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        action_t = time * data_action + (1.0 - time) * base_noise
        target_velocity = data_action - base_noise
        predicted_velocity = self.velocity(obs, action_t, time)
        return (predicted_velocity - target_velocity).pow(2).mean(dim=-1, keepdim=True)


class _ValueBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("FlowRL critic requires at least one hidden layer")
        layers: list[nn.Module] = []
        last_dim = input_dim
        for index, hidden_dim in enumerate(hidden_dims):
            layers.extend((nn.Linear(last_dim, hidden_dim), nn.LayerNorm(hidden_dim)))
            if index > 0:
                layers.append(nn.GELU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 1))
        self.net = nn.Sequential(*layers)
        self.apply(_init_linear)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class FlowRLTwinQ(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        input_dim = int(obs_dim) + int(action_dim)
        self.q1 = _ValueBranch(input_dim, hidden_dims)
        self.q2 = _ValueBranch(input_dim, hidden_dims)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = torch.cat((obs, action), dim=-1)
        return self.q1(inputs), self.q2(inputs)


class FlowRLValue(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims: tuple[int, ...]) -> None:
        super().__init__()
        self.value = _ValueBranch(int(obs_dim), hidden_dims)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.value(obs)
