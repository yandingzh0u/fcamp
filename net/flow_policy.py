from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .common import MIN_POLICY_OBS_DIM, ResidualMLPBlock, sinusoidal_time_embedding


class FlowMatchingPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 29,
        horizon: int = 1,
        hidden_dim: int = 512,
        time_embed_dim: int = 64,
        depth: int = 4,
        action_limit: float = 1.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.chunk_dim = horizon * action_dim
        self.obs_dim = max(obs_dim, MIN_POLICY_OBS_DIM)
        self.time_embed_dim = time_embed_dim
        self.action_limit = float(action_limit)

        self.state_proj = nn.Linear(self.chunk_dim, hidden_dim)
        self.obs_proj = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(ResidualMLPBlock(hidden_dim) for _ in range(depth))
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.chunk_dim),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def _prepare_observation(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] == self.obs_dim:
            return observation
        if observation.shape[-1] > self.obs_dim:
            return observation[..., : self.obs_dim]
        return F.pad(observation, (0, self.obs_dim - observation.shape[-1]))

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
        time_features = sinusoidal_time_embedding(time, self.time_embed_dim)
        hidden = self.state_proj(noisy_actions) + self.obs_proj(observation) + self.time_proj(time_features)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_head(hidden)

    def _integrate_flow(self, observation: torch.Tensor, flow_noise: torch.Tensor, steps: int = 8) -> torch.Tensor:
        self._validate_inputs(observation, flow_noise, steps)
        observation = self._prepare_observation(observation)

        sample = flow_noise
        integration_times = torch.linspace(
            0.0,
            1.0,
            steps + 1,
            device=flow_noise.device,
            dtype=flow_noise.dtype,
        )
        for index in range(steps):
            time = torch.full(
                (flow_noise.shape[0],),
                integration_times[index],
                device=flow_noise.device,
                dtype=flow_noise.dtype,
            )
            dt = integration_times[index + 1] - integration_times[index]
            sample = sample + dt * self.velocity_field(observation, sample, time)
        return sample

    def _action_transform(self, pre_tanh_value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(pre_tanh_value) * self.action_limit

    def forward(self, observation: torch.Tensor, noise: torch.Tensor, steps: int = 8) -> torch.Tensor:
        return self._action_transform(self._integrate_flow(observation, noise, steps=steps)).view(
            noise.shape[0],
            self.horizon,
            self.action_dim,
        )
