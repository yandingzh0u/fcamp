from __future__ import annotations

import torch
from torch import nn


def flow_ode_mean(
    model_output: torch.Tensor,
    latents: torch.Tensor,
    sigmas: torch.Tensor,
    index: int,
) -> torch.Tensor:
    """Advance one deterministic Flow ODE step."""

    sigma = sigmas[index].to(device=model_output.device, dtype=model_output.dtype)
    sigma_next = sigmas[index + 1].to(device=model_output.device, dtype=model_output.dtype)
    dt = sigma_next - sigma
    return latents + model_output * dt


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


def _build_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, h))
        layers.append(_activation(activation))
        last = h
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class FlowMatchingPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dims: tuple[int, ...],
        activation: str,
        action_squash_scale: float,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.chunk_dim = horizon * action_dim
        self.obs_dim = obs_dim
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")

        # Ordered causal velocity: v_k depends only on z_0..z_k. Keep module
        # construction order stable because it is part of the seeded baseline.
        hidden = int(hidden_dims[-1])
        self.obs_encoder = _build_mlp(
            self.obs_dim + 1, tuple(hidden_dims), hidden, activation
        )
        self.frame_pos_embed = nn.Parameter(torch.zeros(self.horizon, hidden))
        nn.init.normal_(self.frame_pos_embed, std=0.02)
        self.token_encoder = _build_mlp(
            self.action_dim + hidden,
            tuple(hidden_dims),
            hidden,
            activation,
        )
        self.causal_cell = nn.GRUCell(hidden, hidden)
        self.vel_head = _build_mlp(
            hidden * 2, tuple(hidden_dims), self.action_dim, activation
        )
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
        b = observation.shape[0]
        chunk = noisy_actions.view(b, self.horizon, self.action_dim)
        obs_h = self.obs_encoder(torch.cat([observation, time.unsqueeze(-1)], dim=-1))
        frame_pos = self.frame_pos_embed.unsqueeze(0).expand(b, -1, -1)
        token_h = self.token_encoder(torch.cat([chunk, frame_pos], dim=-1))
        state = obs_h
        velocity_frames: list[torch.Tensor] = []
        for frame_idx in range(self.horizon):
            state = self.causal_cell(token_h[:, frame_idx], state)
            velocity_frames.append(self.vel_head(torch.cat([obs_h, state], dim=-1)))
        return torch.stack(velocity_frames, dim=1).reshape(b, self.chunk_dim)

    def _action_transform(
        self,
        action_value: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> torch.Tensor:
        """Convert causal residuals into bounded absolute PD commands."""

        scale = self.action_squash_scale
        leading_shape = action_value.shape[:-1]
        chunk = action_value.view(*leading_shape, self.horizon, self.action_dim)
        prev = prev_action.reshape(*leading_shape, self.action_dim)
        eps = 1.0e-6
        prev_normalized = torch.clamp(prev / scale, -1.0 + eps, 1.0 - eps)
        latent_action = scale * torch.atanh(prev_normalized)
        actions = []
        for frame_idx in range(self.horizon):
            latent_action = latent_action + chunk[..., frame_idx, :]
            actions.append(scale * torch.tanh(latent_action / scale))
        return torch.stack(actions, dim=-2).reshape(
            *leading_shape, self.chunk_dim
        )
