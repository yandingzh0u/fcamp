from __future__ import annotations

import copy
import math

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    key = name.lower()
    if key not in activations:
        raise ValueError(f"Unsupported activation: {name}")
    return activations[key]()


class FourierTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        if embedding_dim < 8 or embedding_dim % 8:
            raise ValueError("PolicyFlow timestep_embed_dim must be divisible by 8")
        self.register_buffer(
            "frequencies", torch.randn(embedding_dim // 8) * 16.0
        )
        self.mlp = nn.Sequential(
            nn.Linear(2 * (embedding_dim // 8), embedding_dim),
            nn.Mish(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 1:
            time = time.unsqueeze(-1)
        angles = time * (2.0 * math.pi * self.frequencies).unsqueeze(0)
        return self.mlp(torch.cat((torch.cos(angles), torch.sin(angles)), dim=-1))


class PolicyFlowVelocity(nn.Module):
    """Official PolicyFlow MLP velocity field and linear observation conditioner."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str,
        timestep_embed_dim: int,
    ) -> None:
        super().__init__()
        self.obs_condition = nn.Linear(obs_dim, timestep_embed_dim)
        self.time_embedding = FourierTimeEmbedding(timestep_embed_dim)

        layers: list[nn.Module] = []
        input_dim = action_dim + timestep_embed_dim
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, hidden_dim), _activation(activation)))
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))
        self.velocity_net = nn.Sequential(*layers)

    def forward(
        self,
        observation: torch.Tensor,
        noisy_action: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.obs_condition(observation) + self.time_embedding(time)
        return self.velocity_net(torch.cat((noisy_action, condition), dim=-1))


class PolicyFlowActor(nn.Module):
    """Continuous normalizing flow used by the official PolicyFlow PPO objective.

    Rollouts use midpoint ODE integration followed by an additive diagonal
    Gaussian perturbation. During optimization, the old flow snapshot stays
    frozen and the velocity-field difference parameterizes the likelihood ratio.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str,
        flow_steps: int,
        timestep_embed_dim: int,
        init_noise_std: float,
    ) -> None:
        super().__init__()
        if flow_steps < 1:
            raise ValueError("PolicyFlow flow_steps must be positive")
        if init_noise_std <= 0.0:
            raise ValueError("PolicyFlow init_noise_std must be positive")
        self.action_dim = int(action_dim)
        self.flow_steps = int(flow_steps)
        self.current = PolicyFlowVelocity(
            obs_dim,
            action_dim,
            hidden_dims,
            activation,
            timestep_embed_dim,
        )
        self.last = copy.deepcopy(self.current).requires_grad_(False)
        self.log_std = nn.Parameter(
            torch.full((action_dim,), math.log(float(init_noise_std)), dtype=torch.float32)
        )

    @property
    def std(self) -> torch.Tensor:
        return self.log_std.clamp(-20.0, 4.0).exp()

    def sample_prior(
        self,
        observation: torch.Tensor,
        base_noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = observation.shape[0]
        if base_noise is None:
            base_noise = torch.randn(
                batch_size,
                self.action_dim,
                device=observation.device,
                dtype=observation.dtype,
            )
        action = base_noise
        dt = 1.0 / self.flow_steps
        for step in range(self.flow_steps):
            t0 = torch.full(
                (batch_size,),
                step * dt,
                device=observation.device,
                dtype=observation.dtype,
            )
            velocity = self.current(observation, action, t0)
            midpoint = action + 0.5 * dt * velocity
            action = action + dt * self.current(observation, midpoint, t0 + 0.5 * dt)
        return action, base_noise

    def sample_action(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prior, base_noise = self.sample_prior(observation)
        std = self.std.expand_as(prior)
        delta = std * torch.randn_like(prior)
        log_prob = torch.distributions.Normal(torch.zeros_like(prior), std).log_prob(delta).sum(-1)
        return prior + delta, {
            "prior": prior,
            "base_noise": base_noise,
            "delta": delta,
            "std": std,
            "log_prob": log_prob,
        }

    def deterministic(self, observation: torch.Tensor) -> torch.Tensor:
        base = torch.zeros(
            observation.shape[0],
            self.action_dim,
            device=observation.device,
            dtype=observation.dtype,
        )
        prior, _ = self.sample_prior(observation, base_noise=base)
        return prior

    def flow_variation(
        self,
        observation: torch.Tensor,
        old_prior: torch.Tensor,
        base_noise: torch.Tensor,
        *,
        compute_brownian: bool,
        time_index: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = observation.shape[0]
        # PolicyFlow samples from ODE grid points and their midpoints.
        if time_index is None:
            time_index = torch.randint(
                0,
                2 * self.flow_steps + 1,
                (batch_size,),
                device=observation.device,
            )
        time = time_index.to(dtype=observation.dtype) / (2.0 * self.flow_steps)
        alpha = time.unsqueeze(-1)
        noisy_action = (1.0 - alpha) * base_noise + alpha * old_prior

        with torch.no_grad():
            old_velocity = self.last(observation, noisy_action, time)
        new_velocity = self.current(observation, noisy_action, time)
        delta_velocity = new_velocity - old_velocity

        if compute_brownian:
            brownian = torch.nn.functional.mse_loss(
                (1.0 - alpha) * new_velocity,
                noisy_action - alpha * old_velocity,
            )
        else:
            brownian = new_velocity.new_zeros(())
        return delta_velocity, self.std.expand_as(old_prior), brownian

    @torch.no_grad()
    def snapshot_last(self) -> None:
        self.last.load_state_dict(self.current.state_dict())
