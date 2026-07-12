from __future__ import annotations

import math

import torch
from torch import nn
from torch.distributions import Normal


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
        "mish": nn.Mish,
    }
    if normalized not in activations:
        raise ValueError(f"Unknown activation {name!r}")
    return activations[normalized]()


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(last_dim, hidden_dim), _activation(activation)))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class ReinFlowPolicy(nn.Module):
    """Stochastic flow Markov chain used by ReinFlow-R."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dims: tuple[int, ...],
        noise_hidden_dims: tuple[int, ...],
        activation: str,
        flow_steps: int,
        timestep_embed_dim: int,
        action_scale: float,
        min_std: float,
        max_std: float,
        randn_clip_value: float,
    ) -> None:
        super().__init__()
        if timestep_embed_dim < 2 or timestep_embed_dim % 2:
            raise ValueError("ReinFlow timestep_embed_dim must be positive and even")
        if not (0.0 < min_std <= max_std):
            raise ValueError("ReinFlow requires 0 < min_std <= max_std")
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.chunk_dim = self.horizon * self.action_dim
        self.flow_steps = int(flow_steps)
        self.timestep_embed_dim = int(timestep_embed_dim)
        self.action_scale = float(action_scale)
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.randn_clip_value = float(randn_clip_value)

        input_dim = self.obs_dim + self.timestep_embed_dim + self.chunk_dim
        self.velocity_net = _mlp(input_dim, hidden_dims, self.chunk_dim, activation)
        noise_input_dim = self.obs_dim + self.timestep_embed_dim
        self.noise_net = _mlp(noise_input_dim, noise_hidden_dims, self.chunk_dim, "tanh")
        self.register_buffer("logvar_min", torch.tensor(math.log(self.min_std**2)))
        self.register_buffer("logvar_max", torch.tensor(math.log(self.max_std**2)))

    def _embed_time(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim == 1:
            time = time.unsqueeze(-1)
        frequencies = 2.0 ** torch.arange(
            self.timestep_embed_dim // 2, device=time.device, dtype=time.dtype
        )
        phase = time * frequencies
        return torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)

    def velocity(self, obs: torch.Tensor, action: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        batch = obs.shape[0]
        action_flat = action.reshape(batch, self.chunk_dim)
        inputs = torch.cat((obs, self._embed_time(time), action_flat), dim=-1)
        return self.velocity_net(inputs).reshape(batch, self.horizon, self.action_dim)

    def noise_std(self, obs: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        raw_logvar = self.noise_net(torch.cat((obs, self._embed_time(time)), dim=-1))
        interpolation = 0.5 * (torch.tanh(raw_logvar) + 1.0)
        logvar = self.logvar_min + (self.logvar_max - self.logvar_min) * interpolation
        return torch.exp(0.5 * logvar).reshape(obs.shape[0], self.horizon, self.action_dim)

    def transition_parameters(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        step: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time = torch.full(
            (obs.shape[0], 1),
            float(step) / self.flow_steps,
            device=obs.device,
            dtype=obs.dtype,
        )
        velocity = self.velocity(obs, action, time)
        mean = (action + velocity / self.flow_steps).clamp(-self.action_scale, self.action_scale)
        std = self.noise_std(obs, time)
        return mean, std, velocity

    @torch.no_grad()
    def sample_chain(
        self,
        obs: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch = obs.shape[0]
        if deterministic:
            action = torch.zeros(
                batch, self.horizon, self.action_dim, device=obs.device, dtype=obs.dtype
            )
        else:
            action = torch.randn(
                batch, self.horizon, self.action_dim, device=obs.device, dtype=obs.dtype
            )
        chains = torch.empty(
            batch,
            self.flow_steps + 1,
            self.horizon,
            self.action_dim,
            device=obs.device,
            dtype=obs.dtype,
        )
        chains[:, 0] = action
        for step in range(self.flow_steps):
            mean, std, _ = self.transition_parameters(obs, action, step)
            if deterministic:
                action = mean
            else:
                epsilon = torch.randn_like(action).clamp(
                    -self.randn_clip_value, self.randn_clip_value
                )
                action = mean + std.detach() * epsilon
            if step == self.flow_steps - 1:
                action = action.clamp(-self.action_scale, self.action_scale)
            chains[:, step + 1] = action
        log_prob, entropy, stats = self.chain_log_prob(obs, chains)
        return action, chains, log_prob, {"entropy": entropy, **stats}

    def chain_log_prob(
        self,
        obs: torch.Tensor,
        chains: torch.Tensor,
        *,
        account_for_initial_stochasticity: bool = True,
        normalize_denoising_horizon: bool = True,
        normalize_action_dimension: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        batch = obs.shape[0]
        if chains.shape != (
            batch,
            self.flow_steps + 1,
            self.horizon,
            self.action_dim,
        ):
            raise ValueError(f"Unexpected ReinFlow chain shape {tuple(chains.shape)}")

        log_prob = torch.zeros(batch, device=obs.device, dtype=obs.dtype)
        entropy = torch.zeros_like(log_prob)
        counted_steps = 0
        if account_for_initial_stochasticity:
            initial_dist = Normal(torch.zeros_like(chains[:, 0]), torch.ones_like(chains[:, 0]))
            log_prob = log_prob + initial_dist.log_prob(chains[:, 0]).sum(dim=(-2, -1))
            entropy = entropy + initial_dist.entropy().sum(dim=(-2, -1))
            counted_steps += 1

        transition_log_probs: list[torch.Tensor] = []
        transition_entropies: list[torch.Tensor] = []
        transition_stds: list[torch.Tensor] = []
        velocity_rms: list[torch.Tensor] = []
        for step in range(self.flow_steps):
            mean, std, velocity = self.transition_parameters(obs, chains[:, step], step)
            transition_dist = Normal(mean, std)
            step_log_prob = transition_dist.log_prob(chains[:, step + 1]).sum(dim=(-2, -1))
            step_entropy = transition_dist.entropy().sum(dim=(-2, -1))
            log_prob = log_prob + step_log_prob
            entropy = entropy + step_entropy
            transition_log_probs.append(step_log_prob)
            transition_entropies.append(step_entropy)
            transition_stds.append(std.mean(dim=(-2, -1)))
            velocity_rms.append(velocity.pow(2).mean(dim=(-2, -1)).sqrt())
            counted_steps += 1

        if normalize_denoising_horizon:
            log_prob = log_prob / counted_steps
            entropy = entropy / counted_steps
        if normalize_action_dimension:
            log_prob = log_prob / self.chunk_dim
            entropy = entropy / self.chunk_dim
        stats = {
            "transition_log_probs": torch.stack(transition_log_probs, dim=1),
            "transition_entropies": torch.stack(transition_entropies, dim=1),
            "transition_stds": torch.stack(transition_stds, dim=1),
            "velocity_rms": torch.stack(velocity_rms, dim=1),
        }
        return log_prob, entropy, stats
