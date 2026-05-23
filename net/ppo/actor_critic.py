from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


def _make_mlp(input_dim: int, output_dim: int, hidden_dims: tuple[int, ...], activation: str) -> nn.Sequential:
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive, got {input_dim}")
    if output_dim <= 0:
        raise ValueError(f"output_dim must be positive, got {output_dim}")
    if not hidden_dims:
        raise ValueError("hidden_dims must contain at least one layer")

    layers: list[nn.Module] = []
    in_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(_activation(activation))
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


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


class GaussianActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        actor_hidden_dims: tuple[int, ...] = (512, 256, 128),
        critic_hidden_dims: tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.critic_obs_dim = int(critic_obs_dim)
        self.action_dim = int(action_dim)
        self.actor = _make_mlp(self.obs_dim, self.action_dim, actor_hidden_dims, activation)
        self.critic = _make_mlp(self.critic_obs_dim, 1, critic_hidden_dims, activation)
        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(float(init_noise_std) * torch.ones(self.action_dim))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(float(init_noise_std) * torch.ones(self.action_dim)))
        else:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution has not been initialized.")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution has not been initialized.")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution has not been initialized.")
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        del dones

    def update_distribution(self, observation: torch.Tensor) -> None:
        mean = self.actor(observation)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observation: torch.Tensor, critic_observation: torch.Tensor) -> dict[str, torch.Tensor]:
        self.update_distribution(observation)
        actions = self.distribution.sample()
        return {
            "actions": actions,
            "values": self.evaluate(critic_observation),
            "log_probs": self.get_actions_log_prob(actions),
            "mean": self.action_mean,
            "sigma": self.action_std,
        }

    def act_inference(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor(observation)

    def evaluate(self, critic_observation: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_observation).squeeze(-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Action distribution has not been initialized.")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate_actions(
        self,
        observation: torch.Tensor,
        critic_observation: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.update_distribution(observation)
        log_prob = self.get_actions_log_prob(actions)
        value = self.evaluate(critic_observation)
        entropy = self.entropy
        return log_prob, value, entropy, self.action_mean, self.action_std
