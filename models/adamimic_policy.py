from __future__ import annotations

import copy

import torch
from torch import nn
from torch.distributions import Normal


def _activation(name: str) -> nn.Module:
    table = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
        "mish": nn.Mish,
    }
    key = name.lower()
    if key not in table:
        raise KeyError(f"Unsupported activation {name!r}")
    return table[key]()


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    last_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, int(hidden_dim)))
        layers.append(_activation(activation))
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


class AdaMimicActorCritic(nn.Module):
    """AdaMimic-style two-level Gaussian policy.

    The policy samples a continuous keyframe-time action first, then conditions
    the low-level joint action on that sampled time.  In this repo the current
    G1 environment still advances one reference frame per `env.step`, so the
    method executes only the control part and trains the time action as a
    high-level auxiliary policy head.
    """

    def __init__(
        self,
        *,
        actor_obs_dim: int,
        critic_obs_dim: int,
        control_action_dim: int,
        actor_hidden_dims: tuple[int, ...],
        critic_hidden_dims: tuple[int, ...],
        activation: str,
        init_noise_std: float,
        infer_keyframe_time: bool,
        actor_time_scale_range: tuple[float, float],
        fixed_dt: float,
        time_min_std: float,
        residual_delta: bool = False,
        residual_time_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        self.actor_obs_dim = int(actor_obs_dim)
        self.critic_obs_dim = int(critic_obs_dim)
        self.control_action_dim = int(control_action_dim)
        self.action_dim = self.control_action_dim + 1
        self.infer_keyframe_time = bool(infer_keyframe_time)
        self.actor_time_scale_low = float(actor_time_scale_range[0])
        self.actor_time_scale_high = float(actor_time_scale_range[1])
        self.actor_time_scale = self.actor_time_scale_high - self.actor_time_scale_low
        self.fixed_dt = float(fixed_dt)
        self.time_min_std = float(time_min_std)
        self.residual_delta = bool(residual_delta)
        self.residual_time_threshold = float(residual_time_threshold)

        self.actor = _mlp(
            self.actor_obs_dim + 1,
            actor_hidden_dims,
            self.control_action_dim,
            activation,
        )
        self.actor_delta = copy.deepcopy(self.actor) if self.residual_delta else None
        if self.infer_keyframe_time:
            self.actor_time = nn.Sequential(
                *_mlp(self.actor_obs_dim, actor_hidden_dims, 1, activation),
                nn.Sigmoid(),
            )
        else:
            self.actor_time = None

        self.critic_low = _mlp(
            self.critic_obs_dim + 1,
            critic_hidden_dims,
            1,
            activation,
        )
        self.critic_delta = copy.deepcopy(self.critic_low) if self.residual_delta else None
        self.critic_high = _mlp(
            self.critic_obs_dim,
            critic_hidden_dims,
            1,
            activation,
        )

        self.std = nn.Parameter(float(init_noise_std) * torch.ones(self.control_action_dim))
        self.distribution: Normal | None = None
        self.distribution_time: Normal | None = None
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None or self.distribution_time is None:
            raise RuntimeError("Distribution has not been initialized")
        return torch.cat([self.distribution.mean, self.distribution_time.mean], dim=-1)

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None or self.distribution_time is None:
            raise RuntimeError("Distribution has not been initialized")
        return torch.cat([self.distribution.stddev, self.distribution_time.stddev], dim=-1)

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None or self.distribution_time is None:
            raise RuntimeError("Distribution has not been initialized")
        return self.distribution.entropy().sum(dim=-1)

    def _time_mean(self, obs: torch.Tensor) -> torch.Tensor:
        if self.infer_keyframe_time and self.actor_time is not None:
            delta_time = self.actor_time(obs) * self.actor_time_scale + self.actor_time_scale_low
            return delta_time + self.fixed_dt
        return torch.full((obs.shape[0], 1), self.fixed_dt, device=obs.device, dtype=obs.dtype)

    def _time_std(self, reference: torch.Tensor) -> torch.Tensor:
        std = max(abs(self.actor_time_scale) / 2.0, self.time_min_std)
        return reference * 0.0 + std

    def update_distribution_high(self, obs: torch.Tensor) -> None:
        time_mean = self._time_mean(obs)
        self.distribution_time = Normal(time_mean, self._time_std(time_mean))

    def update_distribution_low(self, obs: torch.Tensor, action_time: torch.Tensor) -> None:
        low_input = torch.cat([obs, action_time.detach()], dim=-1)
        action_mean = self.actor(low_input)
        if self.actor_delta is not None:
            delta_time = action_time - self.fixed_dt
            mask = delta_time.abs() < self.residual_time_threshold
            delta_time = torch.where(mask, torch.zeros_like(delta_time), delta_time)
            action_mean = action_mean + self.actor_delta(low_input) * delta_time
        self.distribution = Normal(action_mean, action_mean * 0.0 + self.std)

    def update_distribution(self, obs: torch.Tensor) -> None:
        self.update_distribution_high(obs)
        assert self.distribution_time is not None
        time_action = self.distribution_time.mean
        self.update_distribution_low(obs, time_action)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        self.update_distribution_high(obs)
        assert self.distribution_time is not None
        low = self.fixed_dt + self.actor_time_scale_low
        high = self.fixed_dt + self.actor_time_scale_high
        if high < low:
            low, high = high, low
        action_time = self.distribution_time.sample().clamp(low, high)
        self.update_distribution_low(obs, action_time)
        assert self.distribution is not None
        control_action = self.distribution.sample()
        return torch.cat([control_action, action_time], dim=-1)

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        action_time = self._time_mean(obs)
        low_input = torch.cat([obs, action_time], dim=-1)
        control_action = self.actor(low_input)
        if self.actor_delta is not None:
            delta_time = action_time - self.fixed_dt
            mask = delta_time.abs() < self.residual_time_threshold
            delta_time = torch.where(mask, torch.zeros_like(delta_time), delta_time)
            control_action = control_action + self.actor_delta(low_input) * delta_time
        return torch.cat([control_action, action_time], dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.distribution is None or self.distribution_time is None:
            raise RuntimeError("Distribution has not been initialized")
        control_logp = self.distribution.log_prob(actions[:, :-1]).sum(dim=-1)
        time_logp = self.distribution_time.log_prob(actions[:, -1:]).sum(dim=-1)
        return control_logp, time_logp

    def evaluate_low(self, critic_obs: torch.Tensor, action_time: torch.Tensor) -> torch.Tensor:
        critic = self.critic_delta if self.critic_delta is not None else self.critic_low
        return critic(torch.cat([critic_obs, action_time], dim=-1)).squeeze(-1)

    def evaluate_high(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic_high(critic_obs).squeeze(-1)

    def freeze_base_actor(self) -> None:
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        for parameter in self.critic_low.parameters():
            parameter.requires_grad_(False)
