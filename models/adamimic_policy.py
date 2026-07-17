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
        layers.extend((nn.Linear(last_dim, int(hidden_dim)), _activation(activation)))
        last_dim = int(hidden_dim)
    layers.append(nn.Linear(last_dim, int(output_dim)))
    return nn.Sequential(*layers)


class AdaMimicActorCritic(nn.Module):
    """Exact TrackActorCritic/TrackActorCriticDelta topology used by AdaMimic."""

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
        num_critics: int = 2,
        residual_delta: bool = False,
        residual_time_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        if not infer_keyframe_time:
            raise ValueError("AdaMimic requires infer_keyframe_time=true")
        if num_critics != 2:
            raise ValueError("Official AdaMimic uses exactly two reward groups/critics")

        self.actor_obs_dim = int(actor_obs_dim)
        self.critic_obs_dim = int(critic_obs_dim)
        self.control_action_dim = int(control_action_dim)
        self.action_dim = self.control_action_dim + 1
        self.num_critics = int(num_critics)
        self.infer_keyframe_time = True
        self.actor_time_scale_low = float(actor_time_scale_range[0])
        self.actor_time_scale_high = float(actor_time_scale_range[1])
        self.actor_time_scale = self.actor_time_scale_high - self.actor_time_scale_low
        if self.actor_time_scale < 0.0:
            raise ValueError("actor_time_scale_range must be ordered [low, high]")
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
        # Official Delta constructs these copies before loading the stage-1 base.
        self.actor_delta = copy.deepcopy(self.actor) if self.residual_delta else None
        self.actor_time = nn.Sequential(
            *_mlp(self.actor_obs_dim, actor_hidden_dims, 1, activation),
            nn.Sigmoid(),
        )

        self.critics = nn.ModuleList(
            _mlp(self.critic_obs_dim + 1, critic_hidden_dims, 1, activation)
            for _ in range(self.num_critics)
        )
        self.critics_delta = copy.deepcopy(self.critics) if self.residual_delta else None
        self.critics_time = nn.ModuleList(
            _mlp(self.critic_obs_dim, critic_hidden_dims, 1, activation)
            for _ in range(self.num_critics)
        )

        self.std = nn.Parameter(float(init_noise_std) * torch.ones(self.control_action_dim))
        self.distribution: Normal | None = None
        self.distribution_time: Normal | None = None
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None or self.distribution_time is None:
            raise RuntimeError("Distribution has not been initialized")
        return torch.cat((self.distribution.mean, self.distribution_time.mean), dim=-1)

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None or self.distribution_time is None:
            raise RuntimeError("Distribution has not been initialized")
        return torch.cat((self.distribution.stddev, self.distribution_time.stddev), dim=-1)

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Distribution has not been initialized")
        # AdaMimic deliberately excludes the time distribution entropy.
        return self.distribution.entropy().sum(dim=-1)

    def _time_mean(self, obs: torch.Tensor) -> torch.Tensor:
        delta = self.actor_time(obs) * self.actor_time_scale + self.actor_time_scale_low
        return delta + self.fixed_dt

    def _time_std(self, reference: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(reference) + max(self.actor_time_scale / 2.0, self.time_min_std)

    def update_distribution_high(self, obs: torch.Tensor) -> None:
        time_mean = self._time_mean(obs)
        self.distribution_time = Normal(time_mean, self._time_std(time_mean))

    def _control_mean(
        self,
        obs: torch.Tensor,
        action_time: torch.Tensor,
        *,
        detach_time: bool,
    ) -> torch.Tensor:
        actor_time = action_time.detach() if detach_time else action_time
        actor_input = torch.cat((obs, actor_time), dim=-1)
        action_mean = self.actor(actor_input)
        if self.actor_delta is not None:
            dt = action_time - self.fixed_dt
            dt = torch.where(dt.abs() < self.residual_time_threshold, torch.zeros_like(dt), dt)
            action_mean = action_mean + self.actor_delta(actor_input) * dt
        return action_mean

    def update_distribution_low(self, obs: torch.Tensor, action_time: torch.Tensor) -> None:
        action_mean = self._control_mean(obs, action_time, detach_time=True)
        self.distribution = Normal(action_mean, torch.zeros_like(action_mean) + self.std)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        self.update_distribution_high(obs)
        assert self.distribution_time is not None
        action_time = self.distribution_time.sample().clamp(
            self.fixed_dt + self.actor_time_scale_low,
            self.fixed_dt + self.actor_time_scale_high,
        )
        self.update_distribution_low(obs, action_time)
        assert self.distribution is not None
        return torch.cat((self.distribution.sample(), action_time), dim=-1)

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        action_time = self._time_mean(obs)
        return torch.cat((self._control_mean(obs, action_time, detach_time=False), action_time), dim=-1)

    def get_actions_log_prob(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.distribution is None or self.distribution_time is None:
            raise RuntimeError("Distribution has not been initialized")
        return (
            self.distribution.log_prob(actions[:, :-1]).sum(dim=-1),
            self.distribution_time.log_prob(actions[:, -1:]).sum(dim=-1),
        )

    def evaluate_low(
        self,
        critic_obs: torch.Tensor,
        action_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        critic_input = torch.cat((critic_obs, action_time), dim=-1) if action_time is not None else critic_obs
        if critic_input.shape[-1] != self.critic_obs_dim + 1:
            raise ValueError(
                f"low critic expects {self.critic_obs_dim + 1} features, got {critic_input.shape[-1]}"
            )
        critics = self.critics_delta if self.critics_delta is not None else self.critics
        return torch.cat(tuple(critic(critic_input) for critic in critics), dim=-1)

    def evaluate_high(self, critic_obs: torch.Tensor) -> torch.Tensor:
        if critic_obs.shape[-1] != self.critic_obs_dim:
            raise ValueError(f"high critic expects {self.critic_obs_dim} features, got {critic_obs.shape[-1]}")
        return torch.cat(tuple(critic(critic_obs) for critic in self.critics_time), dim=-1)

    def freeze_base_actor(self) -> None:
        for parameter in self.actor.parameters():
            parameter.requires_grad_(False)
        for parameter in self.critics.parameters():
            parameter.requires_grad_(False)
