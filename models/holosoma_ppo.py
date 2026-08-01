"""HOLOSOMA's plain PPO actor, critic, normalizer, and rollout storage.

The implementation is intentionally kept structurally identical to the
audited HOLOSOMA G1 WBT PPO at commit
``c5c836c68f423ac4565f57801ff4ff47ea56e5ac``.  Environment adaptation lives
in :mod:`method.fixed_reward`; this module contains only PPO primitives.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.distributions import Normal


class EmpiricalNormalization(nn.Module):
    """Normalize values using HOLOSOMA's exact online update contract."""

    def __init__(self, shape: int, device, eps: float = 1.0e-2, until=None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.device = device
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        center: bool = True,
        update: bool = True,
    ) -> torch.Tensor:
        if x.shape[1:] != self._mean.shape[1:]:
            raise ValueError(
                f"Expected input of shape (*,{self._mean.shape[1:]}), got {x.shape}"
            )
        if self.training and update:
            self.update(x)
        if center:
            return (x - self._mean) / (self._std + self.eps)
        return x / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        if self.until is not None and self.count >= self.until:
            return
        global_batch_size = x.shape[0]
        batch_mean = torch.mean(x, dim=0, keepdim=True)
        batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)
        new_count = self.count + global_batch_size
        delta = batch_mean - self._mean
        self._mean.copy_(
            self._mean + delta * (global_batch_size / new_count)
        )
        delta2 = batch_mean - self._mean
        m_a = self._var * self.count
        m_b = batch_var * global_batch_size
        m2 = (
            m_a
            + m_b
            + delta2.pow(2)
            * (self.count * global_batch_size / new_count)
        )
        self._var.copy_(m2 / new_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(new_count)


def build_mlp(
    input_dim: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    activation: str,
) -> nn.Sequential:
    """Build the exact default-initialized HOLOSOMA MLP."""

    layers: list[nn.Module] = []
    activation_module = getattr(nn, activation)()
    if len(hidden_dims) == 0:
        layers.append(nn.Linear(input_dim, output_dim))
    else:
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation_module)
        for layer_idx in range(len(hidden_dims)):
            if layer_idx == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[layer_idx], output_dim))
            else:
                layers.append(
                    nn.Linear(hidden_dims[layer_idx], hidden_dims[layer_idx + 1])
                )
                layers.append(activation_module)
    return nn.Sequential(*layers)


class PPOActor(nn.Module):
    """Unsquashed diagonal Normal actor used by HOLOSOMA PPO."""

    def __init__(
        self,
        observation_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str,
        num_actions: int,
        init_noise_std: float,
    ) -> None:
        super().__init__()
        self.actor_module = build_mlp(
            observation_dim,
            hidden_dims,
            num_actions,
            activation,
        )
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    @property
    def actor(self) -> nn.Module:
        return self.actor_module

    def reset(self, dones=None) -> None:
        del dones

    @property
    def action_mean(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Actor distribution has not been initialized")
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Actor distribution has not been initialized")
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Actor distribution has not been initialized")
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, actor_obs: torch.Tensor) -> None:
        mean = self.actor(actor_obs)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, actor_obs: torch.Tensor) -> torch.Tensor:
        self.update_distribution(actor_obs)
        if self.distribution is None:  # pragma: no cover - established above
            raise RuntimeError("Actor distribution has not been initialized")
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.distribution is None:
            raise RuntimeError("Actor distribution has not been initialized")
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return self.actor(actor_obs)


class PPOCritic(nn.Module):
    """Scalar MLP critic used by HOLOSOMA PPO."""

    def __init__(
        self,
        observation_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str,
    ) -> None:
        super().__init__()
        self.critic_module = build_mlp(
            observation_dim,
            hidden_dims,
            1,
            activation,
        )

    @property
    def critic(self) -> nn.Module:
        return self.critic_module

    def reset(self, dones=None) -> None:
        del dones

    def evaluate(self, critic_obs: torch.Tensor) -> torch.Tensor:
        return self.critic(critic_obs)


class RolloutStorage:
    """HOLOSOMA's tensor storage and once-per-update minibatch shuffle."""

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.step = 0
        self._buffers: dict[str, Tensor] = {}

    def register(
        self,
        key: str,
        shape: tuple[int, ...] | list[int] = (),
        dtype: torch.dtype = torch.float,
    ) -> None:
        if key in self._buffers:
            raise ValueError(f"Key {key!r} already registered")
        if not isinstance(shape, (list, tuple)):
            raise ValueError("shape must be a list or tuple")
        self._buffers[key] = torch.zeros(
            (self.num_transitions_per_env, self.num_envs, *shape),
            dtype=dtype,
            device=self.device,
        )

    def add(self, **data: Tensor) -> None:
        if self.step >= self.num_transitions_per_env:
            raise RuntimeError(
                f"Buffer overflow: step {self.step} >= {self.num_transitions_per_env}"
            )
        for key, value in data.items():
            if key not in self._buffers:
                continue
            if value.requires_grad:
                raise ValueError(
                    f"Cannot store tensor with requires_grad=True for key {key!r}"
                )
            self._buffers[key][self.step].copy_(value)
        self.step += 1

    def __getitem__(self, key: str) -> Tensor:
        if key not in self._buffers:
            raise KeyError(f"Key {key!r} not registered")
        return self._buffers[key]

    def __setitem__(self, key: str, value: Tensor) -> None:
        if key not in self._buffers:
            raise KeyError(f"Key {key!r} not registered")
        if value.requires_grad:
            raise ValueError("Cannot store tensor with requires_grad=True")
        self._buffers[key].copy_(value)

    def clear(self) -> None:
        self.step = 0

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size,
            requires_grad=False,
            device=self.device,
        )
        flattened = {
            key: buffer.flatten(0, 1)
            for key, buffer in self._buffers.items()
        }
        for _ in range(num_epochs):
            for mini_batch_index in range(num_mini_batches):
                start = mini_batch_index * mini_batch_size
                end = (mini_batch_index + 1) * mini_batch_size
                batch_indices = indices[start:end]
                yield {
                    key: flattened[key][batch_indices]
                    for key in self._buffers
                }
