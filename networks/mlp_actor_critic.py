from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal


def _activation(name: str) -> nn.Module:
    return {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh, "silu": nn.SiLU}[name.lower()]()


def _build_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, h))
        layers.append(_activation(activation))
        last = h
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):


    def __init__(self, obs_dim: int, action_dim: int, hidden_dims, activation: str, init_noise_std: float):
        super().__init__()
        self.net = _build_mlp(obs_dim, tuple(hidden_dims), action_dim, activation)
        self.std = nn.Parameter(init_noise_std * torch.ones(action_dim))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs: torch.Tensor) -> None:
        mean = self.net(obs)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden_dims, activation: str):
        super().__init__()
        self.net = _build_mlp(obs_dim, tuple(hidden_dims), 1, activation)

    def evaluate(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class EmpiricalNormalization(nn.Module):


    def __init__(self, shape: int, device, eps: float = 1e-2, until: int | None = None):
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0).to(device))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0).to(device))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long).to(device))

    @torch.no_grad()
    def forward(self, x: torch.Tensor, center: bool = True, update: bool = True) -> torch.Tensor:
        if self.training and update:
            self._update(x)
        if center:
            return (x - self._mean) / (self._std + self.eps)
        return x / (self._std + self.eps)

    @torch.no_grad()
    def _update(self, x: torch.Tensor) -> None:
        if self.until is not None and self.count >= self.until:
            return
        batch_size = x.shape[0]
        batch_mean = torch.mean(x, dim=0, keepdim=True)
        batch_var = torch.var(x, dim=0, keepdim=True, unbiased=False)
        new_count = self.count + batch_size


        delta = batch_mean - self._mean
        self._mean.copy_(self._mean + delta * (batch_size / new_count))
        m_a = self._var * self.count
        m_b = batch_var * batch_size
        M2 = m_a + m_b + delta.pow(2) * (self.count * batch_size / new_count)
        self._var.copy_(M2 / new_count)
        self._std.copy_(self._var.sqrt())
        self.count.copy_(new_count)
