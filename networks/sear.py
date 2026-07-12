from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SphericalResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = self.linear2(F.relu(self.linear1(self.norm(hidden))))
        hidden = hidden + residual
        return F.normalize(hidden, dim=-1) * math.sqrt(hidden.shape[-1])


class SEARActor(nn.Module):
    """SimbaV2-style squashed Gaussian actor over a complete action chunk."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int,
        num_blocks: int,
        action_scale: float,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.action_scale = float(action_scale)
        self.input_norm = nn.LayerNorm(obs_dim)
        self.input = nn.Linear(obs_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            SphericalResidualBlock(hidden_dim) for _ in range(num_blocks)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.mean = nn.Linear(hidden_dim, self.horizon * self.action_dim)
        self.log_std = nn.Linear(hidden_dim, self.horizon * self.action_dim)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.weight)
        nn.init.zeros_(self.log_std.bias)

    def distribution_parameters(
        self, observation: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.input(self.input_norm(observation)))
        hidden = F.normalize(hidden, dim=-1) * math.sqrt(hidden.shape[-1])
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.output_norm(hidden)
        mean = self.mean(hidden).view(-1, self.horizon, self.action_dim)
        log_std = self.log_std(hidden).view(-1, self.horizon, self.action_dim)
        log_std = -5.0 + 3.5 * (torch.tanh(log_std) + 1.0)
        return mean, log_std

    def sample(
        self,
        observation: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution_parameters(observation)
        std = log_std.exp()
        pre_tanh = mean if deterministic else mean + std * torch.randn_like(mean)
        squashed = torch.tanh(pre_tanh)
        action = self.action_scale * squashed
        distribution = torch.distributions.Normal(mean, std)
        log_prob = distribution.log_prob(pre_tanh)
        log_prob = log_prob - torch.log(
            self.action_scale * (1.0 - squashed.square()) + 1.0e-6
        )
        return action, log_prob.sum(dim=-1)


class CausalDistributionalCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        num_bins: int,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("SEAR critic hidden_dim must be divisible by num_heads")
        self.horizon = int(horizon)
        self.obs_encoder = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(horizon, hidden_dim))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_blocks,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.value_head = nn.Linear(hidden_dim, num_bins)
        causal_mask = torch.triu(
            torch.ones(horizon, horizon, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)

    def forward(
        self, observation: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        if action_chunk.shape[1] != self.horizon:
            raise ValueError(
                f"Expected SEAR chunk horizon {self.horizon}, got {action_chunk.shape[1]}"
            )
        context = self.obs_encoder(observation).unsqueeze(1)
        tokens = self.action_encoder(action_chunk) + context + self.position.unsqueeze(0)
        hidden = self.transformer(tokens, mask=self.causal_mask)
        return self.value_head(self.output_norm(hidden))


class SEARTwinCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dim: int,
        num_heads: int,
        num_blocks: int,
        num_bins: int,
        value_min: float,
        value_max: float,
    ) -> None:
        super().__init__()
        kwargs = dict(
            obs_dim=obs_dim,
            action_dim=action_dim,
            horizon=horizon,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_blocks=num_blocks,
            num_bins=num_bins,
        )
        self.q1 = CausalDistributionalCritic(**kwargs)
        self.q2 = CausalDistributionalCritic(**kwargs)
        self.register_buffer("support", torch.linspace(value_min, value_max, num_bins))

    def forward(
        self, observation: torch.Tensor, action_chunk: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(observation, action_chunk), self.q2(observation, action_chunk)

    def expected(self, logits: torch.Tensor) -> torch.Tensor:
        return (torch.softmax(logits, dim=-1) * self.support).sum(dim=-1)

    def target_distribution(self, values: torch.Tensor) -> torch.Tensor:
        clipped = values.clamp(float(self.support[0]), float(self.support[-1]))
        spacing = self.support[1] - self.support[0]
        position = (clipped - self.support[0]) / spacing
        lower = position.floor().long().clamp(0, self.support.numel() - 1)
        upper = (lower + 1).clamp(0, self.support.numel() - 1)
        upper_weight = position - lower.to(position.dtype)
        upper_weight = torch.where(lower == upper, torch.zeros_like(upper_weight), upper_weight)
        distribution = torch.zeros(
            *values.shape,
            self.support.numel(),
            device=values.device,
            dtype=values.dtype,
        )
        distribution.scatter_add_(-1, lower.unsqueeze(-1), (1.0 - upper_weight).unsqueeze(-1))
        distribution.scatter_add_(-1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
        return distribution
