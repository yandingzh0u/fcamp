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


class ActionConditionedChunkCritic(nn.Module):
    """Action-conditioned multi-horizon critic for SFPO.

    Produces two outputs from a chunk-start state ``s`` and an action chunk
    ``[a_0, ..., a_{h-1}]``:

    * ``V(s)``: a state-only baseline (no action dependence).
    * ``Q_prefix(s, action_chunk) -> [Q_1, ..., Q_h]``: ``Q_k`` is the value of
      executing the prefix ``[a_0, ..., a_{k-1}]`` from ``s``. Prefixes are
      causal by construction (prefix ``k`` only consumes frames ``0..k-1``).

    Architecture (MLP-based, no transformer):

        obs_encoder(s)             -> h_obs          (shared)
        frame_encoder(a_i)         -> h_frame_i      (shared across frames)
        h_prefix_k = cumsum(h_frame_0..k-1) + prefix_pos_embed[k]
        Q_k = q_head([h_obs, h_prefix_k])
        V(s) = v_head(h_obs)

    The cumulative sum over per-frame encodings is the simplest causal prefix
    aggregation for small horizons (h=4); ``prefix_pos_embed`` lets the head
    distinguish prefix lengths.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        hidden_dims: tuple[int, ...],
        activation: str,
    ):
        super().__init__()
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.horizon = horizon
        hidden = int(hidden_dims[-1])

        self.obs_encoder = _build_mlp(obs_dim, tuple(hidden_dims), hidden, activation)
        self.frame_encoder = _build_mlp(action_dim, tuple(hidden_dims), hidden, activation)
        self.prefix_pos_embed = nn.Parameter(torch.zeros(horizon, hidden))
        nn.init.normal_(self.prefix_pos_embed, std=0.02)
        self.q_head = _build_mlp(hidden * 2, tuple(hidden_dims), 1, activation)
        self.v_head = _build_mlp(hidden, tuple(hidden_dims), 1, activation)

    def evaluate_v(self, obs: torch.Tensor) -> torch.Tensor:
        """State-only value ``V(s)`` -> ``[batch, 1]``."""
        h_obs = self.obs_encoder(obs)
        return self.v_head(h_obs)

    def evaluate_q_prefix(self, obs: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Per-prefix action-conditioned values ``[Q_1, ..., Q_h]`` -> ``[batch, horizon]``.

        ``action_chunk`` has shape ``[batch, horizon, action_dim]``.
        """
        if action_chunk.ndim != 3:
            raise ValueError(
                f"action_chunk must be [batch, horizon, action_dim], got {tuple(action_chunk.shape)}"
            )
        if action_chunk.shape[1] != self.horizon or action_chunk.shape[2] != self.action_dim:
            raise ValueError(
                f"action_chunk expected [..., {self.horizon}, {self.action_dim}], "
                f"got {tuple(action_chunk.shape)}"
            )
        h_obs = self.obs_encoder(obs)  # [B, H]
        h_frame = self.frame_encoder(action_chunk)  # [B, h, H]
        # causal prefix aggregation: prefix k aggregates frames 0..k-1
        h_prefix = torch.cumsum(h_frame, dim=1)  # [B, h, H]
        h_prefix = h_prefix + self.prefix_pos_embed.unsqueeze(0)  # [B, h, H]
        h_obs_exp = h_obs.unsqueeze(1).expand(-1, self.horizon, -1)  # [B, h, H]
        h_cat = torch.cat([h_obs_exp, h_prefix], dim=-1)  # [B, h, 2H]
        q = self.q_head(h_cat).squeeze(-1)  # [B, h]
        return q


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
