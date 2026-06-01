from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

MIN_POLICY_OBS_DIM = 154


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


class FlowMatchingPolicy(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 29,
        horizon: int = 1,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        action_squash_scale: float = 5.0,
        basis_count: int = 0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        # Temporal trajectory prior: the flow operates in a COEFFICIENT latent space of
        # `basis_count` low-frequency modes per joint, expanded to the `horizon`-frame action
        # chunk by a fixed temporal basis. This constrains executed chunks to the smooth
        # trajectory manifold so an iid white-noise latent cannot produce high-frequency
        # (jittery) chunks. basis_count <= 0 OR basis_count == horizon with the identity basis
        # reproduces the legacy flat per-frame parametrization byte-for-byte.
        if basis_count is None or basis_count <= 0:
            basis_count = horizon
        self.basis_count = int(min(max(1, basis_count), horizon))
        # latent / flow / log_prob all live in coefficient space (chunk_dim keeps its meaning
        # as "the dimensionality the flow + transition log_prob operate on").
        self.chunk_dim = self.basis_count * action_dim
        self.action_chunk_dim = horizon * action_dim
        self.obs_dim = max(obs_dim, MIN_POLICY_OBS_DIM)
        self.hidden_dims = tuple(hidden_dims)
        self.activation = activation
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        layers: list[nn.Module] = []
        in_dim = self.obs_dim + self.chunk_dim + 1
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(_activation(activation))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.chunk_dim))
        self.velocity_net = nn.Sequential(*layers)
        self.init_noise_std = float(init_noise_std)
        if action_squash_scale <= 0.0:
            raise ValueError(f"action_squash_scale must be > 0, got {action_squash_scale}")
        self.action_squash_scale = float(action_squash_scale)
        # Fixed temporal basis B: (horizon, basis_count). chunk[t] = sum_k B[t,k] * coeff[k].
        # Identity when basis_count == horizon (legacy flat parametrization, byte-identical).
        self.register_buffer("temporal_basis", self._build_temporal_basis(horizon, self.basis_count))

    @staticmethod
    def _build_temporal_basis(horizon: int, basis_count: int) -> torch.Tensor:
        if basis_count == horizon:
            return torch.eye(horizon, dtype=torch.float32)
        # Low-frequency DCT-II modes (k = 0..basis_count-1), columns scaled so that an
        # iid N(0,1) coefficient vector maps to a per-frame action with unit-ish variance
        # (keeps the action magnitude comparable to the legacy flat parametrization).
        t = torch.arange(horizon, dtype=torch.float32).unsqueeze(1)  # (horizon, 1)
        k = torch.arange(basis_count, dtype=torch.float32).unsqueeze(0)  # (1, basis_count)
        basis = torch.cos(torch.pi * (t + 0.5) * k / horizon)  # (horizon, basis_count)
        # Normalize columns to unit norm so each coefficient contributes comparable energy.
        basis = basis / basis.norm(dim=0, keepdim=True).clamp(min=1e-6)
        # Scale so that summing basis_count unit-variance coeffs gives ~unit per-frame std.
        basis = basis * (horizon ** 0.5) / (basis_count ** 0.5)
        return basis

    def _prepare_observation(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] == self.obs_dim:
            return observation
        if observation.shape[-1] > self.obs_dim:
            return observation[..., : self.obs_dim]
        return F.pad(observation, (0, self.obs_dim - observation.shape[-1]))

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
        net_input = torch.cat([observation, noisy_actions, time.unsqueeze(-1)], dim=-1)
        return self.velocity_net(net_input)

    def _action_transform(self, action_value: torch.Tensor) -> torch.Tensor:
        scale = self.action_squash_scale
        if self.basis_count == self.horizon:
            # Identity basis: legacy flat per-frame parametrization (byte-identical).
            return scale * torch.tanh(action_value / scale)
        # action_value is the coefficient latent: (..., basis_count * action_dim).
        leading_shape = action_value.shape[:-1]
        coeff = action_value.view(*leading_shape, self.basis_count, self.action_dim)
        # Expand to per-frame chunk via fixed temporal basis: (horizon, basis_count) @ coeff.
        basis = self.temporal_basis.to(dtype=coeff.dtype)  # (horizon, basis_count)
        chunk = torch.einsum("hk,...ka->...ha", basis, coeff)  # (..., horizon, action_dim)
        squashed = scale * torch.tanh(chunk / scale)
        return squashed.reshape(*leading_shape, self.action_chunk_dim)
