from __future__ import annotations

import torch
from torch import nn


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


def _build_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int, activation: str) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(last, h))
        layers.append(_activation(activation))
        last = h
    layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class FlowMatchingPolicy(nn.Module):
    """One H-frame Flow mean and one shared per-frame joint covariance."""

    # Bounding the lower-triangular perturbation keeps every diagonal of
    # I + perturbation positive and prevents an ill-conditioned covariance.
    JOINT_CHOLESKY_SHAPE_RADIUS = 0.5

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 29,
        horizon: int = 1,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon

        self.chunk_dim = horizon * action_dim
        self.obs_dim = obs_dim
        hidden_dims = tuple(hidden_dims)
        if not hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        self.velocity_net = _build_mlp(
            self.obs_dim + self.chunk_dim + 1,
            hidden_dims,
            self.chunk_dim,
            activation,
        )
        output_layer = self.velocity_net[-1]
        if not isinstance(output_layer, nn.Linear):
            raise TypeError("velocity_net must end in nn.Linear")
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)

        lower = torch.tril_indices(self.action_dim, self.action_dim)
        self.joint_cholesky_raw = nn.Parameter(torch.zeros(lower.shape[1]))
        self.register_buffer("_joint_lower_indices", lower, persistent=False)

    def velocity_field(
        self,
        observation: torch.Tensor,
        flow_state: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        return self.velocity_net(
            torch.cat([observation, flow_state, time.unsqueeze(-1)], dim=-1)
        )

    def joint_cholesky_shape(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Materialize the shared A x A bounded Cholesky shape."""
        raw = self.joint_cholesky_raw.to(device=device, dtype=dtype)
        if not bool(torch.isfinite(raw).all()):
            raise FloatingPointError("CPS Cholesky parameters are non-finite")
        lower = self._joint_lower_indices.to(device=device)
        radius = float(self.JOINT_CHOLESKY_SHAPE_RADIUS)
        raw_norm = torch.linalg.vector_norm(raw)
        projection = torch.clamp(
            torch.as_tensor(radius, device=device, dtype=dtype)
            / raw_norm.clamp(min=torch.finfo(dtype).eps),
            max=1.0,
        )
        perturbation = torch.zeros(
            self.action_dim,
            self.action_dim,
            device=device,
            dtype=dtype,
        )
        perturbation[lower[0], lower[1]] = raw * projection
        chol = torch.eye(
            self.action_dim,
            device=device,
            dtype=dtype,
        ) + perturbation
        diagonal = torch.diagonal(chol)
        if not bool(torch.isfinite(chol).all()) or not bool((diagonal > 0.0).all()):
            raise FloatingPointError(
                "CPS Cholesky shape must be finite with a positive diagonal"
            )
        return chol
