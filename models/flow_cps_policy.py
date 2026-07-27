from __future__ import annotations

import torch
from torch import nn

from components.nn import build_mlp


class FlowMatchingPolicy(nn.Module):
    # The learnable covariance is a shape around the identity.  Bounding the
    # Frobenius norm of its Cholesky perturbation makes the parameterization
    # identifiable after physical trace normalization and guarantees
    #   singular_values(L_shape) in [1-r, 1+r].
    # With r=0.5 the covariance condition number is therefore at most 9.
    CPS_CHOLESKY_SHAPE_RADIUS = 0.5

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 29,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.hidden_dims = tuple(hidden_dims)
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")

        # One position-free velocity field is shared by every planned token.
        # Sequence construction belongs to the caller, which can feed the
        # previously generated command state back through the observation
        # without introducing a chunk-offset parameter or recurrent reset.
        hidden = int(hidden_dims[-1])
        self.obs_encoder = build_mlp(self.obs_dim + 1, tuple(hidden_dims), hidden, activation)
        self.token_encoder = build_mlp(
            self.action_dim,
            tuple(hidden_dims),
            hidden,
            activation,
        )
        self.vel_head = build_mlp(hidden * 2, tuple(hidden_dims), self.action_dim, activation)

        # Exploration covariance is shared by every token. Cross-token
        # dependence is represented by the planned command state, never by an
        # H-specific covariance whose correlation disappears at chunk edges.
        lower = torch.tril_indices(self.action_dim, self.action_dim)
        diagonal = lower[0] == lower[1]
        self.cps_cholesky_raw = nn.Parameter(torch.zeros(lower.shape[1]))
        self.register_buffer("_cps_lower_indices", lower, persistent=False)
        self.register_buffer("_cps_diagonal_mask", diagonal, persistent=False)

    def _prepare_observation(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Expected observation dim {self.obs_dim}, got {observation.shape[-1]}"
            )
        return observation

    def _validate_inputs(self, observation: torch.Tensor, flow_token: torch.Tensor, steps: int) -> None:
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if flow_token.ndim != 2 or flow_token.shape[-1] != self.action_dim:
            raise ValueError(
                "flow_token must have shape "
                f"[B,{self.action_dim}], got {tuple(flow_token.shape)}"
            )
        if observation.ndim != 2:
            raise ValueError(
                f"observation must have shape [B,{self.obs_dim}], got {tuple(observation.shape)}"
            )
        if observation.shape[0] != flow_token.shape[0]:
            raise ValueError(
                f"Observation batch size {observation.shape[0]} must match "
                f"Flow token batch size {flow_token.shape[0]}"
            )

    def velocity_field(
        self,
        observation: torch.Tensor,
        flow_token: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        observation = self._prepare_observation(observation)
        if time.ndim != 1 or time.shape[0] != observation.shape[0]:
            raise ValueError(f"time must have shape ({observation.shape[0]},), got {tuple(time.shape)}")
        obs_h = self.obs_encoder(torch.cat([observation, time.unsqueeze(-1)], dim=-1))
        token_h = self.token_encoder(flow_token)
        return self.vel_head(torch.cat([obs_h, token_h], dim=-1))

    def raw_cps_cholesky(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Materialize a full, bounded, positive-diagonal covariance shape."""
        raw = self.cps_cholesky_raw.to(device=device, dtype=dtype)
        lower = self._cps_lower_indices.to(device=device)
        radius = float(self.CPS_CHOLESKY_SHAPE_RADIUS)
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
        return torch.eye(
            self.action_dim,
            device=device,
            dtype=dtype,
        ) + perturbation
