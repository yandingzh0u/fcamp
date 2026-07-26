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
        horizon: int = 1,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        causal_velocity: bool = True,
        causal_arch: str = "prefix_cumsum",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.chunk_dim = horizon * action_dim
        self.obs_dim = obs_dim
        self.hidden_dims = tuple(hidden_dims)
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        if not bool(causal_velocity):
            raise ValueError("FlowMatchingPolicy only supports causal_velocity=True")
        requested_causal_arch = str(causal_arch)
        if requested_causal_arch not in {"causal_gru", "prefix_cumsum"}:
            raise ValueError(f"Unsupported causal_arch: {requested_causal_arch}")
        # ``prefix_cumsum`` is retained as a constructor compatibility alias.
        self.causal_velocity = True
        self.causal_arch = "causal_gru"

        # Ordered causal velocity: v_k depends only on z_0..z_k, while a
        # GRU state preserves the order of that prefix. Token content and
        # position are concatenated *before* nonlinear encoding so the
        # association between z_i and its offset cannot be lost.
        #
        # This makes logp_k a genuine conditional density
        # log pi(z_k | s, z_0..z_{k-1}), so per-frame clipped-ratio is valid.
        hidden = int(hidden_dims[-1])
        self.obs_encoder = build_mlp(self.obs_dim + 1, tuple(hidden_dims), hidden, activation)
        self.frame_pos_embed = nn.Parameter(torch.zeros(self.horizon, hidden))
        nn.init.normal_(self.frame_pos_embed, std=0.02)
        self.token_encoder = build_mlp(
            self.action_dim + hidden,
            tuple(hidden_dims),
            hidden,
            activation,
        )
        self.causal_cell = nn.GRUCell(hidden, hidden)
        self.vel_head = build_mlp(hidden * 2, tuple(hidden_dims), self.action_dim, activation)
        lower = torch.tril_indices(self.chunk_dim, self.chunk_dim)
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

    def _validate_inputs(self, observation: torch.Tensor, flow_state: torch.Tensor, steps: int) -> None:
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")
        if flow_state.shape[-1] != self.chunk_dim:
            raise ValueError(
                f"Expected Flow state dim {self.chunk_dim}, got {flow_state.shape[-1]}"
            )
        if observation.shape[0] != flow_state.shape[0]:
            raise ValueError(
                f"Observation batch size {observation.shape[0]} must match "
                f"Flow state batch size {flow_state.shape[0]}"
            )

    def velocity_field(
        self,
        observation: torch.Tensor,
        flow_state: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        observation = self._prepare_observation(observation)
        if time.ndim != 1 or time.shape[0] != observation.shape[0]:
            raise ValueError(f"time must have shape ({observation.shape[0]},), got {tuple(time.shape)}")
        b = observation.shape[0]
        chunk = flow_state.view(b, self.horizon, self.action_dim)
        obs_h = self.obs_encoder(torch.cat([observation, time.unsqueeze(-1)], dim=-1))
        frame_pos = self.frame_pos_embed.unsqueeze(0).expand(b, -1, -1)
        token_h = self.token_encoder(torch.cat([chunk, frame_pos], dim=-1))
        state = obs_h
        velocity_frames: list[torch.Tensor] = []
        for frame_idx in range(self.horizon):
            state = self.causal_cell(token_h[:, frame_idx], state)
            velocity_frames.append(self.vel_head(torch.cat([obs_h, state], dim=-1)))
        vel = torch.stack(velocity_frames, dim=1)
        return vel.reshape(b, self.chunk_dim)

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
            self.chunk_dim,
            self.chunk_dim,
            device=device,
            dtype=dtype,
        )
        perturbation[lower[0], lower[1]] = raw * projection
        return torch.eye(
            self.chunk_dim,
            device=device,
            dtype=dtype,
        ) + perturbation
