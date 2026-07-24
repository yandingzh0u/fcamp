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
        causal_velocity: bool = False,
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
        self.causal_velocity = bool(causal_velocity)
        requested_causal_arch = str(causal_arch)
        if self.causal_velocity and requested_causal_arch not in {"causal_gru", "prefix_cumsum"}:
            raise ValueError(f"Unsupported causal_arch: {requested_causal_arch}")
        # ``prefix_cumsum`` is retained as a constructor compatibility alias.
        # The old sum-based implementation was permutation-invariant inside a
        # prefix, so it did not represent an ordered conditional trajectory.
        self.causal_arch = "causal_gru" if self.causal_velocity else requested_causal_arch
        if not self.causal_velocity:
            # Full-chunk MLP: v_k depends on the whole z_0..z_{h-1} (non-causal).
            layers: list[nn.Module] = []
            in_dim = self.obs_dim + self.chunk_dim + 1
            for hidden_dim in self.hidden_dims:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(_activation(activation))
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, self.chunk_dim))
            self.velocity_net = nn.Sequential(*layers)
        else:
            # Ordered causal velocity: v_k depends only on z_0..z_k, while a
            # GRU state preserves the order of that prefix. Token content and
            # position are concatenated *before* nonlinear encoding so the
            # association between z_i and its offset cannot be lost.
            #
            # This makes logp_k a genuine conditional density
            # log pi(z_k | s, z_0..z_{k-1}), so per-frame clipped-ratio is valid.
            #   obs_h    = obs_encoder([obs, time])
            #   token_i  = token_encoder([z_i, frame_pos_i])
            #   state_i  = GRUCell(token_i, state_{i-1}), state_-1 = obs_h
            #   v_i      = vel_head([obs_h, state_i])
            hidden = int(hidden_dims[-1])
            self._causal_hidden = hidden
            self.obs_encoder = _build_mlp(self.obs_dim + 1, tuple(hidden_dims), hidden, activation)
            self.frame_pos_embed = nn.Parameter(torch.zeros(self.horizon, hidden))
            nn.init.normal_(self.frame_pos_embed, std=0.02)
            self.token_encoder = _build_mlp(
                self.action_dim + hidden,
                tuple(hidden_dims),
                hidden,
                activation,
            )
            self.causal_cell = nn.GRUCell(hidden, hidden)
            self.vel_head = _build_mlp(hidden * 2, tuple(hidden_dims), self.action_dim, activation)
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
        if not self.causal_velocity:
            net_input = torch.cat([observation, flow_state, time.unsqueeze(-1)], dim=-1)
            return self.velocity_net(net_input)
        # Ordered causal-GRU velocity: v_k depends only on z_0..z_k.
        # flow_state: [B, chunk_dim] -> [B, h, A]
        b = observation.shape[0]
        chunk = flow_state.view(b, self.horizon, self.action_dim)
        obs_h = self.obs_encoder(torch.cat([observation, time.unsqueeze(-1)], dim=-1))  # [B, H]
        frame_pos = self.frame_pos_embed.unsqueeze(0).expand(b, -1, -1)
        token_h = self.token_encoder(torch.cat([chunk, frame_pos], dim=-1))  # [B, h, H]
        state = obs_h
        velocity_frames: list[torch.Tensor] = []
        for frame_idx in range(self.horizon):
            state = self.causal_cell(token_h[:, frame_idx], state)
            velocity_frames.append(self.vel_head(torch.cat([obs_h, state], dim=-1)))
        vel = torch.stack(velocity_frames, dim=1)  # [B, h, A]
        return vel.reshape(b, self.chunk_dim)

    def reshape_raw_targets(self, raw_target_rate: torch.Tensor) -> torch.Tensor:
        """Return the frame-major raw target-rate chunk as ``[..., H, A]``."""
        if raw_target_rate.shape[-1] != self.chunk_dim:
            raise ValueError(
                f"Expected raw target-rate dim {self.chunk_dim}, got {raw_target_rate.shape[-1]}"
            )
        return raw_target_rate.view(*raw_target_rate.shape[:-1], self.horizon, self.action_dim)

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
