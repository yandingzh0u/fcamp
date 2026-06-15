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
        # Latent-space C1 trajectory parametrization. The flow operates in a COEFFICIENT latent
        # of `basis_count` low-frequency ACCELERATION modes per joint. Those (tanh-bounded)
        # accelerations are integrated TWICE into a latent displacement D with D[0]=D[1]=0, the
        # boundary state is mapped to the UNBOUNDED pre-tanh latent, a latent velocity is formed,
        # and a SINGLE final tanh squashes everything:
        #   z_t = atanh(prev_action/scale) + t*z_velocity + D[t]@accel ; action = scale*tanh(z_t)
        # This makes C0 EXACT in action space (action[0]==prev_action) and C1 EXACT in latent
        # space, while every action is strictly bounded in (-scale, scale). A hard action bound
        # and exact action-space C1 at an arbitrary boundary are mathematically incompatible, so
        # C1 is enforced in latent space. No execution-time stitch/blend/clamp. The boundary
        # state (prev_action a_{t-1}, prev_prev_action a_{t-2}) is supplied to `_action_transform`.
        if basis_count is None or basis_count <= 0:
            basis_count = horizon
        # The twice-integrated displacement basis has D[0]=D[1]=0, so it carries at most
        # horizon-2 independent modes; clamp accordingly (and >=1).
        max_basis = max(1, horizon - 2)
        self.basis_count = int(min(max(1, basis_count), max_basis))
        # latent / flow / log_prob all live in coefficient (acceleration) space.
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
        # Acceleration coefficients are tanh-bounded by accel_scale before double integration so
        # the latent trajectory cannot diverge; tuned so an iid latent gives a moderate curve.
        self.accel_scale = float(action_squash_scale)
        # Fixed displacement basis D: (horizon, basis_count) with D[0,:]==0 AND D[1,:]==0. Built
        # by double-integrating a full-rank acceleration basis.
        self.register_buffer("displacement_basis", self._build_displacement_basis(horizon, self.basis_count))
        # Bounded decaying velocity kernel h[t]: h[0]=0, h[1]=1, increments decay geometrically
        # by `velocity_decay`, so h saturates at 1/(1-decay). Inherited latent velocity uses this
        # instead of a permanent linear ramp -> zero network output is a smooth stop, not drift.
        self.velocity_decay = 0.7
        self.register_buffer("velocity_kernel", self._build_velocity_kernel(horizon, self.velocity_decay))

    @staticmethod
    def _build_velocity_kernel(horizon: int, decay: float) -> torch.Tensor:
        # h[0]=0; h[t] = sum_{s=0}^{t-1} decay^s for t>=1. h[1]-h[0]=1 (exact C1 at boundary);
        # bounded by 1/(1-decay). Shape (horizon, 1) for broadcasting over action dims.
        h = torch.zeros(horizon, dtype=torch.float32)
        running = 0.0
        increment = 1.0
        for t in range(1, horizon):
            running += increment
            h[t] = running
            increment *= decay
        return h.view(horizon, 1)

    @staticmethod
    def _build_displacement_basis(horizon: int, basis_count: int) -> torch.Tensor:
        """Twice-integrated low-frequency acceleration basis; first TWO rows are exactly zero.

        accel[t,k]  = cos(pi*(t+0.5)*k/horizon)      (DCT-II modes incl. k=0 constant accel)
        vel[t,k]    = sum_{s=0..t-1} accel[s,k]       => vel[0,k] = 0
        D[t,k]      = sum_{s=0..t-1} vel[s,k]         => D[0,k] = 0 and D[1,k] = vel[0,k] = 0

        D[0]=D[1]=0 means the first two chunk frames are fixed purely by the latent boundary
        state (z_prev, z_velocity), giving exact C1 continuity in latent space. The constant acceleration
        mode integrates to a quadratic ramp (NOT subtracted), so no column collapses and the
        basis stays full column rank (rank == basis_count for basis_count <= horizon-2).
        Columns are normalized so an iid N(0,1) coefficient maps to ~unit-magnitude per-frame
        displacement (comparable exploration scale to the previous parametrization).
        """
        if horizon <= 2:
            # Too short for a double-integrated intra-chunk trajectory.
            return torch.zeros(horizon, basis_count, dtype=torch.float32)
        t = torch.arange(horizon, dtype=torch.float32).unsqueeze(1)  # (horizon, 1)
        k = torch.arange(basis_count, dtype=torch.float32).unsqueeze(0)  # (1, basis_count)
        accel = torch.cos(torch.pi * (t + 0.5) * k / horizon)  # (horizon, basis_count)
        velocity = torch.cumsum(accel, dim=0)  # vel[t] = sum_{s<=t} accel[s]
        velocity = torch.cat([torch.zeros_like(velocity[:1]), velocity[:-1]], dim=0)  # shift: vel[0]=0
        displacement = torch.cumsum(velocity, dim=0)  # D[t] = sum_{s<=t} vel[s]
        displacement = torch.cat(
            [torch.zeros_like(displacement[:1]), displacement[:-1]], dim=0
        )  # shift: D[0]=0, D[1]=vel[0]=0
        displacement = displacement / displacement.norm(dim=0, keepdim=True).clamp(min=1e-6)
        displacement = displacement * (horizon ** 0.5) / (basis_count ** 0.5)
        return displacement

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

    def _action_transform(
        self,
        action_value: torch.Tensor,
        start_action: torch.Tensor | None = None,
        start_prev_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Map the coefficient (acceleration) latent to a C1 action chunk via LATENT-space
        integration, with a single final tanh so every action is strictly in (-scale, scale).

        IMPORTANT mathematical fact: a hard action bound and exact action-space C1 at an
        arbitrary boundary state are mutually impossible (prev=4.9, vel=+0.2 would demand
        next=5.1, out of bound). We therefore make C0 EXACT in action space and C1 EXACT in
        the unbounded latent (pre-tanh) space, then squash once:

          z_prev   = atanh(start_action / scale)
          z_prev2  = atanh(start_prev_action / scale)
          z_vel    = z_prev - z_prev2                      (latent velocity)
          z_t      = z_prev + t * z_vel + D[t] @ accel     (D[0]=D[1]=0)
          action_t = scale * tanh(z_t)

        Then action[0] = scale*tanh(z_prev) = start_action exactly (C0), and the latent
        trajectory is C1 (z[1]-z[0] = z_vel from the previous boundary). No post-hoc clamp /
        blend / stitch; bounded by construction. rollout / inference / validation / log-prob
        recompute all use this single transform.

        start_action      : previous chunk's last executed frame (== a_{t-1}). None -> 0.
        start_prev_action : the frame before that (== a_{t-2}); with start_action gives the
                            latent velocity. None -> equals start_action (zero latent velocity).
        """
        scale = self.action_squash_scale
        leading_shape = action_value.shape[:-1]
        if self.horizon <= 2:
            chunk = action_value.view(*leading_shape, self.horizon, self.action_dim)
            return (scale * torch.tanh(chunk / scale)).reshape(*leading_shape, self.action_chunk_dim)

        # Bounded acceleration coefficients -> twice-integrated latent displacement (D[0]=D[1]=0).
        accel_scale = self.accel_scale
        coeff = action_value.view(*leading_shape, self.basis_count, self.action_dim)
        accel = accel_scale * torch.tanh(coeff / accel_scale)
        basis = self.displacement_basis.to(dtype=coeff.dtype)  # (horizon, basis_count), rows 0,1 == 0
        displacement = torch.einsum("hk,...ka->...ha", basis, accel)  # (..., horizon, action_dim)

        # Boundary state -> latent (pre-tanh) space. atanh is clamped to keep it finite even if
        # the incoming action sits exactly on the bound.
        def _to_latent(value: torch.Tensor | None) -> torch.Tensor:
            if value is None:
                return torch.zeros(*leading_shape, self.action_dim, device=coeff.device, dtype=coeff.dtype)
            value = torch.broadcast_to(
                value.to(device=coeff.device, dtype=coeff.dtype), (*leading_shape, self.action_dim)
            )
            normalized = torch.clamp(value / scale, -1.0 + 1e-6, 1.0 - 1e-6)
            # Inverse of the final squash action = scale * tanh(z): z = atanh(action/scale).
            return torch.atanh(normalized)

        z_prev = _to_latent(start_action)
        z_prev2 = _to_latent(start_prev_action) if start_prev_action is not None else z_prev
        z_velocity = z_prev - z_prev2  # latent-space velocity (C1 continuity in latent space)

        # z_prev / z_velocity are DIMENSIONLESS latent (pre-tanh) quantities, while `displacement`
        # (= D @ accel) is in ACTION units. Convert the displacement into latent units by
        # dividing by `scale` before adding, so the small-signal decode gain at z~0 is 1 (not
        # `scale`). Then action = scale * tanh(z_t) recovers the action units once.
        #
        # The inherited latent velocity is applied through a BOUNDED, DECAYING kernel h[t]
        # (h[0]=0, h[1]=1, increments geometrically decaying by `velocity_decay`), NOT a
        # permanent linear ramp t*z_velocity. h[1]-h[0]==1 keeps C1 EXACT at the boundary, but
        # h saturates at 1/(1-decay), so a ZERO network output decays the inherited velocity to
        # rest (a smooth stop) instead of extrapolating forever. This removes the residual drift
        # (frame0_abs creeping up over a rollout) the linear ramp caused.
        velocity_kernel = self.velocity_kernel.to(dtype=coeff.dtype)  # (horizon, 1), h[0]=0,h[1]=1
        z_t = z_prev.unsqueeze(-2) + z_velocity.unsqueeze(-2) * velocity_kernel + displacement / scale
        chunk = scale * torch.tanh(z_t)  # strictly in (-scale, scale)
        return chunk.reshape(*leading_shape, self.action_chunk_dim)
