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
    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 29,
        horizon: int = 1,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        activation: str = "elu",
        action_squash_scale: float = 5.0,
        causal_velocity: bool = False,
        causal_arch: str = "prefix_cumsum",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon


        self.chunk_dim = horizon * action_dim
        self.action_chunk_dim = horizon * action_dim
        self.obs_dim = obs_dim
        self.hidden_dims = tuple(hidden_dims)
        if not self.hidden_dims:
            raise ValueError("hidden_dims must contain at least one layer")
        self.causal_velocity = bool(causal_velocity)
        self.causal_arch = str(causal_arch)
        if self.causal_velocity and self.causal_arch != "prefix_cumsum":
            raise ValueError(f"Unsupported causal_arch: {self.causal_arch}")
        if not self.causal_velocity:
            # Full-chunk MLP: v_k depends on the whole z_0..z_{h-1} (non-causal).
            # Kept for backward compatibility with non-SFPO algorithms.
            layers: list[nn.Module] = []
            in_dim = self.obs_dim + self.chunk_dim + 1
            for hidden_dim in self.hidden_dims:
                layers.append(nn.Linear(in_dim, hidden_dim))
                layers.append(_activation(activation))
                in_dim = hidden_dim
            layers.append(nn.Linear(in_dim, self.chunk_dim))
            self.velocity_net = nn.Sequential(*layers)
        else:
            # Causal prefix-cumsum velocity: v_k depends only on z_0..z_k.
            # This makes logp_k a genuine conditional density
            # log pi(u_k | s, u_0..u_{k-1}), so per-frame PPO ratio is valid.
            #   obs_h   = obs_encoder([obs, time])              (shared)
            #   frame_h_i = frame_encoder(z_i) + frame_pos_i    (per-frame)
            #   prefix_h_k = sum_{i<=k} frame_h_i + prefix_pos_k  (causal cumsum)
            #   v_k = vel_head([obs_h, prefix_h_k])
            hidden = int(hidden_dims[-1])
            self._causal_hidden = hidden
            self.obs_encoder = _build_mlp(self.obs_dim + 1, tuple(hidden_dims), hidden, activation)
            self.frame_encoder = _build_mlp(self.action_dim, tuple(hidden_dims), hidden, activation)
            self.frame_pos_embed = nn.Parameter(torch.zeros(self.horizon, hidden))
            self.prefix_pos_embed = nn.Parameter(torch.zeros(self.horizon, hidden))
            nn.init.normal_(self.frame_pos_embed, std=0.02)
            nn.init.normal_(self.prefix_pos_embed, std=0.02)
            self.vel_head = _build_mlp(hidden * 2, tuple(hidden_dims), self.action_dim, activation)
        if action_squash_scale <= 0.0:
            raise ValueError(f"action_squash_scale must be > 0, got {action_squash_scale}")
        self.action_squash_scale = float(action_squash_scale)
        # Maximum per-frame action delta for the direct-delta smooth transform.
        # When None, _action_transform falls back to the absolute squash
        # (backward-compatible). When set (scalar or per-joint vector of shape
        # [action_dim]), the executed chunk is a causal smooth trajectory whose
        # per-frame step is bounded by max_delta, matching the environment's
        # action-rate penalty contract.
        self.action_max_delta: torch.Tensor | None = None
        # Action transform selector. Authoritative for "residual_absolute"
        # (v6). "absolute"/"delta" still fall through to the action_max_delta-
        # driven branches below for backward compatibility with non-SFPO callers
        # (e.g. MixGRPO, which never sets action_max_delta and relies on the
        # absolute squash). SFPO.build() keeps this consistent with
        # action_max_delta for absolute/delta.
        self.action_transform: str = "absolute"

    def set_action_max_delta(self, max_delta) -> None:
        if max_delta is None:
            self.action_max_delta = None
            return
        # Scalar: store as python float so it broadcasts to any device tensor in
        # _action_transform without device-mismatch. Vector (per-joint): store
        # as a tensor on the policy's current parameter device.
        if isinstance(max_delta, (int, float)):
            self.action_max_delta = float(max_delta)
            return
        t = torch.as_tensor(max_delta, dtype=torch.float32)
        if t.ndim == 0:
            self.action_max_delta = float(t.item())
            return
        if t.shape[-1] != self.action_dim and t.numel() != 1:
            raise ValueError(
                f"action_max_delta must be scalar or [action_dim={self.action_dim}], got shape {tuple(t.shape)}"
            )
        dev = next(self.parameters()).device if list(self.parameters()) else t.device
        self.action_max_delta = t.to(dev)

    def _prepare_observation(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != self.obs_dim:
            raise ValueError(
                f"Expected observation dim {self.obs_dim}, got {observation.shape[-1]}"
            )
        return observation

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
        if not self.causal_velocity:
            net_input = torch.cat([observation, noisy_actions, time.unsqueeze(-1)], dim=-1)
            return self.velocity_net(net_input)
        # Causal prefix-cumsum velocity: v_k depends only on z_0..z_k.
        # noisy_actions: [B, chunk_dim] -> [B, h, A]
        b = observation.shape[0]
        chunk = noisy_actions.view(b, self.horizon, self.action_dim)
        obs_h = self.obs_encoder(torch.cat([observation, time.unsqueeze(-1)], dim=-1))  # [B, H]
        frame_h = self.frame_encoder(chunk) + self.frame_pos_embed.unsqueeze(0)  # [B, h, H]
        prefix_h = torch.cumsum(frame_h, dim=1) + self.prefix_pos_embed.unsqueeze(0)  # [B, h, H]
        obs_h_exp = obs_h.unsqueeze(1).expand(-1, self.horizon, -1)  # [B, h, H]
        vel = self.vel_head(torch.cat([obs_h_exp, prefix_h], dim=-1))  # [B, h, A]
        return vel.reshape(b, self.chunk_dim)

    def _action_transform(self, action_value: torch.Tensor, prev_action: torch.Tensor | None = None) -> torch.Tensor:
        """Squash raw flow latents into executable actions.

        If ``prev_action is None`` or ``action_max_delta is None``: fall back to
        the absolute squash ``scale * tanh(raw / scale)`` (backward-compatible).

        Otherwise apply a **direct-delta** causal smooth transform: the flow
        latent IS the bounded delta from the last executed action (not an
        absolute target that then gets bounded). This is cleaner than
        target-then-bound because the policy directly parameterizes the
        quantity the environment penalizes (``a_i - a_{i-1}``):

            delta_i = max_delta * tanh(raw_i)   # bounded per-frame step
            a_i     = clamp(prev + delta_i, -scale, scale)  # absolute bound
            prev    = a_i                          # causal update

        The final clamp to ``[-scale, scale]`` guards against long-run drift:
        bounded deltas alone do not prevent the action from random-walking
        outside the joint limit over many chunks. ``max_delta`` may be a scalar
        or a per-joint vector of shape ``[action_dim]``. ``prev_action`` is the
        raw (un-normalized) last action ``[..., action_dim]``.
        """
        scale = self.action_squash_scale
        leading_shape = action_value.shape[:-1]
        chunk = action_value.view(*leading_shape, self.horizon, self.action_dim)
        if self.action_transform == "residual_absolute" and prev_action is not None:
            # v6 residual-absolute: anchor on prev_action in latent space, add
            # UNBOUNDED per-frame residuals, squash back. Zero residual = hold
            # the current action; a large residual can swing a frame to +/-scale
            # in one step (PPO-like recovery authority, what v4's hard delta cap
            # starved during push recovery). Causal: a_k depends on z_0..z_k
            # only, so the per-frame PPO ratio stays legal.
            prev = prev_action.reshape(*leading_shape, self.action_dim)  # [..., A]
            eps = 1.0e-6
            prev_normalized = torch.clamp(prev / scale, -1.0 + eps, 1.0 - eps)
            u = scale * torch.atanh(prev_normalized)  # latent anchor [..., A]
            actions = []
            for i in range(self.horizon):
                u = u + chunk[..., i, :]  # raw residual (unbounded), [..., A]
                a_i = scale * torch.tanh(u / scale)  # [..., A]
                actions.append(a_i)
            out = torch.stack(actions, dim=-2)  # [..., h, A]
            return out.reshape(*leading_shape, self.action_chunk_dim)
        if prev_action is None or self.action_max_delta is None:
            squashed = scale * torch.tanh(chunk / scale)
            return squashed.reshape(*leading_shape, self.action_chunk_dim)
        # direct-delta causal transform: latent -> bounded delta -> action.
        # A final absolute clamp to [-scale, scale] guards against long-run
        # drift: bounded deltas alone do not prevent the action from random-
        # walking outside the joint limit over many chunks. The env's own clamp
        # ([-100, 100]) is far too wide to act as an algorithmic safeguard.
        max_delta = self.action_max_delta  # scalar or [action_dim]
        scale = self.action_squash_scale
        prev = prev_action.reshape(*leading_shape, 1, self.action_dim)  # [..., 1, A]
        actions = []
        for i in range(self.horizon):
            delta = max_delta * torch.tanh(chunk[..., i, :])
            a_i = prev.squeeze(-2) + delta
            a_i = torch.clamp(a_i, -scale, scale)
            actions.append(a_i)
            prev = a_i.unsqueeze(-2)
        out = torch.stack(actions, dim=-2)  # [..., h, A]
        return out.reshape(*leading_shape, self.action_chunk_dim)
