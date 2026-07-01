from __future__ import annotations

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    return {"elu": nn.ELU, "relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}[normalized]()


class FPOActor(nn.Module):


    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...],
        activation: str = "elu",
        *,
        actor_scale: float = 1.0,
        mlp_output_scale: float = 1.0,
        timestep_embed_dim: int = 8,
        cfm_loss_reduction: str = "mean",
        sampling_steps: int = 4,
        action_perturb_std: float = 0.1,
        cfm_loss_t_inverse_cdf_beta: float = 1.0,
    ):
        super().__init__()
        if timestep_embed_dim % 2 != 0 or timestep_embed_dim <= 0:
            raise ValueError(f"timestep_embed_dim must be positive and even, got {timestep_embed_dim}")
        if cfm_loss_reduction not in ("mean", "sum", "sqrt"):
            raise ValueError(f"cfm_loss_reduction must be mean|sum|sqrt, got {cfm_loss_reduction!r}")
        self.obs_dim = int(obs_dim)
        self.num_actions = int(action_dim)
        self.actor_scale = float(actor_scale)
        self.mlp_output_scale = float(mlp_output_scale)
        self.timestep_embed_dim = int(timestep_embed_dim)
        self.cfm_loss_reduction = cfm_loss_reduction
        self.sampling_steps = max(1, int(sampling_steps))
        self.action_perturb_std = float(action_perturb_std)
        self.cfm_loss_t_inverse_cdf_beta = float(cfm_loss_t_inverse_cdf_beta)

        in_dim = self.obs_dim + self.timestep_embed_dim + self.num_actions
        layers: list[nn.Module] = []
        last = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(_activation(activation))
            last = h
        layers.append(nn.Linear(last, self.num_actions))
        self.actor = nn.Sequential(*layers)


    def _embed_timestep(self, t: torch.Tensor) -> torch.Tensor:

        if t.shape[-1] != 1:
            raise ValueError(f"timestep must have last dim 1, got {tuple(t.shape)}")
        freqs = 2 ** torch.arange(self.timestep_embed_dim // 2, device=t.device, dtype=t.dtype)
        scaled = t * freqs
        return torch.cat([torch.cos(scaled), torch.sin(scaled)], dim=-1)


    def _integrate_flow(self, observations: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:

        device = observations.device
        dtype = observations.dtype
        steps = self.sampling_steps
        full_t = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
        t_current = full_t[:-1]
        dt = full_t[1:] - full_t[:-1]
        batch = observations.shape[0]
        for i in range(steps):
            t_val = t_current[i].reshape(1, 1)
            embedded_t = self._embed_timestep(t_val).expand(batch, -1)
            mlp_output = self.actor(torch.cat([observations, embedded_t, x_t], dim=-1))
            velocity = self.mlp_output_scale * mlp_output
            x_t = x_t + velocity * dt[i]
        return x_t

    def act(self, observations: torch.Tensor) -> torch.Tensor:

        if observations.ndim != 2:
            raise ValueError(f"observations must be (batch, obs_dim), got {tuple(observations.shape)}")
        batch = observations.shape[0]
        if self.training:
            x_t = torch.randn(batch, self.num_actions, device=observations.device, dtype=observations.dtype)
        else:
            x_t = torch.zeros(batch, self.num_actions, device=observations.device, dtype=observations.dtype)
        x_t = self._integrate_flow(observations, x_t)
        actions = self.actor_scale * x_t
        if self.training and self.action_perturb_std > 0.0:
            actions = actions + self.action_perturb_std * torch.randn_like(actions)
        return actions

    def act_inference(self, observations: torch.Tensor, eval_mode: str = "zero") -> torch.Tensor:

        if observations.ndim != 2:
            raise ValueError(f"observations must be (batch, obs_dim), got {tuple(observations.shape)}")
        batch = observations.shape[0]
        if eval_mode == "zero":
            x_t = torch.zeros(batch, self.num_actions, device=observations.device, dtype=observations.dtype)
        elif eval_mode == "random":
            x_t = torch.randn(batch, self.num_actions, device=observations.device, dtype=observations.dtype)
        else:
            raise ValueError(f"Unknown eval_mode: {eval_mode!r}")
        x_t = self._integrate_flow(observations, x_t)
        return self.actor_scale * x_t


    def _reduce(self, sq_err: torch.Tensor) -> torch.Tensor:
        if self.cfm_loss_reduction == "mean":
            return sq_err.mean(dim=-1)
        if self.cfm_loss_reduction == "sum":
            return sq_err.sum(dim=-1)
        return sq_err.sum(dim=-1) / (sq_err.shape[-1] ** 0.5)

    def get_cfm_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        eps: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        B, A = actions.shape
        M = eps.shape[1]
        if eps.shape != (B, M, A):
            raise ValueError(f"eps must be {(B, M, A)}, got {tuple(eps.shape)}")
        if t.shape != (B, M, 1):
            raise ValueError(f"t must be {(B, M, 1)}, got {tuple(t.shape)}")
        scaled_actions = actions / self.actor_scale
        embedded_t = self._embed_timestep(t)
        x_t = t * eps + (1.0 - t) * scaled_actions[:, None, :]
        obs_exp = observations[:, None, :].expand(B, M, observations.shape[-1])
        mlp_output = self.actor(torch.cat([obs_exp, embedded_t, x_t], dim=-1))
        velocity_pred = self.mlp_output_scale * mlp_output
        x0_pred = x_t - t * velocity_pred
        x1_pred = x0_pred + velocity_pred
        target_velocity = eps - scaled_actions[:, None, :]
        loss = self._reduce((velocity_pred - target_velocity) ** 2)
        return loss, x1_pred, x0_pred

    def sample_cfm_timesteps(self, batch: int, num_mc: int, device, dtype=torch.float32) -> torch.Tensor:

        u = torch.rand(batch, num_mc, 1, device=device, dtype=dtype)
        beta = self.cfm_loss_t_inverse_cdf_beta
        return 0.005 + 0.99 * (1.0 - (1.0 - u) ** (1.0 / beta))
