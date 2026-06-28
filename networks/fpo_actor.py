"""Official-aligned single-step Flow actor for FPO++.

This mirrors the actor half of amazon-far/fpo-control `modules/actor_critic.py::ActorCritic`
(the G1 motion-tracking configuration), as opposed to the shared `FlowMatchingPolicy`
(used by MixGRPO/PPO) which squashes actions through a tanh and conditions the velocity net
on a scalar timestep. The two key differences that matter for FPO++ are:

  1. NO tanh squash. The executed action is a LINEAR map of the flow endpoint:
         action = actor_scale * x_t                      (+ action_perturb during training)
     The CFM ratio is constructed from the CFM loss of the *executed* action `a` itself
     (scaled back by `a / actor_scale`), so the advantage (which belongs to `a`) and the
     CFM-loss gradient act on the same variable. With a nonlinear tanh the CFM loss lives on
     the pre-squash latent `z` while the advantage belongs to `a = squash(z)`, and the
     tanh Jacobian does NOT cancel in the FPO ratio (an approximate-likelihood ratio, not a
     true likelihood ratio), mis-aligning the gradient -- the root cause we are fixing.

  2. A multi-dimensional sinusoidal timestep embedding (default 8-dim cos/sin), so the
     velocity network input is [obs, embed(t), x_t] of width obs+timestep_embed_dim+action.

The flow time convention matches the official code: t = 1 is noise, t = 0 is the action.
`act()` integrates x_t from noise (t=1) down to the action (t=0); `get_cfm_loss` interpolates
x_t = t*eps + (1-t)*scaled_action and targets velocity eps - scaled_action.
"""
from __future__ import annotations

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    normalized = name.lower()
    return {"elu": nn.ELU, "relu": nn.ReLU, "silu": nn.SiLU, "tanh": nn.Tanh}[normalized]()


class FPOActor(nn.Module):
    """Single-step flow-matching actor (linear action map + timestep embedding).

    Args:
        obs_dim:            actor observation dimension (already normalized at call time).
        action_dim:         env action dimension (flow operates directly in action space).
        hidden_dims:        actor MLP hidden widths.
        activation:         activation name (elu for the G1 baselines).
        actor_scale:        linear scale a = actor_scale * x_t (official actor_scale, default 1).
        mlp_output_scale:   scale applied to the raw velocity-net output (official, default 1).
        timestep_embed_dim: width of the sinusoidal timestep embedding (official 8).
        cfm_loss_reduction: "mean" | "sum" | "sqrt" reduction over the action dim (tracking: mean).
        sampling_steps:     Euler integration steps generating one action (flow_steps).
        action_perturb_std: std of Gaussian noise added to the action during training (entropy reg).
        cfm_loss_t_inverse_cdf_beta: Beta(1, beta) inverse-CDF shaping of the CFM timesteps.
    """

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

    # ------------------------------------------------------------------ timestep embedding
    def _embed_timestep(self, t: torch.Tensor) -> torch.Tensor:
        """Embed a (*, 1) timestep into (*, timestep_embed_dim) via cos/sin of 2**k * t."""
        if t.shape[-1] != 1:
            raise ValueError(f"timestep must have last dim 1, got {tuple(t.shape)}")
        freqs = 2 ** torch.arange(self.timestep_embed_dim // 2, device=t.device, dtype=t.dtype)
        scaled = t * freqs
        return torch.cat([torch.cos(scaled), torch.sin(scaled)], dim=-1)

    # ------------------------------------------------------------------ flow integration
    def _integrate_flow(self, observations: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        """Euler-integrate the velocity field from t=1 (noise) to t=0 (action)."""
        device = observations.device
        dtype = observations.dtype
        steps = self.sampling_steps
        full_t = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=dtype)
        t_current = full_t[:-1]
        dt = full_t[1:] - full_t[:-1]  # negative steps (1 -> 0)
        batch = observations.shape[0]
        for i in range(steps):
            t_val = t_current[i].reshape(1, 1)
            embedded_t = self._embed_timestep(t_val).expand(batch, -1)
            mlp_output = self.actor(torch.cat([observations, embedded_t, x_t], dim=-1))
            velocity = self.mlp_output_scale * mlp_output
            x_t = x_t + velocity * dt[i]
        return x_t

    def act(self, observations: torch.Tensor) -> torch.Tensor:
        """Sample one action per observation. Training -> x_t ~ N(0, I) + action_perturb;
        eval (module in .eval()) -> x_t = 0 (deterministic zero-sampling)."""
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
        """Deterministic inference. eval_mode="zero" integrates from x_t = 0 (no perturb)."""
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

    # ------------------------------------------------------------------ CFM loss
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
        """Conditional flow-matching loss of the EXECUTED action (official formulation).

            scaled_action = action / actor_scale
            x_t           = t*eps + (1-t)*scaled_action        # t=1 noise, t=0 action
            v_pred        = mlp_output_scale * actor([obs, embed(t), x_t])
            target        = eps - scaled_action
            x0_pred       = x_t - t*v_pred  ;  x1_pred = x0_pred + v_pred
            loss          = reduce_d (v_pred - target)^2

        Args:
            observations: (B, obs_dim)
            actions:      (B, A) executed action
            eps:          (B, M, A) noises ~ N(0, I)
            t:            (B, M, 1) flow timesteps in (0, 1)

        Returns:
            loss    (B, M)
            x1_pred (B, M, A)   used for the KL drift estimate
            x0_pred (B, M, A)
        """
        B, A = actions.shape
        M = eps.shape[1]
        if eps.shape != (B, M, A):
            raise ValueError(f"eps must be {(B, M, A)}, got {tuple(eps.shape)}")
        if t.shape != (B, M, 1):
            raise ValueError(f"t must be {(B, M, 1)}, got {tuple(t.shape)}")
        scaled_actions = actions / self.actor_scale
        embedded_t = self._embed_timestep(t)  # (B, M, embed)
        x_t = t * eps + (1.0 - t) * scaled_actions[:, None, :]  # (B, M, A)
        obs_exp = observations[:, None, :].expand(B, M, observations.shape[-1])
        mlp_output = self.actor(torch.cat([obs_exp, embedded_t, x_t], dim=-1))
        velocity_pred = self.mlp_output_scale * mlp_output
        x0_pred = x_t - t * velocity_pred
        x1_pred = x0_pred + velocity_pred
        target_velocity = eps - scaled_actions[:, None, :]
        loss = self._reduce((velocity_pred - target_velocity) ** 2)
        return loss, x1_pred, x0_pred

    def sample_cfm_timesteps(self, batch: int, num_mc: int, device, dtype=torch.float32) -> torch.Tensor:
        """Sample CFM timesteps via the Beta(1, beta) inverse CDF, scaled to [0.005, 0.995]:
            t = 0.005 + 0.99 * (1 - (1 - u)^(1/beta)),  u ~ U(0, 1).
        Returns (batch, num_mc, 1)."""
        u = torch.rand(batch, num_mc, 1, device=device, dtype=dtype)
        beta = self.cfm_loss_t_inverse_cdf_beta
        return 0.005 + 0.99 * (1.0 - (1.0 - u) ** (1.0 / beta))
