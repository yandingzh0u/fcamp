"""Shared single-token Flow-CPS for bounded target-position increments.

The policy and exploration covariance have no chunk-offset parameters. Higher
level methods construct an H-token plan by repeatedly applying this one-token
distribution, while PPO evaluates the exact Gaussian density of each token.
"""

from __future__ import annotations

import math
from collections import deque

import torch
from torch import nn

from components.normalization.running_stats import EmpiricalNormalization
from method.base import Algorithm
from models.flow_cps_policy import FlowMatchingPolicy


def _consume_legacy_value_critic_initialization(
    observation_dim: int,
    hidden_dims: tuple[int, ...],
) -> None:
    """Preserve schema-16 fresh-run RNG after removing its discarded critic."""

    dimensions = (int(observation_dim) + 2, *map(int, hidden_dims), 1)
    for input_dim, output_dim in zip(dimensions, dimensions[1:]):
        nn.Linear(input_dim, output_dim)


class FlowCPSBase(Algorithm):
    def build(self) -> None:
        cfg = self.cfg
        env = self.env
        if int(cfg.horizon) < 1:
            raise ValueError(f"horizon must be >= 1, got {cfg.horizon}")
        if int(cfg.flow_steps) < 1:
            raise ValueError(f"flow_steps must be >= 1, got {cfg.flow_steps}")
        if int(cfg.rollout_env_steps) <= 0:
            raise ValueError(f"rollout_env_steps must be > 0, got {cfg.rollout_env_steps}")
        if int(cfg.rollout_env_steps) % int(cfg.horizon) != 0:
            raise ValueError(
                f"rollout_env_steps ({cfg.rollout_env_steps}) must be divisible by horizon ({cfg.horizon})."
            )

        self.num_act = env.action_dim
        self.base_actor_obs_dim = int(env.observation_dim)
        self.actor_obs_dim = self.base_actor_obs_dim
        self.critic_obs_dim = env.critic_observation_dim
        self.horizon_h = int(cfg.horizon)
        self.action_low = env.action_low
        self.action_high = env.action_high
        if (
            self.action_low.shape != (self.num_act,)
            or self.action_high.shape != (self.num_act,)
            or not bool(torch.isfinite(self.action_low).all())
            or not bool(torch.isfinite(self.action_high).all())
            or not bool((self.action_high > self.action_low).all())
        ):
            raise ValueError(
                "action_low/action_high must be finite [action_dim] tensors "
                "with strictly positive ranges"
            )
        self.target_action_mid = 0.5 * (
            self.action_low + self.action_high
        )
        self.target_action_half_range = 0.5 * (
            self.action_high - self.action_low
        )

        self._policy = FlowMatchingPolicy(
            obs_dim=self.actor_obs_dim,
            action_dim=self.num_act,
            hidden_dims=tuple(cfg.actor_hidden_dims),
            activation=cfg.activation,
        ).to(env.device)
        self.cps_target_increment_rms = float(
            cfg.cps_target_increment_rms
        )
        self.cps_trainable = bool(getattr(cfg, "cps_trainable", True))
        if (
            not math.isfinite(self.cps_target_increment_rms)
            or self.cps_target_increment_rms <= 0.0
        ):
            raise ValueError(
                "cps_target_increment_rms must be finite and > 0, got "
                f"{self.cps_target_increment_rms}"
            )
        self._cps_flat_dim = self.num_act
        self._policy.cps_cholesky_raw.requires_grad_(self.cps_trainable)
        target_increment_response = self._build_cps_target_increment_response()
        self._policy.register_buffer(
            "cps_target_increment_response",
            target_increment_response,
            persistent=False,
        )
        self._policy.register_buffer(
            "cps_target_increment_rms",
            torch.as_tensor(
                self.cps_target_increment_rms,
                device=env.device,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        identity_energy = target_increment_response.square().sum() / float(
            self.num_act
        )
        self._policy.register_buffer(
            "flow_mean_raw_scale",
            torch.as_tensor(
                self.cps_target_increment_rms,
                device=env.device,
                dtype=torch.float32,
            )
            / torch.sqrt(identity_energy.clamp(min=1.0e-12)),
            persistent=False,
        )
        self.flow_critic_steps = int(getattr(cfg, "flow_critic_steps", cfg.flow_steps))
        self.flow_critic_samples = int(getattr(cfg, "flow_critic_samples", 4))
        self.flow_critic_fm_samples = int(getattr(cfg, "flow_critic_fm_samples", 1))
        _consume_legacy_value_critic_initialization(
            self.critic_obs_dim,
            tuple(cfg.critic_hidden_dims),
        )
        self.chunk_dim = self.horizon_h * self.num_act

        self.empirical_normalization = bool(cfg.empirical_normalization)
        if self.empirical_normalization:
            self.actor_obs_normalizer: nn.Module = EmpiricalNormalization(self.actor_obs_dim, env.device)
        else:
            self.actor_obs_normalizer = nn.Identity()

        self.learning_rate = float(cfg.policy_lr)
        self.critic_learning_rate = float(cfg.value_lr)
        self.min_lr = 1e-5
        if self.learning_rate <= 0.0:
            raise ValueError(f"policy_lr must be > 0, got {cfg.policy_lr}")
        if self.critic_learning_rate <= 0.0:
            raise ValueError(f"value_lr must be > 0, got {cfg.value_lr}")

        self.actor_optimizer = torch.optim.AdamW(
            self._policy.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=float(cfg.weight_decay),
        )

        self.max_episode_steps = env.max_episode_steps
        self._policy_module = nn.ModuleDict({"actor": self._policy})
        self._init_train_episode_stats()

    # ------------------------------------------------------------------ #
    # Algorithm interface properties
    # ------------------------------------------------------------------ #
    @property
    def policy(self) -> nn.Module:
        return self._policy_module

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.actor_optimizer

    @property
    def horizon(self) -> int:
        return int(self.cfg.horizon)

    @property
    def kl_units(self) -> int:
        # Actor loss uses per-frame / per-prefix ratios (not a joint chunk
        # ratio), and the adaptive-LR KL is the masked MEAN per-frame KL.
        # So desired_kl is compared directly against a per-frame KL budget:
        # kl_units = 1 (no horizon scaling).
        return 1

    def extra_checkpoint_state(self) -> dict:
        return {
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "learning_rate": float(self.learning_rate),
            "critic_learning_rate": float(self.critic_learning_rate),
            "actor_obs_normalizer": self.actor_obs_normalizer.state_dict() if self.empirical_normalization else None,
        }

    def load_extra_checkpoint_state(self, payload: dict, reset_optimizer: bool = False) -> None:
        if reset_optimizer:
            self.learning_rate = float(self.cfg.policy_lr)
            self.critic_learning_rate = float(self.cfg.value_lr)
            for group in self.actor_optimizer.param_groups:
                group["lr"] = self.learning_rate
            for group in self.critic_optimizer.param_groups:
                group["lr"] = self.critic_learning_rate
        elif payload:
            self.learning_rate = float(payload.get("learning_rate", self.learning_rate))
            self.critic_learning_rate = float(payload.get("critic_learning_rate", self.critic_learning_rate))
            for group in self.actor_optimizer.param_groups:
                group["lr"] = self.learning_rate
            if "critic_optimizer" in payload:
                self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
            for group in self.critic_optimizer.param_groups:
                group["lr"] = self.critic_learning_rate
        if not payload:
            return
        if self.empirical_normalization:
            if payload.get("actor_obs_normalizer") is not None:
                self.actor_obs_normalizer.load_state_dict(payload["actor_obs_normalizer"])

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    def _chunks_per_update(self) -> int:
        return max(1, int(self.cfg.rollout_env_steps) // max(1, self.horizon_h))

    def _build_cps_target_increment_response(self) -> torch.Tensor:
        """Return the per-token raw-to-target-increment zero-point Jacobian.

        In bounded target coordinates, one raw token has local derivative
        ``action_half_range`` at the midpoint. Calibrating this one-step map
        makes the exploration scale independent of H and therefore identical
        at internal and cross-chunk tokens.
        """

        return torch.diag(
            self.target_action_half_range.to(
                device=self.env.device,
                dtype=torch.float32,
            )
        )

    def _norm_actor(self, obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        return self.actor_obs_normalizer(obs, update=update) if self.empirical_normalization else obs

    def _init_train_episode_stats(self) -> None:
        env = self.env
        self._train_reward_sum = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._train_episode_length = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._train_reward_buffer: deque[float] = deque(maxlen=100)
        self._train_length_buffer: deque[float] = deque(maxlen=100)

    def _record_train_episode_stats(self, rewards, dones, step_counts=None) -> None:
        self._train_reward_sum += rewards.to(dtype=torch.float32)
        if step_counts is None:
            self._train_episode_length += 1.0
        else:
            self._train_episode_length += step_counts.to(dtype=torch.float32)
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() == 0:
            return
        self._train_reward_buffer.extend(self._train_reward_sum.index_select(0, done_ids).detach().cpu().tolist())
        self._train_length_buffer.extend(self._train_episode_length.index_select(0, done_ids).detach().cpu().tolist())
        self._train_reward_sum[done_ids] = 0.0
        self._train_episode_length[done_ids] = 0.0

    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """Return one deterministic raw target-increment token."""

        actor_obs = self._norm_actor(obs, update=False)
        return self._flow_mean_raw(actor_obs).unsqueeze(1)

    def _raw_coordinate_to_target_action(
        self,
        raw_coordinate: torch.Tensor,
    ) -> torch.Tensor:
        """Map an unconstrained absolute coordinate into the action domain."""

        if raw_coordinate.shape[-1] != self.num_act:
            raise ValueError(
                f"raw_coordinate last dimension must be {self.num_act}, "
                f"got {tuple(raw_coordinate.shape)}"
            )
        mid = self.target_action_mid.to(
            device=raw_coordinate.device,
            dtype=raw_coordinate.dtype,
        )
        half_range = self.target_action_half_range.to(
            device=raw_coordinate.device,
            dtype=raw_coordinate.dtype,
        )
        return mid + half_range * torch.tanh(raw_coordinate)

    def _target_residual_to_target_action(
        self,
        raw_residual: torch.Tensor,
        target_anchor: torch.Tensor,
    ) -> torch.Tensor:
        """Update the persistent bounded target by one raw increment.

        ``raw_residual == 0`` holds ``target_anchor`` exactly.  The environment
        owns and carries that anchor across every frame and chunk.
        """

        if (
            raw_residual.ndim != 2
            or raw_residual.shape[-1] != self.num_act
            or target_anchor.shape != raw_residual.shape
        ):
            raise ValueError(
                "raw_residual and target_anchor must have matching "
                f"[B,{self.num_act}] shapes, got "
                f"{tuple(raw_residual.shape)} and {tuple(target_anchor.shape)}"
            )
        mid = self.target_action_mid.to(
            device=target_anchor.device,
            dtype=target_anchor.dtype,
        )
        half_range = self.target_action_half_range.to(
            device=target_anchor.device,
            dtype=target_anchor.dtype,
        )
        normalized_anchor = (target_anchor - mid) / half_range
        anchor_raw = torch.atanh(
            normalized_anchor.clamp(
                min=-1.0 + 1.0e-6,
                max=1.0 - 1.0e-6,
            )
        )
        return self._raw_coordinate_to_target_action(
            anchor_raw + raw_residual
        )

    def _flow_mean_raw(
        self,
        actor_obs: torch.Tensor,
    ) -> torch.Tensor:
        """Return one differentiable Flow mean in raw-token space."""

        batch = actor_obs.shape[0]
        latent = torch.zeros(
            batch,
            self.num_act,
            device=actor_obs.device,
            dtype=actor_obs.dtype,
        )
        self._policy._validate_inputs(actor_obs, latent, int(self.cfg.flow_steps))
        obs_prep = self._policy._prepare_observation(actor_obs)
        steps = int(self.cfg.flow_steps)
        sigma_schedule = torch.linspace(1.0, 0.0, steps + 1, device=actor_obs.device, dtype=actor_obs.dtype)
        for step_index in range(steps):
            sigma = sigma_schedule[step_index]
            timestep = torch.full(
                (batch,),
                float(sigma.item()),
                device=latent.device,
                dtype=latent.dtype,
            )
            velocity = self._policy.velocity_field(obs_prep, latent, timestep)
            sigma_now = sigma_schedule[step_index].to(
                device=velocity.device,
                dtype=velocity.dtype,
            )
            sigma_next = sigma_schedule[step_index + 1].to(
                device=velocity.device,
                dtype=velocity.dtype,
            )
            latent = latent + velocity * (sigma_next - sigma_now)
        # The Flow network evolves in a fixed likelihood-standardized
        # coordinate.  Mapping its output through the initial effective CPS
        # standard deviation keeps one network unit commensurate with one
        # exploration standard deviation.  The mapping is immutable; trainable
        # covariance shape changes cannot silently rescale the actor mean.
        residual = latent * self._policy.flow_mean_raw_scale.to(
            device=latent.device, dtype=latent.dtype
        )
        return residual

    def _raw_cps_cholesky(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return self._policy.raw_cps_cholesky(device=device, dtype=dtype)

    def _effective_cps_cholesky(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the shared per-token Cholesky used everywhere."""

        chol = self._raw_cps_cholesky(device=device, dtype=dtype)
        response = self._policy.cps_target_increment_response.to(
            device=device,
            dtype=dtype,
        )
        response_chol = response @ chol
        target_increment_energy = (
            response_chol.square().sum() / float(self.num_act)
        )
        target_increment_scale = self._policy.cps_target_increment_rms.to(
            device=device,
            dtype=dtype,
        ) / torch.sqrt(target_increment_energy.clamp(min=1.0e-12))
        return chol * target_increment_scale, target_increment_energy

    @torch.no_grad()
    def _final_cps_statistics(self) -> dict[str, float]:
        """Describe the exact final covariance used by sampling and PPO."""

        device = self.env.device
        effective_chol, raw_target_increment_energy = (
            self._effective_cps_cholesky(
                device=device,
                dtype=torch.float32,
            )
        )
        covariance = effective_chol @ effective_chol.transpose(0, 1)
        covariance_diag = torch.diagonal(covariance)
        covariance_offdiag = covariance - torch.diag_embed(covariance_diag)
        response = self._policy.cps_target_increment_response.to(
            device=device,
            dtype=torch.float32,
        )
        achieved_target_increment_rms = torch.sqrt(
            (response @ effective_chol).square().sum()
            / float(self.num_act)
        )
        aggregate_scale = self._policy.cps_target_increment_rms / torch.sqrt(
            raw_target_increment_energy.clamp(min=1.0e-12)
        )
        covariance_logdet = 2.0 * torch.log(
            torch.diagonal(effective_chol).clamp(min=1.0e-12)
        ).sum()
        raw_shape_norm = torch.linalg.vector_norm(
            self._policy.cps_cholesky_raw.detach()
        )
        shape_radius = float(
            self._policy.CPS_CHOLESKY_SHAPE_RADIUS
        )
        return {
            "policy/cps_target_increment_rms_target": float(
                self._policy.cps_target_increment_rms.item()
            ),
            "policy/cps_target_increment_rms_achieved": float(
                achieved_target_increment_rms.item()
            ),
            "policy/cps_target_increment_scale": float(
                aggregate_scale.item()
            ),
            "policy/cps_cov_trace_mean": float(covariance_diag.mean().item()),
            "policy/cps_cov_trace_min": float(covariance_diag.min().item()),
            "policy/cps_cov_trace_max": float(covariance_diag.max().item()),
            "policy/cps_cov_logdet": float(covariance_logdet.item()),
            "policy/cps_cov_offdiag_abs": float(
                covariance_offdiag.abs().mean().item()
            ),
            "policy/cps_shape_norm": float(
                min(float(raw_shape_norm.item()), shape_radius)
            ),
            "policy/cps_shape_raw_norm": float(
                raw_shape_norm.item()
            ),
            "policy/cps_shape_radius": shape_radius,
            "policy/flow_mean_raw_scale": float(
                self._policy.flow_mean_raw_scale.item()
            ),
            "policy/cps_params": float(
                self._policy.cps_cholesky_raw.numel()
            ),
        }

    def _final_cps_conditional_log_prob(
        self,
        raw_z: torch.Tensor,
        mean_z: torch.Tensor,
    ) -> torch.Tensor:
        """Return the exact shared-Gaussian log density per token.

        Any leading dimensions are preserved; only the final action dimension
        belongs to the distribution.
        """

        if raw_z.shape != mean_z.shape or raw_z.shape[-1] != self._cps_flat_dim:
            raise ValueError(
                "raw_z and mean_z must have matching shapes ending in "
                f"{self._cps_flat_dim}, "
                f"got {tuple(raw_z.shape)} and {tuple(mean_z.shape)}"
            )
        chol, _ = self._effective_cps_cholesky(device=raw_z.device, dtype=raw_z.dtype)
        residual = (raw_z - mean_z).reshape(-1, self.num_act)
        whitened = torch.linalg.solve_triangular(
            chol,
            residual.transpose(0, 1),
            upper=False,
        ).transpose(0, 1)
        log_diag = torch.log(torch.diagonal(chol).clamp(min=1.0e-12))
        component = (
            -0.5 * (whitened.square() + math.log(2.0 * math.pi))
            - log_diag.view(1, self.num_act)
        )
        return component.sum(dim=-1).reshape(raw_z.shape[:-1])

    def _final_cps_expected_conditional_kl(
        self,
        old_mean: torch.Tensor,
        new_mean: torch.Tensor,
        old_chol: torch.Tensor,
        new_chol: torch.Tensor,
    ) -> torch.Tensor:
        """Return exact old||new Gaussian KL at every supplied context."""

        if (
            old_mean.shape != new_mean.shape
            or old_mean.ndim != 3
            or old_mean.shape[-1] != self.num_act
        ):
            raise ValueError(
                "old_mean and new_mean must have matching "
                f"[B,H,{self.num_act}] shapes"
            )
        expected_chol_shape = (self.num_act, self.num_act)
        if old_chol.shape != expected_chol_shape or new_chol.shape != expected_chol_shape:
            raise ValueError(
                "old_chol and new_chol must both have shape "
                f"{expected_chol_shape}"
            )

        # KL acceptance operates near zero, so avoid float32 cancellation.
        work_dtype = torch.float64
        old_mu = old_mean.to(dtype=work_dtype)
        new_mu = new_mean.to(dtype=work_dtype)
        old_l = old_chol.to(device=old_mean.device, dtype=work_dtype)
        new_l = new_chol.to(device=old_mean.device, dtype=work_dtype)
        covariance_whitened = torch.linalg.solve_triangular(
            new_l,
            old_l,
            upper=False,
        )
        flat_mean_delta = (old_mu - new_mu).reshape(-1, self.num_act)
        mean_whitened = torch.linalg.solve_triangular(
            new_l,
            flat_mean_delta.transpose(0, 1),
            upper=False,
        ).transpose(0, 1)
        logdet_ratio = 2.0 * (
            torch.log(torch.diagonal(new_l)).sum()
            - torch.log(torch.diagonal(old_l)).sum()
        )
        covariance_term = (
            covariance_whitened.square().sum()
            - float(self.num_act)
            + logdet_ratio
        )
        conditional_kl = 0.5 * (
            covariance_term
            + mean_whitened.square().sum(dim=-1)
        )
        minimum = float(conditional_kl.min().item())
        if minimum < -1.0e-7:
            raise FloatingPointError(
                "Gaussian conditional KL became materially negative: "
                f"min={minimum:.3e}"
            )
        return conditional_kl.clamp_min(0.0).reshape(
            old_mean.shape[:-1]
        ).to(dtype=old_mean.dtype)

    def _sample_final_cps(
        self,
        actor_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample the sole actor random variable after deterministic Flow ODE."""

        mean_z = self._flow_mean_raw(actor_obs)
        chol, _ = self._effective_cps_cholesky(device=actor_obs.device, dtype=actor_obs.dtype)
        epsilon = torch.randn_like(mean_z)
        raw_z = mean_z + epsilon @ chol.transpose(0, 1)
        conditional_logp = self._final_cps_conditional_log_prob(raw_z, mean_z)
        return raw_z, mean_z, conditional_logp

    def _recompute_final_cps_log_prob(
        self,
        actor_obs: torch.Tensor,
        raw_z: torch.Tensor,
    ) -> torch.Tensor:
        if (
            actor_obs.shape[:-1] != raw_z.shape[:-1]
            or actor_obs.shape[-1] != self.actor_obs_dim
            or raw_z.shape[-1] != self.num_act
        ):
            raise ValueError(
                "actor_obs and raw_z must have matching leading dimensions "
                f"and end in {self.actor_obs_dim} and {self.num_act}, got "
                f"{tuple(actor_obs.shape)} and {tuple(raw_z.shape)}"
            )
        mean_z = self._flow_mean_raw(
            actor_obs.reshape(-1, self.actor_obs_dim)
        ).reshape(raw_z.shape)
        return self._final_cps_conditional_log_prob(raw_z, mean_z)

    # ------------------------------------------------------------------ #
    # Env reset / rollout entry points
    # ------------------------------------------------------------------ #
    def initial_reset(self) -> torch.Tensor:
        obs = self.env.reset()
        if bool(self.cfg.init_at_random_ep_len) and self.max_episode_steps > 0:
            self.env.episode_steps = torch.randint_like(self.env.episode_steps, high=int(self.max_episode_steps))
        self._obs = obs
        self._critic_obs = self.env.get_critic_observation()
        return obs

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        # clipped-ratio-aligned continuous stream across updates (same as Flow-CPS).
        return self._obs

    def _policy_mini_batch_size(self, sample_count: int) -> int:
        num_mini_batches = max(1, int(self.cfg.num_mini_batches))
        return max(1, math.ceil(sample_count / num_mini_batches))

    def _policy_micro_batch_size(self, batch_size: int) -> int:
        if int(self.cfg.micro_batch_size) <= 0:
            return max(1, batch_size)
        return max(1, min(batch_size, int(self.cfg.micro_batch_size)))

    def _add_sampler_metrics(self, metrics: dict) -> None:
        stats = self.env.adaptive_sampling_stats()
        for key, value in stats.items():
            value = float(value)
            if math.isfinite(value):
                metrics[f"sampler/{key}"] = value
