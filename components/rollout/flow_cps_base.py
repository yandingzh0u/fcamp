"""Flow-CPS with a deterministic Flow mean and one final control distribution.

Flow integration is deterministic.  Its final frame-major HxA output is the
mean of the raw target-rate variable executed by the environment's stateful
rate decoder.  CPS owns exactly one state-independent dense Cholesky factor in
that same variable.  A fixed decoder-response metric applies one aggregate
physical scaling to the factor; sampling and likelihood evaluation use that
identical effective Cholesky.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from collections import deque

import torch
from torch import nn

from method.base import Algorithm
from components.optim.kl_scheduler import adaptive_lr_from_kl
from models.flow_value_critic import FlowChunkValueCritic
from models.flow_cps_policy import FlowMatchingPolicy
from models.flow_sampling import flow_ode_mean
from models.mlp_actor_critic import EmpiricalNormalization


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

        self._policy = FlowMatchingPolicy(
            obs_dim=self.actor_obs_dim,
            action_dim=self.num_act,
            horizon=self.horizon_h,
            hidden_dims=tuple(cfg.actor_hidden_dims),
            activation=cfg.activation,
            causal_velocity=True,
            causal_arch="prefix_cumsum",
        ).to(env.device)
        self.cps_physical_rms = float(cfg.cps_physical_rms)
        self.cps_trainable = bool(getattr(cfg, "cps_trainable", True))
        if not math.isfinite(self.cps_physical_rms) or self.cps_physical_rms <= 0.0:
            raise ValueError(
                f"cps_physical_rms must be finite and > 0, got {self.cps_physical_rms}"
            )
        self._cps_flat_dim = self.horizon_h * self.num_act
        self._policy.cps_cholesky_raw.requires_grad_(self.cps_trainable)
        physical_response, physical_rms, decoder_contract = self._build_cps_physical_response()
        self._policy.register_buffer("cps_physical_response", physical_response, persistent=False)
        self._policy.register_buffer(
            "cps_physical_rms",
            torch.as_tensor(physical_rms, device=env.device, dtype=torch.float32),
            persistent=False,
        )
        identity_energy = physical_response.square().sum() / float(
            3 * self.num_act
        )
        self._policy.register_buffer(
            "flow_mean_raw_scale",
            torch.as_tensor(
                physical_rms,
                device=env.device,
                dtype=torch.float32,
            )
            / torch.sqrt(identity_energy.clamp(min=1.0e-12)),
            persistent=False,
        )
        self.cps_decoder_dt, self.cps_decoder_rho, self.cps_decoder_rate_limit = decoder_contract
        self.flow_critic_steps = int(getattr(cfg, "flow_critic_steps", cfg.flow_steps))
        self.flow_critic_samples = int(getattr(cfg, "flow_critic_samples", 4))
        self.flow_critic_fm_samples = int(getattr(cfg, "flow_critic_fm_samples", 1))
        self.critic = FlowChunkValueCritic(
            self.critic_obs_dim,
            tuple(cfg.critic_hidden_dims),
            cfg.activation,
            flow_steps=self.flow_critic_steps,
            eval_samples=self.flow_critic_samples,
        ).to(env.device)
        self.chunk_dim = self._policy.chunk_dim

        self.empirical_normalization = bool(cfg.empirical_normalization)
        if self.empirical_normalization:
            self.actor_obs_normalizer: nn.Module = EmpiricalNormalization(self.actor_obs_dim, env.device)
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(self.critic_obs_dim, env.device)
        else:
            self.actor_obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        self.learning_rate = float(cfg.policy_lr)
        self.critic_learning_rate = float(cfg.value_lr)
        self.max_lr = 1e-2
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
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=self.critic_learning_rate,
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=float(cfg.critic_weight_decay),
        )

        self.advantage_normalization = str(cfg.advantage_normalization).lower()
        # KL controller: prefix KL (cumsum per-frame log-ratio / prefix length),
        # updated before each minibatch optimizer step (clipped-ratio-aligned), with an
        # actor epoch early-stop when prefix KL exceeds the adaptive threshold.
        self.kl_early_stop_factor = float(getattr(cfg, "kl_early_stop_factor", 4.0))

        self.max_episode_steps = env.max_episode_steps
        self._policy_module = nn.ModuleDict({"actor": self._policy, "critic": self.critic})
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
            "critic_obs_normalizer": self.critic_obs_normalizer.state_dict() if self.empirical_normalization else None,
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
            if payload.get("critic_obs_normalizer") is not None:
                self.critic_obs_normalizer.load_state_dict(payload["critic_obs_normalizer"])

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    def _chunks_per_update(self) -> int:
        return max(1, int(self.cfg.rollout_env_steps) // max(1, self.horizon_h))

    def _training_rollout_horizon(self) -> int:
        return self.horizon_h * self._chunks_per_update()

    def _build_cps_physical_response(
        self,
    ) -> tuple[torch.Tensor, float, tuple[float, float, torch.Tensor]]:
        """Build the fixed zero-point response used to scale final CPS noise.

        All rows have normalized-action units.  They contain the mean per-frame
        action delta, mean per-frame second difference, and the next H physical
        action deltas caused by the carried terminal rate while the future
        target perturbation is zero.  The last group prices exactly the state
        injected across one chunk boundary.  It intentionally does not collapse
        an infinite tail into one cumulative displacement: the actor emits a
        fresh target on every future frame.
        """
        env = self.env
        device = env.device
        dtype = torch.float32
        dt = float(getattr(env, "dt"))
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"decoder dt must be finite and > 0, got {dt}")

        rho_value = env.command_rate_decay
        if torch.is_tensor(rho_value):
            if rho_value.numel() != 1:
                raise ValueError("command_rate_decay must be a scalar")
            rho = float(rho_value.detach().item())
        else:
            rho = float(rho_value)
        if not math.isfinite(rho) or not (0.0 < rho < 1.0):
            raise ValueError(f"command_rate_decay must be in (0, 1), got {rho}")

        rate_limit = torch.as_tensor(
            env.command_rate_limit,
            device=device,
            dtype=dtype,
        ).flatten()
        if rate_limit.numel() == 1:
            rate_limit = rate_limit.expand(self.num_act).clone()
        if rate_limit.shape != (self.num_act,):
            raise ValueError(
                f"command_rate_limit must be scalar or [{self.num_act}], got {tuple(rate_limit.shape)}"
            )
        if not bool(torch.isfinite(rate_limit).all()) or not bool((rate_limit > 0.0).all()):
            raise ValueError("command_rate_limit must contain only finite positive values")

        h = self.horizon_h
        temporal = torch.zeros(h, h, device=device, dtype=dtype)
        for row in range(h):
            for col in range(row + 1):
                temporal[row, col] = (1.0 - rho) * (rho ** (row - col))
        rate_response = torch.kron(temporal, torch.diag(rate_limit))
        delta_response = dt * rate_response

        difference = torch.eye(h, device=device, dtype=dtype)
        if h > 1:
            difference[1:, :-1] -= torch.eye(h - 1, device=device, dtype=dtype)
        d2_response = torch.kron(
            difference,
            torch.eye(self.num_act, device=device, dtype=dtype),
        ) @ delta_response

        terminal_rate_response = rate_response[-self.num_act :, :]
        tail_frames = [
            dt * (rho ** (frame + 1)) * terminal_rate_response
            for frame in range(h)
        ]
        tail_response = torch.cat(tail_frames, dim=0)
        response = torch.cat(
            [
                delta_response / math.sqrt(float(h)),
                d2_response / math.sqrt(float(h)),
                tail_response / math.sqrt(float(h)),
            ],
            dim=0,
        )
        return response, self.cps_physical_rms, (dt, rho, rate_limit.detach().clone())

    def _norm_actor(self, obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        return self.actor_obs_normalizer(obs, update=update) if self.empirical_normalization else obs

    def _norm_critic(self, obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        return self.critic_obs_normalizer(obs, update=update) if self.empirical_normalization else obs

    def _init_train_episode_stats(self) -> None:
        env = self.env
        self._train_reward_sum = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._train_episode_length = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._train_reward_buffer: deque[float] = deque(maxlen=100)
        self._train_length_buffer: deque[float] = deque(maxlen=100)
        self._train_completed_episodes = 0

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
        self._train_completed_episodes += int(done_ids.numel())
        self._train_reward_sum[done_ids] = 0.0
        self._train_episode_length[done_ids] = 0.0

    def _deterministic_actor_raw_targets(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return self._flow_mean_raw(actor_obs).view(
            actor_obs.shape[0],
            self.horizon_h,
            self.num_act,
        )

    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        """Return deterministic raw target rates.

        The Algorithm interface retains this historical name, but Stage-B
        deployment passes these values through the environment decoder one
        physical frame at a time.  They are not absolute actions.
        """
        actor_obs = self._norm_actor(obs, update=False)
        return self._flow_mean_raw(actor_obs).view(
            obs.shape[0],
            self.horizon_h,
            self.num_act,
        )

    def _flow_mean_raw(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Differentiable deterministic Flow ODE output in final raw-z space."""
        batch = actor_obs.shape[0]
        latent = torch.zeros(batch, self.chunk_dim, device=actor_obs.device, dtype=actor_obs.dtype)
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
            latent = flow_ode_mean(velocity, latent, sigma_schedule, step_index)
        # The Flow network evolves in a fixed likelihood-standardized
        # coordinate.  Mapping its output through the initial effective CPS
        # standard deviation keeps one network unit commensurate with one
        # exploration standard deviation.  The mapping is immutable; trainable
        # covariance shape changes cannot silently rescale the actor mean.
        return latent * self._policy.flow_mean_raw_scale.to(
            device=latent.device,
            dtype=latent.dtype,
        )

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
        """Return the one Cholesky used by both sampling and likelihood.

        The response already averages delta and d2 over H.  Dividing by three
        action-sized feature groups makes ``physical_energy`` their aggregate
        mean square.  Exactly one scalar then maps it to cps_physical_rms.
        """
        chol = self._raw_cps_cholesky(device=device, dtype=dtype)
        response = self._policy.cps_physical_response.to(device=device, dtype=dtype)
        response_chol = response @ chol
        physical_energy = response_chol.square().sum() / float(3 * self.num_act)
        physical_scale = self._policy.cps_physical_rms.to(device=device, dtype=dtype) / torch.sqrt(
            physical_energy.clamp(min=1.0e-12)
        )
        return chol * physical_scale, physical_energy

    @torch.no_grad()
    def _final_cps_statistics(self) -> dict[str, float]:
        """Describe the exact final covariance used by sampling and PPO."""

        device = self.env.device
        effective_chol, raw_physical_energy = self._effective_cps_cholesky(
            device=device,
            dtype=torch.float32,
        )
        covariance = effective_chol @ effective_chol.transpose(0, 1)
        covariance_diag = torch.diagonal(covariance)
        covariance_offdiag = covariance - torch.diag_embed(covariance_diag)
        response = self._policy.cps_physical_response.to(
            device=device,
            dtype=torch.float32,
        )
        achieved_physical_rms = torch.sqrt(
            (response @ effective_chol).square().sum()
            / float(3 * self.num_act)
        )
        aggregate_scale = self._policy.cps_physical_rms / torch.sqrt(
            raw_physical_energy.clamp(min=1.0e-12)
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
            "policy/cps_physical_rms_target": float(
                self._policy.cps_physical_rms.item()
            ),
            "policy/cps_physical_rms_achieved": float(
                achieved_physical_rms.item()
            ),
            "policy/cps_physical_scale": float(aggregate_scale.item()),
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
        """Exact frame-conditional decomposition of the final joint Gaussian."""
        if raw_z.shape != mean_z.shape or raw_z.shape[-1] != self._cps_flat_dim:
            raise ValueError(
                f"raw_z and mean_z must have matching [B,{self._cps_flat_dim}] shapes, "
                f"got {tuple(raw_z.shape)} and {tuple(mean_z.shape)}"
            )
        chol, _ = self._effective_cps_cholesky(device=raw_z.device, dtype=raw_z.dtype)
        residual = raw_z - mean_z
        whitened = torch.linalg.solve_triangular(
            chol,
            residual.transpose(0, 1),
            upper=False,
        ).transpose(0, 1)
        log_diag = torch.log(torch.diagonal(chol).clamp(min=1.0e-12))
        component = (
            -0.5 * (whitened.square() + math.log(2.0 * math.pi))
            - log_diag.view(1, self._cps_flat_dim)
        )
        return component.view(raw_z.shape[0], self.horizon_h, self.num_act).sum(dim=-1)

    def _final_cps_expected_conditional_kl(
        self,
        old_mean: torch.Tensor,
        new_mean: torch.Tensor,
        old_chol: torch.Tensor,
        new_chol: torch.Tensor,
    ) -> torch.Tensor:
        """Return exact old||new expected conditional KL for every frame.

        For a joint Gaussian, the chain rule makes frame-k conditional KL equal
        to the difference between adjacent prefix-marginal KLs.  Leading blocks
        of a frame-major lower Cholesky are exactly those prefix marginals, so
        this remains exact for a fully dense cross-frame covariance.
        """
        if (
            old_mean.shape != new_mean.shape
            or old_mean.ndim != 2
            or old_mean.shape[-1] != self._cps_flat_dim
        ):
            raise ValueError(
                "old_mean and new_mean must have matching "
                f"[B,{self._cps_flat_dim}] shapes"
            )
        expected_chol_shape = (self._cps_flat_dim, self._cps_flat_dim)
        if old_chol.shape != expected_chol_shape or new_chol.shape != expected_chol_shape:
            raise ValueError(
                "old_chol and new_chol must both have shape "
                f"{expected_chol_shape}"
            )

        # Prefix subtraction can lose a few ulps in float32 even when the
        # distributions match.  The acceptance statistic is small and worth
        # evaluating in float64.
        work_dtype = torch.float64
        old_mu = old_mean.to(dtype=work_dtype)
        new_mu = new_mean.to(dtype=work_dtype)
        old_l = old_chol.to(device=old_mean.device, dtype=work_dtype)
        new_l = new_chol.to(device=old_mean.device, dtype=work_dtype)
        previous_prefix = torch.zeros(
            old_mean.shape[0],
            device=old_mean.device,
            dtype=work_dtype,
        )
        conditional: list[torch.Tensor] = []
        for frame in range(self.horizon_h):
            prefix_dim = (frame + 1) * self.num_act
            old_prefix_l = old_l[:prefix_dim, :prefix_dim]
            new_prefix_l = new_l[:prefix_dim, :prefix_dim]
            covariance_whitened = torch.linalg.solve_triangular(
                new_prefix_l,
                old_prefix_l,
                upper=False,
            )
            mean_whitened = torch.linalg.solve_triangular(
                new_prefix_l,
                (old_mu[:, :prefix_dim] - new_mu[:, :prefix_dim]).transpose(0, 1),
                upper=False,
            ).transpose(0, 1)
            logdet_ratio = 2.0 * (
                torch.log(torch.diagonal(new_prefix_l)).sum()
                - torch.log(torch.diagonal(old_prefix_l)).sum()
            )
            prefix_kl = 0.5 * (
                covariance_whitened.square().sum()
                + mean_whitened.square().sum(dim=-1)
                - float(prefix_dim)
                + logdet_ratio
            )
            frame_kl = prefix_kl - previous_prefix
            minimum = float(frame_kl.min().item())
            if minimum < -1.0e-7:
                raise FloatingPointError(
                    "Gaussian conditional KL became materially negative: "
                    f"frame={frame} min={minimum:.3e}"
                )
            conditional.append(frame_kl.clamp_min(0.0))
            previous_prefix = prefix_kl
        return torch.stack(conditional, dim=-1).to(dtype=old_mean.dtype)

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
        mean_z = self._flow_mean_raw(actor_obs)
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

    def _record_first_done(
        self,
        *,
        new_done,
        timeouts,
        info,
        global_step,
        ever_done,
        first_done_step,
        first_done_phase,
        first_done_anchor_pos,
        first_done_anchor_ori,
        first_done_ee_body,
        first_done_timeout,
        first_done_motion_complete,
    ) -> None:
        newly_done = (~ever_done) & new_done
        if not bool(newly_done.any()):
            return
        ids = newly_done.nonzero(as_tuple=False).squeeze(-1)
        first_done_step[ids] = int(global_step)
        first_done_timeout[ids] = timeouts[ids]
        dterms = info["done_terms"]
        if "motion_complete" in dterms:
            first_done_motion_complete[ids] = dterms["motion_complete"].bool()[ids]
        if "anchor_pos_bad" in dterms:
            first_done_anchor_pos[ids] = dterms["anchor_pos_bad"].bool()[ids]
        if "anchor_ori_bad" in dterms:
            first_done_anchor_ori[ids] = dterms["anchor_ori_bad"].bool()[ids]
        if "ee_body_bad" in dterms:
            first_done_ee_body[ids] = dterms["ee_body_bad"].bool()[ids]
        tps = info.get("termination_phase_steps")
        if torch.is_tensor(tps):
            first_done_phase[ids] = tps.long().to(first_done_phase.device)[ids]
        ever_done[ids] = True

    # ------------------------------------------------------------------ #
    # Collect: per-frame storage + multi-horizon prefix targets
    # ------------------------------------------------------------------ #
    def collect(self, current_obs: torch.Tensor) -> dict:
        env = self.env
        device = env.device
        n_envs = env.num_envs
        chunks = self._chunks_per_update()
        h = self.horizon_h
        gamma = float(self.cfg.discount_gamma)

        actor_obs_buf = torch.zeros(chunks, n_envs, self.actor_obs_dim, device=device)
        critic_obs_buf = torch.zeros(chunks, n_envs, self.critic_obs_dim, device=device)
        actions_buf = torch.zeros(chunks, n_envs, h, self.num_act, device=device)
        raw_z_buf = torch.zeros(chunks, n_envs, self.chunk_dim, device=device)
        old_log_probs_buf = torch.zeros(chunks, n_envs, h, device=device)
        # per-frame storage
        reward_raw_buf = torch.zeros(chunks, n_envs, h, device=device)
        reward_masked_buf = torch.zeros(chunks, n_envs, h, device=device)
        alive_frame_buf = torch.zeros(chunks, n_envs, h, dtype=torch.bool, device=device)
        done_frame_buf = torch.zeros(chunks, n_envs, h, dtype=torch.bool, device=device)
        failure_frame_buf = torch.zeros(chunks, n_envs, h, dtype=torch.bool, device=device)
        timeout_frame_buf = torch.zeros(chunks, n_envs, h, dtype=torch.bool, device=device)
        motion_complete_frame_buf = torch.zeros(chunks, n_envs, h, dtype=torch.bool, device=device)
        next_critic_obs_buf = torch.zeros(chunks, n_envs, h, self.critic_obs_dim, device=device)

        obs = current_obs
        critic_obs = self._critic_obs
        action_abs_max = 0.0
        first_chunk_infos: list[dict] = []
        rollout_info_items: list[tuple] = []
        done_terms_union: dict[str, torch.Tensor] = {}
        collection_start_phases = (
            env.phase_steps.detach().clone()
            if hasattr(env, "phase_steps")
            else torch.zeros(n_envs, dtype=torch.long, device=device)
        )

        total_steps = chunks * h
        ever_done = torch.zeros(n_envs, dtype=torch.bool, device=device)
        first_done_step = torch.full((n_envs,), total_steps, dtype=torch.long, device=device)
        first_done_phase = torch.full((n_envs,), -1, dtype=torch.long, device=device)
        first_done_anchor_pos = torch.zeros(n_envs, dtype=torch.bool, device=device)
        first_done_anchor_ori = torch.zeros(n_envs, dtype=torch.bool, device=device)
        first_done_ee_body = torch.zeros(n_envs, dtype=torch.bool, device=device)
        first_done_timeout = torch.zeros(n_envs, dtype=torch.bool, device=device)
        first_done_motion_complete = torch.zeros(n_envs, dtype=torch.bool, device=device)
        metric_cross_chunk_delta_sum = torch.zeros((), device=device, dtype=obs.dtype)
        metric_cross_chunk_delta_count = torch.zeros((), device=device, dtype=obs.dtype)

        with torch.no_grad():
            for chunk_idx in range(chunks):
                actor_obs_n = self._norm_actor(obs)
                critic_obs_n = self._norm_critic(critic_obs)
                # The environment observation contract keeps last_action in the
                # final A columns. It is used only for execution diagnostics;
                # the actor random variable is raw target rate.
                prev_action = obs[..., -self.num_act:].detach()

                raw_z, _, conditional_logp = self._sample_final_cps(actor_obs_n)
                raw_z_chunk = raw_z.view(n_envs, h, self.num_act)
                alive_in_chunk = torch.ones(n_envs, dtype=torch.bool, device=device)

                for frame_idx in range(h):
                    alive_before_frame = alive_in_chunk.clone()
                    raw_target_t = raw_z_chunk[:, frame_idx, :]
                    next_obs, reward, done, info = env.step_raw_target_rate(
                        raw_target_t,
                        active_mask=alive_before_frame,
                        auto_reset=False,
                    )
                    applied_action = info["applied_action"]
                    if not torch.is_tensor(applied_action):
                        raise TypeError("info['applied_action'] must be a tensor")
                    applied_action = applied_action.detach().to(
                        device=device,
                        dtype=actions_buf.dtype,
                    )
                    if applied_action.shape != (n_envs, self.num_act):
                        raise ValueError(
                            "info['applied_action'] must have shape "
                            f"({n_envs}, {self.num_act}), got {tuple(applied_action.shape)}"
                        )
                    actions_buf[chunk_idx, :, frame_idx] = applied_action
                    action_abs_max = max(
                        action_abs_max,
                        float(applied_action.abs().max().item()),
                    )
                    next_critic_obs = env.get_critic_observation()
                    if chunk_idx == 0 and frame_idx == 0:
                        first_chunk_infos.append(info)

                    active_f = alive_before_frame.to(dtype=reward.dtype)
                    reward_raw_buf[chunk_idx, :, frame_idx] = reward.detach().to(dtype=reward_raw_buf.dtype)
                    reward_masked_buf[chunk_idx, :, frame_idx] = (
                        reward.detach().to(dtype=reward_masked_buf.dtype) * active_f
                    )
                    alive_frame_buf[chunk_idx, :, frame_idx] = alive_before_frame.detach()
                    next_critic_obs_buf[chunk_idx, :, frame_idx] = (
                        self._norm_critic(next_critic_obs, update=False).detach()
                    )

                    timeouts = info["done_terms"]["time_out"].bool()
                    motion_complete = info["done_terms"].get("motion_complete")
                    motion_complete = (
                        motion_complete.bool() if torch.is_tensor(motion_complete) else torch.zeros_like(timeouts)
                    )
                    failure_terms = (
                        info["done_terms"]["anchor_pos_bad"].bool()
                        | info["done_terms"]["anchor_ori_bad"].bool()
                        | info["done_terms"]["ee_body_bad"].bool()
                    )
                    done_b = done.bool()
                    new_done = alive_before_frame & done_b
                    new_timeout = new_done & timeouts
                    new_motion_complete = new_done & motion_complete
                    new_failure = new_done & failure_terms & (~timeouts) & (~motion_complete)

                    done_frame_buf[chunk_idx, :, frame_idx] = new_done.detach()
                    failure_frame_buf[chunk_idx, :, frame_idx] = new_failure.detach()
                    timeout_frame_buf[chunk_idx, :, frame_idx] = new_timeout.detach()
                    motion_complete_frame_buf[chunk_idx, :, frame_idx] = new_motion_complete.detach()

                    if bool(new_done.any()):
                        self._record_first_done(
                            new_done=new_done,
                            timeouts=timeouts,
                            info=info,
                            global_step=chunk_idx * h + frame_idx,
                            ever_done=ever_done,
                            first_done_step=first_done_step,
                            first_done_phase=first_done_phase,
                            first_done_anchor_pos=first_done_anchor_pos,
                            first_done_anchor_ori=first_done_anchor_ori,
                            first_done_ee_body=first_done_ee_body,
                            first_done_timeout=first_done_timeout,
                            first_done_motion_complete=first_done_motion_complete,
                        )
                    self._record_train_episode_stats(
                        (reward.detach().to(dtype=torch.float32) * active_f),
                        new_done.detach(),
                        step_counts=active_f.detach(),
                    )
                    rollout_info_items.append((info, alive_before_frame.detach()))
                    for key, val in info["done_terms"].items():
                        b = val.bool()
                        done_terms_union[key] = b.clone() if key not in done_terms_union else (done_terms_union[key] | b)

                    alive_in_chunk = alive_before_frame & ~done_b
                    obs = next_obs
                    critic_obs = next_critic_obs

                # Chunk-end reset: dead envs restart so the next chunk begins a
                # fresh episode; alive envs keep running (continuous clipped-ratio stream).
                chunk_done = done_frame_buf[chunk_idx].any(dim=-1)
                if bool(chunk_done.any()):
                    reset_ids = chunk_done.nonzero(as_tuple=False).squeeze(-1)
                    reset_phases = self.env.sample_phase_indices(reset_ids.numel(), horizon=max(1, h))
                    reset_obs = self.env.reset_envs(reset_ids, phase_indices=reset_phases)
                    obs[reset_ids] = reset_obs
                    critic_obs = self.env.get_critic_observation()

                chunk_first_action = actions_buf[chunk_idx, :, 0, :]
                # first-action delta = |a_0 - prev_action|, where prev_action is
                # the raw env last action at chunk start (from prev_action_buf).
                # This is the true cross-chunk continuity metric and is NOT
                # polluted by chunk-end resets (prev_action_buf[chunk_idx] holds
                # the env's last_action right before this chunk: the
                # phase-reference command for freshly reset envs, or the
                # previous chunk's last action for alive envs).
                first_action_delta = (chunk_first_action - prev_action).abs().mean(dim=-1)
                metric_cross_chunk_delta_sum = metric_cross_chunk_delta_sum + first_action_delta.sum()
                metric_cross_chunk_delta_count = metric_cross_chunk_delta_count + torch.tensor(
                    float(n_envs), device=device, dtype=obs.dtype
                )

                actor_obs_buf[chunk_idx] = actor_obs_n
                critic_obs_buf[chunk_idx] = critic_obs_n
                raw_z_buf[chunk_idx] = raw_z.detach()
                old_log_probs_buf[chunk_idx] = conditional_logp.detach()

            # ---- batched critic evaluation (state-only V at chunk start) ----
            critic_obs_flat = critic_obs_buf.reshape(chunks * n_envs, self.critic_obs_dim)
            values_v = self.critic.evaluate(critic_obs_flat).reshape(chunks, n_envs, 1)
            chunk_values = values_v.squeeze(-1)

            # Per-frame V is diagnostic only. The value loss below trains the
            # critic on chunk-start targets, not on per-frame targets.
            next_critic_obs_flat = next_critic_obs_buf.reshape(chunks * n_envs * h, self.critic_obs_dim)
            frame_next_values = self.critic.evaluate(next_critic_obs_flat).reshape(chunks, n_envs, h)
            frame_values = torch.cat(
                [
                    values_v.expand(-1, -1, h)[..., :1],  # frame 0: chunk-start V
                    frame_next_values[..., :-1],  # frame j>0: previous frame's next-V (diagnostic)
                ],
                dim=-1,
            )

            # ---- chunk return + terminal-aware bootstrap ----
            gamma_pow_reward = gamma ** torch.arange(h, device=device, dtype=reward_masked_buf.dtype)
            chunk_discounted_return = (reward_masked_buf * gamma_pow_reward.view(1, 1, h)).sum(dim=-1)

            frame_idx_grid = torch.arange(h, device=device).view(1, 1, h).expand(chunks, n_envs, h)
            masked_done_idx = torch.where(done_frame_buf, frame_idx_grid, torch.full_like(frame_idx_grid, h))
            death_frame = masked_done_idx.min(dim=-1).values
            chunk_done = death_frame < h
            chunk_failure = failure_frame_buf.any(dim=-1)
            boot_frame = death_frame.clamp(max=h - 1)
            boot_obs = next_critic_obs_buf.gather(
                2,
                boot_frame.view(chunks, n_envs, 1, 1).expand(-1, -1, 1, self.critic_obs_dim),
            ).squeeze(2)
            boot_values = self.critic.evaluate(boot_obs.reshape(chunks * n_envs, self.critic_obs_dim)).reshape(
                chunks, n_envs
            )
            boot_discount = gamma ** (boot_frame.to(dtype=boot_values.dtype) + 1.0)
            chunk_bootstrap = torch.where(
                chunk_failure,
                torch.zeros_like(boot_values),
                boot_discount * boot_values,
            )
            chunk_one_step_target = chunk_discounted_return + chunk_bootstrap

            # ---- cross-chunk GAE on macro transitions ----
            chunk_td_delta = chunk_one_step_target - chunk_values
            chunk_advantages = torch.zeros_like(chunk_td_delta)
            gae = torch.zeros(n_envs, device=device, dtype=chunk_td_delta.dtype)
            gae_lambda = float(getattr(self.cfg, "gae_lambda", 0.95))
            chunk_gamma_lambda = (gamma ** h) * (gae_lambda ** h)
            chunk_cont = (~chunk_done).to(dtype=chunk_td_delta.dtype)
            for chunk_i in range(chunks - 1, -1, -1):
                gae = chunk_td_delta[chunk_i] + chunk_gamma_lambda * chunk_cont[chunk_i] * gae
                chunk_advantages[chunk_i] = gae

            chunk_v_targets = chunk_values + chunk_advantages

            # Actor advantage = chunk GAE advantage. Each executed frame ratio
            # carries the same chunk-level credit.
            valid_prefix_mask = alive_frame_buf.clone()
            chunk_valid_mask = alive_frame_buf[..., 0].clone()
            normalized_chunk_advantages = self._normalize_chunk_advantages(chunk_advantages, chunk_valid_mask)
            advantages = normalized_chunk_advantages.unsqueeze(-1).expand(-1, -1, h) * valid_prefix_mask
            raw_gae_advantages = chunk_advantages.unsqueeze(-1).expand(-1, -1, h) * valid_prefix_mask
            frame_v_targets = chunk_v_targets.unsqueeze(-1).expand_as(frame_values)

            # ---- diagnostic prefix targets (for logging only) ----
            zero_bootstrap = torch.zeros_like(frame_next_values)
            frame_bootstrap = torch.where(failure_frame_buf, zero_bootstrap, frame_next_values)
            gamma_pow_boot = gamma ** torch.arange(1, h + 1, device=device, dtype=frame_bootstrap.dtype)
            disc_rewards = reward_masked_buf * gamma_pow_reward.view(1, 1, h)
            cum_disc_rewards = torch.cumsum(disc_rewards, dim=-1)
            disc_bootstrap = frame_bootstrap * gamma_pow_boot.view(1, 1, h)
            prefix_targets = cum_disc_rewards + disc_bootstrap  # T_{k+1}, diagnostic
            v_targets = chunk_v_targets.unsqueeze(-1)

        self._obs = obs
        self._critic_obs = critic_obs
        chunk_return_realized = chunk_one_step_target  # [chunks, n_envs], reward plus terminal-aware bootstrap
        # chunk_raw_return: target-side raw, used for advantage/value computations.
        chunk_raw_return = (reward_masked_buf * gamma_pow_reward.view(1, 1, h)).sum(dim=-1)
        # chunk_env_raw_return: environment-side raw reward sum.
        alive_f = alive_frame_buf.to(dtype=reward_raw_buf.dtype)
        chunk_env_raw_return = (reward_raw_buf * alive_f * gamma_pow_reward.view(1, 1, h)).sum(dim=-1)
        # No hand-written failure cost remains; this diagnostic should stay 0.
        failure_cost_return = chunk_raw_return - chunk_env_raw_return
        chunk_live_frames = alive_frame_buf.to(dtype=torch.float32).sum(dim=-1)  # [chunks, n_envs]

        return {
            "actor_obs": actor_obs_buf,
            "critic_obs": critic_obs_buf,
            "actions": actions_buf,
            "raw_z": raw_z_buf,
            "old_log_probs": old_log_probs_buf,
            "values_v": values_v,
            "frame_values": frame_values,
            "frame_v_targets": frame_v_targets,
            "chunk_values": chunk_values,
            "chunk_v_targets": chunk_v_targets,
            "chunk_advantages": chunk_advantages,
            "chunk_one_step_target": chunk_one_step_target,
            "chunk_bootstrap": chunk_bootstrap,
            "chunk_cont": chunk_cont,
            "chunk_valid_mask": chunk_valid_mask,
            "prefix_targets": prefix_targets,
            "v_targets": v_targets,
            "advantages": advantages,
            "raw_gae_advantages": raw_gae_advantages,
            "valid_prefix_mask": valid_prefix_mask,
            "death_frame": death_frame,
            "frame_bootstrap": frame_bootstrap,
            "frame_next_values": frame_next_values,
            "reward_raw": reward_raw_buf,
            "reward_masked": reward_masked_buf,
            "chunk_env_raw_return": chunk_env_raw_return,
            "failure_cost_return": failure_cost_return,
            "alive_frame": alive_frame_buf,
            "done_frame": done_frame_buf,
            "failure_frame": failure_frame_buf,
            "timeout_frame": timeout_frame_buf,
            "motion_complete_frame": motion_complete_frame_buf,
            "next_critic_obs": next_critic_obs_buf,
            "chunk_return_realized": chunk_return_realized,
            "chunk_raw_return": chunk_raw_return,
            "chunk_live_frames": chunk_live_frames,
            "done_terms_union": done_terms_union,
            "rollout_info_items": rollout_info_items,
            "first_chunk_infos": first_chunk_infos,
            "first_done_step": first_done_step,
            "first_done_phase": first_done_phase,
            "first_done_ee_body": first_done_ee_body,
            "first_done_anchor_pos": first_done_anchor_pos,
            "first_done_anchor_ori": first_done_anchor_ori,
            "first_done_timeout": first_done_timeout,
            "first_done_motion_complete": first_done_motion_complete,
            "collection_start_phases": collection_start_phases,
            "metric_cross_chunk_delta_sum": metric_cross_chunk_delta_sum,
            "metric_cross_chunk_delta_count": metric_cross_chunk_delta_count,
            "action_abs_max": action_abs_max,
            "next_observation": obs,
        }

    def _normalize_advantages(self, adv: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Normalize per-prefix (default), globally, or not at all.

        Invalid prefixes (post-death) are masked to zero after normalization.
        """
        if self.advantage_normalization == "none":
            return adv * mask
        if self.advantage_normalization == "global":
            valid = adv[mask]
            if valid.numel() == 0:
                return adv * mask
            mean = valid.mean()
            std = valid.std(unbiased=False) + 1.0e-8
            return ((adv - mean) / std) * mask
        # per_prefix
        out = torch.zeros_like(adv)
        for k in range(self.horizon_h):
            col_mask = mask[..., k]
            col = adv[..., k][col_mask]
            if col.numel() == 0:
                continue
            mean = col.mean()
            std = col.std(unbiased=False) + 1.0e-8
            out[..., k] = (adv[..., k] - mean) / std
        return out * mask

    def _normalize_chunk_advantages(self, adv: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Normalize chunk advantages over valid chunk starts."""
        if self.advantage_normalization == "none":
            return adv * mask
        valid = adv[mask]
        if valid.numel() == 0:
            return adv * mask
        mean = valid.mean()
        std = valid.std(unbiased=False) + 1.0e-8
        return ((adv - mean) / std) * mask

    # ------------------------------------------------------------------ #
    # Update: per-frame ratio + chunk advantage + chunk critic loss
    # ------------------------------------------------------------------ #
    def _policy_mini_batch_size(self, sample_count: int) -> int:
        num_mini_batches = max(1, int(self.cfg.num_mini_batches))
        return max(1, math.ceil(sample_count / num_mini_batches))

    def _policy_micro_batch_size(self, batch_size: int) -> int:
        if int(self.cfg.micro_batch_size) <= 0:
            return max(1, batch_size)
        return max(1, min(batch_size, int(self.cfg.micro_batch_size)))

    def _update_adaptive_learning_rates(self, observed_kl: float) -> None:
        # Only the ACTOR learning rate is tied to the policy KL. The critic LR
        # is decoupled: it stays at value_lr so an aggressive KL-driven actor
        # slowdown does not also starve the state-V critic of gradient signal.
        desired_kl = float(self.cfg.desired_kl)
        if desired_kl <= 0.0:
            return
        new_actor_lr, _ = adaptive_lr_from_kl(
            raw_kl=observed_kl,
            kl_units=self.kl_units,
            target_per_step=desired_kl,
            lr=self.learning_rate,
            min_lr=self.min_lr,
            max_lr=self.max_lr,
        )
        self.learning_rate = new_actor_lr
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.learning_rate

    @abstractmethod
    def update(self, rollout: dict, collect_time: float) -> dict:
        """Reference updater for tests; production FCAMP supplies its own updater.

        Keeping this implementation behind an abstract method lets the isolated
        Flow-CPS test harness exercise the historical reference algorithm without
        making ``FlowCPSBase`` a constructible production algorithm.
        """
        import time as _time

        device = self.env.device
        chunks, n_envs = rollout["actions"].shape[:2]
        h = self.horizon_h
        raw_batch_size = chunks * n_envs

        actor_obs = rollout["actor_obs"].reshape(raw_batch_size, self.actor_obs_dim)
        critic_obs = rollout["critic_obs"].reshape(raw_batch_size, self.critic_obs_dim)
        raw_z = rollout["raw_z"].reshape(raw_batch_size, self.chunk_dim)
        old_log_probs = rollout["old_log_probs"].reshape(raw_batch_size, h)
        chunk_v_targets = rollout["chunk_v_targets"].reshape(raw_batch_size)
        advantages = rollout["advantages"].reshape(raw_batch_size, h)
        raw_gae_advantages = rollout["raw_gae_advantages"].reshape(raw_batch_size, h)
        valid_prefix_mask = rollout["valid_prefix_mask"].reshape(raw_batch_size, h)
        chunk_valid_mask = rollout["chunk_valid_mask"].reshape(raw_batch_size)
        batch_size = actor_obs.shape[0]

        mini_batch_size = self._policy_mini_batch_size(batch_size)
        clip_range = float(self.cfg.clip_range)
        value_coef = float(self.cfg.value_loss_coef)
        probe_count = min(128, batch_size)
        with torch.no_grad():
            probe_obs = actor_obs[:probe_count]
            probe_raw_before = self._deterministic_actor_raw_targets(probe_obs)
            params_before = [p.detach().clone() for p in self._policy.parameters()]

        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "loss": 0.0,
            "ratio": 0.0,
            "ratio_min": float("inf"),
            "ratio_max": 0.0,
            "clip_frac": 0.0,
            "kl_loss": 0.0,
            "logprob_delta_abs": 0.0,
            "old_log_prob": 0.0,
            "new_log_prob": 0.0,
            "grad_norm": 0.0,
            "grad_norm_critic": 0.0,
            "v_target_mean": 0.0,
            "valid_prefix_frac": 0.0,
            "early_stop_epoch": float(int(self.cfg.policy_epochs)),
        }
        per_frame_kl_sum = torch.zeros(h, device=device)
        per_frame_ratio_sum = torch.zeros(h, device=device)
        per_frame_clip_sum = torch.zeros(h, device=device)
        per_frame_adv_abs_sum = torch.zeros(h, device=device)
        per_frame_count = torch.zeros(h, device=device)
        update_count = 0
        micro_batch_count = 0
        early_stopped_epoch = int(self.cfg.policy_epochs)

        t1 = _time.perf_counter()
        actor_frozen = False  # set True by KL early-stop; critic keeps training
        for epoch in range(int(self.cfg.policy_epochs)):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mini_batch_size):
                idx = perm[start:start + mini_batch_size]
                if idx.numel() == 0:
                    continue
                mb_size = int(idx.numel())
                micro_batch_size = self._policy_micro_batch_size(mb_size)
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)

                mb_totals = {k: 0.0 for k in (
                    "policy_loss", "value_loss", "loss", "ratio",
                    "clip_frac", "kl_loss", "logprob_delta_abs", "old_log_prob",
                    "new_log_prob", "v_target_mean", "valid_prefix_frac",
                )}
                mb_ratio_min = float("inf")
                mb_ratio_max = 0.0
                # accumulate per-frame log-ratio across micro-batches for the
                # prefix-KL controller (computed once per minibatch, before the
                # optimizer step -- clipped-ratio-aligned).
                mb_prefix_kl_accum = torch.zeros(h, device=device)
                mb_prefix_kl_count = torch.zeros(h, device=device)
                for micro_start in range(0, mb_size, micro_batch_size):
                    micro_end = min(micro_start + micro_batch_size, mb_size)
                    sub = idx[micro_start:micro_end]
                    weight = float(sub.numel()) / float(mb_size)

                    old_per_frame_logp = old_log_probs[sub]
                    new_per_frame_logp = self._recompute_final_cps_log_prob(
                        actor_obs[sub],
                        raw_z[sub],
                    )
                    per_frame_log_ratio = new_per_frame_logp - old_per_frame_logp  # [micro, h]
                    # Match the MixGRPO/Flow-CPS training object: the KL proxy
                    # is defined on the same log-prob surrogate used by the
                    # clipped policy ratio, not on a separate Gaussian transition KL.
                    kl_per_frame = 0.5 * per_frame_log_ratio.square()
                    # Per-frame clipped policy ratio (NOT a prefix/joint cumsum ratio).
                    # Each executed chunk component gets its own per-frame
                    # likelihood ratio, but all executed frames carry the same
                    # chunk GAE advantage.
                    ratio = torch.exp(per_frame_log_ratio)  # [micro, h]
                    adv = advantages[sub]  # [micro, h]
                    mask = valid_prefix_mask[sub].to(dtype=ratio.dtype)  # [micro, h]
                    mask_sum = mask.sum().clamp(min=1.0)

                    # Flat clipped-ratio clip (same [1-eps, 1+eps] for every frame).
                    clip_low = 1.0 - clip_range
                    clip_high = 1.0 + clip_range
                    unclipped = -adv * ratio
                    clipped = -adv * torch.clamp(ratio, clip_low, clip_high)
                    policy_loss_per = torch.maximum(unclipped, clipped)  # [micro, h]
                    policy_loss = (policy_loss_per * mask).sum() / mask_sum

                    # Chunk critic loss: flow V models f366's chunk-GAE value
                    # targets. Actor credit remains the original cross-chunk
                    # GAE; only the critic function class changes.
                    critic_obs_sub = critic_obs[sub]  # [micro, D]
                    chunk_mask = chunk_valid_mask[sub].to(dtype=critic_obs_sub.dtype)
                    chunk_mask_sum = chunk_mask.sum().clamp(min=1.0)
                    value_loss_per = self.critic.flow_matching_loss_v(
                        critic_obs_sub,
                        chunk_v_targets[sub],
                        fm_samples=self.flow_critic_fm_samples,
                    )
                    value_loss = (value_loss_per * chunk_mask).sum() / chunk_mask_sum

                    # When the KL early-stop has frozen the actor, only the
                    # critic loss is backpropagated and the actor optimizer step
                    # is skipped. This lets the state-V critic keep learning
                    # while the policy is held still. logged_loss is always
                    # defined (policy_loss + value_loss) so the diagnostics below
                    # never read a stale variable; the frozen branch just does
                    # not backprop the policy part.
                    logged_loss = policy_loss + value_coef * value_loss
                    if actor_frozen:
                        (value_coef * value_loss * weight).backward()
                    else:
                        (logged_loss * weight).backward()

                    with torch.no_grad():
                        # prefix KL (cumsum per-frame log-ratio / prefix length):
                        # the KL controller must be in the same units as the actor
                        # ratio (which is per-frame here), but we also track the
                        # prefix KL for early-stop diagnostics. The adaptive LR
                        # uses the masked mean per-frame KL (kl_units=1).
                        kl_per_frame_mean = (kl_per_frame * mask).sum() / mask_sum
                        mb_prefix_kl_accum += (kl_per_frame * mask).sum(dim=0).detach()
                        mb_prefix_kl_count += mask.sum(dim=0).detach()
                        clip_per_frame = ((ratio < clip_low) | (ratio > clip_high)).to(dtype=ratio.dtype)
                        mb_totals["policy_loss"] += float(policy_loss.item()) * weight
                        mb_totals["value_loss"] += float(value_loss.item()) * weight
                        mb_totals["loss"] += float(logged_loss.item()) * weight
                        mb_totals["ratio"] += float((ratio * mask).sum().item() / mask_sum.item()) * weight
                        mb_totals["clip_frac"] += float((clip_per_frame * mask).sum().item() / mask_sum.item()) * weight
                        mb_totals["kl_loss"] += float(kl_per_frame_mean.item()) * weight
                        mb_totals["logprob_delta_abs"] += float((per_frame_log_ratio.abs() * mask).sum().item() / mask_sum.item()) * weight
                        mb_totals["old_log_prob"] += float((old_per_frame_logp * mask).sum().item() / mask_sum.item()) * weight
                        mb_totals["new_log_prob"] += float((new_per_frame_logp * mask).sum().item() / mask_sum.item()) * weight
                        mb_totals["v_target_mean"] += float(
                            (chunk_v_targets[sub] * chunk_mask).sum().item() / chunk_mask_sum.item()
                        ) * weight
                        mb_totals["valid_prefix_frac"] += float(mask.mean().item()) * weight
                        mb_ratio_min = min(mb_ratio_min, float(ratio.min().item()))
                        mb_ratio_max = max(mb_ratio_max, float(ratio.max().item()))
                        per_frame_kl_sum += (kl_per_frame * mask).sum(dim=0)
                        per_frame_ratio_sum += (ratio * mask).sum(dim=0)
                        per_frame_clip_sum += (clip_per_frame * mask).sum(dim=0)
                        per_frame_adv_abs_sum += (adv.abs() * mask).sum(dim=0)
                        per_frame_count += mask.sum(dim=0)
                    micro_batch_count += 1

                # ---- KL controller: update ACTOR LR BEFORE the optimizer step ----
                # (clipped-ratio-aligned, critic LR is decoupled and stays at value_lr).
                # Skipped when the actor is already frozen by an earlier epoch's
                # early-stop.
                if not actor_frozen:
                    self._update_adaptive_learning_rates(mb_totals["kl_loss"])

                grad_norm_critic = nn.utils.clip_grad_norm_(self.critic.parameters(), float(self.cfg.max_grad_norm))
                self.critic_optimizer.step()
                if actor_frozen:
                    grad_norm = torch.tensor(0.0, device=device)
                else:
                    grad_norm = nn.utils.clip_grad_norm_(self._policy.parameters(), float(self.cfg.max_grad_norm))
                    self.actor_optimizer.step()

                for key in mb_totals:
                    totals[key] += mb_totals[key]
                totals["ratio_min"] = min(totals["ratio_min"], mb_ratio_min)
                totals["ratio_max"] = max(totals["ratio_max"], mb_ratio_max)
                totals["grad_norm"] += float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                totals["grad_norm_critic"] += float(
                    grad_norm_critic.item() if torch.is_tensor(grad_norm_critic) else grad_norm_critic
                )
                update_count += 1

            # ---- actor epoch early-stop on prefix KL ----
            # ---- actor epoch early-stop on per-frame KL ----
            # If the epoch-averaged KL exceeds kl_early_stop_factor * desired_kl,
            # freeze the actor for the remaining epochs (skip policy loss
            # backprop + actor step) but keep training the critic. This avoids
            # destructive policy moves while the state-V critic keeps learning.
            desired_kl = float(self.cfg.desired_kl)
            if (
                not actor_frozen
                and desired_kl > 0.0
                and update_count > 0
            ):
                epoch_kl = totals["kl_loss"] / max(update_count, 1)
                if epoch_kl > self.kl_early_stop_factor * desired_kl:
                    actor_frozen = True
                    early_stopped_epoch = epoch + 1

        update_time = _time.perf_counter() - t1
        denom = max(update_count, 1)
        kl_raw = totals["kl_loss"] / denom
        kl_units = self.kl_units
        kl_per_step = kl_raw / kl_units
        pf_count = per_frame_count.clamp(min=1.0)
        per_frame_kl_mean = (per_frame_kl_sum / pf_count).detach().cpu().tolist()
        per_frame_ratio_mean = (per_frame_ratio_sum / pf_count).detach().cpu().tolist()
        per_frame_clip_mean = (per_frame_clip_sum / pf_count).detach().cpu().tolist()
        per_frame_adv_abs = (per_frame_adv_abs_sum / pf_count).detach().cpu().tolist()

        with torch.no_grad():
            probe_raw_after = self._deterministic_actor_raw_targets(probe_obs)
            raw_z_mean_delta = torch.mean(
                torch.abs(probe_raw_after - probe_raw_before)
            )
            param_delta_sq = torch.zeros((), device=device)
            param_count = 0
            for param, before in zip(self._policy.parameters(), params_before, strict=True):
                delta = param.detach() - before
                param_delta_sq = param_delta_sq + torch.sum(delta * delta)
                param_count += delta.numel()
            param_rms_delta = torch.sqrt(param_delta_sq / max(param_count, 1))

        update_metrics = {
            "flow_cps/loss": totals["loss"] / denom,
            "flow_cps/policy_loss": totals["policy_loss"] / denom,
            "flow_cps/value_loss": totals["value_loss"] / denom,
            "flow_cps/ratio": totals["ratio"] / denom,
            "flow_cps/ratio_min": totals["ratio_min"] if totals["ratio_min"] != float("inf") else 0.0,
            "flow_cps/ratio_max": totals["ratio_max"],
            "flow_cps/clip_frac": totals["clip_frac"] / denom,
            "flow_cps/kl_loss": kl_raw,
            "flow_cps/kl_raw": kl_raw,
            "flow_cps/kl_per_step": kl_per_step,
            "flow_cps/kl_target_per_step": desired_kl,
            "flow_cps/kl_target_raw": desired_kl * kl_units,
            "flow_cps/kl_units": float(kl_units),
            "flow_cps/logprob_delta_abs": totals["logprob_delta_abs"] / denom,
            "flow_cps/old_log_prob": totals["old_log_prob"] / denom,
            "flow_cps/new_log_prob": totals["new_log_prob"] / denom,
            "flow_cps/grad_norm": totals["grad_norm"] / denom,
            "flow_cps/grad_norm_critic": totals["grad_norm_critic"] / denom,
            "flow_cps/lr": self.learning_rate,
            "flow_cps/critic_lr": self.critic_learning_rate,
            "flow_cps/effective_mini_batch_size": float(mini_batch_size),
            "flow_cps/sample_count": float(batch_size),
            "flow_cps/raw_sample_count": float(raw_batch_size),
            "flow_cps/micro_batch_size": float(self._policy_micro_batch_size(mini_batch_size)),
            "flow_cps/micro_batches": float(micro_batch_count),
            "flow_cps/optimizer_steps": float(update_count),
            "flow_cps/cps_distribution_count": 1.0,
            "flow_cps/v_target_mean": totals["v_target_mean"] / denom,
            "flow_cps/valid_prefix_frac": totals["valid_prefix_frac"] / denom,
            "flow_cps/gae_lambda": float(getattr(self.cfg, "gae_lambda", 0.95)),
            "flow_cps/critic_unit": 1.0,  # 1 = chunk
            "flow_cps/actor_advantage_unit": 1.0,  # 1 = chunk GAE
            "flow_cps/kl_early_stop_factor": float(self.kl_early_stop_factor),
            "flow_cps/early_stop_epoch": float(early_stopped_epoch),
            "flow_cps/advantage_normalization": float({"per_prefix": 0.0, "global": 1.0, "none": 2.0}[self.advantage_normalization]),
            "policy/raw_z_mean_delta": float(raw_z_mean_delta.item()),
            "policy/param_rms_delta": float(param_rms_delta.item()),
            "policy/raw_adv_mean": float(raw_gae_advantages.mean().item()),
            "policy/raw_adv_std": float(raw_gae_advantages.std(unbiased=False).item()),
        }
        update_metrics.update(self._final_cps_statistics())
        for k in range(h):
            update_metrics[f"flow_cps/kl_frame_{k}"] = float(per_frame_kl_mean[k]) if k < len(per_frame_kl_mean) else float("nan")
            update_metrics[f"flow_cps/ratio_frame_{k}"] = float(per_frame_ratio_mean[k]) if k < len(per_frame_ratio_mean) else float("nan")
            update_metrics[f"flow_cps/clip_frame_{k}"] = float(per_frame_clip_mean[k]) if k < len(per_frame_clip_mean) else float("nan")
            update_metrics[f"flow_cps/adv_abs_frame_{k}"] = float(per_frame_adv_abs[k]) if k < len(per_frame_adv_abs) else float("nan")
        return self._build_metrics(rollout, update_metrics, collect_time, update_time)

    # ------------------------------------------------------------------ #
    # Metrics & logging
    # ------------------------------------------------------------------ #
    def _build_metrics(self, rollout, update_metrics, collect_time, update_time) -> dict:
        actions = rollout["actions"]
        alive = rollout["alive_frame"].to(dtype=actions.dtype)
        reward_raw = rollout["reward_raw"]
        reward_masked = rollout["reward_masked"]
        done_frame = rollout["done_frame"]
        failure_frame = rollout["failure_frame"]
        timeout_frame = rollout["timeout_frame"]
        death_frame = rollout["death_frame"]
        valid_prefix_mask = rollout["valid_prefix_mask"].to(dtype=actions.dtype)
        prefix_targets = rollout["prefix_targets"]
        v_targets = rollout["v_targets"]
        values_v = rollout["values_v"]
        frame_values = rollout["frame_values"]
        frame_v_targets = rollout["frame_v_targets"]
        chunk_values = rollout["chunk_values"]
        chunk_v_targets = rollout["chunk_v_targets"]
        chunk_advantages = rollout["chunk_advantages"]
        chunk_one_step_target = rollout["chunk_one_step_target"]
        chunk_bootstrap = rollout["chunk_bootstrap"]
        chunk_cont = rollout["chunk_cont"]
        chunk_valid_mask = rollout["chunk_valid_mask"]
        chunk_realized = rollout["chunk_return_realized"]
        chunk_raw = rollout["chunk_raw_return"]  # target-side (with failure cost)
        chunk_env_raw = rollout["chunk_env_raw_return"]  # env-side (no failure cost)
        failure_cost_return = rollout["failure_cost_return"]
        live_steps = rollout["chunk_live_frames"]
        first_done_step = rollout["first_done_step"]
        # first_done is ANY first termination (failure, timeout, motion_complete).
        # For logging, exclude timeout and motion_complete so "failed" matches
        # the failure_frame definition used in absorbing-failure targets.
        first_done_timeout_flag = rollout["first_done_timeout"]
        first_done_motion_complete = rollout["first_done_motion_complete"]
        failed = (
            (first_done_step < self._training_rollout_horizon())
            & (~first_done_timeout_flag)
            & (~first_done_motion_complete)
        )
        timeout = rollout["first_done_timeout"]
        metric_cross_chunk_delta_sum = rollout.get("metric_cross_chunk_delta_sum")
        metric_cross_chunk_delta_count = rollout.get("metric_cross_chunk_delta_count")
        if metric_cross_chunk_delta_sum is not None and float(metric_cross_chunk_delta_count.item()) > 0.0:
            cross_chunk_delta = float((metric_cross_chunk_delta_sum / metric_cross_chunk_delta_count).item())
        else:
            cross_chunk_delta = float("nan")

        h = self.horizon_h
        valid_raw_rewards = chunk_raw.reshape(-1)
        valid_realized = chunk_realized.reshape(-1)
        valid_dones = done_frame.any(dim=-1).reshape(-1)
        safe_live_steps = live_steps.clamp(min=1.0)
        reward_per_live_step = chunk_raw / safe_live_steps
        official_scale_reward = reward_per_live_step * self.max_episode_steps

        per_frame_abs = actions.abs().mean(dim=-1)
        frame0_abs = self._alive_weighted_mean(per_frame_abs[..., 0], alive[..., 0])
        frame1_abs = self._alive_weighted_mean(per_frame_abs[..., 1], alive[..., 1]) if actions.shape[2] > 1 else float("nan")
        applied_abs = per_frame_abs[rollout["alive_frame"]]
        if applied_abs.numel() > 0:
            action_abs_mean = float(applied_abs.mean().item())
            action_abs_p95 = float(torch.quantile(applied_abs.flatten(), 0.95).item())
            action_abs_p99 = float(torch.quantile(applied_abs.flatten(), 0.99).item())
            action_abs_max = float(applied_abs.max().item())
        else:
            action_abs_mean = action_abs_p95 = action_abs_p99 = action_abs_max = 0.0
        if actions.shape[2] > 1:
            delta = (actions[..., 1:, :] - actions[..., :-1, :]).abs().mean(dim=-1)
            in_chunk_delta = self._alive_weighted_mean(delta, alive[..., 1:])
        else:
            in_chunk_delta = float("nan")

        metrics = {
            **update_metrics,
            "method/name": "flow_cps_base",
            "rollout/reward_step_mean": float(
                (reward_raw * alive).sum().item() / max(float(alive.sum().item()), 1.0)
            ),
            "rollout/chunk_return_mean": float(valid_realized.mean().item()) if valid_realized.numel() > 0 else 0.0,
            "rollout/chunk_return_std": float(valid_realized.std(unbiased=False).item()) if valid_realized.numel() > 0 else 0.0,
            "rollout/chunk_raw_return_mean": float(valid_raw_rewards.mean().item()) if valid_raw_rewards.numel() > 0 else 0.0,
            # chunk_env_raw: true per-step env reward sum (NO failure cost).
            "rollout/chunk_env_raw_return_mean": float(chunk_env_raw.mean().item()),
            # failure_cost: target-side raw minus env-side raw (negative). Near 0
            # -> survival-driven; very negative -> failure-dominated.
            "rollout/failure_cost_return_mean": float(failure_cost_return.mean().item()),
            "rollout/done_frac": float(valid_dones.float().mean().item()) if valid_dones.numel() > 0 else 0.0,
            "rollout/failure_frac": float(failure_frame.any(dim=-1).float().mean().item()),
            "rollout/timeout_frac": float(timeout_frame.any(dim=-1).float().mean().item()),
            "rollout/live_steps_mean": float(live_steps.mean().item()),
            "rollout/live_steps_min": float(live_steps.min().item()),
            "rollout/live_steps_p50": float(torch.quantile(live_steps, 0.50).item()),
            "rollout/live_steps_p95": float(torch.quantile(live_steps, 0.95).item()),
            "rollout/live_steps_max": float(live_steps.max().item()),
            "rollout/reward_per_live_step": float(reward_per_live_step.mean().item()),
            "rollout/max_episode_return_projection": float(official_scale_reward.mean().item()),
            "rollout/success_frac": float((~failed).float().mean().item()),
            "rollout/failure_frac_first_done": float((failed & (~timeout)).float().mean().item()),
            "rollout/first_failure_step_mean": float(first_done_step[failed].float().mean().item()) if bool(failed.any()) else float("nan"),
            "rollout/first_failure_step_min": float(first_done_step[failed].min().item()) if bool(failed.any()) else float("nan"),
            "rollout/first_failure_step_max": float(first_done_step[failed].max().item()) if bool(failed.any()) else float("nan"),
            "rollout/death_frame_mean": float(death_frame.float().mean().item()),
            "rollout/valid_prefix_frac": float(valid_prefix_mask.mean().item()),
            "phase/start_mean": float(rollout["collection_start_phases"].float().mean().item()),
            "phase/start_min": float(rollout["collection_start_phases"].min().item()),
            "phase/start_max": float(rollout["collection_start_phases"].max().item()),
            "timing/collect_s": collect_time,
            "timing/update_s": update_time,
            "act/abs_mean": action_abs_mean,
            "act/abs_p95": action_abs_p95,
            "act/abs_p99": action_abs_p99,
            "act/abs_max": action_abs_max,
            "act/abs_max_all": float(rollout.get("action_abs_max", 0.0)),
            "act/frame0_abs_mean": frame0_abs,
            "act/frame1_abs_mean": frame1_abs,
            "act/in_chunk_delta_abs": in_chunk_delta,
            "act/cross_chunk_delta_abs": cross_chunk_delta,
        }
        # per-frame critic diagnostics (V only; Q prefix dropped)
        for k in range(h):
            col_mask = rollout["valid_prefix_mask"][..., k]
            if bool(col_mask.any()):
                metrics[f"critic/frame_v_{k+1}_mean"] = float(frame_values[..., k][col_mask].mean().item())
                metrics[f"critic/frame_v_target_{k+1}_mean"] = float(frame_v_targets[..., k][col_mask].mean().item())
                metrics[f"critic/prefix_target_{k+1}_mean"] = float(prefix_targets[..., k][col_mask].mean().item())
                metrics[f"critic/valid_prefix_{k+1}_frac"] = float(col_mask.float().mean().item())
            else:
                metrics[f"critic/frame_v_{k+1}_mean"] = float("nan")
                metrics[f"critic/frame_v_target_{k+1}_mean"] = float("nan")
                metrics[f"critic/prefix_target_{k+1}_mean"] = float("nan")
                metrics[f"critic/valid_prefix_{k+1}_frac"] = 0.0
        metrics["critic/v_mean"] = float(values_v.mean().item())
        metrics["critic/v_target_mean"] = float(v_targets.mean().item())
        if bool(chunk_valid_mask.any()):
            metrics["critic/chunk_v_mean"] = float(chunk_values[chunk_valid_mask].mean().item())
            metrics["critic/chunk_v_target_mean"] = float(chunk_v_targets[chunk_valid_mask].mean().item())
            metrics["critic/chunk_one_step_target_mean"] = float(
                chunk_one_step_target[chunk_valid_mask].mean().item()
            )
            metrics["critic/chunk_bootstrap_mean"] = float(chunk_bootstrap[chunk_valid_mask].mean().item())
            metrics["critic/chunk_adv_mean"] = float(chunk_advantages[chunk_valid_mask].mean().item())
            metrics["critic/chunk_adv_std"] = float(chunk_advantages[chunk_valid_mask].std(unbiased=False).item())
            metrics["critic/chunk_cont_frac"] = float(chunk_cont[chunk_valid_mask].mean().item())
        else:
            metrics["critic/chunk_v_mean"] = float("nan")
            metrics["critic/chunk_v_target_mean"] = float("nan")
            metrics["critic/chunk_one_step_target_mean"] = float("nan")
            metrics["critic/chunk_bootstrap_mean"] = float("nan")
            metrics["critic/chunk_adv_mean"] = float("nan")
            metrics["critic/chunk_adv_std"] = float("nan")
            metrics["critic/chunk_cont_frac"] = 0.0
        metrics["critic/chunk_valid_frac"] = float(chunk_valid_mask.float().mean().item())

        for key, mask in rollout["done_terms_union"].items():
            metrics[f"done/{key}_frac"] = float(mask.float().mean().item())
        self._add_first_failure_metrics(metrics, rollout, failed)
        self._add_reward_metrics(metrics, rollout)
        self._add_action_group_metrics(metrics, actions, alive)
        self._add_sampler_metrics(metrics)
        self._add_reward_weighted_metrics(metrics)
        if self._train_reward_buffer:
            metrics["train/mean_reward"] = float(sum(self._train_reward_buffer) / len(self._train_reward_buffer))
            metrics["train/mean_episode_length"] = float(sum(self._train_length_buffer) / len(self._train_length_buffer))
        else:
            metrics["train/mean_reward"] = float("nan")
            metrics["train/mean_episode_length"] = float("nan")
        metrics["train/recent_episode_count"] = float(len(self._train_reward_buffer))
        metrics["train/completed_episodes"] = float(self._train_completed_episodes)
        return metrics

    def _alive_weighted_mean(self, values: torch.Tensor, weights: torch.Tensor) -> float:
        denom = weights.sum()
        if float(denom.item()) <= 0.0:
            return float("nan")
        return float(((values * weights).sum() / denom).item())

    def _add_first_failure_metrics(self, metrics: dict, rollout: dict, failed: torch.Tensor) -> None:
        phase = rollout["first_done_phase"]
        valid_phase = phase[phase >= 0]
        if valid_phase.numel() > 0:
            metrics["rollout/first_failure_phase_mean"] = float(valid_phase.float().mean().item())
            metrics["rollout/first_failure_phase_min"] = float(valid_phase.min().item())
            metrics["rollout/first_failure_phase_max"] = float(valid_phase.max().item())
        else:
            metrics["rollout/first_failure_phase_mean"] = float("nan")
            metrics["rollout/first_failure_phase_min"] = float("nan")
            metrics["rollout/first_failure_phase_max"] = float("nan")
        metrics["rollout/first_failure_anchor_pos_frac"] = float((rollout["first_done_anchor_pos"] & failed).float().mean().item())
        metrics["rollout/first_failure_anchor_ori_frac"] = float((rollout["first_done_anchor_ori"] & failed).float().mean().item())
        metrics["rollout/first_failure_ee_body_frac"] = float((rollout["first_done_ee_body"] & failed).float().mean().item())
        metrics["rollout/first_failure_timeout_frac"] = float((rollout["first_done_timeout"] & failed).float().mean().item())

    def _add_reward_metrics(self, metrics: dict, rollout: dict) -> None:
        first_chunk_infos = rollout["first_chunk_infos"]
        for info in first_chunk_infos:
            for key, value in info["reward_terms"].items():
                metrics[f"reward/{key}_mean"] = float(value.mean().item())

        reward_sums: dict[str, float] = {}
        weight_sum = 0.0
        done_sums: dict[str, float] = {}
        for info, valid in rollout["rollout_info_items"]:
            valid_f = valid.float()
            weight = float(valid_f.sum().item())
            if weight <= 0.0:
                continue
            weight_sum += weight
            for key, value in info["done_terms"].items():
                done_sums[key] = done_sums.get(key, 0.0) + float((value.float() * valid_f).sum().item())
            for key, value in info["reward_terms"].items():
                reward_sums[key] = reward_sums.get(key, 0.0) + float((value * valid_f).sum().item())
        if weight_sum > 0.0:
            for key, value_sum in reward_sums.items():
                metrics[f"reward_rollout/{key}_mean"] = value_sum / weight_sum
            for key, value_sum in done_sums.items():
                metrics[f"done_rollout/{key}_frac"] = value_sum / weight_sum

    def _add_action_group_metrics(self, metrics: dict, actions: torch.Tensor, alive: torch.Tensor) -> None:
        denom = alive.sum().clamp(min=1.0)
        act_abs = (actions.abs() * alive.unsqueeze(-1)).sum(dim=(0, 1, 2)) / denom
        if act_abs.numel() < 29:
            return
        legs_idx = list(range(0, 12))
        waist_idx = [12, 13, 14]
        arms_idx = list(range(15, 29))
        metrics["act/legs_abs"] = float(act_abs[legs_idx].mean().item())
        metrics["act/waist_abs"] = float(act_abs[waist_idx].mean().item())
        metrics["act/arms_abs"] = float(act_abs[arms_idx].mean().item())
        metrics["act/l_shoulder_pitch"] = float(act_abs[15].item())
        metrics["act/r_shoulder_pitch"] = float(act_abs[22].item())
        metrics["act/l_shoulder_roll"] = float(act_abs[16].item())
        metrics["act/r_shoulder_roll"] = float(act_abs[23].item())
        metrics["act/l_shoulder_yaw"] = float(act_abs[17].item())
        metrics["act/r_shoulder_yaw"] = float(act_abs[24].item())
        metrics["act/l_elbow"] = float(act_abs[18].item())
        metrics["act/r_elbow"] = float(act_abs[25].item())
        metrics["act/l_wrist_roll"] = float(act_abs[19].item())
        metrics["act/r_wrist_roll"] = float(act_abs[26].item())
        metrics["act/l_wrist_pitch"] = float(act_abs[20].item())
        metrics["act/r_wrist_pitch"] = float(act_abs[27].item())
        metrics["act/l_wrist_yaw"] = float(act_abs[21].item())
        metrics["act/r_wrist_yaw"] = float(act_abs[28].item())

    def _add_sampler_metrics(self, metrics: dict) -> None:
        stats = self.env.adaptive_sampling_stats()
        for key, value in stats.items():
            value = float(value)
            if math.isfinite(value):
                metrics[f"sampler/{key}"] = value

    def _add_reward_weighted_metrics(self, metrics: dict) -> None:
        reward_weights = {
            "action_rate": -self.env.config.action_rate_weight,
            "joint_limit": -10.0,
            "anchor_pos_reward": 0.5,
            "anchor_ori_reward": 0.5,
            "body_pos_reward": 1.0,
            "body_ori_reward": 1.0,
            "body_lin_vel_reward": 1.0,
            "body_ang_vel_reward": 1.0,
            "undesired_contacts": -0.1,
        }
        weighted_positive = 0.0
        weighted_penalty = 0.0
        for name, weight in reward_weights.items():
            rollout_key = f"reward_rollout/{name}_mean"
            chunk_key = f"reward/{name}_mean"
            if rollout_key in metrics:
                raw_value = metrics[rollout_key]
            elif chunk_key in metrics:
                raw_value = metrics[chunk_key]
            else:
                continue
            contribution = weight * raw_value * self.env.dt
            metrics[f"reward_weighted/{name}"] = contribution
            if contribution >= 0.0:
                weighted_positive += contribution
            else:
                weighted_penalty += contribution
        metrics["reward_weighted/positive"] = weighted_positive
        metrics["reward_weighted/penalty"] = weighted_penalty
        metrics["reward_weighted/total"] = weighted_positive + weighted_penalty

    def log(self, update_idx: int, max_updates: int, metrics: dict) -> None:
        h = self.horizon_h
        frame_v_means = " ".join(
            f"V{k+1}={metrics.get(f'critic/frame_v_{k+1}_mean', float('nan')):.4f}" for k in range(h)
        )
        tgt_means = " ".join(
            f"T{k+1}={metrics.get(f'critic/prefix_target_{k+1}_mean', float('nan')):.4f}" for k in range(h)
        )
        kl_frames = " ".join(
            f"f{k}={metrics.get(f'flow_cps/kl_frame_{k}', float('nan')):.6f}" for k in range(h)
        )
        print(
            f"[UPDATE] {update_idx}/{max_updates} "
            f"reward_step={metrics['rollout/reward_step_mean']:.5f} "
            f"env_raw={metrics.get('rollout/chunk_env_raw_return_mean', float('nan')):.5f} "
            f"target_raw={metrics['rollout/chunk_raw_return_mean']:.5f} "
            f"done_frac={metrics['rollout/done_frac']:.5f} "
            f"mean_reward={metrics.get('train/mean_reward', float('nan')):.5f} "
            f"mean_len={metrics.get('train/mean_episode_length', float('nan')):.2f}",
            flush=True,
        )
        print(
            f"[Flow-CPS] loss={metrics['flow_cps/loss']:.5f} "
            f"policy={metrics['flow_cps/policy_loss']:.5f} "
            f"value={metrics['flow_cps/value_loss']:.5f} "
            f"ratio={metrics['flow_cps/ratio']:.4f} "
            f"[{metrics['flow_cps/ratio_min']:.3f},{metrics['flow_cps/ratio_max']:.3f}] "
            f"clip={metrics['flow_cps/clip_frac']:.4f} "
            f"kl_raw={metrics['flow_cps/kl_raw']:.6f} "
            f"kl/step={metrics['flow_cps/kl_per_step']:.6f} "
            f"target/step={metrics['flow_cps/kl_target_per_step']:.4f} "
            f"kl_units={metrics['flow_cps/kl_units']:.0f} "
            f"grad={metrics['flow_cps/grad_norm']:.4f} "
            f"grad_c={metrics['flow_cps/grad_norm_critic']:.4f} "
            f"lr={metrics['flow_cps/lr']:.6f} critic_lr={metrics['flow_cps/critic_lr']:.6f} "
            f"early_stop@{metrics['flow_cps/early_stop_epoch']:.0f}",
            flush=True,
        )
        print(
            f"[CRITIC] unit=flow_chunk_gae V={metrics['critic/chunk_v_mean']:.4f} "
            f"V_tgt={metrics['critic/chunk_v_target_mean']:.4f} "
            f"one_step={metrics['critic/chunk_one_step_target_mean']:.4f} "
            f"boot={metrics['critic/chunk_bootstrap_mean']:.4f} "
            f"adv={metrics['critic/chunk_adv_mean']:.4f}/{metrics['critic/chunk_adv_std']:.4f} "
            f"cont={metrics['critic/chunk_cont_frac']:.4f} "
            f"valid_pfx={metrics['flow_cps/valid_prefix_frac']:.4f} "
            f"death_frame={metrics['rollout/death_frame_mean']:.3f} "
            f"gae_lambda={metrics['flow_cps/gae_lambda']:.3f} "
            "max_delta=nan "
            f"raw_adv_mean={metrics['policy/raw_adv_mean']:.4f} "
            f"raw_adv_std={metrics['policy/raw_adv_std']:.4f} "
            f"| {frame_v_means} | {tgt_means}",
            flush=True,
        )
        print(
            f"[KL_FRAME] {kl_frames} "
            f"clip0={metrics.get('flow_cps/clip_frame_0', float('nan')):.4f} "
            f"adv0={metrics.get('flow_cps/adv_abs_frame_0', float('nan')):.4f}",
            flush=True,
        )
        print(
            f"[ROLLOUT] live={metrics.get('rollout/live_steps_mean', float('nan')):.2f} "
            f"survive={metrics.get('rollout/success_frac', float('nan')):.5f} "
            f"per_step={metrics.get('rollout/reward_per_live_step', float('nan')):.5f} "
            f"episode_projection={metrics.get('rollout/max_episode_return_projection', float('nan')):.2f}",
            flush=True,
        )
        print(
            f"[POLICY_DETAIL] samples={metrics.get('flow_cps/sample_count', float('nan')):.0f} "
            f"mb={metrics.get('flow_cps/effective_mini_batch_size', float('nan')):.0f} "
            f"micro_mb={metrics.get('flow_cps/micro_batch_size', float('nan')):.0f} "
            f"logp_delta_abs={metrics.get('flow_cps/logprob_delta_abs', float('nan')):.5f} "
            f"old_logp={metrics.get('flow_cps/old_log_prob', float('nan')):.5f} "
            f"new_logp={metrics.get('flow_cps/new_log_prob', float('nan')):.5f} "
            f"physical_rms={metrics.get('policy/cps_physical_rms_achieved', float('nan')):.5f} "
            f"target={metrics.get('policy/cps_physical_rms_target', float('nan')):.5f} "
            f"scale={metrics.get('policy/cps_physical_scale', float('nan')):.5f} "
            f"cov_var={metrics.get('policy/cps_cov_trace_mean', float('nan')):.4f} "
            f"[{metrics.get('policy/cps_cov_trace_min', float('nan')):.4f},{metrics.get('policy/cps_cov_trace_max', float('nan')):.4f}] "
            f"cov_logdet={metrics.get('policy/cps_cov_logdet', float('nan')):.4f} "
            f"cov_offdiag={metrics.get('policy/cps_cov_offdiag_abs', float('nan')):.4f} "
            f"cps_params={metrics.get('policy/cps_params', float('nan')):.0f}",
            flush=True,
        )
        print(
            f"[TRAIN] mean_reward={metrics.get('train/mean_reward', float('nan')):.5f} "
            f"mean_len={metrics.get('train/mean_episode_length', float('nan')):.2f} "
            f"recent_eps={metrics.get('train/recent_episode_count', 0.0):.0f} "
            f"completed_eps={metrics.get('train/completed_episodes', 0.0):.0f}",
            flush=True,
        )
        print(
            f"[UPDATE_EFFECT] raw_z_mean_delta={metrics.get('policy/raw_z_mean_delta', float('nan')):.8f} "
            f"param_rms_delta={metrics.get('policy/param_rms_delta', float('nan')):.8f}",
            flush=True,
        )
        print(f"[TIME] collect={metrics['timing/collect_s']:.3f}s update={metrics['timing/update_s']:.3f}s", flush=True)
        print(
            f"[DONE] timeout={metrics.get('done/time_out_frac', 0.0):.5f} "
            f"anchor_pos={metrics.get('done/anchor_pos_bad_frac', 0.0):.5f} "
            f"anchor_ori={metrics.get('done/anchor_ori_bad_frac', 0.0):.5f} "
            f"ee_body={metrics.get('done/ee_body_bad_frac', 0.0):.5f}",
            flush=True,
        )
        print(
            f"[TRACK_ROLLOUT] "
            f"anchor_pos={metrics.get('reward_rollout/anchor_pos_reward_mean', float('nan')):.5f} "
            f"anchor_ori={metrics.get('reward_rollout/anchor_ori_reward_mean', float('nan')):.5f} "
            f"body_pos={metrics.get('reward_rollout/body_pos_reward_mean', float('nan')):.5f} "
            f"body_ori={metrics.get('reward_rollout/body_ori_reward_mean', float('nan')):.5f} "
            f"body_lin={metrics.get('reward_rollout/body_lin_vel_reward_mean', float('nan')):.5f} "
            f"body_ang={metrics.get('reward_rollout/body_ang_vel_reward_mean', float('nan')):.5f}",
            flush=True,
        )
        print(
            f"[ACT_SUMMARY] abs_mean={metrics.get('act/abs_mean', float('nan')):.4f} "
            f"abs_p95={metrics.get('act/abs_p95', float('nan')):.4f} "
            f"abs_p99={metrics.get('act/abs_p99', float('nan')):.4f} "
            f"abs_max={metrics.get('act/abs_max', float('nan')):.4f} "
            f"legs={metrics.get('act/legs_abs', float('nan')):.4f} "
            f"waist={metrics.get('act/waist_abs', float('nan')):.4f} "
            f"arms={metrics.get('act/arms_abs', float('nan')):.4f}",
            flush=True,
        )
        print(
            f"[FRAME_DIAG] frame0_abs={metrics.get('act/frame0_abs_mean', float('nan')):.4f} "
            f"frame1_abs={metrics.get('act/frame1_abs_mean', float('nan')):.4f} "
            f"in_chunk_delta={metrics.get('act/in_chunk_delta_abs', float('nan')):.4f} "
            f"cross_chunk_delta={metrics.get('act/cross_chunk_delta_abs', float('nan')):.4f}",
            flush=True,
        )
        print(
            f"[TRAIN_COST] action_rate={metrics.get('reward_rollout/action_rate_mean', float('nan')):.5f} "
            f"joint_limit={metrics.get('reward_rollout/joint_limit_mean', float('nan')):.5f} "
            f"contacts={metrics.get('reward_rollout/undesired_contacts_mean', float('nan')):.5f} "
            f"| weighted act_rate={metrics.get('reward_weighted/action_rate', float('nan')):.5f} "
            f"contacts={metrics.get('reward_weighted/undesired_contacts', float('nan')):.5f} "
            f"pos={metrics.get('reward_weighted/positive', float('nan')):.5f} "
            f"penalty={metrics.get('reward_weighted/penalty', float('nan')):.5f} "
            f"total={metrics.get('reward_weighted/total', float('nan')):.5f}",
            flush=True,
        )
        print(
            f"[RETURN_SPLIT] env_raw={metrics.get('rollout/chunk_env_raw_return_mean', float('nan')):.5f} "
            f"target_raw={metrics.get('rollout/chunk_raw_return_mean', float('nan')):.5f} "
            f"failure_cost={metrics.get('rollout/failure_cost_return_mean', float('nan')):.5f} "
            f"realized={metrics.get('rollout/chunk_return_mean', float('nan')):.5f}",
            flush=True,
        )

    def log_banner(self) -> None:
        cfg = self.cfg
        env = self.env
        print("[INFO] Starting Flow-CPS training (smooth action chunk + chunk GAE + flow chunk-start V)", flush=True)
        print(
            f"[INFO] task={env.task.name} terrain={env.task.terrain} motion_file={env.task.motion_file}",
            flush=True,
        )
        print(
            f"[INFO] component=flow_cps_base actor_obs_dim={self.actor_obs_dim} critic_obs_dim={self.critic_obs_dim} "
            f"action_dim={self.num_act} horizon={cfg.horizon} rollout_chunks={self._chunks_per_update()} "
            f"rollout_env_steps={cfg.rollout_env_steps} flow_steps={cfg.flow_steps} "
            f"cps_physical_rms={cfg.cps_physical_rms}",
            flush=True,
        )
        print(
            f"[INFO] critic_unit=flow_chunk_start state_only_critic=True prefix_q=False "
            f"causal_velocity={self._policy.causal_velocity} causal_arch={self._policy.causal_arch} "
            f"actor_variable=raw_target_rate "
            f"exploration=single_final_dense_cholesky_cps cps_trainable={self.cps_trainable} "
            f"cps_params={self._policy.cps_cholesky_raw.numel()} "
            f"terminal_failure_cost=False "
            f"flow_ode_deterministic=True physical_metric_scaled=True "
            f"per_frame_ratio_ppo=True flat_clip=True "
            f"actor_advantage=gae_chunk "
            f"per_frame_kl_adaptive_lr=True kl_units=1 kl_early_stop_factor={self.kl_early_stop_factor} "
            f"advantage_norm={cfg.advantage_normalization} "
            f"gamma={cfg.discount_gamma} gae_lambda={float(getattr(cfg, 'gae_lambda', 0.95)):.3f} "
            f"cross_chunk_gae=True td_bootstrap_unit=chunk "
            f"flow_value_loss=True flow_critic_steps={self.flow_critic_steps} "
            f"flow_critic_samples={self.flow_critic_samples} "
            f"flow_critic_fm_samples={self.flow_critic_fm_samples} "
            f"chunk_gamma={float(cfg.discount_gamma) ** int(cfg.horizon):.6f}",
            flush=True,
        )
        print(
            f"[INFO] actor_hidden_dims={list(cfg.actor_hidden_dims)} critic_hidden_dims={list(cfg.critic_hidden_dims)} "
            f"activation={cfg.activation} "
            f"empirical_normalization={cfg.empirical_normalization}",
            flush=True,
        )
        print(
            f"[INFO] policy_epochs={cfg.policy_epochs} clip_range={cfg.clip_range} "
            f"desired_kl={cfg.desired_kl} value_loss_coef={cfg.value_loss_coef} "
            f"value_loss=flow_matching "
            f"num_mini_batches={cfg.num_mini_batches} micro_batch={cfg.micro_batch_size} "
            f"policy_lr={cfg.policy_lr} critic_lr={cfg.value_lr}",
            flush=True,
        )
