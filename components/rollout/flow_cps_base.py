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
from collections import deque

import torch
from torch import nn

from method.base import Algorithm
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

