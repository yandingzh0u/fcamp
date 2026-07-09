

"""SFPO: causal flow policy with trace-normalized low-rank CPS exploration.

The actor keeps one stochastic path: a deterministic residual flow backbone plus
in-flow CPS exploration. Fresh noise is sampled in the 4xaction_dim trajectory
space through a learned diagonal-plus-low-rank covariance whose trace is
normalized to preserve the scalar CPS noise budget. The exact covariance
transition density is used for the PPO ratio/KL; there is no action-Gaussian
exploration branch or hand-written failure penalty.
"""

from __future__ import annotations

import math
from collections import deque

import torch
from torch import nn
from torch.nn import functional as F

from algorithms.base import Algorithm
from algorithms.kl_scheduler import adaptive_lr_from_kl
from networks.flow_critic import FlowChunkValueCritic
from networks.flow_policy import FlowMatchingPolicy
from networks.flow_sampling import flow_ode_mean
from networks.mlp_actor_critic import EmpiricalNormalization


class SFPO(Algorithm):
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
        self.actor_obs_dim = env.observation_dim
        self.critic_obs_dim = env.critic_observation_dim
        self.horizon_h = int(cfg.horizon)
        self.action_chunk_dim = self.horizon_h * self.num_act

        self._policy = FlowMatchingPolicy(
            obs_dim=self.actor_obs_dim,
            action_dim=self.num_act,
            horizon=self.horizon_h,
            hidden_dims=tuple(cfg.actor_hidden_dims),
            activation=cfg.activation,
            action_squash_scale=float(cfg.action_squash_scale),
            causal_velocity=True,
            causal_arch="prefix_cumsum",
        ).to(env.device)
        self._policy.set_action_max_delta(None)
        self.action_transform = "residual_absolute"
        self._policy.action_transform = self.action_transform
        self.cps_noise_level = float(cfg.cps_noise_level)
        self.cps_trainable = bool(getattr(cfg, "cps_trainable", True))
        if not (0.0 < self.cps_noise_level < 1.0):
            raise ValueError(f"cps_noise_level must be in (0, 1), got {self.cps_noise_level}")
        steps = int(cfg.flow_steps)
        self._cps_flat_dim = self.horizon_h * self.num_act
        self.cps_cov_rank = int(getattr(cfg, "cps_cov_rank", 8))
        if self.cps_cov_rank < 0:
            raise ValueError(f"cps_cov_rank must be >= 0, got {self.cps_cov_rank}")
        init_diag = math.log(math.exp(1.0) - 1.0)
        self._policy.cps_diag_raw = nn.Parameter(
            torch.full((steps, self._cps_flat_dim), init_diag, device=env.device)
        )
        self._policy.cps_lowrank_raw = nn.Parameter(
            1.0e-3 * torch.randn(steps, self._cps_flat_dim, self.cps_cov_rank, device=env.device)
        )
        self._policy.cps_diag_raw.requires_grad_(self.cps_trainable)
        self._policy.cps_lowrank_raw.requires_grad_(self.cps_trainable)
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
        # updated before each minibatch optimizer step (PPO-aligned), with an
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

    def _train_step_indices(self, device) -> torch.Tensor:
        return torch.arange(int(self.cfg.flow_steps), device=device, dtype=torch.long)

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

    def _deterministic_actor_actions(self, actor_obs: torch.Tensor, prev_action: torch.Tensor | None = None) -> torch.Tensor:
        return self._flow_mean_actions(actor_obs, prev_action=prev_action)

    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        actor_obs = self._norm_actor(obs, update=False)
        # prev_action is the raw (un-normalized) last action, which lives in the
        # last action_dim columns of the raw actor observation. Extract it from
        # the raw obs BEFORE normalization so the smooth transform anchors on the
        # true last executed action, not a normalized surrogate.
        prev_action = obs[..., -self.num_act:].detach()
        return self._flow_mean_actions(actor_obs, prev_action=prev_action)

    def _flow_mean_latent(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Differentiable zero-noise CPS final latent used for eval/probing."""
        batch = actor_obs.shape[0]
        latent = torch.zeros(batch, self.chunk_dim, device=actor_obs.device, dtype=actor_obs.dtype)
        self._policy._validate_inputs(actor_obs, latent, int(self.cfg.flow_steps))
        obs_prep = self._policy._prepare_observation(actor_obs)
        steps = int(self.cfg.flow_steps)
        sigma_schedule = torch.linspace(1.0, 0.0, steps + 1, device=actor_obs.device, dtype=actor_obs.dtype)
        for step_index in range(steps):
            next_path, _, _ = self._cps_backbone_step(obs_prep, latent, sigma_schedule, step_index)
            latent = self._residual_from_path(next_path).reshape(batch, self.chunk_dim)
        return latent

    def _flow_mean_actions(self, actor_obs: torch.Tensor, prev_action: torch.Tensor | None = None) -> torch.Tensor:
        mean_latent = self._flow_mean_latent(actor_obs)
        return self._policy._action_transform(mean_latent, prev_action=prev_action).view(
            actor_obs.shape[0], self.horizon_h, self.num_act
        )

    def _cps_step_coeffs(
        self,
        step_index: int,
        sigma_schedule: torch.Tensor,
        reference: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma = sigma_schedule[step_index].to(device=reference.device, dtype=reference.dtype)
        sigma_next = sigma_schedule[step_index + 1].to(device=reference.device, dtype=reference.dtype)
        delta_sigma = torch.clamp(sigma - sigma_next, min=0.0)
        sqrt_delta = torch.sqrt(delta_sigma)
        eta = torch.as_tensor(self.cps_noise_level, device=reference.device, dtype=reference.dtype)
        beta = 0.5 * math.pi * eta
        del sigma_next
        # Mean-preserving Action-CPS applies coefficient preservation to the
        # exploration offset process, not to the deterministic flow backbone.
        # The fresh structured noise is trace-normalized, so the average
        # Brownian energy is still exactly the scalar CPS budget below.
        noise_coeff = torch.sin(beta) * sqrt_delta
        predicted_sq = torch.clamp(1.0 - noise_coeff.square(), min=1.0e-12)
        predicted_coeff = torch.sqrt(predicted_sq)
        return predicted_coeff, noise_coeff, eta

    def _cps_covariance_factors(
        self,
        step_index: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        diag = F.softplus(self._policy.cps_diag_raw[step_index].to(device=device, dtype=dtype)) + 1.0e-4
        lowrank = self._policy.cps_lowrank_raw[step_index].to(device=device, dtype=dtype)
        trace = (diag.square().sum() + lowrank.square().sum()).clamp(min=1.0e-12)
        scale = torch.sqrt(trace / float(self._cps_flat_dim))
        diag = diag / scale
        lowrank = lowrank / scale
        cov = torch.diag_embed(diag.square()) + lowrank @ lowrank.transpose(0, 1)
        chol = torch.linalg.cholesky(cov)
        log_diag_chol = torch.log(torch.diagonal(chol).clamp(min=1.0e-8))
        trace_normalized = (diag.square().sum() + lowrank.square().sum()) / float(self._cps_flat_dim)
        return diag, lowrank, cov, chol, log_diag_chol, trace_normalized

    def _cps_backbone_step(
        self,
        obs_prep: torch.Tensor,
        mean_latent: torch.Tensor,
        sigma_schedule: torch.Tensor,
        step_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = mean_latent.shape[0]
        sigma = sigma_schedule[step_index]
        timestep_batch = torch.full((batch,), float(sigma.item()), device=mean_latent.device, dtype=mean_latent.dtype)
        model_output = self._policy.velocity_field(obs_prep, mean_latent, timestep_batch)
        latent = mean_latent.view(batch, self.horizon_h, self.num_act)
        mean_next = flow_ode_mean(model_output, mean_latent, sigma_schedule, step_index)
        predicted_coeff, noise_coeff, _ = self._cps_step_coeffs(step_index, sigma_schedule, latent)
        next_mean_path = self._path_from_residual(mean_next.view(batch, self.horizon_h, self.num_act))
        return next_mean_path, predicted_coeff, noise_coeff

    @staticmethod
    def _path_from_residual(residual: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(residual, dim=-2)

    @staticmethod
    def _residual_from_path(path: torch.Tensor) -> torch.Tensor:
        prev = torch.cat([torch.zeros_like(path[..., :1, :]), path[..., :-1, :]], dim=-2)
        return path - prev

    @staticmethod
    def _chunk_path_from_innovations(innovations: torch.Tensor) -> torch.Tensor:
        """Causal, invertible frame smoother for action-path exploration noise.

        Innovations are independent in density space. Their normalized
        cumulative path is an action-level trajectory perturbation, not a
        residual-latent perturbation.
        """
        horizon = innovations.shape[-2]
        norm = torch.sqrt(
            torch.arange(1, horizon + 1, device=innovations.device, dtype=innovations.dtype)
        ).view(*((1,) * (innovations.ndim - 2)), horizon, 1)
        return torch.cumsum(innovations, dim=-2) / norm

    @staticmethod
    def _innovations_from_chunk_path(path_noise: torch.Tensor) -> torch.Tensor:
        horizon = path_noise.shape[-2]
        norm = torch.sqrt(
            torch.arange(1, horizon + 1, device=path_noise.device, dtype=path_noise.dtype)
        ).view(*((1,) * (path_noise.ndim - 2)), horizon, 1)
        cumulative = path_noise * norm
        first = cumulative[..., :1, :]
        rest = cumulative[..., 1:, :] - cumulative[..., :-1, :]
        return torch.cat([first, rest], dim=-2)

    def _sample_cps_innovation(
        self,
        *,
        batch: int,
        step_index: int,
        noise_coeff: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        diag, lowrank, _, _, _, _ = self._cps_covariance_factors(step_index, device=device, dtype=dtype)
        eps_diag = torch.randn(batch, self._cps_flat_dim, device=device, dtype=dtype)
        eps_rank = torch.randn(batch, self.cps_cov_rank, device=device, dtype=dtype)
        innovation = eps_diag * diag.view(1, -1) + eps_rank @ lowrank.transpose(0, 1)
        return (innovation * noise_coeff).view(batch, self.horizon_h, self.num_act)

    def _cps_innovation_log_prob(
        self,
        innovation: torch.Tensor,
        noise_coeff: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        # Exact low-rank-plus-diagonal CPS density. Cholesky orders dimensions
        # by frame, so grouping component log-probs back into frames gives a
        # causal conditional decomposition of the joint chunk density.
        batch = innovation.shape[0]
        safe_std = torch.clamp(noise_coeff, min=1.0e-6)
        _, _, _, chol, log_diag_chol, _ = self._cps_covariance_factors(
            step_index,
            device=innovation.device,
            dtype=innovation.dtype,
        )
        flat = innovation.reshape(batch, self._cps_flat_dim) / safe_std
        whitened = torch.linalg.solve_triangular(
            chol,
            flat.transpose(0, 1),
            upper=False,
        ).transpose(0, 1)
        component_log_prob = (
            -0.5 * (whitened.square() + math.log(2.0 * math.pi))
            - log_diag_chol.view(1, self._cps_flat_dim)
            - torch.log(safe_std)
        )
        return component_log_prob.view(batch, self.horizon_h, self.num_act).sum(dim=-1)

    def _sample_cps_path(
        self,
        actor_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = actor_obs.shape[0]
        steps = int(self.cfg.flow_steps)
        mean_latent = torch.zeros(batch, self.chunk_dim, device=actor_obs.device, dtype=actor_obs.dtype)
        sampled_latent = torch.zeros(batch, self.chunk_dim, device=actor_obs.device, dtype=actor_obs.dtype)
        self._policy._validate_inputs(actor_obs, sampled_latent, steps)
        obs_prep = self._policy._prepare_observation(actor_obs)
        sigma_schedule = torch.linspace(1.0, 0.0, steps + 1, device=actor_obs.device, dtype=actor_obs.dtype)

        latent_path = torch.zeros(batch, steps + 1, self.chunk_dim, device=actor_obs.device, dtype=actor_obs.dtype)
        step_log_probs = torch.zeros(batch, steps, self.horizon_h, device=actor_obs.device, dtype=actor_obs.dtype)
        step_noise_coeffs = torch.zeros(
            batch, steps, self.horizon_h, self.num_act, device=actor_obs.device, dtype=actor_obs.dtype
        )
        latent_path[:, 0, :] = sampled_latent

        for step_index in range(steps):
            next_mean_path, predicted_coeff, noise_coeff = self._cps_backbone_step(
                obs_prep,
                mean_latent,
                sigma_schedule,
                step_index,
            )
            offset = sampled_latent.view(batch, self.horizon_h, self.num_act) - mean_latent.view(
                batch, self.horizon_h, self.num_act
            )
            offset_path = self._path_from_residual(offset)
            mean_path = next_mean_path + predicted_coeff * offset_path
            innovation = self._sample_cps_innovation(
                batch=batch,
                step_index=step_index,
                noise_coeff=noise_coeff,
                device=actor_obs.device,
                dtype=actor_obs.dtype,
            )
            random_path = self._chunk_path_from_innovations(innovation)
            sample_path = mean_path + random_path
            sample = self._residual_from_path(sample_path)
            step_log_probs[:, step_index, :] = self._cps_innovation_log_prob(innovation, noise_coeff, step_index)
            step_noise_coeffs[:, step_index] = noise_coeff
            mean_latent = self._residual_from_path(next_mean_path).reshape(batch, self.chunk_dim)
            sampled_latent = sample.reshape(batch, self.chunk_dim)
            latent_path[:, step_index + 1, :] = sampled_latent

        return sampled_latent, latent_path, step_log_probs, step_noise_coeffs

    def _recompute_cps_path_stats(
        self,
        actor_obs: torch.Tensor,
        latent_path: torch.Tensor,
    ) -> torch.Tensor:
        batch = actor_obs.shape[0]
        steps = int(self.cfg.flow_steps)
        mean_latent = torch.zeros(batch, self.chunk_dim, device=actor_obs.device, dtype=actor_obs.dtype)
        self._policy._validate_inputs(actor_obs, mean_latent, steps)
        obs_prep = self._policy._prepare_observation(actor_obs)
        sigma_schedule = torch.linspace(1.0, 0.0, steps + 1, device=actor_obs.device, dtype=actor_obs.dtype)

        step_log_probs = torch.zeros(batch, steps, self.horizon_h, device=actor_obs.device, dtype=actor_obs.dtype)

        for step_index in range(steps):
            sampled_latent = latent_path[:, step_index, :]
            next_latent = latent_path[:, step_index + 1, :].view(batch, self.horizon_h, self.num_act)
            next_mean_path, predicted_coeff, noise_coeff = self._cps_backbone_step(
                obs_prep,
                mean_latent,
                sigma_schedule,
                step_index,
            )
            offset = sampled_latent.view(batch, self.horizon_h, self.num_act) - mean_latent.view(
                batch, self.horizon_h, self.num_act
            )
            offset_path = self._path_from_residual(offset)
            mean_path = next_mean_path + predicted_coeff * offset_path
            next_path = self._path_from_residual(next_latent)
            path_noise = next_path - mean_path
            innovation = self._innovations_from_chunk_path(path_noise)
            step_log_probs[:, step_index, :] = self._cps_innovation_log_prob(innovation, noise_coeff, step_index)
            mean_latent = self._residual_from_path(next_mean_path).reshape(batch, self.chunk_dim)

        return step_log_probs

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
        # PPO-aligned continuous stream across updates (same as SFPO).
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
        flow_steps = int(self.cfg.flow_steps)

        actor_obs_buf = torch.zeros(chunks, n_envs, self.actor_obs_dim, device=device)
        critic_obs_buf = torch.zeros(chunks, n_envs, self.critic_obs_dim, device=device)
        actions_buf = torch.zeros(chunks, n_envs, h, self.num_act, device=device)
        latents_buf = torch.zeros(chunks, n_envs, flow_steps + 1, self.chunk_dim, device=device)
        old_log_probs_buf = torch.zeros(chunks, n_envs, flow_steps, h, device=device)
        old_cps_noise_coeff_buf = torch.zeros(chunks, n_envs, flow_steps, h, self.num_act, device=device)
        # prev_action for the smooth transform: the raw last executed action at
        # chunk start. Stored so update can recompute the same action transform.
        prev_action_buf = torch.zeros(chunks, n_envs, self.num_act, device=device)
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
        train_step_indices = None
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
                # prev_action = raw last executed action (last num_act cols of
                # raw actor obs). The smooth transform anchors the chunk on this.
                prev_action = obs[..., -self.num_act:].detach()
                prev_action_buf[chunk_idx] = prev_action

                (
                    final_latent,
                    latent_path,
                    step_log_probs,
                    step_noise_coeffs,
                ) = self._sample_cps_path(actor_obs_n)
                action_chunk = self._policy._action_transform(
                    final_latent,
                    prev_action=prev_action,
                ).view(n_envs, h, self.num_act)
                train_step_indices = self._train_step_indices(device)
                action_abs_max = max(action_abs_max, float(action_chunk.abs().max().item()))

                alive_in_chunk = torch.ones(n_envs, dtype=torch.bool, device=device)

                for frame_idx in range(h):
                    alive_before_frame = alive_in_chunk.clone()
                    action_t = action_chunk[:, frame_idx, :]
                    if bool((~alive_before_frame).any()):
                        action_t = torch.where(
                            alive_before_frame.unsqueeze(-1), action_t, torch.zeros_like(action_t)
                        )
                    next_obs, reward, done, info = env.step(action_t, auto_reset=False)
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
                # fresh episode; alive envs keep running (continuous PPO stream).
                chunk_done = done_frame_buf[chunk_idx].any(dim=-1)
                if bool(chunk_done.any()):
                    reset_ids = chunk_done.nonzero(as_tuple=False).squeeze(-1)
                    reset_phases = self.env.sample_phase_indices(reset_ids.numel(), horizon=max(1, h))
                    reset_obs = self.env.reset_envs(reset_ids, phase_indices=reset_phases)
                    obs[reset_ids] = reset_obs
                    critic_obs = self.env.get_critic_observation()

                chunk_first_action = action_chunk[:, 0, :].detach()
                chunk_last_action = action_chunk[:, h - 1, :].detach()
                # first-action delta = |a_0 - prev_action|, where prev_action is
                # the raw env last action at chunk start (from prev_action_buf).
                # This is the true cross-chunk continuity metric and is NOT
                # polluted by chunk-end resets (prev_action_buf[chunk_idx] holds
                # the env's last_action right before this chunk, which is 0 for
                # freshly-reset envs and the previous chunk's last action for
                # alive envs).
                first_action_delta = (chunk_first_action - prev_action).abs().mean(dim=-1)
                metric_cross_chunk_delta_sum = metric_cross_chunk_delta_sum + first_action_delta.sum()
                metric_cross_chunk_delta_count = metric_cross_chunk_delta_count + torch.tensor(
                    float(n_envs), device=device, dtype=obs.dtype
                )

                actor_obs_buf[chunk_idx] = actor_obs_n
                critic_obs_buf[chunk_idx] = critic_obs_n
                actions_buf[chunk_idx] = action_chunk.detach()
                latents_buf[chunk_idx] = latent_path.detach()
                old_log_probs_buf[chunk_idx] = step_log_probs.detach()
                old_cps_noise_coeff_buf[chunk_idx] = step_noise_coeffs.detach()

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
        if train_step_indices is None:
            train_step_indices = self._train_step_indices(device)

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
            "latents": latents_buf,
            "old_log_probs": old_log_probs_buf,
            "old_cps_noise_coeff": old_cps_noise_coeff_buf,
            "prev_action": prev_action_buf,
            "train_step_indices": train_step_indices,
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

    def update(self, rollout: dict, collect_time: float) -> dict:
        import time as _time

        device = self.env.device
        chunks, n_envs = rollout["actions"].shape[:2]
        h = self.horizon_h
        flow_steps = int(self.cfg.flow_steps)
        raw_batch_size = chunks * n_envs

        actor_obs = rollout["actor_obs"].reshape(raw_batch_size, self.actor_obs_dim)
        critic_obs = rollout["critic_obs"].reshape(raw_batch_size, self.critic_obs_dim)
        latent_path = rollout["latents"].reshape(raw_batch_size, flow_steps + 1, self.chunk_dim)
        old_log_probs = rollout["old_log_probs"].reshape(raw_batch_size, flow_steps, h)
        prev_action = rollout["prev_action"].reshape(raw_batch_size, self.num_act)
        chunk_v_targets = rollout["chunk_v_targets"].reshape(raw_batch_size)
        advantages = rollout["advantages"].reshape(raw_batch_size, h)
        raw_gae_advantages = rollout["raw_gae_advantages"].reshape(raw_batch_size, h)
        valid_prefix_mask = rollout["valid_prefix_mask"].reshape(raw_batch_size, h)
        chunk_valid_mask = rollout["chunk_valid_mask"].reshape(raw_batch_size)
        batch_size = actor_obs.shape[0]

        mini_batch_size = self._policy_mini_batch_size(batch_size)
        clip_range = float(self.cfg.clip_range)
        value_coef = float(self.cfg.value_loss_coef)
        train_step_indices = rollout["train_step_indices"]

        probe_count = min(128, batch_size)
        with torch.no_grad():
            probe_obs = actor_obs[:probe_count]
            probe_prev_action = prev_action[:probe_count]
            probe_action_before = self._deterministic_actor_actions(probe_obs, prev_action=probe_prev_action)
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
                # optimizer step -- PPO-aligned).
                mb_prefix_kl_accum = torch.zeros(h, device=device)
                mb_prefix_kl_count = torch.zeros(h, device=device)
                for micro_start in range(0, mb_size, micro_batch_size):
                    micro_end = min(micro_start + micro_batch_size, mb_size)
                    sub = idx[micro_start:micro_end]
                    weight = float(sub.numel()) / float(mb_size)

                    old_log_probs_sub = old_log_probs[sub]  # [micro, steps, h]
                    # Flow-CPS optimizes a timestep-averaged surrogate rather
                    # than exponentiating the joint product over all denoising
                    # steps. This keeps the PPO ratio in the same units as the
                    # official MixGRPO CPS objective.
                    old_per_frame_logp = old_log_probs_sub.mean(dim=1)  # [micro, h]
                    new_log_probs = self._recompute_cps_path_stats(
                        actor_obs[sub],
                        latent_path[sub],
                    )
                    new_per_frame_logp = new_log_probs.mean(dim=1)
                    per_frame_log_ratio = new_per_frame_logp - old_per_frame_logp  # [micro, h]
                    # Match the MixGRPO/Flow-CPS training object: the KL proxy
                    # is defined on the same log-prob surrogate used by the
                    # PPO ratio, not on a separate Gaussian transition KL.
                    kl_per_frame = 0.5 * per_frame_log_ratio.square()
                    # Per-frame PPO ratio (NOT a prefix/joint cumsum ratio).
                    # Each executed chunk component gets its own per-frame
                    # likelihood ratio, but all executed frames carry the same
                    # chunk GAE advantage.
                    ratio = torch.exp(per_frame_log_ratio)  # [micro, h]
                    adv = advantages[sub]  # [micro, h]
                    mask = valid_prefix_mask[sub].to(dtype=ratio.dtype)  # [micro, h]
                    mask_sum = mask.sum().clamp(min=1.0)

                    # Flat PPO clip (same [1-eps, 1+eps] for every frame).
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
                # (PPO-aligned, critic LR is decoupled and stays at value_lr).
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
            probe_action_after = self._deterministic_actor_actions(probe_obs, prev_action=probe_prev_action)
            action_delta = torch.mean(torch.abs(probe_action_after - probe_action_before))
            param_delta_sq = torch.zeros((), device=device)
            param_count = 0
            for param, before in zip(self._policy.parameters(), params_before, strict=True):
                delta = param.detach() - before
                param_delta_sq = param_delta_sq + torch.sum(delta * delta)
                param_count += delta.numel()
            param_rms_delta = torch.sqrt(param_delta_sq / max(param_count, 1))
            # Report the train-time CPS noise coefficient averaged across the
            # scheduler. It is step dependent because Action-CPS scales fresh
            # Brownian path noise by sqrt(delta_sigma).
            cps_eta = torch.full((flow_steps,), self.cps_noise_level, device=device, dtype=torch.float32)
            sigma_schedule = torch.linspace(1.0, 0.0, flow_steps + 1, device=device, dtype=torch.float32)
            sigma = sigma_schedule[:-1]
            sigma_next = sigma_schedule[1:]
            sqrt_delta = torch.sqrt(torch.clamp(sigma - sigma_next, min=0.0))
            cps_noise_coeff = torch.sin(0.5 * math.pi * cps_eta) * sqrt_delta
            cps_pred_coeff = torch.sqrt(torch.clamp(1.0 - cps_noise_coeff.square(), min=1.0e-12))
            cov_traces = []
            cov_logdets = []
            cov_diag = []
            cov_offdiag_abs = []
            cov_lowrank_energy = []
            for step_index in range(flow_steps):
                diag_factor, lowrank, cov, _, log_diag_chol, trace_normalized = self._cps_covariance_factors(
                    step_index,
                    device=device,
                    dtype=torch.float32,
                )
                diag = torch.diagonal(cov)
                offdiag = cov - torch.diag_embed(diag)
                cov_traces.append(trace_normalized)
                cov_logdets.append(2.0 * log_diag_chol.sum())
                cov_diag.append(diag)
                cov_offdiag_abs.append(offdiag.abs().mean())
                cov_lowrank_energy.append(lowrank.square().sum() / float(self._cps_flat_dim))
            cps_cov_trace = torch.stack(cov_traces)
            cps_cov_logdet = torch.stack(cov_logdets)
            cps_cov_diag = torch.stack(cov_diag)
            cps_cov_offdiag_abs = torch.stack(cov_offdiag_abs)
            cps_cov_lowrank_energy = torch.stack(cov_lowrank_energy)

        update_metrics = {
            "sfpo/loss": totals["loss"] / denom,
            "sfpo/policy_loss": totals["policy_loss"] / denom,
            "sfpo/value_loss": totals["value_loss"] / denom,
            "sfpo/ratio": totals["ratio"] / denom,
            "sfpo/ratio_min": totals["ratio_min"] if totals["ratio_min"] != float("inf") else 0.0,
            "sfpo/ratio_max": totals["ratio_max"],
            "sfpo/clip_frac": totals["clip_frac"] / denom,
            "sfpo/kl_loss": kl_raw,
            "sfpo/kl_raw": kl_raw,
            "sfpo/kl_per_step": kl_per_step,
            "sfpo/kl_target_per_step": desired_kl,
            "sfpo/kl_target_raw": desired_kl * kl_units,
            "sfpo/kl_units": float(kl_units),
            "sfpo/logprob_delta_abs": totals["logprob_delta_abs"] / denom,
            "sfpo/old_log_prob": totals["old_log_prob"] / denom,
            "sfpo/new_log_prob": totals["new_log_prob"] / denom,
            "sfpo/grad_norm": totals["grad_norm"] / denom,
            "sfpo/grad_norm_critic": totals["grad_norm_critic"] / denom,
            "sfpo/lr": self.learning_rate,
            "sfpo/critic_lr": self.critic_learning_rate,
            "sfpo/effective_mini_batch_size": float(mini_batch_size),
            "sfpo/sample_count": float(batch_size),
            "sfpo/raw_sample_count": float(raw_batch_size),
            "sfpo/micro_batch_size": float(self._policy_micro_batch_size(mini_batch_size)),
            "sfpo/micro_batches": float(micro_batch_count),
            "sfpo/optimizer_steps": float(update_count),
            "sfpo/cps_train_steps": float(train_step_indices.numel()),
            "sfpo/v_target_mean": totals["v_target_mean"] / denom,
            "sfpo/valid_prefix_frac": totals["valid_prefix_frac"] / denom,
            "sfpo/gae_lambda": float(getattr(self.cfg, "gae_lambda", 0.95)),
            "sfpo/critic_unit": 1.0,  # 1 = chunk
            "sfpo/actor_advantage_unit": 1.0,  # 1 = chunk GAE
            "sfpo/action_transform": 2.0,  # residual_absolute
            "sfpo/kl_early_stop_factor": float(self.kl_early_stop_factor),
            "sfpo/early_stop_epoch": float(early_stopped_epoch),
            "sfpo/advantage_normalization": float({"per_prefix": 0.0, "global": 1.0, "none": 2.0}[self.advantage_normalization]),
            "policy/action_delta": float(action_delta.item()),
            "policy/param_rms_delta": float(param_rms_delta.item()),
            "policy/raw_adv_mean": float(raw_gae_advantages.mean().item()),
            "policy/raw_adv_std": float(raw_gae_advantages.std(unbiased=False).item()),
            "policy/cps_base_eta": float(self.cps_noise_level),
            "policy/cps_eta_mean": float(cps_eta.mean().item()),
            "policy/cps_eta_min": float(cps_eta.min().item()),
            "policy/cps_eta_max": float(cps_eta.max().item()),
            "policy/cps_noise_coeff_mean": float(cps_noise_coeff.mean().item()),
            "policy/cps_noise_coeff_min": float(cps_noise_coeff.min().item()),
            "policy/cps_noise_coeff_max": float(cps_noise_coeff.max().item()),
            "policy/cps_pred_coeff_mean": float(cps_pred_coeff.mean().item()),
            "policy/cps_energy_error": float(
                torch.max(
                    torch.abs(cps_pred_coeff.square() + cps_noise_coeff.square() * cps_cov_trace - 1.0)
                ).item()
            ),
            "policy/cps_cov_trace_mean": float(cps_cov_trace.mean().item()),
            "policy/cps_cov_trace_min": float(cps_cov_trace.min().item()),
            "policy/cps_cov_trace_max": float(cps_cov_trace.max().item()),
            "policy/cps_cov_logdet_mean": float(cps_cov_logdet.mean().item()),
            "policy/cps_cov_diag_mean": float(cps_cov_diag.mean().item()),
            "policy/cps_cov_diag_min": float(cps_cov_diag.min().item()),
            "policy/cps_cov_diag_max": float(cps_cov_diag.max().item()),
            "policy/cps_cov_offdiag_abs": float(cps_cov_offdiag_abs.mean().item()),
            "policy/cps_cov_lowrank_energy": float(cps_cov_lowrank_energy.mean().item()),
            "policy/cps_cov_rank": float(self.cps_cov_rank),
            "policy/cps_params": float(self._policy.cps_diag_raw.numel() + self._policy.cps_lowrank_raw.numel()),
        }
        for k in range(h):
            update_metrics[f"sfpo/kl_frame_{k}"] = float(per_frame_kl_mean[k]) if k < len(per_frame_kl_mean) else float("nan")
            update_metrics[f"sfpo/ratio_frame_{k}"] = float(per_frame_ratio_mean[k]) if k < len(per_frame_ratio_mean) else float("nan")
            update_metrics[f"sfpo/clip_frame_{k}"] = float(per_frame_clip_mean[k]) if k < len(per_frame_clip_mean) else float("nan")
            update_metrics[f"sfpo/adv_abs_frame_{k}"] = float(per_frame_adv_abs[k]) if k < len(per_frame_adv_abs) else float("nan")
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
            "algo/name": "sfpo",
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
        for key in ("top_bin", "top_prob", "failed_sum", "entropy", "peak_bin"):
            metrics[f"sampler/{key}"] = float(stats.get(key, float("nan")))

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
            f"f{k}={metrics.get(f'sfpo/kl_frame_{k}', float('nan')):.6f}" for k in range(h)
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
            f"[SFPO] loss={metrics['sfpo/loss']:.5f} "
            f"policy={metrics['sfpo/policy_loss']:.5f} "
            f"value={metrics['sfpo/value_loss']:.5f} "
            f"ratio={metrics['sfpo/ratio']:.4f} "
            f"[{metrics['sfpo/ratio_min']:.3f},{metrics['sfpo/ratio_max']:.3f}] "
            f"clip={metrics['sfpo/clip_frac']:.4f} "
            f"kl_raw={metrics['sfpo/kl_raw']:.6f} "
            f"kl/step={metrics['sfpo/kl_per_step']:.6f} "
            f"target/step={metrics['sfpo/kl_target_per_step']:.4f} "
            f"kl_units={metrics['sfpo/kl_units']:.0f} "
            f"grad={metrics['sfpo/grad_norm']:.4f} "
            f"grad_c={metrics['sfpo/grad_norm_critic']:.4f} "
            f"lr={metrics['sfpo/lr']:.6f} critic_lr={metrics['sfpo/critic_lr']:.6f} "
            f"early_stop@{metrics['sfpo/early_stop_epoch']:.0f}",
            flush=True,
        )
        print(
            f"[CRITIC] unit=flow_chunk_gae V={metrics['critic/chunk_v_mean']:.4f} "
            f"V_tgt={metrics['critic/chunk_v_target_mean']:.4f} "
            f"one_step={metrics['critic/chunk_one_step_target_mean']:.4f} "
            f"boot={metrics['critic/chunk_bootstrap_mean']:.4f} "
            f"adv={metrics['critic/chunk_adv_mean']:.4f}/{metrics['critic/chunk_adv_std']:.4f} "
            f"cont={metrics['critic/chunk_cont_frac']:.4f} "
            f"valid_pfx={metrics['sfpo/valid_prefix_frac']:.4f} "
            f"death_frame={metrics['rollout/death_frame_mean']:.3f} "
            f"gae_lambda={metrics['sfpo/gae_lambda']:.3f} "
            "max_delta=nan "
            f"raw_adv_mean={metrics['policy/raw_adv_mean']:.4f} "
            f"raw_adv_std={metrics['policy/raw_adv_std']:.4f} "
            f"| {frame_v_means} | {tgt_means}",
            flush=True,
        )
        print(
            f"[KL_FRAME] {kl_frames} "
            f"clip0={metrics.get('sfpo/clip_frame_0', float('nan')):.4f} "
            f"adv0={metrics.get('sfpo/adv_abs_frame_0', float('nan')):.4f}",
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
            f"[POLICY_DETAIL] samples={metrics.get('sfpo/sample_count', float('nan')):.0f} "
            f"mb={metrics.get('sfpo/effective_mini_batch_size', float('nan')):.0f} "
            f"micro_mb={metrics.get('sfpo/micro_batch_size', float('nan')):.0f} "
            f"logp_delta_abs={metrics.get('sfpo/logprob_delta_abs', float('nan')):.5f} "
            f"old_logp={metrics.get('sfpo/old_log_prob', float('nan')):.5f} "
            f"new_logp={metrics.get('sfpo/new_log_prob', float('nan')):.5f} "
            f"cps_steps={metrics.get('sfpo/cps_train_steps', float('nan')):.0f} "
            f"base_eta={metrics.get('policy/cps_base_eta', float('nan')):.4f} "
            f"eta={metrics.get('policy/cps_eta_mean', float('nan')):.4f} "
            f"[{metrics.get('policy/cps_eta_min', float('nan')):.4f},{metrics.get('policy/cps_eta_max', float('nan')):.4f}] "
            f"cps_noise={metrics.get('policy/cps_noise_coeff_mean', float('nan')):.4f} "
            f"cps_noise_max={metrics.get('policy/cps_noise_coeff_max', float('nan')):.4f} "
            f"cps_energy_err={metrics.get('policy/cps_energy_error', float('nan')):.2e} "
            f"cov_trace={metrics.get('policy/cps_cov_trace_mean', float('nan')):.4f} "
            f"[{metrics.get('policy/cps_cov_trace_min', float('nan')):.4f},{metrics.get('policy/cps_cov_trace_max', float('nan')):.4f}] "
            f"cov_logdet={metrics.get('policy/cps_cov_logdet_mean', float('nan')):.4f} "
            f"cov_diag={metrics.get('policy/cps_cov_diag_mean', float('nan')):.4f} "
            f"[{metrics.get('policy/cps_cov_diag_min', float('nan')):.4f},{metrics.get('policy/cps_cov_diag_max', float('nan')):.4f}] "
            f"cov_offdiag={metrics.get('policy/cps_cov_offdiag_abs', float('nan')):.4f} "
            f"lowrank_energy={metrics.get('policy/cps_cov_lowrank_energy', float('nan')):.4f} "
            f"rank={metrics.get('policy/cps_cov_rank', float('nan')):.0f} "
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
            f"[UPDATE_EFFECT] action_delta={metrics.get('policy/action_delta', float('nan')):.8f} "
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
        print("[INFO] Starting SFPO training (smooth action chunk + chunk GAE + flow chunk-start V)", flush=True)
        print(
            f"[INFO] task={env.task.name} terrain={env.task.terrain} motion_file={env.task.motion_file}",
            flush=True,
        )
        print(
            f"[INFO] algo=sfpo actor_obs_dim={self.actor_obs_dim} critic_obs_dim={self.critic_obs_dim} "
            f"action_dim={self.num_act} horizon={cfg.horizon} rollout_chunks={self._chunks_per_update()} "
            f"rollout_env_steps={cfg.rollout_env_steps} flow_steps={cfg.flow_steps} "
            f"cps_noise_level={cfg.cps_noise_level}",
            flush=True,
        )
        print(
            f"[INFO] critic_unit=flow_chunk_start state_only_critic=True prefix_q=False "
            f"causal_velocity={self._policy.causal_velocity} causal_arch={self._policy.causal_arch} "
            f"action_transform={self.action_transform} action_max_delta=none "
            f"exploration=trace_normalized_lowrank_diag_cps_inside_flow cps_trainable={self.cps_trainable} "
            f"cps_cov_rank={self.cps_cov_rank} "
            f"cps_params={self._policy.cps_diag_raw.numel() + self._policy.cps_lowrank_raw.numel()} "
            f"terminal_failure_cost=False "
            f"cps_path_coordinates=True offset_energy_preserving=True lowrank_diag_transition_density=True "
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
            f"activation={cfg.activation} action_squash_scale={cfg.action_squash_scale} "
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
