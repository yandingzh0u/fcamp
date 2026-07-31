

"""Flow-CPS with direct residual exploration and an identifiable global scale.

The actor keeps the proven causal residual decoder and per-frame PPO contract.
At every flow step, CPS noise is injected directly into residual/action-increment
coordinates.  All H offsets draw independent innovations from the same learned
joint covariance shape; a separate learned scalar owns the total noise budget.
The exact per-offset transition density is used by clipped PPO.
"""

from __future__ import annotations

import math
from collections import deque

import torch
from torch import nn
from torch.nn import functional as F

from components.optim.kl_scheduler import adaptive_lr_from_kl
from components.normalization.running_stats import EmpiricalNormalization
from models.flow_cps_policy import FlowMatchingPolicy, flow_ode_mean


class FlowCPSBase:
    FLOW_CRITIC_SAMPLES = 4
    FLOW_CRITIC_FM_SAMPLES = 1

    def __init__(self, cfg, env):
        self.cfg = cfg
        self.env = env

    def build(self) -> None:
        cfg = self.cfg
        env = self.env
        self.num_act = env.action_dim
        self.actor_obs_dim = int(env.observation_dim)
        self.critic_obs_dim = env.critic_observation_dim
        self.horizon_h = int(cfg.horizon)

        self._policy = FlowMatchingPolicy(
            obs_dim=self.actor_obs_dim,
            action_dim=self.num_act,
            horizon=self.horizon_h,
            hidden_dims=tuple(cfg.actor_hidden_dims),
            activation=cfg.activation,
            action_squash_scale=float(cfg.action_squash_scale),
        ).to(env.device)
        steps = int(cfg.flow_steps)
        self.cps_cov_rank = int(cfg.cps_cov_rank)
        init_diag = math.log(math.exp(1.0) - 1.0)
        self._policy.cps_diag_raw = nn.Parameter(
            torch.full((steps, self.num_act), init_diag, device=env.device)
        )
        self._policy.cps_lowrank_raw = nn.Parameter(
            1.0e-3
            * torch.randn(
                steps,
                self.num_act,
                self.cps_cov_rank,
                device=env.device,
            )
        )
        eta_init = float(cfg.cps_noise_init)
        eta_eps = 1.0e-4
        eta_unit = (eta_init - eta_eps) / (1.0 - 2.0 * eta_eps)
        eta_raw_init = math.log(eta_unit / (1.0 - eta_unit))
        self._policy.cps_eta_raw = nn.Parameter(
            torch.tensor(eta_raw_init, device=env.device)
        )
        self.chunk_dim = self._policy.chunk_dim

        self.actor_obs_normalizer = EmpiricalNormalization(
            self.actor_obs_dim, env.device
        )

        self.learning_rate = float(cfg.policy_lr)
        self.critic_learning_rate = float(cfg.value_lr)
        self.max_lr = 1e-2
        self.min_lr = 1e-5

        self.actor_optimizer = torch.optim.AdamW(
            self._policy.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=float(cfg.weight_decay),
        )
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
            "actor_obs_normalizer": self.actor_obs_normalizer.state_dict(),
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
        if payload.get("actor_obs_normalizer") is not None:
            self.actor_obs_normalizer.load_state_dict(payload["actor_obs_normalizer"])

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    def _chunks_per_update(self) -> int:
        return max(1, int(self.cfg.rollout_env_steps) // max(1, self.horizon_h))

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
        actor_obs = self.actor_obs_normalizer(obs)
        # prev_action is the raw (un-normalized) last action, which lives in the
        # last action_dim columns of the raw actor observation. Extract it from
        # the raw obs BEFORE normalization so the smooth transform anchors on the
        # true last executed action, not a normalized surrogate.
        prev_action = obs[..., -self.num_act:].detach()
        return self._flow_mean_actions(actor_obs, prev_action=prev_action)

    def evaluation_step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        return self.env.step(actions)

    def snapshot_runtime_state(self):
        return None

    def restore_runtime_state(self, state) -> None:
        del state

    def _flow_mean_latent(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Differentiable zero-noise CPS final latent used for eval/probing."""
        batch = actor_obs.shape[0]
        latent = torch.zeros(batch, self.chunk_dim, device=actor_obs.device, dtype=actor_obs.dtype)
        self._policy._validate_inputs(actor_obs, latent, int(self.cfg.flow_steps))
        obs_prep = self._policy._prepare_observation(actor_obs)
        steps = int(self.cfg.flow_steps)
        sigma_schedule = torch.linspace(1.0, 0.0, steps + 1, device=actor_obs.device, dtype=actor_obs.dtype)
        for step_index in range(steps):
            latent, _, _ = self._cps_backbone_step(
                obs_prep,
                latent,
                sigma_schedule,
                step_index,
            )
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
        eta_eps = torch.as_tensor(
            1.0e-4,
            device=reference.device,
            dtype=reference.dtype,
        )
        eta_unit = torch.sigmoid(
            self._policy.cps_eta_raw.to(
                device=reference.device,
                dtype=reference.dtype,
            )
        )
        eta = eta_eps + (1.0 - 2.0 * eta_eps) * eta_unit
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

    def _cps_eta_value(self) -> torch.Tensor:
        eta_eps = torch.as_tensor(
            1.0e-4,
            device=self._policy.cps_eta_raw.device,
            dtype=self._policy.cps_eta_raw.dtype,
        )
        return eta_eps + (1.0 - 2.0 * eta_eps) * torch.sigmoid(
            self._policy.cps_eta_raw
        )

    def _cps_covariance_factors(
        self,
        step_index: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        diag = (
            F.softplus(
                self._policy.cps_diag_raw[step_index].to(
                    device=device,
                    dtype=dtype,
                )
            )
            + 1.0e-4
        )
        lowrank = self._policy.cps_lowrank_raw[step_index].to(
            device=device,
            dtype=dtype,
        )
        trace = (
            diag.square().sum() + lowrank.square().sum()
        ).clamp(min=1.0e-12)
        scale = torch.sqrt(trace / float(self.num_act))
        diag = diag / scale
        lowrank = lowrank / scale
        cov = (
            torch.diag_embed(diag.square())
            + lowrank @ lowrank.transpose(0, 1)
        )
        chol = torch.linalg.cholesky(cov)
        log_diag_chol = torch.log(torch.diagonal(chol).clamp(min=1.0e-8))
        trace_normalized = (
            diag.square().sum() + lowrank.square().sum()
        ) / float(self.num_act)
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
        mean_next = flow_ode_mean(model_output, mean_latent, sigma_schedule, step_index)
        predicted_coeff, noise_coeff, _ = self._cps_step_coeffs(
            step_index,
            sigma_schedule,
            mean_latent,
        )
        return mean_next, predicted_coeff, noise_coeff

    def _sample_cps_innovation(
        self,
        *,
        batch: int,
        step_index: int,
        noise_coeff: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        diag, lowrank, _, _, _, _ = self._cps_covariance_factors(
            step_index,
            device=device,
            dtype=dtype,
        )
        eps_diag = torch.randn(
            batch,
            self.horizon_h,
            self.num_act,
            device=device,
            dtype=dtype,
        )
        eps_rank = torch.randn(
            batch,
            self.horizon_h,
            self.cps_cov_rank,
            device=device,
            dtype=dtype,
        )
        innovation = (
            eps_diag * diag.view(1, 1, self.num_act)
            + torch.matmul(eps_rank, lowrank.transpose(0, 1))
        )
        return innovation * noise_coeff

    def _cps_innovation_log_prob(
        self,
        innovation: torch.Tensor,
        noise_coeff: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        # Exact shared joint covariance density.  Offsets use independent
        # random realizations but the same learned action-space shape.
        batch = innovation.shape[0]
        safe_std = torch.clamp(noise_coeff, min=1.0e-6)
        _, _, _, chol, log_diag_chol, _ = self._cps_covariance_factors(
            step_index,
            device=innovation.device,
            dtype=innovation.dtype,
        )
        flat = innovation.reshape(
            batch * self.horizon_h,
            self.num_act,
        ) / safe_std
        whitened = torch.linalg.solve_triangular(
            chol,
            flat.transpose(0, 1),
            upper=False,
        ).transpose(0, 1)
        component_log_prob = (
            -0.5 * (whitened.square() + math.log(2.0 * math.pi))
            - log_diag_chol.view(1, self.num_act)
            - torch.log(safe_std)
        )
        return component_log_prob.sum(dim=-1).view(
            batch,
            self.horizon_h,
        )

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
        innovation_square_sums = torch.zeros(
            self.horizon_h,
            device=actor_obs.device,
            dtype=actor_obs.dtype,
        )
        latent_path[:, 0, :] = sampled_latent

        for step_index in range(steps):
            mean_next, predicted_coeff, noise_coeff = self._cps_backbone_step(
                obs_prep,
                mean_latent,
                sigma_schedule,
                step_index,
            )
            conditional_mean = mean_next + predicted_coeff * (
                sampled_latent - mean_latent
            )
            innovation = self._sample_cps_innovation(
                batch=batch,
                step_index=step_index,
                noise_coeff=noise_coeff,
                device=actor_obs.device,
                dtype=actor_obs.dtype,
            )
            sample = conditional_mean.view(
                batch,
                self.horizon_h,
                self.num_act,
            ) + innovation
            step_log_probs[:, step_index, :] = (
                self._cps_innovation_log_prob(
                    innovation,
                    noise_coeff,
                    step_index,
                )
            )
            innovation_square_sums += innovation.square().sum(dim=(0, 2))
            mean_latent = mean_next
            sampled_latent = sample.reshape(batch, self.chunk_dim)
            latent_path[:, step_index + 1, :] = sampled_latent

        innovation_mean_squares = innovation_square_sums / float(
            batch * steps * self.num_act
        )
        return (
            sampled_latent,
            latent_path,
            step_log_probs,
            innovation_mean_squares,
        )

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
            next_latent = latent_path[:, step_index + 1, :]
            mean_next, predicted_coeff, noise_coeff = self._cps_backbone_step(
                obs_prep,
                mean_latent,
                sigma_schedule,
                step_index,
            )
            conditional_mean = mean_next + predicted_coeff * (
                sampled_latent - mean_latent
            )
            innovation = (
                next_latent - conditional_mean
            ).view(batch, self.horizon_h, self.num_act)
            step_log_probs[:, step_index, :] = (
                self._cps_innovation_log_prob(
                    innovation,
                    noise_coeff,
                    step_index,
                )
            )
            mean_latent = mean_next

        return step_log_probs

    # ------------------------------------------------------------------ #
    # Optimizer helpers
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

    def _add_sampler_metrics(self, metrics: dict) -> None:
        stats = self.env.adaptive_sampling_stats()
        for key, value in stats.items():
            value = float(value)
            if not math.isfinite(value):
                raise FloatingPointError(
                    f"adaptive sampler statistic {key!r} is non-finite"
                )
            metrics[f"sampler/{key}"] = value
