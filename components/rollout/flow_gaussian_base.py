

"""Causal deterministic flow mean with trainable action-path Gaussian exploration.

The flow network predicts a four-action residual mean.  Residuals are first
integrated into the pre-tanh absolute-action path, where one independent,
trainable Gaussian standard deviation is applied per horizon offset and
joint.  The sampled path is differenced back into residuals before the
existing action transform.  PPO therefore operates on the exact action-path
Gaussian density; there is no stochastic flow path or hidden covariance
process.
"""

from __future__ import annotations

import math
from collections import deque

import torch
from torch import nn

from components.optim.kl_scheduler import adaptive_lr_from_kl
from components.normalization.running_stats import EmpiricalNormalization
from models.flow_chunk_policy import FlowMatchingPolicy, flow_ode_mean


class FlowGaussianBase:
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
        self.action_path_std_min = float(cfg.gaussian_path_std_min)
        self.action_path_std_max = float(cfg.gaussian_path_std_max)
        self._policy.action_path_log_std = nn.Parameter(
            torch.full(
                (self.horizon_h, self.num_act),
                math.log(float(cfg.gaussian_path_init_std)),
                device=env.device,
            )
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
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        return self.env.step(actions)

    def snapshot_runtime_state(self):
        return None

    def restore_runtime_state(self, state) -> None:
        del state

    def _flow_mean_latent(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Differentiable deterministic final residual latent."""
        batch = actor_obs.shape[0]
        latent = torch.zeros(batch, self.chunk_dim, device=actor_obs.device, dtype=actor_obs.dtype)
        self._policy._validate_inputs(actor_obs, latent, int(self.cfg.flow_steps))
        obs_prep = self._policy._prepare_observation(actor_obs)
        steps = int(self.cfg.flow_steps)
        sigma_schedule = torch.linspace(1.0, 0.0, steps + 1, device=actor_obs.device, dtype=actor_obs.dtype)
        for step_index in range(steps):
            sigma = sigma_schedule[step_index]
            timestep_batch = torch.full(
                (batch,),
                float(sigma.item()),
                device=latent.device,
                dtype=latent.dtype,
            )
            model_output = self._policy.velocity_field(
                obs_prep,
                latent,
                timestep_batch,
            )
            latent = flow_ode_mean(
                model_output,
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

    @staticmethod
    def _path_from_residual(residual: torch.Tensor) -> torch.Tensor:
        return torch.cumsum(residual, dim=-2)

    @staticmethod
    def _residual_from_path(path: torch.Tensor) -> torch.Tensor:
        previous = torch.cat(
            (torch.zeros_like(path[..., :1, :]), path[..., :-1, :]),
            dim=-2,
        )
        return path - previous

    def _bounded_action_path_log_std(self) -> torch.Tensor:
        return self._policy.action_path_log_std.clamp(
            min=math.log(self.action_path_std_min),
            max=math.log(self.action_path_std_max),
        )

    @torch.no_grad()
    def _clamp_action_path_log_std_(self) -> None:
        self._policy.action_path_log_std.clamp_(
            min=math.log(self.action_path_std_min),
            max=math.log(self.action_path_std_max),
        )

    @staticmethod
    def _diagonal_gaussian_log_prob(
        sample: torch.Tensor,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> torch.Tensor:
        standardized = (sample - mean) * torch.exp(-log_std)
        component = (
            -0.5 * standardized.square()
            - log_std
            - 0.5 * math.log(2.0 * math.pi)
        )
        return component.sum(dim=-1)

    def _sample_gaussian_action_path(
        self,
        actor_obs: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        batch = actor_obs.shape[0]
        mean_residual = self._flow_mean_latent(actor_obs).view(
            batch,
            self.horizon_h,
            self.num_act,
        )
        mean_path = self._path_from_residual(mean_residual)
        log_std = self._bounded_action_path_log_std().to(
            device=actor_obs.device,
            dtype=actor_obs.dtype,
        )
        sampled_path = (
            mean_path
            + torch.exp(log_std) * torch.randn_like(mean_path)
        )
        sampled_residual = self._residual_from_path(sampled_path)
        log_prob = self._diagonal_gaussian_log_prob(
            sampled_path,
            mean_path,
            log_std,
        )
        return (
            sampled_residual.reshape(batch, self.chunk_dim),
            sampled_path,
            log_prob,
            mean_path,
            log_std,
        )

    def _recompute_gaussian_action_path_stats(
        self,
        actor_obs: torch.Tensor,
        sampled_path: torch.Tensor,
        old_mean_path: torch.Tensor,
        old_log_std: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = actor_obs.shape[0]
        sample = sampled_path.view(
            batch,
            self.horizon_h,
            self.num_act,
        )
        new_mean_residual = self._flow_mean_latent(actor_obs).view_as(
            sample
        )
        new_mean_path = self._path_from_residual(new_mean_residual)
        new_log_std = self._bounded_action_path_log_std().to(
            device=actor_obs.device,
            dtype=actor_obs.dtype,
        )
        old_mean_path = old_mean_path.view_as(sample)
        old_log_std = old_log_std.to(
            device=actor_obs.device,
            dtype=actor_obs.dtype,
        )
        if old_log_std.shape != (self.horizon_h, self.num_act):
            raise ValueError(
                "old Gaussian log_std must have shape "
                f"({self.horizon_h}, {self.num_act})"
            )
        new_log_prob = self._diagonal_gaussian_log_prob(
            sample,
            new_mean_path,
            new_log_std,
        )
        old_variance = torch.exp(2.0 * old_log_std)
        new_variance = torch.exp(2.0 * new_log_std)
        analytic_kl = (
            new_log_std
            - old_log_std
            + (
                old_variance
                + (old_mean_path - new_mean_path).square()
            )
            / (2.0 * new_variance)
            - 0.5
        ).sum(dim=-1)
        entropy = (
            new_log_std
            + 0.5 * math.log(2.0 * math.pi * math.e)
        ).sum(dim=-1).unsqueeze(0).expand(batch, -1)
        return new_log_prob, analytic_kl, entropy

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
            if math.isfinite(value):
                metrics[f"sampler/{key}"] = value
