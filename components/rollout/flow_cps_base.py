"""Flow-CPS over an H-frame sequence of bounded action innovations."""

from __future__ import annotations

import math
from collections import deque

import torch
from torch import nn

from method.base import Algorithm
from components.normalization.running_stats import EmpiricalNormalization
from models.flow_cps_policy import FlowMatchingPolicy


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
        ).to(env.device)
        self.cps_raw_rms = float(cfg.cps_raw_rms)
        if not math.isfinite(self.cps_raw_rms) or self.cps_raw_rms <= 0.0:
            raise ValueError(
                f"cps_raw_rms must be finite and > 0, got {self.cps_raw_rms}"
            )
        self.innovation_step_bound = float(cfg.innovation_step_bound)
        if (
            not math.isfinite(self.innovation_step_bound)
            or self.innovation_step_bound <= 0.0
        ):
            raise ValueError(
                "innovation_step_bound must be finite and > 0, got "
                f"{self.innovation_step_bound}"
            )

        action_low = env.action_low.to(device=env.device, dtype=torch.float32)
        action_high = env.action_high.to(device=env.device, dtype=torch.float32)
        if action_low.shape != (self.num_act,) or action_high.shape != (self.num_act,):
            raise ValueError(
                f"Action bounds must have shape ({self.num_act},), got "
                f"{tuple(action_low.shape)} and {tuple(action_high.shape)}"
            )
        if not bool(torch.isfinite(action_low).all()) or not bool(
            torch.isfinite(action_high).all()
        ):
            raise ValueError("Action bounds must be finite")
        if not torch.allclose(action_low, -action_high, atol=1.0e-7, rtol=0.0):
            raise ValueError("Innovation decoder requires symmetric action bounds")
        if not torch.allclose(
            action_high,
            action_high[:1].expand_as(action_high),
            atol=1.0e-7,
            rtol=0.0,
        ):
            raise ValueError("Innovation decoder requires one shared action bound")
        self.policy_action_bound = float(action_high[0].item())
        if self.innovation_step_bound >= 2.0 * self.policy_action_bound:
            raise ValueError(
                "innovation_step_bound must be below the full action range "
                f"{2.0 * self.policy_action_bound}"
            )

        self._cps_flat_dim = self.horizon_h * self.num_act
        self._policy.register_buffer(
            "cps_raw_rms",
            torch.as_tensor(
                self.cps_raw_rms,
                device=env.device,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.chunk_dim = self._policy.chunk_dim

        self.actor_obs_normalizer = EmpiricalNormalization(
            self.actor_obs_dim, env.device
        )

        self.learning_rate = float(cfg.policy_lr)
        self.min_lr = 1e-5
        if self.learning_rate <= 0.0:
            raise ValueError(f"policy_lr must be > 0, got {cfg.policy_lr}")

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

    def extra_checkpoint_state(self) -> dict:
        return {
            "learning_rate": float(self.learning_rate),
            "actor_obs_normalizer": self.actor_obs_normalizer.state_dict(),
        }

    def load_extra_checkpoint_state(self, payload: dict, reset_optimizer: bool = False) -> None:
        if reset_optimizer:
            self.learning_rate = float(self.cfg.policy_lr)
            for group in self.actor_optimizer.param_groups:
                group["lr"] = self.learning_rate
        elif payload:
            self.learning_rate = float(payload.get("learning_rate", self.learning_rate))
            for group in self.actor_optimizer.param_groups:
                group["lr"] = self.learning_rate
        if not payload:
            return
        if payload.get("actor_obs_normalizer") is not None:
            self.actor_obs_normalizer.load_state_dict(
                payload["actor_obs_normalizer"]
            )

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    def _chunks_per_update(self) -> int:
        return max(1, int(self.cfg.rollout_env_steps) // max(1, self.horizon_h))

    def _norm_actor(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_obs_normalizer(obs, update=False)


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



    def _flow_mean_innovation(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Integrate the deterministic mean of all H raw innovations."""
        if actor_obs.ndim != 2 or actor_obs.shape[-1] != self.actor_obs_dim:
            raise ValueError(
                f"actor_obs must have shape [B,{self.actor_obs_dim}], "
                f"got {tuple(actor_obs.shape)}"
            )
        batch = actor_obs.shape[0]
        latent = torch.zeros(batch, self.chunk_dim, device=actor_obs.device, dtype=actor_obs.dtype)
        steps = int(self.cfg.flow_steps)
        for step_index in range(steps):
            timestep = torch.full(
                (batch,),
                float(step_index) / float(steps),
                device=latent.device,
                dtype=latent.dtype,
            )
            velocity = self._policy.velocity_field(actor_obs, latent, timestep)
            latent = latent + velocity / float(steps)
        return latent

    def _action_chunk_from_innovations(
        self,
        raw_innovations: torch.Tensor,
        initial_action: torch.Tensor,
    ) -> torch.Tensor:
        """Map [B,H*A] innovations and [B,A] state to bounded [B,H,A] actions."""

        batch = initial_action.shape[0]
        expected_action_shape = (batch, self.num_act)
        expected_innovation_shape = (batch, self.chunk_dim)
        if initial_action.shape != expected_action_shape:
            raise ValueError(
                f"initial_action must have shape {expected_action_shape}, "
                f"got {tuple(initial_action.shape)}"
            )
        if raw_innovations.shape != expected_innovation_shape:
            raise ValueError(
                f"raw_innovations must have shape {expected_innovation_shape}, "
                f"got {tuple(raw_innovations.shape)}"
            )
        if not bool(torch.isfinite(initial_action).all()):
            raise FloatingPointError("initial_action contains non-finite values")
        if not bool(torch.isfinite(raw_innovations).all()):
            raise FloatingPointError("raw_innovations contain non-finite values")

        bound = torch.as_tensor(
            self.policy_action_bound,
            device=initial_action.device,
            dtype=initial_action.dtype,
        )
        normalized_initial = initial_action / bound
        violation = (normalized_initial.abs() - 1.0).clamp_min(0.0).max()
        if float(violation.item()) > 1.0e-6:
            raise ValueError(
                "initial_action exceeds the policy action bound by "
                f"{float(violation.item()) * self.policy_action_bound:.3e}"
            )
        innovations = raw_innovations.to(dtype=initial_action.dtype).view(
            batch,
            self.horizon_h,
            self.num_act,
        )
        normalized_step_bound = 2.0 * math.atanh(
            float(self.innovation_step_bound)
            / (2.0 * float(self.policy_action_bound))
        )
        action = initial_action
        actions = []
        for innovation in innovations.unbind(dim=1):
            # tanh(x + eta) addition identity.  Re-anchoring after every frame
            # makes H=8, H=4+4 and H=1x8 bitwise identical in finite precision.
            step = torch.tanh(
                normalized_step_bound * torch.tanh(innovation)
            )
            normalized_action = action / bound
            action = bound * (
                (normalized_action + step)
                / (1.0 + normalized_action * step)
            )
            actions.append(action)
        return torch.stack(actions, dim=1)

    def _effective_joint_cholesky(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return shared joint Cholesky with exact target raw RMS per frame."""

        chol = self._policy.joint_cholesky_shape(
            device=device,
            dtype=dtype,
        )
        shape_energy = chol.square().sum() / float(self.num_act)
        scale = self._policy.cps_raw_rms.to(
            device=device,
            dtype=dtype,
        ) / torch.sqrt(shape_energy)
        return chol * scale

    @torch.no_grad()
    def _cps_statistics(self) -> dict[str, float]:
        """Describe the shared iid innovation covariance used by PPO."""

        device = self.env.device
        effective_chol = self._effective_joint_cholesky(
            device=device,
            dtype=torch.float32,
        )
        covariance = effective_chol @ effective_chol.transpose(0, 1)
        covariance_diag = torch.diagonal(covariance)
        covariance_offdiag = covariance - torch.diag_embed(covariance_diag)
        achieved_raw_rms = torch.sqrt(
            effective_chol.square().sum() / float(self.num_act)
        )
        covariance_logdet = 2.0 * torch.log(
            torch.diagonal(effective_chol).clamp(min=1.0e-12)
        ).sum()
        raw_shape_norm = torch.linalg.vector_norm(
            self._policy.joint_cholesky_raw.detach()
        )
        actor_obs_scale = (
            self.actor_obs_normalizer._std
            + float(self.actor_obs_normalizer.eps)
        )
        shape_radius = float(
            self._policy.JOINT_CHOLESKY_SHAPE_RADIUS
        )
        return {
            "policy/cps_raw_rms_target": float(
                self._policy.cps_raw_rms.item()
            ),
            "policy/cps_raw_rms_achieved": float(
                achieved_raw_rms.item()
            ),
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
            "policy/actor_obs_normalizer_count": float(
                self.actor_obs_normalizer.count.item()
            ),
            "policy/actor_obs_normalizer_mean_abs": float(
                self.actor_obs_normalizer._mean.abs().mean().item()
            ),
            "policy/actor_obs_normalizer_scale_min": float(
                actor_obs_scale.min().item()
            ),
            "policy/actor_obs_normalizer_scale_mean": float(
                actor_obs_scale.mean().item()
            ),
            "policy/actor_obs_normalizer_scale_max": float(
                actor_obs_scale.max().item()
            ),
            "policy/actor_obs_normalizer_frozen": 1.0,
        }

    def _innovation_frame_log_prob(
        self,
        raw_innovation: torch.Tensor,
        mean_innovation: torch.Tensor,
    ) -> torch.Tensor:
        """Ordinary 29-D Gaussian log probability for each iid time frame."""

        if (
            raw_innovation.shape != mean_innovation.shape
            or raw_innovation.shape[-1] != self._cps_flat_dim
        ):
            raise ValueError(
                "raw_innovation and mean_innovation must have matching "
                f"[B,{self._cps_flat_dim}] shapes, got "
                f"{tuple(raw_innovation.shape)} and {tuple(mean_innovation.shape)}"
            )
        chol = self._effective_joint_cholesky(
            device=raw_innovation.device,
            dtype=raw_innovation.dtype,
        )
        residual = (raw_innovation - mean_innovation).view(
            -1,
            self.num_act,
        )
        whitened = torch.linalg.solve_triangular(
            chol,
            residual.transpose(0, 1),
            upper=False,
        ).transpose(0, 1)
        normalizer = (
            0.5 * float(self.num_act) * math.log(2.0 * math.pi)
            + torch.log(torch.diagonal(chol)).sum()
        )
        return (
            -0.5 * whitened.square().sum(dim=-1) - normalizer
        ).view(raw_innovation.shape[0], self.horizon_h)

    def _expected_innovation_frame_kl(
        self,
        old_mean: torch.Tensor,
        new_mean: torch.Tensor,
        old_chol: torch.Tensor,
        new_chol: torch.Tensor,
    ) -> torch.Tensor:
        """Return exact old||new KL for every independent frame."""

        if (
            old_mean.shape != new_mean.shape
            or old_mean.ndim != 2
            or old_mean.shape[-1] != self._cps_flat_dim
        ):
            raise ValueError(
                "old_mean and new_mean must have matching "
                f"[B,{self._cps_flat_dim}] shapes"
            )
        expected_chol_shape = (self.num_act, self.num_act)
        if old_chol.shape != expected_chol_shape or new_chol.shape != expected_chol_shape:
            raise ValueError(
                "old_chol and new_chol must both have shape "
                f"{expected_chol_shape}"
            )

        work_dtype = torch.float64
        old_mu = old_mean.view(
            -1,
            self.horizon_h,
            self.num_act,
        ).to(dtype=work_dtype)
        new_mu = new_mean.view_as(old_mu).to(dtype=work_dtype)
        old_l = old_chol.to(device=old_mean.device, dtype=work_dtype)
        new_l = new_chol.to(device=old_mean.device, dtype=work_dtype)
        covariance_whitened = torch.linalg.solve_triangular(
            new_l,
            old_l,
            upper=False,
        )
        mean_difference = (old_mu - new_mu).reshape(-1, self.num_act)
        mean_whitened = torch.linalg.solve_triangular(
            new_l,
            mean_difference.transpose(0, 1),
            upper=False,
        ).transpose(0, 1).view(old_mean.shape[0], self.horizon_h, self.num_act)
        covariance_term = (
            covariance_whitened.square().sum()
            - float(self.num_act)
            + 2.0
            * (
                torch.log(torch.diagonal(new_l)).sum()
                - torch.log(torch.diagonal(old_l)).sum()
            )
        )
        frame_kl = 0.5 * (
            covariance_term + mean_whitened.square().sum(dim=-1)
        )
        minimum = float(frame_kl.min().item())
        if minimum < -1.0e-7:
            raise FloatingPointError(
                f"Gaussian frame KL became materially negative: min={minimum:.3e}"
            )
        return frame_kl.clamp_min(0.0).to(dtype=old_mean.dtype)

    def _sample_innovations(
        self,
        actor_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample H iid-in-time, joint-across-joints raw innovations."""

        mean_innovation = self._flow_mean_innovation(actor_obs)
        chol = self._effective_joint_cholesky(
            device=actor_obs.device,
            dtype=actor_obs.dtype,
        )
        mean_frames = mean_innovation.view(-1, self.horizon_h, self.num_act)
        epsilon = torch.randn_like(mean_frames)
        raw_innovation = (
            mean_frames + epsilon @ chol.transpose(0, 1)
        ).reshape_as(mean_innovation)
        frame_logp = self._innovation_frame_log_prob(
            raw_innovation,
            mean_innovation,
        )
        return raw_innovation, mean_innovation, frame_logp

    def _recompute_innovation_log_prob(
        self,
        actor_obs: torch.Tensor,
        raw_innovation: torch.Tensor,
    ) -> torch.Tensor:
        mean_innovation = self._flow_mean_innovation(actor_obs)
        return self._innovation_frame_log_prob(
            raw_innovation,
            mean_innovation,
        )

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        del update_idx
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
