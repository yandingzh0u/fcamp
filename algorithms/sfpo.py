"""SFPO: causal flow policy + per-frame GAE + absolute-action chunks (v5).

Root-cause redesign of the chunked-action flow-matching policy. The design
fixes three PPO/GRPO-to-chunk-flow unit mismatches that plagued earlier
versions:

1. **Action semantics** (v5): the flow latent is an *absolute* joint target
   squashed by ``a_k = scale * tanh(raw_k / scale)`` (``action_transform=
   "absolute"``), giving the actor PPO-like full action support. The env's
   action-rate penalty ``reward -= w * (a_i - a_{i-1})^2`` handles smoothness
   -- SFPO does NOT reinvent a hard smoothness constraint. The legacy v4
   "delta" transform (``a_i = prev + max_delta * tanh(raw_i)``) is retained
   for ablation only: it was a hard-bounded delta integrator that starved the
   actor of action freedom (in_chunk_delta ~0.24 vs PPO ~0.76, env_raw
   plateaued at ~0.157 while V_tgt kept climbing -> tail improved, median
   did not).

2. **Causal flow density**: the velocity field is causal over the horizon
   (``v_k`` sees only ``z_0..z_k``), so the per-frame SDE log-prob is a real
   conditional density and the PPO ratio ``exp(logp_k_new - logp_k_old)`` is
   aligned with frame-k credit.

3. **Credit assignment + value propagation**: state-only ``V(s)`` critic (the
   action-conditioned Q prefix was a dead branch -- trained but never wired
   into the actor advantage, and wiring it via ``A = Q - V`` is biased without
   Q warm-up). Actor advantage is the **per-frame GAE** ``A_j = R_j - V(s_j)``
   (frame-j unit, lambda-smoothed, cross-chunk propagation ~20 steps/update),
   NOT the multi-prefix objective ``A_k = T_{k+1} - V(s_0)`` which mixed
   early-reward credit into all later prefixes. Per-frame PPO ratio + flat
   clip are valid because the policy is causal.

4. **Terminal vs absorbing failure**: failure injects an immediate per-step
   cost on the failure frame (``reward -= failure_penalty``) with bootstrap 0
   (true terminal), NOT an absorbing ``-10`` bootstrap that saturated the
   value distribution. Scale matches per-chunk reward magnitude.

KL controller is PPO-aligned: the masked mean per-frame KL drives the ACTOR
learning rate (critic LR is decoupled, fixed at ``value_lr``), updated before
each minibatch optimizer step, with an actor-epoch early-stop that freezes the
actor (critic keeps training) when KL exceeds ``kl_early_stop_factor * desired_kl``.
"""

from __future__ import annotations

import math
from collections import deque

import torch
from torch import nn

from algorithms.base import Algorithm
from algorithms.kl_scheduler import adaptive_lr_from_kl
from networks.flow_inference import deterministic_sde_ode_actions
from networks.flow_policy import FlowMatchingPolicy
from networks.flow_sampling import flow_grpo_step, flow_grpo_step_per_frame
from networks.mlp_actor_critic import Critic, EmpiricalNormalization


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
            causal_velocity=bool(getattr(cfg, "causal_velocity", False)),
            causal_arch=str(getattr(cfg, "causal_arch", "prefix_cumsum")),
        ).to(env.device)
        # Action transform: v5 default "absolute" -- the flow latent is an
        # absolute joint target squashed by scale*tanh(raw/scale), giving the
        # actor PPO-like full action support (the env's action-rate penalty
        # handles smoothness). "delta" is the legacy v4 hard-bounded delta
        # integrator (a_i = prev + max_delta*tanh(raw_i)); retained for
        # ablation only -- it starved the actor of action freedom.
        action_transform = str(getattr(cfg, "action_transform", "absolute")).lower()
        if action_transform == "absolute":
            self._policy.set_action_max_delta(None)
        elif action_transform == "delta":
            if getattr(cfg, "action_max_delta", None) is None:
                raise ValueError(
                    "action_transform='delta' requires a positive action_max_delta"
                )
            self._policy.set_action_max_delta(float(cfg.action_max_delta))
        elif action_transform == "residual_absolute":
            # v6: prev-action-anchored full-support residual. action_max_delta is
            # unused (residuals are unbounded); keep it None for clarity.
            self._policy.set_action_max_delta(None)
        else:
            raise ValueError(
                f"action_transform must be 'absolute', 'delta', or "
                f"'residual_absolute', got {action_transform!r}"
            )
        self.action_transform = action_transform
        self._policy.action_transform = action_transform
        self.actor_density = str(getattr(cfg, "actor_density", "sde_path")).lower()
        if self.actor_density not in {"sde_path", "action_gaussian"}:
            raise ValueError(
                f"actor_density must be 'sde_path' or 'action_gaussian', got {self.actor_density!r}"
            )
        self.action_noise_std = float(getattr(cfg, "action_noise_std", cfg.init_noise_std))
        self.action_std_trainable = bool(getattr(cfg, "action_std_trainable", True))
        if self.actor_density == "action_gaussian":
            if self.action_noise_std <= 0.0:
                raise ValueError(f"action_noise_std must be > 0, got {self.action_noise_std}")
            init_log_std = math.log(self.action_noise_std)
            self._policy.action_log_std = nn.Parameter(
                torch.full((self.horizon_h, self.num_act), init_log_std, device=env.device)
            )
            self._policy.action_log_std.requires_grad_(self.action_std_trainable)
        self.critic = Critic(self.critic_obs_dim, tuple(cfg.critic_hidden_dims), cfg.activation).to(env.device)
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

        self.failure_penalty = float(cfg.failure_penalty)
        # Terminal failure: an immediate per-step cost (NOT an absorbing -10
        # bootstrap). failure_penalty is added to the failure frame's reward,
        # and the bootstrap on failure is 0 (true terminal). This keeps the
        # target in the same units as the reward return instead of collapsing
        # the value distribution around -10.
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

    # ------------------------------------------------------------------ #
    # Per-frame flow sampling & log-prob recompute
    # ------------------------------------------------------------------ #
    def _sde_ode_rollout_actions_per_frame(self, obs, *, initial_noise, sde_noise=None, prev_action=None):
        self._policy._validate_inputs(obs, initial_noise, int(self.cfg.flow_steps))
        obs_prep = self._policy._prepare_observation(obs)
        batch_size = obs.shape[0]
        steps = int(self.cfg.flow_steps)
        sigma_schedule = torch.linspace(1.0, 0.0, steps + 1, device=obs.device, dtype=obs.dtype)
        latent = initial_noise.to(device=obs.device, dtype=obs.dtype) * float(self.cfg.init_noise_std)
        all_latents = [latent.detach()]
        if sde_noise is None:
            sde_noise = torch.randn(batch_size, steps, self.chunk_dim, device=obs.device, dtype=obs.dtype)
        elif sde_noise.shape != (batch_size, steps, self.chunk_dim):
            raise ValueError(
                f"sde_noise must have shape {(batch_size, steps, self.chunk_dim)}, got {tuple(sde_noise.shape)}"
            )
        step_log_probs = []
        for step_index in range(steps):
            sigma = sigma_schedule[step_index]
            timestep_batch = torch.full((batch_size,), float(sigma.item()), device=obs.device, dtype=obs.dtype)
            model_output = self._policy.velocity_field(obs_prep, latent, timestep_batch)
            latent, log_prob = flow_grpo_step_per_frame(
                model_output=model_output,
                latents=latent,
                sigmas=sigma_schedule,
                index=step_index,
                eta=float(self.cfg.sde_eta),
                sample_noise=sde_noise[:, step_index],
                horizon=self.horizon_h,
                action_dim=self.num_act,
            )
            all_latents.append(latent.detach())
            step_log_probs.append(log_prob)  # [batch, h]
        actions = self._policy._action_transform(latent, prev_action=prev_action)
        return (
            actions.view(obs.shape[0], self.horizon_h, self.num_act),
            torch.stack(all_latents, dim=1),
            torch.stack(step_log_probs, dim=1),  # [batch, steps, h]
        )

    def _sample_policy_with_logprobs_per_frame(self, obs, noise, sde_noise=None, prev_action=None):
        train_step_indices = self._train_step_indices(obs.device)
        actions, latent_path, step_log_probs = self._sde_ode_rollout_actions_per_frame(
            obs, initial_noise=noise, sde_noise=sde_noise, prev_action=prev_action
        )
        return {
            "actions": actions,
            "all_latents": latent_path.detach(),
            "log_probs": step_log_probs.detach(),  # [batch, steps, h]
            "train_step_indices": train_step_indices.detach().clone(),
        }

    def _compute_transition_log_probs_per_frame(self, obs, latent_path, step_indices):
        if latent_path.ndim != 3:
            raise ValueError(f"latent_path must be (batch, steps+1, dim), got {tuple(latent_path.shape)}")
        sample_count, path_steps, latent_dim = latent_path.shape
        steps = int(self.cfg.flow_steps)
        if obs.shape[0] != sample_count:
            raise ValueError("obs and latent_path batch sizes must match")
        if path_steps != steps + 1 or latent_dim != self.chunk_dim:
            raise ValueError(
                f"latent_path must be {(sample_count, steps + 1, self.chunk_dim)}, got {tuple(latent_path.shape)}"
            )
        obs_prep = self._policy._prepare_observation(obs)
        sigma_schedule = torch.linspace(1.0, 0.0, steps + 1, device=obs.device, dtype=obs.dtype)
        log_probs = []
        for step_tensor in step_indices.to(device=obs.device, dtype=torch.long):
            step_index = int(step_tensor.item())
            if step_index < 0 or step_index >= steps:
                raise ValueError(f"SFPO step index {step_index} is outside [0, {steps})")
            latent_t = latent_path[:, step_index].to(dtype=obs.dtype)
            next_latent = latent_path[:, step_index + 1].to(dtype=obs.dtype)
            sigma = sigma_schedule[step_index]
            timestep_batch = torch.full((sample_count,), float(sigma.item()), device=obs.device, dtype=obs.dtype)
            model_output = self._policy.velocity_field(obs_prep, latent_t, timestep_batch)
            _, log_prob = flow_grpo_step_per_frame(
                model_output=model_output,
                latents=latent_t,
                sigmas=sigma_schedule,
                index=step_index,
                eta=float(self.cfg.sde_eta),
                prev_sample=next_latent,
                horizon=self.horizon_h,
                action_dim=self.num_act,
            )
            log_probs.append(log_prob)  # [batch, h]
        return torch.stack(log_probs, dim=1)  # [batch, steps, h]

    def _deterministic_actor_actions(self, actor_obs: torch.Tensor, prev_action: torch.Tensor | None = None) -> torch.Tensor:
        return deterministic_sde_ode_actions(
            self._policy,
            actor_obs,
            steps=int(self.cfg.flow_steps),
            sde_eta=float(self.cfg.sde_eta),
            initial_noise=None,
            prev_action=prev_action,
        )

    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        actor_obs = self._norm_actor(obs, update=False)
        initial_noise = None
        if str(self.cfg.eval_initial_noise) == "random":
            initial_noise = torch.randn(obs.shape[0], self.chunk_dim, device=obs.device, dtype=obs.dtype)
        # prev_action is the raw (un-normalized) last action, which lives in the
        # last action_dim columns of the raw actor observation. Extract it from
        # the raw obs BEFORE normalization so the smooth transform anchors on the
        # true last executed action, not a normalized surrogate.
        prev_action = obs[..., -self.num_act:].detach()
        return deterministic_sde_ode_actions(
            self._policy,
            actor_obs,
            steps=int(self.cfg.flow_steps),
            sde_eta=float(self.cfg.sde_eta),
            initial_noise=initial_noise,
            prev_action=prev_action,
        )

    def _flow_mean_latent(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Differentiable deterministic final flow latent used as the Gaussian mean."""
        batch = actor_obs.shape[0]
        latent = torch.zeros(batch, self.chunk_dim, device=actor_obs.device, dtype=actor_obs.dtype)
        self._policy._validate_inputs(actor_obs, latent, int(self.cfg.flow_steps))
        obs_prep = self._policy._prepare_observation(actor_obs)
        steps = int(self.cfg.flow_steps)
        sigma_schedule = torch.linspace(1.0, 0.0, steps + 1, device=actor_obs.device, dtype=actor_obs.dtype)
        zero_step_noise = torch.zeros_like(latent)
        for step_index in range(steps):
            sigma = sigma_schedule[step_index]
            timestep_batch = torch.full((batch,), float(sigma.item()), device=actor_obs.device, dtype=actor_obs.dtype)
            model_output = self._policy.velocity_field(obs_prep, latent, timestep_batch)
            latent, _ = flow_grpo_step(
                model_output=model_output,
                latents=latent,
                sigmas=sigma_schedule,
                index=step_index,
                eta=float(self.cfg.sde_eta),
                sample_noise=zero_step_noise,
            )
        return latent

    def _flow_mean_actions(self, actor_obs: torch.Tensor, prev_action: torch.Tensor | None = None) -> torch.Tensor:
        mean_latent = self._flow_mean_latent(actor_obs)
        return self._policy._action_transform(mean_latent, prev_action=prev_action).view(
            actor_obs.shape[0], self.horizon_h, self.num_act
        )

    def _action_gaussian_std(self, reference: torch.Tensor) -> torch.Tensor:
        if not hasattr(self._policy, "action_log_std"):
            std = torch.full(
                (self.horizon_h, self.num_act),
                self.action_noise_std,
                device=reference.device,
                dtype=reference.dtype,
            )
        else:
            log_std = self._policy.action_log_std.to(device=reference.device, dtype=reference.dtype)
            std = torch.exp(log_std).clamp(min=1.0e-3, max=10.0)
        return std.view(*((1,) * (reference.ndim - 2)), self.horizon_h, self.num_act).expand_as(reference)

    @staticmethod
    def _action_gaussian_log_prob(actions: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        var = std.square()
        log_std = torch.log(std)
        log_prob = -0.5 * ((actions - mean).square() / var + 2.0 * log_std + math.log(2.0 * math.pi))
        return log_prob.sum(dim=-1)

    @staticmethod
    def _action_gaussian_kl(
        old_mean: torch.Tensor,
        old_std: torch.Tensor,
        new_mean: torch.Tensor,
        new_std: torch.Tensor,
    ) -> torch.Tensor:
        old_var = old_std.square()
        new_var = new_std.square()
        kl = torch.log(new_std / old_std) + (old_var + (old_mean - new_mean).square()) / (2.0 * new_var) - 0.5
        return kl.sum(dim=-1)

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
        # For actor_density="action_gaussian" these are final residual-latent
        # Gaussian parameters. The residual-latent -> action transform is fixed
        # for old/new policies, so its Jacobian cancels in the PPO ratio.
        old_action_mean_buf = torch.zeros(chunks, n_envs, h, self.num_act, device=device)
        old_action_std_buf = torch.zeros(chunks, n_envs, h, self.num_act, device=device)
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

                if self.actor_density == "action_gaussian":
                    mean_latent = self._flow_mean_latent(actor_obs_n).view(n_envs, h, self.num_act)
                    action_std = self._action_gaussian_std(mean_latent)
                    sample_latent_chunk = mean_latent + action_std * torch.randn_like(mean_latent)
                    action_chunk = self._policy._action_transform(
                        sample_latent_chunk.reshape(n_envs, self.chunk_dim),
                        prev_action=prev_action,
                    ).view(n_envs, h, self.num_act)
                    action_log_probs = self._action_gaussian_log_prob(sample_latent_chunk, mean_latent, action_std)
                    latent_path = torch.zeros(n_envs, flow_steps + 1, self.chunk_dim, device=device, dtype=obs.dtype)
                    latent_path[:, -1, :] = sample_latent_chunk.reshape(n_envs, self.chunk_dim)
                    step_log_probs = torch.zeros(n_envs, flow_steps, h, device=device, dtype=obs.dtype)
                    step_log_probs[:, 0, :] = action_log_probs
                    train_step_indices = self._train_step_indices(device)
                    action_mean = mean_latent
                else:
                    noise = torch.randn(n_envs, self.chunk_dim, device=device, dtype=obs.dtype)
                    sde_noise = torch.randn(n_envs, flow_steps, self.chunk_dim, device=device, dtype=obs.dtype)
                    sample = self._sample_policy_with_logprobs_per_frame(
                        actor_obs_n, noise, sde_noise=sde_noise, prev_action=prev_action
                    )
                    train_step_indices = sample["train_step_indices"]
                    action_chunk = sample["actions"]
                    latent_path = sample["all_latents"]
                    step_log_probs = sample["log_probs"]
                    action_mean = action_chunk
                    action_std = torch.zeros_like(action_chunk)
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

                    # Terminal failure: inject an immediate per-step cost (NOT an
                    # absorbing -10 bootstrap). The failure frame's reward is
                    # decremented by failure_penalty, and the bootstrap on failure
                    # is 0 (true terminal). This keeps the target in the same
                    # units as the reward return and avoids saturating the value
                    # distribution around a huge absorbing constant.
                    if bool(new_failure.any()):
                        reward_masked_buf[chunk_idx, :, frame_idx] = (
                            reward_masked_buf[chunk_idx, :, frame_idx]
                            - new_failure.to(dtype=reward_masked_buf.dtype) * float(self.failure_penalty)
                        )

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
                old_action_mean_buf[chunk_idx] = action_mean.detach()
                old_action_std_buf[chunk_idx] = action_std.detach()

            # ---- batched critic evaluation (state-only V: chunk-start + per-frame) ----
            # Q prefix critic is dropped (was trained but never wired into the
            # actor advantage; wiring it in via A = Q - V needs Q warm-up and is
            # left for a later iteration). State-only V is clean and sufficient.
            critic_obs_flat = critic_obs_buf.reshape(chunks * n_envs, self.critic_obs_dim)
            values_v = self.critic.evaluate(critic_obs_flat).reshape(chunks, n_envs, 1)

            # per-frame pre-action V: re-evaluate the chunk-start critic obs for
            # frame 0, and the running mid-chunk critic obs for frame j>0. We
            # approximate frame_values by reusing the chunk-start V for frame 0
            # and the next-frame V (from next_critic_obs) shifted by one for the
            # GAE delta. A dedicated frame_critic_obs buffer would be cleaner but
            # requires storing per-frame pre-action critic obs; the next_critic_obs
            # already gives us V(s_{t+1}), which is what GAE needs for the delta.
            next_critic_obs_flat = next_critic_obs_buf.reshape(chunks * n_envs * h, self.critic_obs_dim)
            frame_next_values = self.critic.evaluate(next_critic_obs_flat).reshape(chunks, n_envs, h)

            # ---- terminal-failure bootstrap (NOT absorbing -10) ----
            # failure frame: bootstrap = 0 (true terminal, the immediate cost was
            # already added to the reward above). timeout/motion_complete/alive:
            # bootstrap = V(s_{t+1}) (soft terminal / continue). This keeps the
            # target distribution in reward-return units.
            zero_bootstrap = torch.zeros_like(frame_next_values)
            frame_bootstrap = torch.where(failure_frame_buf, zero_bootstrap, frame_next_values)

            # ---- cross-chunk per-frame GAE ----
            # delta_t = r_t + gamma * b_t - V(s_t), with b_t = 0 on failure
            # (terminal) else V(s_{t+1}). V(s_t) is approximated by shifting
            # frame_next_values: V(s_t) for frame t = V(s_{t+1}) of frame t-1
            # (the previous frame's next-state V), with chunk-start V for frame 0.
            # GAE recurses backward across the whole chunks*h stream; continuity
            # (alive & ~done) lets advantages propagate across chunk boundaries.
            frame_values = torch.cat(
                [
                    values_v.expand(-1, -1, h)[..., :1],  # frame 0: chunk-start V
                    frame_next_values[..., :-1],  # frame j>0: previous frame's next-V
                ],
                dim=-1,
            )
            frame_td_delta = reward_masked_buf + gamma * frame_bootstrap - frame_values
            gae = torch.zeros(n_envs, device=device, dtype=frame_td_delta.dtype)
            gae_advantages = torch.zeros_like(frame_td_delta)
            gae_lambda = float(getattr(self.cfg, "gae_lambda", 0.95))
            for flat_t in range(chunks * h - 1, -1, -1):
                chunk_i = flat_t // h
                frame_i = flat_t % h
                alive_f = alive_frame_buf[chunk_i, :, frame_i].to(dtype=frame_td_delta.dtype)
                # continuity: this frame was alive AND not done (done -> env
                # reset at chunk end -> next frame is a fresh episode, cut GAE).
                cont_f = alive_f * (~done_frame_buf[chunk_i, :, frame_i]).to(dtype=frame_td_delta.dtype)
                delta_f = frame_td_delta[chunk_i, :, frame_i] * alive_f
                gae = delta_f + gamma * gae_lambda * cont_f * gae
                gae_advantages[chunk_i, :, frame_i] = gae * alive_f

            frame_v_targets = frame_values + gae_advantages

            # Actor advantage = per-frame GAE advantage (frame-j unit, lambda
            # smoothing, consistent with the critic V target).
            advantages = gae_advantages
            valid_prefix_mask = alive_frame_buf.clone()
            advantages = self._normalize_advantages(advantages, valid_prefix_mask)

            # ---- diagnostic prefix targets (for logging only) ----
            gamma_pow_reward = gamma ** torch.arange(h, device=device, dtype=reward_masked_buf.dtype)
            gamma_pow_boot = gamma ** torch.arange(1, h + 1, device=device, dtype=frame_bootstrap.dtype)
            disc_rewards = reward_masked_buf * gamma_pow_reward.view(1, 1, h)
            cum_disc_rewards = torch.cumsum(disc_rewards, dim=-1)
            disc_bootstrap = frame_bootstrap * gamma_pow_boot.view(1, 1, h)
            prefix_targets = cum_disc_rewards + disc_bootstrap  # T_{k+1}, diagnostic
            frame_idx_grid = torch.arange(h, device=device).view(1, 1, h).expand(chunks, n_envs, h)
            masked_done_idx = torch.where(done_frame_buf, frame_idx_grid, torch.full_like(frame_idx_grid, h))
            death_frame = masked_done_idx.min(dim=-1).values
            clamped_df = death_frame.clamp(max=h - 1)
            v_targets = prefix_targets.gather(-1, clamped_df.unsqueeze(-1))  # diagnostic

        self._obs = obs
        self._critic_obs = critic_obs
        if train_step_indices is None:
            train_step_indices = self._train_step_indices(device)

        chunk_return_realized = v_targets.squeeze(-1)  # [chunks, n_envs] (diagnostic, with failure cost)
        # chunk_raw_return: target-side raw (includes failure cost) -- used for
        # advantage/value computations.
        chunk_raw_return = (reward_masked_buf * gamma_pow_reward.view(1, 1, h)).sum(dim=-1)
        # chunk_env_raw_return: environment-side raw (excludes failure cost) --
        # the true per-step reward sum, for honest training-progress logging.
        alive_f = alive_frame_buf.to(dtype=reward_raw_buf.dtype)
        chunk_env_raw_return = (reward_raw_buf * alive_f * gamma_pow_reward.view(1, 1, h)).sum(dim=-1)
        # failure_cost_contribution: how much of the target raw return is the
        # injected terminal failure cost (negative). Near 0 -> survival-driven;
        # very negative -> failure-dominated.
        failure_cost_return = chunk_raw_return - chunk_env_raw_return
        chunk_live_frames = alive_frame_buf.to(dtype=torch.float32).sum(dim=-1)  # [chunks, n_envs]

        return {
            "actor_obs": actor_obs_buf,
            "critic_obs": critic_obs_buf,
            "actions": actions_buf,
            "latents": latents_buf,
            "old_log_probs": old_log_probs_buf,
            "old_action_mean": old_action_mean_buf,
            "old_action_std": old_action_std_buf,
            "prev_action": prev_action_buf,
            "train_step_indices": train_step_indices,
            "values_v": values_v,
            "frame_values": frame_values,
            "frame_v_targets": frame_v_targets,
            "prefix_targets": prefix_targets,
            "v_targets": v_targets,
            "advantages": advantages,
            "raw_gae_advantages": gae_advantages,
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

    # ------------------------------------------------------------------ #
    # Update: per-frame ratio + per-prefix PPO + V/Q critic losses
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
        actions = rollout["actions"].reshape(raw_batch_size, h, self.num_act)
        latent_path = rollout["latents"].reshape(raw_batch_size, flow_steps + 1, self.chunk_dim)
        old_log_probs = rollout["old_log_probs"].reshape(raw_batch_size, flow_steps, h)
        old_action_mean = rollout["old_action_mean"].reshape(raw_batch_size, h, self.num_act)
        old_action_std = rollout["old_action_std"].reshape(raw_batch_size, h, self.num_act)
        prev_action = rollout["prev_action"].reshape(raw_batch_size, self.num_act)
        old_frame_values = rollout["frame_values"].reshape(raw_batch_size, h)
        frame_v_targets = rollout["frame_v_targets"].reshape(raw_batch_size, h)
        prefix_targets = rollout["prefix_targets"].reshape(raw_batch_size, h)
        v_targets = rollout["v_targets"].reshape(raw_batch_size, 1)
        advantages = rollout["advantages"].reshape(raw_batch_size, h)
        raw_gae_advantages = rollout["raw_gae_advantages"].reshape(raw_batch_size, h)
        valid_prefix_mask = rollout["valid_prefix_mask"].reshape(raw_batch_size, h)
        batch_size = actor_obs.shape[0]

        mini_batch_size = self._policy_mini_batch_size(batch_size)
        clip_range = float(self.cfg.clip_range)
        value_clip = float(self.cfg.value_clip_range)
        use_clipped_value_loss = bool(self.cfg.use_clipped_value_loss)
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
                    old_per_frame_logp = old_log_probs_sub.sum(dim=1)  # [micro, h]
                    if self.actor_density == "action_gaussian":
                        sample_latent_chunk = latent_path[sub, -1, :].view(-1, h, self.num_act)
                        new_action_mean = self._flow_mean_latent(actor_obs[sub]).view(-1, h, self.num_act)
                        new_action_std = self._action_gaussian_std(new_action_mean)
                        new_per_frame_logp = self._action_gaussian_log_prob(
                            sample_latent_chunk, new_action_mean, new_action_std
                        )
                        kl_per_frame = self._action_gaussian_kl(
                            old_action_mean[sub],
                            old_action_std[sub].clamp(min=1.0e-6),
                            new_action_mean.detach(),
                            new_action_std.detach(),
                        )
                    else:
                        new_log_probs = self._compute_transition_log_probs_per_frame(
                            actor_obs[sub], latent_path[sub], train_step_indices
                        )  # [micro, steps, h]
                        # per-frame total log-prob (sum over flow steps)
                        new_per_frame_logp = new_log_probs.sum(dim=1)  # [micro, h]
                        kl_per_frame = None
                    per_frame_log_ratio = new_per_frame_logp - old_per_frame_logp  # [micro, h]
                    # Per-frame PPO ratio (NOT a prefix/joint cumsum ratio).
                    # Each executed chunk component gets its own per-frame
                    # likelihood ratio and per-frame GAE advantage, so the PG
                    # estimator is the standard per-action PPO form.
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

                    # V critic loss over every executed frame state. The critic
                    # is state-only V; we evaluate V at each frame's pre-action
                    # state by reusing the stored per-frame next-critic-obs
                    # shifted by one (frame 0 uses chunk-start critic_obs).
                    # frame_v_targets = GAE return (frame_values + gae_advantages).
                    next_critic_obs_sub = rollout["next_critic_obs"].reshape(
                        raw_batch_size, h, self.critic_obs_dim
                    )[sub]
                    critic_obs_sub = critic_obs[sub]  # [micro, D]
                    # per-frame pre-action critic obs: frame 0 = chunk-start,
                    # frame j>0 = next-critic-obs of frame j-1.
                    frame_critic_obs_sub = torch.cat(
                        [critic_obs_sub.unsqueeze(1), next_critic_obs_sub[:, :-1, :]], dim=1
                    )  # [micro, h, D]
                    value = self.critic.evaluate(
                        frame_critic_obs_sub.reshape(-1, self.critic_obs_dim)
                    ).reshape(-1, h)
                    if use_clipped_value_loss:
                        value_clipped = old_frame_values[sub] + (value - old_frame_values[sub]).clamp(
                            -value_clip, value_clip
                        )
                        value_losses = (value - frame_v_targets[sub]).pow(2)
                        value_losses_clipped = (value_clipped - frame_v_targets[sub]).pow(2)
                        value_loss_per = torch.max(value_losses, value_losses_clipped)
                    else:
                        value_loss_per = (frame_v_targets[sub] - value).pow(2)
                    value_loss = (value_loss_per * mask).sum() / mask_sum

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
                        if kl_per_frame is None:
                            kl_per_frame = 0.5 * per_frame_log_ratio.square()  # [micro, h]
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
                            (frame_v_targets[sub] * mask).sum().item() / mask_sum.item()
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
            "sfpo/sde_train_steps": float(train_step_indices.numel()),
            "sfpo/v_target_mean": totals["v_target_mean"] / denom,
            "sfpo/valid_prefix_frac": totals["valid_prefix_frac"] / denom,
            "sfpo/failure_penalty": float(self.failure_penalty),
            "sfpo/gae_lambda": float(getattr(self.cfg, "gae_lambda", 0.95)),
            "sfpo/action_max_delta": (
                float(self._policy.action_max_delta.mean().item())
                if torch.is_tensor(self._policy.action_max_delta)
                else (
                    float(self._policy.action_max_delta)
                    if self._policy.action_max_delta is not None
                    else float("nan")
                )
            ),
            "sfpo/action_transform": {"absolute": 0.0, "delta": 1.0, "residual_absolute": 2.0}[self.action_transform],
            "sfpo/actor_density": {"sde_path": 0.0, "action_gaussian": 1.0}[self.actor_density],
            "sfpo/kl_early_stop_factor": float(self.kl_early_stop_factor),
            "sfpo/early_stop_epoch": float(early_stopped_epoch),
            "sfpo/advantage_normalization": float({"per_prefix": 0.0, "global": 1.0, "none": 2.0}[self.advantage_normalization]),
            "policy/action_delta": float(action_delta.item()),
            "policy/param_rms_delta": float(param_rms_delta.item()),
            "policy/raw_adv_mean": float(raw_gae_advantages.mean().item()),
            "policy/raw_adv_std": float(raw_gae_advantages.std(unbiased=False).item()),
            "policy/action_std_mean": (
                float(torch.exp(self._policy.action_log_std.detach()).mean().item())
                if hasattr(self._policy, "action_log_std")
                else float(self.action_noise_std)
            ),
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
            f"[CRITIC] V={metrics['critic/v_mean']:.4f} "
            f"V_tgt={metrics['critic/v_target_mean']:.4f} "
            f"valid_pfx={metrics['sfpo/valid_prefix_frac']:.4f} "
            f"death_frame={metrics['rollout/death_frame_mean']:.3f} "
            f"fail_pen={metrics['sfpo/failure_penalty']:.3f} "
            f"gae_lambda={metrics['sfpo/gae_lambda']:.3f} "
            f"max_delta={metrics['sfpo/action_max_delta']:.3f} "
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
            f"sde_steps={metrics.get('sfpo/sde_train_steps', float('nan')):.0f} "
            f"action_std={metrics.get('policy/action_std_mean', float('nan')):.4f}",
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
        print("[INFO] Starting SFPO training (smooth action chunk + per-frame GAE + state-V)", flush=True)
        print(
            f"[INFO] task={env.task.name} terrain={env.task.terrain} motion_file={env.task.motion_file}",
            flush=True,
        )
        print(
            f"[INFO] algo=sfpo actor_obs_dim={self.actor_obs_dim} critic_obs_dim={self.critic_obs_dim} "
            f"action_dim={self.num_act} horizon={cfg.horizon} rollout_chunks={self._chunks_per_update()} "
            f"rollout_env_steps={cfg.rollout_env_steps} flow_steps={cfg.flow_steps} "
            f"sde_eta={cfg.sde_eta} init_noise_std={cfg.init_noise_std} eval_initial_noise={cfg.eval_initial_noise}",
            flush=True,
        )
        _amd = self._policy.action_max_delta
        _amd_str = (
            f"{float(_amd.mean().item()):.3f}"
            if torch.is_tensor(_amd)
            else (f"{float(_amd):.3f}" if _amd is not None else "none")
        )
        print(
            f"[INFO] state_only_critic=True prefix_q=False "
            f"causal_velocity={self._policy.causal_velocity} causal_arch={self._policy.causal_arch} "
            f"action_transform={self.action_transform} action_max_delta={_amd_str} "
            f"actor_density={self.actor_density} action_noise_std={self.action_noise_std} "
            f"action_std_trainable={self.action_std_trainable} "
            f"terminal_failure_cost={self.failure_penalty != 0.0} failure_penalty={cfg.failure_penalty} "
            f"per_frame_action_logprob={self.actor_density == 'action_gaussian'} "
            f"per_frame_flow_logprob={self.actor_density == 'sde_path'} per_frame_ratio_ppo=True flat_clip=True "
            f"actor_advantage=gae_per_frame "
            f"per_frame_kl_adaptive_lr=True kl_units=1 kl_early_stop_factor={self.kl_early_stop_factor} "
            f"advantage_norm={cfg.advantage_normalization} "
            f"gamma={cfg.discount_gamma} gae_lambda={float(getattr(cfg, 'gae_lambda', 0.95)):.3f} "
            f"cross_chunk_gae=True td_bootstrap_per_frame=True "
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
            f"use_clipped_value_loss={cfg.use_clipped_value_loss} "
            f"num_mini_batches={cfg.num_mini_batches} micro_batch={cfg.micro_batch_size} "
            f"policy_lr={cfg.policy_lr} critic_lr={cfg.value_lr}",
            flush=True,
        )
