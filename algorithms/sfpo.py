from __future__ import annotations

import math
from collections import deque

import torch
from torch import nn

from algorithms.base import Algorithm
from algorithms.kl_scheduler import adaptive_lr_from_kl
from networks.flow_inference import deterministic_sde_ode_actions
from networks.flow_policy import FlowMatchingPolicy
from networks.flow_sampling import flow_grpo_step
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
        self.action_chunk_dim = int(cfg.horizon) * self.num_act

        self._policy = FlowMatchingPolicy(
            obs_dim=self.actor_obs_dim,
            action_dim=self.num_act,
            horizon=int(cfg.horizon),
            hidden_dims=tuple(cfg.actor_hidden_dims),
            activation=cfg.activation,
            action_squash_scale=float(cfg.action_squash_scale),
        ).to(env.device)
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

        self.max_episode_steps = env.max_episode_steps
        self._policy_module = nn.ModuleDict({"actor": self._policy, "critic": self.critic})
        self._init_train_episode_stats()

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
        # Raw KL is a chunk-level (joint) quantity scaling with horizon;
        # normalize by horizon so desired_kl is a per-control-step budget.
        return max(1, int(self.cfg.horizon))

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

    def _chunks_per_update(self) -> int:
        return max(1, int(self.cfg.rollout_env_steps) // max(1, int(self.cfg.horizon)))

    def _training_rollout_horizon(self) -> int:
        return int(self.cfg.horizon) * self._chunks_per_update()

    def _train_step_indices(self, device) -> torch.Tensor:
        return torch.arange(int(self.cfg.flow_steps), device=device, dtype=torch.long)

    def _norm_actor(self, obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        return self.actor_obs_normalizer(obs, update=update) if self.empirical_normalization else obs

    def _norm_critic(self, obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        return self.critic_obs_normalizer(obs, update=update) if self.empirical_normalization else obs

    def _norm_actor_with_mask(self, obs: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        if not self.empirical_normalization:
            return obs
        if bool(active.any()):
            self.actor_obs_normalizer(obs[active], update=True)
        return self.actor_obs_normalizer(obs, update=False)

    def _norm_critic_with_mask(self, obs: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        if not self.empirical_normalization:
            return obs
        if bool(active.any()):
            self.critic_obs_normalizer(obs[active], update=True)
        return self.critic_obs_normalizer(obs, update=False)

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

    def _sde_ode_rollout_actions(self, obs, *, initial_noise, sde_noise=None):
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
            latent, log_prob = flow_grpo_step(
                model_output=model_output,
                latents=latent,
                sigmas=sigma_schedule,
                index=step_index,
                eta=float(self.cfg.sde_eta),
                sample_noise=sde_noise[:, step_index],
            )
            all_latents.append(latent.detach())
            step_log_probs.append(log_prob)
        actions = self._policy._action_transform(latent)
        return actions, torch.stack(all_latents, dim=1), torch.stack(step_log_probs, dim=1)

    def _sample_policy_with_logprobs(self, obs, noise, sde_noise=None):
        train_step_indices = self._train_step_indices(obs.device)
        actions, latent_path, step_log_probs = self._sde_ode_rollout_actions(
            obs, initial_noise=noise, sde_noise=sde_noise
        )
        return {
            "actions": actions.view(obs.shape[0], self._policy.horizon, self._policy.action_dim),
            "all_latents": latent_path.detach(),
            "log_probs": step_log_probs.detach(),
            "train_step_indices": train_step_indices.detach().clone(),
        }

    def _training_anchor_phases(self) -> torch.Tensor:
        anchors = self.env.sample_phase_indices(self.env.num_envs, self._training_rollout_horizon())
        return anchors.to(device=self.env.device, dtype=torch.long)

    def _compute_transition_log_probs(self, obs, latent_path, step_indices):
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
            _, log_prob = flow_grpo_step(
                model_output=model_output,
                latents=latent_t,
                sigmas=sigma_schedule,
                index=step_index,
                eta=float(self.cfg.sde_eta),
                prev_sample=next_latent,
            )
            log_probs.append(log_prob)
        return torch.stack(log_probs, dim=1)

    def _deterministic_actor_actions(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return deterministic_sde_ode_actions(
            self._policy,
            actor_obs,
            steps=int(self.cfg.flow_steps),
            sde_eta=float(self.cfg.sde_eta),
            initial_noise=None,
        )

    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        actor_obs = self._norm_actor(obs, update=False)
        initial_noise = None
        if str(self.cfg.eval_initial_noise) == "random":
            initial_noise = torch.randn(obs.shape[0], self.chunk_dim, device=obs.device, dtype=obs.dtype)
        return deterministic_sde_ode_actions(
            self._policy,
            actor_obs,
            steps=int(self.cfg.flow_steps),
            sde_eta=float(self.cfg.sde_eta),
            initial_noise=initial_noise,
        )

    def initial_reset(self) -> torch.Tensor:
        obs = self.env.reset()
        if bool(self.cfg.init_at_random_ep_len) and self.max_episode_steps > 0:
            self.env.episode_steps = torch.randint_like(self.env.episode_steps, high=int(self.max_episode_steps))
        self._obs = obs
        self._critic_obs = self.env.get_critic_observation()
        return obs

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        # PPO-aligned: keep the running environment across updates. Episode
        # boundaries are handled by chunk-internal masks and chunk-end resets
        # inside collect(); we do NOT force-reset here, so samples form one
        # continuous PPO-style stream instead of the old first-life per-update
        # rollout. Episode-stat accumulators persist and reset on done.
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
    ) -> None:
        newly_done = (~ever_done) & new_done
        if not bool(newly_done.any()):
            return
        ids = newly_done.nonzero(as_tuple=False).squeeze(-1)
        first_done_step[ids] = int(global_step)
        first_done_timeout[ids] = timeouts[ids]
        dterms = info["done_terms"]
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

    def collect(self, current_obs: torch.Tensor) -> dict:
        env = self.env
        device = env.device
        n_envs = env.num_envs
        chunks = self._chunks_per_update()
        horizon = int(self.cfg.horizon)
        gamma = float(self.cfg.discount_gamma)

        actor_obs_buf = torch.zeros(chunks, n_envs, self.actor_obs_dim, device=device)
        critic_obs_buf = torch.zeros(chunks, n_envs, self.critic_obs_dim, device=device)
        values_buf = torch.zeros(chunks, n_envs, 1, device=device)
        rewards_buf = torch.zeros(chunks, n_envs, 1, device=device)
        raw_rewards_buf = torch.zeros(chunks, n_envs, 1, device=device)
        dones_buf = torch.zeros(chunks, n_envs, 1, dtype=torch.bool, device=device)
        timeouts_buf = torch.zeros(chunks, n_envs, 1, dtype=torch.bool, device=device)
        valid_mask_buf = torch.ones(chunks, n_envs, 1, dtype=torch.bool, device=device)
        live_frames_buf = torch.zeros(chunks, n_envs, 1, device=device)
        actions_buf = torch.zeros(chunks, n_envs, horizon, self.num_act, device=device)
        latents_buf = torch.zeros(chunks, n_envs, int(self.cfg.flow_steps) + 1, self.chunk_dim, device=device)
        old_log_probs_buf = torch.zeros(chunks, n_envs, int(self.cfg.flow_steps), device=device)
        reward_frame_buf = torch.zeros(chunks, n_envs, horizon, device=device)
        alive_frame_buf = torch.zeros(chunks, n_envs, horizon, dtype=torch.bool, device=device)

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

        total_steps = chunks * horizon
        ever_done = torch.zeros(n_envs, dtype=torch.bool, device=device)
        first_done_step = torch.full((n_envs,), total_steps, dtype=torch.long, device=device)
        first_done_phase = torch.full((n_envs,), -1, dtype=torch.long, device=device)
        first_done_anchor_pos = torch.zeros(n_envs, dtype=torch.bool, device=device)
        first_done_anchor_ori = torch.zeros(n_envs, dtype=torch.bool, device=device)
        first_done_ee_body = torch.zeros(n_envs, dtype=torch.bool, device=device)
        first_done_timeout = torch.zeros(n_envs, dtype=torch.bool, device=device)
        metric_cross_chunk_delta_sum = torch.zeros((), device=device, dtype=obs.dtype)
        metric_cross_chunk_delta_count = torch.zeros((), device=device, dtype=obs.dtype)
        prev_chunk_last_action = None

        with torch.no_grad():
            for chunk_idx in range(chunks):
                actor_obs_n = self._norm_actor(obs)
                critic_obs_n = self._norm_critic(critic_obs)
                value = self.critic.evaluate(critic_obs_n).detach()

                noise = torch.randn(n_envs, self.chunk_dim, device=device, dtype=obs.dtype)
                sde_noise = torch.randn(n_envs, int(self.cfg.flow_steps), self.chunk_dim, device=device, dtype=obs.dtype)
                sample = self._sample_policy_with_logprobs(actor_obs_n, noise, sde_noise=sde_noise)
                train_step_indices = sample["train_step_indices"]
                action_chunk = sample["actions"]
                action_abs_max = max(action_abs_max, float(action_chunk.abs().max().item()))

                chunk_reward = torch.zeros(n_envs, device=device, dtype=obs.dtype)
                chunk_raw_reward = torch.zeros(n_envs, device=device, dtype=obs.dtype)
                chunk_done = torch.zeros(n_envs, dtype=torch.bool, device=device)
                chunk_timeout = torch.zeros(n_envs, dtype=torch.bool, device=device)
                chunk_live_frames = torch.zeros(n_envs, device=device, dtype=obs.dtype)
                # Chunk-internal alive mask (MixGRPO-style): a done env stays dead
                # for the remainder of THIS chunk only, then is reset at chunk end
                # so the next chunk starts a fresh episode. valid_mask stays all-True
                # across chunks (NO first-life truncation, NO cross-chunk invalidation).
                alive_in_chunk = torch.ones(n_envs, dtype=torch.bool, device=device)

                for frame_idx in range(horizon):
                    alive_before_frame = alive_in_chunk.clone()
                    action_t = action_chunk[:, frame_idx, :]
                    if bool((~alive_before_frame).any()):
                        action_t = torch.where(
                            alive_before_frame.unsqueeze(-1), action_t, torch.zeros_like(action_t)
                        )
                    # auto_reset=False: done envs are NOT reset mid-chunk. Their
                    # remaining frames are masked out (zero action, no reward
                    # contribution) and they are reset once at chunk end.
                    next_obs, reward, done, info = env.step(action_t, auto_reset=False)
                    next_critic_obs = env.get_critic_observation()
                    if chunk_idx == 0 and frame_idx == 0:
                        first_chunk_infos.append(info)

                    active_f = alive_before_frame.to(dtype=chunk_reward.dtype)
                    discount = gamma ** frame_idx
                    reward_live = reward.to(dtype=chunk_reward.dtype) * active_f
                    chunk_raw_reward = chunk_raw_reward + discount * reward_live
                    chunk_reward = chunk_reward + discount * reward_live
                    chunk_live_frames = chunk_live_frames + active_f
                    reward_frame_buf[chunk_idx, :, frame_idx] = reward.detach().to(dtype=reward_frame_buf.dtype)
                    alive_frame_buf[chunk_idx, :, frame_idx] = alive_before_frame.detach()

                    timeouts = info["done_terms"]["time_out"].bool()
                    done_b = done.bool()
                    new_done = alive_before_frame & done_b
                    new_timeout = new_done & timeouts
                    if bool(new_timeout.any()):
                        terminal_critic_obs = info.get("final_critic_observation")
                        if terminal_critic_obs is None:
                            terminal_critic_obs = next_critic_obs
                        terminal_value = self.critic.evaluate(
                            self._norm_critic(terminal_critic_obs, update=False)
                        ).detach().squeeze(1)
                        chunk_reward = chunk_reward + (gamma ** (frame_idx + 1)) * terminal_value * new_timeout.float()

                    chunk_done = chunk_done | new_done
                    chunk_timeout = chunk_timeout | new_timeout
                    if bool(new_done.any()):
                        self._record_first_done(
                            new_done=new_done,
                            timeouts=timeouts,
                            info=info,
                            global_step=chunk_idx * horizon + frame_idx,
                            ever_done=ever_done,
                            first_done_step=first_done_step,
                            first_done_phase=first_done_phase,
                            first_done_anchor_pos=first_done_anchor_pos,
                            first_done_anchor_ori=first_done_anchor_ori,
                            first_done_ee_body=first_done_ee_body,
                            first_done_timeout=first_done_timeout,
                        )
                    self._record_train_episode_stats(
                        reward_live.detach(),
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

                # Chunk-end reset: restart done envs so the next chunk begins a
                # fresh episode. Alive envs keep their running state -> continuous
                # PPO-style flow across chunks. This does NOT invalidate the
                # current chunk's sample (valid_mask stays True for every chunk).
                if bool(chunk_done.any()):
                    reset_ids = chunk_done.nonzero(as_tuple=False).squeeze(-1)
                    reset_phases = self.env.sample_phase_indices(
                        reset_ids.numel(), horizon=max(1, int(self.cfg.horizon))
                    )
                    reset_obs = self.env.reset_envs(reset_ids, phase_indices=reset_phases)
                    obs[reset_ids] = reset_obs
                    critic_obs = self.env.get_critic_observation()

                chunk_first_action = action_chunk[:, 0, :].detach()
                chunk_last_action = action_chunk[:, horizon - 1, :].detach()
                if prev_chunk_last_action is not None:
                    delta = (chunk_first_action - prev_chunk_last_action).abs().mean(dim=-1)
                    metric_cross_chunk_delta_sum = metric_cross_chunk_delta_sum + delta.sum()
                    metric_cross_chunk_delta_count = metric_cross_chunk_delta_count + torch.tensor(
                        float(n_envs), device=device, dtype=obs.dtype
                    )
                prev_chunk_last_action = chunk_last_action

                actor_obs_buf[chunk_idx] = actor_obs_n
                critic_obs_buf[chunk_idx] = critic_obs_n
                values_buf[chunk_idx] = value
                rewards_buf[chunk_idx, :, 0] = chunk_reward.detach()
                raw_rewards_buf[chunk_idx, :, 0] = chunk_raw_reward.detach()
                dones_buf[chunk_idx, :, 0] = chunk_done.detach()
                timeouts_buf[chunk_idx, :, 0] = chunk_timeout.detach()
                valid_mask_buf[chunk_idx, :, 0] = True
                live_frames_buf[chunk_idx, :, 0] = chunk_live_frames.detach()
                actions_buf[chunk_idx] = action_chunk.detach()
                latents_buf[chunk_idx] = sample["all_latents"].detach()
                old_log_probs_buf[chunk_idx] = sample["log_probs"].detach()

            last_critic_obs = self._norm_critic(critic_obs, update=False)
            last_values = self.critic.evaluate(last_critic_obs).detach()
            # Bootstrap at rollout end: done chunks are blocked by dones_buf
            # (next_nonterminal=0 -> no bootstrap), alive envs bootstrap with
            # their running value. Chunk-end resets keep the stream continuous
            # across updates. valid_mask is all-True for every chunk.
            alive_at_end = torch.ones(n_envs, 1, dtype=torch.bool, device=device)
            returns, advantages = self._compute_chunk_gae(
                last_values,
                values_buf,
                dones_buf,
                rewards_buf,
                valid_mask=valid_mask_buf,
                alive_at_end=alive_at_end,
            )

        self._obs = obs
        self._critic_obs = critic_obs
        if train_step_indices is None:
            train_step_indices = self._train_step_indices(device)
        return {
            "actor_obs": actor_obs_buf,
            "critic_obs": critic_obs_buf,
            "actions": actions_buf,
            "latents": latents_buf,
            "old_log_probs": old_log_probs_buf,
            "train_step_indices": train_step_indices,
            "values": values_buf,
            "returns": returns,
            "advantages": advantages,
            "rewards": rewards_buf,
            "raw_rewards": raw_rewards_buf,
            "dones": dones_buf,
            "timeouts": timeouts_buf,
            "valid_mask": valid_mask_buf,
            "alive_at_end": alive_at_end,
            "live_frames": live_frames_buf,
            "reward_frame": reward_frame_buf,
            "alive_frame": alive_frame_buf,
            "done_terms_union": done_terms_union,
            "rollout_info_items": rollout_info_items,
            "first_chunk_infos": first_chunk_infos,
            "first_done_step": first_done_step,
            "first_done_phase": first_done_phase,
            "first_done_ee_body": first_done_ee_body,
            "first_done_anchor_pos": first_done_anchor_pos,
            "first_done_anchor_ori": first_done_anchor_ori,
            "first_done_timeout": first_done_timeout,
            "collection_start_phases": collection_start_phases,
            "metric_cross_chunk_delta_sum": metric_cross_chunk_delta_sum,
            "metric_cross_chunk_delta_count": metric_cross_chunk_delta_count,
            "action_abs_max": action_abs_max,
            "next_observation": obs,
        }

    def _compute_chunk_gae(self, last_values, values, dones, rewards, valid_mask=None, alive_at_end=None):
        horizon = max(1, int(self.cfg.horizon))
        chunk_gamma = float(self.cfg.discount_gamma) ** horizon
        chunk_lambda = float(self.cfg.gae_lambda) ** horizon
        if valid_mask is None:
            valid_mask = torch.ones_like(dones, dtype=torch.bool)
        if alive_at_end is None:
            alive_at_end = torch.ones_like(last_values, dtype=torch.bool)
        advantage = torch.zeros_like(values[0])
        returns = torch.zeros_like(values)
        for step in reversed(range(returns.shape[0])):
            if step == returns.shape[0] - 1:
                next_values = last_values
                next_valid = alive_at_end.float()
            else:
                next_values = values[step + 1]
                next_valid = valid_mask[step + 1].float()
            current_valid = valid_mask[step].float()
            next_nonterminal = (1.0 - dones[step].float()) * next_valid
            delta = rewards[step] + next_nonterminal * chunk_gamma * next_values - values[step]
            advantage = delta + next_nonterminal * chunk_gamma * chunk_lambda * advantage
            advantage = advantage * current_valid
            returns[step] = advantage + values[step]
        advantages = returns - values
        normalized = torch.zeros_like(advantages)
        valid_advantages = advantages[valid_mask]
        if valid_advantages.numel() > 0:
            normalized[valid_mask] = (
                (valid_advantages - valid_advantages.mean())
                / (valid_advantages.std() + 1.0e-8)
            )
        return returns, normalized

    def _policy_mini_batch_size(self, sample_count: int) -> int:
        num_mini_batches = max(1, int(self.cfg.num_mini_batches))
        return max(1, math.ceil(sample_count / num_mini_batches))

    def _policy_micro_batch_size(self, batch_size: int) -> int:
        if int(self.cfg.micro_batch_size) <= 0:
            return max(1, batch_size)
        return max(1, min(batch_size, int(self.cfg.micro_batch_size)))

    def _update_adaptive_learning_rates(self, observed_kl: float) -> None:
        desired_kl = float(self.cfg.desired_kl)
        if desired_kl <= 0.0:
            return
        # Normalize raw chunk KL by horizon before comparing to desired_kl,
        # so desired_kl is a per-control-step budget (h-independent).
        new_actor_lr, _ = adaptive_lr_from_kl(
            raw_kl=observed_kl,
            kl_units=self.kl_units,
            target_per_step=desired_kl,
            lr=self.learning_rate,
            min_lr=self.min_lr,
            max_lr=self.max_lr,
        )
        new_critic_lr, _ = adaptive_lr_from_kl(
            raw_kl=observed_kl,
            kl_units=self.kl_units,
            target_per_step=desired_kl,
            lr=self.critic_learning_rate,
            min_lr=self.min_lr,
            max_lr=self.max_lr,
        )
        self.learning_rate = new_actor_lr
        self.critic_learning_rate = new_critic_lr
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.learning_rate
        for group in self.critic_optimizer.param_groups:
            group["lr"] = self.critic_learning_rate

    def update(self, rollout: dict, collect_time: float) -> dict:
        import time as _time

        device = self.env.device
        chunks, n_envs = rollout["actions"].shape[:2]
        raw_batch_size = chunks * n_envs
        valid_flat = rollout["valid_mask"].reshape(raw_batch_size)
        actor_obs = rollout["actor_obs"].reshape(raw_batch_size, self.actor_obs_dim)[valid_flat]
        critic_obs = rollout["critic_obs"].reshape(raw_batch_size, self.critic_obs_dim)[valid_flat]
        latent_path = rollout["latents"].reshape(raw_batch_size, int(self.cfg.flow_steps) + 1, self.chunk_dim)[valid_flat]
        old_log_probs = rollout["old_log_probs"].reshape(raw_batch_size, int(self.cfg.flow_steps))[valid_flat]
        returns = rollout["returns"].reshape(raw_batch_size, 1)[valid_flat]
        old_values = rollout["values"].reshape(raw_batch_size, 1)[valid_flat]
        advantages = rollout["advantages"].reshape(raw_batch_size, 1)[valid_flat]
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
            probe_action_before = self._deterministic_actor_actions(probe_obs)
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
        }
        update_count = 0
        micro_batch_count = 0

        t1 = _time.perf_counter()
        for _ in range(int(self.cfg.policy_epochs)):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mini_batch_size):
                idx = perm[start:start + mini_batch_size]
                if idx.numel() == 0:
                    continue
                mb_size = int(idx.numel())
                micro_batch_size = self._policy_micro_batch_size(mb_size)
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)

                mb_totals = {k: 0.0 for k in ("policy_loss", "value_loss", "loss", "ratio", "clip_frac", "kl_loss", "logprob_delta_abs", "old_log_prob", "new_log_prob")}
                mb_ratio_min = float("inf")
                mb_ratio_max = 0.0
                for micro_start in range(0, mb_size, micro_batch_size):
                    micro_end = min(micro_start + micro_batch_size, mb_size)
                    sub = idx[micro_start:micro_end]
                    weight = float(sub.numel()) / float(mb_size)
                    new_log_probs = self._compute_transition_log_probs(actor_obs[sub], latent_path[sub], train_step_indices)
                    old_log_probs_sub = old_log_probs[sub]
                    log_ratio = new_log_probs - old_log_probs_sub
                    ratio = torch.exp(log_ratio)
                    adv = advantages[sub]
                    adv_steps = adv
                    unclipped = -adv_steps * ratio
                    clipped = -adv_steps * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                    policy_loss = torch.maximum(unclipped, clipped).mean()

                    value = self.critic.evaluate(critic_obs[sub])
                    if use_clipped_value_loss:
                        value_clipped = old_values[sub] + (value - old_values[sub]).clamp(-value_clip, value_clip)
                        value_losses = (value - returns[sub]).pow(2)
                        value_losses_clipped = (value_clipped - returns[sub]).pow(2)
                        value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = (returns[sub] - value).pow(2).mean()
                    loss = policy_loss + value_coef * value_loss
                    (loss * weight).backward()

                    with torch.no_grad():
                        mb_totals["policy_loss"] += float(policy_loss.item()) * weight
                        mb_totals["value_loss"] += float(value_loss.item()) * weight
                        mb_totals["loss"] += float(loss.item()) * weight
                        mb_totals["ratio"] += float(ratio.mean().item()) * weight
                        mb_totals["clip_frac"] += float((torch.abs(ratio - 1.0) > clip_range).float().mean().item()) * weight
                        mb_totals["kl_loss"] += float((0.5 * log_ratio.square()).mean().item()) * weight
                        mb_totals["logprob_delta_abs"] += float(log_ratio.abs().mean().item()) * weight
                        mb_totals["old_log_prob"] += float(old_log_probs_sub.mean().item()) * weight
                        mb_totals["new_log_prob"] += float(new_log_probs.mean().item()) * weight
                        mb_ratio_min = min(mb_ratio_min, float(ratio.min().item()))
                        mb_ratio_max = max(mb_ratio_max, float(ratio.max().item()))
                    micro_batch_count += 1

                self._update_adaptive_learning_rates(mb_totals["kl_loss"])
                grad_norm = nn.utils.clip_grad_norm_(self._policy.parameters(), float(self.cfg.max_grad_norm))
                grad_norm_critic = nn.utils.clip_grad_norm_(self.critic.parameters(), float(self.cfg.max_grad_norm))
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                for key in mb_totals:
                    totals[key] += mb_totals[key]
                totals["ratio_min"] = min(totals["ratio_min"], mb_ratio_min)
                totals["ratio_max"] = max(totals["ratio_max"], mb_ratio_max)
                totals["grad_norm"] += float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                totals["grad_norm_critic"] += float(
                    grad_norm_critic.item() if torch.is_tensor(grad_norm_critic) else grad_norm_critic
                )
                update_count += 1

        update_time = _time.perf_counter() - t1
        denom = max(update_count, 1)
        kl_raw = totals["kl_loss"] / denom
        kl_units = self.kl_units
        kl_per_step = kl_raw / kl_units
        desired_kl = float(self.cfg.desired_kl)
        with torch.no_grad():
            probe_action_after = self._deterministic_actor_actions(probe_obs)
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
            "sfpo/advantage_abs_mean": float(advantages.abs().mean().item()),
            "sfpo/return_mean": float(returns.mean().item()),
            "sfpo/value_mean": float(old_values.mean().item()),
            "policy/action_delta": float(action_delta.item()),
            "policy/param_rms_delta": float(param_rms_delta.item()),
        }
        return self._build_metrics(rollout, update_metrics, collect_time, update_time)

    def _build_metrics(self, rollout, update_metrics, collect_time, update_time) -> dict:
        rewards = rollout["rewards"]
        raw_rewards = rollout["raw_rewards"]
        dones = rollout["dones"]
        valid_mask = rollout["valid_mask"]
        live_frames = rollout["live_frames"]
        actions = rollout["actions"]
        alive = rollout["alive_frame"].to(dtype=actions.dtype)
        reward_frame = rollout["reward_frame"]
        valid_raw_rewards = raw_rewards[valid_mask]
        valid_rewards = rewards[valid_mask]
        valid_dones = dones[valid_mask]
        live_steps = live_frames.sum(dim=0).squeeze(-1)
        safe_live_steps = live_steps.clamp(min=1.0)
        reward_per_live_step = raw_rewards.sum(dim=0).squeeze(-1) / safe_live_steps
        official_scale_reward = reward_per_live_step * self.max_episode_steps
        first_done_step = rollout["first_done_step"]
        failed = first_done_step < self._training_rollout_horizon()
        timeout = rollout["first_done_timeout"]
        metric_cross_chunk_delta_sum = rollout.get("metric_cross_chunk_delta_sum")
        metric_cross_chunk_delta_count = rollout.get("metric_cross_chunk_delta_count")
        if metric_cross_chunk_delta_sum is not None and float(metric_cross_chunk_delta_count.item()) > 0.0:
            cross_chunk_delta = float((metric_cross_chunk_delta_sum / metric_cross_chunk_delta_count).item())
        else:
            cross_chunk_delta = float("nan")

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
            "rollout/reward_step_mean": float((reward_frame * alive).sum().item() / max(float(alive.sum().item()), 1.0)),
            "rollout/chunk_return_mean": float(valid_raw_rewards.mean().item()) if valid_raw_rewards.numel() > 0 else 0.0,
            "rollout/chunk_return_std": float(valid_raw_rewards.std(unbiased=False).item()) if valid_raw_rewards.numel() > 0 else 0.0,
            "rollout/chunk_objective_mean": float(valid_rewards.mean().item()) if valid_rewards.numel() > 0 else 0.0,
            "rollout/done_frac": float(valid_dones.float().mean().item()) if valid_dones.numel() > 0 else 0.0,
            "rollout/timeout_frac": float(rollout["timeouts"][valid_mask].float().mean().item()) if valid_dones.numel() > 0 else 0.0,
            "rollout/valid_frac": float(valid_mask.float().mean().item()),
            "rollout/live_steps_mean": float(live_steps.float().mean().item()),
            "rollout/live_steps_min": float(live_steps.float().min().item()),
            "rollout/live_steps_p50": float(torch.quantile(live_steps.float(), 0.50).item()),
            "rollout/live_steps_p95": float(torch.quantile(live_steps.float(), 0.95).item()),
            "rollout/live_steps_max": float(live_steps.float().max().item()),
            "rollout/reward_per_live_step": float(reward_per_live_step.mean().item()),
            "rollout/max_episode_return_projection": float(official_scale_reward.mean().item()),
            "rollout/success_frac": float((~failed).float().mean().item()),
            "rollout/failure_frac": float((failed & (~timeout)).float().mean().item()),
            "rollout/first_failure_step_mean": float(first_done_step[failed].float().mean().item()) if bool(failed.any()) else float("nan"),
            "rollout/first_failure_step_min": float(first_done_step[failed].min().item()) if bool(failed.any()) else float("nan"),
            "rollout/first_failure_step_max": float(first_done_step[failed].max().item()) if bool(failed.any()) else float("nan"),
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
        print(
            f"[UPDATE] {update_idx}/{max_updates} "
            f"reward_step={metrics['rollout/reward_step_mean']:.5f} "
            f"chunk={metrics['rollout/chunk_return_mean']:.5f} "
            f"done_frac={metrics['rollout/done_frac']:.5f} "
            f"mean_reward={metrics.get('train/mean_reward', float('nan')):.5f} "
            f"mean_len={metrics.get('train/mean_episode_length', float('nan')):.2f}",
            flush=True,
        )
        print(
            f"[SFPO] loss={metrics['sfpo/loss']:.5f} policy_loss={metrics['sfpo/policy_loss']:.5f} "
            f"value_loss={metrics['sfpo/value_loss']:.5f} ratio={metrics['sfpo/ratio']:.4f} "
            f"[{metrics['sfpo/ratio_min']:.3f},{metrics['sfpo/ratio_max']:.3f}] "
            f"clip={metrics['sfpo/clip_frac']:.4f} kl_raw={metrics['sfpo/kl_raw']:.6f} "
            f"kl/step={metrics['sfpo/kl_per_step']:.6f} "
            f"target_raw={metrics['sfpo/kl_target_raw']:.4f} "
            f"target/step={metrics['sfpo/kl_target_per_step']:.4f} "
            f"kl_units={metrics['sfpo/kl_units']:.0f} "
            f"grad={metrics['sfpo/grad_norm']:.4f} grad_c={metrics['sfpo/grad_norm_critic']:.4f} "
            f"lr={metrics['sfpo/lr']:.6f} critic_lr={metrics['sfpo/critic_lr']:.6f}",
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
            f"raw_samples={metrics.get('sfpo/raw_sample_count', float('nan')):.0f} "
            f"mb={metrics.get('sfpo/effective_mini_batch_size', float('nan')):.0f} "
            f"micro_mb={metrics.get('sfpo/micro_batch_size', float('nan')):.0f} "
            f"logp_delta_abs={metrics.get('sfpo/logprob_delta_abs', float('nan')):.5f} "
            f"old_logp={metrics.get('sfpo/old_log_prob', float('nan')):.5f} "
            f"new_logp={metrics.get('sfpo/new_log_prob', float('nan')):.5f} "
            f"adv_abs={metrics.get('sfpo/advantage_abs_mean', float('nan')):.5f} "
            f"sde_steps={metrics.get('sfpo/sde_train_steps', float('nan')):.0f}",
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

    def log_banner(self) -> None:
        cfg = self.cfg
        env = self.env
        print("[INFO] Starting SFPO training", flush=True)
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
        print(
            f"[INFO] stochastic_flow=True grouped_rollout=False critic_advantage=True "
            f"ppo_aligned_rollout=True auto_reset=False chunk_end_reset=True "
            f"chunk_internal_alive_mask=True first_life_trajectory=False "
            f"terminal_penalty_disabled=True entropy_bonus=False "
            f"gamma={cfg.discount_gamma} lambda={cfg.gae_lambda} "
            f"chunk_gamma={float(cfg.discount_gamma) ** int(cfg.horizon):.6f} "
            f"chunk_lambda={float(cfg.gae_lambda) ** int(cfg.horizon):.6f}",
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
            f"desired_kl={cfg.desired_kl} "
            f"value_loss_coef={cfg.value_loss_coef} use_clipped_value_loss={cfg.use_clipped_value_loss} "
            f"num_mini_batches={cfg.num_mini_batches} micro_batch={cfg.micro_batch_size} "
            f"policy_lr={cfg.policy_lr} critic_lr={cfg.value_lr}",
            flush=True,
        )
