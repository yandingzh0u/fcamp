from __future__ import annotations

from collections import deque

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

from algorithms.base import Algorithm
from algorithms.kl_scheduler import adaptive_lr_from_kl
from networks.mlp_actor_critic import Critic, EmpiricalNormalization, GaussianActor


class SFPO(Algorithm):
    """Chunk-level PPO.

    SFPO keeps PPO's Gaussian actor, scalar value critic, clipped surrogate,
    clipped value loss, entropy bonus, and adaptive KL learning-rate schedule.
    The only structural change is the action space: one policy sample is a
    fixed-horizon action chunk, executed open-loop for ``horizon`` env steps.
    """

    def build(self) -> None:
        cfg = self.cfg
        env = self.env
        if int(cfg.horizon) < 1:
            raise ValueError(f"horizon must be >= 1, got {cfg.horizon}")
        if int(cfg.num_steps_per_env) <= 0:
            raise ValueError(f"num_steps_per_env must be > 0, got {cfg.num_steps_per_env}")
        if int(cfg.num_steps_per_env) % int(cfg.horizon) != 0:
            raise ValueError(
                f"num_steps_per_env ({cfg.num_steps_per_env}) must be divisible by horizon ({cfg.horizon})."
            )

        self.num_act = env.action_dim
        self.actor_obs_dim = env.observation_dim
        self.critic_obs_dim = env.critic_observation_dim
        self.horizon_h = int(cfg.horizon)
        self.chunk_action_dim = self.horizon_h * self.num_act

        self.actor = GaussianActor(
            self.actor_obs_dim,
            self.chunk_action_dim,
            tuple(cfg.actor_hidden_dims),
            cfg.activation,
            cfg.init_noise_std,
        ).to(env.device)
        self.critic = Critic(self.critic_obs_dim, tuple(cfg.critic_hidden_dims), cfg.activation).to(env.device)

        self.empirical_normalization = bool(cfg.empirical_normalization)
        if self.empirical_normalization:
            self.actor_obs_normalizer: nn.Module = EmpiricalNormalization(self.actor_obs_dim, env.device)
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(self.critic_obs_dim, env.device)
        else:
            self.actor_obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        self.actor_learning_rate = float(cfg.actor_learning_rate)
        self.critic_learning_rate = float(cfg.critic_learning_rate)
        self.max_lr = 1e-2
        self.min_lr = 1e-5
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=self.actor_learning_rate, weight_decay=cfg.weight_decay
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=self.critic_learning_rate, weight_decay=cfg.critic_weight_decay
        )

        self.num_steps_per_env = int(cfg.num_steps_per_env)
        self.num_chunks_per_env = self.num_steps_per_env // self.horizon_h
        self.max_episode_steps = env.max_episode_steps
        self._policy_module = nn.ModuleDict({"actor": self.actor, "critic": self.critic})
        self._init_train_episode_stats()

    @property
    def policy(self) -> nn.Module:
        return self._policy_module

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.actor_optimizer

    @property
    def horizon(self) -> int:
        return int(getattr(self, "horizon_h", self.cfg.horizon))

    @property
    def kl_units(self) -> int:
        # Raw policy KL is the joint KL of an h-step action chunk. Normalize by
        # h so desired_kl remains a per-control-step budget like PPO.
        return int(getattr(self, "horizon_h", self.cfg.horizon))

    def extra_checkpoint_state(self) -> dict:
        return {
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
            "actor_obs_normalizer": self.actor_obs_normalizer.state_dict() if self.empirical_normalization else None,
            "critic_obs_normalizer": self.critic_obs_normalizer.state_dict() if self.empirical_normalization else None,
        }

    def load_extra_checkpoint_state(self, payload: dict, reset_optimizer: bool = False) -> None:
        if not payload:
            return
        if reset_optimizer:
            self.actor_learning_rate = float(self.cfg.actor_learning_rate)
            self.critic_learning_rate = float(self.cfg.critic_learning_rate)
            for group in self.actor_optimizer.param_groups:
                group["lr"] = self.actor_learning_rate
            for group in self.critic_optimizer.param_groups:
                group["lr"] = self.critic_learning_rate
        else:
            if "critic_optimizer" in payload:
                self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
            self.actor_learning_rate = float(payload.get("actor_learning_rate", self.actor_learning_rate))
            self.critic_learning_rate = float(payload.get("critic_learning_rate", self.critic_learning_rate))
            for group in self.actor_optimizer.param_groups:
                group["lr"] = self.actor_learning_rate
            for group in self.critic_optimizer.param_groups:
                group["lr"] = self.critic_learning_rate
        if self.empirical_normalization:
            if payload.get("actor_obs_normalizer") is not None:
                self.actor_obs_normalizer.load_state_dict(payload["actor_obs_normalizer"])
            if payload.get("critic_obs_normalizer") is not None:
                self.critic_obs_normalizer.load_state_dict(payload["critic_obs_normalizer"])

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

    def _record_episode_stats(self, rewards: torch.Tensor, dones: torch.Tensor, active: torch.Tensor) -> None:
        active_f = active.to(dtype=torch.float32)
        self._train_reward_sum += rewards.to(dtype=torch.float32) * active_f
        self._train_episode_length += active_f
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() == 0:
            return
        self._train_reward_buffer.extend(self._train_reward_sum.index_select(0, done_ids).detach().cpu().tolist())
        self._train_length_buffer.extend(self._train_episode_length.index_select(0, done_ids).detach().cpu().tolist())
        self._train_completed_episodes += int(done_ids.numel())
        self._train_reward_sum[done_ids] = 0.0
        self._train_episode_length[done_ids] = 0.0

    def initial_reset(self) -> torch.Tensor:
        obs = self.env.reset()
        if bool(self.cfg.init_at_random_ep_len) and self.max_episode_steps > 0:
            self.env.episode_steps = torch.randint_like(self.env.episode_steps, high=int(self.max_episode_steps))
        self._obs = obs
        self._critic_obs = self.env.get_critic_observation()
        return obs

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        return self._obs

    def _record_first_done(
        self,
        *,
        new_done: torch.Tensor,
        timeouts: torch.Tensor,
        info: dict,
        global_step: int,
        ever_done: torch.Tensor,
        first_done_step: torch.Tensor,
        first_done_phase: torch.Tensor,
        first_done_anchor_pos: torch.Tensor,
        first_done_anchor_ori: torch.Tensor,
        first_done_ee_body: torch.Tensor,
        first_done_timeout: torch.Tensor,
        first_done_motion_complete: torch.Tensor,
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
        phase = info.get("termination_phase_steps")
        if torch.is_tensor(phase):
            first_done_phase[ids] = phase.long().to(first_done_phase.device)[ids]
        ever_done[ids] = True

    def collect(self, current_obs: torch.Tensor) -> dict:
        env = self.env
        device = env.device
        n_envs = env.num_envs
        chunks = self.num_chunks_per_env
        h = self.horizon_h
        gamma = float(self.cfg.discount_gamma)

        actor_obs_buf = torch.zeros(chunks, n_envs, self.actor_obs_dim, device=device)
        critic_obs_buf = torch.zeros(chunks, n_envs, self.critic_obs_dim, device=device)
        actions_buf = torch.zeros(chunks, n_envs, h, self.num_act, device=device)
        values_buf = torch.zeros(chunks, n_envs, 1, device=device)
        logp_buf = torch.zeros(chunks, n_envs, 1, device=device)
        mu_buf = torch.zeros(chunks, n_envs, self.chunk_action_dim, device=device)
        sigma_buf = torch.zeros(chunks, n_envs, self.chunk_action_dim, device=device)
        reward_raw_buf = torch.zeros(chunks, n_envs, h, device=device)
        reward_masked_buf = torch.zeros(chunks, n_envs, h, device=device)
        alive_frame_buf = torch.zeros(chunks, n_envs, h, dtype=torch.bool, device=device)
        done_frame_buf = torch.zeros(chunks, n_envs, h, dtype=torch.bool, device=device)
        timeout_frame_buf = torch.zeros(chunks, n_envs, h, dtype=torch.bool, device=device)
        motion_complete_frame_buf = torch.zeros(chunks, n_envs, h, dtype=torch.bool, device=device)
        failure_frame_buf = torch.zeros(chunks, n_envs, h, dtype=torch.bool, device=device)
        next_critic_obs_buf = torch.zeros(chunks, n_envs, h, self.critic_obs_dim, device=device)

        obs = self._obs
        critic_obs = self._critic_obs
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
        action_abs_max = 0.0

        with torch.no_grad():
            for chunk_idx in range(chunks):
                actor_obs_n = self._norm_actor(obs)
                critic_obs_n = self._norm_critic(critic_obs)
                chunk_flat = self.actor.act(actor_obs_n)
                chunk_actions = chunk_flat.view(n_envs, h, self.num_act)
                value = self.critic.evaluate(critic_obs_n).detach()
                logp = self.actor.get_actions_log_prob(chunk_flat).detach().unsqueeze(1)
                action_abs_max = max(action_abs_max, float(chunk_actions.abs().max().item()))

                actor_obs_buf[chunk_idx] = actor_obs_n
                critic_obs_buf[chunk_idx] = critic_obs_n
                actions_buf[chunk_idx] = chunk_actions.detach()
                values_buf[chunk_idx] = value
                logp_buf[chunk_idx] = logp
                mu_buf[chunk_idx] = self.actor.action_mean.detach()
                sigma_buf[chunk_idx] = self.actor.action_std.detach()

                alive_in_chunk = torch.ones(n_envs, dtype=torch.bool, device=device)
                for frame_idx in range(h):
                    alive_before = alive_in_chunk.clone()
                    action_t = chunk_actions[:, frame_idx, :]
                    if bool((~alive_before).any()):
                        action_t = torch.where(alive_before.unsqueeze(-1), action_t, torch.zeros_like(action_t))
                    next_obs, reward, done, info = env.step(action_t, auto_reset=False)
                    next_critic_obs = env.get_critic_observation()
                    if chunk_idx == 0 and frame_idx == 0:
                        first_chunk_infos.append(info)

                    active_f = alive_before.to(dtype=reward.dtype)
                    reward_raw_buf[chunk_idx, :, frame_idx] = reward.detach()
                    reward_masked_buf[chunk_idx, :, frame_idx] = reward.detach() * active_f
                    alive_frame_buf[chunk_idx, :, frame_idx] = alive_before
                    next_critic_obs_buf[chunk_idx, :, frame_idx] = self._norm_critic(
                        next_critic_obs, update=False
                    ).detach()

                    dterms = info["done_terms"]
                    timeouts = dterms["time_out"].bool()
                    motion_complete = dterms.get("motion_complete")
                    motion_complete = (
                        motion_complete.bool() if torch.is_tensor(motion_complete) else torch.zeros_like(timeouts)
                    )
                    failure_terms = torch.zeros_like(timeouts)
                    for name in ("anchor_pos_bad", "anchor_ori_bad", "ee_body_bad"):
                        if name in dterms:
                            failure_terms = failure_terms | dterms[name].bool()
                    done_b = done.bool()
                    new_done = alive_before & done_b
                    new_timeout = new_done & timeouts
                    new_motion_complete = new_done & motion_complete
                    new_failure = new_done & failure_terms & (~timeouts) & (~motion_complete)

                    done_frame_buf[chunk_idx, :, frame_idx] = new_done
                    timeout_frame_buf[chunk_idx, :, frame_idx] = new_timeout
                    motion_complete_frame_buf[chunk_idx, :, frame_idx] = new_motion_complete
                    failure_frame_buf[chunk_idx, :, frame_idx] = new_failure
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
                    self._record_episode_stats(reward.detach(), new_done.detach(), alive_before.detach())
                    rollout_info_items.append((info, alive_before.detach()))
                    for key, val in dterms.items():
                        b = val.bool()
                        done_terms_union[key] = b.clone() if key not in done_terms_union else (done_terms_union[key] | b)

                    alive_in_chunk = alive_before & ~done_b
                    obs = next_obs
                    critic_obs = next_critic_obs

                chunk_done = done_frame_buf[chunk_idx].any(dim=-1)
                if bool(chunk_done.any()):
                    reset_ids = chunk_done.nonzero(as_tuple=False).squeeze(-1)
                    reset_phases = env.sample_phase_indices(reset_ids.numel(), horizon=max(1, h))
                    reset_obs = env.reset_envs(reset_ids, phase_indices=reset_phases)
                    obs[reset_ids] = reset_obs
                    critic_obs = env.get_critic_observation()

            gamma_pow = gamma ** torch.arange(h, device=device, dtype=reward_masked_buf.dtype)
            chunk_reward_return = (reward_masked_buf * gamma_pow.view(1, 1, h)).sum(dim=-1)
            frame_idx_grid = torch.arange(h, device=device).view(1, 1, h).expand(chunks, n_envs, h)
            masked_done_idx = torch.where(done_frame_buf, frame_idx_grid, torch.full_like(frame_idx_grid, h))
            death_frame = masked_done_idx.min(dim=-1).values
            chunk_done = death_frame < h
            chunk_timeout = timeout_frame_buf.any(dim=-1)
            bootstrap_frame = death_frame.clamp(max=h - 1)
            bootstrap_obs = next_critic_obs_buf.gather(
                2,
                bootstrap_frame.view(chunks, n_envs, 1, 1).expand(-1, -1, 1, self.critic_obs_dim),
            ).squeeze(2)
            bootstrap_values = self.critic.evaluate(
                bootstrap_obs.reshape(chunks * n_envs, self.critic_obs_dim)
            ).reshape(chunks, n_envs)
            bootstrap_discount = gamma ** (bootstrap_frame.to(dtype=bootstrap_values.dtype) + 1.0)
            terminal_without_bootstrap = chunk_done & (~chunk_timeout)
            chunk_bootstrap = torch.where(
                terminal_without_bootstrap,
                torch.zeros_like(bootstrap_values),
                bootstrap_discount * bootstrap_values,
            )
            chunk_one_step_target = chunk_reward_return + chunk_bootstrap

            chunk_values = values_buf.squeeze(-1)
            chunk_delta = chunk_one_step_target - chunk_values
            chunk_advantages = torch.zeros_like(chunk_delta)
            gae = torch.zeros(n_envs, device=device, dtype=chunk_delta.dtype)
            chunk_cont = (~chunk_done).to(dtype=chunk_delta.dtype)
            macro_gamma_lambda = (gamma * float(self.cfg.gae_lambda)) ** h
            for chunk_i in range(chunks - 1, -1, -1):
                gae = chunk_delta[chunk_i] + macro_gamma_lambda * chunk_cont[chunk_i] * gae
                chunk_advantages[chunk_i] = gae
            chunk_returns = chunk_values + chunk_advantages
            chunk_valid_mask = alive_frame_buf[..., 0].clone()
            advantages = self._normalize_chunk_advantages(chunk_advantages, chunk_valid_mask).unsqueeze(-1)
            raw_advantages = chunk_advantages.unsqueeze(-1)

        self._obs = obs
        self._critic_obs = critic_obs
        alive_f = alive_frame_buf.to(dtype=reward_raw_buf.dtype)
        chunk_env_raw_return = (reward_raw_buf * alive_f * gamma_pow.view(1, 1, h)).sum(dim=-1)
        failure_cost_return = torch.zeros_like(chunk_env_raw_return)
        chunk_live_frames = alive_f.sum(dim=-1)
        frame_bootstrap = torch.where(
            done_frame_buf & (~timeout_frame_buf),
            torch.zeros_like(next_critic_obs_buf[..., 0]),
            self.critic.evaluate(next_critic_obs_buf.reshape(chunks * n_envs * h, self.critic_obs_dim))
            .reshape(chunks, n_envs, h),
        )

        return {
            "actor_obs": actor_obs_buf,
            "critic_obs": critic_obs_buf,
            "actions": actions_buf,
            "values": values_buf,
            "logp": logp_buf,
            "mu": mu_buf,
            "sigma": sigma_buf,
            "returns": chunk_returns.unsqueeze(-1),
            "advantages": advantages,
            "raw_advantages": raw_advantages,
            "chunk_values": chunk_values,
            "chunk_v_targets": chunk_returns,
            "chunk_advantages": chunk_advantages,
            "chunk_one_step_target": chunk_one_step_target,
            "chunk_bootstrap": chunk_bootstrap,
            "chunk_cont": chunk_cont,
            "chunk_valid_mask": chunk_valid_mask,
            "reward_raw": reward_raw_buf,
            "reward_masked": reward_masked_buf,
            "chunk_raw_return": chunk_reward_return,
            "chunk_env_raw_return": chunk_env_raw_return,
            "failure_cost_return": failure_cost_return,
            "chunk_return_realized": chunk_one_step_target,
            "chunk_live_frames": chunk_live_frames,
            "alive_frame": alive_frame_buf,
            "valid_prefix_mask": alive_frame_buf.clone(),
            "done_frame": done_frame_buf,
            "timeout_frame": timeout_frame_buf,
            "motion_complete_frame": motion_complete_frame_buf,
            "failure_frame": failure_frame_buf,
            "death_frame": death_frame,
            "frame_next_values": frame_bootstrap,
            "frame_bootstrap": frame_bootstrap,
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
            "action_abs_max": action_abs_max,
            "next_observation": obs,
        }

    def _normalize_chunk_advantages(self, adv: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = adv[mask]
        if valid.numel() == 0:
            return adv * mask
        return ((adv - valid.mean()) / (valid.std(unbiased=False) + 1.0e-8)) * mask

    def update(self, rollout: dict, collect_time: float) -> dict:
        import time as _time

        device = self.env.device
        chunks, n_envs = rollout["actions"].shape[:2]
        batch_size = chunks * n_envs
        flat = lambda x: x.reshape(batch_size, -1)
        actor_obs = flat(rollout["actor_obs"])
        critic_obs = flat(rollout["critic_obs"])
        actions = rollout["actions"].reshape(batch_size, self.chunk_action_dim)
        old_logp = flat(rollout["logp"])
        old_values = flat(rollout["values"])
        returns = flat(rollout["returns"])
        advantages = flat(rollout["advantages"])
        old_mu = flat(rollout["mu"])
        old_sigma = flat(rollout["sigma"])

        num_mini_batches = int(self.cfg.num_mini_batches)
        mini_batch_size = max(1, batch_size // num_mini_batches)
        epochs = int(self.cfg.num_learning_epochs)
        clip = float(self.cfg.clip_range)
        value_clip = float(self.cfg.value_clip_range)

        totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "ratio": 0.0,
            "ratio_min": float("inf"),
            "ratio_max": 0.0,
            "clip_frac": 0.0,
            "logprob_delta_abs": 0.0,
            "old_log_prob": 0.0,
            "new_log_prob": 0.0,
        }
        num_updates = 0
        grad_norm_accum = 0.0
        grad_norm_critic_accum = 0.0
        per_frame_kl_sum = torch.zeros(self.horizon_h, device=device)
        per_frame_kl_count = torch.zeros(self.horizon_h, device=device)

        probe_count = min(128, batch_size)
        with torch.no_grad():
            probe_obs = actor_obs[:probe_count]
            probe_before = self._chunk_mean(probe_obs)
            params_before = [p.detach().clone() for p in self.actor.parameters()]

        t1 = _time.perf_counter()
        for _epoch in range(epochs):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mini_batch_size):
                idx = perm[start:start + mini_batch_size]
                if idx.numel() == 0:
                    continue
                mb_actor_obs = actor_obs[idx]
                mb_critic_obs = critic_obs[idx]
                mb_actions = actions[idx]
                mb_old_logp = old_logp[idx]
                mb_old_values = old_values[idx]
                mb_returns = returns[idx]
                mb_adv = advantages[idx]
                mb_old_mu = old_mu[idx]
                mb_old_sigma = old_sigma[idx]

                self.actor.update_distribution(mb_actor_obs)
                value_batch = self.critic.evaluate(mb_critic_obs)
                logp_batch = self.actor.get_actions_log_prob(mb_actions)
                mu_batch = self.actor.action_mean
                sigma_batch = self.actor.action_std
                entropy_batch = self.actor.entropy / float(self.horizon_h)

                if self.cfg.desired_kl is not None and self.cfg.desired_kl > 0.0:
                    with torch.no_grad():
                        kl_per_dim = kl_divergence(Normal(mb_old_mu, mb_old_sigma), Normal(mu_batch, sigma_batch))
                        kl = kl_per_dim.sum(-1)
                        kl_mean = kl.mean()
                        frame_kl = kl_per_dim.view(-1, self.horizon_h, self.num_act).sum(-1)
                    self._update_lr(kl_mean)
                else:
                    kl_mean = torch.zeros((), device=device)
                    frame_kl = torch.zeros(idx.numel(), self.horizon_h, device=device)

                ratio = torch.exp(logp_batch - mb_old_logp.squeeze(-1))
                surrogate = -mb_adv.squeeze(-1) * ratio
                surrogate_clipped = -mb_adv.squeeze(-1) * torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                value_clipped = mb_old_values + (value_batch - mb_old_values).clamp(-value_clip, value_clip)
                value_losses = (value_batch - mb_returns).pow(2)
                value_losses_clipped = (value_clipped - mb_returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()

                entropy_loss = entropy_batch.mean()
                actor_loss = surrogate_loss - float(self.cfg.entropy_coef) * entropy_loss
                critic_loss = float(self.cfg.value_loss_coef) * value_loss
                loss = actor_loss + critic_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                grad_norm_critic = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                with torch.no_grad():
                    clip_frac = (torch.abs(ratio - 1.0) > clip).float().mean()
                    logp_delta = logp_batch - mb_old_logp.squeeze(-1)
                    totals["loss"] += float(loss.item())
                    totals["policy_loss"] += float(surrogate_loss.item())
                    totals["value_loss"] += float(value_loss.item())
                    totals["entropy"] += float(entropy_loss.item())
                    totals["kl"] += float(kl_mean.item())
                    totals["ratio"] += float(ratio.mean().item())
                    totals["ratio_min"] = min(totals["ratio_min"], float(ratio.min().item()))
                    totals["ratio_max"] = max(totals["ratio_max"], float(ratio.max().item()))
                    totals["clip_frac"] += float(clip_frac.item())
                    totals["logprob_delta_abs"] += float(logp_delta.abs().mean().item())
                    totals["old_log_prob"] += float(mb_old_logp.mean().item())
                    totals["new_log_prob"] += float(logp_batch.mean().item())
                    per_frame_kl_sum += frame_kl.sum(dim=0)
                    per_frame_kl_count += torch.full((self.horizon_h,), float(frame_kl.shape[0]), device=device)
                    grad_norm_accum += float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                    grad_norm_critic_accum += float(
                        grad_norm_critic.item() if torch.is_tensor(grad_norm_critic) else grad_norm_critic
                    )
                    num_updates += 1

        update_time = _time.perf_counter() - t1
        denom = max(num_updates, 1)
        for key in totals:
            if key not in {"ratio_min", "ratio_max"}:
                totals[key] /= denom
        grad_norm_accum /= denom
        grad_norm_critic_accum /= denom

        with torch.no_grad():
            probe_after = self._chunk_mean(probe_obs)
            action_delta = float(torch.mean(torch.abs(probe_after - probe_before)).item())
            param_delta_sq = torch.zeros((), device=device)
            param_count = 0
            for param, before in zip(self.actor.parameters(), params_before, strict=True):
                delta = param.detach() - before
                param_delta_sq = param_delta_sq + torch.sum(delta * delta)
                param_count += delta.numel()
            param_rms_delta = float(torch.sqrt(param_delta_sq / max(param_count, 1)).item())

        kl_raw = totals["kl"]
        kl_units = self.kl_units
        kl_per_step = kl_raw / max(1, kl_units)
        update_metrics = {
            "sfpo/loss": totals["loss"],
            "sfpo/policy_loss": totals["policy_loss"],
            "sfpo/value_loss": totals["value_loss"],
            "sfpo/entropy": totals["entropy"],
            "sfpo/ratio": totals["ratio"],
            "sfpo/ratio_min": totals["ratio_min"] if totals["ratio_min"] != float("inf") else 0.0,
            "sfpo/ratio_max": totals["ratio_max"],
            "sfpo/clip_frac": totals["clip_frac"],
            "sfpo/kl_raw": kl_raw,
            "sfpo/kl_per_step": kl_per_step,
            "sfpo/kl_target_per_step": float(self.cfg.desired_kl),
            "sfpo/kl_target_raw": float(self.cfg.desired_kl) * kl_units,
            "sfpo/kl_units": float(kl_units),
            "sfpo/logprob_delta_abs": totals["logprob_delta_abs"],
            "sfpo/old_log_prob": totals["old_log_prob"],
            "sfpo/new_log_prob": totals["new_log_prob"],
            "sfpo/grad_norm": grad_norm_accum,
            "sfpo/grad_norm_critic": grad_norm_critic_accum,
            "sfpo/actor_lr": self.actor_learning_rate,
            "sfpo/critic_lr": self.critic_learning_rate,
            "sfpo/action_std_mean": float(self.actor.std.detach().mean().item()),
            "sfpo/sample_count": float(batch_size),
            "sfpo/effective_mini_batch_size": float(mini_batch_size),
            "policy/action_delta": action_delta,
            "policy/param_rms_delta": param_rms_delta,
        }
        frame_kl_mean = per_frame_kl_sum / per_frame_kl_count.clamp(min=1.0)
        for frame_idx in range(self.horizon_h):
            update_metrics[f"sfpo/kl_frame_{frame_idx}"] = float(frame_kl_mean[frame_idx].item())
        return self._build_metrics(rollout, update_metrics, collect_time, update_time)

    def _update_lr(self, kl_mean: torch.Tensor) -> None:
        desired_kl = float(self.cfg.desired_kl)
        if desired_kl <= 0.0:
            return
        new_actor_lr, _ = adaptive_lr_from_kl(
            raw_kl=kl_mean,
            kl_units=self.kl_units,
            target_per_step=desired_kl,
            lr=self.actor_learning_rate,
            min_lr=self.min_lr,
            max_lr=self.max_lr,
        )
        new_critic_lr, _ = adaptive_lr_from_kl(
            raw_kl=kl_mean,
            kl_units=self.kl_units,
            target_per_step=desired_kl,
            lr=self.critic_learning_rate,
            min_lr=self.min_lr,
            max_lr=self.max_lr,
        )
        self.actor_learning_rate = new_actor_lr
        self.critic_learning_rate = new_critic_lr
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.actor_learning_rate
        for group in self.critic_optimizer.param_groups:
            group["lr"] = self.critic_learning_rate

    def _chunk_mean(self, actor_obs: torch.Tensor) -> torch.Tensor:
        return self.actor.act_inference(actor_obs).view(actor_obs.shape[0], self.horizon_h, self.num_act)

    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        actor_obs = self._norm_actor(obs, update=False)
        return self._chunk_mean(actor_obs)

    def _build_metrics(self, rollout: dict, update_metrics: dict, collect_time: float, update_time: float) -> dict:
        actions = rollout["actions"]
        alive = rollout["alive_frame"].to(dtype=actions.dtype)
        reward_raw = rollout["reward_raw"]
        done_frame = rollout["done_frame"]
        failure_frame = rollout["failure_frame"]
        timeout_frame = rollout["timeout_frame"]
        live_steps = rollout["chunk_live_frames"]
        chunk_raw = rollout["chunk_raw_return"]
        chunk_env_raw = rollout["chunk_env_raw_return"]
        chunk_realized = rollout["chunk_return_realized"]
        chunk_values = rollout["chunk_values"]
        chunk_targets = rollout["chunk_v_targets"]
        chunk_advantages = rollout["chunk_advantages"]
        chunk_bootstrap = rollout["chunk_bootstrap"]
        chunk_cont = rollout["chunk_cont"]
        chunk_valid_mask = rollout["chunk_valid_mask"]
        first_done_step = rollout["first_done_step"]
        first_done_timeout = rollout["first_done_timeout"]
        first_done_motion_complete = rollout["first_done_motion_complete"]
        failed = (
            (first_done_step < self.num_steps_per_env)
            & (~first_done_timeout)
            & (~first_done_motion_complete)
        )

        alive_sum = alive.sum().clamp(min=1.0)
        sampled_abs = actions.abs()
        applied_abs = sampled_abs[rollout["alive_frame"]]
        if applied_abs.numel() > 0:
            action_abs_mean = float(applied_abs.mean().item())
            action_abs_p95 = float(torch.quantile(applied_abs.flatten(), 0.95).item())
            action_abs_p99 = float(torch.quantile(applied_abs.flatten(), 0.99).item())
            action_abs_max = float(applied_abs.max().item())
        else:
            action_abs_mean = action_abs_p95 = action_abs_p99 = action_abs_max = 0.0
        if self.horizon_h > 1:
            action_delta = (actions[..., 1:, :] - actions[..., :-1, :]).abs().mean(dim=-1)
            in_chunk_delta = float((action_delta * alive[..., 1:]).sum().item() / alive[..., 1:].sum().clamp(min=1.0).item())
        else:
            in_chunk_delta = float("nan")

        reward_per_live_step = chunk_env_raw / live_steps.clamp(min=1.0)
        metrics = {
            **update_metrics,
            "algo/name": "sfpo",
            "rollout/reward_step_mean": float((reward_raw * alive).sum().item() / alive_sum.item()),
            "rollout/chunk_return_mean": float(chunk_realized.mean().item()),
            "rollout/chunk_return_std": float(chunk_realized.std(unbiased=False).item()),
            "rollout/chunk_raw_return_mean": float(chunk_raw.mean().item()),
            "rollout/chunk_env_raw_return_mean": float(chunk_env_raw.mean().item()),
            "rollout/failure_cost_return_mean": 0.0,
            "rollout/done_frac": float(done_frame.any(dim=-1).float().mean().item()),
            "rollout/failure_frac": float(failure_frame.any(dim=-1).float().mean().item()),
            "rollout/timeout_frac": float(timeout_frame.any(dim=-1).float().mean().item()),
            "rollout/live_steps_mean": float(live_steps.mean().item()),
            "rollout/live_steps_min": float(live_steps.min().item()),
            "rollout/live_steps_p50": float(torch.quantile(live_steps, 0.50).item()),
            "rollout/live_steps_p95": float(torch.quantile(live_steps, 0.95).item()),
            "rollout/live_steps_max": float(live_steps.max().item()),
            "rollout/reward_per_live_step": float(reward_per_live_step.mean().item()),
            "rollout/max_episode_return_projection": float((reward_per_live_step * self.max_episode_steps).mean().item()),
            "rollout/success_frac": float((~failed).float().mean().item()),
            "rollout/failure_frac_first_done": float(failed.float().mean().item()),
            "rollout/first_failure_step_mean": float(first_done_step[failed].float().mean().item()) if bool(failed.any()) else float("nan"),
            "rollout/first_failure_step_min": float(first_done_step[failed].min().item()) if bool(failed.any()) else float("nan"),
            "rollout/first_failure_step_max": float(first_done_step[failed].max().item()) if bool(failed.any()) else float("nan"),
            "rollout/death_frame_mean": float(rollout["death_frame"].float().mean().item()),
            "rollout/valid_prefix_frac": float(rollout["valid_prefix_mask"].float().mean().item()),
            "phase/start_mean": float(rollout["collection_start_phases"].float().mean().item()),
            "phase/start_min": float(rollout["collection_start_phases"].min().item()),
            "phase/start_max": float(rollout["collection_start_phases"].max().item()),
            "critic/chunk_v_mean": float(chunk_values[chunk_valid_mask].mean().item()) if bool(chunk_valid_mask.any()) else float("nan"),
            "critic/chunk_v_target_mean": float(chunk_targets[chunk_valid_mask].mean().item()) if bool(chunk_valid_mask.any()) else float("nan"),
            "critic/chunk_one_step_target_mean": float(rollout["chunk_one_step_target"][chunk_valid_mask].mean().item()) if bool(chunk_valid_mask.any()) else float("nan"),
            "critic/chunk_bootstrap_mean": float(chunk_bootstrap[chunk_valid_mask].mean().item()) if bool(chunk_valid_mask.any()) else float("nan"),
            "critic/chunk_adv_mean": float(chunk_advantages[chunk_valid_mask].mean().item()) if bool(chunk_valid_mask.any()) else float("nan"),
            "critic/chunk_adv_std": float(chunk_advantages[chunk_valid_mask].std(unbiased=False).item()) if bool(chunk_valid_mask.any()) else float("nan"),
            "critic/chunk_cont_frac": float(chunk_cont[chunk_valid_mask].mean().item()) if bool(chunk_valid_mask.any()) else 0.0,
            "critic/chunk_valid_frac": float(chunk_valid_mask.float().mean().item()),
            "timing/collect_s": collect_time,
            "timing/update_s": update_time,
            "act/abs_mean": action_abs_mean,
            "act/abs_p95": action_abs_p95,
            "act/abs_p99": action_abs_p99,
            "act/abs_max": action_abs_max,
            "act/abs_max_all": float(rollout.get("action_abs_max", 0.0)),
            "act/in_chunk_delta_abs": in_chunk_delta,
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
        for info in rollout["first_chunk_infos"]:
            for key, value in info["reward_terms"].items():
                metrics[f"reward/{key}_mean"] = float(value.mean().item())

        reward_sums: dict[str, float] = {}
        done_sums: dict[str, float] = {}
        weight_sum = 0.0
        for info, valid in rollout["rollout_info_items"]:
            valid_f = valid.float()
            weight = float(valid_f.sum().item())
            if weight <= 0.0:
                continue
            weight_sum += weight
            for key, value in info["reward_terms"].items():
                reward_sums[key] = reward_sums.get(key, 0.0) + float((value * valid_f).sum().item())
            for key, value in info["done_terms"].items():
                done_sums[key] = done_sums.get(key, 0.0) + float((value.float() * valid_f).sum().item())
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
        frame_kl = " ".join(
            f"f{k}={metrics.get(f'sfpo/kl_frame_{k}', float('nan')):.6f}" for k in range(self.horizon_h)
        )
        print(
            f"[UPDATE] {update_idx}/{max_updates} "
            f"reward_step={metrics['rollout/reward_step_mean']:.5f} "
            f"chunk_return={metrics['rollout/chunk_return_mean']:.5f} "
            f"done_frac={metrics['rollout/done_frac']:.5f} "
            f"mean_reward={metrics.get('train/mean_reward', float('nan')):.5f} "
            f"mean_len={metrics.get('train/mean_episode_length', float('nan')):.2f}",
            flush=True,
        )
        print(
            f"[SFPO] loss={metrics['sfpo/loss']:.5f} "
            f"policy={metrics['sfpo/policy_loss']:.5f} "
            f"value={metrics['sfpo/value_loss']:.5f} "
            f"entropy={metrics['sfpo/entropy']:.5f} "
            f"ratio={metrics['sfpo/ratio']:.4f} "
            f"[{metrics['sfpo/ratio_min']:.3f},{metrics['sfpo/ratio_max']:.3f}] "
            f"clip={metrics['sfpo/clip_frac']:.4f} "
            f"kl_raw={metrics['sfpo/kl_raw']:.6f} "
            f"kl/step={metrics['sfpo/kl_per_step']:.6f} "
            f"target_raw={metrics['sfpo/kl_target_raw']:.4f} "
            f"target/step={metrics['sfpo/kl_target_per_step']:.4f} "
            f"kl_units={metrics['sfpo/kl_units']:.0f} "
            f"actor_lr={metrics['sfpo/actor_lr']:.6f} critic_lr={metrics['sfpo/critic_lr']:.6f} "
            f"grad={metrics['sfpo/grad_norm']:.4f} grad_c={metrics['sfpo/grad_norm_critic']:.4f} "
            f"action_std={metrics['sfpo/action_std_mean']:.4f}",
            flush=True,
        )
        print(
            f"[CHUNK] h={self.horizon_h} samples={metrics.get('sfpo/sample_count', float('nan')):.0f} "
            f"mb={metrics.get('sfpo/effective_mini_batch_size', float('nan')):.0f} "
            f"V={metrics['critic/chunk_v_mean']:.4f} "
            f"V_tgt={metrics['critic/chunk_v_target_mean']:.4f} "
            f"one_step={metrics['critic/chunk_one_step_target_mean']:.4f} "
            f"boot={metrics['critic/chunk_bootstrap_mean']:.4f} "
            f"adv={metrics['critic/chunk_adv_mean']:.4f}/{metrics['critic/chunk_adv_std']:.4f} "
            f"cont={metrics['critic/chunk_cont_frac']:.4f} "
            f"live={metrics.get('rollout/live_steps_mean', float('nan')):.2f} "
            f"{frame_kl}",
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
            f"in_chunk_delta={metrics.get('act/in_chunk_delta_abs', float('nan')):.4f} "
            f"legs={metrics.get('act/legs_abs', float('nan')):.4f} "
            f"waist={metrics.get('act/waist_abs', float('nan')):.4f} "
            f"arms={metrics.get('act/arms_abs', float('nan')):.4f}",
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

    def log_banner(self) -> None:
        cfg = self.cfg
        env = self.env
        print("[INFO] Starting SFPO training (chunk-level PPO)", flush=True)
        print(
            f"[INFO] task={env.task.name} terrain={env.task.terrain} motion_file={env.task.motion_file}",
            flush=True,
        )
        print(
            f"[INFO] algo=sfpo actor_obs_dim={self.actor_obs_dim} critic_obs_dim={self.critic_obs_dim} "
            f"action_dim={self.num_act} horizon={self.horizon_h} chunk_action_dim={self.chunk_action_dim} "
            f"num_envs={env.num_envs} num_steps_per_env={self.num_steps_per_env} "
            f"chunks_per_env={self.num_chunks_per_env}",
            flush=True,
        )
        print(
            f"[INFO] ppo_objective=joint_chunk_ratio gamma={cfg.discount_gamma} "
            f"gae_lambda={cfg.gae_lambda} macro_gamma_lambda={(float(cfg.discount_gamma) * float(cfg.gae_lambda)) ** self.horizon_h:.6f} "
            f"clip={cfg.clip_range} value_clip={cfg.value_clip_range} entropy_coef={cfg.entropy_coef} "
            f"value_loss_coef={cfg.value_loss_coef} desired_kl_per_step={cfg.desired_kl} "
            f"raw_kl_target={float(cfg.desired_kl) * self.kl_units:.6f} kl_units={self.kl_units}",
            flush=True,
        )
        print(
            f"[INFO] num_learning_epochs={cfg.num_learning_epochs} num_mini_batches={cfg.num_mini_batches} "
            f"actor_lr={cfg.actor_learning_rate} critic_lr={cfg.critic_learning_rate} "
            f"init_noise_std={cfg.init_noise_std} empirical_normalization={cfg.empirical_normalization} "
            f"init_at_random_ep_len={cfg.init_at_random_ep_len} "
            f"actor_hidden_dims={list(cfg.actor_hidden_dims)} critic_hidden_dims={list(cfg.critic_hidden_dims)} "
            f"activation={cfg.activation}",
            flush=True,
        )
