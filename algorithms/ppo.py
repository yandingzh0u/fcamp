"""PPO algorithm plugin, ported from holosoma's PPO numerical core.

Standard RSL-RL style actor-critic PPO: Gaussian MLP actor + MLP value critic, GAE returns /
advantages, clipped surrogate + clipped value loss, KL-adaptive learning rate, optional
empirical observation normalization, timeout value bootstrap.

Runs on the unified core trainer loop and the crawl env (H=1 single-step actions). Logs in the
project's [UPDATE]/[PPO]/[DONE]/... format (not holosoma's loguru/tensorboard), so this is a
faithful re-implementation of the algorithm, not a bit-exact clone of holosoma's process.
"""
from __future__ import annotations

from collections import deque

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

from algorithms.base import Algorithm
from core.logging import log_shared_tracking, log_shared_update_diagnostics
from networks.mlp_actor_critic import Critic, EmpiricalNormalization, GaussianActor


class PPO(Algorithm):
    name = "ppo"

    # ------------------------------------------------------------------ build
    def build(self) -> None:
        cfg = self.cfg
        env = self.env
        if env.action_dim != cfg.action_dim:
            raise ValueError(f"Expected env action_dim {env.action_dim}, got {cfg.action_dim}")
        self.num_act = cfg.action_dim
        self.actor_obs_dim = env.observation_dim
        self.critic_obs_dim = env.critic_observation_dim
        device = env.device

        self.actor = GaussianActor(
            self.actor_obs_dim, self.num_act, tuple(cfg.actor_hidden_dims), cfg.activation, cfg.init_noise_std
        ).to(device)
        self.critic = Critic(self.critic_obs_dim, tuple(cfg.actor_hidden_dims), cfg.activation).to(device)

        self.empirical_normalization = bool(cfg.empirical_normalization)
        if self.empirical_normalization:
            self.actor_obs_normalizer: nn.Module = EmpiricalNormalization(self.actor_obs_dim, device)
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(self.critic_obs_dim, device)
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
            self.critic.parameters(), lr=self.critic_learning_rate, weight_decay=cfg.weight_decay
        )

        self.num_steps_per_env = int(cfg.num_steps_per_env)
        self.max_episode_steps = int(getattr(env.task_cfg, "max_episode_steps", -1))
        self._init_train_episode_stats()

        # Combined actor-critic for state_dict checkpointing through the base interface.
        self._policy_module = nn.ModuleDict({"actor": self.actor, "critic": self.critic})
        self._opt_view = self.actor_optimizer  # core checkpoint saves this; we override save/load state below

    @property
    def policy(self) -> nn.Module:
        return self._policy_module

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.actor_optimizer

    def extra_checkpoint_state(self) -> dict:
        return {
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
            "actor_obs_normalizer": self.actor_obs_normalizer.state_dict() if self.empirical_normalization else None,
            "critic_obs_normalizer": self.critic_obs_normalizer.state_dict() if self.empirical_normalization else None,
        }

    def load_extra_checkpoint_state(self, payload: dict) -> None:
        if not payload:
            return
        if "critic_optimizer" in payload and not bool(getattr(self.cfg, "reset_optimizer_on_resume", False)):
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        self.actor_learning_rate = float(payload.get("actor_learning_rate", self.actor_learning_rate))
        self.critic_learning_rate = float(payload.get("critic_learning_rate", self.critic_learning_rate))
        if self.empirical_normalization:
            if payload.get("actor_obs_normalizer") is not None:
                self.actor_obs_normalizer.load_state_dict(payload["actor_obs_normalizer"])
            if payload.get("critic_obs_normalizer") is not None:
                self.critic_obs_normalizer.load_state_dict(payload["critic_obs_normalizer"])

    # ------------------------------------------------------------------ episode stats
    def _init_train_episode_stats(self) -> None:
        env = self.env
        self._train_reward_sum = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._train_episode_length = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._train_reward_buffer: deque[float] = deque(maxlen=100)
        self._train_length_buffer: deque[float] = deque(maxlen=100)
        self._train_completed_episodes = 0

    def _record_episode_stats(self, rewards, dones) -> None:
        self._train_reward_sum += rewards.to(dtype=torch.float32)
        self._train_episode_length += 1.0
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() == 0:
            return
        self._train_reward_buffer.extend(self._train_reward_sum.index_select(0, done_ids).detach().cpu().tolist())
        self._train_length_buffer.extend(self._train_episode_length.index_select(0, done_ids).detach().cpu().tolist())
        self._train_completed_episodes += int(done_ids.numel())
        self._train_reward_sum[done_ids] = 0.0
        self._train_episode_length[done_ids] = 0.0

    # ------------------------------------------------------------------ obs helpers
    def _norm_actor(self, obs, update=True):
        return self.actor_obs_normalizer(obs, update=update) if self.empirical_normalization else obs

    def _norm_critic(self, obs, update=True):
        return self.critic_obs_normalizer(obs, update=update) if self.empirical_normalization else obs

    # ------------------------------------------------------------------ resets
    def initial_reset(self) -> torch.Tensor:
        obs = self.env.reset()
        # init_at_random_ep_len: stagger episode timers like holosoma so envs don't all time
        # out together (only meaningful with a finite episode cap).
        if bool(self.cfg.init_at_random_ep_len) and self.max_episode_steps > 0:
            self.env.episode_steps = torch.randint_like(self.env.episode_steps, high=int(self.max_episode_steps))
        self._obs = obs
        self._critic_obs = self.env.get_critic_observation()
        return obs

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        # PPO rolls continuously (auto_reset inside step); nothing to re-sample per update.
        return self._obs

    # ------------------------------------------------------------------ rollout
    def collect(self, current_obs: torch.Tensor) -> dict:
        env = self.env
        device = env.device
        N = env.num_envs
        T = self.num_steps_per_env
        gamma = float(self.cfg.discount_gamma)

        actor_obs_buf = torch.zeros(T, N, self.actor_obs_dim, device=device)
        critic_obs_buf = torch.zeros(T, N, self.critic_obs_dim, device=device)
        actions_buf = torch.zeros(T, N, self.num_act, device=device)
        rewards_buf = torch.zeros(T, N, 1, device=device)
        dones_buf = torch.zeros(T, N, 1, dtype=torch.bool, device=device)
        values_buf = torch.zeros(T, N, 1, device=device)
        logp_buf = torch.zeros(T, N, 1, device=device)
        mu_buf = torch.zeros(T, N, self.num_act, device=device)
        sigma_buf = torch.zeros(T, N, self.num_act, device=device)

        obs = self._obs
        critic_obs = self._critic_obs
        done_terms_union: dict[str, torch.Tensor] = {}
        rollout_info_items: list[tuple] = []
        first_chunk_infos: list[dict] = []

        with torch.no_grad():
            for t in range(T):
                actor_obs = self._norm_actor(obs)
                critic_obs_n = self._norm_critic(critic_obs)
                actions = self.actor.act(actor_obs)
                values = self.critic.evaluate(critic_obs_n).detach()
                logp = self.actor.get_actions_log_prob(actions).detach().unsqueeze(1)
                mu = self.actor.action_mean.detach()
                sigma = self.actor.action_std.detach()

                next_obs, rewards, dones, infos = env.step(actions, auto_reset=True)
                next_critic_obs = env.get_critic_observation()

                # Timeout value bootstrap: add gamma * V(final_state) for envs that timed out.
                final_rewards = torch.zeros_like(rewards)
                time_outs = infos["done_terms"]["time_out"]
                if bool(time_outs.any()) and "final_critic_observation" in infos:
                    fco = self._norm_critic(infos["final_critic_observation"], update=False)
                    final_values = self.critic.evaluate(fco).detach().squeeze(1)
                    final_rewards = final_rewards + gamma * final_values * time_outs.to(dtype=rewards.dtype)

                actor_obs_buf[t] = actor_obs
                critic_obs_buf[t] = critic_obs_n
                actions_buf[t] = actions
                values_buf[t] = values
                logp_buf[t] = logp
                mu_buf[t] = mu
                sigma_buf[t] = sigma
                rewards_buf[t] = (rewards + final_rewards).view(-1, 1)
                dones_buf[t] = dones.view(-1, 1)

                self._record_episode_stats(rewards, dones)
                if t == 0:
                    first_chunk_infos.append(infos)
                rollout_info_items.append((infos, (~dones).detach()))
                for key, val in infos["done_terms"].items():
                    done_terms_union[key] = val.bool().clone() if key not in done_terms_union else (done_terms_union[key] | val.bool())

                obs = next_obs
                critic_obs = next_critic_obs

            last_critic_obs = self._norm_critic(critic_obs, update=False)
            last_values = self.critic.evaluate(last_critic_obs).detach()
            returns, advantages = self._compute_gae(last_values, values_buf, dones_buf, rewards_buf, gamma)

        self._obs = obs
        self._critic_obs = critic_obs
        return {
            "actor_obs": actor_obs_buf, "critic_obs": critic_obs_buf, "actions": actions_buf,
            "values": values_buf, "logp": logp_buf, "mu": mu_buf, "sigma": sigma_buf,
            "returns": returns, "advantages": advantages, "rewards": rewards_buf, "dones": dones_buf,
            "done_terms_union": done_terms_union, "rollout_info_items": rollout_info_items,
            "first_chunk_infos": first_chunk_infos, "next_observation": obs,
        }

    def _compute_gae(self, last_values, values, dones, rewards, gamma):
        lam = float(self.cfg.gae_lambda)
        advantage = torch.zeros_like(values[0])
        returns = torch.zeros_like(values)
        T = returns.shape[0]
        for step in reversed(range(T)):
            next_values = last_values if step == T - 1 else values[step + 1]
            next_nonterminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_nonterminal * gamma * next_values - values[step]
            advantage = delta + next_nonterminal * gamma * lam * advantage
            returns[step] = advantage + values[step]
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    # ------------------------------------------------------------------ update
    def update(self, rollout: dict, collect_time: float) -> dict:
        import time as _time
        device = self.env.device
        N = self.env.num_envs
        T = self.num_steps_per_env
        flat = lambda x: x.reshape(T * N, -1)
        actor_obs = flat(rollout["actor_obs"])
        critic_obs = flat(rollout["critic_obs"])
        actions = flat(rollout["actions"])
        old_logp = flat(rollout["logp"])
        old_values = flat(rollout["values"])
        returns = flat(rollout["returns"])
        advantages = flat(rollout["advantages"])
        old_mu = flat(rollout["mu"])
        old_sigma = flat(rollout["sigma"])

        batch_size = T * N
        num_mini_batches = int(self.cfg.num_mini_batches)
        mini_batch_size = batch_size // num_mini_batches
        epochs = int(self.cfg.num_learning_epochs)
        clip = float(self.cfg.clip_range)

        totals = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0, "kl": 0.0}
        num_updates = epochs * num_mini_batches
        grad_norm_accum = 0.0

        t1 = _time.perf_counter()
        for _ in range(epochs):
            perm = torch.randperm(batch_size, device=device)
            for mb in range(num_mini_batches):
                idx = perm[mb * mini_batch_size:(mb + 1) * mini_batch_size]
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
                entropy_batch = self.actor.entropy

                # KL-adaptive LR.
                if self.cfg.desired_kl is not None and self.cfg.desired_kl > 0.0:
                    with torch.no_grad():
                        kl = kl_divergence(Normal(mb_old_mu, mb_old_sigma), Normal(mu_batch, sigma_batch)).sum(-1)
                        kl_mean = torch.mean(kl)
                    self._update_lr(kl_mean)
                else:
                    kl_mean = torch.zeros((), device=device)

                ratio = torch.exp(logp_batch - torch.squeeze(mb_old_logp))
                surrogate = -torch.squeeze(mb_adv) * ratio
                surrogate_clipped = -torch.squeeze(mb_adv) * torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                value_clipped = mb_old_values + (value_batch - mb_old_values).clamp(-clip, clip)
                value_losses = (value_batch - mb_returns).pow(2)
                value_losses_clipped = (value_clipped - mb_returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()

                entropy_loss = entropy_batch.mean()
                actor_loss = surrogate_loss - float(self.cfg.entropy_coef) * entropy_loss
                critic_loss = float(self.cfg.value_loss_coef) * value_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                (actor_loss + critic_loss).backward()
                gn_a = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                totals["value"] += float(value_loss.item())
                totals["surrogate"] += float(surrogate_loss.item())
                totals["entropy"] += float(entropy_loss.item())
                totals["kl"] += float(kl_mean.item())
                grad_norm_accum += float(gn_a.item() if torch.is_tensor(gn_a) else gn_a)

        update_time = _time.perf_counter() - t1
        for key in totals:
            totals[key] /= num_updates
        grad_norm_accum /= num_updates

        return self._build_metrics(rollout, totals, grad_norm_accum, collect_time, update_time)

    def _update_lr(self, kl_mean: torch.Tensor) -> None:
        desired_kl = float(self.cfg.desired_kl)
        if kl_mean > desired_kl * 2.0:
            self.actor_learning_rate = max(self.min_lr, self.actor_learning_rate / 1.5)
            self.critic_learning_rate = max(self.min_lr, self.critic_learning_rate / 1.5)
        elif 0.0 < kl_mean < desired_kl / 2.0:
            self.actor_learning_rate = min(self.max_lr, self.actor_learning_rate * 1.5)
            self.critic_learning_rate = min(self.max_lr, self.critic_learning_rate * 1.5)
        for pg in self.actor_optimizer.param_groups:
            pg["lr"] = self.actor_learning_rate
        for pg in self.critic_optimizer.param_groups:
            pg["lr"] = self.critic_learning_rate

    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        actor_obs = self._norm_actor(obs, update=False)
        mean = self.actor.act_inference(actor_obs)
        return mean.unsqueeze(1)  # (N, 1, action_dim): horizon=1 chunk for validation/play

    # ------------------------------------------------------------------ metrics + logging
    def _build_metrics(self, rollout, totals, grad_norm, collect_time, update_time) -> dict:
        env = self.env
        rewards = rollout["rewards"]
        dones = rollout["dones"]
        metrics = {
            "ppo/value_loss": totals["value"],
            "ppo/surrogate_loss": totals["surrogate"],
            "ppo/entropy": totals["entropy"],
            "ppo/kl": totals["kl"],
            "ppo/actor_lr": self.actor_learning_rate,
            "ppo/critic_lr": self.critic_learning_rate,
            "ppo/grad_norm": grad_norm,
            "ppo/action_std_mean": float(self.actor.std.detach().mean().item()),
            "rollout/reward_step_mean": float(rewards.mean().item()),
            "rollout/done_frac": float(dones.float().mean().item()),
            "timing/collect_s": collect_time,
            "timing/update_s": update_time,
            "policy/action_delta": 0.0,
            "policy/param_rms_delta": 0.0,
        }
        # done-cause fractions over the whole rollout
        for key, mask in rollout["done_terms_union"].items():
            metrics[f"done/{key}_frac"] = float(mask.float().mean().item())
        # rollout reward terms (alive-weighted), reusing the shared tracking log
        rollout_reward_sums: dict[str, float] = {}
        rollout_done_union: dict[str, torch.Tensor] = {}
        weight_sum = 0.0
        for info, valid in rollout["rollout_info_items"]:
            vf = valid.float()
            w = float(vf.sum().item())
            if w <= 0.0:
                continue
            weight_sum += w
            for k, v in info["reward_terms"].items():
                rollout_reward_sums[k] = rollout_reward_sums.get(k, 0.0) + float((v * vf).sum().item())
            for k, v in info["done_terms"].items():
                md = v.bool() & valid.bool()
                rollout_done_union[k] = md.clone() if k not in rollout_done_union else (rollout_done_union[k] | md)
        if weight_sum > 0.0:
            for k, s in rollout_reward_sums.items():
                metrics[f"reward_rollout/{k}_mean"] = s / weight_sum
            for k, m in rollout_done_union.items():
                metrics[f"done_rollout/{k}_frac"] = float(m.float().mean().item())
        # chunk-0 reward terms for [TRACK_CHUNK0]
        for info in rollout["first_chunk_infos"]:
            for k, v in info["reward_terms"].items():
                metrics[f"reward/{k}_mean"] = float(v.mean().item())
        # train episode stats
        if self._train_reward_buffer:
            metrics["train/mean_reward"] = float(sum(self._train_reward_buffer) / len(self._train_reward_buffer))
            metrics["train/mean_episode_length"] = float(sum(self._train_length_buffer) / len(self._train_length_buffer))
        else:
            metrics["train/mean_reward"] = float("nan")
            metrics["train/mean_episode_length"] = float("nan")
        metrics["train/recent_episode_count"] = float(len(self._train_reward_buffer))
        metrics["train/completed_episodes"] = float(self._train_completed_episodes)
        # first-failure placeholders (PPO has no chunk concept; report NaN to keep log shape)
        for k in ("first_failure_chunk_mean", "first_failure_chunk_min", "first_failure_chunk_max",
                  "first_failure_phase_mean", "first_failure_phase_min", "first_failure_phase_max"):
            metrics[f"rollout/{k}"] = float("nan")
        return metrics

    def log(self, update_idx: int, max_updates: int, metrics: dict) -> None:
        print(
            f"[UPDATE] {update_idx}/{max_updates} "
            f"reward_step={metrics['rollout/reward_step_mean']:.5f} "
            f"done_frac={metrics['rollout/done_frac']:.5f} "
            f"mean_reward={metrics.get('train/mean_reward', float('nan')):.5f} "
            f"mean_len={metrics.get('train/mean_episode_length', float('nan')):.2f}",
            flush=True,
        )
        print(
            f"[PPO] value_loss={metrics['ppo/value_loss']:.5f} "
            f"surrogate={metrics['ppo/surrogate_loss']:.5f} "
            f"entropy={metrics['ppo/entropy']:.5f} kl={metrics['ppo/kl']:.5f} "
            f"actor_lr={metrics['ppo/actor_lr']:.6f} critic_lr={metrics['ppo/critic_lr']:.6f} "
            f"grad={metrics['ppo/grad_norm']:.5f} action_std={metrics['ppo/action_std_mean']:.4f}",
            flush=True,
        )
        log_shared_update_diagnostics(metrics, failure_label="FIRST_FAILURE", index_name="chunk")
        log_shared_tracking(metrics)

    def log_banner(self) -> None:
        cfg = self.cfg
        env = self.env
        print("[INFO] Starting PPO training", flush=True)
        print(f"[INFO] motion_file={env.task_cfg.motion_file}", flush=True)
        print(
            f"[INFO] algo=ppo actor_obs_dim={self.actor_obs_dim} critic_obs_dim={self.critic_obs_dim} "
            f"action_dim={self.num_act} num_envs={env.num_envs} num_steps_per_env={self.num_steps_per_env} "
            f"num_learning_epochs={cfg.num_learning_epochs} num_mini_batches={cfg.num_mini_batches} "
            f"gamma={cfg.discount_gamma} lam={cfg.gae_lambda} clip={cfg.clip_range} "
            f"entropy_coef={cfg.entropy_coef} value_loss_coef={cfg.value_loss_coef} "
            f"desired_kl={cfg.desired_kl} actor_lr={cfg.actor_learning_rate} critic_lr={cfg.critic_learning_rate} "
            f"init_noise_std={cfg.init_noise_std} empirical_normalization={cfg.empirical_normalization} "
            f"init_at_random_ep_len={cfg.init_at_random_ep_len} "
            f"actor_hidden_dims={list(cfg.actor_hidden_dims)} activation={cfg.activation}",
            flush=True,
        )
