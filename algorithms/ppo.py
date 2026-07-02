from __future__ import annotations

from collections import deque

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

from algorithms.base import Algorithm
from algorithms.kl_scheduler import adaptive_lr_from_kl
from networks.mlp_actor_critic import Critic, EmpiricalNormalization, GaussianActor


class PPO(Algorithm):

    def build(self) -> None:
        cfg = self.cfg
        env = self.env
        self.num_act = env.action_dim
        self.actor_obs_dim = env.observation_dim
        self.critic_obs_dim = env.critic_observation_dim
        device = env.device

        self.actor = GaussianActor(
            self.actor_obs_dim, self.num_act, tuple(cfg.actor_hidden_dims), cfg.activation, cfg.init_noise_std
        ).to(device)


        self.critic = Critic(self.critic_obs_dim, tuple(cfg.critic_hidden_dims), cfg.activation).to(device)

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
            self.critic.parameters(), lr=self.critic_learning_rate, weight_decay=cfg.critic_weight_decay
        )

        self.num_steps_per_env = int(cfg.num_steps_per_env)
        self.max_episode_steps = env.max_episode_steps
        self._init_train_episode_stats()


        self._policy_module = nn.ModuleDict({"actor": self.actor, "critic": self.critic})

    @property
    def policy(self) -> nn.Module:
        return self._policy_module

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.actor_optimizer

    @property
    def horizon(self) -> int:
        return 1

    @property
    def kl_units(self) -> int:
        # PPO is a single-step policy: raw KL is already per-control-step.
        return 1

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


            for pg in self.actor_optimizer.param_groups:
                pg["lr"] = self.actor_learning_rate
            for pg in self.critic_optimizer.param_groups:
                pg["lr"] = self.critic_learning_rate
        else:
            if "critic_optimizer" in payload:
                self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
            self.actor_learning_rate = float(payload.get("actor_learning_rate", self.actor_learning_rate))
            self.critic_learning_rate = float(payload.get("critic_learning_rate", self.critic_learning_rate))
        if self.empirical_normalization:
            if payload.get("actor_obs_normalizer") is not None:
                self.actor_obs_normalizer.load_state_dict(payload["actor_obs_normalizer"])
            if payload.get("critic_obs_normalizer") is not None:
                self.critic_obs_normalizer.load_state_dict(payload["critic_obs_normalizer"])


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


    def _norm_actor(self, obs, update=True):
        return self.actor_obs_normalizer(obs, update=update) if self.empirical_normalization else obs

    def _norm_critic(self, obs, update=True):
        return self.critic_obs_normalizer(obs, update=update) if self.empirical_normalization else obs


    def initial_reset(self) -> torch.Tensor:
        obs = self.env.reset()


        if bool(self.cfg.init_at_random_ep_len) and self.max_episode_steps > 0:
            self.env.episode_steps = torch.randint_like(self.env.episode_steps, high=int(self.max_episode_steps))
        self._obs = obs
        self._critic_obs = self.env.get_critic_observation()
        return obs

    def reset_for_update(self, update_idx: int) -> torch.Tensor:

        return self._obs


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


        ever_done = torch.zeros(N, dtype=torch.bool, device=device)
        first_done_step = torch.full((N,), T, dtype=torch.long, device=device)
        first_done_phase = torch.full((N,), -1, dtype=torch.long, device=device)
        first_done_ee_body = torch.zeros(N, dtype=torch.bool, device=device)
        first_done_anchor_pos = torch.zeros(N, dtype=torch.bool, device=device)
        first_done_anchor_ori = torch.zeros(N, dtype=torch.bool, device=device)
        first_done_timeout = torch.zeros(N, dtype=torch.bool, device=device)

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


                rollout_info_items.append((infos, torch.ones_like(dones, dtype=torch.bool)))
                for key, val in infos["done_terms"].items():
                    done_terms_union[key] = val.bool().clone() if key not in done_terms_union else (done_terms_union[key] | val.bool())


                done_b = dones.bool()
                newly_done = (~ever_done) & done_b
                if bool(newly_done.any()):
                    ids = newly_done.nonzero(as_tuple=False).squeeze(-1)
                    first_done_step[ids] = t
                    first_done_timeout[ids] = time_outs.bool()[ids]
                    dterms = infos["done_terms"]
                    if "ee_body_bad" in dterms:
                        first_done_ee_body[ids] = dterms["ee_body_bad"].bool()[ids]
                    if "anchor_pos_bad" in dterms:
                        first_done_anchor_pos[ids] = dterms["anchor_pos_bad"].bool()[ids]
                    if "anchor_ori_bad" in dterms:
                        first_done_anchor_ori[ids] = dterms["anchor_ori_bad"].bool()[ids]
                    tps = infos.get("termination_phase_steps")
                    if torch.is_tensor(tps):
                        first_done_phase[ids] = tps.long().to(device)[ids]
                    ever_done[ids] = True

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
            "first_done_step": first_done_step, "first_done_phase": first_done_phase,
            "first_done_ee_body": first_done_ee_body, "first_done_anchor_pos": first_done_anchor_pos,
            "first_done_anchor_ori": first_done_anchor_ori, "first_done_timeout": first_done_timeout,
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
        value_clip = float(self.cfg.value_clip_range)

        totals = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0, "kl": 0.0}
        num_updates = epochs * num_mini_batches
        grad_norm_accum = 0.0


        probe_count = min(128, N)
        with torch.no_grad():
            probe_obs_n = rollout["actor_obs"][0, :probe_count]
            probe_before = self.actor.act_inference(probe_obs_n)
            params_before = [p.detach().clone() for p in self.actor.parameters()]

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

                value_clipped = mb_old_values + (value_batch - mb_old_values).clamp(-value_clip, value_clip)
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

        with torch.no_grad():
            probe_after = self.actor.act_inference(probe_obs_n)
            action_delta = float(torch.mean(torch.abs(probe_after - probe_before)).item())
            param_delta_sq = torch.zeros((), device=device)
            param_count = 0
            for p, before in zip(self.actor.parameters(), params_before, strict=True):
                d = p.detach() - before
                param_delta_sq = param_delta_sq + torch.sum(d * d)
                param_count += d.numel()
            param_rms_delta = float(torch.sqrt(param_delta_sq / max(param_count, 1)).item())

        return self._build_metrics(
            rollout, totals, grad_norm_accum, collect_time, update_time, action_delta, param_rms_delta
        )

    def _update_lr(self, kl_mean: torch.Tensor) -> None:
        desired_kl = float(self.cfg.desired_kl)
        if desired_kl <= 0.0:
            return
        # kl_units=1 for PPO, so this is behavior-preserving while sharing the
        # same per-step KL contract as SFPO/MixGRPO.
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
        for pg in self.actor_optimizer.param_groups:
            pg["lr"] = self.actor_learning_rate
        for pg in self.critic_optimizer.param_groups:
            pg["lr"] = self.critic_learning_rate

    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        actor_obs = self._norm_actor(obs, update=False)
        mean = self.actor.act_inference(actor_obs)
        return mean.unsqueeze(1)


    def _build_metrics(self, rollout, totals, grad_norm, collect_time, update_time,
                       action_delta, param_rms_delta) -> dict:
        rewards = rollout["rewards"]
        dones = rollout["dones"]
        desired_kl = float(self.cfg.desired_kl)
        kl_units = self.kl_units
        kl_raw = totals["kl"]
        kl_per_step = kl_raw / max(1, kl_units)
        metrics = {
            "ppo/value_loss": totals["value"],
            "ppo/surrogate_loss": totals["surrogate"],
            "ppo/entropy": totals["entropy"],
            "ppo/kl": kl_raw,
            "ppo/kl_raw": kl_raw,
            "ppo/kl_per_step": kl_per_step,
            "ppo/kl_target_per_step": desired_kl,
            "ppo/kl_target_raw": desired_kl * kl_units,
            "ppo/kl_units": float(kl_units),
            "ppo/actor_lr": self.actor_learning_rate,
            "ppo/critic_lr": self.critic_learning_rate,
            "ppo/grad_norm": grad_norm,
            "ppo/action_std_mean": float(self.actor.std.detach().mean().item()),
            "rollout/reward_step_mean": float(rewards.mean().item()),
            "rollout/done_frac": float(dones.float().mean().item()),
            "timing/collect_s": collect_time,
            "timing/update_s": update_time,
            "policy/action_delta": action_delta,
            "policy/param_rms_delta": param_rms_delta,
        }

        for key, mask in rollout["done_terms_union"].items():
            metrics[f"done/{key}_frac"] = float(mask.float().mean().item())

        rollout_reward_sums: dict[str, float] = {}
        weight_sum = 0.0


        done_rollout_sums: dict[str, float] = {}
        done_rollout_steps = 0
        for info, valid in rollout["rollout_info_items"]:
            done_rollout_steps += 1
            for k, v in info["done_terms"].items():
                done_rollout_sums[k] = done_rollout_sums.get(k, 0.0) + float(v.float().mean().item())
            vf = valid.float()
            w = float(vf.sum().item())
            if w <= 0.0:
                continue
            weight_sum += w
            for k, v in info["reward_terms"].items():
                rollout_reward_sums[k] = rollout_reward_sums.get(k, 0.0) + float((v * vf).sum().item())
        if weight_sum > 0.0:
            for k, s in rollout_reward_sums.items():
                metrics[f"reward_rollout/{k}_mean"] = s / weight_sum
        if done_rollout_steps > 0:
            for k, s in done_rollout_sums.items():
                metrics[f"done_rollout/{k}_frac"] = s / done_rollout_steps

        for info in rollout["first_chunk_infos"]:
            for k, v in info["reward_terms"].items():
                metrics[f"reward/{k}_mean"] = float(v.mean().item())

        if self._train_reward_buffer:
            metrics["train/mean_reward"] = float(sum(self._train_reward_buffer) / len(self._train_reward_buffer))
            metrics["train/mean_episode_length"] = float(sum(self._train_length_buffer) / len(self._train_length_buffer))
        else:
            metrics["train/mean_reward"] = float("nan")
            metrics["train/mean_episode_length"] = float("nan")
        metrics["train/recent_episode_count"] = float(len(self._train_reward_buffer))
        metrics["train/completed_episodes"] = float(self._train_completed_episodes)
        self._add_first_failure_metrics(metrics, rollout)


        with torch.no_grad():
            greedy = self.actor.act_inference(rollout["actor_obs"][0])
            g_abs = greedy.abs()
            g_flat = g_abs.reshape(-1)
            act_abs = g_abs.mean(dim=0)
            metrics["act/greedy_abs_mean"] = float(g_flat.mean().item())
            metrics["act/greedy_abs_p95"] = float(torch.quantile(g_flat, 0.95).item())
            metrics["act/greedy_abs_p99"] = float(torch.quantile(g_flat, 0.99).item())
            metrics["act/greedy_abs_max"] = float(g_flat.max().item())
            self._add_joint_group_metrics(metrics, act_abs)

            sampled = rollout["actions"].reshape(-1, self.num_act).abs()
            s_flat = sampled.reshape(-1)
            metrics["act/rollout_abs_mean"] = float(s_flat.mean().item())
            metrics["act/rollout_abs_p95"] = float(torch.quantile(s_flat, 0.95).item())
            metrics["act/rollout_abs_p99"] = float(torch.quantile(s_flat, 0.99).item())
            metrics["act/rollout_abs_max"] = float(s_flat.max().item())
        self._add_reward_weighted_metrics(metrics)
        self._add_sampler_metrics(metrics)
        return metrics

    def _add_sampler_metrics(self, metrics: dict) -> None:
        stats = self.env.adaptive_sampling_stats()
        for key in ("top_bin", "top_prob", "failed_sum", "entropy", "peak_bin"):
            metrics[f"sampler/{key}"] = float(stats.get(key, float("nan")))


    def _add_first_failure_metrics(self, metrics: dict, rollout: dict) -> None:

        T = self.num_steps_per_env
        fds = rollout["first_done_step"]
        died = fds < T
        timeout = rollout["first_done_timeout"]
        if bool(died.any()):
            steps = fds[died].float()
            metrics["rollout/first_failure_chunk_mean"] = float(steps.mean().item())
            metrics["rollout/first_failure_chunk_min"] = float(steps.min().item())
            metrics["rollout/first_failure_chunk_max"] = float(steps.max().item())
        else:
            for k in ("first_failure_chunk_mean", "first_failure_chunk_min", "first_failure_chunk_max"):
                metrics[f"rollout/{k}"] = float("nan")
        phase = rollout["first_done_phase"]
        valid_phase = phase[phase >= 0]
        if valid_phase.numel() > 0:
            metrics["rollout/first_failure_phase_mean"] = float(valid_phase.float().mean().item())
            metrics["rollout/first_failure_phase_min"] = float(valid_phase.min().item())
            metrics["rollout/first_failure_phase_max"] = float(valid_phase.max().item())
        else:
            for k in ("first_failure_phase_mean", "first_failure_phase_min", "first_failure_phase_max"):
                metrics[f"rollout/{k}"] = float("nan")
        metrics["rollout/first_failure_ee_body_frac"] = float((rollout["first_done_ee_body"] & died).float().mean().item())
        metrics["rollout/first_failure_anchor_pos_frac"] = float((rollout["first_done_anchor_pos"] & died).float().mean().item())
        metrics["rollout/first_failure_anchor_ori_frac"] = float((rollout["first_done_anchor_ori"] & died).float().mean().item())
        metrics["rollout/first_failure_timeout_frac"] = float((timeout & died).float().mean().item())
        metrics["rollout/failure_frac"] = float((died & (~timeout)).float().mean().item())

    def _add_joint_group_metrics(self, metrics: dict, act_abs: torch.Tensor) -> None:

        if act_abs.numel() < 29:
            return
        legs_idx = list(range(0, 12)); waist_idx = [12, 13, 14]; arms_idx = list(range(15, 29))
        metrics["act/legs_abs"] = float(act_abs[legs_idx].mean().item())
        metrics["act/waist_abs"] = float(act_abs[waist_idx].mean().item())
        metrics["act/arms_abs"] = float(act_abs[arms_idx].mean().item())
        metrics["act/l_shoulder_pitch"] = float(act_abs[15].item()); metrics["act/r_shoulder_pitch"] = float(act_abs[22].item())
        metrics["act/l_shoulder_roll"] = float(act_abs[16].item()); metrics["act/r_shoulder_roll"] = float(act_abs[23].item())
        metrics["act/l_shoulder_yaw"] = float(act_abs[17].item()); metrics["act/r_shoulder_yaw"] = float(act_abs[24].item())
        metrics["act/l_elbow"] = float(act_abs[18].item()); metrics["act/r_elbow"] = float(act_abs[25].item())
        metrics["act/l_wrist_roll"] = float(act_abs[19].item()); metrics["act/r_wrist_roll"] = float(act_abs[26].item())
        metrics["act/l_wrist_pitch"] = float(act_abs[20].item()); metrics["act/r_wrist_pitch"] = float(act_abs[27].item())
        metrics["act/l_wrist_yaw"] = float(act_abs[21].item()); metrics["act/r_wrist_yaw"] = float(act_abs[28].item())

    def _add_reward_weighted_metrics(self, metrics: dict) -> None:

        action_rate_weight = self.env.config.action_rate_weight
        dt = self.env.dt
        reward_weights = {
            "action_rate": -action_rate_weight,
            "joint_limit": -10.0, "anchor_pos_reward": 0.5, "anchor_ori_reward": 0.5,
            "body_pos_reward": 1.0, "body_ori_reward": 1.0, "body_lin_vel_reward": 1.0,
            "body_ang_vel_reward": 1.0, "undesired_contacts": -0.1,
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
            contribution = weight * raw_value * dt
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
            f"done_frac={metrics['rollout/done_frac']:.5f} "
            f"mean_reward={metrics.get('train/mean_reward', float('nan')):.5f} "
            f"mean_len={metrics.get('train/mean_episode_length', float('nan')):.2f}",
            flush=True,
        )
        print(
            f"[PPO] value_loss={metrics['ppo/value_loss']:.5f} "
            f"surrogate={metrics['ppo/surrogate_loss']:.5f} "
            f"entropy={metrics['ppo/entropy']:.5f} kl_raw={metrics['ppo/kl_raw']:.5f} "
            f"kl/step={metrics['ppo/kl_per_step']:.5f} "
            f"target_raw={metrics['ppo/kl_target_raw']:.4f} "
            f"target/step={metrics['ppo/kl_target_per_step']:.4f} "
            f"actor_lr={metrics['ppo/actor_lr']:.6f} critic_lr={metrics['ppo/critic_lr']:.6f} "
            f"grad={metrics['ppo/grad_norm']:.5f} action_std={metrics['ppo/action_std_mean']:.4f}",
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
            f"[DONE_ROLLOUT] timeout={metrics.get('done_rollout/time_out_frac', 0.0):.5f} "
            f"anchor_pos={metrics.get('done_rollout/anchor_pos_bad_frac', 0.0):.5f} "
            f"anchor_ori={metrics.get('done_rollout/anchor_ori_bad_frac', 0.0):.5f} "
            f"ee_body={metrics.get('done_rollout/ee_body_bad_frac', 0.0):.5f}",
            flush=True,
        )
        print(
            f"[FIRST_FAILURE] "
            f"step_mean={metrics.get('rollout/first_failure_chunk_mean', float('nan')):.2f} "
            f"step_min={metrics.get('rollout/first_failure_chunk_min', float('nan')):.0f} "
            f"step_max={metrics.get('rollout/first_failure_chunk_max', float('nan')):.0f} "
            f"phase_mean={metrics.get('rollout/first_failure_phase_mean', float('nan')):.2f} "
            f"phase_min={metrics.get('rollout/first_failure_phase_min', float('nan')):.0f} "
            f"phase_max={metrics.get('rollout/first_failure_phase_max', float('nan')):.0f} "
            f"anchor_pos={metrics.get('rollout/first_failure_anchor_pos_frac', 0.0):.5f} "
            f"anchor_ori={metrics.get('rollout/first_failure_anchor_ori_frac', 0.0):.5f} "
            f"ee_body={metrics.get('rollout/first_failure_ee_body_frac', 0.0):.5f} "
            f"timeout={metrics.get('rollout/first_failure_timeout_frac', 0.0):.5f}",
            flush=True,
        )
        print(
            f"[TRACK_CHUNK0] "
            f"anchor_pos={metrics.get('reward/anchor_pos_reward_mean', float('nan')):.5f} "
            f"anchor_ori={metrics.get('reward/anchor_ori_reward_mean', float('nan')):.5f} "
            f"body_pos={metrics.get('reward/body_pos_reward_mean', float('nan')):.5f} "
            f"body_ori={metrics.get('reward/body_ori_reward_mean', float('nan')):.5f} "
            f"body_lin={metrics.get('reward/body_lin_vel_reward_mean', float('nan')):.5f} "
            f"body_ang={metrics.get('reward/body_ang_vel_reward_mean', float('nan')):.5f}",
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
            f"[TRAIN_COST] "
            f"action_rate={metrics.get('reward/action_rate_mean', float('nan')):.5f} "
            f"joint_limit={metrics.get('reward/joint_limit_mean', float('nan')):.5f} "
            f"undesired_contacts={metrics.get('reward/undesired_contacts_mean', float('nan')):.5f}",
            flush=True,
        )
        print(
            f"[REWARD_WEIGHTED] "
            f"pos={metrics.get('reward_weighted/positive', float('nan')):.5f} "
            f"penalty={metrics.get('reward_weighted/penalty', float('nan')):.5f} "
            f"total={metrics.get('reward_weighted/total', float('nan')):.5f} "
            f"act_rate={metrics.get('reward_weighted/action_rate', float('nan')):.5f} "
            f"contacts={metrics.get('reward_weighted/undesired_contacts', float('nan')):.5f}",
            flush=True,
        )
        print(
            f"[ACT_GREEDY] "
            f"abs_mean={metrics.get('act/greedy_abs_mean', float('nan')):.4f} "
            f"abs_p95={metrics.get('act/greedy_abs_p95', float('nan')):.4f} "
            f"abs_p99={metrics.get('act/greedy_abs_p99', float('nan')):.4f} "
            f"abs_max={metrics.get('act/greedy_abs_max', float('nan')):.4f} "
            f"legs={metrics.get('act/legs_abs', float('nan')):.4f} "
            f"waist={metrics.get('act/waist_abs', float('nan')):.4f} "
            f"arms={metrics.get('act/arms_abs', float('nan')):.4f}",
            flush=True,
        )
        print(
            f"[ACT_ROLLOUT] "
            f"abs_mean={metrics.get('act/rollout_abs_mean', float('nan')):.4f} "
            f"abs_p95={metrics.get('act/rollout_abs_p95', float('nan')):.4f} "
            f"abs_p99={metrics.get('act/rollout_abs_p99', float('nan')):.4f} "
            f"abs_max={metrics.get('act/rollout_abs_max', float('nan')):.4f}",
            flush=True,
        )
        print(
            f"[SAMPLER] "
            f"top_bin={metrics.get('sampler/top_bin', float('nan')):.0f} "
            f"top_prob={metrics.get('sampler/top_prob', float('nan')):.3f} "
            f"peak_bin={metrics.get('sampler/peak_bin', float('nan')):.0f} "
            f"failed_sum={metrics.get('sampler/failed_sum', float('nan')):.4f} "
            f"entropy={metrics.get('sampler/entropy', float('nan')):.4f}",
            flush=True,
        )
        print(
            f"[TRAIN_BODY] "
            f"torso_ori={metrics.get('reward/diag_torso_ori_deg_mean', float('nan')):.2f}deg "
            f"l_wrist_ori={metrics.get('reward/diag_left_wrist_ori_deg_mean', float('nan')):.2f}deg "
            f"r_wrist_ori={metrics.get('reward/diag_right_wrist_ori_deg_mean', float('nan')):.2f}deg "
            f"l_elbow_ori={metrics.get('reward/diag_left_elbow_ori_deg_mean', float('nan')):.2f}deg "
            f"r_elbow_ori={metrics.get('reward/diag_right_elbow_ori_deg_mean', float('nan')):.2f}deg "
            f"l_shoulder_ori={metrics.get('reward/diag_left_shoulder_ori_deg_mean', float('nan')):.2f}deg "
            f"r_shoulder_ori={metrics.get('reward/diag_right_shoulder_ori_deg_mean', float('nan')):.2f}deg",
            flush=True,
        )
        print(
            f"[TRAIN_ANG] "
            f"torso={metrics.get('reward/diag_torso_ang_vel_mean', float('nan')):.3f} "
            f"l_wrist={metrics.get('reward/diag_left_wrist_ang_vel_mean', float('nan')):.3f} "
            f"r_wrist={metrics.get('reward/diag_right_wrist_ang_vel_mean', float('nan')):.3f} "
            f"l_elbow={metrics.get('reward/diag_left_elbow_ang_vel_mean', float('nan')):.3f} "
            f"r_elbow={metrics.get('reward/diag_right_elbow_ang_vel_mean', float('nan')):.3f} "
            f"l_shoulder={metrics.get('reward/diag_left_shoulder_ang_vel_mean', float('nan')):.3f} "
            f"r_shoulder={metrics.get('reward/diag_right_shoulder_ang_vel_mean', float('nan')):.3f}",
            flush=True,
        )
        print(
            f"[TRAIN_ACT] "
            f"l_wrist_r={metrics.get('act/l_wrist_roll', float('nan')):.4f} "
            f"l_wrist_p={metrics.get('act/l_wrist_pitch', float('nan')):.4f} "
            f"l_wrist_y={metrics.get('act/l_wrist_yaw', float('nan')):.4f} "
            f"r_wrist_r={metrics.get('act/r_wrist_roll', float('nan')):.4f} "
            f"r_wrist_p={metrics.get('act/r_wrist_pitch', float('nan')):.4f} "
            f"r_wrist_y={metrics.get('act/r_wrist_yaw', float('nan')):.4f} "
            f"l_elbow={metrics.get('act/l_elbow', float('nan')):.4f} "
            f"r_elbow={metrics.get('act/r_elbow', float('nan')):.4f}",
            flush=True,
        )

    def log_banner(self) -> None:
        cfg = self.cfg
        env = self.env
        print("[INFO] Starting PPO training", flush=True)
        print(
            f"[INFO] task={env.task.name} terrain={env.task.terrain} motion_file={env.task.motion_file}",
            flush=True,
        )
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
