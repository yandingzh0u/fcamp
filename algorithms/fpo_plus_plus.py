from __future__ import annotations

from collections import deque

import torch
from torch import nn

from algorithms.base import Algorithm
from networks.fpo_actor import FPOActor
from networks.mlp_actor_critic import Critic, EmpiricalNormalization


def clamp_ste(x: torch.Tensor, *, min: float | None = None, max: float | None = None) -> torch.Tensor:

    clamped = x.clamp(min=min, max=max)
    return x + (clamped - x).detach()


def fpo_ratio(old_cfm: torch.Tensor, new_cfm: torch.Tensor, delta_clip: float) -> torch.Tensor:

    diff = old_cfm - new_cfm
    if delta_clip > 0.0:
        diff = clamp_ste(diff, max=float(delta_clip))
    return torch.exp(diff)


def aspo_objective(ratio: torch.Tensor, advantage: torch.Tensor, clip: float) -> torch.Tensor:

    if clip <= 0.0:
        raise ValueError(f"ASPO/PPO clip must be > 0, got {clip}")
    ppo = torch.minimum(ratio * advantage, torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage)
    spo = ratio * advantage - advantage.abs() / (2.0 * clip) * (ratio - 1.0) ** 2
    return torch.where(advantage >= 0.0, ppo, spo)


class FPOPlusPlus(Algorithm):

    def build(self) -> None:
        cfg = self.cfg
        env = self.env
        if int(cfg.horizon) != 1:
            raise ValueError(f"FPO++ follows the official h=1 policy, got horizon={cfg.horizon}")
        self.num_act = env.action_dim
        self.num_steps_per_env = max(1, int(cfg.num_steps_per_env))
        self.actor_obs_dim = env.observation_dim
        self.critic_obs_dim = env.critic_observation_dim
        device = env.device

        self.flow_steps = max(1, int(cfg.flow_steps))
        self.actor = FPOActor(
            obs_dim=self.actor_obs_dim,
            action_dim=self.num_act,
            hidden_dims=tuple(cfg.actor_hidden_dims),
            activation=cfg.activation,
            actor_scale=float(cfg.actor_scale),
            mlp_output_scale=float(cfg.mlp_output_scale),
            timestep_embed_dim=int(cfg.timestep_embed_dim),
            cfm_loss_reduction=str(cfg.cfm_loss_reduction),
            sampling_steps=self.flow_steps,
            action_perturb_std=float(cfg.action_perturb_std),
            cfm_loss_t_inverse_cdf_beta=float(cfg.cfm_loss_t_inverse_cdf_beta),
        ).to(device)
        self.actor.train()


        self.critic = Critic(self.critic_obs_dim, tuple(cfg.critic_hidden_dims), cfg.activation).to(device)

        self.chunk_dim = self.num_act

        self.empirical_normalization = bool(cfg.empirical_normalization)
        if self.empirical_normalization:
            self.actor_obs_normalizer: nn.Module = EmpiricalNormalization(self.actor_obs_dim, device)
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(self.critic_obs_dim, device)
        else:
            self.actor_obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()


        self.learning_rate = float(cfg.policy_lr)
        if float(cfg.value_lr) <= 0.0:
            raise ValueError(f"value_lr must be > 0, got {cfg.value_lr}")
        self.critic_learning_rate = float(cfg.value_lr)
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(), lr=self.learning_rate,
            betas=(0.9, 0.999), weight_decay=float(cfg.weight_decay),
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=self.critic_learning_rate,
            betas=(0.9, 0.999), weight_decay=float(cfg.critic_weight_decay),
        )

        self.num_mc = max(1, int(cfg.fpo_num_mc))

        self.cfm_diff_clamp_max = float(cfg.fpo_delta_clip)
        self.cfm_loss_clamp = float(cfg.fpo_cfm_loss_clamp)
        self.cfm_loss_clamp_neg_adv = bool(cfg.cfm_loss_clamp_neg_adv)
        self.cfm_loss_clamp_neg_adv_max = float(cfg.cfm_loss_clamp_neg_adv_max)
        self.adv_clamp = float(cfg.fpo_adv_clamp)
        self.schedule = str(cfg.schedule)
        self.desired_kl = float(cfg.desired_kl)
        self.num_micro_batches = max(1, int(cfg.num_micro_batches))
        self.lr_min = 1e-5
        self.lr_max = 1e-2

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
        return int(self.cfg.horizon)

    def extra_checkpoint_state(self) -> dict:
        return {
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "learning_rate": self.learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
            "actor_obs_normalizer": self.actor_obs_normalizer.state_dict() if self.empirical_normalization else None,
            "critic_obs_normalizer": self.critic_obs_normalizer.state_dict() if self.empirical_normalization else None,
        }

    def load_extra_checkpoint_state(self, payload: dict, reset_optimizer: bool = False) -> None:
        if not payload:
            return
        if reset_optimizer:


            for pg in self.actor_optimizer.param_groups:
                pg["lr"] = self.learning_rate
            for pg in self.critic_optimizer.param_groups:
                pg["lr"] = self.critic_learning_rate
        else:

            self.learning_rate = float(payload.get("learning_rate", self.learning_rate))
            self.critic_learning_rate = float(
                payload.get("critic_learning_rate", self.critic_learning_rate)
            )
            for pg in self.actor_optimizer.param_groups:
                pg["lr"] = self.learning_rate
            if "critic_optimizer" in payload:
                self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
            for pg in self.critic_optimizer.param_groups:
                pg["lr"] = self.critic_learning_rate
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
        M = self.num_mc
        A = self.num_act
        T = self.num_steps_per_env
        gamma = float(self.cfg.discount_gamma)

        actor_obs_buf = torch.zeros(T, N, self.actor_obs_dim, device=device)
        critic_obs_buf = torch.zeros(T, N, self.critic_obs_dim, device=device)
        actions_buf = torch.zeros(T, N, A, device=device)
        cfm_t_buf = torch.zeros(T, N, M, 1, device=device)
        cfm_eps_buf = torch.zeros(T, N, M, A, device=device)
        old_cfm_buf = torch.zeros(T, N, M, device=device)
        x1_pred_buf = torch.zeros(T, N, M, A, device=device)
        values_buf = torch.zeros(T, N, 1, device=device)
        rewards_buf = torch.zeros(T, N, 1, device=device)
        dones_buf = torch.zeros(T, N, 1, dtype=torch.bool, device=device)

        obs = self._obs
        critic_obs = self._critic_obs
        done_terms_union: dict[str, torch.Tensor] = {}
        rollout_info_items: list[tuple] = []
        first_chunk_infos: list[dict] = []
        action_abs_max = 0.0


        ever_done = torch.zeros(N, dtype=torch.bool, device=device)
        first_done_step = torch.full((N,), T, dtype=torch.long, device=device)
        first_done_phase = torch.full((N,), -1, dtype=torch.long, device=device)
        first_done_ee_body = torch.zeros(N, dtype=torch.bool, device=device)
        first_done_anchor_pos = torch.zeros(N, dtype=torch.bool, device=device)
        first_done_anchor_ori = torch.zeros(N, dtype=torch.bool, device=device)
        first_done_timeout = torch.zeros(N, dtype=torch.bool, device=device)

        with torch.no_grad():
            for t in range(T):
                actor_obs_n = self._norm_actor(obs)
                critic_obs_n = self._norm_critic(critic_obs)
                value = self.critic.evaluate(critic_obs_n).detach()

                action = self.actor.act(actor_obs_n).detach()
                action_abs_max = max(action_abs_max, float(action.abs().max().item()))

                cfm_eps = torch.randn(N, M, A, device=device)
                cfm_t = self.actor.sample_cfm_timesteps(N, M, device=device)
                old_cfm, x1_pred, _ = self.actor.get_cfm_loss(actor_obs_n, action, cfm_eps, cfm_t)
                old_cfm = old_cfm.detach()
                x1_pred = x1_pred.detach()

                next_obs, reward, done, info = env.step(action, auto_reset=True)
                next_critic_obs = env.get_critic_observation()

                done_b = done.bool()
                time_outs = info["done_terms"]["time_out"]
                time_outs_b = time_outs.bool()


                newly_done = (~ever_done) & done_b
                if bool(newly_done.any()):
                    ids = newly_done.nonzero(as_tuple=False).squeeze(-1)
                    first_done_step[ids] = t
                    first_done_timeout[ids] = time_outs_b[ids]
                    dterms = info["done_terms"]
                    if "ee_body_bad" in dterms:
                        first_done_ee_body[ids] = dterms["ee_body_bad"].bool()[ids]
                    if "anchor_pos_bad" in dterms:
                        first_done_anchor_pos[ids] = dterms["anchor_pos_bad"].bool()[ids]
                    if "anchor_ori_bad" in dterms:
                        first_done_anchor_ori[ids] = dterms["anchor_ori_bad"].bool()[ids]
                    tps = info.get("termination_phase_steps")
                    if torch.is_tensor(tps):
                        first_done_phase[ids] = tps.long().to(device)[ids]
                    ever_done[ids] = True


                final_rewards = torch.zeros_like(reward)
                if bool(time_outs.any()) and "final_critic_observation" in info:
                    fco = self._norm_critic(info["final_critic_observation"], update=False)
                    final_values = self.critic.evaluate(fco).detach().squeeze(1)
                    final_rewards = final_rewards + gamma * final_values * time_outs.to(dtype=reward.dtype)

                actor_obs_buf[t] = actor_obs_n
                critic_obs_buf[t] = critic_obs_n
                actions_buf[t] = action
                cfm_t_buf[t] = cfm_t
                cfm_eps_buf[t] = cfm_eps
                old_cfm_buf[t] = old_cfm
                x1_pred_buf[t] = x1_pred
                values_buf[t] = value
                rewards_buf[t] = (reward + final_rewards).view(-1, 1)
                dones_buf[t] = done.view(-1, 1)

                self._record_episode_stats(reward, done)
                if t == 0:
                    first_chunk_infos.append(info)


                valid = torch.ones_like(done, dtype=torch.bool)
                rollout_info_items.append((info, valid.detach()))
                for key, val in info["done_terms"].items():
                    b = val.bool()
                    done_terms_union[key] = b.clone() if key not in done_terms_union else (done_terms_union[key] | b)

                obs = next_obs
                critic_obs = next_critic_obs

            last_critic_obs = self._norm_critic(critic_obs, update=False)
            last_values = self.critic.evaluate(last_critic_obs).detach()
            returns, advantages = self._compute_gae(last_values, values_buf, dones_buf, rewards_buf, gamma)

        self._obs = obs
        self._critic_obs = critic_obs


        return {
            "actor_obs": actor_obs_buf, "critic_obs": critic_obs_buf, "actions": actions_buf,
            "cfm_t": cfm_t_buf, "cfm_eps": cfm_eps_buf, "old_cfm": old_cfm_buf, "x1_pred": x1_pred_buf,
            "values": values_buf, "returns": returns, "advantages": advantages,
            "rewards": rewards_buf, "dones": dones_buf,
            "done_terms_union": done_terms_union, "rollout_info_items": rollout_info_items,
            "first_chunk_infos": first_chunk_infos, "action_abs_max": action_abs_max,
            "first_done_step": first_done_step, "first_done_phase": first_done_phase,
            "first_done_ee_body": first_done_ee_body, "first_done_anchor_pos": first_done_anchor_pos,
            "first_done_anchor_ori": first_done_anchor_ori, "first_done_timeout": first_done_timeout,
            "next_observation": obs,
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
        T, N = rollout["actions"].shape[0], rollout["actions"].shape[1]
        B = T * N
        M = self.num_mc
        A = self.num_act

        actor_obs = rollout["actor_obs"].reshape(B, self.actor_obs_dim)
        critic_obs = rollout["critic_obs"].reshape(B, self.critic_obs_dim)
        actions = rollout["actions"].reshape(B, A)
        cfm_t = rollout["cfm_t"].reshape(B, M, 1)
        cfm_eps = rollout["cfm_eps"].reshape(B, M, A)
        old_cfm = rollout["old_cfm"].reshape(B, M)
        old_x1_pred = rollout["x1_pred"].reshape(B, M, A)
        returns = rollout["returns"].reshape(B, 1)
        advantages = rollout["advantages"].reshape(B, 1)
        old_values = rollout["values"].reshape(B, 1)

        num_mini_batches = max(1, int(self.cfg.num_mini_batches))
        mini_batch_size = max(1, B // num_mini_batches)
        epochs = int(self.cfg.num_learning_epochs)
        clip = float(self.cfg.clip_range)
        value_clip = float(self.cfg.value_clip_range)
        value_coef = float(self.cfg.value_loss_coef)
        use_clipped_value_loss = bool(self.cfg.use_clipped_value_loss)

        totals = {
            "actor_loss": 0.0, "value_loss": 0.0, "ratio": 0.0, "ratio_min": float("inf"),
            "ratio_max": 0.0, "clip_frac": 0.0, "cfm_new": 0.0, "cfm_old": 0.0, "grad_norm": 0.0,
            "grad_norm_critic": 0.0, "kl": 0.0, "ratio_mc_std": 0.0,
            "cfm_log_ratio_mean": 0.0, "cfm_log_ratio_std": 0.0,
            "positive_adv_frac": 0.0, "aspo_positive": 0.0, "aspo_negative": 0.0,
        }
        num_updates = 0


        probe_count = min(128, N)
        with torch.no_grad():
            probe_obs_n = rollout["actor_obs"][0, :probe_count]
            probe_before = self.actor.act_inference(probe_obs_n, eval_mode="zero")
            params_before = [p.detach().clone() for p in self.actor.parameters()]

        t1 = _time.perf_counter()
        for _ in range(epochs):
            perm = torch.randperm(B, device=device)
            for mb in range(num_mini_batches):
                idx = perm[mb * mini_batch_size:(mb + 1) * mini_batch_size]
                mb_size = idx.numel()
                if mb_size == 0:
                    continue

                mb_adv_full = advantages[idx].clamp(-self.adv_clamp, self.adv_clamp)

                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                micro_chunks = torch.chunk(torch.arange(mb_size, device=device), self.num_micro_batches)
                agg = {k: 0.0 for k in (
                    "actor_loss", "value_loss", "ratio", "clip_frac", "cfm_new", "cfm_old", "kl",
                    "ratio_mc_std", "cfm_log_ratio_mean", "cfm_log_ratio_std",
                    "positive_adv_frac", "aspo_positive", "aspo_negative",
                )}
                agg_ratio_min = float("inf")
                agg_ratio_max = 0.0
                for sub in micro_chunks:
                    if sub.numel() == 0:
                        continue
                    weight = float(sub.numel()) / float(mb_size)
                    li = idx[sub]
                    new_cfm, x1_pred, _ = self.actor.get_cfm_loss(
                        actor_obs[li], actions[li], cfm_eps[li], cfm_t[li]
                    )
                    value = self.critic.evaluate(critic_obs[li])
                    mb_adv = mb_adv_full[sub]
                    mb_old_cfm = old_cfm[li]


                    if self.cfm_loss_clamp > 0.0:
                        mb_old_cfm = mb_old_cfm.clamp(max=self.cfm_loss_clamp)
                        new_cfm = new_cfm.clamp(max=self.cfm_loss_clamp)


                    if self.cfm_loss_clamp_neg_adv:
                        new_cfm = torch.where(
                            mb_adv < 0, new_cfm.clamp(max=self.cfm_loss_clamp_neg_adv_max), new_cfm
                        )

                    ratio = fpo_ratio(mb_old_cfm, new_cfm, self.cfm_diff_clamp_max)
                    surrogate = aspo_objective(ratio, mb_adv, clip)
                    actor_loss = -surrogate.mean()


                    if use_clipped_value_loss:
                        mb_old_values = old_values[li]
                        value_clipped = mb_old_values + (value - mb_old_values).clamp(-value_clip, value_clip)
                        value_losses = (value - returns[li]).pow(2)
                        value_losses_clipped = (value_clipped - returns[li]).pow(2)
                        value_loss = torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = (returns[li] - value).pow(2).mean()

                    loss = actor_loss + value_coef * value_loss
                    (loss * weight).backward()

                    with torch.no_grad():
                        agg["actor_loss"] += float(actor_loss.item()) * weight
                        agg["value_loss"] += float(value_loss.item()) * weight
                        agg["ratio"] += float(ratio.mean().item()) * weight
                        agg["clip_frac"] += float((torch.abs(ratio - 1.0) > clip).float().mean().item()) * weight
                        agg["cfm_new"] += float(new_cfm.mean().item()) * weight
                        agg["cfm_old"] += float(mb_old_cfm.mean().item()) * weight
                        log_ratio = mb_old_cfm - new_cfm
                        positive = (mb_adv >= 0.0).expand_as(surrogate)
                        negative = ~positive
                        agg["ratio_mc_std"] += float(
                            ratio.std(dim=1, unbiased=False).mean().item()
                        ) * weight
                        agg["cfm_log_ratio_mean"] += float(log_ratio.mean().item()) * weight
                        agg["cfm_log_ratio_std"] += float(log_ratio.std(unbiased=False).item()) * weight
                        agg["positive_adv_frac"] += float(positive.float().mean().item()) * weight
                        if bool(positive.any()):
                            agg["aspo_positive"] += float(surrogate[positive].mean().item()) * weight
                        if bool(negative.any()):
                            agg["aspo_negative"] += float(surrogate[negative].mean().item()) * weight
                        agg_ratio_min = min(agg_ratio_min, float(ratio.min().item()))
                        agg_ratio_max = max(agg_ratio_max, float(ratio.max().item()))

                        kl_micro = ((x1_pred.detach() - old_x1_pred[li]) ** 2).mean()
                        agg["kl"] += float(kl_micro.item()) * weight


                if self.schedule == "adaptive":
                    kl_mean = agg["kl"]
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(self.lr_min, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(self.lr_max, self.learning_rate * 1.5)
                    for group in self.actor_optimizer.param_groups:
                        group["lr"] = self.learning_rate


                grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                grad_norm_critic = nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                totals["actor_loss"] += agg["actor_loss"]
                totals["value_loss"] += agg["value_loss"]
                totals["ratio"] += agg["ratio"]
                totals["ratio_min"] = min(totals["ratio_min"], agg_ratio_min)
                totals["ratio_max"] = max(totals["ratio_max"], agg_ratio_max)
                totals["clip_frac"] += agg["clip_frac"]
                totals["cfm_new"] += agg["cfm_new"]
                totals["cfm_old"] += agg["cfm_old"]
                totals["kl"] += agg["kl"]
                for key in (
                    "ratio_mc_std", "cfm_log_ratio_mean", "cfm_log_ratio_std",
                    "positive_adv_frac", "aspo_positive", "aspo_negative",
                ):
                    totals[key] += agg[key]
                totals["grad_norm"] += float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                totals["grad_norm_critic"] += float(grad_norm_critic.item() if torch.is_tensor(grad_norm_critic) else grad_norm_critic)
                num_updates += 1

        update_time = _time.perf_counter() - t1
        denom = max(num_updates, 1)
        with torch.no_grad():
            probe_after = self.actor.act_inference(probe_obs_n, eval_mode="zero")
            action_delta = float(torch.mean(torch.abs(probe_after - probe_before)).item())
            param_delta_sq = torch.zeros((), device=device)
            param_count = 0
            for p, before in zip(self.actor.parameters(), params_before, strict=True):
                d = p.detach() - before
                param_delta_sq = param_delta_sq + torch.sum(d * d)
                param_count += d.numel()
            param_rms_delta = float(torch.sqrt(param_delta_sq / max(param_count, 1)).item())

        agg_out = {
            "actor_loss": totals["actor_loss"] / denom,
            "value_loss": totals["value_loss"] / denom,
            "ratio": totals["ratio"] / denom,
            "ratio_min": totals["ratio_min"] if totals["ratio_min"] != float("inf") else 0.0,
            "ratio_max": totals["ratio_max"],
            "clip_frac": totals["clip_frac"] / denom,
            "cfm_new": totals["cfm_new"] / denom,
            "cfm_old": totals["cfm_old"] / denom,
            "grad_norm": totals["grad_norm"] / denom,
            "grad_norm_critic": totals["grad_norm_critic"] / denom,
            "kl": totals["kl"] / denom,
            "ratio_mc_std": totals["ratio_mc_std"] / denom,
            "cfm_log_ratio_mean": totals["cfm_log_ratio_mean"] / denom,
            "cfm_log_ratio_std": totals["cfm_log_ratio_std"] / denom,
            "positive_adv_frac": totals["positive_adv_frac"] / denom,
            "aspo_positive": totals["aspo_positive"] / denom,
            "aspo_negative": totals["aspo_negative"] / denom,
            "action_delta": action_delta,
            "param_rms_delta": param_rms_delta,
            "mini_batch_size": float(mini_batch_size),
        }
        return self._build_metrics(rollout, agg_out, collect_time, update_time)


    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:

        actor_obs_n = self._norm_actor(obs, update=False)
        action = self.actor.act_inference(actor_obs_n, eval_mode="zero")
        return action.unsqueeze(1)


    def _add_first_failure_metrics(self, metrics: dict, rollout: dict) -> None:

        T = self.num_steps_per_env
        fds = rollout.get("first_done_step")
        if fds is None:
            for k in ("first_failure_chunk_mean", "first_failure_chunk_min", "first_failure_chunk_max",
                      "first_failure_phase_mean", "first_failure_phase_min", "first_failure_phase_max"):
                metrics[f"rollout/{k}"] = float("nan")
            return
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

    def _build_metrics(self, rollout, agg, collect_time, update_time) -> dict:
        rewards = rollout["rewards"]
        dones = rollout["dones"]
        metrics = {
            "fpo_pp/actor_loss": agg["actor_loss"],
            "fpo_pp/value_loss": agg["value_loss"],
            "fpo_pp/ratio": agg["ratio"],
            "fpo_pp/ratio_min": agg["ratio_min"],
            "fpo_pp/ratio_max": agg["ratio_max"],
            "fpo_pp/clip_frac": agg["clip_frac"],
            "fpo_pp/cfm_new": agg["cfm_new"],
            "fpo_pp/cfm_old": agg["cfm_old"],
            "fpo_pp/cfm_log_ratio_mean": agg["cfm_log_ratio_mean"],
            "fpo_pp/cfm_log_ratio_std": agg["cfm_log_ratio_std"],
            "fpo_pp/ratio_mc_std": agg["ratio_mc_std"],
            "fpo_pp/positive_adv_frac": agg["positive_adv_frac"],
            "fpo_pp/aspo_positive": agg["aspo_positive"],
            "fpo_pp/aspo_negative": agg["aspo_negative"],
            "fpo_pp/grad_norm_actor": agg["grad_norm"],
            "fpo_pp/grad_norm_critic": agg["grad_norm_critic"],
            "fpo_pp/kl_x1_mse": agg["kl"],
            "fpo_pp/actor_lr": self.learning_rate,
            "fpo_pp/critic_lr": self.critic_learning_rate,
            "budget/physical_transitions": float(self.num_steps_per_env * self.env.num_envs),
            "budget/policy_decisions": float(self.num_steps_per_env * self.env.num_envs),
            "budget/cfm_mc_terms": float(self.num_steps_per_env * self.env.num_envs * self.num_mc),
            "rollout/reward_step_mean": float(rewards.mean().item()),
            "rollout/done_frac": float(dones.float().mean().item()),
            "act/abs_max_all": float(rollout.get("action_abs_max", 0.0)),
            "timing/collect_s": collect_time,
            "timing/update_s": update_time,
            "policy/action_delta": agg["action_delta"],
            "policy/param_rms_delta": agg["param_rms_delta"],
            "policy/effective_mini_batch_size": agg["mini_batch_size"],
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
            greedy = self.actor.act_inference(rollout["actor_obs"][0], eval_mode="zero")
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
            f"[FPO++] actor_loss={metrics['fpo_pp/actor_loss']:.5f} "
            f"value_loss={metrics['fpo_pp/value_loss']:.5f} "
            f"ratio={metrics['fpo_pp/ratio']:.4f} "
            f"[{metrics['fpo_pp/ratio_min']:.3f},{metrics['fpo_pp/ratio_max']:.3f}] "
            f"clip_frac={metrics['fpo_pp/clip_frac']:.4f} "
            f"kl_x1={metrics['fpo_pp/kl_x1_mse']:.6f} "
            f"grad_a={metrics['fpo_pp/grad_norm_actor']:.4f} "
            f"grad_c={metrics['fpo_pp/grad_norm_critic']:.4f} "
            f"actor_lr={metrics['fpo_pp/actor_lr']:.6f} "
            f"critic_lr={metrics['fpo_pp/critic_lr']:.6f}",
            flush=True,
        )
        print(
            f"[FPO++_CORE] cfm_old={metrics['fpo_pp/cfm_old']:.4f} "
            f"cfm_new={metrics['fpo_pp/cfm_new']:.4f} "
            f"log_ratio={metrics['fpo_pp/cfm_log_ratio_mean']:.5f}+/-"
            f"{metrics['fpo_pp/cfm_log_ratio_std']:.5f} "
            f"mc_ratio_std={metrics['fpo_pp/ratio_mc_std']:.5f} "
            f"positive_adv_frac={metrics['fpo_pp/positive_adv_frac']:.3f} "
            f"aspo_pos={metrics['fpo_pp/aspo_positive']:.5f} "
            f"aspo_neg={metrics['fpo_pp/aspo_negative']:.5f}",
            flush=True,
        )
        print(
            f"[BUDGET] physical_transitions={metrics['budget/physical_transitions']:.0f} "
            f"policy_decisions={metrics['budget/policy_decisions']:.0f} "
            f"cfm_mc_terms={metrics['budget/cfm_mc_terms']:.0f}",
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
            f"abs_max={metrics.get('act/rollout_abs_max', float('nan')):.4f} "
            f"abs_max_all={metrics.get('act/abs_max_all', float('nan')):.4f}",
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
        print("[INFO] Starting FPO++ training", flush=True)
        print(
            f"[INFO] task={env.task.name} terrain={env.task.terrain} motion_file={env.task.motion_file}",
            flush=True,
        )
        print(
            f"[INFO] algorithm=fpo++ actor_obs_dim={self.actor_obs_dim} critic_obs_dim={self.critic_obs_dim} "
            f"action_dim={self.num_act} horizon={self.horizon} "
            f"chunk_dim={self.chunk_dim} num_envs={env.num_envs} num_steps_per_env={self.num_steps_per_env} "
            f"flow_steps={self.flow_steps} num_mc={self.num_mc} actor_scale={self.actor.actor_scale} "
            f"action_perturb_std={self.actor.action_perturb_std} timestep_embed_dim={self.actor.timestep_embed_dim} "
            f"cfm_reduction={self.actor.cfm_loss_reduction} clip={cfg.clip_range} "
            f"cfm_diff_clamp_max={self.cfm_diff_clamp_max} cfm_loss_clamp={self.cfm_loss_clamp} "
            f"adv_clamp={self.adv_clamp} schedule={self.schedule} desired_kl={self.desired_kl} "
            f"num_micro_batches={self.num_micro_batches} "
            f"num_learning_epochs={cfg.num_learning_epochs} num_mini_batches={cfg.num_mini_batches} "
            f"gamma={cfg.discount_gamma} lam={cfg.gae_lambda} value_loss_coef={cfg.value_loss_coef} "
            f"use_clipped_value_loss={cfg.use_clipped_value_loss} "
            f"actor_lr={cfg.policy_lr} critic_lr={self.critic_learning_rate} "
            f"adaptive_actor_only={self.schedule == 'adaptive'} weight_decay={cfg.weight_decay} "
            f"empirical_normalization={cfg.empirical_normalization} "
            f"actor_hidden_dims={list(cfg.actor_hidden_dims)} activation={cfg.activation}",
            flush=True,
        )
