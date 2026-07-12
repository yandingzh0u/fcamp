from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import torch
from torch import nn

from algorithms.base import Algorithm
from networks.mlp_actor_critic import Critic, EmpiricalNormalization
from networks.reinflow import ReinFlowPolicy


class ReinFlow(Algorithm):
    """ReinFlow-R: PPO over exact stochastic flow-chain transition likelihoods."""

    def build(self) -> None:
        cfg = self.cfg
        env = self.env
        self.horizon_h = int(cfg.horizon)
        if self.horizon_h != 4:
            raise ValueError(f"This ReinFlow comparison uses the official h=4, got {self.horizon_h}")
        if int(cfg.rollout_env_steps) % self.horizon_h:
            raise ValueError("ReinFlow rollout_env_steps must be divisible by horizon")
        self.actor_obs_dim = int(env.observation_dim)
        self.critic_obs_dim = int(env.critic_observation_dim)
        self.action_dim = int(env.action_dim)
        self.chunk_dim = self.horizon_h * self.action_dim
        self.flow_steps = int(cfg.flow_steps)
        self.rollout_env_steps = int(cfg.rollout_env_steps)
        self.decision_steps = self.rollout_env_steps // self.horizon_h
        self.device = torch.device(env.device)

        self.actor = ReinFlowPolicy(
            self.actor_obs_dim,
            self.action_dim,
            self.horizon_h,
            tuple(cfg.actor_hidden_dims),
            tuple(cfg.noise_hidden_dims),
            cfg.activation,
            self.flow_steps,
            int(cfg.timestep_embed_dim),
            float(cfg.action_scale),
            float(cfg.min_denoising_std),
            float(cfg.max_denoising_std),
            float(cfg.randn_clip_value),
        ).to(self.device)
        self.critic = Critic(
            self.critic_obs_dim, tuple(cfg.critic_hidden_dims), cfg.activation
        ).to(self.device)
        self._load_pretrained_actor(str(cfg.pretrained_actor_path))

        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=float(cfg.policy_lr),
            weight_decay=float(cfg.weight_decay),
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=float(cfg.value_lr),
            weight_decay=float(cfg.critic_weight_decay),
        )
        self.empirical_normalization = bool(cfg.empirical_normalization)
        if self.empirical_normalization:
            self.actor_obs_normalizer: nn.Module = EmpiricalNormalization(
                self.actor_obs_dim, self.device
            )
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(
                self.critic_obs_dim, self.device
            )
        else:
            self.actor_obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        self._policy_module = nn.ModuleDict({"actor": self.actor, "critic": self.critic})
        self._init_episode_stats()
        self._parameter_counts = {
            "velocity": sum(parameter.numel() for parameter in self.actor.velocity_net.parameters()),
            "noise": sum(parameter.numel() for parameter in self.actor.noise_net.parameters()),
            "critic": sum(parameter.numel() for parameter in self.critic.parameters()),
        }

    def _load_pretrained_actor(self, checkpoint: str) -> None:
        if not checkpoint:
            self.pretrained_actor_loaded = False
            return
        path = Path(checkpoint).expanduser().resolve()
        payload = torch.load(path, map_location=self.device)
        if isinstance(payload, dict):
            for key in ("actor", "policy", "model"):
                if key in payload and isinstance(payload[key], dict):
                    payload = payload[key]
                    break
        self.actor.load_state_dict(payload, strict=True)
        self.pretrained_actor_loaded = True

    @property
    def policy(self) -> nn.Module:
        return self._policy_module

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.actor_optimizer

    @property
    def horizon(self) -> int:
        return self.horizon_h

    def _norm_actor(self, obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        if self.empirical_normalization:
            return self.actor_obs_normalizer(obs, update=update)
        return obs

    def _norm_critic(self, obs: torch.Tensor, update: bool = True) -> torch.Tensor:
        if self.empirical_normalization:
            return self.critic_obs_normalizer(obs, update=update)
        return obs

    def _init_episode_stats(self) -> None:
        env = self.env
        self._episode_rewards = torch.zeros(env.num_envs, device=env.device)
        self._episode_lengths = torch.zeros(env.num_envs, device=env.device)
        self._reward_history: deque[float] = deque(maxlen=100)
        self._length_history: deque[float] = deque(maxlen=100)
        self._completed_episodes = 0

    def _record_episode_stats(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        active_float = active.float()
        self._episode_rewards += rewards * active_float
        self._episode_lengths += active_float
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() == 0:
            return
        self._reward_history.extend(self._episode_rewards[done_ids].detach().cpu().tolist())
        self._length_history.extend(self._episode_lengths[done_ids].detach().cpu().tolist())
        self._completed_episodes += int(done_ids.numel())
        self._episode_rewards[done_ids] = 0.0
        self._episode_lengths[done_ids] = 0.0

    def initial_reset(self) -> torch.Tensor:
        obs = self.env.reset()
        if bool(self.cfg.init_at_random_ep_len) and self.env.max_episode_steps > 0:
            self.env.episode_steps = torch.randint_like(
                self.env.episode_steps, high=int(self.env.max_episode_steps)
            )
        self._obs = obs
        self._critic_obs = self.env.get_critic_observation()
        return obs

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        return self._obs

    def collect(self, current_obs: torch.Tensor) -> dict:
        env = self.env
        n_envs = env.num_envs
        chunks = self.decision_steps
        horizon = self.horizon_h
        gamma = float(self.cfg.discount_gamma)

        actor_obs_buffer = torch.empty(chunks, n_envs, self.actor_obs_dim, device=self.device)
        critic_obs_buffer = torch.empty(chunks, n_envs, self.critic_obs_dim, device=self.device)
        chain_buffer = torch.empty(
            chunks,
            n_envs,
            self.flow_steps + 1,
            horizon,
            self.action_dim,
            device=self.device,
        )
        action_buffer = torch.empty(
            chunks, n_envs, horizon, self.action_dim, device=self.device
        )
        old_log_prob_buffer = torch.empty(chunks, n_envs, device=self.device)
        value_buffer = torch.empty(chunks, n_envs, device=self.device)
        reward_buffer = torch.zeros(chunks, n_envs, horizon, device=self.device)
        alive_buffer = torch.zeros(
            chunks, n_envs, horizon, dtype=torch.bool, device=self.device
        )
        done_buffer = torch.zeros_like(alive_buffer)
        failure_buffer = torch.zeros_like(alive_buffer)
        timeout_buffer = torch.zeros_like(alive_buffer)
        next_critic_obs_buffer = torch.empty(
            chunks, n_envs, horizon, self.critic_obs_dim, device=self.device
        )
        transition_std_sum = torch.zeros(self.flow_steps, device=self.device)
        transition_log_prob_sum = torch.zeros(self.flow_steps, device=self.device)
        velocity_rms_sum = torch.zeros(self.flow_steps, device=self.device)
        done_sums: dict[str, float] = {}
        reward_sums: dict[str, float] = {}
        reward_weight = 0.0
        obs = self._obs
        critic_obs = self._critic_obs

        with torch.no_grad():
            for chunk in range(chunks):
                actor_obs_n = self._norm_actor(obs)
                critic_obs_n = self._norm_critic(critic_obs)
                value = self.critic.evaluate(critic_obs_n).squeeze(-1)
                action_chunk, chains, old_log_prob, chain_stats = self.actor.sample_chain(actor_obs_n)
                actor_obs_buffer[chunk] = actor_obs_n
                critic_obs_buffer[chunk] = critic_obs_n
                value_buffer[chunk] = value
                action_buffer[chunk] = action_chunk
                chain_buffer[chunk] = chains
                old_log_prob_buffer[chunk] = old_log_prob
                transition_std_sum += chain_stats["transition_stds"].mean(dim=0)
                transition_log_prob_sum += chain_stats["transition_log_probs"].mean(dim=0)
                velocity_rms_sum += chain_stats["velocity_rms"].mean(dim=0)

                alive = torch.ones(n_envs, dtype=torch.bool, device=self.device)
                for frame in range(horizon):
                    active = alive.clone()
                    action = torch.where(
                        active.unsqueeze(-1), action_chunk[:, frame], torch.zeros_like(action_chunk[:, frame])
                    )
                    next_obs, reward, done, info = env.step(action, auto_reset=False)
                    next_critic_obs = env.get_critic_observation()
                    next_critic_obs_buffer[chunk, :, frame] = self._norm_critic(
                        next_critic_obs, update=False
                    )
                    active_float = active.float()
                    reward_buffer[chunk, :, frame] = reward * active_float
                    alive_buffer[chunk, :, frame] = active
                    timeout = info["done_terms"]["time_out"].bool()
                    failure = (
                        info["done_terms"]["anchor_pos_bad"].bool()
                        | info["done_terms"]["anchor_ori_bad"].bool()
                        | info["done_terms"]["ee_body_bad"].bool()
                    )
                    new_done = active & done.bool()
                    done_buffer[chunk, :, frame] = new_done
                    failure_buffer[chunk, :, frame] = new_done & failure & ~timeout
                    timeout_buffer[chunk, :, frame] = new_done & timeout
                    self._record_episode_stats(reward, new_done, active)
                    for key, value_tensor in info["done_terms"].items():
                        done_sums[key] = done_sums.get(key, 0.0) + float(
                            (value_tensor.bool() & new_done).float().mean().item()
                        )
                    active_count = float(active_float.sum().item())
                    reward_weight += active_count
                    for key, value_tensor in info["reward_terms"].items():
                        reward_sums[key] = reward_sums.get(key, 0.0) + float(
                            (value_tensor * active_float).sum().item()
                        )
                    alive = active & ~done.bool()
                    obs = next_obs
                    critic_obs = next_critic_obs

                chunk_done = done_buffer[chunk].any(dim=-1)
                if bool(chunk_done.any()):
                    reset_ids = chunk_done.nonzero(as_tuple=False).squeeze(-1)
                    reset_phases = env.sample_phase_indices(reset_ids.numel(), horizon=horizon)
                    reset_obs = env.reset_envs(reset_ids, phase_indices=reset_phases)
                    obs[reset_ids] = reset_obs
                    critic_obs = env.get_critic_observation()

            next_values = self.critic.evaluate(
                next_critic_obs_buffer.reshape(-1, self.critic_obs_dim)
            ).reshape(chunks, n_envs, horizon)
            gamma_powers = gamma ** torch.arange(horizon, device=self.device)
            chunk_rewards = (reward_buffer * gamma_powers.view(1, 1, horizon)).sum(dim=-1)
            frame_indices = torch.arange(horizon, device=self.device).view(1, 1, horizon)
            terminal_indices = torch.where(
                done_buffer,
                frame_indices.expand(chunks, n_envs, horizon),
                torch.full_like(done_buffer, horizon, dtype=torch.long),
            )
            death_frame = terminal_indices.min(dim=-1).values
            chunk_done = death_frame < horizon
            bootstrap_frame = death_frame.clamp(max=horizon - 1)
            bootstrap_value = next_values.gather(
                2, bootstrap_frame.unsqueeze(-1)
            ).squeeze(-1)
            chunk_failure = failure_buffer.any(dim=-1)
            bootstrap_discount = gamma ** (bootstrap_frame.float() + 1.0)
            bootstrap = torch.where(
                chunk_failure, torch.zeros_like(bootstrap_value), bootstrap_discount * bootstrap_value
            )
            one_step_target = chunk_rewards + bootstrap
            td_delta = one_step_target - value_buffer
            advantages = torch.zeros_like(td_delta)
            gae = torch.zeros(n_envs, device=self.device)
            macro_gamma_lambda = (gamma**horizon) * (float(self.cfg.gae_lambda) ** horizon)
            continuation = (~chunk_done).float()
            for chunk in range(chunks - 1, -1, -1):
                gae = td_delta[chunk] + macro_gamma_lambda * continuation[chunk] * gae
                advantages[chunk] = gae
            returns = value_buffer + advantages

        self._obs = obs
        self._critic_obs = critic_obs
        return {
            "actor_obs": actor_obs_buffer,
            "critic_obs": critic_obs_buffer,
            "chains": chain_buffer,
            "actions": action_buffer,
            "old_log_prob": old_log_prob_buffer,
            "values": value_buffer,
            "returns": returns,
            "advantages": advantages,
            "rewards": reward_buffer,
            "alive": alive_buffer,
            "dones": done_buffer,
            "failures": failure_buffer,
            "timeouts": timeout_buffer,
            "chunk_rewards": chunk_rewards,
            "transition_stds": transition_std_sum / chunks,
            "transition_log_probs": transition_log_prob_sum / chunks,
            "velocity_rms": velocity_rms_sum / chunks,
            "done_means": {
                key: value / self.rollout_env_steps for key, value in done_sums.items()
            },
            "reward_means": {
                key: value / reward_weight for key, value in reward_sums.items()
            } if reward_weight else {},
            "next_observation": obs,
        }

    def update(self, rollout: dict, collect_time: float) -> dict:
        update_start = time.perf_counter()
        chunks, n_envs = rollout["old_log_prob"].shape
        batch_size = chunks * n_envs
        actor_obs = rollout["actor_obs"].reshape(batch_size, self.actor_obs_dim)
        critic_obs = rollout["critic_obs"].reshape(batch_size, self.critic_obs_dim)
        chains = rollout["chains"].reshape(
            batch_size,
            self.flow_steps + 1,
            self.horizon_h,
            self.action_dim,
        )
        old_log_prob = rollout["old_log_prob"].reshape(batch_size)
        old_values = rollout["values"].reshape(batch_size)
        returns = rollout["returns"].reshape(batch_size)
        advantages = rollout["advantages"].reshape(batch_size)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        num_mini_batches = int(self.cfg.num_mini_batches)
        mini_batch_size = max(1, batch_size // num_mini_batches)
        epochs = int(self.cfg.policy_epochs)
        clip = float(self.cfg.clip_range)
        totals = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "ratio": 0.0,
            "clip_frac": 0.0,
            "approx_kl": 0.0,
            "actor_grad": 0.0,
            "critic_grad": 0.0,
            "new_log_prob": 0.0,
            "old_log_prob": 0.0,
            "noise_std": 0.0,
            "velocity_rms": 0.0,
            "logprob_clamp_frac": 0.0,
        }
        actor_updates = 0
        critic_updates = 0
        actor_stopped = False
        stopped_epoch = -1
        first_ratio = None
        ratio_min = float("inf")
        ratio_max = 0.0

        for epoch in range(epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            for mini_batch in range(num_mini_batches):
                index = permutation[
                    mini_batch * mini_batch_size : (mini_batch + 1) * mini_batch_size
                ]
                if index.numel() == 0:
                    continue

                value = self.critic.evaluate(critic_obs[index]).squeeze(-1)
                value_loss = 0.5 * (value - returns[index]).pow(2).mean()
                self.critic_optimizer.zero_grad(set_to_none=True)
                (float(self.cfg.value_loss_coef) * value_loss).backward()
                critic_grad = nn.utils.clip_grad_norm_(
                    self.critic.parameters(), float(self.cfg.max_grad_norm)
                )
                self.critic_optimizer.step()
                totals["value_loss"] += float(value_loss.item())
                totals["critic_grad"] += float(critic_grad.item())
                critic_updates += 1

                if actor_stopped:
                    continue
                new_log_prob_raw, entropy, chain_stats = self.actor.chain_log_prob(
                    actor_obs[index],
                    chains[index],
                    account_for_initial_stochasticity=bool(
                        self.cfg.account_for_initial_stochasticity
                    ),
                    normalize_denoising_horizon=bool(self.cfg.normalize_denoising_horizon),
                    normalize_action_dimension=bool(self.cfg.normalize_action_dimension),
                )
                old_log_prob_raw = old_log_prob[index]
                logprob_min = float(self.cfg.logprob_min)
                logprob_max = float(self.cfg.logprob_max)
                new_log_prob = new_log_prob_raw.clamp(logprob_min, logprob_max)
                old_log_prob_batch = old_log_prob_raw.clamp(logprob_min, logprob_max)
                log_ratio = new_log_prob - old_log_prob_batch
                ratio = torch.exp(log_ratio)
                advantage = advantages[index]
                objective = torch.minimum(
                    ratio * advantage,
                    ratio.clamp(1.0 - clip, 1.0 + clip) * advantage,
                )
                policy_loss = -objective.mean()
                entropy_mean = entropy.mean()
                actor_loss = policy_loss - float(self.cfg.entropy_coef) * entropy_mean
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                if first_ratio is None:
                    first_ratio = float(ratio.mean().item())
                    if abs(first_ratio - 1.0) > 1e-5:
                        raise RuntimeError(
                            f"ReinFlow first on-policy ratio must be 1, got {first_ratio:.8f}"
                        )

                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                actor_grad = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), float(self.cfg.max_grad_norm)
                )
                self.actor_optimizer.step()
                totals["policy_loss"] += float(policy_loss.item())
                totals["entropy"] += float(entropy_mean.item())
                totals["ratio"] += float(ratio.mean().item())
                totals["clip_frac"] += float(
                    ((ratio - 1.0).abs() > clip).float().mean().item()
                )
                totals["approx_kl"] += float(approx_kl.item())
                totals["actor_grad"] += float(actor_grad.item())
                totals["new_log_prob"] += float(new_log_prob_raw.mean().item())
                totals["old_log_prob"] += float(old_log_prob_raw.mean().item())
                totals["noise_std"] += float(chain_stats["transition_stds"].mean().item())
                totals["velocity_rms"] += float(chain_stats["velocity_rms"].mean().item())
                totals["logprob_clamp_frac"] += float(
                    ((new_log_prob_raw < logprob_min) | (new_log_prob_raw > logprob_max))
                    .float()
                    .mean()
                    .item()
                )
                ratio_min = min(ratio_min, float(ratio.min().item()))
                ratio_max = max(ratio_max, float(ratio.max().item()))
                actor_updates += 1
                if float(self.cfg.target_kl) > 0.0 and float(approx_kl.item()) > float(
                    self.cfg.target_kl
                ):
                    actor_stopped = True
                    stopped_epoch = epoch

        actor_denominator = max(actor_updates, 1)
        critic_denominator = max(critic_updates, 1)
        metrics = {
            "reinflow/policy_loss": totals["policy_loss"] / actor_denominator,
            "reinflow/value_loss": totals["value_loss"] / critic_denominator,
            "reinflow/entropy_rate": totals["entropy"] / actor_denominator,
            "reinflow/ratio": totals["ratio"] / actor_denominator,
            "reinflow/ratio_min": ratio_min if ratio_min != float("inf") else 0.0,
            "reinflow/ratio_max": ratio_max,
            "reinflow/clip_frac": totals["clip_frac"] / actor_denominator,
            "reinflow/approx_kl": totals["approx_kl"] / actor_denominator,
            "reinflow/first_ratio": float(first_ratio if first_ratio is not None else 0.0),
            "reinflow/actor_grad": totals["actor_grad"] / actor_denominator,
            "reinflow/critic_grad": totals["critic_grad"] / critic_denominator,
            "reinflow/new_log_prob": totals["new_log_prob"] / actor_denominator,
            "reinflow/old_log_prob": totals["old_log_prob"] / actor_denominator,
            "reinflow/logprob_clamp_frac": totals["logprob_clamp_frac"] / actor_denominator,
            "reinflow/noise_std": totals["noise_std"] / actor_denominator,
            "reinflow/velocity_rms": totals["velocity_rms"] / actor_denominator,
            "reinflow/actor_updates": float(actor_updates),
            "reinflow/critic_updates": float(critic_updates),
            "reinflow/early_stop_epoch": float(stopped_epoch),
            "reinflow/actor_lr": float(self.actor_optimizer.param_groups[0]["lr"]),
            "reinflow/critic_lr": float(self.critic_optimizer.param_groups[0]["lr"]),
            "budget/physical_transitions": float(self.rollout_env_steps * self.env.num_envs),
            "budget/policy_decisions": float(self.decision_steps * self.env.num_envs),
            "budget/chain_transitions": float(
                self.decision_steps * self.env.num_envs * self.flow_steps
            ),
            "budget/actor_sample_uses": float(actor_updates * mini_batch_size),
            "budget/critic_sample_uses": float(critic_updates * mini_batch_size),
            "params/velocity_trainable": float(self._parameter_counts["velocity"]),
            "params/noise_trainable": float(self._parameter_counts["noise"]),
            "params/critic_trainable": float(self._parameter_counts["critic"]),
            "rollout/reward_step_mean": float(
                rollout["rewards"].sum().item() / rollout["alive"].float().sum().clamp_min(1).item()
            ),
            "rollout/chunk_reward_mean": float(rollout["chunk_rewards"].mean().item()),
            "rollout/done_frac": float(rollout["dones"].float().mean().item()),
            "rollout/live_frames_mean": float(rollout["alive"].float().sum(dim=-1).mean().item()),
            "act/rollout_abs_mean": float(rollout["actions"].abs().mean().item()),
            "act/rollout_abs_max": float(rollout["actions"].abs().max().item()),
            "timing/collect_s": float(collect_time),
            "timing/update_s": float(time.perf_counter() - update_start),
        }
        for step in range(self.flow_steps):
            metrics[f"reinflow/collect_std_step_{step}"] = float(
                rollout["transition_stds"][step].item()
            )
            metrics[f"reinflow/collect_logprob_step_{step}"] = float(
                rollout["transition_log_probs"][step].item()
            )
            metrics[f"reinflow/collect_velocity_rms_step_{step}"] = float(
                rollout["velocity_rms"][step].item()
            )
        for key, value in rollout["done_means"].items():
            metrics[f"done_rollout/{key}_frac"] = float(value)
        for key, value in rollout["reward_means"].items():
            metrics[f"reward_rollout/{key}_mean"] = float(value)
        metrics["train/mean_reward"] = (
            float(sum(self._reward_history) / len(self._reward_history))
            if self._reward_history
            else float("nan")
        )
        metrics["train/mean_episode_length"] = (
            float(sum(self._length_history) / len(self._length_history))
            if self._length_history
            else float("nan")
        )
        metrics["train/completed_episodes"] = float(self._completed_episodes)
        sampler = self.env.adaptive_sampling_stats()
        for key in ("top_bin", "top_prob", "failed_sum", "entropy", "peak_bin"):
            metrics[f"sampler/{key}"] = float(sampler.get(key, float("nan")))
        return metrics

    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        actor_obs = self._norm_actor(obs, update=False)
        actions, _, _, _ = self.actor.sample_chain(actor_obs, deterministic=True)
        return actions

    def extra_checkpoint_state(self) -> dict:
        return {
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_obs_normalizer": self.actor_obs_normalizer.state_dict()
            if self.empirical_normalization
            else None,
            "critic_obs_normalizer": self.critic_obs_normalizer.state_dict()
            if self.empirical_normalization
            else None,
        }

    def load_extra_checkpoint_state(self, payload: dict, reset_optimizer: bool = False) -> None:
        if not payload:
            return
        if not reset_optimizer and "critic_optimizer" in payload:
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        if self.empirical_normalization:
            if payload.get("actor_obs_normalizer") is not None:
                self.actor_obs_normalizer.load_state_dict(payload["actor_obs_normalizer"])
            if payload.get("critic_obs_normalizer") is not None:
                self.critic_obs_normalizer.load_state_dict(payload["critic_obs_normalizer"])

    def log(self, update_idx: int, max_updates: int, metrics: dict) -> None:
        print(
            f"[UPDATE] {update_idx}/{max_updates} "
            f"reward_step={metrics['rollout/reward_step_mean']:.5f} "
            f"chunk_reward={metrics['rollout/chunk_reward_mean']:.5f} "
            f"done_frac={metrics['rollout/done_frac']:.5f} "
            f"mean_reward={metrics['train/mean_reward']:.5f} "
            f"mean_len={metrics['train/mean_episode_length']:.2f}",
            flush=True,
        )
        print(
            f"[REINFLOW] policy={metrics['reinflow/policy_loss']:.5f} "
            f"value={metrics['reinflow/value_loss']:.5f} "
            f"ratio={metrics['reinflow/ratio']:.4f} "
            f"[{metrics['reinflow/ratio_min']:.3f},{metrics['reinflow/ratio_max']:.3f}] "
            f"clip={metrics['reinflow/clip_frac']:.4f} "
            f"kl={metrics['reinflow/approx_kl']:.6f} "
            f"entropy={metrics['reinflow/entropy_rate']:.4f} "
            f"early_stop={metrics['reinflow/early_stop_epoch']:.0f}",
            flush=True,
        )
        print(
            f"[REINFLOW_CHAIN] old_logp={metrics['reinflow/old_log_prob']:.5f} "
            f"new_logp={metrics['reinflow/new_log_prob']:.5f} "
            f"clamp_frac={metrics['reinflow/logprob_clamp_frac']:.4f} "
            f"noise_std={metrics['reinflow/noise_std']:.4f} "
            f"velocity_rms={metrics['reinflow/velocity_rms']:.4f} "
            f"first_ratio={metrics['reinflow/first_ratio']:.6f}",
            flush=True,
        )
        step_stats = " ".join(
            f"s{step}={metrics[f'reinflow/collect_std_step_{step}']:.4f}"
            for step in range(self.flow_steps)
        )
        print(f"[REINFLOW_STD] {step_stats}", flush=True)
        print(
            f"[REINFLOW_LR] actor={metrics['reinflow/actor_lr']:.6f} "
            f"critic={metrics['reinflow/critic_lr']:.6f} "
            f"grad_actor={metrics['reinflow/actor_grad']:.4f} "
            f"grad_critic={metrics['reinflow/critic_grad']:.4f}",
            flush=True,
        )
        print(
            f"[BUDGET] physical_transitions={metrics['budget/physical_transitions']:.0f} "
            f"policy_decisions={metrics['budget/policy_decisions']:.0f} "
            f"chain_transitions={metrics['budget/chain_transitions']:.0f} "
            f"actor_sample_uses={metrics['budget/actor_sample_uses']:.0f} "
            f"critic_sample_uses={metrics['budget/critic_sample_uses']:.0f}",
            flush=True,
        )
        print(
            f"[TIME] collect={metrics['timing/collect_s']:.3f}s "
            f"update={metrics['timing/update_s']:.3f}s",
            flush=True,
        )

    def log_banner(self) -> None:
        cfg = self.cfg
        env = self.env
        print("[INFO] Starting ReinFlow-R training", flush=True)
        print(
            f"[INFO] task={env.task.name} terrain={env.task.terrain} motion_file={env.task.motion_file}",
            flush=True,
        )
        print(
            f"[INFO] algorithm=reinflow actor_obs_dim={self.actor_obs_dim} "
            f"critic_obs_dim={self.critic_obs_dim} action_dim={self.action_dim} "
            f"horizon={self.horizon_h} chunk_dim={self.chunk_dim} num_envs={env.num_envs} "
            f"rollout_env_steps={self.rollout_env_steps} policy_decisions={self.decision_steps} "
            f"flow_steps={self.flow_steps} min_std={cfg.min_denoising_std} "
            f"max_std={cfg.max_denoising_std} clip={cfg.clip_range} target_kl={cfg.target_kl} "
            f"logprob_clamp=[{cfg.logprob_min},{cfg.logprob_max}] "
            f"normalize_denoising={cfg.normalize_denoising_horizon} "
            f"normalize_action_dim={cfg.normalize_action_dimension} "
            f"gamma_per_frame={cfg.discount_gamma} gamma_chunk={float(cfg.discount_gamma) ** self.horizon_h:.6f} "
            f"lambda_chunk={float(cfg.gae_lambda) ** self.horizon_h:.6f} "
            f"actor_lr={cfg.policy_lr} critic_lr={cfg.value_lr} "
            f"pretrained_actor_loaded={self.pretrained_actor_loaded}",
            flush=True,
        )
        print(
            f"[INFO] trainable_params velocity={self._parameter_counts['velocity']} "
            f"noise={self._parameter_counts['noise']} critic={self._parameter_counts['critic']}",
            flush=True,
        )
