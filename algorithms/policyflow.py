from __future__ import annotations

import time
from collections import deque

import torch
from torch import nn

from algorithms.base import Algorithm
from networks.mlp_actor_critic import Critic, EmpiricalNormalization
from networks.policyflow import PolicyFlowActor


class PolicyFlow(Algorithm):
    """Official on-policy PolicyFlow objective with a one-action CNF policy."""

    def build(self) -> None:
        cfg = self.cfg
        env = self.env
        if int(cfg.horizon) != 1:
            raise ValueError(f"Official PolicyFlow uses action horizon h=1, got {cfg.horizon}")
        self.device = torch.device(env.device)
        self.obs_dim = int(env.observation_dim)
        self.critic_obs_dim = int(env.critic_observation_dim)
        self.action_dim = int(env.action_dim)
        self.rollout_steps = int(cfg.rollout_env_steps)

        self.actor = PolicyFlowActor(
            self.obs_dim,
            self.action_dim,
            tuple(cfg.actor_hidden_dims),
            cfg.activation,
            int(cfg.flow_steps),
            int(cfg.timestep_embed_dim),
            float(cfg.init_noise_std),
        ).to(self.device)
        self.critic = Critic(
            self.critic_obs_dim,
            tuple(cfg.critic_hidden_dims),
            cfg.activation,
        ).to(self.device)

        self.empirical_normalization = bool(cfg.empirical_normalization)
        if self.empirical_normalization:
            self.actor_obs_normalizer: nn.Module = EmpiricalNormalization(self.obs_dim, self.device)
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(
                self.critic_obs_dim, self.device
            )
        else:
            self.actor_obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        self.actor_learning_rate = float(cfg.actor_learning_rate)
        self.critic_learning_rate = float(cfg.critic_learning_rate)
        actor_parameters = [parameter for parameter in self.actor.parameters() if parameter.requires_grad]
        self._optimizer = torch.optim.AdamW(
            (
                {
                    "params": actor_parameters,
                    "lr": self.actor_learning_rate,
                    "weight_decay": float(cfg.weight_decay),
                    "name": "actor",
                },
                {
                    "params": self.critic.parameters(),
                    "lr": self.critic_learning_rate,
                    "weight_decay": float(cfg.critic_weight_decay),
                    "name": "critic",
                },
            )
        )
        self._policy_module = nn.ModuleDict({"actor": self.actor, "critic": self.critic})
        self._trainable_counts = {
            "actor": sum(parameter.numel() for parameter in actor_parameters),
            "critic": sum(parameter.numel() for parameter in self.critic.parameters()),
        }
        self._init_episode_stats()

    @property
    def policy(self) -> nn.Module:
        return self._policy_module

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def horizon(self) -> int:
        return 1

    def _norm_actor(self, observation: torch.Tensor, *, update: bool) -> torch.Tensor:
        if self.empirical_normalization:
            return self.actor_obs_normalizer(observation, update=update)
        return observation

    def _norm_critic(self, observation: torch.Tensor, *, update: bool) -> torch.Tensor:
        if self.empirical_normalization:
            return self.critic_obs_normalizer(observation, update=update)
        return observation

    def _init_episode_stats(self) -> None:
        env = self.env
        self._episode_rewards = torch.zeros(env.num_envs, device=env.device)
        self._episode_lengths = torch.zeros(env.num_envs, device=env.device)
        self._reward_history: deque[float] = deque(maxlen=100)
        self._length_history: deque[float] = deque(maxlen=100)
        self._completed_episodes = 0

    def _record_episode_stats(self, rewards: torch.Tensor, dones: torch.Tensor) -> None:
        self._episode_rewards += rewards
        self._episode_lengths += 1.0
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() == 0:
            return
        self._reward_history.extend(self._episode_rewards[done_ids].detach().cpu().tolist())
        self._length_history.extend(self._episode_lengths[done_ids].detach().cpu().tolist())
        self._completed_episodes += int(done_ids.numel())
        self._episode_rewards[done_ids] = 0.0
        self._episode_lengths[done_ids] = 0.0

    def initial_reset(self) -> torch.Tensor:
        observation = self.env.reset()
        if bool(self.cfg.init_at_random_ep_len) and self.env.max_episode_steps > 0:
            self.env.episode_steps = torch.randint_like(
                self.env.episode_steps, high=int(self.env.max_episode_steps)
            )
        self._obs = observation
        self._critic_obs = self.env.get_critic_observation()
        return observation

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        return self._obs

    def collect(self, current_obs: torch.Tensor) -> dict:
        env = self.env
        num_envs = env.num_envs
        steps = self.rollout_steps
        actor_obs = torch.empty(steps, num_envs, self.obs_dim, device=self.device)
        critic_obs = torch.empty(steps, num_envs, self.critic_obs_dim, device=self.device)
        base_noise = torch.empty(steps, num_envs, self.action_dim, device=self.device)
        prior = torch.empty_like(base_noise)
        delta = torch.empty_like(base_noise)
        action_std = torch.empty_like(base_noise)
        actions = torch.empty_like(base_noise)
        old_log_prob = torch.empty(steps, num_envs, 1, device=self.device)
        rewards = torch.empty(steps, num_envs, 1, device=self.device)
        dones = torch.empty(steps, num_envs, 1, dtype=torch.bool, device=self.device)
        values = torch.empty(steps, num_envs, 1, device=self.device)
        done_sums: dict[str, float] = {}
        reward_sums: dict[str, float] = {}

        observation = self._obs
        critic_observation = self._critic_obs
        gamma = float(self.cfg.discount_gamma)
        with torch.no_grad():
            for step in range(steps):
                actor_observation = self._norm_actor(observation, update=True)
                critic_observation_n = self._norm_critic(critic_observation, update=True)
                action, action_info = self.actor.sample_action(actor_observation)
                value = self.critic.evaluate(critic_observation_n)
                next_observation, reward, done, info = env.step(action, auto_reset=True)
                next_critic_observation = env.get_critic_observation()

                timeout = info["done_terms"]["time_out"].bool()
                bootstrap_reward = torch.zeros_like(reward)
                if bool(timeout.any()) and "final_critic_observation" in info:
                    final_critic = self._norm_critic(
                        info["final_critic_observation"], update=False
                    )
                    final_value = self.critic.evaluate(final_critic).squeeze(-1)
                    bootstrap_reward = gamma * final_value * timeout.to(reward.dtype)

                actor_obs[step] = actor_observation
                critic_obs[step] = critic_observation_n
                base_noise[step] = action_info["base_noise"]
                prior[step] = action_info["prior"]
                delta[step] = action_info["delta"]
                action_std[step] = action_info["std"]
                actions[step] = action
                old_log_prob[step, :, 0] = action_info["log_prob"]
                values[step] = value
                rewards[step, :, 0] = reward + bootstrap_reward
                dones[step, :, 0] = done
                self._record_episode_stats(reward, done.bool())
                for key, value_term in info["done_terms"].items():
                    done_sums[key] = done_sums.get(key, 0.0) + float(
                        value_term.float().mean().item()
                    )
                for key, value_term in info["reward_terms"].items():
                    reward_sums[key] = reward_sums.get(key, 0.0) + float(value_term.mean().item())
                observation = next_observation
                critic_observation = next_critic_observation

            last_critic = self._norm_critic(critic_observation, update=False)
            last_values = self.critic.evaluate(last_critic)
            returns, advantages = self._compute_gae(last_values, values, dones, rewards)

        self._obs = observation
        self._critic_obs = critic_observation
        return {
            "actor_obs": actor_obs,
            "critic_obs": critic_obs,
            "base_noise": base_noise,
            "prior": prior,
            "delta": delta,
            "action_std": action_std,
            "actions": actions,
            "old_log_prob": old_log_prob,
            "values": values,
            "returns": returns,
            "advantages": advantages,
            "rewards": rewards,
            "dones": dones,
            "done_means": {key: value / steps for key, value in done_sums.items()},
            "reward_means": {key: value / steps for key, value in reward_sums.items()},
            "next_observation": observation,
        }

    def _compute_gae(
        self,
        last_values: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gamma = float(self.cfg.discount_gamma)
        gae_lambda = float(self.cfg.gae_lambda)
        advantage = torch.zeros_like(last_values)
        returns = torch.empty_like(values)
        for step in reversed(range(values.shape[0])):
            next_value = last_values if step == values.shape[0] - 1 else values[step + 1]
            not_done = 1.0 - dones[step].float()
            delta = rewards[step] + gamma * not_done * next_value - values[step]
            advantage = delta + gamma * gae_lambda * not_done * advantage
            returns[step] = values[step] + advantage
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1.0e-8)
        return returns, advantages

    @staticmethod
    def _gaussian_kl(
        new_mean: torch.Tensor,
        new_std: torch.Tensor,
        old_std: torch.Tensor,
    ) -> torch.Tensor:
        return (
            torch.log(new_std / old_std + 1.0e-5)
            + (old_std.square() + new_mean.square()) / (2.0 * new_std.square())
            - 0.5
        ).sum(-1)

    def _adapt_learning_rate(self, kl: float) -> None:
        target = float(self.cfg.desired_kl)
        if target <= 0.0:
            return
        for group in self._optimizer.param_groups:
            if kl > 2.0 * target:
                group["lr"] = max(float(group["lr"]) / 1.5, 1.0e-6)
            elif kl < 0.5 * target:
                group["lr"] = min(float(group["lr"]) * 1.5, 1.0e-2)
        self.actor_learning_rate = float(self._optimizer.param_groups[0]["lr"])
        self.critic_learning_rate = float(self._optimizer.param_groups[1]["lr"])

    def update(self, rollout: dict, collect_time: float) -> dict:
        update_start = time.perf_counter()
        steps = self.rollout_steps
        num_envs = self.env.num_envs
        batch_size = steps * num_envs
        flatten = lambda tensor: tensor.reshape(batch_size, -1)
        actor_obs = flatten(rollout["actor_obs"])
        critic_obs = flatten(rollout["critic_obs"])
        base_noise = flatten(rollout["base_noise"])
        prior = flatten(rollout["prior"])
        delta = flatten(rollout["delta"])
        old_std = flatten(rollout["action_std"])
        old_log_prob = flatten(rollout["old_log_prob"]).squeeze(-1)
        old_values = flatten(rollout["values"])
        returns = flatten(rollout["returns"])
        advantages = flatten(rollout["advantages"]).squeeze(-1)

        epochs = int(self.cfg.num_learning_epochs)
        mini_batches = int(self.cfg.num_mini_batches)
        mini_batch_size = batch_size // mini_batches
        if mini_batch_size < 1:
            raise ValueError("PolicyFlow mini-batch count exceeds rollout batch size")
        totals = {
            "policy": 0.0,
            "value": 0.0,
            "entropy": 0.0,
            "brownian": 0.0,
            "kl": 0.0,
            "ratio": 0.0,
            "clip_fraction": 0.0,
            "grad_norm": 0.0,
            "delta_velocity": 0.0,
        }
        optimization_steps = 0
        all_kls: list[float] = []
        clip_range = float(self.cfg.clip_range)
        value_clip = float(self.cfg.value_clip_range)

        probe_count = min(128, num_envs)
        probe_obs = rollout["actor_obs"][0, :probe_count]
        with torch.no_grad():
            probe_before = self.actor.deterministic(probe_obs)
            parameters_before = [
                parameter.detach().clone()
                for parameter in self.actor.parameters()
                if parameter.requires_grad
            ]

        for _ in range(epochs):
            permutation = torch.randperm(batch_size, device=self.device)
            for mini_batch in range(mini_batches):
                index = permutation[
                    mini_batch * mini_batch_size : (mini_batch + 1) * mini_batch_size
                ]
                delta_velocity, new_std, brownian = self.actor.flow_variation(
                    actor_obs[index],
                    prior[index],
                    base_noise[index],
                    compute_brownian=float(self.cfg.brownian_reg_coef) > 0.0,
                )
                distribution = torch.distributions.Normal(delta_velocity, new_std)
                new_log_prob = distribution.log_prob(delta[index]).sum(-1)
                ratio = torch.exp(new_log_prob - old_log_prob[index])
                unclipped = advantages[index] * ratio
                clipped = advantages[index] * ratio.clamp(
                    1.0 - clip_range, 1.0 + clip_range
                )
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                entropy = distribution.entropy().sum(-1).mean()
                brownian_loss = float(self.cfg.brownian_reg_coef) * brownian

                predicted_values = self.critic.evaluate(critic_obs[index])
                clipped_values = old_values[index] + (
                    predicted_values - old_values[index]
                ).clamp(-value_clip, value_clip)
                value_loss = torch.maximum(
                    (predicted_values - returns[index]).square(),
                    (clipped_values - returns[index]).square(),
                ).mean()
                total_loss = (
                    policy_loss
                    - float(self.cfg.gaussian_entropy_coef) * entropy
                    + brownian_loss
                    + float(self.cfg.value_loss_coef) * value_loss
                )

                self._optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                trainable_parameters = [
                    parameter
                    for group in self._optimizer.param_groups
                    for parameter in group["params"]
                ]
                grad_norm = nn.utils.clip_grad_norm_(
                    trainable_parameters, float(self.cfg.max_grad_norm)
                )
                self._optimizer.step()

                with torch.no_grad():
                    kl = self._gaussian_kl(delta_velocity, new_std, old_std[index]).mean()
                    clip_fraction = ((ratio - 1.0).abs() > clip_range).float().mean()
                kl_value = float(kl.item())
                all_kls.append(kl_value)
                totals["policy"] += float(policy_loss.item())
                totals["value"] += float(value_loss.item())
                totals["entropy"] += float(entropy.item())
                totals["brownian"] += float(brownian_loss.item())
                totals["kl"] += kl_value
                totals["ratio"] += float(ratio.mean().item())
                totals["clip_fraction"] += float(clip_fraction.item())
                totals["grad_norm"] += float(
                    grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                )
                totals["delta_velocity"] += float(delta_velocity.abs().mean().item())
                optimization_steps += 1
            self._adapt_learning_rate(sum(all_kls) / max(len(all_kls), 1))

        for key in totals:
            totals[key] /= max(optimization_steps, 1)
        with torch.no_grad():
            probe_after = self.actor.deterministic(probe_obs)
            action_delta = float((probe_after - probe_before).abs().mean().item())
            squared_delta = torch.zeros((), device=self.device)
            parameter_count = 0
            for parameter, before in zip(
                (p for p in self.actor.parameters() if p.requires_grad),
                parameters_before,
                strict=True,
            ):
                squared_delta += (parameter.detach() - before).square().sum()
                parameter_count += parameter.numel()
            parameter_rms_delta = float(
                torch.sqrt(squared_delta / max(parameter_count, 1)).item()
            )
        self.actor.snapshot_last()

        metrics = {
            "policyflow/policy_loss": totals["policy"],
            "policyflow/value_loss": totals["value"],
            "policyflow/gaussian_entropy": totals["entropy"],
            "policyflow/brownian_loss": totals["brownian"],
            "policyflow/kl": totals["kl"],
            "policyflow/ratio_mean": totals["ratio"],
            "policyflow/clip_fraction": totals["clip_fraction"],
            "policyflow/delta_velocity_abs": totals["delta_velocity"],
            "policyflow/grad_norm": totals["grad_norm"],
            "policyflow/noise_std": float(self.actor.std.detach().mean().item()),
            "policyflow/actor_lr": self.actor_learning_rate,
            "policyflow/critic_lr": self.critic_learning_rate,
            "policyflow/optimization_steps": float(optimization_steps),
            "budget/physical_transitions": float(steps * num_envs),
            "budget/policy_decisions": float(steps * num_envs),
            "params/actor_trainable": float(self._trainable_counts["actor"]),
            "params/critic_trainable": float(self._trainable_counts["critic"]),
            "rollout/reward_step_mean": float(rollout["rewards"].mean().item()),
            "rollout/done_frac": float(rollout["dones"].float().mean().item()),
            "act/rollout_abs_mean": float(rollout["actions"].abs().mean().item()),
            "act/rollout_abs_max": float(rollout["actions"].abs().max().item()),
            "policy/action_delta": action_delta,
            "policy/param_rms_delta": parameter_rms_delta,
            "timing/collect_s": float(collect_time),
            "timing/update_s": float(time.perf_counter() - update_start),
            "train/mean_reward": (
                float(sum(self._reward_history) / len(self._reward_history))
                if self._reward_history
                else float("nan")
            ),
            "train/mean_episode_length": (
                float(sum(self._length_history) / len(self._length_history))
                if self._length_history
                else float("nan")
            ),
            "train/completed_episodes": float(self._completed_episodes),
        }
        for key, value in rollout["done_means"].items():
            metrics[f"done_rollout/{key}_frac"] = float(value)
        for key, value in rollout["reward_means"].items():
            metrics[f"reward_rollout/{key}_mean"] = float(value)
        sampler = self.env.adaptive_sampling_stats()
        for key in ("top_bin", "top_prob", "failed_sum", "entropy", "peak_bin"):
            metrics[f"sampler/{key}"] = float(sampler.get(key, float("nan")))
        return metrics

    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        normalized = self._norm_actor(obs, update=False)
        return self.actor.deterministic(normalized).unsqueeze(1)

    def extra_checkpoint_state(self) -> dict:
        return {
            "actor_obs_normalizer": (
                self.actor_obs_normalizer.state_dict() if self.empirical_normalization else None
            ),
            "critic_obs_normalizer": (
                self.critic_obs_normalizer.state_dict() if self.empirical_normalization else None
            ),
            "actor_learning_rate": self.actor_learning_rate,
            "critic_learning_rate": self.critic_learning_rate,
        }

    def load_extra_checkpoint_state(self, payload: dict, reset_optimizer: bool = False) -> None:
        if not payload:
            return
        if self.empirical_normalization:
            if payload.get("actor_obs_normalizer") is not None:
                self.actor_obs_normalizer.load_state_dict(payload["actor_obs_normalizer"])
            if payload.get("critic_obs_normalizer") is not None:
                self.critic_obs_normalizer.load_state_dict(payload["critic_obs_normalizer"])
        if not reset_optimizer:
            self.actor_learning_rate = float(
                payload.get("actor_learning_rate", self.actor_learning_rate)
            )
            self.critic_learning_rate = float(
                payload.get("critic_learning_rate", self.critic_learning_rate)
            )

    def log(self, update_idx: int, max_updates: int, metrics: dict) -> None:
        print(
            f"[UPDATE] {update_idx}/{max_updates} "
            f"reward_step={metrics['rollout/reward_step_mean']:.5f} "
            f"done_frac={metrics['rollout/done_frac']:.5f} "
            f"mean_reward={metrics['train/mean_reward']:.5f} "
            f"mean_len={metrics['train/mean_episode_length']:.2f}",
            flush=True,
        )
        print(
            f"[POLICYFLOW] policy={metrics['policyflow/policy_loss']:.5f} "
            f"value={metrics['policyflow/value_loss']:.5f} "
            f"entropy={metrics['policyflow/gaussian_entropy']:.5f} "
            f"brownian={metrics['policyflow/brownian_loss']:.5f} "
            f"kl={metrics['policyflow/kl']:.6f} ratio={metrics['policyflow/ratio_mean']:.5f} "
            f"clip_frac={metrics['policyflow/clip_fraction']:.4f}",
            flush=True,
        )
        print(
            f"[POLICYFLOW_FIELD] delta_v_abs={metrics['policyflow/delta_velocity_abs']:.6f} "
            f"noise_std={metrics['policyflow/noise_std']:.5f} "
            f"actor_lr={metrics['policyflow/actor_lr']:.6f} "
            f"critic_lr={metrics['policyflow/critic_lr']:.6f} "
            f"grad={metrics['policyflow/grad_norm']:.5f}",
            flush=True,
        )
        print(
            f"[BUDGET] physical_transitions={metrics['budget/physical_transitions']:.0f} "
            f"policy_decisions={metrics['budget/policy_decisions']:.0f} "
            f"optimizer_steps={metrics['policyflow/optimization_steps']:.0f}",
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
        print("[INFO] Starting official-objective PolicyFlow training", flush=True)
        print(
            f"[INFO] algorithm=policyflow actor_obs_dim={self.obs_dim} "
            f"critic_obs_dim={self.critic_obs_dim} action_dim={self.action_dim} "
            f"horizon=1 num_envs={env.num_envs} rollout_env_steps={self.rollout_steps} "
            f"flow_steps={cfg.flow_steps} integrator=midpoint "
            f"brownian_reg_coef={cfg.brownian_reg_coef} "
            f"gaussian_entropy_coef={cfg.gaussian_entropy_coef} "
            f"epochs={cfg.num_learning_epochs} mini_batches={cfg.num_mini_batches} "
            f"actor_lr={cfg.actor_learning_rate} critic_lr={cfg.critic_learning_rate}",
            flush=True,
        )
        print(
            f"[INFO] trainable_params actor={self._trainable_counts['actor']} "
            f"critic={self._trainable_counts['critic']}",
            flush=True,
        )

