from __future__ import annotations

import copy
import time
from collections import deque

import torch
from torch import nn

from algorithms.base import Algorithm
from networks.flowrl import FlowRLActor, FlowRLTwinQ, FlowRLValue
from networks.mlp_actor_critic import EmpiricalNormalization


class TensorReplayBuffer:
    """Device-local ring buffer with an optional recent-sample fraction."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        action_dim: int,
        device: torch.device | str,
        recent_fraction: float,
        recent_window: int,
    ) -> None:
        self.capacity = int(capacity)
        self.recent_fraction = float(recent_fraction)
        self.recent_window = int(recent_window)
        self.obs = torch.empty(self.capacity, obs_dim, device=device)
        self.actions = torch.empty(self.capacity, action_dim, device=device)
        self.rewards = torch.empty(self.capacity, 1, device=device)
        self.next_obs = torch.empty(self.capacity, obs_dim, device=device)
        self.masks = torch.empty(self.capacity, 1, device=device)
        self.is_offline = torch.empty(self.capacity, dtype=torch.bool, device=device)
        self.position = 0
        self.size = 0

    @torch.no_grad()
    def add_batch(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        masks: torch.Tensor,
        is_offline: bool | torch.Tensor = False,
    ) -> None:
        batch = obs.shape[0]
        if isinstance(is_offline, torch.Tensor):
            source = is_offline.to(device=self.obs.device, dtype=torch.bool).reshape(-1)
            if source.shape[0] != batch:
                raise ValueError(
                    f"Replay source labels must have shape [{batch}], got {tuple(source.shape)}"
                )
        else:
            source = torch.full((batch,), bool(is_offline), device=self.obs.device, dtype=torch.bool)
        if batch >= self.capacity:
            obs = obs[-self.capacity :]
            actions = actions[-self.capacity :]
            rewards = rewards[-self.capacity :]
            next_obs = next_obs[-self.capacity :]
            masks = masks[-self.capacity :]
            source = source[-self.capacity :]
            batch = self.capacity
        first = min(batch, self.capacity - self.position)
        second = batch - first
        target = slice(self.position, self.position + first)
        self.obs[target].copy_(obs[:first])
        self.actions[target].copy_(actions[:first])
        self.rewards[target].copy_(rewards[:first])
        self.next_obs[target].copy_(next_obs[:first])
        self.masks[target].copy_(masks[:first])
        self.is_offline[target].copy_(source[:first])
        if second:
            target = slice(0, second)
            self.obs[target].copy_(obs[first:])
            self.actions[target].copy_(actions[first:])
            self.rewards[target].copy_(rewards[first:])
            self.next_obs[target].copy_(next_obs[first:])
            self.masks[target].copy_(masks[first:])
            self.is_offline[target].copy_(source[first:])
        self.position = (self.position + batch) % self.capacity
        self.size = min(self.capacity, self.size + batch)

    def sample(
        self,
        batch_size: int,
        *,
        include_source: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        if self.size < 1:
            raise RuntimeError("Cannot sample an empty replay buffer")
        batch_size = min(int(batch_size), self.size)
        recent_count = min(int(round(batch_size * self.recent_fraction)), batch_size)
        uniform_count = batch_size - recent_count
        indices: list[torch.Tensor] = []
        if uniform_count:
            indices.append(torch.randint(self.size, (uniform_count,), device=self.obs.device))
        if recent_count:
            window = min(self.recent_window, self.size)
            offsets = torch.randint(window, (recent_count,), device=self.obs.device)
            newest = (self.position - 1) % self.capacity
            indices.append((newest - offsets) % self.capacity)
        index = torch.cat(indices, dim=0)
        batch = (
            self.obs[index],
            self.actions[index],
            self.rewards[index],
            self.next_obs[index],
            self.masks[index],
        )
        if include_source:
            return (*batch, self.is_offline[index])
        return batch

    def source_counts(self) -> tuple[int, int]:
        valid = self.is_offline[: self.size]
        offline = int(valid.sum().item())
        return offline, int(self.size - offline)


def expectile_loss(error: torch.Tensor, expectile: float) -> tuple[torch.Tensor, torch.Tensor]:
    weight = torch.where(error >= 0.0, expectile, 1.0 - expectile)
    return (weight * error.pow(2)).mean(), weight


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_parameter, source_parameter in zip(target.parameters(), source.parameters(), strict=True):
        target_parameter.mul_(1.0 - tau).add_(source_parameter, alpha=tau)


class FlowRL(Algorithm):
    """Online off-policy FlowRL port from bytedance/FlowRL."""

    def build(self) -> None:
        cfg = self.cfg
        env = self.env
        if int(cfg.horizon) != 1:
            raise ValueError(f"Official FlowRL is a one-action policy, got horizon={cfg.horizon}")
        self.obs_dim = int(env.observation_dim)
        self.action_dim = int(env.action_dim)
        self.rollout_steps = int(cfg.rollout_env_steps)
        self.device = torch.device(env.device)

        self.actor = FlowRLActor(
            self.obs_dim,
            self.action_dim,
            tuple(cfg.actor_hidden_dims),
            cfg.activation,
            int(cfg.flow_steps),
            float(cfg.action_scale),
        ).to(self.device)
        self.online_q = FlowRLTwinQ(
            self.obs_dim, self.action_dim, tuple(cfg.critic_hidden_dims)
        ).to(self.device)
        self.target_q = copy.deepcopy(self.online_q).to(self.device)
        self.behavior_q = FlowRLTwinQ(
            self.obs_dim, self.action_dim, tuple(cfg.critic_hidden_dims)
        ).to(self.device)
        self.target_behavior_q = copy.deepcopy(self.behavior_q).to(self.device)
        self.value = FlowRLValue(self.obs_dim, tuple(cfg.critic_hidden_dims)).to(self.device)
        for module in (self.target_q, self.target_behavior_q):
            module.requires_grad_(False)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=float(cfg.policy_lr), weight_decay=float(cfg.weight_decay)
        )
        self.online_q_optimizer = torch.optim.Adam(
            self.online_q.parameters(), lr=float(cfg.critic_lr), weight_decay=float(cfg.critic_weight_decay)
        )
        self.behavior_q_optimizer = torch.optim.Adam(
            self.behavior_q.parameters(), lr=float(cfg.critic_lr), weight_decay=float(cfg.critic_weight_decay)
        )
        self.value_optimizer = torch.optim.Adam(
            self.value.parameters(), lr=float(cfg.value_lr), weight_decay=float(cfg.critic_weight_decay)
        )

        self.empirical_normalization = bool(cfg.empirical_normalization)
        if self.empirical_normalization:
            self.obs_normalizer: nn.Module = EmpiricalNormalization(self.obs_dim, self.device)
        else:
            self.obs_normalizer = nn.Identity()

        self.replay = TensorReplayBuffer(
            int(cfg.replay_capacity),
            self.obs_dim,
            self.action_dim,
            self.device,
            float(cfg.recent_fraction),
            int(cfg.recent_window),
        )
        self.total_env_frames = 0
        self.total_transitions = 0
        self.gradient_step = 0
        self._init_episode_stats()
        self._policy_module = nn.ModuleDict(
            {
                "actor": self.actor,
                "online_q": self.online_q,
                "target_q": self.target_q,
                "behavior_q": self.behavior_q,
                "target_behavior_q": self.target_behavior_q,
                "value": self.value,
            }
        )
        self._trainable_counts = {
            "actor": sum(parameter.numel() for parameter in self.actor.parameters() if parameter.requires_grad),
            "online_q": sum(parameter.numel() for parameter in self.online_q.parameters() if parameter.requires_grad),
            "behavior_q": sum(parameter.numel() for parameter in self.behavior_q.parameters() if parameter.requires_grad),
            "value": sum(parameter.numel() for parameter in self.value.parameters() if parameter.requires_grad),
        }

    @property
    def policy(self) -> nn.Module:
        return self._policy_module

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.actor_optimizer

    @property
    def horizon(self) -> int:
        return int(self.cfg.horizon)

    def _norm(self, obs: torch.Tensor, update: bool) -> torch.Tensor:
        if self.empirical_normalization:
            return self.obs_normalizer(obs, update=update)
        return obs

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
        obs = self.env.reset()
        if bool(self.cfg.init_at_random_ep_len) and self.env.max_episode_steps > 0:
            self.env.episode_steps = torch.randint_like(
                self.env.episode_steps, high=int(self.env.max_episode_steps)
            )
        self._obs = obs
        return obs

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        return self._obs

    def collect(self, current_obs: torch.Tensor) -> dict:
        env = self.env
        num_envs = env.num_envs
        actions_buffer = torch.empty(
            self.rollout_steps, num_envs, self.action_dim, device=self.device
        )
        rewards_buffer = torch.empty(self.rollout_steps, num_envs, device=self.device)
        dones_buffer = torch.empty(
            self.rollout_steps, num_envs, dtype=torch.bool, device=self.device
        )
        done_sums: dict[str, float] = {}
        reward_sums: dict[str, float] = {}
        warmup_actions = 0
        obs = self._obs

        with torch.no_grad():
            for step in range(self.rollout_steps):
                normalized_obs = self._norm(obs, update=True)
                if self.total_env_frames + step < int(self.cfg.warmup_env_steps):
                    action = (2.0 * torch.rand(num_envs, self.action_dim, device=self.device) - 1.0)
                    action = action * float(self.cfg.action_scale)
                    warmup_actions += num_envs
                else:
                    action, _ = self.actor.sample(normalized_obs)
                    if float(self.cfg.exploration_noise) > 0.0:
                        noise = torch.rand_like(action) * 0.01 * float(self.cfg.exploration_noise)
                        action = action + noise.clamp(-0.25, 0.25)
                    action = action.clamp(-float(self.cfg.action_scale), float(self.cfg.action_scale))

                next_obs, reward, done, info = env.step(action, auto_reset=True)
                timeout = info["done_terms"]["time_out"].bool()
                replay_next_obs = next_obs.clone()
                if bool(timeout.any()) and "final_observation" in info:
                    replay_next_obs[timeout] = info["final_observation"][timeout]
                bootstrap_mask = ((~done.bool()) | timeout).float().unsqueeze(-1)
                self.replay.add_batch(
                    obs,
                    action,
                    reward.unsqueeze(-1),
                    replay_next_obs,
                    bootstrap_mask,
                )
                actions_buffer[step] = action
                rewards_buffer[step] = reward
                dones_buffer[step] = done.bool()
                self._record_episode_stats(reward, done.bool())
                for key, value in info["done_terms"].items():
                    done_sums[key] = done_sums.get(key, 0.0) + float(value.float().mean().item())
                for key, value in info["reward_terms"].items():
                    reward_sums[key] = reward_sums.get(key, 0.0) + float(value.mean().item())
                obs = next_obs

        self.total_env_frames += self.rollout_steps
        self.total_transitions += self.rollout_steps * num_envs
        self._obs = obs
        return {
            "actions": actions_buffer,
            "rewards": rewards_buffer,
            "dones": dones_buffer,
            "done_means": {key: value / self.rollout_steps for key, value in done_sums.items()},
            "reward_means": {key: value / self.rollout_steps for key, value in reward_sums.items()},
            "warmup_actions": warmup_actions,
            "next_observation": obs,
        }

    def _step_optimizer(
        self,
        module: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss: torch.Tensor,
    ) -> float:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(module.parameters(), float(self.cfg.max_grad_norm))
        optimizer.step()
        return float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)

    def _update_critics(self, batch: tuple[torch.Tensor, ...]) -> dict[str, float | torch.Tensor]:
        obs, action, reward, next_obs, mask = batch
        obs_n = self._norm(obs, update=False)
        next_obs_n = self._norm(next_obs, update=False)
        gamma = float(self.cfg.discount_gamma)

        with torch.no_grad():
            next_action, _ = self.actor.sample(next_obs_n)
            target_q1, target_q2 = self.target_q(next_obs_n, next_action)
            td_target = reward + gamma * mask * torch.minimum(target_q1, target_q2)
        q1, q2 = self.online_q(obs_n, action)
        online_q_loss = (q1 - td_target).pow(2).mean() + (q2 - td_target).pow(2).mean()
        online_q_grad = self._step_optimizer(self.online_q, self.online_q_optimizer, online_q_loss)

        with torch.no_grad():
            behavior_target = reward + gamma * mask * self.value(next_obs_n)
        behavior_q1, behavior_q2 = self.behavior_q(obs_n, action)
        q_buffer = torch.minimum(behavior_q1, behavior_q2).detach()
        behavior_q_loss = (behavior_q1 - behavior_target).pow(2).mean()
        behavior_q_loss = behavior_q_loss + (behavior_q2 - behavior_target).pow(2).mean()
        behavior_q_grad = self._step_optimizer(
            self.behavior_q, self.behavior_q_optimizer, behavior_q_loss
        )

        with torch.no_grad():
            target_behavior_q1, target_behavior_q2 = self.target_behavior_q(obs_n, action)
            target_behavior_q = torch.minimum(target_behavior_q1, target_behavior_q2)
        value_prediction = self.value(obs_n)
        value_error = target_behavior_q - value_prediction
        value_loss, expectile_weight = expectile_loss(value_error, float(self.cfg.expectile))
        value_grad = self._step_optimizer(self.value, self.value_optimizer, value_loss)
        return {
            "online_q_loss": float(online_q_loss.item()),
            "behavior_q_loss": float(behavior_q_loss.item()),
            "value_loss": float(value_loss.item()),
            "online_q_grad": online_q_grad,
            "behavior_q_grad": behavior_q_grad,
            "value_grad": value_grad,
            "td_target": float(td_target.mean().item()),
            "behavior_target": float(behavior_target.mean().item()),
            "expectile_high_frac": float((value_error >= 0.0).float().mean().item()),
            "expectile_weight": float(expectile_weight.mean().item()),
            "q_buffer": q_buffer,
            "obs_n": obs_n,
            "action": action,
        }

    def _update_actor(self, critic_output: dict[str, float | torch.Tensor]) -> dict[str, float]:
        obs_n = critic_output["obs_n"]
        action = critic_output["action"]
        q_buffer = critic_output["q_buffer"]
        assert isinstance(obs_n, torch.Tensor)
        assert isinstance(action, torch.Tensor)
        assert isinstance(q_buffer, torch.Tensor)

        self.online_q.requires_grad_(False)
        policy_action, _ = self.actor.sample(obs_n)
        q1_pi, q2_pi = self.online_q(obs_n, policy_action)
        min_q_pi = torch.minimum(q1_pi, q2_pi)
        q_gap = torch.relu(q_buffer - min_q_pi.detach())
        weights = torch.exp(q_gap - q_gap.mean()) * float(self.cfg.w2_lambda)
        weights = weights.clamp(float(self.cfg.cfm_weight_min), float(self.cfg.cfm_weight_max))

        base_noise = torch.randn_like(action).clamp(-1.0, 1.0)
        cfm_time = torch.rand(action.shape[0], 1, device=self.device)
        cfm_per_sample = self.actor.cfm_loss(obs_n, action, base_noise, cfm_time)
        cfm_scalar = cfm_per_sample.mean()
        weighted_cfm = (weights * cfm_scalar).mean()
        actor_loss = (-min_q_pi + weights * cfm_scalar).mean()
        actor_grad = self._step_optimizer(self.actor, self.actor_optimizer, actor_loss)
        self.online_q.requires_grad_(True)
        return {
            "actor_loss": float(actor_loss.item()),
            "actor_grad": actor_grad,
            "q_pi": float(min_q_pi.mean().item()),
            "q_buffer": float(q_buffer.mean().item()),
            "q_gap": float(q_gap.mean().item()),
            "cfm_loss": float(cfm_scalar.item()),
            "weighted_cfm": float(weighted_cfm.item()),
            "cfm_weight_mean": float(weights.mean().item()),
            "cfm_weight_min": float(weights.min().item()),
            "cfm_weight_max": float(weights.max().item()),
        }

    def update(self, rollout: dict, collect_time: float) -> dict:
        update_start = time.perf_counter()
        keys = (
            "online_q_loss", "behavior_q_loss", "value_loss", "online_q_grad",
            "behavior_q_grad", "value_grad", "td_target", "behavior_target",
            "expectile_high_frac", "expectile_weight",
        )
        totals = {key: 0.0 for key in keys}
        actor_keys = (
            "actor_loss", "actor_grad", "q_pi", "q_buffer", "q_gap", "cfm_loss",
            "weighted_cfm", "cfm_weight_mean", "cfm_weight_min", "cfm_weight_max",
        )
        actor_totals = {key: 0.0 for key in actor_keys}
        actor_updates = 0
        sampled_transitions = 0
        gradient_steps = int(self.cfg.gradient_steps_per_update)
        for _ in range(gradient_steps):
            sample_size = min(int(self.cfg.replay_batch_size), self.replay.size)
            batch = self.replay.sample(sample_size)
            critic_output = self._update_critics(batch)
            for key in keys:
                totals[key] += float(critic_output[key])
            sampled_transitions += sample_size
            if self.gradient_step % int(self.cfg.policy_delay) == 0:
                actor_output = self._update_actor(critic_output)
                for key in actor_keys:
                    actor_totals[key] += actor_output[key]
                actor_updates += 1
                soft_update(self.target_q, self.online_q, float(self.cfg.target_tau))
                soft_update(self.target_behavior_q, self.behavior_q, float(self.cfg.target_tau))
            self.gradient_step += 1

        denominator = max(gradient_steps, 1)
        actor_denominator = max(actor_updates, 1)
        metrics = {f"flowrl/{key}": value / denominator for key, value in totals.items()}
        metrics.update(
            {f"flowrl/{key}": value / actor_denominator for key, value in actor_totals.items()}
        )
        metrics.update(
            {
                "flowrl/actor_updates": float(actor_updates),
                "flowrl/critic_updates": float(gradient_steps),
                "flowrl/actor_lr": float(self.actor_optimizer.param_groups[0]["lr"]),
                "flowrl/online_q_lr": float(self.online_q_optimizer.param_groups[0]["lr"]),
                "flowrl/behavior_q_lr": float(self.behavior_q_optimizer.param_groups[0]["lr"]),
                "flowrl/value_lr": float(self.value_optimizer.param_groups[0]["lr"]),
                "flowrl/replay_size": float(self.replay.size),
                "flowrl/replay_capacity": float(self.replay.capacity),
                "flowrl/replay_utilization": float(self.replay.size / self.replay.capacity),
                "budget/physical_transitions": float(self.rollout_steps * self.env.num_envs),
                "budget/policy_decisions": float(self.rollout_steps * self.env.num_envs),
                "budget/replay_samples": float(sampled_transitions),
                "budget/actor_replay_samples": float(
                    actor_updates * min(int(self.cfg.replay_batch_size), self.replay.size)
                ),
                "budget/warmup_actions": float(rollout["warmup_actions"]),
                "params/actor_trainable": float(self._trainable_counts["actor"]),
                "params/online_q_trainable": float(self._trainable_counts["online_q"]),
                "params/behavior_q_trainable": float(self._trainable_counts["behavior_q"]),
                "params/value_trainable": float(self._trainable_counts["value"]),
                "rollout/reward_step_mean": float(rollout["rewards"].mean().item()),
                "rollout/done_frac": float(rollout["dones"].float().mean().item()),
                "act/rollout_abs_mean": float(rollout["actions"].abs().mean().item()),
                "act/rollout_abs_max": float(rollout["actions"].abs().max().item()),
                "timing/collect_s": float(collect_time),
                "timing/update_s": float(time.perf_counter() - update_start),
            }
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
        normalized_obs = self._norm(obs, update=False)
        action, _ = self.actor.sample(normalized_obs, deterministic=True)
        return action.unsqueeze(1)

    def extra_checkpoint_state(self) -> dict:
        return {
            "online_q_optimizer": self.online_q_optimizer.state_dict(),
            "behavior_q_optimizer": self.behavior_q_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "obs_normalizer": self.obs_normalizer.state_dict() if self.empirical_normalization else None,
            "total_env_frames": self.total_env_frames,
            "total_transitions": self.total_transitions,
            "gradient_step": self.gradient_step,
            "replay_size_at_save": self.replay.size,
        }

    def load_extra_checkpoint_state(self, payload: dict, reset_optimizer: bool = False) -> None:
        if not payload:
            return
        if not reset_optimizer:
            for name, optimizer in (
                ("online_q_optimizer", self.online_q_optimizer),
                ("behavior_q_optimizer", self.behavior_q_optimizer),
                ("value_optimizer", self.value_optimizer),
            ):
                if name in payload:
                    optimizer.load_state_dict(payload[name])
        if self.empirical_normalization and payload.get("obs_normalizer") is not None:
            self.obs_normalizer.load_state_dict(payload["obs_normalizer"])
        self.total_env_frames = int(payload.get("total_env_frames", self.total_env_frames))
        self.total_transitions = int(payload.get("total_transitions", self.total_transitions))
        self.gradient_step = int(payload.get("gradient_step", self.gradient_step))
        saved_replay_size = int(payload.get("replay_size_at_save", 0))
        if saved_replay_size:
            print(
                f"[FLOWRL] replay contents are intentionally not checkpointed; "
                f"discarded {saved_replay_size} saved entries and refill online.",
                flush=True,
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
            f"[FLOWRL_Q] online={metrics['flowrl/online_q_loss']:.5f} "
            f"behavior={metrics['flowrl/behavior_q_loss']:.5f} "
            f"value={metrics['flowrl/value_loss']:.5f} "
            f"td_target={metrics['flowrl/td_target']:.4f} "
            f"behavior_target={metrics['flowrl/behavior_target']:.4f} "
            f"expectile_high={metrics['flowrl/expectile_high_frac']:.3f}",
            flush=True,
        )
        print(
            f"[FLOWRL_ACTOR] loss={metrics['flowrl/actor_loss']:.5f} "
            f"q_pi={metrics['flowrl/q_pi']:.4f} q_buffer={metrics['flowrl/q_buffer']:.4f} "
            f"q_gap={metrics['flowrl/q_gap']:.4f} cfm={metrics['flowrl/cfm_loss']:.5f} "
            f"w2={metrics['flowrl/weighted_cfm']:.5f} "
            f"weight={metrics['flowrl/cfm_weight_mean']:.4f} "
            f"[{metrics['flowrl/cfm_weight_min']:.4f},{metrics['flowrl/cfm_weight_max']:.4f}]",
            flush=True,
        )
        print(
            f"[FLOWRL_LR] actor={metrics['flowrl/actor_lr']:.6f} "
            f"online_q={metrics['flowrl/online_q_lr']:.6f} "
            f"behavior_q={metrics['flowrl/behavior_q_lr']:.6f} "
            f"value={metrics['flowrl/value_lr']:.6f}",
            flush=True,
        )
        print(
            f"[BUDGET] physical_transitions={metrics['budget/physical_transitions']:.0f} "
            f"policy_decisions={metrics['budget/policy_decisions']:.0f} "
            f"replay_samples={metrics['budget/replay_samples']:.0f} "
            f"actor_replay_samples={metrics['budget/actor_replay_samples']:.0f} "
            f"critic_steps={metrics['flowrl/critic_updates']:.0f} "
            f"actor_steps={metrics['flowrl/actor_updates']:.0f}",
            flush=True,
        )
        print(
            f"[REPLAY] size={metrics['flowrl/replay_size']:.0f}/"
            f"{metrics['flowrl/replay_capacity']:.0f} "
            f"util={metrics['flowrl/replay_utilization']:.4f} "
            f"warmup_actions={metrics['budget/warmup_actions']:.0f}",
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
        print("[INFO] Starting ByteDance FlowRL training", flush=True)
        print(
            f"[INFO] task={env.task.name} terrain={env.task.terrain} motion_file={env.task.motion_file}",
            flush=True,
        )
        print(
            f"[INFO] algorithm=flowrl actor_obs_dim={self.obs_dim} action_dim={self.action_dim} "
            f"horizon={self.horizon} num_envs={env.num_envs} rollout_env_steps={self.rollout_steps} "
            f"flow_steps={cfg.flow_steps} integrator=midpoint action_scale={cfg.action_scale} "
            f"replay_capacity={cfg.replay_capacity} replay_batch_size={cfg.replay_batch_size} "
            f"gradient_steps_per_update={cfg.gradient_steps_per_update} policy_delay={cfg.policy_delay} "
            f"expectile={cfg.expectile} target_tau={cfg.target_tau} w2_lambda={cfg.w2_lambda} "
            f"actor_lr={cfg.policy_lr} critic_lr={cfg.critic_lr} value_lr={cfg.value_lr} "
            f"actor_hidden_dims={list(cfg.actor_hidden_dims)} "
            f"critic_hidden_dims={list(cfg.critic_hidden_dims)}",
            flush=True,
        )
        print(
            f"[INFO] trainable_params actor={self._trainable_counts['actor']} "
            f"online_q={self._trainable_counts['online_q']} "
            f"behavior_q={self._trainable_counts['behavior_q']} "
            f"value={self._trainable_counts['value']}",
            flush=True,
        )
