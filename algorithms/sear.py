from __future__ import annotations

import copy
import math
import time
from collections import deque

import torch
from torch import nn

from algorithms.base import Algorithm
from algorithms.flowrl import soft_update
from networks.mlp_actor_critic import EmpiricalNormalization
from networks.sear import SEARActor, SEARTwinCritic


def discounted_prefix_sum(values: torch.Tensor, gamma: float) -> torch.Tensor:
    horizon = values.shape[1]
    discounts = torch.pow(
        values.new_tensor(float(gamma)),
        torch.arange(horizon, device=values.device, dtype=values.dtype),
    )
    return (values * discounts).cumsum(dim=1)


class SequenceReplayBuffer:
    """Vector-environment ring buffer that samples contiguous action chunks."""

    def __init__(
        self,
        capacity: int,
        num_envs: int,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        device: torch.device | str,
    ) -> None:
        self.num_envs = int(num_envs)
        self.horizon = int(horizon)
        self.capacity_rows = max(self.horizon + 1, int(capacity) // self.num_envs)
        self.capacity = self.capacity_rows * self.num_envs
        shape = (self.capacity_rows, self.num_envs)
        self.obs = torch.empty(*shape, obs_dim, device=device)
        self.actions = torch.empty(*shape, action_dim, device=device)
        self.rewards = torch.empty(*shape, device=device)
        self.next_obs = torch.empty(*shape, obs_dim, device=device)
        self.masks = torch.empty(*shape, device=device)
        self.continuation = torch.empty(*shape, dtype=torch.bool, device=device)
        self.position = 0
        self.size_rows = 0

    @property
    def size(self) -> int:
        return self.size_rows * self.num_envs

    @torch.no_grad()
    def add_row(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_observation: torch.Tensor,
        mask: torch.Tensor,
        continuation: torch.Tensor,
    ) -> None:
        self.obs[self.position].copy_(observation)
        self.actions[self.position].copy_(action)
        self.rewards[self.position].copy_(reward)
        self.next_obs[self.position].copy_(next_observation)
        self.masks[self.position].copy_(mask)
        self.continuation[self.position].copy_(continuation)
        self.position = (self.position + 1) % self.capacity_rows
        self.size_rows = min(self.capacity_rows, self.size_rows + 1)

    def _physical_rows(self, logical_rows: torch.Tensor) -> torch.Tensor:
        oldest = self.position if self.size_rows == self.capacity_rows else 0
        return (oldest + logical_rows) % self.capacity_rows

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        valid_starts = self.size_rows - self.horizon + 1
        if valid_starts < 1:
            raise RuntimeError(
                f"SEAR replay needs at least {self.horizon} rows, has {self.size_rows}"
            )
        batch_size = int(batch_size)
        selected_rows: list[torch.Tensor] = []
        selected_envs: list[torch.Tensor] = []
        remaining = batch_size
        for _ in range(12):
            if remaining <= 0:
                break
            candidates = max(remaining * 2, 64)
            logical_start = torch.randint(
                valid_starts, (candidates,), device=self.obs.device
            )
            env_index = torch.randint(
                self.num_envs, (candidates,), device=self.obs.device
            )
            offsets = torch.arange(self.horizon, device=self.obs.device)
            rows = self._physical_rows(logical_start[:, None] + offsets[None, :])
            if self.horizon > 1:
                valid = self.continuation[
                    rows[:, :-1], env_index[:, None].expand(-1, self.horizon - 1)
                ].all(dim=1)
            else:
                valid = torch.ones(candidates, dtype=torch.bool, device=self.obs.device)
            rows = rows[valid][:remaining]
            env_index = env_index[valid][:remaining]
            if rows.numel():
                selected_rows.append(rows)
                selected_envs.append(env_index)
                remaining -= rows.shape[0]
        if remaining > 0:
            raise RuntimeError(
                "SEAR replay could not find enough within-episode contiguous chunks"
            )
        rows = torch.cat(selected_rows, dim=0)
        env_index = torch.cat(selected_envs, dim=0)
        expanded_env = env_index[:, None].expand(-1, self.horizon)
        return (
            self.obs[rows[:, 0], env_index],
            self.actions[rows, expanded_env],
            self.rewards[rows, expanded_env],
            self.next_obs[rows, expanded_env],
            self.masks[rows, expanded_env],
        )


class SEAR(Algorithm):
    """Pure-online SEAR with random replanning and multi-horizon chunk critics."""

    def build(self) -> None:
        cfg = self.cfg
        env = self.env
        self.device = torch.device(env.device)
        self.obs_dim = int(env.observation_dim)
        self.action_dim = int(env.action_dim)
        self._horizon = int(cfg.horizon)
        self.rollout_steps = int(cfg.rollout_env_steps)
        if self._horizon < 2:
            raise ValueError("SEAR comparison is configured as an action-chunk policy with h>=2")

        self.actor = SEARActor(
            self.obs_dim,
            self.action_dim,
            self._horizon,
            int(cfg.actor_hidden_dim),
            int(cfg.actor_num_blocks),
            float(cfg.action_scale),
        ).to(self.device)
        critic_kwargs = dict(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            horizon=self._horizon,
            hidden_dim=int(cfg.critic_hidden_dim),
            num_heads=int(cfg.critic_num_heads),
            num_blocks=int(cfg.critic_num_blocks),
            num_bins=int(cfg.num_value_bins),
            value_min=float(cfg.value_min),
            value_max=float(cfg.value_max),
        )
        self.critic = SEARTwinCritic(**critic_kwargs).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False).to(self.device)
        self.temperature = nn.ParameterDict(
            {
                "log_alpha": nn.Parameter(
                    torch.tensor(math.log(float(cfg.init_alpha)), device=self.device)
                )
            }
        )
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=float(cfg.actor_lr),
            weight_decay=float(cfg.weight_decay),
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=float(cfg.critic_lr),
            weight_decay=float(cfg.weight_decay),
        )
        self.alpha_optimizer = torch.optim.AdamW(
            self.temperature.parameters(), lr=float(cfg.alpha_lr), weight_decay=0.0
        )
        self.empirical_normalization = bool(cfg.empirical_normalization)
        if self.empirical_normalization:
            self.obs_normalizer: nn.Module = EmpiricalNormalization(self.obs_dim, self.device)
        else:
            self.obs_normalizer = nn.Identity()
        self.replay = SequenceReplayBuffer(
            int(cfg.replay_capacity),
            env.num_envs,
            self.obs_dim,
            self.action_dim,
            self._horizon,
            self.device,
        )
        self._queued_chunks = torch.zeros(
            env.num_envs,
            self._horizon,
            self.action_dim,
            device=self.device,
        )
        self._chunk_index = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
        self._prefix_remaining = torch.zeros_like(self._chunk_index)
        self.total_env_frames = 0
        self.total_transitions = 0
        self.gradient_step = 0
        self._init_episode_stats()
        self._policy_module = nn.ModuleDict(
            {
                "actor": self.actor,
                "critic": self.critic,
                "target_critic": self.target_critic,
                "temperature": self.temperature,
            }
        )
        self._trainable_counts = {
            "actor": sum(parameter.numel() for parameter in self.actor.parameters()),
            "critic": sum(parameter.numel() for parameter in self.critic.parameters()),
            "temperature": 1,
        }

    @property
    def alpha(self) -> torch.Tensor:
        return self.temperature["log_alpha"].exp()

    @property
    def policy(self) -> nn.Module:
        return self._policy_module

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.actor_optimizer

    @property
    def horizon(self) -> int:
        return self._horizon

    def _norm(self, observation: torch.Tensor, *, update: bool) -> torch.Tensor:
        if self.empirical_normalization:
            return self.obs_normalizer(observation, update=update)
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
        self._prefix_remaining.zero_()
        return observation

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        return self._obs

    def collect(self, current_obs: torch.Tensor) -> dict:
        env = self.env
        num_envs = env.num_envs
        actions = torch.empty(
            self.rollout_steps, num_envs, self.action_dim, device=self.device
        )
        rewards = torch.empty(self.rollout_steps, num_envs, device=self.device)
        dones = torch.empty(
            self.rollout_steps, num_envs, dtype=torch.bool, device=self.device
        )
        done_sums: dict[str, float] = {}
        reward_sums: dict[str, float] = {}
        policy_decisions = 0
        prefix_total = 0
        prefix_count = 0
        observation = self._obs
        with torch.no_grad():
            for step in range(self.rollout_steps):
                normalized = self._norm(observation, update=True)
                replan = self._prefix_remaining <= 0
                replan_ids = replan.nonzero(as_tuple=False).squeeze(-1)
                if replan_ids.numel():
                    new_chunks, _ = self.actor.sample(normalized.index_select(0, replan_ids))
                    prefix_lengths = torch.randint(
                        1,
                        self._horizon + 1,
                        (replan_ids.numel(),),
                        device=self.device,
                    )
                    self._queued_chunks[replan_ids] = new_chunks
                    self._chunk_index[replan_ids] = 0
                    self._prefix_remaining[replan_ids] = prefix_lengths
                    policy_decisions += int(replan_ids.numel())
                    prefix_total += int(prefix_lengths.sum().item())
                    prefix_count += int(prefix_lengths.numel())

                env_index = torch.arange(num_envs, device=self.device)
                action = self._queued_chunks[env_index, self._chunk_index]
                next_observation, reward, done, info = env.step(action, auto_reset=True)
                timeout = info["done_terms"]["time_out"].bool()
                replay_next_observation = next_observation.clone()
                if bool(timeout.any()) and "final_observation" in info:
                    replay_next_observation[timeout] = info["final_observation"][timeout]
                bootstrap_mask = ((~done.bool()) | timeout).float()
                self.replay.add_row(
                    observation,
                    action,
                    reward,
                    replay_next_observation,
                    bootstrap_mask,
                    ~done.bool(),
                )
                actions[step] = action
                rewards[step] = reward
                dones[step] = done.bool()
                self._record_episode_stats(reward, done.bool())
                for key, value in info["done_terms"].items():
                    done_sums[key] = done_sums.get(key, 0.0) + float(
                        value.float().mean().item()
                    )
                for key, value in info["reward_terms"].items():
                    reward_sums[key] = reward_sums.get(key, 0.0) + float(value.mean().item())

                self._chunk_index += 1
                self._prefix_remaining -= 1
                self._prefix_remaining[done.bool()] = 0
                observation = next_observation

        self.total_env_frames += self.rollout_steps
        self.total_transitions += self.rollout_steps * num_envs
        self._obs = observation
        return {
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "done_means": {
                key: value / self.rollout_steps for key, value in done_sums.items()
            },
            "reward_means": {
                key: value / self.rollout_steps for key, value in reward_sums.items()
            },
            "policy_decisions": policy_decisions,
            "prefix_mean": prefix_total / max(prefix_count, 1),
            "next_observation": observation,
        }

    def _critic_update(
        self, batch: tuple[torch.Tensor, ...]
    ) -> dict[str, float | torch.Tensor]:
        observation, action_chunk, rewards, next_observations, masks = batch
        batch_size = observation.shape[0]
        horizon = self._horizon
        gamma = float(self.cfg.discount_gamma)
        observation_n = self._norm(observation, update=False)
        next_observations_n = self._norm(
            next_observations.reshape(batch_size * horizon, self.obs_dim), update=False
        )
        with torch.no_grad():
            next_chunks, next_log_prob = self.actor.sample(next_observations_n)
            next_q1_logits, next_q2_logits = self.target_critic(
                next_observations_n, next_chunks
            )
            next_q1 = self.target_critic.expected(next_q1_logits).view(
                batch_size, horizon, horizon
            )
            next_q2 = self.target_critic.expected(next_q2_logits).view(
                batch_size, horizon, horizon
            )
            diagonal = torch.arange(horizon, device=self.device)
            next_q = torch.minimum(
                next_q1[:, diagonal, diagonal], next_q2[:, diagonal, diagonal]
            )
            next_log_prob = next_log_prob.view(batch_size, horizon, horizon)
            entropy_prefix = discounted_prefix_sum(
                next_log_prob.reshape(batch_size * horizon, horizon), gamma
            ).view(batch_size, horizon, horizon)
            entropy_prefix = entropy_prefix[:, diagonal, diagonal]
            soft_next_q = next_q - self.alpha.detach() * entropy_prefix
            reward_prefix = discounted_prefix_sum(rewards, gamma)
            bootstrap_mask = masks.cumprod(dim=1)
            gamma_powers = torch.pow(
                rewards.new_tensor(gamma),
                torch.arange(1, horizon + 1, device=self.device, dtype=rewards.dtype),
            )
            targets = reward_prefix + gamma_powers * bootstrap_mask * soft_next_q
            target_distribution = self.target_critic.target_distribution(targets)

        q1_logits, q2_logits = self.critic(observation_n, action_chunk)
        q1_loss = -(target_distribution * torch.log_softmax(q1_logits, dim=-1)).sum(-1).mean()
        q2_loss = -(target_distribution * torch.log_softmax(q2_logits, dim=-1)).sum(-1).mean()
        critic_loss = q1_loss + q2_loss
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad = nn.utils.clip_grad_norm_(
            self.critic.parameters(), float(self.cfg.max_grad_norm)
        )
        self.critic_optimizer.step()
        with torch.no_grad():
            q1_value = self.critic.expected(q1_logits)
            q2_value = self.critic.expected(q2_logits)
            clipped_fraction = (
                (targets <= float(self.cfg.value_min))
                | (targets >= float(self.cfg.value_max))
            ).float().mean()
        return {
            "critic_loss": float(critic_loss.item()),
            "critic_grad": float(
                critic_grad.item() if torch.is_tensor(critic_grad) else critic_grad
            ),
            "q_mean": float(torch.minimum(q1_value, q2_value).mean().item()),
            "target_mean": float(targets.mean().item()),
            "target_min": float(targets.min().item()),
            "target_max": float(targets.max().item()),
            "target_clipped_fraction": float(clipped_fraction.item()),
            "observation_n": observation_n,
        }

    def _actor_update(self, observation_n: torch.Tensor) -> dict[str, float]:
        gamma = float(self.cfg.discount_gamma)
        self.critic.requires_grad_(False)
        action_chunk, log_prob = self.actor.sample(observation_n)
        q1_logits, q2_logits = self.critic(observation_n, action_chunk)
        q1 = self.critic.expected(q1_logits)[:, -1]
        q2 = self.critic.expected(q2_logits)[:, -1]
        q_value = torch.minimum(q1, q2)
        entropy_term = discounted_prefix_sum(log_prob, gamma)[:, -1]
        actor_loss = (self.alpha.detach() * entropy_term - q_value).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = nn.utils.clip_grad_norm_(
            self.actor.parameters(), float(self.cfg.max_grad_norm)
        )
        self.actor_optimizer.step()
        self.critic.requires_grad_(True)

        joint_log_prob = log_prob.detach().sum(dim=1)
        target_entropy = -float(self.cfg.target_entropy_scale) * self._horizon * self.action_dim
        alpha_loss = -(
            self.temperature["log_alpha"] * (joint_log_prob + target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        return {
            "actor_loss": float(actor_loss.item()),
            "actor_grad": float(
                actor_grad.item() if torch.is_tensor(actor_grad) else actor_grad
            ),
            "q_pi": float(q_value.mean().item()),
            "log_prob": float(joint_log_prob.mean().item()),
            "alpha": float(self.alpha.detach().item()),
            "alpha_loss": float(alpha_loss.item()),
            "target_entropy": target_entropy,
        }

    def update(self, rollout: dict, collect_time: float) -> dict:
        update_start = time.perf_counter()
        critic_keys = (
            "critic_loss",
            "critic_grad",
            "q_mean",
            "target_mean",
            "target_min",
            "target_max",
            "target_clipped_fraction",
        )
        actor_keys = (
            "actor_loss",
            "actor_grad",
            "q_pi",
            "log_prob",
            "alpha",
            "alpha_loss",
            "target_entropy",
        )
        critic_totals = {key: 0.0 for key in critic_keys}
        actor_totals = {key: 0.0 for key in actor_keys}
        gradient_steps = int(self.cfg.gradient_steps_per_update)
        batch_size = min(int(self.cfg.replay_batch_size), self.replay.size)
        for _ in range(gradient_steps):
            batch = self.replay.sample(batch_size)
            critic_output = self._critic_update(batch)
            for key in critic_keys:
                critic_totals[key] += float(critic_output[key])
            actor_output = self._actor_update(critic_output["observation_n"])
            for key in actor_keys:
                actor_totals[key] += actor_output[key]
            soft_update(self.target_critic, self.critic, float(self.cfg.target_tau))
            self.gradient_step += 1

        for key in critic_totals:
            critic_totals[key] /= max(gradient_steps, 1)
        for key in actor_totals:
            actor_totals[key] /= max(gradient_steps, 1)
        replay_frames = gradient_steps * batch_size * self._horizon
        physical_transitions = self.rollout_steps * self.env.num_envs
        metrics = {f"sear/{key}": value for key, value in critic_totals.items()}
        metrics.update({f"sear/{key}": value for key, value in actor_totals.items()})
        metrics.update(
            {
                "sear/critic_updates": float(gradient_steps),
                "sear/actor_updates": float(gradient_steps),
                "sear/actor_lr": float(self.actor_optimizer.param_groups[0]["lr"]),
                "sear/critic_lr": float(self.critic_optimizer.param_groups[0]["lr"]),
                "sear/alpha_lr": float(self.alpha_optimizer.param_groups[0]["lr"]),
                "sear/replay_size": float(self.replay.size),
                "sear/replay_capacity": float(self.replay.capacity),
                "sear/replay_rows": float(self.replay.size_rows),
                "sear/random_prefix_mean": float(rollout["prefix_mean"]),
                "sear/utd_frames": float(replay_frames / max(physical_transitions, 1)),
                "budget/physical_transitions": float(physical_transitions),
                "budget/policy_decisions": float(rollout["policy_decisions"]),
                "budget/replay_sequences": float(gradient_steps * batch_size),
                "budget/replay_frames": float(replay_frames),
                "params/actor_trainable": float(self._trainable_counts["actor"]),
                "params/critic_trainable": float(self._trainable_counts["critic"]),
                "rollout/reward_step_mean": float(rollout["rewards"].mean().item()),
                "rollout/done_frac": float(rollout["dones"].float().mean().item()),
                "act/rollout_abs_mean": float(rollout["actions"].abs().mean().item()),
                "act/rollout_abs_max": float(rollout["actions"].abs().max().item()),
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
        )
        for key, value in rollout["done_means"].items():
            metrics[f"done_rollout/{key}_frac"] = float(value)
        for key, value in rollout["reward_means"].items():
            metrics[f"reward_rollout/{key}_mean"] = float(value)
        sampler = self.env.adaptive_sampling_stats()
        for key in ("top_bin", "top_prob", "failed_sum", "entropy", "peak_bin"):
            metrics[f"sampler/{key}"] = float(sampler.get(key, float("nan")))
        return metrics

    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        normalized = self._norm(obs, update=False)
        action_chunk, _ = self.actor.sample(normalized, deterministic=True)
        return action_chunk

    def extra_checkpoint_state(self) -> dict:
        return {
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "obs_normalizer": (
                self.obs_normalizer.state_dict() if self.empirical_normalization else None
            ),
            "total_env_frames": self.total_env_frames,
            "total_transitions": self.total_transitions,
            "gradient_step": self.gradient_step,
            "replay_size_at_save": self.replay.size,
        }

    def load_extra_checkpoint_state(self, payload: dict, reset_optimizer: bool = False) -> None:
        if not payload:
            return
        if not reset_optimizer:
            if "critic_optimizer" in payload:
                self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
            if "alpha_optimizer" in payload:
                self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
        if self.empirical_normalization and payload.get("obs_normalizer") is not None:
            self.obs_normalizer.load_state_dict(payload["obs_normalizer"])
        self.total_env_frames = int(payload.get("total_env_frames", self.total_env_frames))
        self.total_transitions = int(payload.get("total_transitions", self.total_transitions))
        self.gradient_step = int(payload.get("gradient_step", self.gradient_step))
        saved_replay_size = int(payload.get("replay_size_at_save", 0))
        if saved_replay_size:
            print(
                f"[SEAR] replay is not checkpointed; discarded {saved_replay_size} entries.",
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
            f"[SEAR_Q] loss={metrics['sear/critic_loss']:.5f} "
            f"q={metrics['sear/q_mean']:.4f} target={metrics['sear/target_mean']:.4f} "
            f"range=[{metrics['sear/target_min']:.4f},{metrics['sear/target_max']:.4f}] "
            f"clipped={metrics['sear/target_clipped_fraction']:.5f}",
            flush=True,
        )
        print(
            f"[SEAR_ACTOR] loss={metrics['sear/actor_loss']:.5f} "
            f"q_pi={metrics['sear/q_pi']:.4f} logp={metrics['sear/log_prob']:.4f} "
            f"alpha={metrics['sear/alpha']:.6f} "
            f"target_entropy={metrics['sear/target_entropy']:.1f}",
            flush=True,
        )
        print(
            f"[SEAR_CHUNK] horizon={self._horizon} "
            f"random_prefix_mean={metrics['sear/random_prefix_mean']:.3f} "
            f"policy_decisions={metrics['budget/policy_decisions']:.0f} "
            f"physical_transitions={metrics['budget/physical_transitions']:.0f}",
            flush=True,
        )
        print(
            f"[BUDGET] replay_sequences={metrics['budget/replay_sequences']:.0f} "
            f"replay_frames={metrics['budget/replay_frames']:.0f} "
            f"utd_frames={metrics['sear/utd_frames']:.3f} "
            f"critic_steps={metrics['sear/critic_updates']:.0f} "
            f"actor_steps={metrics['sear/actor_updates']:.0f}",
            flush=True,
        )
        print(
            f"[REPLAY] rows={metrics['sear/replay_rows']:.0f} "
            f"transitions={metrics['sear/replay_size']:.0f}/"
            f"{metrics['sear/replay_capacity']:.0f}",
            flush=True,
        )
        print(
            f"[TIME] collect={metrics['timing/collect_s']:.3f}s "
            f"update={metrics['timing/update_s']:.3f}s",
            flush=True,
        )

    def log_banner(self) -> None:
        cfg = self.cfg
        print("[INFO] Starting pure-online SEAR action-chunk training", flush=True)
        print(
            f"[INFO] algorithm=sear actor_obs_dim={self.obs_dim} action_dim={self.action_dim} "
            f"horizon={self._horizon} num_envs={self.env.num_envs} "
            f"rollout_env_steps={self.rollout_steps} random_replanning=true "
            f"critic=causal_distributional_transformer heads={cfg.critic_num_heads} "
            f"blocks={cfg.critic_num_blocks} bins={cfg.num_value_bins} "
            f"value_range=[{cfg.value_min},{cfg.value_max}] target_tau={cfg.target_tau} "
            f"replay_batch_size={cfg.replay_batch_size} "
            f"gradient_steps_per_update={cfg.gradient_steps_per_update}",
            flush=True,
        )
        print(
            f"[INFO] trainable_params actor={self._trainable_counts['actor']} "
            f"critic={self._trainable_counts['critic']} temperature=1",
            flush=True,
        )

