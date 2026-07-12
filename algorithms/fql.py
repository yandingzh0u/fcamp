from __future__ import annotations

import copy
import time
from collections import deque
from itertools import chain

import torch
from torch import nn

from algorithms.base import Algorithm
from algorithms.flowrl import TensorReplayBuffer, soft_update
from core.offline_dataset import load_offline_transition_dataset
from networks.fql import FQLTwinQ, FQLVectorField
from networks.mlp_actor_critic import EmpiricalNormalization


def aggregate_twin_q(
    q1: torch.Tensor,
    q2: torch.Tensor,
    aggregation: str,
) -> torch.Tensor:
    if aggregation == "min":
        return torch.minimum(q1, q2)
    if aggregation == "mean":
        return 0.5 * (q1 + q2)
    raise ValueError(f"FQL q_aggregation must be 'mean' or 'min', got {aggregation!r}")


FQL_STEP_METRICS = (
    "critic_loss",
    "bc_flow_loss",
    "distill_loss",
    "q_loss",
    "actor_loss",
    "q_mean",
    "q_min",
    "q_max",
    "actor_q",
    "q_abs_mean",
    "q_scale",
    "td_target",
    "policy_mse",
    "target_actor_gap",
    "target_flow_abs",
    "actor_action_abs",
    "velocity_rms",
    "critic_grad",
    "bc_flow_grad",
    "one_step_grad",
    "target_critic_rms_gap",
)


class FQL(Algorithm):
    """Offline-to-online port of seohongpark/fql for continuous robot control."""

    def build(self) -> None:
        cfg = self.cfg
        env = self.env
        if int(cfg.horizon) != 1:
            raise ValueError(f"Official FQL uses one-step actions, got horizon={cfg.horizon}")
        self.obs_dim = int(env.observation_dim)
        self.action_dim = int(env.action_dim)
        self.rollout_steps = int(cfg.rollout_env_steps)
        self.device = torch.device(env.device)

        self.bc_flow = FQLVectorField(
            self.obs_dim,
            self.action_dim,
            tuple(cfg.actor_hidden_dims),
            cfg.activation,
            bool(cfg.actor_layer_norm),
            time_conditioned=True,
        ).to(self.device)
        self.one_step_actor = FQLVectorField(
            self.obs_dim,
            self.action_dim,
            tuple(cfg.actor_hidden_dims),
            cfg.activation,
            bool(cfg.actor_layer_norm),
            time_conditioned=False,
        ).to(self.device)
        self.critic = FQLTwinQ(
            self.obs_dim,
            self.action_dim,
            tuple(cfg.critic_hidden_dims),
            cfg.activation,
            bool(cfg.critic_layer_norm),
        ).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)
        self.target_critic.requires_grad_(False)

        self.bc_flow_optimizer = torch.optim.Adam(
            self.bc_flow.parameters(),
            lr=float(cfg.flow_lr),
            weight_decay=float(cfg.weight_decay),
        )
        self.one_step_optimizer = torch.optim.Adam(
            self.one_step_actor.parameters(),
            lr=float(cfg.policy_lr),
            weight_decay=float(cfg.weight_decay),
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(cfg.critic_lr),
            weight_decay=float(cfg.critic_weight_decay),
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
        self.offline_dataset_metadata: dict = {}
        self.offline_seed_transitions = 0
        self.offline_pretrain_samples = 0
        self._offline_pretrain_metrics: dict[str, float] = {}
        self._offline_pretrain_time_last = 0.0
        self._offline_pretraining_done = int(cfg.offline_pretrain_gradient_steps) == 0
        self._load_offline_replay()
        self.total_env_frames = 0
        self.total_transitions = 0
        self.gradient_step = 0
        self._init_episode_stats()
        self._policy_module = nn.ModuleDict(
            {
                "bc_flow": self.bc_flow,
                "one_step_actor": self.one_step_actor,
                "critic": self.critic,
                "target_critic": self.target_critic,
            }
        )
        self._trainable_counts = {
            "bc_flow": sum(p.numel() for p in self.bc_flow.parameters() if p.requires_grad),
            "one_step_actor": sum(
                p.numel() for p in self.one_step_actor.parameters() if p.requires_grad
            ),
            "critic": sum(p.numel() for p in self.critic.parameters() if p.requires_grad),
        }

    def _load_offline_replay(self) -> None:
        dataset_path = str(self.cfg.offline_dataset_path)
        if not dataset_path:
            return
        tensors, metadata = load_offline_transition_dataset(
            dataset_path,
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            action_limit=float(self.cfg.action_scale),
        )
        dataset_environment_scale = float(metadata.get("environment_action_scale", 1.0))
        expected_environment_scale = float(self.cfg.environment_action_scale)
        if abs(dataset_environment_scale - expected_environment_scale) > 1e-6:
            raise ValueError(
                f"Offline dataset environment_action_scale={dataset_environment_scale} does not "
                f"match FQL config value {expected_environment_scale}"
            )
        action_coordinate = str(metadata.get("action_coordinate", "environment_action"))
        if action_coordinate not in {"normalized_fql_action", "environment_action"}:
            raise ValueError(f"Unsupported offline action_coordinate {action_coordinate!r}")
        if action_coordinate == "environment_action" and expected_environment_scale != 1.0:
            raise ValueError(
                "Datasets in environment_action coordinates are only valid when "
                "environment_action_scale=1.0"
            )
        transition_count = int(tensors["observations"].shape[0])
        if transition_count > self.replay.capacity:
            raise ValueError(
                f"Offline dataset has {transition_count} transitions but replay_capacity="
                f"{self.replay.capacity}; refusing to discard offline data"
            )
        transfer_batch = min(65536, transition_count)
        for start in range(0, transition_count, transfer_batch):
            stop = min(start + transfer_batch, transition_count)
            obs = tensors["observations"][start:stop].to(self.device)
            if self.empirical_normalization:
                self._norm(obs, update=True)
            self.replay.add_batch(
                obs,
                tensors["actions"][start:stop].to(self.device),
                tensors["rewards"][start:stop].to(self.device),
                tensors["next_observations"][start:stop].to(self.device),
                tensors["masks"][start:stop].to(self.device),
                is_offline=True,
            )
        self.offline_dataset_metadata = metadata
        self.offline_seed_transitions = transition_count
        print(
            f"[FQL_OFFLINE_LOAD] path={metadata['path']} transitions={transition_count} "
            f"teacher={metadata.get('teacher_algorithm', 'unknown')} "
            f"action_limit={metadata['observed_action_limit']:.6f}",
            flush=True,
        )

    @property
    def policy(self) -> nn.Module:
        return self._policy_module

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.one_step_optimizer

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
        self._offline_pretrain_time_last = 0.0
        if not self._offline_pretraining_done:
            if update_idx == 1:
                self._run_offline_pretraining()
            else:
                self._offline_pretraining_done = True
                print(
                    f"[FQL_OFFLINE_PRETRAIN] skipped at update={update_idx}; "
                    "assuming a resumed checkpoint already contains pretrained weights.",
                    flush=True,
                )
        return self._obs

    def _run_offline_pretraining(self) -> None:
        gradient_steps = int(self.cfg.offline_pretrain_gradient_steps)
        if gradient_steps < 1:
            self._offline_pretraining_done = True
            return
        if self.offline_seed_transitions < 1:
            raise RuntimeError("FQL offline pretraining requested without loaded offline transitions")

        started = time.perf_counter()
        totals = {key: 0.0 for key in FQL_STEP_METRICS}
        sampled_transitions = 0
        sampled_offline = 0
        print(
            f"[FQL_OFFLINE_PRETRAIN_START] steps={gradient_steps} "
            f"dataset_transitions={self.offline_seed_transitions} "
            f"batch_size={min(int(self.cfg.replay_batch_size), self.replay.size)}",
            flush=True,
        )
        for step in range(gradient_steps):
            sample_size = min(int(self.cfg.replay_batch_size), self.replay.size)
            sampled = self.replay.sample(sample_size, include_source=True)
            output = self._joint_gradient_step(sampled[:5])
            for key in FQL_STEP_METRICS:
                totals[key] += output[key]
            sampled_transitions += sample_size
            sampled_offline += int(sampled[5].sum().item())
            self.gradient_step += 1
            should_log = (
                step == 0
                or step + 1 == gradient_steps
                or (step + 1) % int(self.cfg.offline_pretrain_log_every) == 0
            )
            if should_log:
                print(
                    f"[FQL_OFFLINE_PRETRAIN] step={step + 1}/{gradient_steps} "
                    f"critic={output['critic_loss']:.5f} cfm={output['bc_flow_loss']:.5f} "
                    f"distill={output['distill_loss']:.5f} q_loss={output['q_loss']:.5f}",
                    flush=True,
                )

        elapsed = time.perf_counter() - started
        denominator = max(gradient_steps, 1)
        self._offline_pretrain_metrics = {
            f"fql_pretrain/{key}": value / denominator for key, value in totals.items()
        }
        self._offline_pretrain_metrics.update(
            {
                "fql_pretrain/gradient_steps": float(gradient_steps),
                "fql_pretrain/samples": float(sampled_transitions),
                "fql_pretrain/offline_sample_fraction": float(
                    sampled_offline / max(sampled_transitions, 1)
                ),
                "fql_pretrain/time_s": float(elapsed),
            }
        )
        self.offline_pretrain_samples += sampled_transitions
        self._offline_pretrain_time_last = elapsed
        self._offline_pretraining_done = True
        print(
            f"[FQL_OFFLINE_PRETRAIN_DONE] steps={gradient_steps} samples={sampled_transitions} "
            f"offline_fraction={sampled_offline / max(sampled_transitions, 1):.4f} "
            f"time={elapsed:.3f}s",
            flush=True,
        )

    def _one_step_actions(
        self,
        normalized_obs: torch.Tensor,
        *,
        deterministic: bool = False,
        noises: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noises is None:
            if deterministic:
                noises = torch.zeros(
                    normalized_obs.shape[0],
                    self.action_dim,
                    device=normalized_obs.device,
                    dtype=normalized_obs.dtype,
                )
            else:
                noises = torch.randn(
                    normalized_obs.shape[0],
                    self.action_dim,
                    device=normalized_obs.device,
                    dtype=normalized_obs.dtype,
                )
        actions = self.one_step_actor(normalized_obs, noises)
        scale = float(self.cfg.action_scale)
        return actions.clamp(-scale, scale), noises

    @torch.no_grad()
    def compute_flow_actions(
        self,
        normalized_obs: torch.Tensor,
        noises: torch.Tensor,
    ) -> torch.Tensor:
        actions = noises
        flow_steps = int(self.cfg.flow_steps)
        dt = 1.0 / flow_steps
        for step in range(flow_steps):
            time_tensor = torch.full(
                (normalized_obs.shape[0], 1),
                step / flow_steps,
                device=normalized_obs.device,
                dtype=normalized_obs.dtype,
            )
            actions = actions + dt * self.bc_flow(normalized_obs, actions, time_tensor)
        scale = float(self.cfg.action_scale)
        return actions.clamp(-scale, scale)

    def collect(self, current_obs: torch.Tensor) -> dict:
        env = self.env
        num_envs = env.num_envs
        actions_buffer = torch.empty(
            self.rollout_steps, num_envs, self.action_dim, device=self.device
        )
        policy_actions_buffer = torch.empty_like(actions_buffer)
        rewards_buffer = torch.empty(self.rollout_steps, num_envs, device=self.device)
        dones_buffer = torch.empty(
            self.rollout_steps, num_envs, dtype=torch.bool, device=self.device
        )
        done_sums: dict[str, float] = {}
        reward_sums: dict[str, float] = {}
        warmup_actions = 0
        policy_actions = 0
        effective_warmup_steps = (
            0 if self.offline_pretrain_samples > 0 else int(self.cfg.warmup_env_steps)
        )
        obs = self._obs

        with torch.no_grad():
            for step in range(self.rollout_steps):
                normalized_obs = self._norm(obs, update=True)
                if self.total_env_frames + step < effective_warmup_steps:
                    scale = float(self.cfg.action_scale)
                    policy_action = scale * (2.0 * torch.rand(
                        num_envs, self.action_dim, device=self.device
                    ) - 1.0)
                    warmup_actions += num_envs
                else:
                    policy_action, _ = self._one_step_actions(normalized_obs)
                    policy_actions += num_envs
                action = policy_action * float(self.cfg.environment_action_scale)

                next_obs, reward, done, info = env.step(action, auto_reset=True)
                timeout = info["done_terms"]["time_out"].bool()
                replay_next_obs = next_obs.clone()
                if bool(timeout.any()) and "final_observation" in info:
                    replay_next_obs[timeout] = info["final_observation"][timeout]
                bootstrap_mask = ((~done.bool()) | timeout).float().unsqueeze(-1)
                self.replay.add_batch(
                    obs,
                    policy_action,
                    reward.unsqueeze(-1),
                    replay_next_obs,
                    bootstrap_mask,
                )
                actions_buffer[step] = action
                policy_actions_buffer[step] = policy_action
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
            "normalized_actions": policy_actions_buffer,
            "rewards": rewards_buffer,
            "dones": dones_buffer,
            "done_means": {key: value / self.rollout_steps for key, value in done_sums.items()},
            "reward_means": {key: value / self.rollout_steps for key, value in reward_sums.items()},
            "warmup_actions": warmup_actions,
            "policy_actions": policy_actions,
            "effective_warmup_steps": effective_warmup_steps,
            "next_observation": obs,
        }

    @staticmethod
    def _grad_norm(module: nn.Module) -> float:
        squared = torch.zeros((), device=next(module.parameters()).device)
        for parameter in module.parameters():
            if parameter.grad is not None:
                squared = squared + parameter.grad.detach().pow(2).sum()
        return float(squared.sqrt().item())

    @torch.no_grad()
    def _target_distance(self) -> float:
        squared = torch.zeros((), device=self.device)
        count = 0
        for target, online in zip(
            self.target_critic.parameters(), self.critic.parameters(), strict=True
        ):
            squared += (target - online).pow(2).sum()
            count += target.numel()
        return float((squared / max(count, 1)).sqrt().item())

    def _joint_gradient_step(self, batch: tuple[torch.Tensor, ...]) -> dict[str, float]:
        obs, replay_action, reward, next_obs, mask = batch
        obs_n = self._norm(obs, update=False)
        next_obs_n = self._norm(next_obs, update=False)

        with torch.no_grad():
            next_action, _ = self._one_step_actions(next_obs_n)
            target_q1, target_q2 = self.target_critic(next_obs_n, next_action)
            next_q = aggregate_twin_q(target_q1, target_q2, str(self.cfg.q_aggregation))
            td_target = reward + float(self.cfg.discount_gamma) * mask * next_q

        q1, q2 = self.critic(obs_n, replay_action)
        critic_loss = 0.5 * ((q1 - td_target).pow(2).mean() + (q2 - td_target).pow(2).mean())

        base_noise = torch.randn_like(replay_action)
        cfm_time = torch.rand(replay_action.shape[0], 1, device=self.device)
        interpolated_action = (1.0 - cfm_time) * base_noise + cfm_time * replay_action
        target_velocity = replay_action - base_noise
        predicted_velocity = self.bc_flow(obs_n, interpolated_action, cfm_time)
        bc_flow_loss = (predicted_velocity - target_velocity).pow(2).mean()

        distill_noise = torch.randn_like(replay_action)
        target_flow_action = self.compute_flow_actions(obs_n, distill_noise)
        raw_actor_action = self.one_step_actor(obs_n, distill_noise)
        distill_loss = (raw_actor_action - target_flow_action).pow(2).mean()

        self.critic.requires_grad_(False)
        scale = float(self.cfg.action_scale)
        actor_action = raw_actor_action.clamp(-scale, scale)
        actor_q1, actor_q2 = self.critic(obs_n, actor_action)
        actor_q = 0.5 * (actor_q1 + actor_q2)
        self.critic.requires_grad_(True)
        q_scale = torch.ones((), device=self.device)
        if bool(self.cfg.normalize_q_loss):
            q_scale = (1.0 / actor_q.detach().abs().mean().clamp_min(1e-6)).detach()
        q_loss = -q_scale * actor_q.mean()
        actor_loss = bc_flow_loss + float(self.cfg.alpha) * distill_loss + q_loss

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.bc_flow_optimizer.zero_grad(set_to_none=True)
        self.one_step_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        actor_loss.backward()
        critic_grad = self._grad_norm(self.critic)
        bc_flow_grad = self._grad_norm(self.bc_flow)
        one_step_grad = self._grad_norm(self.one_step_actor)
        nn.utils.clip_grad_norm_(self.critic.parameters(), float(self.cfg.max_grad_norm))
        nn.utils.clip_grad_norm_(
            chain(self.bc_flow.parameters(), self.one_step_actor.parameters()),
            float(self.cfg.max_grad_norm),
        )
        self.critic_optimizer.step()
        self.bc_flow_optimizer.step()
        self.one_step_optimizer.step()
        soft_update(self.target_critic, self.critic, float(self.cfg.target_tau))

        with torch.no_grad():
            sampled_action, _ = self._one_step_actions(obs_n)
            policy_mse = (sampled_action - replay_action).pow(2).mean()
            target_actor_gap = (actor_action - target_flow_action).pow(2).mean()
        return {
            "critic_loss": float(critic_loss.item()),
            "bc_flow_loss": float(bc_flow_loss.item()),
            "distill_loss": float(distill_loss.item()),
            "q_loss": float(q_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "q_mean": float(0.5 * (q1.mean().item() + q2.mean().item())),
            "q_min": float(torch.minimum(q1, q2).min().item()),
            "q_max": float(torch.maximum(q1, q2).max().item()),
            "actor_q": float(actor_q.mean().item()),
            "q_abs_mean": float(actor_q.abs().mean().item()),
            "q_scale": float(q_scale.item()),
            "td_target": float(td_target.mean().item()),
            "policy_mse": float(policy_mse.item()),
            "target_actor_gap": float(target_actor_gap.item()),
            "target_flow_abs": float(target_flow_action.abs().mean().item()),
            "actor_action_abs": float(actor_action.abs().mean().item()),
            "velocity_rms": float(predicted_velocity.pow(2).mean().sqrt().item()),
            "critic_grad": critic_grad,
            "bc_flow_grad": bc_flow_grad,
            "one_step_grad": one_step_grad,
            "target_critic_rms_gap": self._target_distance(),
        }

    def update(self, rollout: dict, collect_time: float) -> dict:
        update_start = time.perf_counter()
        totals = {key: 0.0 for key in FQL_STEP_METRICS}
        sampled_transitions = 0
        sampled_offline = 0
        gradient_steps = int(self.cfg.gradient_steps_per_update)
        for _ in range(gradient_steps):
            sample_size = min(int(self.cfg.replay_batch_size), self.replay.size)
            sampled = self.replay.sample(sample_size, include_source=True)
            output = self._joint_gradient_step(sampled[:5])
            for key in FQL_STEP_METRICS:
                totals[key] += output[key]
            sampled_transitions += sample_size
            sampled_offline += int(sampled[5].sum().item())
            self.gradient_step += 1

        denominator = max(gradient_steps, 1)
        metrics = {f"fql/{key}": value / denominator for key, value in totals.items()}
        final_batch_size = min(int(self.cfg.replay_batch_size), self.replay.size)
        replay_offline, replay_online = self.replay.source_counts()
        metrics.update(
            {
                "fql/gradient_steps": float(gradient_steps),
                "fql/flow_lr": float(self.bc_flow_optimizer.param_groups[0]["lr"]),
                "fql/policy_lr": float(self.one_step_optimizer.param_groups[0]["lr"]),
                "fql/critic_lr": float(self.critic_optimizer.param_groups[0]["lr"]),
                "fql/replay_size": float(self.replay.size),
                "fql/replay_capacity": float(self.replay.capacity),
                "fql/replay_utilization": float(self.replay.size / self.replay.capacity),
                "fql/replay_offline_transitions": float(replay_offline),
                "fql/replay_online_transitions": float(replay_online),
                "fql/update_offline_sample_fraction": float(
                    sampled_offline / max(sampled_transitions, 1)
                ),
                "budget/physical_transitions": float(self.rollout_steps * self.env.num_envs),
                "budget/policy_decisions": float(self.rollout_steps * self.env.num_envs),
                "budget/replay_samples": float(sampled_transitions),
                "budget/flow_training_evals": float(sampled_transitions),
                "budget/flow_target_evals": float(sampled_transitions * int(self.cfg.flow_steps)),
                "budget/onestep_training_evals": float(2 * sampled_transitions),
                "budget/warmup_actions": float(rollout["warmup_actions"]),
                "budget/rollout_policy_actions": float(rollout["policy_actions"]),
                "budget/offline_seed_transitions": float(self.offline_seed_transitions),
                "budget/offline_pretrain_samples": float(self.offline_pretrain_samples),
                "params/bc_flow_trainable": float(self._trainable_counts["bc_flow"]),
                "params/one_step_actor_trainable": float(self._trainable_counts["one_step_actor"]),
                "params/critic_trainable": float(self._trainable_counts["critic"]),
                "rollout/reward_step_mean": float(rollout["rewards"].mean().item()),
                "rollout/done_frac": float(rollout["dones"].float().mean().item()),
                "act/rollout_abs_mean": float(rollout["actions"].abs().mean().item()),
                "act/rollout_abs_max": float(rollout["actions"].abs().max().item()),
                "act/normalized_abs_mean": float(
                    rollout["normalized_actions"].abs().mean().item()
                ),
                "act/normalized_abs_max": float(
                    rollout["normalized_actions"].abs().max().item()
                ),
                "timing/collect_s": float(collect_time),
                "timing/collect_env_s": float(
                    max(0.0, collect_time - self._offline_pretrain_time_last)
                ),
                "timing/offline_pretrain_s": float(self._offline_pretrain_time_last),
                "timing/update_s": float(time.perf_counter() - update_start),
                "fql/current_batch_size": float(final_batch_size),
                "fql/effective_warmup_steps": float(rollout["effective_warmup_steps"]),
            }
        )
        metrics.update(self._offline_pretrain_metrics)
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
        action, _ = self._one_step_actions(normalized_obs, deterministic=True)
        return (action * float(self.cfg.environment_action_scale)).unsqueeze(1)

    def extra_checkpoint_state(self) -> dict:
        return {
            "bc_flow_optimizer": self.bc_flow_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "obs_normalizer": self.obs_normalizer.state_dict() if self.empirical_normalization else None,
            "total_env_frames": self.total_env_frames,
            "total_transitions": self.total_transitions,
            "gradient_step": self.gradient_step,
            "replay_size_at_save": self.replay.size,
            "offline_seed_transitions": self.offline_seed_transitions,
            "offline_pretrain_samples": self.offline_pretrain_samples,
            "offline_pretraining_done": self._offline_pretraining_done,
            "offline_dataset_path": self.offline_dataset_metadata.get("path", ""),
        }

    def load_extra_checkpoint_state(self, payload: dict, reset_optimizer: bool = False) -> None:
        if not payload:
            return
        if not reset_optimizer:
            if "bc_flow_optimizer" in payload:
                self.bc_flow_optimizer.load_state_dict(payload["bc_flow_optimizer"])
            if "critic_optimizer" in payload:
                self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        if self.empirical_normalization and payload.get("obs_normalizer") is not None:
            self.obs_normalizer.load_state_dict(payload["obs_normalizer"])
        self.total_env_frames = int(payload.get("total_env_frames", self.total_env_frames))
        self.total_transitions = int(payload.get("total_transitions", self.total_transitions))
        self.gradient_step = int(payload.get("gradient_step", self.gradient_step))
        self.offline_pretrain_samples = int(
            payload.get("offline_pretrain_samples", self.offline_pretrain_samples)
        )
        self._offline_pretraining_done = bool(
            payload.get("offline_pretraining_done", self._offline_pretraining_done)
        )
        saved_replay_size = int(payload.get("replay_size_at_save", 0))
        if saved_replay_size:
            print(
                f"[FQL] online replay contents are not checkpointed; saved buffer had "
                f"{saved_replay_size} entries and the resumed buffer was reseeded with "
                f"{self.offline_seed_transitions} offline transitions.",
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
            f"[FQL_CRITIC] loss={metrics['fql/critic_loss']:.5f} "
            f"q={metrics['fql/q_mean']:.4f} "
            f"range=[{metrics['fql/q_min']:.4f},{metrics['fql/q_max']:.4f}] "
            f"target={metrics['fql/td_target']:.4f} "
            f"target_gap={metrics['fql/target_critic_rms_gap']:.6f}",
            flush=True,
        )
        print(
            f"[FQL_ACTOR] total={metrics['fql/actor_loss']:.5f} "
            f"cfm={metrics['fql/bc_flow_loss']:.5f} "
            f"distill={metrics['fql/distill_loss']:.5f} "
            f"q_loss={metrics['fql/q_loss']:.5f} actor_q={metrics['fql/actor_q']:.4f} "
            f"q_scale={metrics['fql/q_scale']:.4f}",
            flush=True,
        )
        print(
            f"[FQL_FLOW] target_actor_gap={metrics['fql/target_actor_gap']:.5f} "
            f"target_abs={metrics['fql/target_flow_abs']:.4f} "
            f"actor_abs={metrics['fql/actor_action_abs']:.4f} "
            f"velocity_rms={metrics['fql/velocity_rms']:.4f} "
            f"policy_data_mse={metrics['fql/policy_mse']:.5f}",
            flush=True,
        )
        print(
            f"[FQL_GRAD_LR] flow_grad={metrics['fql/bc_flow_grad']:.4f} "
            f"policy_grad={metrics['fql/one_step_grad']:.4f} "
            f"critic_grad={metrics['fql/critic_grad']:.4f} "
            f"flow_lr={metrics['fql/flow_lr']:.6f} "
            f"policy_lr={metrics['fql/policy_lr']:.6f} "
            f"critic_lr={metrics['fql/critic_lr']:.6f}",
            flush=True,
        )
        print(
            f"[BUDGET] physical_transitions={metrics['budget/physical_transitions']:.0f} "
            f"policy_decisions={metrics['budget/policy_decisions']:.0f} "
            f"replay_samples={metrics['budget/replay_samples']:.0f} "
            f"offline_seed={metrics['budget/offline_seed_transitions']:.0f} "
            f"offline_pretrain_samples={metrics['budget/offline_pretrain_samples']:.0f} "
            f"flow_train_evals={metrics['budget/flow_training_evals']:.0f} "
            f"flow_target_evals={metrics['budget/flow_target_evals']:.0f} "
            f"onestep_evals={metrics['budget/onestep_training_evals']:.0f}",
            flush=True,
        )
        print(
            f"[REPLAY] size={metrics['fql/replay_size']:.0f}/"
            f"{metrics['fql/replay_capacity']:.0f} "
            f"util={metrics['fql/replay_utilization']:.4f} "
            f"offline={metrics['fql/replay_offline_transitions']:.0f} "
            f"online={metrics['fql/replay_online_transitions']:.0f} "
            f"batch_offline_frac={metrics['fql/update_offline_sample_fraction']:.4f} "
            f"warmup_actions={metrics['budget/warmup_actions']:.0f} "
            f"policy_actions={metrics['budget/rollout_policy_actions']:.0f}",
            flush=True,
        )
        print(
            f"[TIME] collect_total={metrics['timing/collect_s']:.3f}s "
            f"collect_env={metrics['timing/collect_env_s']:.3f}s "
            f"offline_pretrain={metrics['timing/offline_pretrain_s']:.3f}s "
            f"update={metrics['timing/update_s']:.3f}s",
            flush=True,
        )

    def log_banner(self) -> None:
        cfg = self.cfg
        env = self.env
        protocol = "offline-to-online" if self.offline_seed_transitions else "online-replay"
        print(f"[INFO] Starting FQL {protocol} training", flush=True)
        print(
            f"[INFO] task={env.task.name} terrain={env.task.terrain} motion_file={env.task.motion_file}",
            flush=True,
        )
        print(
            f"[INFO] algorithm=fql source_policy=one_step_flow actor_obs_dim={self.obs_dim} "
            f"action_dim={self.action_dim} horizon={self.horizon} num_envs={env.num_envs} "
            f"rollout_env_steps={self.rollout_steps} flow_steps={cfg.flow_steps} integrator=euler "
            f"normalized_action_limit={cfg.action_scale} "
            f"environment_action_scale={cfg.environment_action_scale} "
            f"alpha={cfg.alpha} normalize_q_loss={cfg.normalize_q_loss} "
            f"q_aggregation={cfg.q_aggregation} discount={cfg.discount_gamma} "
            f"target_tau={cfg.target_tau} replay_capacity={cfg.replay_capacity} "
            f"replay_batch_size={cfg.replay_batch_size} "
            f"gradient_steps_per_update={cfg.gradient_steps_per_update} "
            f"flow_lr={cfg.flow_lr} policy_lr={cfg.policy_lr} critic_lr={cfg.critic_lr} "
            f"actor_hidden_dims={list(cfg.actor_hidden_dims)} "
            f"critic_hidden_dims={list(cfg.critic_hidden_dims)}",
            flush=True,
        )
        if self.offline_seed_transitions:
            print(
                f"[INFO] fql_protocol=offline_to_online dataset="
                f"{self.offline_dataset_metadata.get('path')} "
                f"offline_transitions={self.offline_seed_transitions} "
                f"offline_pretrain_gradient_steps={cfg.offline_pretrain_gradient_steps} "
                f"online_warmup_steps=0 teacher="
                f"{self.offline_dataset_metadata.get('teacher_algorithm', 'unknown')}",
                flush=True,
            )
        else:
            print(
                f"[INFO] fql_protocol=online_replay_only offline_dataset=none "
                f"random_warmup_physical_steps={cfg.warmup_env_steps}",
                flush=True,
            )
        print(
            f"[INFO] trainable_params bc_flow={self._trainable_counts['bc_flow']} "
            f"one_step_actor={self._trainable_counts['one_step_actor']} "
            f"critic={self._trainable_counts['critic']}",
            flush=True,
        )
