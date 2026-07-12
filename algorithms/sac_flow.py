from __future__ import annotations

import math
import time
from collections import deque

import torch
from torch import nn

from algorithms.base import Algorithm
from algorithms.flowrl import TensorReplayBuffer
from networks.mlp_actor_critic import EmpiricalNormalization
from networks.sac_flow import SACFlowActor, SACFlowTwinQ


class SACFlow(Algorithm):
    """From-scratch SAC Flow-G with the official CrossQ training path."""

    def build(self) -> None:
        cfg = self.cfg
        env = self.env
        if int(cfg.horizon) != 1:
            raise ValueError(f"From-scratch SAC Flow uses h=1, got {cfg.horizon}")
        self.device = torch.device(env.device)
        self.obs_dim = int(env.observation_dim)
        self.action_dim = int(env.action_dim)
        self.rollout_steps = int(cfg.rollout_env_steps)
        actor_hidden_dims = tuple(cfg.actor_hidden_dims)
        self.actor = SACFlowActor(
            self.obs_dim,
            self.action_dim,
            hidden_dim=int(actor_hidden_dims[0]),
            flow_steps=int(cfg.flow_steps),
            action_scale=float(cfg.action_scale),
            time_embed_dim=int(cfg.timestep_embed_dim),
            log_std_hidden_dims=actor_hidden_dims,
            use_batch_renorm=bool(cfg.use_batch_renorm),
            batch_norm_momentum=float(cfg.batch_norm_momentum),
        ).to(self.device)
        self.q_network = SACFlowTwinQ(
            self.obs_dim,
            self.action_dim,
            tuple(cfg.critic_hidden_dims),
            use_batch_renorm=bool(cfg.use_batch_renorm),
            batch_norm_momentum=float(cfg.batch_norm_momentum),
        ).to(self.device)

        self.temperature = nn.ParameterDict(
            {
                "log_alpha": nn.Parameter(
                    torch.tensor(math.log(float(cfg.init_alpha)), device=self.device)
                )
            }
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=float(cfg.policy_lr),
            betas=(0.5, 0.999),
            weight_decay=float(cfg.weight_decay),
        )
        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=float(cfg.critic_lr),
            betas=(0.5, 0.999),
            weight_decay=float(cfg.critic_weight_decay),
        )
        self.alpha_optimizer = torch.optim.Adam(
            self.temperature.parameters(),
            lr=float(cfg.alpha_lr),
            betas=(0.5, 0.999),
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
            recent_fraction=0.0,
            recent_window=1,
        )
        self.total_env_frames = 0
        self.total_transitions = 0
        self.gradient_step = 0
        self._init_episode_stats()
        self._policy_module = nn.ModuleDict(
            {
                "actor": self.actor,
                "q_network": self.q_network,
                "temperature": self.temperature,
            }
        )
        self._trainable_counts = {
            "actor": sum(parameter.numel() for parameter in self.actor.parameters()),
            "critic": sum(parameter.numel() for parameter in self.q_network.parameters()),
            "temperature": sum(parameter.numel() for parameter in self.temperature.parameters()),
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
        return 1

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
        warmup_actions = 0
        observation = self._obs
        with torch.no_grad():
            for step in range(self.rollout_steps):
                normalized = self._norm(observation, update=True)
                if self.total_env_frames + step < int(self.cfg.warmup_env_steps):
                    action = (
                        2.0 * torch.rand(num_envs, self.action_dim, device=self.device) - 1.0
                    ) * float(self.cfg.action_scale)
                    warmup_actions += num_envs
                else:
                    action, _, _ = self.actor.sample(normalized, update_stats=False)
                next_observation, reward, done, info = env.step(action, auto_reset=True)
                timeout = info["done_terms"]["time_out"].bool()
                replay_next_observation = next_observation.clone()
                if bool(timeout.any()) and "final_observation" in info:
                    replay_next_observation[timeout] = info["final_observation"][timeout]
                bootstrap_mask = ((~done.bool()) | timeout).float().unsqueeze(-1)
                self.replay.add_batch(
                    observation,
                    action,
                    reward.unsqueeze(-1),
                    replay_next_observation,
                    bootstrap_mask,
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
            "warmup_actions": warmup_actions,
            "next_observation": observation,
        }

    def _critic_update(
        self, batch: tuple[torch.Tensor, ...]
    ) -> dict[str, float]:
        observation, action, reward, next_observation, mask = batch
        observation_n = self._norm(observation, update=False)
        next_observation_n = self._norm(next_observation, update=False)
        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(
                next_observation_n, update_stats=False
            )

        # CrossQ performs one joint forward pass so current and bootstrap values
        # share exactly the same normalization statistics.
        joined_observation = torch.cat((observation_n, next_observation_n), dim=0)
        joined_action = torch.cat((action, next_action), dim=0)
        q1_all, q2_all = self.q_network(
            joined_observation, joined_action, update_stats=True
        )
        q1, next_q1 = q1_all.chunk(2, dim=0)
        q2, next_q2 = q2_all.chunk(2, dim=0)
        with torch.no_grad():
            soft_next_q = torch.minimum(next_q1, next_q2) - self.alpha.detach() * next_log_prob
            target_q = reward + float(self.cfg.discount_gamma) * mask * soft_next_q
        q_loss = 0.5 * ((q1 - target_q).square().mean() + (q2 - target_q).square().mean())

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            self.q_network.parameters(), float(self.cfg.max_grad_norm)
        )
        self.q_optimizer.step()
        return {
            "q_loss": float(q_loss.item()),
            "q_mean": float(torch.minimum(q1, q2).mean().item()),
            "target_q": float(target_q.mean().item()),
            "next_log_prob": float(next_log_prob.mean().item()),
            "critic_grad": float(
                grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
            ),
        }

    def _actor_and_alpha_update(self, observation: torch.Tensor) -> dict[str, float]:
        observation_n = self._norm(observation, update=False)
        self.q_network.requires_grad_(False)
        action, log_prob, path_info = self.actor.sample(
            observation_n, update_stats=True
        )
        q1, q2 = self.q_network(observation_n, action, update_stats=False)
        min_q = torch.minimum(q1, q2)
        actor_loss = (self.alpha.detach() * log_prob - min_q).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = nn.utils.clip_grad_norm_(
            self.actor.parameters(), float(self.cfg.max_grad_norm)
        )
        self.actor_optimizer.step()
        self.q_network.requires_grad_(True)

        target_entropy = float(self.cfg.target_entropy)
        alpha_loss = (
            self.alpha * (-log_prob.detach() - target_entropy)
        ).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        return {
            "actor_loss": float(actor_loss.item()),
            "actor_grad": float(
                actor_grad.item() if torch.is_tensor(actor_grad) else actor_grad
            ),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.detach().item()),
            "log_prob": float(log_prob.mean().item()),
            "q_pi": float(min_q.mean().item()),
            "path_std": float(path_info["path_std"].mean().item()),
            "gate": float(path_info["gate"].mean().item()),
            "latent_abs": float(path_info["terminal_latent"].abs().mean().item()),
        }

    def update(self, rollout: dict, collect_time: float) -> dict:
        update_start = time.perf_counter()
        critic_keys = ("q_loss", "q_mean", "target_q", "next_log_prob", "critic_grad")
        actor_keys = (
            "actor_loss",
            "actor_grad",
            "alpha_loss",
            "alpha",
            "log_prob",
            "q_pi",
            "path_std",
            "gate",
            "latent_abs",
        )
        critic_totals = {key: 0.0 for key in critic_keys}
        actor_totals = {key: 0.0 for key in actor_keys}
        gradient_steps = int(self.cfg.gradient_steps_per_update)
        actor_updates = 0
        sampled_transitions = 0
        for _ in range(gradient_steps):
            sample_size = min(int(self.cfg.replay_batch_size), self.replay.size)
            batch = self.replay.sample(sample_size)
            critic_output = self._critic_update(batch)
            for key in critic_keys:
                critic_totals[key] += critic_output[key]
            sampled_transitions += sample_size
            self.gradient_step += 1
            if self.gradient_step % int(self.cfg.policy_delay) == 0:
                actor_output = self._actor_and_alpha_update(batch[0])
                for key in actor_keys:
                    actor_totals[key] += actor_output[key]
                actor_updates += 1

        for key in critic_totals:
            critic_totals[key] /= max(gradient_steps, 1)
        for key in actor_totals:
            actor_totals[key] /= max(actor_updates, 1)
        metrics = {f"sac_flow/{key}": value for key, value in critic_totals.items()}
        metrics.update({f"sac_flow/{key}": value for key, value in actor_totals.items()})
        metrics.update(
            {
                "sac_flow/critic_updates": float(gradient_steps),
                "sac_flow/actor_updates": float(actor_updates),
                "sac_flow/actor_lr": float(self.actor_optimizer.param_groups[0]["lr"]),
                "sac_flow/critic_lr": float(self.q_optimizer.param_groups[0]["lr"]),
                "sac_flow/alpha_lr": float(self.alpha_optimizer.param_groups[0]["lr"]),
                "sac_flow/replay_size": float(self.replay.size),
                "sac_flow/replay_capacity": float(self.replay.capacity),
                "budget/physical_transitions": float(
                    self.rollout_steps * self.env.num_envs
                ),
                "budget/policy_decisions": float(
                    self.rollout_steps * self.env.num_envs
                ),
                "budget/replay_samples": float(sampled_transitions),
                "budget/actor_replay_samples": float(
                    actor_updates * min(int(self.cfg.replay_batch_size), self.replay.size)
                ),
                "budget/warmup_actions": float(rollout["warmup_actions"]),
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
        action, _, _ = self.actor.sample(normalized, deterministic=True, update_stats=False)
        return action.unsqueeze(1)

    def extra_checkpoint_state(self) -> dict:
        return {
            "q_optimizer": self.q_optimizer.state_dict(),
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
            if "q_optimizer" in payload:
                self.q_optimizer.load_state_dict(payload["q_optimizer"])
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
                f"[SAC_FLOW] replay is not checkpointed; discarded {saved_replay_size} entries.",
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
            f"[SAC_FLOW_Q] loss={metrics['sac_flow/q_loss']:.5f} "
            f"q={metrics['sac_flow/q_mean']:.5f} target={metrics['sac_flow/target_q']:.5f} "
            f"next_logp={metrics['sac_flow/next_log_prob']:.5f} "
            f"grad={metrics['sac_flow/critic_grad']:.5f}",
            flush=True,
        )
        print(
            f"[SAC_FLOW_ACTOR] loss={metrics['sac_flow/actor_loss']:.5f} "
            f"q_pi={metrics['sac_flow/q_pi']:.5f} logp={metrics['sac_flow/log_prob']:.5f} "
            f"alpha={metrics['sac_flow/alpha']:.6f} "
            f"alpha_loss={metrics['sac_flow/alpha_loss']:.5f}",
            flush=True,
        )
        print(
            f"[SAC_FLOW_PATH] std={metrics['sac_flow/path_std']:.5f} "
            f"gate={metrics['sac_flow/gate']:.5f} "
            f"latent_abs={metrics['sac_flow/latent_abs']:.5f}",
            flush=True,
        )
        print(
            f"[BUDGET] physical_transitions={metrics['budget/physical_transitions']:.0f} "
            f"replay_samples={metrics['budget/replay_samples']:.0f} "
            f"critic_steps={metrics['sac_flow/critic_updates']:.0f} "
            f"actor_steps={metrics['sac_flow/actor_updates']:.0f} "
            f"warmup_actions={metrics['budget/warmup_actions']:.0f}",
            flush=True,
        )
        print(
            f"[REPLAY] size={metrics['sac_flow/replay_size']:.0f}/"
            f"{metrics['sac_flow/replay_capacity']:.0f}",
            flush=True,
        )
        print(
            f"[TIME] collect={metrics['timing/collect_s']:.3f}s "
            f"update={metrics['timing/update_s']:.3f}s",
            flush=True,
        )

    def log_banner(self) -> None:
        cfg = self.cfg
        print("[INFO] Starting from-scratch SAC Flow-G training", flush=True)
        print(
            f"[INFO] algorithm=sac-flow actor_obs_dim={self.obs_dim} "
            f"action_dim={self.action_dim} horizon=1 num_envs={self.env.num_envs} "
            f"rollout_env_steps={self.rollout_steps} flow_steps={cfg.flow_steps} "
            f"stochastic_path=true crossq=true batch_renorm={cfg.use_batch_renorm} "
            f"replay_capacity={cfg.replay_capacity} replay_batch_size={cfg.replay_batch_size} "
            f"gradient_steps_per_update={cfg.gradient_steps_per_update} "
            f"policy_delay={cfg.policy_delay} target_entropy={cfg.target_entropy} "
            f"actor_lr={cfg.policy_lr} critic_lr={cfg.critic_lr} alpha_lr={cfg.alpha_lr}",
            flush=True,
        )
        print(
            f"[INFO] trainable_params actor={self._trainable_counts['actor']} "
            f"critic={self._trainable_counts['critic']} temperature=1",
            flush=True,
        )

