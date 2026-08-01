"""Single-step closed-loop fixed-reward Flow-CPS PPO."""

from __future__ import annotations

import math
import time
from collections import deque

import torch
from torch import nn

from components.credit.task_credit import (
    compute_task_gae,
    normalize_actor_advantage,
    resolve_terminal_masks,
)
from components.normalization.running_stats import EmpiricalNormalization
from components.optim.kl_scheduler import adaptive_lr_from_kl
from components.rollout.fixed_reward_contract import (
    FIXED_REWARD_CHECKPOINT_CONTRACT,
)
from components.rollout.training_streams import (
    CURRICULUM_STREAM,
    PHASE0_STREAM,
    Phase0AttemptTracker,
    Phase0CurriculumStreams,
)
from models.flow_cps_policy import FlowMatchingPolicy
from models.value_critic import ValueCritic


RAW_POSE_TERMS = (
    "anchor_pos_reward",
    "anchor_ori_reward",
    "body_pos_reward",
    "body_ori_reward",
)
RAW_PENALTY_TERMS = (
    "action_rate",
    "joint_limit",
    "undesired_contacts",
)
REWARD_TERM_WEIGHTS = {
    "anchor_pos_reward": 0.5,
    "anchor_ori_reward": 0.5,
    "body_pos_reward": 2.0,
    "body_ori_reward": 2.0,
    "action_rate": -0.1,
    "joint_limit": -10.0,
    "undesired_contacts": -0.1,
}
REWARD_METRIC_NAMES = {
    "anchor_pos_reward": "anchor_pos",
    "anchor_ori_reward": "anchor_ori",
    "body_pos_reward": "body_pos",
    "body_ori_reward": "body_ori",
    "action_rate": "action_rate",
    "joint_limit": "joint_limit",
    "undesired_contacts": "undesired_contacts",
}


def _distribution_metrics(
    prefix: str,
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    flat = values.detach().float().reshape(-1)
    if mask is not None:
        selected = mask.detach().bool().reshape(-1)
        if values.ndim > mask.ndim:
            repeats = values.numel() // mask.numel()
            selected = selected.repeat_interleave(repeats)
        flat = flat[selected]
    if flat.numel() == 0:
        return {
            f"{prefix}/count": 0.0,
            f"{prefix}/mean": -1.0,
            f"{prefix}/rms": -1.0,
            f"{prefix}/min": -1.0,
            f"{prefix}/max": -1.0,
            f"{prefix}/p05": -1.0,
            f"{prefix}/p50": -1.0,
            f"{prefix}/p95": -1.0,
            f"{prefix}/p99": -1.0,
        }
    quantiles = torch.quantile(
        flat,
        torch.tensor([0.05, 0.50, 0.95, 0.99], device=flat.device),
    )
    return {
        f"{prefix}/count": float(flat.numel()),
        f"{prefix}/mean": float(flat.mean().item()),
        f"{prefix}/rms": float(torch.sqrt(flat.square().mean()).item()),
        f"{prefix}/min": float(flat.min().item()),
        f"{prefix}/max": float(flat.max().item()),
        f"{prefix}/p05": float(quantiles[0].item()),
        f"{prefix}/p50": float(quantiles[1].item()),
        f"{prefix}/p95": float(quantiles[2].item()),
        f"{prefix}/p99": float(quantiles[3].item()),
    }


class FixedRewardFlowCPS:
    """PPO with one real observation and one absolute Flow action per step."""

    def __init__(self, cfg, env) -> None:
        self.cfg = cfg
        self.env = env

    def build(self) -> None:
        cfg = self.cfg
        env = self.env
        self.num_act = int(env.action_dim)
        self.actor_obs_dim = int(env.observation_dim)
        self.critic_obs_dim = int(env.critic_observation_dim)

        self._policy = FlowMatchingPolicy(
            obs_dim=self.actor_obs_dim,
            action_dim=self.num_act,
            hidden_dims=cfg.actor_hidden_dims,
            activation=cfg.activation,
            action_limit=float(cfg.action_limit),
            flow_steps=int(cfg.flow_steps),
            cps_noise_init=float(cfg.cps_noise_init),
            cps_cov_rank=int(cfg.cps_cov_rank),
        ).to(env.device)
        self.critic = ValueCritic(
            observation_dim=self.critic_obs_dim,
            hidden_dims=cfg.critic_hidden_dims,
            activation=cfg.activation,
        ).to(env.device)
        self.actor_obs_normalizer = EmpiricalNormalization(
            self.actor_obs_dim, env.device
        )
        self.critic_obs_normalizer = EmpiricalNormalization(
            self.critic_obs_dim, env.device
        )

        self.learning_rate = float(cfg.policy_lr)
        self.critic_learning_rate = float(cfg.value_lr)
        self.min_lr = 1.0e-5
        self.max_lr = 1.0e-2
        self.actor_optimizer = torch.optim.AdamW(
            self._policy.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=float(cfg.weight_decay),
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=self.critic_learning_rate,
            betas=(0.9, 0.999),
            eps=1.0e-8,
            weight_decay=float(cfg.critic_weight_decay),
        )
        self._policy_module = nn.ModuleDict(
            {
                "actor": self._policy,
                "actor_obs_normalizer": self.actor_obs_normalizer,
                "critic": self.critic,
                "critic_obs_normalizer": self.critic_obs_normalizer,
            }
        )

        action_limit = float(cfg.action_limit)
        self.action_low = torch.full(
            (self.num_act,), -action_limit, device=env.device
        )
        self.action_high = torch.full(
            (self.num_act,), action_limit, device=env.device
        )
        env.enable_strict_action_contract(self.action_low, self.action_high)

        self.training_streams = Phase0CurriculumStreams.create(
            env.num_envs,
            phase0_fraction=float(cfg.phase0_fraction),
            phase0_start=int(env.motion_start_phase),
            device=env.device,
        )
        self.phase0_attempts = Phase0AttemptTracker(
            self.training_streams.stream_ids
        )
        env.set_adaptive_failure_eligibility(
            self.training_streams.curriculum_mask
        )

        self._update_index = 0
        self._high_kl_streak = 0
        self._nonpositive_reward_streak = 0
        self._stream_return_sum = torch.zeros(
            env.num_envs, dtype=torch.float32, device=env.device
        )
        self._stream_length_sum = torch.zeros_like(self._stream_return_sum)
        self._stream_return_buffers = {
            PHASE0_STREAM: deque(maxlen=100),
            CURRICULUM_STREAM: deque(maxlen=100),
        }
        self._stream_length_buffers = {
            PHASE0_STREAM: deque(maxlen=100),
            CURRICULUM_STREAM: deque(maxlen=100),
        }
        self._previous_action_delta = torch.zeros(
            env.num_envs, self.num_act, device=env.device
        )
        self._has_action_delta = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    @property
    def policy(self) -> nn.Module:
        return self._policy_module

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.actor_optimizer

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = self.actor_obs_normalizer(observation)
        self._ensure_finite("evaluation/actor_observation", normalized)
        action = self._policy.deterministic_action(normalized)
        self._ensure_finite("evaluation/action", action)
        return action

    def evaluation_step(
        self, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        return self.env.step(action)

    def snapshot_runtime_state(self):
        return None

    def restore_runtime_state(self, state) -> None:
        del state

    # ------------------------------------------------------------------
    # Numerical contracts
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_finite(name: str, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a tensor")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name} contains non-finite values")

    def _ensure_runtime_state_finite(self) -> None:
        data = getattr(getattr(self.env, "robot", None), "data", None)
        if data is None:
            return
        for name in ("joint_pos", "joint_vel", "root_state_w", "body_state_w"):
            value = getattr(data, name, None)
            if torch.is_tensor(value):
                self._ensure_finite(f"state/{name}", value)

    def _ensure_gradients_finite(
        self,
        name: str,
        module: nn.Module,
        grad_norm: torch.Tensor | float,
    ) -> None:
        self._ensure_finite(f"{name}/gradient_norm", torch.as_tensor(grad_norm))
        for parameter_name, parameter in module.named_parameters():
            if parameter.grad is not None:
                self._ensure_finite(
                    f"{name}/gradient/{parameter_name}", parameter.grad
                )

    def _ensure_parameters_finite(self, name: str, module: nn.Module) -> None:
        for parameter_name, parameter in module.named_parameters():
            self._ensure_finite(
                f"{name}/parameter/{parameter_name}", parameter
            )

    def _ensure_optimizer_finite(
        self, name: str, optimizer: torch.optim.Optimizer
    ) -> None:
        for parameter_index, state in enumerate(optimizer.state.values()):
            for state_name, value in state.items():
                if torch.is_tensor(value):
                    self._ensure_finite(
                        f"{name}/state/{parameter_index}/{state_name}", value
                    )

    def _ensure_normalizer_finite(
        self, name: str, normalizer: EmpiricalNormalization
    ) -> None:
        for buffer_name in ("_mean", "_var", "_std", "count"):
            self._ensure_finite(
                f"{name}/{buffer_name}", getattr(normalizer, buffer_name)
            )
        if bool((normalizer._var < 0).any()) or bool(
            (normalizer._std < 0).any()
        ):
            raise FloatingPointError(f"{name} contains a negative scale")

    # ------------------------------------------------------------------
    # Checkpoint contract
    # ------------------------------------------------------------------
    @staticmethod
    def _removed_state_keys(mapping: object) -> set[str]:
        if not isinstance(mapping, dict):
            return set()
        forbidden = (
            "amp_",
            "disc_",
            "discriminator",
            "style_prior",
            "mixed_reward",
            "channel_",
            "history",
            "replay",
            "mmd",
            "world_model",
        )
        return {
            str(key)
            for key in mapping
            if any(token in str(key).lower() for token in forbidden)
        }

    def _validate_checkpoint_contract(
        self, state: object, policy_state: object | None = None
    ) -> None:
        if not isinstance(state, dict):
            raise ValueError(
                "fixed_reward checkpoint lacks schema 14; start a fresh run"
            )
        for key, expected in FIXED_REWARD_CHECKPOINT_CONTRACT.items():
            actual = state.get(key)
            if actual != expected:
                raise ValueError(
                    "fixed_reward checkpoint semantic contract mismatch: "
                    f"{key} expected={expected!r}, actual={actual!r}"
                )
        removed = self._removed_state_keys(state)
        removed.update(self._removed_state_keys(policy_state))
        if removed:
            raise ValueError(
                "fixed_reward checkpoint contains removed state: "
                + ", ".join(sorted(removed))
            )

    def validate_checkpoint_payload(self, payload: dict) -> None:
        state = payload.get("algo_state") if isinstance(payload, dict) else None
        policy_state = payload.get("policy") if isinstance(payload, dict) else None
        self._validate_checkpoint_contract(state, policy_state)

    def extra_checkpoint_state(self) -> dict:
        return {
            **FIXED_REWARD_CHECKPOINT_CONTRACT,
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "learning_rate": float(self.learning_rate),
            "critic_learning_rate": float(self.critic_learning_rate),
            "stream_ids": self.training_streams.stream_ids.detach().cpu(),
            "phase0_stream_count": int(
                self.training_streams.phase0_ids.numel()
            ),
            "phase0_stream_fraction": float(
                self.training_streams.phase0_fraction
            ),
            "phase0_attempt_tracker": self.phase0_attempts.state_dict(),
        }

    def load_extra_checkpoint_state(
        self, payload: dict, reset_optimizer: bool = False
    ) -> None:
        self._validate_checkpoint_contract(payload)
        if reset_optimizer:
            self.learning_rate = float(self.cfg.policy_lr)
            self.critic_learning_rate = float(self.cfg.value_lr)
        else:
            self.learning_rate = float(payload["learning_rate"])
            self.critic_learning_rate = float(payload["critic_learning_rate"])
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.learning_rate
        for group in self.critic_optimizer.param_groups:
            group["lr"] = self.critic_learning_rate

        saved_stream_ids = payload.get("stream_ids")
        if not torch.is_tensor(saved_stream_ids) or not torch.equal(
            saved_stream_ids.to(dtype=torch.int8, device="cpu"),
            self.training_streams.stream_ids.detach().to("cpu"),
        ):
            raise ValueError("checkpoint training-stream assignment differs")
        if int(payload.get("phase0_stream_count", -1)) != int(
            self.training_streams.phase0_ids.numel()
        ):
            raise ValueError("checkpoint phase0 stream count differs")
        if not math.isclose(
            float(payload.get("phase0_stream_fraction", -1.0)),
            float(self.training_streams.phase0_fraction),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("checkpoint stream objective differs")
        self.phase0_attempts.load_state_dict(
            payload.get("phase0_attempt_tracker")
        )

    # ------------------------------------------------------------------
    # Reset and normalization
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _update_normalizer(
        self, normalizer: EmpiricalNormalization, samples: torch.Tensor
    ) -> None:
        flat = samples.reshape(-1, samples.shape[-1])
        if flat.shape[0] == 0:
            raise RuntimeError("normalizer received no samples")
        batch_size = min(max(1, int(self.cfg.micro_batch_size)), 1024)
        for start in range(0, flat.shape[0], batch_size):
            batch = flat[start : start + batch_size]
            self._ensure_finite("normalizer/input", batch)
            normalizer._update(batch)
        self._ensure_normalizer_finite("normalizer", normalizer)

    @torch.no_grad()
    def _evaluate_values(self, observations: torch.Tensor) -> torch.Tensor:
        shape = observations.shape[:-1]
        flat = observations.reshape(-1, self.critic_obs_dim)
        batch_size = max(1, int(self.cfg.micro_batch_size))
        values: list[torch.Tensor] = []
        for start in range(0, flat.shape[0], batch_size):
            value = self.critic(flat[start : start + batch_size])
            self._ensure_finite("critic/value", value)
            values.append(value)
        return torch.cat(values).reshape(shape)

    def _reset_training_streams(
        self, *, randomize_curriculum_episode_age: bool
    ) -> torch.Tensor:
        env = self.env
        self.phase0_attempts.interrupt_inflight()
        env_ids = torch.arange(
            env.num_envs, device=env.device, dtype=torch.long
        )
        phases, reset_streams = self.training_streams.reset_phases(
            env_ids,
            lambda count: env.sample_phase_indices(count, horizon=1),
        )
        observation = env.reset(
            phase_indices=phases, reset_stream_ids=reset_streams
        )
        if (
            randomize_curriculum_episode_age
            and bool(self.cfg.init_at_random_ep_len)
            and env.max_episode_steps > 0
            and self.training_streams.curriculum_ids.numel() > 0
        ):
            ids = self.training_streams.curriculum_ids
            ages = torch.randint(
                0,
                int(env.max_episode_steps),
                (ids.numel(),),
                device=env.device,
                dtype=env.episode_steps.dtype,
            )
            env.set_episode_age(ids, ages)
        self._ensure_finite("observation/reset", observation)
        self._obs = observation
        self._critic_obs = env.get_critic_observation()
        self._ensure_finite("critic_observation/reset", self._critic_obs)
        self._previous_action_delta.zero_()
        self._has_action_delta.zero_()
        self.phase0_attempts.start(self.training_streams.phase0_ids)
        return observation

    def initial_reset(self) -> torch.Tensor:
        return self._reset_training_streams(
            randomize_curriculum_episode_age=True
        )

    def reset_after_resume(self) -> torch.Tensor:
        self._stream_return_sum.zero_()
        self._stream_length_sum.zero_()
        return self._reset_training_streams(
            randomize_curriculum_episode_age=False
        )

    def evaluation_reset(self, phase_indices: torch.Tensor) -> torch.Tensor:
        return self.env.reset(phase_indices=phase_indices)

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        self._update_index = int(update_idx)
        self.phase0_attempts.begin_update()
        return self._obs

    def _record_episode_stats(
        self, reward: torch.Tensor, done: torch.Tensor
    ) -> None:
        self._stream_return_sum += reward.float()
        self._stream_length_sum += 1.0
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() == 0:
            return
        streams = self.training_streams.stream_ids.index_select(0, done_ids)
        for stream_id in (PHASE0_STREAM, CURRICULUM_STREAM):
            ids = done_ids[streams == stream_id]
            if ids.numel() == 0:
                continue
            self._stream_return_buffers[stream_id].extend(
                self._stream_return_sum.index_select(0, ids)
                .detach()
                .cpu()
                .tolist()
            )
            self._stream_length_buffers[stream_id].extend(
                self._stream_length_sum.index_select(0, ids)
                .detach()
                .cpu()
                .tolist()
            )
        self._stream_return_sum[done_ids] = 0.0
        self._stream_length_sum[done_ids] = 0.0

    # ------------------------------------------------------------------
    # Rollout and GAE
    # ------------------------------------------------------------------
    def _reward_contributions(
        self, terms: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for name, weight in REWARD_TERM_WEIGHTS.items():
            if name not in terms:
                raise KeyError(f"reward terms are missing {name}")
            self._ensure_finite(f"reward/raw/{name}", terms[name])
            result[name] = float(weight) * terms[name] * float(self.env.dt)
        return result

    def collect(self, current_obs: torch.Tensor) -> dict:
        env = self.env
        device = env.device
        time_steps = int(self.cfg.rollout_env_steps)
        num_envs = env.num_envs
        flow_steps = int(self.cfg.flow_steps)

        actor_obs = torch.empty(
            time_steps, num_envs, self.actor_obs_dim, device=device
        )
        actor_obs_raw = torch.empty_like(actor_obs)
        critic_obs = torch.empty(
            time_steps, num_envs, self.critic_obs_dim, device=device
        )
        critic_obs_raw = torch.empty_like(critic_obs)
        next_critic_obs = torch.empty_like(critic_obs)
        latent_paths = torch.empty(
            time_steps,
            num_envs,
            flow_steps + 1,
            self.num_act,
            device=device,
        )
        old_log_probs = torch.empty(
            time_steps, num_envs, flow_steps, device=device
        )
        actions = torch.empty(
            time_steps, num_envs, self.num_act, device=device
        )
        mean_actions = torch.empty_like(actions)
        reward = torch.empty(time_steps, num_envs, device=device)
        raw_terms = {
            name: torch.empty_like(reward)
            for name in (*RAW_POSE_TERMS, *RAW_PENALTY_TERMS)
        }
        contributions = {
            name: torch.empty_like(reward) for name in REWARD_TERM_WEIGHTS
        }
        done = torch.zeros(
            time_steps, num_envs, dtype=torch.bool, device=device
        )
        failure = torch.zeros_like(done)
        timeout = torch.zeros_like(done)
        motion_complete = torch.zeros_like(done)
        intervention_edge = torch.zeros_like(done)
        terminal_phase = torch.full(
            (time_steps, num_envs),
            -1.0,
            dtype=torch.float32,
            device=device,
        )
        action_delta = torch.empty_like(actions)
        action_d2 = torch.empty_like(actions)
        action_delta_valid = torch.zeros_like(done)
        action_d2_valid = torch.zeros_like(done)
        reset_action = torch.zeros_like(done)
        joint_vel_jump = torch.empty_like(reward)
        root_lin_vel_jump = torch.empty_like(reward)
        root_ang_vel_jump = torch.empty_like(reward)

        observation = current_obs
        value_observation = self._critic_obs
        collection_start_phases = env.phase_steps.detach().clone()
        reward_identity_abs_max = 0.0
        action_bound_violation_max = 0.0
        innovation_square_sum = torch.zeros(flow_steps, device=device)

        with torch.no_grad():
            for step_index in range(time_steps):
                self._ensure_finite("observation/current", observation)
                self._ensure_finite(
                    "critic_observation/current", value_observation
                )
                normalized_actor_obs = self.actor_obs_normalizer(observation)
                normalized_critic_obs = self.critic_obs_normalizer(
                    value_observation
                )
                self._ensure_finite(
                    "observation/actor_normalized", normalized_actor_obs
                )
                self._ensure_finite(
                    "observation/critic_normalized", normalized_critic_obs
                )

                (
                    action,
                    latent_path,
                    step_log_prob,
                    sample_diagnostics,
                ) = self._policy.sample(normalized_actor_obs)
                mean_action = sample_diagnostics["mean_action"]
                for name, tensor in (
                    ("action", action),
                    ("mean_action", mean_action),
                    ("latent_path", latent_path),
                    ("log_probability", step_log_prob),
                ):
                    self._ensure_finite(f"policy/{name}", tensor)

                below = (self.action_low - action).clamp_min(0.0)
                above = (action - self.action_high).clamp_min(0.0)
                action_bound_violation_max = max(
                    action_bound_violation_max,
                    float(torch.maximum(below, above).max().item()),
                )
                previous_action = env.last_action.detach().clone()
                delta = action - previous_action
                has_delta = self._has_action_delta.clone()
                second_delta = delta - self._previous_action_delta
                action_delta[step_index] = delta
                action_d2[step_index] = second_delta
                action_delta_valid[step_index] = has_delta
                action_d2_valid[step_index] = has_delta
                reset_action[step_index] = ~has_delta
                self._previous_action_delta.copy_(delta)
                self._has_action_delta.fill_(True)

                _, joint_vel_before = env.get_action_joint_state()
                root_velocity_before = env.get_mimic_root_velocity_w()
                (
                    next_observation,
                    step_reward,
                    step_done,
                    info,
                ) = env.step(action)
                terminal_critic_obs = env.get_critic_observation()
                _, joint_vel_after = env.get_action_joint_state()
                root_velocity_after = env.get_mimic_root_velocity_w()
                self._ensure_runtime_state_finite()
                for name, tensor in (
                    ("next_observation", next_observation),
                    ("next_critic_observation", terminal_critic_obs),
                    ("reward", step_reward),
                ):
                    self._ensure_finite(name, tensor)
                if float(step_reward.max().item()) > 0.100001:
                    raise RuntimeError(
                        "fixed reward exceeded its single-step maximum"
                    )

                reward_terms = info["reward_terms"]
                step_contributions = self._reward_contributions(reward_terms)
                reconstructed = torch.zeros_like(step_reward)
                for value in step_contributions.values():
                    reconstructed += value
                identity_error = float(
                    (reconstructed - step_reward).abs().max().item()
                )
                reward_identity_abs_max = max(
                    reward_identity_abs_max, identity_error
                )
                if identity_error > 1.0e-6:
                    raise RuntimeError(
                        "fixed reward decomposition identity exceeded 1e-6"
                    )

                done_terms = info["done_terms"]
                failures = (
                    done_terms["anchor_pos_bad"].bool()
                    | done_terms["anchor_ori_bad"].bool()
                    | done_terms["ee_body_bad"].bool()
                )
                step_failure, step_timeout, step_complete = (
                    resolve_terminal_masks(
                        step_done.bool(),
                        done_terms["time_out"].bool(),
                        done_terms["motion_complete"].bool(),
                        failures,
                    )
                )
                self.phase0_attempts.observe_step(
                    torch.ones_like(step_done, dtype=torch.bool),
                    step_done.bool(),
                    step_failure,
                    step_timeout,
                    step_complete,
                )

                actor_obs[step_index] = normalized_actor_obs
                actor_obs_raw[step_index] = observation
                critic_obs[step_index] = normalized_critic_obs
                critic_obs_raw[step_index] = value_observation
                next_critic_obs[step_index] = self.critic_obs_normalizer(
                    terminal_critic_obs
                )
                latent_paths[step_index] = latent_path
                old_log_probs[step_index] = step_log_prob
                actions[step_index] = action
                mean_actions[step_index] = mean_action
                reward[step_index] = step_reward
                done[step_index] = step_done.bool()
                failure[step_index] = step_failure
                timeout[step_index] = step_timeout
                motion_complete[step_index] = step_complete
                intervention_edge[step_index] = info[
                    "intervention_edge_mask"
                ].bool()
                joint_vel_jump[step_index] = (
                    joint_vel_after - joint_vel_before
                ).abs().mean(dim=-1)
                root_lin_vel_jump[step_index] = (
                    root_velocity_after[:, :3] - root_velocity_before[:, :3]
                ).abs().mean(dim=-1)
                root_ang_vel_jump[step_index] = (
                    root_velocity_after[:, 3:] - root_velocity_before[:, 3:]
                ).abs().mean(dim=-1)
                for name in raw_terms:
                    raw_terms[name][step_index] = reward_terms[name]
                    contributions[name][step_index] = step_contributions[name]
                if bool(step_done.any()):
                    terminal_phase[step_index, step_done.bool()] = info[
                        "termination_phase_steps"
                    ][step_done.bool()].float()

                innovation_rms = sample_diagnostics[
                    "innovation_rms_per_flow_step"
                ]
                self._ensure_finite("cps/innovation_rms", innovation_rms)
                innovation_square_sum += innovation_rms.square()
                self._record_episode_stats(step_reward, step_done.bool())

                observation = next_observation
                value_observation = terminal_critic_obs
                done_ids = step_done.bool().nonzero(
                    as_tuple=False
                ).squeeze(-1)
                if done_ids.numel() > 0:
                    reset_phases, reset_streams = (
                        self.training_streams.reset_phases(
                            done_ids,
                            lambda count: env.sample_phase_indices(
                                count, horizon=1
                            ),
                        )
                    )
                    reset_observation = env.reset_envs(
                        done_ids,
                        phase_indices=reset_phases,
                        reset_stream_ids=reset_streams,
                    )
                    self._ensure_finite(
                        "observation/partial_reset", reset_observation
                    )
                    observation[done_ids] = reset_observation
                    value_observation = env.get_critic_observation()
                    self._previous_action_delta[done_ids] = 0.0
                    self._has_action_delta[done_ids] = False
                    self.phase0_attempts.start(done_ids)

            values = self._evaluate_values(critic_obs)
            next_values = self._evaluate_values(next_critic_obs)

        if action_bound_violation_max > 1.0e-6:
            raise RuntimeError("policy emitted an action outside its domain")
        self._obs = observation
        self._critic_obs = value_observation
        rollout = {
            "actor_obs": actor_obs,
            "actor_obs_raw": actor_obs_raw,
            "critic_obs": critic_obs,
            "critic_obs_raw": critic_obs_raw,
            "latents": latent_paths,
            "old_log_probs": old_log_probs,
            "actions": actions,
            "mean_actions": mean_actions,
            "values": values,
            "next_values": next_values,
            "done": done,
            "failure": failure,
            "timeout": timeout,
            "motion_complete": motion_complete,
            "terminal_phase": terminal_phase,
            "bootstrap_mask": ~failure & ~motion_complete,
            "trace_mask": ~done,
            "reward": reward,
            "reward_raw_terms": raw_terms,
            "reward_contributions": contributions,
            "reward_decomposition_abs_max": reward_identity_abs_max,
            "intervention_edge": intervention_edge,
            "stream_ids": self.training_streams.stream_ids,
            "collection_start_phases": collection_start_phases,
            "action_bound_violation_max": action_bound_violation_max,
            "action_delta": action_delta,
            "action_d2": action_d2,
            "action_delta_valid": action_delta_valid,
            "action_d2_valid": action_d2_valid,
            "reset_action": reset_action,
            "joint_vel_jump": joint_vel_jump,
            "root_lin_vel_jump": root_lin_vel_jump,
            "root_ang_vel_jump": root_ang_vel_jump,
            "cps_innovation_rms": torch.sqrt(
                innovation_square_sum / float(time_steps)
            ),
            "next_observation": observation,
        }
        self._assign_credit(rollout)
        return rollout

    @torch.no_grad()
    def _assign_credit(self, rollout: dict) -> None:
        rewards = rollout["reward"]
        valid = torch.ones_like(rewards, dtype=torch.bool)
        credit = compute_task_gae(
            rewards,
            rollout["values"],
            rollout["next_values"],
            rollout["bootstrap_mask"],
            rollout["trace_mask"],
            valid,
            gamma=float(self.cfg.discount_gamma),
            gae_lambda=float(self.cfg.gae_lambda),
        )
        weights = torch.zeros_like(rewards)
        for _, _, objective_weight, env_ids in self._stream_specs(
            self.training_streams.stream_ids
        ):
            sample_count = rewards.shape[0] * env_ids.numel()
            if sample_count <= 0:
                raise RuntimeError("training stream has no actor samples")
            weights[:, env_ids] = float(objective_weight) / float(sample_count)
        credit = normalize_actor_advantage(credit, valid, weights)
        for name, tensor in (
            ("advantage", credit.advantages),
            ("actor_advantage", credit.actor_advantage),
            ("value_target", credit.value_targets),
        ):
            self._ensure_finite(f"credit/{name}", tensor)
        rollout["advantages"] = credit.actor_advantage
        rollout["raw_advantages"] = credit.advantages
        rollout["value_targets"] = credit.value_targets

    def _stream_specs(
        self, labels: torch.Tensor
    ) -> list[tuple[str, int, float, torch.Tensor]]:
        configured = (
            ("phase0", PHASE0_STREAM, float(self.cfg.phase0_fraction)),
            (
                "curriculum",
                CURRICULUM_STREAM,
                1.0 - float(self.cfg.phase0_fraction),
            ),
        )
        active: list[tuple[str, int, float, torch.Tensor]] = []
        for name, stream_id, weight in configured:
            indices = (labels == stream_id).nonzero(
                as_tuple=False
            ).squeeze(-1)
            if indices.numel() > 0 and weight > 0.0:
                active.append((name, stream_id, weight, indices))
        total_weight = sum(item[2] for item in active)
        if not active or total_weight <= 0.0:
            raise RuntimeError("fixed_reward has no active stream")
        return [
            (name, stream_id, weight / total_weight, indices)
            for name, stream_id, weight, indices in active
        ]

    # ------------------------------------------------------------------
    # PPO and value optimization
    # ------------------------------------------------------------------
    @staticmethod
    def _snapshot_parameters(module: nn.Module) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }

    @staticmethod
    def _parameter_delta(
        module: nn.Module,
        before: dict[str, torch.Tensor],
        predicate=lambda _name: True,
    ) -> float:
        current = dict(module.named_parameters())
        selected = [name for name in before if predicate(name)]
        if not selected:
            raise RuntimeError("parameter group is empty")
        squared = torch.zeros((), device=current[selected[0]].device)
        for name in selected:
            squared += (
                current[name].detach() - before[name].to(current[name])
            ).float().square().sum()
        return float(torch.sqrt(squared).item())

    @staticmethod
    def _gradient_l2(module: nn.Module, predicate=lambda _name: True) -> float:
        squared: torch.Tensor | None = None
        for name, parameter in module.named_parameters():
            if parameter.grad is None or not predicate(name):
                continue
            term = parameter.grad.detach().float().square().sum()
            squared = term if squared is None else squared + term
        return 0.0 if squared is None else float(torch.sqrt(squared).item())

    def _mini_batch_size(self, sample_count: int) -> int:
        return max(
            1,
            math.ceil(sample_count / max(1, int(self.cfg.num_mini_batches))),
        )

    def _micro_batch_size(self, batch_size: int) -> int:
        configured = int(self.cfg.micro_batch_size)
        return batch_size if configured <= 0 else max(
            1, min(batch_size, configured)
        )

    def _update_actor_lr(self, observed_kl: float) -> int:
        if float(self.cfg.desired_kl) <= 0.0:
            return 0
        old_lr = self.learning_rate
        self.learning_rate, _ = adaptive_lr_from_kl(
            raw_kl=observed_kl,
            kl_units=1,
            target_per_step=float(self.cfg.desired_kl),
            lr=self.learning_rate,
            min_lr=self.min_lr,
            max_lr=self.max_lr,
        )
        for group in self.actor_optimizer.param_groups:
            group["lr"] = self.learning_rate
        return -1 if self.learning_rate < old_lr else int(
            self.learning_rate > old_lr
        )

    @torch.no_grad()
    def _policy_ratio_metrics(
        self,
        actor_obs: torch.Tensor,
        latent_paths: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> dict[str, float]:
        batch_size = self._micro_batch_size(actor_obs.shape[0])
        log_ratios: list[torch.Tensor] = []
        factor_deltas: list[torch.Tensor] = []
        for start in range(0, actor_obs.shape[0], batch_size):
            stop = start + batch_size
            new = self._policy.recompute_log_probs(
                actor_obs[start:stop], latent_paths[start:stop]
            )
            delta = new - old_log_probs[start:stop]
            factor_deltas.append(delta)
            log_ratios.append(delta.sum(dim=-1))
        factors = torch.cat(factor_deltas)
        log_ratio = torch.cat(log_ratios)
        ratio = torch.exp(log_ratio)
        self._ensure_finite("policy/post_log_ratio", log_ratio)
        self._ensure_finite("policy/post_ratio", ratio)
        clipped = (
            (ratio < 1.0 - float(self.cfg.clip_range))
            | (ratio > 1.0 + float(self.cfg.clip_range))
        ).float()
        ratio_q = torch.quantile(
            ratio,
            torch.tensor([0.05, 0.50, 0.95], device=ratio.device),
        )
        abs_log_q = torch.quantile(log_ratio.abs(), 0.95)
        return {
            "policy/kl": float((0.5 * log_ratio.square()).mean().item()),
            "policy/flow_factor_kl": float(
                (0.5 * factors.square()).mean().item()
            ),
            "policy/ratio_mean": float(ratio.mean().item()),
            "policy/ratio_min": float(ratio.min().item()),
            "policy/ratio_p05": float(ratio_q[0].item()),
            "policy/ratio_p50": float(ratio_q[1].item()),
            "policy/ratio_p95": float(ratio_q[2].item()),
            "policy/ratio_max": float(ratio.max().item()),
            "policy/clip_fraction": float(clipped.mean().item()),
            "policy/log_ratio_abs_p95": float(abs_log_q.item()),
            "policy/log_ratio_abs_max": float(log_ratio.abs().max().item()),
        }

    def _actor_update(self, rollout: dict) -> dict[str, float]:
        time_steps, num_envs = rollout["reward"].shape
        sample_count = time_steps * num_envs
        actor_obs = rollout["actor_obs"].reshape(
            sample_count, self.actor_obs_dim
        )
        latent_paths = rollout["latents"].reshape(
            sample_count,
            int(self.cfg.flow_steps) + 1,
            self.num_act,
        )
        old_log_probs = rollout["old_log_probs"].reshape(
            sample_count, int(self.cfg.flow_steps)
        )
        advantages = rollout["advantages"].reshape(sample_count)
        labels = rollout["stream_ids"].reshape(1, num_envs).expand(
            time_steps, num_envs
        ).reshape(-1)
        for name, tensor in (
            ("observation", actor_obs),
            ("latent_path", latent_paths),
            ("old_log_probability", old_log_probs),
            ("advantage", advantages),
        ):
            self._ensure_finite(f"policy/input/{name}", tensor)

        # The unchanged sampler and recomputation must agree before PPO.
        preflight_max = 0.0
        with torch.no_grad():
            # Recompute with the same per-step environment batch used during
            # sampling. CUDA GEMM reduction order can differ across batch
            # shapes by one float32 ULP; changing the batch shape is not a
            # policy-density discrepancy.
            sampled_observations = rollout["actor_obs"]
            sampled_paths = rollout["latents"]
            sampled_log_probs = rollout["old_log_probs"]
            for step_index in range(time_steps):
                recomputed = self._policy.recompute_log_probs(
                    sampled_observations[step_index],
                    sampled_paths[step_index],
                )
                preflight_max = max(
                    preflight_max,
                    float(
                        (recomputed - sampled_log_probs[step_index])
                        .abs()
                        .max()
                        .item()
                    ),
                )
        if preflight_max > 3.0e-6:
            raise RuntimeError(
                "sampled/recomputed CPS log probability differs by more than "
                f"3e-6: {preflight_max:.9g}"
            )

        streams = self._stream_specs(labels)
        num_mini_batches = max(1, int(self.cfg.num_mini_batches))
        micro_size = self._micro_batch_size(
            self._mini_batch_size(sample_count)
        )
        clip_low = 1.0 - float(self.cfg.clip_range)
        clip_high = 1.0 + float(self.cfg.clip_range)
        before = self._snapshot_parameters(self._policy)
        lr_start = self.learning_rate
        totals = {
            "loss": 0.0,
            "grad": 0.0,
            "backbone_grad": 0.0,
            "covariance_grad": 0.0,
            "eta_grad": 0.0,
        }
        stream_loss = {name: 0.0 for name, _, _, _ in streams}
        stream_steps = {name: 0 for name, _, _, _ in streams}
        lr_down = lr_up = lr_hold = 0
        optimizer_steps = 0
        early_stop_epoch = int(self.cfg.policy_epochs)

        for epoch in range(int(self.cfg.policy_epochs)):
            epoch_kl = 0.0
            epoch_steps = 0
            splits = {}
            for name, _, weight, indices in streams:
                shuffled = indices[torch.randperm(indices.numel(), device=indices.device)]
                splits[name] = (weight, torch.tensor_split(shuffled, num_mini_batches))
            for mini_index in range(num_mini_batches):
                parts = [
                    (name, weight, groups[mini_index])
                    for name, (weight, groups) in splits.items()
                    if groups[mini_index].numel() > 0
                ]
                if not parts:
                    continue
                self.actor_optimizer.zero_grad(set_to_none=True)
                combined_loss = 0.0
                combined_kl = 0.0
                for name, weight, indices in parts:
                    denominator = float(indices.numel())
                    loss_sum = 0.0
                    kl_sum = 0.0
                    for start in range(0, indices.numel(), micro_size):
                        sub = indices[start : start + micro_size]
                        new_log_probs = self._policy.recompute_log_probs(
                            actor_obs[sub], latent_paths[sub]
                        )
                        delta = new_log_probs - old_log_probs[sub]
                        log_ratio = delta.sum(dim=-1)
                        ratio = torch.exp(log_ratio)
                        self._ensure_finite("policy/log_ratio", log_ratio)
                        self._ensure_finite("policy/ratio", ratio)
                        unclipped = -advantages[sub] * ratio
                        clipped = -advantages[sub] * torch.clamp(
                            ratio, clip_low, clip_high
                        )
                        loss = torch.maximum(unclipped, clipped).sum()
                        (float(weight) * loss / denominator).backward()
                        loss_sum += float(loss.detach().item())
                        kl_sum += float(
                            (0.5 * log_ratio.detach().square()).sum().item()
                        )
                    stream_mean = loss_sum / denominator
                    combined_loss += float(weight) * stream_mean
                    combined_kl += float(weight) * kl_sum / denominator
                    stream_loss[name] += stream_mean
                    stream_steps[name] += 1

                direction = self._update_actor_lr(combined_kl)
                lr_down += int(direction < 0)
                lr_up += int(direction > 0)
                lr_hold += int(direction == 0)
                grad_norm = nn.utils.clip_grad_norm_(
                    self._policy.parameters(),
                    float(self.cfg.max_grad_norm),
                    error_if_nonfinite=True,
                )
                self._ensure_gradients_finite("actor", self._policy, grad_norm)
                totals["loss"] += combined_loss
                totals["grad"] += float(grad_norm)
                totals["backbone_grad"] += self._gradient_l2(
                    self._policy, lambda name: not name.startswith("cps_")
                )
                totals["covariance_grad"] += self._gradient_l2(
                    self._policy,
                    lambda name: name in {"cps_diag_raw", "cps_lowrank_raw"},
                )
                totals["eta_grad"] += self._gradient_l2(
                    self._policy, lambda name: name == "cps_eta_raw"
                )
                self.actor_optimizer.step()
                self._ensure_parameters_finite("actor/after", self._policy)
                self._ensure_optimizer_finite(
                    "actor_optimizer", self.actor_optimizer
                )
                optimizer_steps += 1
                epoch_kl += combined_kl
                epoch_steps += 1
            if (
                float(self.cfg.desired_kl) > 0.0
                and epoch_steps > 0
                and epoch_kl / epoch_steps
                > float(self.cfg.kl_early_stop_factor)
                * float(self.cfg.desired_kl)
            ):
                early_stop_epoch = epoch + 1
                break

        if optimizer_steps == 0:
            raise RuntimeError("actor performed no optimizer step")
        actor_delta = self._parameter_delta(self._policy, before)
        if not math.isfinite(actor_delta) or actor_delta <= 0.0:
            raise RuntimeError("actor parameters did not update")
        denominator = float(optimizer_steps)
        metrics = {
            "policy/loss": totals["loss"] / denominator,
            "policy/grad_norm": totals["grad"] / denominator,
            "policy/backbone_grad_norm": totals["backbone_grad"] / denominator,
            "policy/covariance_grad_norm": totals["covariance_grad"] / denominator,
            "policy/eta_grad_abs": totals["eta_grad"] / denominator,
            "policy/parameter_delta_l2": actor_delta,
            "policy/backbone_parameter_delta_l2": self._parameter_delta(
                self._policy,
                before,
                lambda name: not name.startswith("cps_"),
            ),
            "policy/covariance_parameter_delta_l2": self._parameter_delta(
                self._policy,
                before,
                lambda name: name in {"cps_diag_raw", "cps_lowrank_raw"},
            ),
            "policy/eta_parameter_delta_abs": self._parameter_delta(
                self._policy,
                before,
                lambda name: name == "cps_eta_raw",
            ),
            "policy/lr_start": float(lr_start),
            "policy/lr": float(self.learning_rate),
            "policy/lr_decrease_steps": float(lr_down),
            "policy/lr_increase_steps": float(lr_up),
            "policy/lr_hold_steps": float(lr_hold),
            "policy/optimizer_steps": float(optimizer_steps),
            "policy/early_stop_epoch": float(early_stop_epoch),
            "policy/log_prob_recompute_abs_max": float(preflight_max),
        }
        metrics.update(
            self._policy_ratio_metrics(
                actor_obs, latent_paths, old_log_probs
            )
        )
        for name, _, weight, _ in streams:
            metrics[f"stream/{name}/actor_objective_weight"] = float(weight)
            metrics[f"stream/{name}/actor_loss"] = stream_loss[name] / max(
                stream_steps[name], 1
            )
        return metrics

    def _critic_update(self, rollout: dict) -> dict[str, float]:
        time_steps, num_envs = rollout["reward"].shape
        observations = rollout["critic_obs"].reshape(
            -1, self.critic_obs_dim
        )
        targets = rollout["value_targets"].reshape(-1)
        labels = rollout["stream_ids"].reshape(1, num_envs).expand(
            time_steps, num_envs
        ).reshape(-1)
        self._ensure_finite("critic/input/observation", observations)
        self._ensure_finite("critic/input/target", targets)
        streams = self._stream_specs(labels)
        num_mini_batches = max(1, int(self.cfg.num_mini_batches))
        micro_size = self._micro_batch_size(
            self._mini_batch_size(observations.shape[0])
        )
        before = self._snapshot_parameters(self.critic)
        total_loss = total_grad = 0.0
        optimizer_steps = 0
        stream_loss = {name: 0.0 for name, _, _, _ in streams}
        stream_steps = {name: 0 for name, _, _, _ in streams}

        for _ in range(int(self.cfg.policy_epochs)):
            splits = {}
            for name, _, weight, indices in streams:
                shuffled = indices[torch.randperm(indices.numel(), device=indices.device)]
                splits[name] = (weight, torch.tensor_split(shuffled, num_mini_batches))
            for mini_index in range(num_mini_batches):
                parts = [
                    (name, weight, groups[mini_index])
                    for name, (weight, groups) in splits.items()
                    if groups[mini_index].numel() > 0
                ]
                if not parts:
                    continue
                self.critic_optimizer.zero_grad(set_to_none=True)
                combined_loss = 0.0
                for name, weight, indices in parts:
                    denominator = float(indices.numel())
                    loss_sum = 0.0
                    for start in range(0, indices.numel(), micro_size):
                        sub = indices[start : start + micro_size]
                        prediction = self.critic(observations[sub])
                        loss = (prediction - targets[sub]).square().sum()
                        self._ensure_finite("critic/loss", loss)
                        (float(weight) * loss / denominator).backward()
                        loss_sum += float(loss.detach().item())
                    mean_loss = loss_sum / denominator
                    combined_loss += float(weight) * mean_loss
                    stream_loss[name] += mean_loss
                    stream_steps[name] += 1
                grad_norm = nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    float(self.cfg.max_grad_norm),
                    error_if_nonfinite=True,
                )
                self._ensure_gradients_finite("critic", self.critic, grad_norm)
                self.critic_optimizer.step()
                self._ensure_parameters_finite("critic/after", self.critic)
                self._ensure_optimizer_finite(
                    "critic_optimizer", self.critic_optimizer
                )
                total_loss += combined_loss
                total_grad += float(grad_norm)
                optimizer_steps += 1

        if optimizer_steps == 0:
            raise RuntimeError("critic performed no optimizer step")
        parameter_delta = self._parameter_delta(self.critic, before)
        if not math.isfinite(parameter_delta) or parameter_delta <= 0.0:
            raise RuntimeError("critic parameters did not update")
        with torch.no_grad():
            prediction = self._evaluate_values(
                observations.reshape(time_steps, num_envs, -1)
            ).reshape(-1)
            error = prediction - targets
            target_variance = targets.var(unbiased=False)
            explained_variance = 1.0 - error.var(unbiased=False) / (
                target_variance + 1.0e-8
            )
        metrics = {
            "critic/value_loss": total_loss / float(optimizer_steps),
            "critic/grad_norm": total_grad / float(optimizer_steps),
            "critic/parameter_delta_l2": parameter_delta,
            "critic/lr": float(self.critic_learning_rate),
            "critic/optimizer_steps": float(optimizer_steps),
            "critic/sample_count": float(targets.numel()),
            "critic/mae": float(error.abs().mean().item()),
            "critic/rmse": float(torch.sqrt(error.square().mean()).item()),
            "critic/explained_variance": float(explained_variance.item()),
        }
        for name, _, weight, _ in streams:
            metrics[f"stream/{name}/critic_objective_weight"] = float(weight)
            metrics[f"stream/{name}/critic_value_loss"] = stream_loss[name] / max(
                stream_steps[name], 1
            )
        return metrics

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _stream_metrics(self, rollout: dict) -> dict[str, float]:
        stream_ids = rollout["stream_ids"]
        time_steps = rollout["reward"].shape[0]
        metrics: dict[str, float] = {}
        configured_weights = {
            "phase0": float(self.cfg.phase0_fraction),
            "curriculum": 1.0 - float(self.cfg.phase0_fraction),
        }
        for name, stream_id in (
            ("phase0", PHASE0_STREAM),
            ("curriculum", CURRICULUM_STREAM),
        ):
            env_mask = stream_ids == stream_id
            transition_mask = env_mask.unsqueeze(0).expand(
                time_steps, -1
            )
            stream_done = rollout["done"] & transition_mask
            stream_failure = rollout["failure"] & transition_mask
            stream_timeout = rollout["timeout"] & transition_mask
            stream_complete = rollout["motion_complete"] & transition_mask
            env_count = int(env_mask.sum().item())
            transition_count = int(transition_mask.sum().item())
            terminal_count = int(stream_done.sum().item())
            failure_count = int(stream_failure.sum().item())
            timeout_count = int(stream_timeout.sum().item())
            complete_count = int(stream_complete.sum().item())
            returns = self._stream_return_buffers[stream_id]
            lengths = self._stream_length_buffers[stream_id]
            prefix = f"stream/{name}"
            metrics.update(
                {
                    f"{prefix}/env_count": float(env_count),
                    f"{prefix}/env_fraction": float(
                        env_count / max(stream_ids.numel(), 1)
                    ),
                    f"{prefix}/configured_objective_weight": configured_weights[name],
                    f"{prefix}/transition_count": float(transition_count),
                    f"{prefix}/episode_count": float(len(returns)),
                    f"{prefix}/episode_return_mean": float(
                        sum(returns) / len(returns) if returns else 0.0
                    ),
                    f"{prefix}/episode_length_mean": float(
                        sum(lengths) / len(lengths) if lengths else 0.0
                    ),
                    f"{prefix}/terminal_count": float(terminal_count),
                    f"{prefix}/failure_count": float(failure_count),
                    f"{prefix}/timeout_count": float(timeout_count),
                    f"{prefix}/motion_complete_count": float(complete_count),
                    f"{prefix}/failure_rate": float(
                        failure_count / max(terminal_count, 1)
                    ),
                    f"{prefix}/completion_rate": float(
                        complete_count / max(terminal_count, 1)
                    ),
                    f"{prefix}/sampler_failure_eligible": float(
                        stream_id == CURRICULUM_STREAM
                    ),
                }
            )
            start_phases = rollout["collection_start_phases"][env_mask]
            metrics.update(
                {
                    f"{prefix}/collection_start_mean": float(
                        start_phases.float().mean().item()
                    ),
                    f"{prefix}/collection_start_min": float(
                        start_phases.min().item()
                    ),
                    f"{prefix}/collection_start_max": float(
                        start_phases.max().item()
                    ),
                }
            )
            failure_phases = rollout["terminal_phase"][stream_failure]
            if failure_phases.numel() > 0:
                q = torch.quantile(
                    failure_phases.float(),
                    torch.tensor([0.50, 0.95], device=failure_phases.device),
                )
                metrics[f"{prefix}/failure_phase_mean"] = float(
                    failure_phases.float().mean().item()
                )
                metrics[f"{prefix}/failure_phase_p50"] = float(q[0].item())
                metrics[f"{prefix}/failure_phase_p95"] = float(q[1].item())
            else:
                metrics[f"{prefix}/failure_phase_mean"] = -1.0
                metrics[f"{prefix}/failure_phase_p50"] = -1.0
                metrics[f"{prefix}/failure_phase_p95"] = -1.0
        return metrics

    @torch.no_grad()
    def _cps_metrics(self) -> dict[str, float]:
        eta = self._policy.eta()
        self._ensure_finite("cps/eta", eta)
        eta_value = float(eta.item())
        if not 0.0 < eta_value < 1.0:
            raise RuntimeError("CPS eta left (0, 1)")
        metrics = {
            "cps/eta": eta_value,
            "cps/eta_raw": float(self._policy.cps_eta_raw.item()),
            "health/covariance_psd": 1.0,
        }
        for step_index in range(int(self.cfg.flow_steps)):
            (
                _,
                _,
                covariance,
                cholesky,
                _,
                mean_variance,
            ) = self._policy.covariance_factors(step_index)
            preserved, noise = self._policy.cps_step_coefficients(
                step_index, covariance
            )
            eigenvalues = torch.linalg.eigvalsh(covariance)
            for name, tensor in (
                ("covariance", covariance),
                ("cholesky", cholesky),
                ("eigenvalues", eigenvalues),
                ("mean_variance", mean_variance),
            ):
                self._ensure_finite(f"cps/flow_{step_index}/{name}", tensor)
            if float(eigenvalues.min().item()) <= 0.0:
                raise RuntimeError("CPS covariance is not positive definite")
            if abs(float(mean_variance.item()) - 1.0) > 1.0e-5:
                raise RuntimeError("CPS covariance shape lost normalization")
            probabilities = eigenvalues / eigenvalues.sum()
            effective_rank = torch.exp(
                -(probabilities * torch.log(probabilities.clamp_min(1.0e-12))).sum()
            )
            prefix = f"cps/flow_{step_index}"
            metrics.update(
                {
                    f"{prefix}/preserved_coefficient": float(preserved.item()),
                    f"{prefix}/noise_coefficient": float(noise.item()),
                    f"{prefix}/covariance_trace": float(
                        torch.trace(covariance).item()
                    ),
                    f"{prefix}/shape_mean_variance": float(
                        mean_variance.item()
                    ),
                    f"{prefix}/eigenvalue_min": float(eigenvalues.min().item()),
                    f"{prefix}/eigenvalue_max": float(eigenvalues.max().item()),
                    f"{prefix}/condition_number": float(
                        (eigenvalues.max() / eigenvalues.min()).item()
                    ),
                    f"{prefix}/effective_rank": float(effective_rank.item()),
                    f"{prefix}/cholesky_diag_min": float(
                        torch.diagonal(cholesky).min().item()
                    ),
                    f"{prefix}/cholesky_diag_max": float(
                        torch.diagonal(cholesky).max().item()
                    ),
                }
            )
        return metrics

    def _soft_health_metrics(self, metrics: dict[str, float]) -> dict[str, float]:
        kl = float(metrics["policy/kl"])
        reward_mean = float(metrics["reward/total/mean"])
        self._high_kl_streak = self._high_kl_streak + 1 if kl > 0.04 else 0
        self._nonpositive_reward_streak = (
            self._nonpositive_reward_streak + 1
            if reward_mean <= 0.0
            else 0
        )
        ratio = float(metrics["policy/ratio_mean"])
        clip_fraction = float(metrics["policy/clip_fraction"])
        return {
            "health/high_kl_streak": float(self._high_kl_streak),
            "health/high_kl_warning": float(self._high_kl_streak >= 3),
            "health/high_clip_warning": float(clip_fraction > 0.5),
            "health/ratio_warning": float(ratio < 0.8 or ratio > 1.2),
            "health/nonpositive_reward_streak": float(
                self._nonpositive_reward_streak
            ),
            "health/nonpositive_reward_warning": float(
                self._nonpositive_reward_streak >= 5
            ),
        }

    def update(self, rollout: dict, collect_time: float) -> dict[str, float]:
        update_start = time.perf_counter()
        removed = self._removed_state_keys(rollout)
        if removed:
            raise RuntimeError(
                "rollout contains removed state: " + ", ".join(sorted(removed))
            )
        for name in (
            "reward",
            "values",
            "next_values",
            "advantages",
            "raw_advantages",
            "value_targets",
        ):
            self._ensure_finite(f"rollout/{name}", rollout[name])
        if float(rollout["reward_decomposition_abs_max"]) > 1.0e-6:
            raise RuntimeError("reward decomposition identity failed")
        if float(rollout["action_bound_violation_max"]) > 1.0e-6:
            raise RuntimeError("action-bound contract failed")
        if float(rollout["reward"].max().item()) > 0.100001:
            raise RuntimeError("reward maximum contract failed")

        actor_start = time.perf_counter()
        actor_metrics = self._actor_update(rollout)
        actor_time = time.perf_counter() - actor_start
        critic_start = time.perf_counter()
        critic_metrics = self._critic_update(rollout)
        critic_time = time.perf_counter() - critic_start

        with torch.no_grad():
            self._update_normalizer(
                self.actor_obs_normalizer, rollout["actor_obs_raw"]
            )
            self._update_normalizer(
                self.critic_obs_normalizer, rollout["critic_obs_raw"]
            )

        metrics: dict[str, float] = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        metrics.update(self._stream_metrics(rollout))
        metrics.update(self.phase0_attempts.metrics())
        metrics.update(_distribution_metrics("reward/total", rollout["reward"]))
        metrics.update(
            _distribution_metrics(
                "credit/raw_advantage", rollout["raw_advantages"]
            )
        )
        metrics.update(
            _distribution_metrics(
                "credit/actor_advantage", rollout["advantages"]
            )
        )
        metrics.update(_distribution_metrics("critic/value", rollout["values"]))
        metrics.update(
            _distribution_metrics(
                "critic/return_target", rollout["value_targets"]
            )
        )
        for name in (*RAW_POSE_TERMS, *RAW_PENALTY_TERMS):
            metrics.update(
                _distribution_metrics(
                    f"reward/raw/{REWARD_METRIC_NAMES[name]}",
                    rollout["reward_raw_terms"][name],
                )
            )
        for name in REWARD_TERM_WEIGHTS:
            metrics.update(
                _distribution_metrics(
                    f"reward/contribution/{REWARD_METRIC_NAMES[name]}",
                    rollout["reward_contributions"][name],
                )
            )

        action_abs = rollout["actions"].abs()
        mean_action_abs = rollout["mean_actions"].abs()
        exploration_displacement = (
            rollout["actions"] - rollout["mean_actions"]
        )
        metrics.update(_distribution_metrics("action/absolute", action_abs))
        metrics.update(
            _distribution_metrics("action/deterministic_absolute", mean_action_abs)
        )
        metrics.update(
            _distribution_metrics(
                "action/exploration_displacement", exploration_displacement
            )
        )
        metrics.update(
            _distribution_metrics(
                "action/delta",
                rollout["action_delta"].abs().mean(dim=-1),
                rollout["action_delta_valid"],
            )
        )
        metrics.update(
            _distribution_metrics(
                "action/d2",
                rollout["action_d2"].abs().mean(dim=-1),
                rollout["action_d2_valid"],
            )
        )
        metrics.update(
            _distribution_metrics(
                "action/reset_first_delta",
                rollout["action_delta"].abs().mean(dim=-1),
                rollout["reset_action"],
            )
        )
        for name in (
            "joint_vel_jump",
            "root_lin_vel_jump",
            "root_ang_vel_jump",
        ):
            metrics.update(
                _distribution_metrics(
                    f"dynamics/{name}", rollout[name]
                )
            )

        innovation_rms = rollout["cps_innovation_rms"]
        self._ensure_finite("cps/innovation_rms", innovation_rms)
        for step_index, value in enumerate(innovation_rms):
            metrics[f"cps/flow_{step_index}/innovation_rms"] = float(
                value.item()
            )
        metrics.update(self._cps_metrics())
        metrics.update(
            {
                "reward/decomposition_identity_abs_max": float(
                    rollout["reward_decomposition_abs_max"]
                ),
                "rollout/transition_count": float(rollout["reward"].numel()),
                "rollout/done_fraction": float(rollout["done"].float().mean().item()),
                "rollout/failure_fraction": float(
                    rollout["failure"].float().mean().item()
                ),
                "rollout/timeout_fraction": float(
                    rollout["timeout"].float().mean().item()
                ),
                "rollout/motion_complete_fraction": float(
                    rollout["motion_complete"].float().mean().item()
                ),
                "rollout/bootstrap_fraction": float(
                    rollout["bootstrap_mask"].float().mean().item()
                ),
                "rollout/trace_fraction": float(
                    rollout["trace_mask"].float().mean().item()
                ),
                "phase/start_mean": float(
                    rollout["collection_start_phases"].float().mean().item()
                ),
                "phase/start_min": float(
                    rollout["collection_start_phases"].min().item()
                ),
                "phase/start_max": float(
                    rollout["collection_start_phases"].max().item()
                ),
                "action/saturation_fraction": float(
                    (
                        action_abs
                        >= 0.95 * float(self.cfg.action_limit)
                    )
                    .float()
                    .mean()
                    .item()
                ),
                "action/policy_bound_violation_max": float(
                    rollout["action_bound_violation_max"]
                ),
                "intervention/edge_count": float(
                    rollout["intervention_edge"].sum().item()
                ),
                "timing/collect_s": float(collect_time),
                "timing/actor_update_s": float(actor_time),
                "timing/critic_update_s": float(critic_time),
                "timing/update_s": float(time.perf_counter() - update_start),
                "system/cuda_peak_allocated_gib": float(
                    torch.cuda.max_memory_allocated(self.env.device) / (1024**3)
                    if torch.cuda.is_available()
                    else 0.0
                ),
            }
        )
        for name, value in self.env.adaptive_sampling_stats().items():
            scalar = float(value)
            if not math.isfinite(scalar):
                raise FloatingPointError(
                    f"adaptive sampler statistic {name!r} is non-finite"
                )
            metrics[f"sampler/{name}"] = scalar
        self._ensure_parameters_finite("actor/final", self._policy)
        self._ensure_parameters_finite("critic/final", self.critic)
        metrics["system/parameters_finite"] = 1.0
        metrics.update(self._soft_health_metrics(metrics))
        return metrics

    # ------------------------------------------------------------------
    # Console
    # ------------------------------------------------------------------
    def log(self, update_idx: int, max_updates: int, metrics: dict) -> None:
        print(
            f"[POLICY] update={update_idx}/{max_updates} "
            f"loss={metrics.get('policy/loss', 0.0):.5f} "
            f"kl={metrics.get('policy/kl', 0.0):.6f} "
            f"ratio={metrics.get('policy/ratio_mean', 0.0):.4f} "
            f"ratio_p95={metrics.get('policy/ratio_p95', 0.0):.4f} "
            f"clip={metrics.get('policy/clip_fraction', 0.0):.4f} "
            f"grad={metrics.get('policy/grad_norm', 0.0):.4f} "
            f"delta={metrics.get('policy/parameter_delta_l2', 0.0):.3e} "
            f"lr={metrics.get('policy/lr', 0.0):.3e}",
            flush=True,
        )
        print(
            f"[CPS] eta={metrics.get('cps/eta', 0.0):.6f} "
            f"innovation0={metrics.get('cps/flow_0/innovation_rms', 0.0):.5f} "
            f"cov_grad={metrics.get('policy/covariance_grad_norm', 0.0):.3e} "
            f"cov_delta={metrics.get('policy/covariance_parameter_delta_l2', 0.0):.3e} "
            f"eta_grad={metrics.get('policy/eta_grad_abs', 0.0):.3e} "
            f"eta_delta={metrics.get('policy/eta_parameter_delta_abs', 0.0):.3e} "
            f"logprob_err={metrics.get('policy/log_prob_recompute_abs_max', 0.0):.3e}",
            flush=True,
        )
        print(
            f"[ACTION] abs_rms={metrics.get('action/absolute/rms', 0.0):.5f} "
            f"mean_rms={metrics.get('action/deterministic_absolute/rms', 0.0):.5f} "
            f"explore_rms={metrics.get('action/exploration_displacement/rms', 0.0):.5f} "
            f"delta={metrics.get('action/delta/mean', 0.0):.5f} "
            f"d2={metrics.get('action/d2/mean', 0.0):.5f} "
            f"sat={metrics.get('action/saturation_fraction', 0.0):.5f}",
            flush=True,
        )
        print(
            f"[CRITIC] loss={metrics.get('critic/value_loss', 0.0):.5f} "
            f"value={metrics.get('critic/value/mean', 0.0):.5f} "
            f"return={metrics.get('critic/return_target/mean', 0.0):.5f} "
            f"rmse={metrics.get('critic/rmse', 0.0):.5f} "
            f"ev={metrics.get('critic/explained_variance', 0.0):.4f} "
            f"grad={metrics.get('critic/grad_norm', 0.0):.4f} "
            f"delta={metrics.get('critic/parameter_delta_l2', 0.0):.3e}",
            flush=True,
        )
        print(
            f"[REWARD] total={metrics.get('reward/total/mean', 0.0):.5f} "
            f"anchor_pos={metrics.get('reward/raw/anchor_pos/mean', 0.0):.4f} "
            f"anchor_ori={metrics.get('reward/raw/anchor_ori/mean', 0.0):.4f} "
            f"body_pos={metrics.get('reward/raw/body_pos/mean', 0.0):.4f} "
            f"body_ori={metrics.get('reward/raw/body_ori/mean', 0.0):.4f} "
            f"identity={metrics.get('reward/decomposition_identity_abs_max', 0.0):.3e}",
            flush=True,
        )
        print(
            f"[PENALTY] action_rate={metrics.get('reward/raw/action_rate/mean', 0.0):.5f} "
            f"joint_limit={metrics.get('reward/raw/joint_limit/mean', 0.0):.5f} "
            f"contacts={metrics.get('reward/raw/undesired_contacts/mean', 0.0):.5f} "
            f"action_rate_c={metrics.get('reward/contribution/action_rate/mean', 0.0):.5f} "
            f"joint_limit_c={metrics.get('reward/contribution/joint_limit/mean', 0.0):.5f} "
            f"contacts_c={metrics.get('reward/contribution/undesired_contacts/mean', 0.0):.5f}",
            flush=True,
        )
        print(
            f"[HEALTH] finite={metrics.get('system/parameters_finite', 0.0):.0f} "
            f"action_violation={metrics.get('action/policy_bound_violation_max', 0.0):.3e} "
            f"phase0_return={metrics.get('stream/phase0/episode_return_mean', 0.0):.4f} "
            f"phase0_length={metrics.get('stream/phase0/episode_length_mean', 0.0):.1f} "
            f"phase0_fail={metrics.get('stream/phase0/failure_rate', 0.0):.4f} "
            f"curr_return={metrics.get('stream/curriculum/episode_return_mean', 0.0):.4f} "
            f"curr_length={metrics.get('stream/curriculum/episode_length_mean', 0.0):.1f} "
            f"curr_fail={metrics.get('stream/curriculum/failure_rate', 0.0):.4f}",
            flush=True,
        )

    def log_banner(self) -> None:
        print(
            "[POLICY] method=fixed_reward control=closed_loop_50hz "
            "actor=flow_mlp action=absolute_tanh ppo=primitive_path",
            flush=True,
        )
        print(
            "[CPS] covariance=joint_diagonal_plus_low_rank "
            "shape=unit_mean_variance scale=learned_eta",
            flush=True,
        )
        print(
            f"[CRITIC] type=state_only_scalar_mlp input={self.critic_obs_dim}",
            flush=True,
        )
        print(
            "[REWARD] formula=(0.5*anchor_pos+0.5*anchor_ori+"
            "2*body_pos+2*body_ori-0.1*action_rate-10*joint_limit-"
            "0.1*undesired_contacts)*dt",
            flush=True,
        )
        print(
            "[HEALTH] guards=state,observation,action,reward,value,return,"
            "advantage,loss,gradient,parameter,covariance "
            f"fixed_reward_schema={FIXED_REWARD_CHECKPOINT_CONTRACT['fixed_reward_schema_version']}",
            flush=True,
        )
