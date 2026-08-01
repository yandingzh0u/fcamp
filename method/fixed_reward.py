"""Fixed-reward G1 training with the audited HOLOSOMA G1 WBT PPO.

Only the environment adapter (reward, reset streams, termination and Isaac
Lab stepping) is MimicKit-specific.  The actor, critic, probability law,
normalization, rollout storage, GAE, minibatching, losses, optimizer order,
KL scheduler and numerical hyperparameters follow HOLOSOMA commit
``c5c836c68f423ac4565f57801ff4ff47ea56e5ac``.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque

import torch
from torch import nn
from torch.distributions import Normal, kl_divergence

from components.credit.task_credit import resolve_terminal_masks
from components.rollout.fixed_reward_contract import (
    FIXED_REWARD_CHECKPOINT_CONTRACT,
)
from components.rollout.training_streams import (
    CURRICULUM_STREAM,
    PHASE0_STREAM,
    Phase0AttemptTracker,
    Phase0CurriculumStreams,
)
from models.holosoma_ppo import (
    EmpiricalNormalization,
    PPOActor,
    PPOCritic,
    RolloutStorage,
)


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

HOLOSOMA_UPSTREAM_COMMIT = "c5c836c68f423ac4565f57801ff4ff47ea56e5ac"
HOLOSOMA_SOURCE_SHA256 = {
    "ppo.py": "3da7ce871ad98b6400d663721825e4b345f4c3198371b98e74e57c6113ef0e10",
    "ppo_modules.py": "508ff6485ec1ea3ef33aee749cb7e069623b183d06ea6a17a37cedb516575024",
    "data_utils.py": "ffd8a69af140becb98450c954139b62a9b16b33c87e83c96edd2179be45f85de",
    "algo.py": "c520db1f660de7ac1090409821668fe9df640aa99bc4651bcb5cc56a02d90310",
    "g1_experiment.py": "592ebec1e2bfa18d2c5862a3aeba11d3682b80a8e45db9f4c90380345eaa87bc",
}
HOLOSOMA_PPO_MANIFEST = {
    "activation": "ELU",
    "actor_hidden_dims": [512, 256, 128],
    "actor_learning_rate": 0.001,
    "actor_weight_decay": 0.0,
    "action_clip_value": 100.0,
    "critic_hidden_dims": [512, 256, 128],
    "critic_learning_rate": 0.001,
    "critic_weight_decay": 0.0,
    "desired_kl": 0.01,
    "empirical_normalization": True,
    "entropy_coef": 0.005,
    "gamma": 0.99,
    "init_noise_std": 1.0,
    "lam": 0.95,
    "max_grad_norm": 1.0,
    "num_learning_epochs": 5,
    "num_mini_batches": 4,
    "num_steps_per_env": 24,
    "schedule": "adaptive",
    "use_symmetry": False,
    "value_loss_coef": 1.0,
    "clip_param": 0.2,
}
HOLOSOMA_STATIC_PARITY_SHA256 = hashlib.sha256(
    json.dumps(
        HOLOSOMA_PPO_MANIFEST,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def _distribution_metrics(prefix: str, values: torch.Tensor) -> dict[str, float]:
    flat = values.detach().float().reshape(-1)
    if flat.numel() == 0:
        raise RuntimeError(f"{prefix} has no samples")
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


class FixedRewardPPO:
    """HOLOSOMA PPO connected to the fixed-reward G1 environment."""

    def __init__(self, cfg, env) -> None:
        self.cfg = cfg
        self.env = env

    def build(self) -> None:
        cfg = self.cfg
        env = self.env
        self.num_act = int(env.action_dim)
        self.actor_obs_dim = int(env.observation_dim)
        self.critic_obs_dim = int(env.critic_observation_dim)

        self.actor = PPOActor(
            observation_dim=self.actor_obs_dim,
            hidden_dims=tuple(cfg.actor_hidden_dims),
            activation=str(cfg.activation),
            num_actions=self.num_act,
            init_noise_std=float(cfg.init_noise_std),
        ).to(env.device)
        self.critic = PPOCritic(
            observation_dim=self.critic_obs_dim,
            hidden_dims=tuple(cfg.critic_hidden_dims),
            activation=str(cfg.activation),
        ).to(env.device)
        if not bool(cfg.empirical_normalization):
            raise RuntimeError("Audited HOLOSOMA G1 WBT PPO requires normalization")
        self.actor_obs_normalizer = EmpiricalNormalization(
            self.actor_obs_dim,
            env.device,
        )
        self.critic_obs_normalizer = EmpiricalNormalization(
            self.critic_obs_dim,
            env.device,
        )

        self.actor_learning_rate = float(cfg.actor_learning_rate)
        self.critic_learning_rate = float(cfg.critic_learning_rate)
        self.max_actor_learning_rate = max(self.actor_learning_rate, 1.0e-2)
        self.min_actor_learning_rate = min(self.actor_learning_rate, 1.0e-5)
        self.max_critic_learning_rate = max(self.critic_learning_rate, 1.0e-2)
        self.min_critic_learning_rate = min(self.critic_learning_rate, 1.0e-5)
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=self.actor_learning_rate,
            weight_decay=float(cfg.actor_weight_decay),
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=self.critic_learning_rate,
            weight_decay=float(cfg.critic_weight_decay),
        )
        self._policy_module = nn.ModuleDict(
            {
                "actor": self.actor,
                "actor_obs_normalizer": self.actor_obs_normalizer,
                "critic": self.critic,
                "critic_obs_normalizer": self.critic_obs_normalizer,
            }
        )

        action_clip = float(cfg.action_clip_value)
        self.action_low = torch.full(
            (self.num_act,), -action_clip, device=env.device
        )
        self.action_high = torch.full(
            (self.num_act,), action_clip, device=env.device
        )
        env.enable_strict_action_contract(self.action_low, self.action_high)

        self.storage = RolloutStorage(
            env.num_envs,
            int(cfg.num_steps_per_env),
            device=env.device,
        )
        self.storage.register("actor_obs", (self.actor_obs_dim,), torch.float)
        self.storage.register("critic_obs", (self.critic_obs_dim,), torch.float)
        for key, shape, dtype in (
            ("actions", (self.num_act,), torch.float),
            ("rewards", (1,), torch.float),
            ("dones", (1,), torch.bool),
            ("values", (1,), torch.float),
            ("returns", (1,), torch.float),
            ("advantages", (1,), torch.float),
            ("actions_log_prob", (1,), torch.float),
            ("action_mean", (self.num_act,), torch.float),
            ("action_sigma", (self.num_act,), torch.float),
        ):
            self.storage.register(key, shape, dtype)

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
        self._actor_optimizer_steps_total = 0
        self._critic_optimizer_steps_total = 0
        self._last_minibatch_fingerprint = ""
        self._obs = None
        self._critic_obs = None

    @property
    def policy(self) -> nn.Module:
        return self._policy_module

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self.actor_optimizer

    @staticmethod
    def _ensure_finite(name: str, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a tensor")
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"{name} contains non-finite values")

    @staticmethod
    def _snapshot_parameters(module: nn.Module) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
        }

    @staticmethod
    def _parameter_delta(
        before: dict[str, torch.Tensor], module: nn.Module
    ) -> float:
        total = torch.zeros((), device=next(module.parameters()).device)
        for name, parameter in module.named_parameters():
            total += (parameter.detach() - before[name]).double().square().sum()
        return float(torch.sqrt(total).item())

    def _ensure_runtime_state_finite(self) -> None:
        data = getattr(getattr(self.env, "robot", None), "data", None)
        if data is None:
            return
        for name in ("joint_pos", "joint_vel", "root_state_w", "body_state_w"):
            value = getattr(data, name, None)
            if torch.is_tensor(value):
                self._ensure_finite(f"state/{name}", value)

    def _ensure_train_state_finite(self) -> None:
        for module_name, module in (
            ("actor", self.actor),
            ("critic", self.critic),
        ):
            for parameter_name, parameter in module.named_parameters():
                self._ensure_finite(
                    f"{module_name}/parameter/{parameter_name}", parameter
                )
                if parameter.grad is not None:
                    self._ensure_finite(
                        f"{module_name}/gradient/{parameter_name}",
                        parameter.grad,
                    )
        for optimizer_name, optimizer in (
            ("actor", self.actor_optimizer),
            ("critic", self.critic_optimizer),
        ):
            for state_index, state in enumerate(optimizer.state.values()):
                for state_name, value in state.items():
                    if torch.is_tensor(value):
                        self._ensure_finite(
                            f"{optimizer_name}_optimizer/{state_index}/{state_name}",
                            value,
                        )

    # ------------------------------------------------------------------
    # Evaluation adapter
    # ------------------------------------------------------------------
    @torch.no_grad()
    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = self.actor_obs_normalizer(observation, update=False)
        action = self.actor.act_inference(normalized)
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
    # Checkpoint contract
    # ------------------------------------------------------------------
    def _validate_checkpoint_contract(self, state: object) -> None:
        if not isinstance(state, dict):
            raise ValueError("fixed_reward checkpoint lacks schema 17")
        for key, expected in FIXED_REWARD_CHECKPOINT_CONTRACT.items():
            if state.get(key) != expected:
                raise ValueError(
                    "fixed_reward checkpoint semantic contract mismatch: "
                    f"{key} expected={expected!r}, actual={state.get(key)!r}"
                )

    def validate_checkpoint_payload(self, payload: dict) -> None:
        self._validate_checkpoint_contract(payload.get("algo_state"))

    def extra_checkpoint_state(self) -> dict:
        state = {
            **FIXED_REWARD_CHECKPOINT_CONTRACT,
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "actor_learning_rate": float(self.actor_learning_rate),
            "critic_learning_rate": float(self.critic_learning_rate),
            "stream_ids": self.training_streams.stream_ids.detach().cpu(),
            "phase0_stream_count": int(
                self.training_streams.phase0_ids.numel()
            ),
            "phase0_stream_fraction": float(self.cfg.phase0_fraction),
            "phase0_attempt_tracker": self.phase0_attempts.state_dict(),
            "actor_optimizer_steps_total": int(
                self._actor_optimizer_steps_total
            ),
            "critic_optimizer_steps_total": int(
                self._critic_optimizer_steps_total
            ),
        }
        return state

    def load_extra_checkpoint_state(
        self, payload: dict, reset_optimizer: bool = False
    ) -> None:
        self._validate_checkpoint_contract(payload)
        if not reset_optimizer:
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
            self.actor_learning_rate = float(payload["actor_learning_rate"])
            self.critic_learning_rate = float(payload["critic_learning_rate"])
            for group in self.actor_optimizer.param_groups:
                group["lr"] = self.actor_learning_rate
            for group in self.critic_optimizer.param_groups:
                group["lr"] = self.critic_learning_rate
            self._actor_optimizer_steps_total = int(
                payload["actor_optimizer_steps_total"]
            )
            self._critic_optimizer_steps_total = int(
                payload["critic_optimizer_steps_total"]
            )
        saved_streams = torch.as_tensor(
            payload["stream_ids"],
            device=self.env.device,
            dtype=self.training_streams.stream_ids.dtype,
        )
        if not torch.equal(saved_streams, self.training_streams.stream_ids):
            raise ValueError("checkpoint stream assignment differs")
        if int(payload["phase0_stream_count"]) != int(
            self.training_streams.phase0_ids.numel()
        ):
            raise ValueError("checkpoint phase0 stream count differs")
        if not math.isclose(
            float(payload["phase0_stream_fraction"]),
            float(self.cfg.phase0_fraction),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("checkpoint stream fraction differs")
        self.phase0_attempts.load_state_dict(
            payload["phase0_attempt_tracker"]
        )

    # ------------------------------------------------------------------
    # Reset and episode accounting
    # ------------------------------------------------------------------
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
            phase_indices=phases,
            reset_stream_ids=reset_streams,
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
        self.actor.train()
        self.critic.train()
        self.actor_obs_normalizer.train()
        self.critic_obs_normalizer.train()
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
        if self._obs is None:
            raise RuntimeError("algorithm has not been reset")
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

    def _reward_contributions(
        self, terms: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        for name, weight in REWARD_TERM_WEIGHTS.items():
            if name not in terms:
                raise KeyError(f"reward terms are missing {name}")
            result[name] = float(weight) * terms[name] * float(self.env.dt)
        return result

    # ------------------------------------------------------------------
    # HOLOSOMA rollout and GAE
    # ------------------------------------------------------------------
    def collect(self, current_obs: torch.Tensor) -> dict:
        env = self.env
        cfg = self.cfg
        time_steps = int(cfg.num_steps_per_env)
        num_envs = int(env.num_envs)
        device = env.device
        if self.storage.step != 0:
            raise RuntimeError("rollout storage was not cleared after update")

        reward_raw = torch.empty(time_steps, num_envs, device=device)
        reward_train = torch.empty_like(reward_raw)
        done = torch.zeros(
            time_steps, num_envs, dtype=torch.bool, device=device
        )
        failure = torch.zeros_like(done)
        timeout = torch.zeros_like(done)
        motion_complete = torch.zeros_like(done)
        action_raw = torch.empty(
            time_steps, num_envs, self.num_act, device=device
        )
        action_applied = torch.empty_like(action_raw)
        action_mean = torch.empty_like(action_raw)
        action_delta = torch.empty_like(action_raw)
        action_d2 = torch.empty_like(action_raw)
        action_delta_valid = torch.zeros_like(done)
        action_d2_valid = torch.zeros_like(done)
        joint_vel_jump = torch.empty_like(reward_raw)
        root_lin_vel_jump = torch.empty_like(reward_raw)
        root_ang_vel_jump = torch.empty_like(reward_raw)
        raw_terms = {
            name: torch.empty_like(reward_raw)
            for name in (*RAW_POSE_TERMS, *RAW_PENALTY_TERMS)
        }
        contributions = {
            name: torch.empty_like(reward_raw)
            for name in REWARD_TERM_WEIGHTS
        }
        terminal_phase = torch.full(
            (time_steps, num_envs),
            -1.0,
            dtype=torch.float32,
            device=device,
        )
        observation = current_obs
        if self._critic_obs is None:
            raise RuntimeError("critic observation has not been initialized")
        critic_observation = self._critic_obs
        collection_start_phases = env.phase_steps.detach().clone()
        reward_identity_abs_max = 0.0
        raw_clip_count = 0

        # HOLOSOMA wraps collection in inference_mode.  MimicKit's environment
        # replaces persistent phase/sampler tensors while stepping, so its
        # equivalent adapter must use no_grad to keep those tensors mutable
        # during later validation and reset.  Policy outputs and probabilities
        # are otherwise identical and remain detached from autograd.
        with torch.no_grad():
            for step_index in range(time_steps):
                self._ensure_finite("actor_observation/raw", observation)
                self._ensure_finite(
                    "critic_observation/raw", critic_observation
                )
                normalized_actor_obs = self.actor_obs_normalizer(observation)
                normalized_critic_obs = self.critic_obs_normalizer(
                    critic_observation
                )
                sampled_action = self.actor.act(normalized_actor_obs)
                values = self.critic.evaluate(normalized_critic_obs).detach()
                log_prob = self.actor.get_actions_log_prob(sampled_action).detach()
                means = self.actor.action_mean.detach()
                sigmas = self.actor.action_std.detach()
                for name, tensor in (
                    ("action", sampled_action),
                    ("value", values),
                    ("log_prob", log_prob),
                    ("mean", means),
                    ("sigma", sigmas),
                ):
                    self._ensure_finite(f"policy/{name}", tensor)

                applied_action = torch.clamp(
                    sampled_action,
                    -float(cfg.action_clip_value),
                    float(cfg.action_clip_value),
                )
                raw_clip_count += int((sampled_action != applied_action).sum().item())
                previous_action = env.last_action.detach().clone()
                delta = applied_action - previous_action
                second_delta = delta - self._previous_action_delta
                has_delta = self._has_action_delta.clone()
                self._previous_action_delta.copy_(delta)
                self._has_action_delta.fill_(True)

                _, joint_vel_before = env.get_action_joint_state()
                root_velocity_before = env.get_mimic_root_velocity_w()
                (
                    next_observation,
                    step_reward,
                    step_done,
                    info,
                ) = env.step(sampled_action)
                terminal_critic_obs = env.get_critic_observation()
                _, joint_vel_after = env.get_action_joint_state()
                root_velocity_after = env.get_mimic_root_velocity_w()
                self._ensure_runtime_state_finite()
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
                step_failure, step_timeout, step_complete = resolve_terminal_masks(
                    step_done.bool(),
                    done_terms["time_out"].bool(),
                    done_terms["motion_complete"].bool(),
                    failures,
                )
                self.phase0_attempts.observe_step(
                    torch.ones_like(step_done, dtype=torch.bool),
                    step_done.bool(),
                    step_failure,
                    step_timeout,
                    step_complete,
                )

                final_rewards = torch.zeros_like(step_reward)
                if bool(step_timeout.any()):
                    final_critic_normalized = self.critic_obs_normalizer(
                        terminal_critic_obs,
                        update=False,
                    )
                    final_values = self.critic.evaluate(
                        final_critic_normalized
                    ).detach()
                    final_rewards += float(cfg.gamma) * torch.squeeze(
                        final_values
                        * step_timeout.unsqueeze(1).to(device=device),
                        1,
                    )
                stored_reward = step_reward + final_rewards

                self.storage.add(
                    actor_obs=normalized_actor_obs,
                    critic_obs=normalized_critic_obs,
                    actions=sampled_action,
                    values=values,
                    actions_log_prob=log_prob.unsqueeze(1),
                    action_mean=means,
                    action_sigma=sigmas,
                    rewards=stored_reward.view(-1, 1),
                    dones=step_done.bool().view(-1, 1),
                )
                action_raw[step_index] = sampled_action
                action_applied[step_index] = applied_action
                action_mean[step_index] = means
                action_delta[step_index] = delta
                action_d2[step_index] = second_delta
                action_delta_valid[step_index] = has_delta
                action_d2_valid[step_index] = has_delta
                reward_raw[step_index] = step_reward
                reward_train[step_index] = stored_reward
                done[step_index] = step_done.bool()
                failure[step_index] = step_failure
                timeout[step_index] = step_timeout
                motion_complete[step_index] = step_complete
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
                self._record_episode_stats(step_reward, step_done.bool())

                observation = next_observation
                critic_observation = terminal_critic_obs
                done_ids = step_done.bool().nonzero(
                    as_tuple=False
                ).squeeze(-1)
                if done_ids.numel() > 0:
                    reset_phases, reset_streams = self.training_streams.reset_phases(
                        done_ids,
                        lambda count: env.sample_phase_indices(count, horizon=1),
                    )
                    reset_observation = env.reset_envs(
                        done_ids,
                        phase_indices=reset_phases,
                        reset_stream_ids=reset_streams,
                    )
                    observation[done_ids] = reset_observation
                    critic_observation = env.get_critic_observation()
                    self._previous_action_delta[done_ids] = 0.0
                    self._has_action_delta[done_ids] = False
                    self.actor.reset(step_done)
                    self.critic.reset(step_done)
                    self.phase0_attempts.start(done_ids)

            last_critic_obs = self.critic_obs_normalizer(
                critic_observation,
                update=False,
            )
            last_values = self.critic.evaluate(last_critic_obs).detach()
            returns, advantages = self._compute_returns_and_advantages(
                last_values,
                self.storage["values"],
                self.storage["dones"],
                self.storage["rewards"],
            )
            self.storage["returns"] = returns
            self.storage["advantages"] = advantages

        self._obs = observation
        self._critic_obs = critic_observation
        return {
            "storage": self.storage,
            "reward": reward_raw,
            "training_reward": reward_train,
            "done": done,
            "failure": failure,
            "timeout": timeout,
            "motion_complete": motion_complete,
            "terminal_phase": terminal_phase,
            "stream_ids": self.training_streams.stream_ids,
            "collection_start_phases": collection_start_phases,
            "actions": action_raw,
            "applied_actions": action_applied,
            "mean_actions": action_mean,
            "action_delta": action_delta,
            "action_d2": action_d2,
            "action_delta_valid": action_delta_valid,
            "action_d2_valid": action_d2_valid,
            "joint_vel_jump": joint_vel_jump,
            "root_lin_vel_jump": root_lin_vel_jump,
            "root_ang_vel_jump": root_ang_vel_jump,
            "reward_raw_terms": raw_terms,
            "reward_contributions": contributions,
            "reward_decomposition_abs_max": reward_identity_abs_max,
            "action_clip_count": raw_clip_count,
            "next_observation": observation,
        }

    def _compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        rewards: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantage: torch.Tensor | int = 0
        returns = torch.zeros_like(values)
        num_steps = returns.shape[0]
        for step in reversed(range(num_steps)):
            next_values = last_values if step == num_steps - 1 else values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = (
                rewards[step]
                + next_is_not_terminal * float(self.cfg.gamma) * next_values
                - values[step]
            )
            advantage = (
                delta
                + next_is_not_terminal
                * float(self.cfg.gamma)
                * float(self.cfg.lam)
                * advantage
            )
            returns[step] = advantage + values[step]
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1.0e-8
        )
        self._ensure_finite("gae/returns", returns)
        self._ensure_finite("gae/advantages", advantages)
        return returns, advantages

    # ------------------------------------------------------------------
    # Exact HOLOSOMA PPO update
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_kl_div(
        old_mu_batch: torch.Tensor,
        old_sigma_batch: torch.Tensor,
        mu_batch: torch.Tensor,
        sigma_batch: torch.Tensor,
    ) -> torch.Tensor:
        with torch.inference_mode():
            old_dist = Normal(old_mu_batch, old_sigma_batch)
            new_dist = Normal(mu_batch, sigma_batch)
            return kl_divergence(old_dist, new_dist).sum(-1).mean()

    def _update_learning_rate(self, kl_mean: torch.Tensor) -> None:
        if kl_mean > float(self.cfg.desired_kl) * 2.0:
            self.actor_learning_rate = max(
                self.min_actor_learning_rate,
                self.actor_learning_rate / 1.5,
            )
            self.critic_learning_rate = max(
                self.min_critic_learning_rate,
                self.critic_learning_rate / 1.5,
            )
        elif kl_mean < float(self.cfg.desired_kl) / 2.0 and kl_mean > 0.0:
            self.actor_learning_rate = min(
                self.max_actor_learning_rate,
                self.actor_learning_rate * 1.5,
            )
            self.critic_learning_rate = min(
                self.max_critic_learning_rate,
                self.critic_learning_rate * 1.5,
            )
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = self.actor_learning_rate
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = self.critic_learning_rate

    def _compute_ppo_loss(self, minibatch: dict[str, torch.Tensor]):
        actions_batch = minibatch["actions"]
        target_values_batch = minibatch["values"]
        advantages_batch = minibatch["advantages"]
        returns_batch = minibatch["returns"]
        old_actions_log_prob_batch = minibatch["actions_log_prob"]
        old_mu_batch = minibatch["action_mean"]
        old_sigma_batch = minibatch["action_sigma"]
        actor_obs = minibatch["actor_obs"]
        critic_obs = minibatch["critic_obs"]

        self.actor.act(actor_obs)
        value_batch = self.critic.evaluate(critic_obs)
        actions_log_prob_batch = self.actor.get_actions_log_prob(actions_batch)
        mu_batch = self.actor.action_mean
        sigma_batch = self.actor.action_std
        entropy_batch = self.actor.entropy

        kl_mean = self._compute_kl_div(
            old_mu_batch,
            old_sigma_batch,
            mu_batch,
            sigma_batch,
        )
        self._update_learning_rate(kl_mean)

        ratio = torch.exp(
            actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch)
        )
        surrogate = -torch.squeeze(advantages_batch) * ratio
        surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
            ratio,
            1.0 - float(self.cfg.clip_param),
            1.0 + float(self.cfg.clip_param),
        )
        surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

        value_clipped = target_values_batch + (
            value_batch - target_values_batch
        ).clamp(-float(self.cfg.clip_param), float(self.cfg.clip_param))
        value_losses = (value_batch - returns_batch).pow(2)
        value_losses_clipped = (value_clipped - returns_batch).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()

        symmetry_actor_loss = torch.tensor(0.0, device=self.env.device)
        symmetry_critic_loss = torch.tensor(0.0, device=self.env.device)
        entropy_loss = entropy_batch.mean()
        actor_loss = surrogate_loss - float(self.cfg.entropy_coef) * entropy_loss
        critic_loss = float(self.cfg.value_loss_coef) * value_loss
        clip_fraction = (
            torch.abs(ratio - 1.0) > float(self.cfg.clip_param)
        ).float().mean()
        return {
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "symmetry_actor_loss": symmetry_actor_loss,
            "symmetry_critic_loss": symmetry_critic_loss,
            "value_loss": value_loss,
            "surrogate_loss": surrogate_loss,
            "entropy_loss": entropy_loss,
            "kl_mean": kl_mean,
            "ratio_mean": ratio.mean(),
            "clip_fraction": clip_fraction,
        }

    def _training_step(self) -> dict[str, float]:
        cfg = self.cfg
        actor_before = self._snapshot_parameters(self.actor)
        critic_before = self._snapshot_parameters(self.critic)
        generator = self.storage.mini_batch_generator(
            int(cfg.num_mini_batches),
            int(cfg.num_learning_epochs),
        )
        loss_dict = {
            "Value": 0.0,
            "Surrogate": 0.0,
            "Entropy": 0.0,
            "KL": 0.0,
        }
        actor_grad_sum = 0.0
        critic_grad_sum = 0.0
        first_epoch_fingerprints: list[str] = []
        repeated_epoch_fingerprints: list[str] = []
        for update_number, minibatch in enumerate(generator):
            fingerprint = hashlib.sha256(
                minibatch["actions_log_prob"][:64]
                .detach()
                .cpu()
                .numpy()
                .tobytes()
            ).hexdigest()
            if update_number < int(cfg.num_mini_batches):
                first_epoch_fingerprints.append(fingerprint)
            elif update_number < 2 * int(cfg.num_mini_batches):
                repeated_epoch_fingerprints.append(fingerprint)

            ppo_loss_dict = self._compute_ppo_loss(minibatch)
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            ppo_loss = (
                ppo_loss_dict["actor_loss"]
                + ppo_loss_dict["critic_loss"]
            )
            self._ensure_finite("ppo/loss", ppo_loss)
            ppo_loss.backward()
            actor_grad = nn.utils.clip_grad_norm_(
                self.actor.parameters(), float(cfg.max_grad_norm)
            )
            critic_grad = nn.utils.clip_grad_norm_(
                self.critic.parameters(), float(cfg.max_grad_norm)
            )
            self._ensure_finite("ppo/actor_grad", torch.as_tensor(actor_grad))
            self._ensure_finite("ppo/critic_grad", torch.as_tensor(critic_grad))
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            self._actor_optimizer_steps_total += 1
            self._critic_optimizer_steps_total += 1
            actor_grad_sum += float(actor_grad)
            critic_grad_sum += float(critic_grad)

            loss_dict["Value"] += ppo_loss_dict.pop("value_loss").item()
            loss_dict["Surrogate"] += ppo_loss_dict.pop("surrogate_loss").item()
            loss_dict["Entropy"] += ppo_loss_dict.pop("entropy_loss").item()
            loss_dict["KL"] += ppo_loss_dict.pop("kl_mean").item()
            for key, loss in ppo_loss_dict.items():
                if key not in loss_dict:
                    loss_dict[key] = 0.0
                loss_dict[key] += (
                    loss.item() if torch.is_tensor(loss) else float(loss)
                )

        num_updates = int(cfg.num_learning_epochs) * int(cfg.num_mini_batches)
        if first_epoch_fingerprints != repeated_epoch_fingerprints:
            raise RuntimeError(
                "HOLOSOMA minibatch permutation was not reused across epochs"
            )
        self._last_minibatch_fingerprint = hashlib.sha256(
            "".join(first_epoch_fingerprints).encode("ascii")
        ).hexdigest()
        for key in loss_dict:
            loss_dict[key] /= num_updates
        loss_dict["actor_learning_rate"] = self.actor_learning_rate
        loss_dict["critic_learning_rate"] = self.critic_learning_rate
        loss_dict["actor_grad_norm"] = actor_grad_sum / num_updates
        loss_dict["critic_grad_norm"] = critic_grad_sum / num_updates
        loss_dict["actor_parameter_delta"] = self._parameter_delta(
            actor_before, self.actor
        )
        loss_dict["critic_parameter_delta"] = self._parameter_delta(
            critic_before, self.critic
        )
        loss_dict["optimizer_steps"] = float(num_updates)
        self._ensure_train_state_finite()
        return loss_dict

    # ------------------------------------------------------------------
    # Metrics and console
    # ------------------------------------------------------------------
    def _stream_metrics(self, rollout: dict) -> dict[str, float]:
        metrics: dict[str, float] = {}
        stream_ids = rollout["stream_ids"]
        for name, stream_id in (
            ("phase0", PHASE0_STREAM),
            ("curriculum", CURRICULUM_STREAM),
        ):
            env_mask = stream_ids == stream_id
            transition_mask = env_mask.unsqueeze(0).expand_as(rollout["done"])
            sample_count = int(transition_mask.sum().item())
            done_count = int((rollout["done"] & transition_mask).sum().item())
            failure_count = int(
                (rollout["failure"] & transition_mask).sum().item()
            )
            completion_count = int(
                (rollout["motion_complete"] & transition_mask).sum().item()
            )
            timeout_count = int(
                (rollout["timeout"] & transition_mask).sum().item()
            )
            rewards = rollout["reward"][transition_mask]
            returns = self._stream_return_buffers[stream_id]
            lengths = self._stream_length_buffers[stream_id]
            prefix = f"stream/{name}"
            metrics.update(
                {
                    f"{prefix}/env_count": float(env_mask.sum().item()),
                    f"{prefix}/sample_count": float(sample_count),
                    f"{prefix}/reward_mean": float(rewards.mean().item()),
                    f"{prefix}/done_count": float(done_count),
                    f"{prefix}/failure_rate": float(
                        failure_count / max(done_count, 1)
                    ),
                    f"{prefix}/completion_rate": float(
                        completion_count / max(done_count, 1)
                    ),
                    f"{prefix}/timeout_rate": float(
                        timeout_count / max(done_count, 1)
                    ),
                    f"{prefix}/episode_return_mean": float(
                        sum(returns) / len(returns) if returns else 0.0
                    ),
                    f"{prefix}/episode_length_mean": float(
                        sum(lengths) / len(lengths) if lengths else 0.0
                    ),
                }
            )
        return metrics

    def update(self, rollout: dict, collect_time: float) -> dict[str, float]:
        train_start = time.perf_counter()
        loss = self._training_step()
        learning_time = time.perf_counter() - train_start
        storage = rollout["storage"]
        values = storage["values"].detach()
        returns = storage["returns"].detach()
        advantages = storage["advantages"].detach()
        value_error = returns - values
        return_variance = returns.var(unbiased=False)
        explained_variance = 1.0 - value_error.var(unbiased=False) / (
            return_variance + 1.0e-8
        )

        metrics: dict[str, float] = {
            "Loss/Value": float(loss["Value"]),
            "Loss/Surrogate": float(loss["Surrogate"]),
            "Loss/Entropy": float(loss["Entropy"]),
            "Loss/KL": float(loss["KL"]),
            "Loss/actor_loss": float(loss["actor_loss"]),
            "Loss/critic_loss": float(loss["critic_loss"]),
            "Loss/symmetry_actor_loss": float(loss["symmetry_actor_loss"]),
            "Loss/symmetry_critic_loss": float(loss["symmetry_critic_loss"]),
            "Loss/actor_learning_rate": float(loss["actor_learning_rate"]),
            "Loss/critic_learning_rate": float(loss["critic_learning_rate"]),
            "Policy/mean_noise_std": float(self.actor.std.mean().item()),
            "Policy/ratio_mean": float(loss["ratio_mean"]),
            "Policy/clip_fraction": float(loss["clip_fraction"]),
            "Policy/actor_grad_norm": float(loss["actor_grad_norm"]),
            "Policy/critic_grad_norm": float(loss["critic_grad_norm"]),
            "Policy/actor_parameter_delta": float(
                loss["actor_parameter_delta"]
            ),
            "Policy/critic_parameter_delta": float(
                loss["critic_parameter_delta"]
            ),
            "Policy/optimizer_steps": float(loss["optimizer_steps"]),
            "Policy/action_clip_fraction": float(
                rollout["action_clip_count"] / rollout["actions"].numel()
            ),
            "Policy/static_parity": 1.0,
            "Policy/minibatch_reuse_verified": 1.0,
            "Normalizer/actor_count": float(
                self.actor_obs_normalizer.count.item()
            ),
            "Normalizer/critic_count": float(
                self.critic_obs_normalizer.count.item()
            ),
            "Train/num_samples_update": float(
                self.env.num_envs * int(self.cfg.num_steps_per_env)
            ),
            "Train/advantage_mean": float(advantages.mean().item()),
            "Train/advantage_std": float(advantages.std().item()),
            "Critic/value_mean": float(values.mean().item()),
            "Critic/return_mean": float(returns.mean().item()),
            "Critic/rmse": float(
                torch.sqrt(value_error.square().mean()).item()
            ),
            "Critic/explained_variance": float(explained_variance.item()),
            "reward/decomposition_identity_abs_max": float(
                rollout["reward_decomposition_abs_max"]
            ),
            "timing/collect_s": float(collect_time),
            "timing/learning_s": float(learning_time),
            "system/parameters_finite": 1.0,
        }
        metrics.update(_distribution_metrics("reward/total", rollout["reward"]))
        metrics.update(
            _distribution_metrics("action/raw", rollout["actions"])
        )
        metrics.update(
            _distribution_metrics("action/applied", rollout["applied_actions"])
        )
        metrics.update(
            _distribution_metrics("action/mean", rollout["mean_actions"])
        )
        valid_delta = rollout["action_delta"][rollout["action_delta_valid"]]
        valid_d2 = rollout["action_d2"][rollout["action_d2_valid"]]
        if valid_delta.numel() > 0:
            metrics.update(_distribution_metrics("action/delta", valid_delta))
        if valid_d2.numel() > 0:
            metrics.update(_distribution_metrics("action/d2", valid_d2))
        metrics.update(
            _distribution_metrics("physics/joint_vel_jump", rollout["joint_vel_jump"])
        )
        metrics.update(
            _distribution_metrics("physics/root_lin_vel_jump", rollout["root_lin_vel_jump"])
        )
        metrics.update(
            _distribution_metrics("physics/root_ang_vel_jump", rollout["root_ang_vel_jump"])
        )
        for term_name, values_tensor in rollout["reward_raw_terms"].items():
            metric_name = REWARD_METRIC_NAMES[term_name]
            metrics.update(
                _distribution_metrics(
                    f"reward/raw/{metric_name}", values_tensor
                )
            )
        for term_name, values_tensor in rollout["reward_contributions"].items():
            metric_name = REWARD_METRIC_NAMES[term_name]
            metrics.update(
                _distribution_metrics(
                    f"reward/contribution/{metric_name}", values_tensor
                )
            )
        metrics.update(self._stream_metrics(rollout))
        metrics.update(self.phase0_attempts.metrics())
        self.storage.clear()
        return metrics

    @staticmethod
    def _parity_line() -> str:
        return (
            "[HOLOSOMA_PARITY] "
            "actor_hidden_dims=[512,256,128] critic_hidden_dims=[512,256,128] "
            "activation=ELU init_noise_std=1.0 num_steps_per_env=24 "
            "num_learning_epochs=5 num_mini_batches=4 clip_param=0.2 "
            "gamma=0.99 lam=0.95 value_loss_coef=1.0 entropy_coef=0.005 "
            "actor_learning_rate=0.001 critic_learning_rate=0.001 "
            "actor_weight_decay=0.0 critic_weight_decay=0.0 "
            "max_grad_norm=1.0 schedule=adaptive desired_kl=0.01 "
            "empirical_normalization=true use_symmetry=false"
        )

    def log(self, update_idx: int, max_updates: int, metrics: dict) -> None:
        iteration = int(update_idx) - 1
        if iteration < 3:
            print(self._parity_line(), flush=True)
            print(
                f"[HOLOSOMA_PARITY] static_sha256={HOLOSOMA_STATIC_PARITY_SHA256} "
                f"upstream_commit={HOLOSOMA_UPSTREAM_COMMIT} verified=true",
                flush=True,
            )
        collection_time = float(metrics.get("timing/collect_s", 0.0))
        learning_time = float(metrics.get("timing/learning_s", 0.0))
        total_time = collection_time + learning_time
        transitions = int(metrics.get("samples/env_transitions_total", 0.0))
        fps = transitions if total_time <= 0.0 else int(
            metrics.get("samples/env_transitions_update", 0.0) / total_time
        )
        print(
            f" Learning iteration {iteration}/{max_updates} ",
            flush=True,
        )
        print(
            f"Computation: {fps:.0f} steps/s "
            f"(Collection: {collection_time:.3f}s, Learning {learning_time:.3f}s)",
            flush=True,
        )
        for key in (
            "Value",
            "Surrogate",
            "Entropy",
            "KL",
            "actor_loss",
            "critic_loss",
            "symmetry_actor_loss",
            "symmetry_critic_loss",
            "actor_learning_rate",
            "critic_learning_rate",
        ):
            print(f"{key}: {metrics[f'Loss/{key}']:.4f}", flush=True)
        print(
            f"Policy/mean_noise_std: {metrics['Policy/mean_noise_std']:.4f}",
            flush=True,
        )
        print(f"Total timesteps: {transitions}", flush=True)
        print(
            "[PPO_HEALTH] "
            f"adv_mean={metrics['Train/advantage_mean']:.3e} "
            f"adv_std={metrics['Train/advantage_std']:.6f} "
            f"ratio={metrics['Policy/ratio_mean']:.6f} "
            f"clip={metrics['Policy/clip_fraction']:.6f} "
            f"actor_delta={metrics['Policy/actor_parameter_delta']:.3e} "
            f"critic_delta={metrics['Policy/critic_parameter_delta']:.3e} "
            f"action_clip={metrics['Policy/action_clip_fraction']:.3e} "
            f"reward_identity={metrics['reward/decomposition_identity_abs_max']:.3e}",
            flush=True,
        )
        print(
            "[REWARD] "
            f"total={metrics['reward/total/mean']:.5f} "
            f"anchor_pos={metrics['reward/raw/anchor_pos/mean']:.4f} "
            f"anchor_ori={metrics['reward/raw/anchor_ori/mean']:.4f} "
            f"body_pos={metrics['reward/raw/body_pos/mean']:.4f} "
            f"body_ori={metrics['reward/raw/body_ori/mean']:.4f}",
            flush=True,
        )

    def log_banner(self) -> None:
        print(
            "[PPO] implementation=HOLOSOMA_G1_WBT actor=MLP_diagonal_Normal "
            "critic=asymmetric_MLP action=unsquashed_env_clip_100",
            flush=True,
        )
        print(self._parity_line(), flush=True)
        print(
            f"[HOLOSOMA_PARITY] static_sha256={HOLOSOMA_STATIC_PARITY_SHA256} "
            f"upstream_commit={HOLOSOMA_UPSTREAM_COMMIT} verified=true",
            flush=True,
        )
        print(
            "[HOLOSOMA_SOURCE] "
            + " ".join(
                f"{name}={digest}"
                for name, digest in HOLOSOMA_SOURCE_SHA256.items()
            ),
            flush=True,
        )
        print(
            "[GAE] timeout=reward_bootstrap terminal=no_bootstrap "
            "advantage=global_unbiased_std minibatches=single_permutation_reused",
            flush=True,
        )
        print(
            "[REWARD] formula=(0.5*anchor_pos+0.5*anchor_ori+"
            "2*body_pos+2*body_ori-0.1*action_rate-10*joint_limit-"
            "0.1*undesired_contacts)*dt",
            flush=True,
        )
        print(
            f"[CHECKPOINT] fixed_reward_schema="
            f"{FIXED_REWARD_CHECKPOINT_CONTRACT['fixed_reward_schema_version']}",
            flush=True,
        )
