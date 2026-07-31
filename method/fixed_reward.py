"""Pure fixed-pose-reward Flow-CPS optimization."""

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
from components.rollout.fixed_reward_contract import (
    FIXED_REWARD_CHECKPOINT_CONTRACT,
)
from components.rollout.flow_cps_base import FlowCPSBase
from components.rollout.training_streams import (
    CURRICULUM_STREAM,
    PHASE0_STREAM,
    Phase0AttemptTracker,
    Phase0CurriculumStreams,
)
from models.task_flow_critic import TaskFlowCritic


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


def _masked_stats(
    prefix: str,
    values: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> dict[str, float]:
    flat = values.detach().float().reshape(-1)
    if mask is not None:
        flat = flat[mask.reshape(-1).bool()]
    if flat.numel() == 0:
        return {f"{prefix}/count": 0.0}
    quantiles = torch.quantile(
        flat,
        torch.tensor([0.05, 0.5, 0.95], device=flat.device),
    )
    return {
        f"{prefix}/count": float(flat.numel()),
        f"{prefix}/mean": float(flat.mean().item()),
        f"{prefix}/std": float(flat.std(unbiased=False).item()),
        f"{prefix}/min": float(flat.min().item()),
        f"{prefix}/max": float(flat.max().item()),
        f"{prefix}/p05": float(quantiles[0].item()),
        f"{prefix}/p50": float(quantiles[1].item()),
        f"{prefix}/p95": float(quantiles[2].item()),
    }


class FixedRewardFlowCPS(FlowCPSBase):
    """Flow-CPS with one scalar fixed pose-tracking objective."""

    def build(self) -> None:
        cfg = self.cfg
        env = self.env
        critic_cfg = cfg.critic
        super().build()

        action_limit = float(cfg.action_squash_scale)
        self.action_low = torch.full(
            (self.num_act,),
            -action_limit,
            device=env.device,
        )
        self.action_high = torch.full(
            (self.num_act,),
            action_limit,
            device=env.device,
        )
        env.enable_strict_action_contract(self.action_low, self.action_high)
        self.training_streams = Phase0CurriculumStreams.create(
            env.num_envs,
            phase0_fraction=float(cfg.streams.phase0_fraction),
            phase0_start=int(env.motion_start_phase),
            device=env.device,
        )
        self.phase0_attempts = Phase0AttemptTracker(
            self.training_streams.stream_ids
        )
        env.set_adaptive_failure_eligibility(
            self.training_streams.curriculum_mask
        )

        self.prefix_context_dim = (
            2 * self.critic_obs_dim
            + self.actor_obs_dim
            + self.num_act
            + self.chunk_dim
            + 2 * self.horizon_h
        )
        self.critic = TaskFlowCritic(
            context_dim=self.prefix_context_dim,
            encoder_hidden_dims=critic_cfg.encoder_hidden_dims,
            head_hidden_dims=critic_cfg.head_hidden_dims,
            activation=cfg.activation,
            flow_steps=int(cfg.flow_steps),
            eval_samples=self.FLOW_CRITIC_SAMPLES,
        ).to(env.device)
        self.prefix_context_normalizer = EmpiricalNormalization(
            self.prefix_context_dim,
            env.device,
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
                "prefix_context_normalizer": self.prefix_context_normalizer,
            }
        )
        self._update_index = 0
        self._high_kl_streak = 0
        self._nonpositive_reward_streak = 0
        self._init_stream_episode_stats()

    # ------------------------------------------------------------------ #
    # Hard numerical contracts
    # ------------------------------------------------------------------ #
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
        for name in (
            "joint_pos",
            "joint_vel",
            "root_state_w",
            "body_state_w",
        ):
            value = getattr(data, name, None)
            if torch.is_tensor(value):
                self._ensure_finite(f"state/{name}", value)

    def _ensure_gradients_finite(
        self,
        name: str,
        module: nn.Module,
        grad_norm: torch.Tensor | float,
    ) -> None:
        norm = torch.as_tensor(grad_norm)
        self._ensure_finite(f"{name}/gradient_norm", norm)
        for parameter_name, parameter in module.named_parameters():
            if parameter.grad is not None:
                self._ensure_finite(
                    f"{name}/gradient/{parameter_name}",
                    parameter.grad,
                )

    def _ensure_parameters_finite(self, name: str, module: nn.Module) -> None:
        for parameter_name, parameter in module.named_parameters():
            self._ensure_finite(
                f"{name}/parameter/{parameter_name}",
                parameter,
            )

    def _ensure_optimizer_finite(
        self,
        name: str,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        for parameter_index, state in enumerate(optimizer.state.values()):
            for state_name, value in state.items():
                if torch.is_tensor(value):
                    self._ensure_finite(
                        f"{name}/state/{parameter_index}/{state_name}",
                        value,
                    )

    def _ensure_normalizer_finite(
        self,
        name: str,
        normalizer: EmpiricalNormalization,
    ) -> None:
        for buffer_name in ("_mean", "_var", "_std", "count"):
            value = getattr(normalizer, buffer_name)
            self._ensure_finite(
                f"{name}/{buffer_name}",
                value,
            )
        if bool((normalizer._var < 0).any()) or bool(
            (normalizer._std < 0).any()
        ):
            raise FloatingPointError(
                f"{name} contains an invalid negative scale"
            )
        if int(normalizer.count.item()) < 0:
            raise FloatingPointError(f"{name} contains a negative count")

    # ------------------------------------------------------------------ #
    # Checkpointing
    # ------------------------------------------------------------------ #
    @staticmethod
    def _legacy_checkpoint_keys(mapping: object) -> set[str]:
        if not isinstance(mapping, dict):
            return set()
        legacy: set[str] = set()
        exact = {
            "fcamp_schema_version",
            "warmup_env_transitions",
            "mixed_reward",
            "history",
            "replay",
            "mmd",
        }
        fragments = (
            "discriminator",
            "style_prior",
            "imitation_history",
            "window_replay",
            "temporal_credit",
        )
        for key in mapping:
            key_text = str(key).lower()
            if (
                key_text in exact
                or key_text.startswith("amp_")
                or key_text.startswith("disc_")
                or key_text.startswith("channel_")
                or any(fragment in key_text for fragment in fragments)
                or "mmd" in key_text
            ):
                legacy.add(str(key))
        return legacy

    def _validate_checkpoint_contract(
        self,
        state: object,
        policy_state: object | None = None,
    ) -> None:
        if not isinstance(state, dict):
            raise ValueError(
                "fixed_reward checkpoint is missing its algorithm contract; "
                "start a fresh run"
            )
        expected = FIXED_REWARD_CHECKPOINT_CONTRACT[
            "fixed_reward_schema_version"
        ]
        saved = state.get("fixed_reward_schema_version")
        if type(saved) is not int or saved != expected:
            raise ValueError(
                "fixed_reward checkpoint fixed_reward_schema_version="
                f"{saved!r}, expected {expected}; legacy FCAMP/AMP checkpoints "
                "cannot be loaded"
            )
        for key, expected_value in (
            FIXED_REWARD_CHECKPOINT_CONTRACT.items()
        ):
            saved_value = state.get(key)
            if saved_value != expected_value:
                raise ValueError(
                    "fixed_reward checkpoint semantic contract mismatch: "
                    f"{key} expected={expected_value!r}, "
                    f"actual={saved_value!r}"
                )
        legacy = self._legacy_checkpoint_keys(state)
        legacy.update(self._legacy_checkpoint_keys(policy_state))
        if legacy:
            raise ValueError(
                "fixed_reward checkpoint contains legacy FCAMP/AMP state: "
                + ", ".join(sorted(legacy))
            )

    def validate_checkpoint_payload(self, payload: dict) -> None:
        state = payload.get("algo_state") if isinstance(payload, dict) else None
        policy_state = payload.get("policy") if isinstance(payload, dict) else None
        self._validate_checkpoint_contract(state, policy_state)

    def extra_checkpoint_state(self) -> dict:
        payload = super().extra_checkpoint_state()
        payload.update(
            {
                **FIXED_REWARD_CHECKPOINT_CONTRACT,
                "stream_ids": self.training_streams.stream_ids.detach().cpu(),
                "phase0_stream_count": int(
                    self.training_streams.phase0_ids.numel()
                ),
                "phase0_stream_fraction": float(
                    self.training_streams.phase0_fraction
                ),
                "phase0_attempt_tracker": self.phase0_attempts.state_dict(),
            }
        )
        return payload

    def load_extra_checkpoint_state(
        self,
        payload: dict,
        reset_optimizer: bool = False,
    ) -> None:
        self._validate_checkpoint_contract(payload)
        super().load_extra_checkpoint_state(
            payload,
            reset_optimizer=reset_optimizer,
        )
        saved_stream_ids = payload.get("stream_ids")
        if not torch.is_tensor(saved_stream_ids) or not torch.equal(
            saved_stream_ids.to(dtype=torch.int8, device="cpu"),
            self.training_streams.stream_ids.detach().to("cpu"),
        ):
            raise ValueError(
                "fixed_reward checkpoint training-stream assignment differs"
            )
        if int(payload.get("phase0_stream_count", -1)) != int(
            self.training_streams.phase0_ids.numel()
        ):
            raise ValueError(
                "fixed_reward checkpoint phase0 stream count differs"
            )
        if not math.isclose(
            float(payload.get("phase0_stream_fraction", -1.0)),
            float(self.training_streams.phase0_fraction),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "fixed_reward checkpoint stream objective differs"
            )
        self.phase0_attempts.load_state_dict(
            payload.get("phase0_attempt_tracker")
        )

    # ------------------------------------------------------------------ #
    # Causal value contexts and reset streams
    # ------------------------------------------------------------------ #
    def _prefix_context_raw(
        self,
        current_critic_obs: torch.Tensor,
        chunk_start_critic_obs: torch.Tensor,
        chunk_start_actor_obs: torch.Tensor,
        previous_action: torch.Tensor,
        final_latent: torch.Tensor,
        offset: int,
    ) -> torch.Tensor:
        if not 0 <= int(offset) < self.horizon_h:
            raise ValueError("offset must identify a chunk frame")
        batch = current_critic_obs.shape[0]
        latent = final_latent.reshape(batch, self.horizon_h, self.num_act)
        prefix = torch.zeros_like(latent)
        if offset > 0:
            prefix[:, :offset] = latent[:, :offset]
        prefix_mask = torch.zeros(
            batch,
            self.horizon_h,
            device=latent.device,
            dtype=latent.dtype,
        )
        if offset > 0:
            prefix_mask[:, :offset] = 1.0
        offset_onehot = torch.zeros_like(prefix_mask)
        offset_onehot[:, int(offset)] = 1.0
        context = torch.cat(
            (
                current_critic_obs,
                chunk_start_critic_obs,
                chunk_start_actor_obs,
                previous_action,
                prefix.reshape(batch, -1),
                prefix_mask,
                offset_onehot,
            ),
            dim=-1,
        )
        if context.shape[-1] != self.prefix_context_dim:
            raise RuntimeError(
                "fixed_reward prefix context has incompatible dimension"
            )
        self._ensure_finite("critic/context_raw", context)
        return context

    @torch.no_grad()
    def _evaluate_prefix_values(self, contexts: torch.Tensor) -> torch.Tensor:
        flat = contexts.reshape(-1, self.prefix_context_dim)
        batch_size = max(1, int(self.cfg.micro_batch_size))
        outputs: list[torch.Tensor] = []
        for start in range(0, flat.shape[0], batch_size):
            values = self.critic.evaluate(flat[start : start + batch_size])
            self._ensure_finite("critic/value", values)
            outputs.append(values)
        return torch.cat(outputs, dim=0).reshape(*contexts.shape[:-1])

    @torch.no_grad()
    def _update_empirical_normalizer_chunked(
        self,
        normalizer: EmpiricalNormalization,
        samples: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> None:
        flat = samples.reshape(-1, samples.shape[-1])
        if valid is not None:
            indices = valid.reshape(-1).bool().nonzero(
                as_tuple=False
            ).squeeze(-1)
        else:
            indices = torch.arange(flat.shape[0], device=flat.device)
        if indices.numel() == 0:
            raise RuntimeError("normalizer received no valid samples")
        batch_size = min(max(1, int(self.cfg.micro_batch_size)), 1024)
        for start in range(0, indices.numel(), batch_size):
            batch = flat.index_select(
                0,
                indices[start : start + batch_size],
            )
            self._ensure_finite("normalizer/input", batch)
            normalizer._update(batch)
            self._ensure_normalizer_finite("normalizer", normalizer)

    def _init_stream_episode_stats(self) -> None:
        env = self.env
        self._stream_return_sum = torch.zeros(
            env.num_envs,
            dtype=torch.float32,
            device=env.device,
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

    def _record_stream_episode_stats(
        self,
        rewards: torch.Tensor,
        done: torch.Tensor,
        active: torch.Tensor,
    ) -> None:
        self._stream_return_sum += rewards.to(dtype=torch.float32)
        self._stream_length_sum += active.to(dtype=torch.float32)
        done_ids = done.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() == 0:
            return
        done_streams = self.training_streams.stream_ids.index_select(
            0,
            done_ids,
        )
        for stream_id in (PHASE0_STREAM, CURRICULUM_STREAM):
            local_ids = done_ids[done_streams == stream_id]
            if local_ids.numel() == 0:
                continue
            self._stream_return_buffers[stream_id].extend(
                self._stream_return_sum.index_select(
                    0,
                    local_ids,
                ).detach().cpu().tolist()
            )
            self._stream_length_buffers[stream_id].extend(
                self._stream_length_sum.index_select(
                    0,
                    local_ids,
                ).detach().cpu().tolist()
            )
        self._stream_return_sum[done_ids] = 0.0
        self._stream_length_sum[done_ids] = 0.0

    def _reset_training_streams(
        self,
        *,
        randomize_curriculum_episode_age: bool,
    ) -> torch.Tensor:
        env = self.env
        self.phase0_attempts.interrupt_inflight()
        env_ids = torch.arange(
            env.num_envs,
            device=env.device,
            dtype=torch.long,
        )
        phases, reset_streams = self.training_streams.reset_phases(
            env_ids,
            lambda count: env.sample_phase_indices(
                count,
                horizon=max(1, self.horizon_h),
            ),
        )
        obs = env.reset(
            phase_indices=phases,
            reset_stream_ids=reset_streams,
        )
        if (
            randomize_curriculum_episode_age
            and bool(self.cfg.init_at_random_ep_len)
            and env.max_episode_steps > 0
            and self.training_streams.curriculum_ids.numel() > 0
        ):
            curriculum_ids = self.training_streams.curriculum_ids
            random_age = torch.randint(
                0,
                int(env.max_episode_steps),
                (curriculum_ids.numel(),),
                device=env.device,
                dtype=env.episode_steps.dtype,
            )
            env.set_episode_age(curriculum_ids, random_age)
        self._ensure_finite("observation/reset", obs)
        self._obs = obs
        self._critic_obs = env.get_critic_observation()
        self._ensure_finite("critic_observation/reset", self._critic_obs)
        self.phase0_attempts.start(self.training_streams.phase0_ids)
        return obs

    def initial_reset(self) -> torch.Tensor:
        return self._reset_training_streams(
            randomize_curriculum_episode_age=True,
        )

    def reset_after_resume(self) -> torch.Tensor:
        self._stream_return_sum.zero_()
        self._stream_length_sum.zero_()
        return self._reset_training_streams(
            randomize_curriculum_episode_age=False,
        )

    def evaluation_reset(self, phase_indices: torch.Tensor) -> torch.Tensor:
        return self.env.reset(phase_indices=phase_indices)

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        self._update_index = int(update_idx)
        self.phase0_attempts.begin_update()
        return self._obs

    # ------------------------------------------------------------------ #
    # Rollout and scalar credit
    # ------------------------------------------------------------------ #
    @staticmethod
    def _chronological(value: torch.Tensor) -> torch.Tensor:
        chunks, envs, horizon = value.shape[:3]
        tail = value.shape[3:]
        return value.permute(
            0,
            2,
            1,
            *range(3, value.ndim),
        ).reshape(chunks * horizon, envs, *tail)

    @staticmethod
    def _chunk_layout(value: torch.Tensor, chunks: int, horizon: int) -> torch.Tensor:
        envs = value.shape[1]
        tail = value.shape[2:]
        return value.reshape(chunks, horizon, envs, *tail).permute(
            0,
            2,
            1,
            *range(3, 3 + len(tail)),
        )

    def _reward_contributions(
        self,
        terms: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        dt = float(self.env.dt)
        contributions: dict[str, torch.Tensor] = {}
        for key, weight in REWARD_TERM_WEIGHTS.items():
            if key not in terms:
                raise KeyError(f"reward terms are missing {key}")
            value = terms[key]
            self._ensure_finite(f"reward/raw/{key}", value)
            contributions[key] = float(weight) * value * dt
        return contributions

    def collect(self, current_obs: torch.Tensor) -> dict:
        env = self.env
        device = env.device
        n_envs = env.num_envs
        chunks = self._chunks_per_update()
        horizon = self.horizon_h
        flow_steps = int(self.cfg.flow_steps)

        actor_obs_buf = torch.zeros(
            chunks,
            n_envs,
            self.actor_obs_dim,
            device=device,
        )
        actor_obs_raw_buf = torch.zeros_like(actor_obs_buf)
        latent_path_buf = torch.zeros(
            chunks,
            n_envs,
            flow_steps + 1,
            self.chunk_dim,
            device=device,
        )
        old_log_probs_buf = torch.zeros(
            chunks,
            n_envs,
            flow_steps,
            horizon,
            device=device,
        )
        context_buf = torch.zeros(
            chunks,
            n_envs,
            horizon,
            self.prefix_context_dim,
            device=device,
        )
        next_context_buf = torch.zeros_like(context_buf)
        context_raw_buf = torch.zeros_like(context_buf)
        next_context_raw_buf = torch.zeros_like(context_buf)
        reward_buf = torch.zeros(chunks, n_envs, horizon, device=device)
        raw_term_bufs = {
            key: torch.zeros_like(reward_buf)
            for key in (*RAW_POSE_TERMS, *RAW_PENALTY_TERMS)
        }
        contribution_bufs = {
            key: torch.zeros_like(reward_buf)
            for key in REWARD_TERM_WEIGHTS
        }
        valid_buf = torch.zeros(
            chunks,
            n_envs,
            horizon,
            dtype=torch.bool,
            device=device,
        )
        intervention_edge_buf = torch.zeros_like(valid_buf)
        done_buf = torch.zeros_like(valid_buf)
        failure_buf = torch.zeros_like(valid_buf)
        timeout_buf = torch.zeros_like(valid_buf)
        motion_complete_buf = torch.zeros_like(valid_buf)
        bootstrap_buf = torch.zeros_like(valid_buf)
        trace_buf = torch.zeros_like(valid_buf)
        terminal_phase_buf = torch.full(
            (chunks, n_envs, horizon),
            -1.0,
            dtype=torch.float32,
            device=device,
        )

        obs = current_obs
        critic_obs = self._critic_obs
        self._ensure_finite("observation/collection_start", obs)
        self._ensure_finite("critic_observation/collection_start", critic_obs)
        collection_start_phases = env.phase_steps.detach().clone()
        action_abs_max = 0.0
        action_bound_violation_max = 0.0
        reward_identity_abs_max = 0.0
        cps_innovation_mean_square_sum = torch.zeros(
            horizon,
            device=device,
        )

        with torch.no_grad():
            for chunk_idx in range(chunks):
                chunk_actor_raw = obs.clone()
                chunk_critic_raw = critic_obs.clone()
                actor_obs_n = self.actor_obs_normalizer(chunk_actor_raw)
                self._ensure_finite("observation/actor_normalized", actor_obs_n)
                previous_action = chunk_actor_raw[
                    ..., -self.num_act :
                ].detach()
                (
                    final_latent,
                    latent_path,
                    old_log_probs,
                    innovation_mean_squares,
                ) = self._sample_cps_path(actor_obs_n)
                self._ensure_finite("action/final_latent", final_latent)
                self._ensure_finite("action/latent_path", latent_path)
                self._ensure_finite("action/old_log_prob", old_log_probs)
                self._ensure_finite(
                    "cps/innovation_mean_squares",
                    innovation_mean_squares,
                )
                cps_innovation_mean_square_sum += innovation_mean_squares
                action_chunk = self._policy._action_transform(
                    final_latent,
                    prev_action=previous_action,
                ).view(n_envs, horizon, self.num_act)
                self._ensure_finite("action/chunk", action_chunk)
                action_abs_max = max(
                    action_abs_max,
                    float(action_chunk.abs().max().item()),
                )
                below = (self.action_low - action_chunk).clamp_min(0.0)
                above = (action_chunk - self.action_high).clamp_min(0.0)
                action_bound_violation_max = max(
                    action_bound_violation_max,
                    float(torch.maximum(below, above).max().item()),
                )

                actor_obs_buf[chunk_idx] = actor_obs_n
                actor_obs_raw_buf[chunk_idx] = chunk_actor_raw
                latent_path_buf[chunk_idx] = latent_path
                old_log_probs_buf[chunk_idx] = old_log_probs

                alive = torch.ones(
                    n_envs,
                    dtype=torch.bool,
                    device=device,
                )
                for frame_idx in range(horizon):
                    alive_before = alive.clone()
                    current_context_raw = self._prefix_context_raw(
                        critic_obs,
                        chunk_critic_raw,
                        chunk_actor_raw,
                        previous_action,
                        final_latent,
                        frame_idx,
                    )
                    action_t = action_chunk[:, frame_idx]
                    action_t = torch.where(
                        alive_before.unsqueeze(-1),
                        action_t,
                        torch.zeros_like(action_t),
                    )
                    self._ensure_finite("action/primitive", action_t)
                    next_obs, reward, done, info = env.step(action_t)
                    next_critic_obs = env.get_critic_observation()
                    self._ensure_runtime_state_finite()
                    self._ensure_finite("observation/next", next_obs)
                    self._ensure_finite(
                        "critic_observation/next",
                        next_critic_obs,
                    )
                    self._ensure_finite("reward/scalar", reward)
                    if float(reward.max().item()) > 0.100001:
                        raise RuntimeError(
                            "fixed reward exceeded the theoretical single-step maximum"
                        )

                    active_float = alive_before.to(dtype=reward.dtype)
                    reward_terms = info["reward_terms"]
                    contributions = self._reward_contributions(reward_terms)
                    reconstructed = torch.zeros_like(reward)
                    for value in contributions.values():
                        reconstructed += value
                    identity_error = (reconstructed - reward).abs()
                    active_identity = identity_error[alive_before]
                    if active_identity.numel() > 0:
                        current_identity_max = float(
                            active_identity.max().item()
                        )
                        reward_identity_abs_max = max(
                            reward_identity_abs_max,
                            current_identity_max,
                        )
                        if current_identity_max > 1.0e-6:
                            raise RuntimeError(
                                "fixed reward decomposition identity exceeded 1e-6"
                            )

                    reward_buf[chunk_idx, :, frame_idx] = (
                        reward * active_float
                    )
                    for key in raw_term_bufs:
                        raw_term_bufs[key][chunk_idx, :, frame_idx] = (
                            reward_terms[key] * active_float
                        )
                        contribution_bufs[key][
                            chunk_idx, :, frame_idx
                        ] = contributions[key] * active_float
                    valid_buf[chunk_idx, :, frame_idx] = alive_before
                    intervention_edges = info[
                        "intervention_edge_mask"
                    ].bool()
                    intervention_edge_buf[
                        chunk_idx, :, frame_idx
                    ] = intervention_edges & alive_before

                    done_terms = info["done_terms"]
                    timeouts = done_terms["time_out"].bool()
                    motion_complete = done_terms["motion_complete"].bool()
                    failures = (
                        done_terms["anchor_pos_bad"].bool()
                        | done_terms["anchor_ori_bad"].bool()
                        | done_terms["ee_body_bad"].bool()
                    )
                    new_done = alive_before & done.bool()
                    (
                        new_failure,
                        new_timeout,
                        new_motion_complete,
                    ) = resolve_terminal_masks(
                        new_done,
                        timeouts,
                        motion_complete,
                        failures,
                    )
                    done_buf[chunk_idx, :, frame_idx] = new_done
                    failure_buf[chunk_idx, :, frame_idx] = new_failure
                    timeout_buf[chunk_idx, :, frame_idx] = new_timeout
                    motion_complete_buf[
                        chunk_idx, :, frame_idx
                    ] = new_motion_complete
                    self.phase0_attempts.observe_step(
                        alive_before,
                        new_done,
                        new_failure,
                        new_timeout,
                        new_motion_complete,
                    )
                    if bool(new_done.any()):
                        terminal_phase_buf[
                            chunk_idx, new_done, frame_idx
                        ] = info["termination_phase_steps"][
                            new_done
                        ].to(dtype=torch.float32)

                    bootstrap_buf[chunk_idx, :, frame_idx] = (
                        alive_before
                        & ~new_failure
                        & ~new_motion_complete
                    )
                    trace_buf[chunk_idx, :, frame_idx] = (
                        alive_before & ~new_done
                    )

                    if frame_idx < horizon - 1:
                        next_context_raw = self._prefix_context_raw(
                            next_critic_obs,
                            chunk_critic_raw,
                            chunk_actor_raw,
                            previous_action,
                            final_latent,
                            frame_idx + 1,
                        )
                    else:
                        next_context_raw = self._prefix_context_raw(
                            next_critic_obs,
                            next_critic_obs,
                            next_obs,
                            action_t,
                            torch.zeros_like(final_latent),
                            0,
                        )
                    context_raw_buf[
                        chunk_idx, :, frame_idx
                    ] = current_context_raw
                    next_context_raw_buf[
                        chunk_idx, :, frame_idx
                    ] = next_context_raw
                    normalized_context = self.prefix_context_normalizer(
                        current_context_raw
                    )
                    normalized_next_context = self.prefix_context_normalizer(
                        next_context_raw
                    )
                    self._ensure_finite(
                        "critic/context",
                        normalized_context,
                    )
                    self._ensure_finite(
                        "critic/next_context",
                        normalized_next_context,
                    )
                    context_buf[
                        chunk_idx, :, frame_idx
                    ] = normalized_context
                    next_context_buf[
                        chunk_idx, :, frame_idx
                    ] = normalized_next_context

                    active_reward = reward * active_float
                    self._record_train_episode_stats(
                        active_reward,
                        new_done,
                        step_counts=active_float,
                    )
                    self._record_stream_episode_stats(
                        active_reward,
                        new_done,
                        alive_before,
                    )
                    alive = alive_before & ~done.bool()
                    obs = next_obs
                    critic_obs = next_critic_obs

                chunk_done = done_buf[chunk_idx].any(dim=-1)
                if bool(chunk_done.any()):
                    reset_ids = chunk_done.nonzero(
                        as_tuple=False
                    ).squeeze(-1)
                    reset_phases, reset_streams = (
                        self.training_streams.reset_phases(
                            reset_ids,
                            lambda count: env.sample_phase_indices(
                                count,
                                horizon=max(1, horizon),
                            ),
                        )
                    )
                    reset_obs = env.reset_envs(
                        reset_ids,
                        phase_indices=reset_phases,
                        reset_stream_ids=reset_streams,
                    )
                    self._ensure_finite(
                        "observation/partial_reset",
                        reset_obs,
                    )
                    obs[reset_ids] = reset_obs
                    critic_obs = env.get_critic_observation()
                    self._ensure_finite(
                        "critic_observation/partial_reset",
                        critic_obs,
                    )
                    self.phase0_attempts.start(reset_ids)

            values = self._evaluate_prefix_values(context_buf)
            next_values = self._evaluate_prefix_values(next_context_buf)

        if action_bound_violation_max > 1.0e-6:
            raise RuntimeError(
                "fixed_reward actor emitted an action outside its command domain"
            )
        if not bool(valid_buf.any()):
            raise RuntimeError("fixed_reward rollout contains no valid samples")
        self._obs = obs
        self._critic_obs = critic_obs
        cps_offset_innovation_rms = torch.sqrt(
            cps_innovation_mean_square_sum / float(chunks)
        )
        rollout = {
            "actor_obs": actor_obs_buf,
            "actor_obs_raw": actor_obs_raw_buf,
            "latents": latent_path_buf,
            "old_log_probs": old_log_probs_buf,
            "contexts": context_buf,
            "contexts_raw": context_raw_buf,
            "next_contexts_raw": next_context_raw_buf,
            "values": values,
            "next_values": next_values,
            "valid": valid_buf,
            "done": done_buf,
            "failure": failure_buf,
            "timeout": timeout_buf,
            "motion_complete": motion_complete_buf,
            "terminal_phase": terminal_phase_buf,
            "bootstrap_mask": bootstrap_buf,
            "trace_mask": trace_buf,
            "reward": reward_buf,
            "reward_raw_terms": raw_term_bufs,
            "reward_contributions": contribution_bufs,
            "reward_decomposition_abs_max": reward_identity_abs_max,
            "intervention_edge": intervention_edge_buf,
            "stream_ids": self.training_streams.stream_ids,
            "collection_start_phases": collection_start_phases,
            "action_abs_max": action_abs_max,
            "action_bound_violation_max": action_bound_violation_max,
            "cps_offset_innovation_rms": cps_offset_innovation_rms,
            "next_observation": obs,
        }
        self._assign_credit(rollout)
        return rollout

    @torch.no_grad()
    def _assign_credit(self, rollout: dict) -> None:
        chunks, n_envs, horizon = rollout["valid"].shape
        reward_time = self._chronological(rollout["reward"])
        values_time = self._chronological(rollout["values"])
        next_values_time = self._chronological(rollout["next_values"])
        bootstrap_time = self._chronological(rollout["bootstrap_mask"])
        trace_time = self._chronological(rollout["trace_mask"])
        valid_time = self._chronological(rollout["valid"])
        credit = compute_task_gae(
            reward_time,
            values_time,
            next_values_time,
            bootstrap_time,
            trace_time,
            valid_time,
            gamma=float(self.cfg.discount_gamma),
            gae_lambda=float(self.cfg.gae_lambda),
        )

        normalization_weights = torch.zeros_like(
            reward_time,
            dtype=reward_time.dtype,
        )
        for _, _, objective_weight, env_ids in self._stream_specs(
            self.training_streams.stream_ids
        ):
            stream_valid = valid_time.index_select(1, env_ids)
            valid_count = int(stream_valid.sum().item())
            if valid_count <= 0:
                raise RuntimeError(
                    "fixed_reward stream has no valid actor samples"
                )
            stream_weights = (
                stream_valid.to(dtype=reward_time.dtype)
                * (float(objective_weight) / float(valid_count))
            )
            normalization_weights.index_copy_(
                1,
                env_ids,
                stream_weights,
            )
        credit = normalize_actor_advantage(
            credit,
            valid_time,
            normalization_weights,
        )
        self._ensure_finite("credit/advantage", credit.advantages)
        self._ensure_finite(
            "credit/actor_advantage",
            credit.actor_advantage,
        )
        self._ensure_finite("credit/value_target", credit.value_targets)
        rollout["advantages"] = self._chunk_layout(
            credit.actor_advantage,
            chunks,
            horizon,
        )
        rollout["raw_advantages"] = self._chunk_layout(
            credit.advantages,
            chunks,
            horizon,
        )
        rollout["value_targets"] = self._chunk_layout(
            credit.value_targets,
            chunks,
            horizon,
        )

    # ------------------------------------------------------------------ #
    # Objective stream partition
    # ------------------------------------------------------------------ #
    def _stream_specs(
        self,
        labels: torch.Tensor,
    ) -> list[tuple[str, int, float, torch.Tensor]]:
        configured_phase0 = float(self.cfg.streams.phase0_fraction)
        configured = (
            ("phase0", PHASE0_STREAM, configured_phase0),
            (
                "curriculum",
                CURRICULUM_STREAM,
                1.0 - configured_phase0,
            ),
        )
        active: list[tuple[str, int, float, torch.Tensor]] = []
        for name, stream_id, weight in configured:
            indices = (labels == stream_id).nonzero(
                as_tuple=False
            ).squeeze(-1)
            if indices.numel() > 0 and weight > 0.0:
                active.append((name, stream_id, weight, indices))
        weight_sum = sum(item[2] for item in active)
        if not active or weight_sum <= 0.0:
            raise RuntimeError("fixed_reward has no active training stream")
        return [
            (name, stream_id, weight / weight_sum, indices)
            for name, stream_id, weight, indices in active
        ]

    # ------------------------------------------------------------------ #
    # Actor and value optimization
    # ------------------------------------------------------------------ #
    @staticmethod
    def _snapshot_parameters(module: nn.Module) -> list[torch.Tensor]:
        return [
            parameter.detach().clone()
            for parameter in module.parameters()
            if parameter.requires_grad
        ]

    @staticmethod
    def _parameter_delta_l2(
        module: nn.Module,
        before: list[torch.Tensor],
    ) -> float:
        trainable = [
            parameter
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        if len(trainable) != len(before):
            raise RuntimeError("trainable parameter set changed during update")
        squared = torch.zeros((), device=trainable[0].device)
        for parameter, old_parameter in zip(trainable, before):
            squared += (
                parameter.detach() - old_parameter.to(parameter)
            ).float().square().sum()
        return float(torch.sqrt(squared).item())

    @staticmethod
    def _named_parameter_snapshot(
        module: nn.Module,
        predicate,
    ) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
            if parameter.requires_grad and predicate(name)
        }

    @staticmethod
    def _named_parameter_delta_l2(
        module: nn.Module,
        before: dict[str, torch.Tensor],
    ) -> float:
        if not before:
            raise RuntimeError("parameter group is empty")
        current = dict(module.named_parameters())
        squared = torch.zeros(
            (),
            device=next(iter(current.values())).device,
        )
        for name, old_parameter in before.items():
            if name not in current:
                raise RuntimeError(
                    f"trainable parameter {name!r} disappeared"
                )
            squared += (
                current[name].detach()
                - old_parameter.to(current[name])
            ).float().square().sum()
        return float(torch.sqrt(squared).item())

    def _actor_update(self, rollout: dict) -> dict[str, float]:
        device = self.env.device
        chunks, n_envs = rollout["valid"].shape[:2]
        batch_size = chunks * n_envs
        horizon = self.horizon_h
        flow_steps = int(self.cfg.flow_steps)
        actor_obs = rollout["actor_obs"].reshape(
            batch_size,
            self.actor_obs_dim,
        )
        latent_path = rollout["latents"].reshape(
            batch_size,
            flow_steps + 1,
            self.chunk_dim,
        )
        old_log_probs = rollout["old_log_probs"].reshape(
            batch_size,
            flow_steps,
            horizon,
        )
        advantages = rollout["advantages"].reshape(batch_size, horizon)
        valid = rollout["valid"].reshape(batch_size, horizon)
        for name, tensor in (
            ("actor_observation", actor_obs),
            ("latent_path", latent_path),
            ("old_log_probability", old_log_probs),
            ("advantage", advantages),
        ):
            self._ensure_finite(f"actor/input/{name}", tensor)

        stream_labels = rollout["stream_ids"].reshape(
            1,
            n_envs,
        ).expand(chunks, n_envs).reshape(-1)
        stream_specs = self._stream_specs(stream_labels)
        num_mini_batches = max(1, int(self.cfg.num_mini_batches))
        micro_batch_size = self._policy_micro_batch_size(
            self._policy_mini_batch_size(batch_size)
        )
        clip_low = 1.0 - float(self.cfg.clip_range)
        clip_high = 1.0 + float(self.cfg.clip_range)
        metric_names = (
            "policy_loss",
            "kl",
            "per_factor_kl",
            "full_chunk_path_kl",
            "ratio",
            "clip",
        )
        totals = {
            **{name: 0.0 for name in metric_names},
            "grad_norm": 0.0,
            "joint_log_ratio_abs_max": 0.0,
        }
        stream_totals = {
            name: {key: 0.0 for key in metric_names}
            for name, _, _, _ in stream_specs
        }
        stream_steps = {
            name: 0
            for name, _, _, _ in stream_specs
        }
        frame_totals = {
            name: {
                "kl": torch.zeros(horizon, device=device),
                "ratio": torch.zeros(horizon, device=device),
                "clip": torch.zeros(horizon, device=device),
                "count": torch.zeros(horizon, device=device),
            }
            for name, _, _, _ in stream_specs
        }
        steps = 0
        early_stop_epoch = int(self.cfg.policy_epochs)
        actor_lr_start = float(self.learning_rate)
        lr_decrease_steps = 0
        lr_increase_steps = 0
        lr_hold_steps = 0
        before = self._snapshot_parameters(self._policy)
        backbone_before = self._named_parameter_snapshot(
            self._policy,
            lambda name: not name.startswith("cps_"),
        )
        covariance_before = self._named_parameter_snapshot(
            self._policy,
            lambda name: name in {
                "cps_diag_raw",
                "cps_lowrank_raw",
            },
        )
        eta_before = self._policy.cps_eta_raw.detach().clone()
        self._ensure_parameters_finite("actor/before", self._policy)

        for epoch in range(int(self.cfg.policy_epochs)):
            epoch_kl_sum = 0.0
            epoch_steps = 0
            stream_splits: dict[
                str,
                tuple[float, tuple[torch.Tensor, ...]],
            ] = {}
            for name, _, objective_weight, indices in stream_specs:
                shuffled = indices.index_select(
                    0,
                    torch.randperm(indices.numel(), device=device),
                )
                stream_splits[name] = (
                    objective_weight,
                    torch.tensor_split(
                        shuffled,
                        num_mini_batches,
                    ),
                )
            for mini_batch_index in range(num_mini_batches):
                parts = [
                    (
                        name,
                        objective_weight,
                        splits[mini_batch_index],
                    )
                    for name, (
                        objective_weight,
                        splits,
                    ) in stream_splits.items()
                    if splits[mini_batch_index].numel() > 0
                ]
                if not parts:
                    continue
                self.actor_optimizer.zero_grad(set_to_none=True)
                combined_metrics = {
                    key: 0.0
                    for key in metric_names
                }
                for name, objective_weight, indices in parts:
                    valid_denominator = float(
                        valid.index_select(0, indices).sum().item()
                    )
                    if valid_denominator <= 0.0:
                        raise RuntimeError(
                            f"fixed_reward {name} actor minibatch "
                            "has no valid frames"
                        )
                    stream_sums = {
                        key: 0.0
                        for key in metric_names
                    }
                    factor_denominator = (
                        valid_denominator * flow_steps
                    )
                    chunk_denominator = float(
                        (
                            valid.index_select(0, indices).sum(dim=1) > 0
                        ).sum().item()
                    )
                    for micro_start in range(
                        0,
                        indices.numel(),
                        micro_batch_size,
                    ):
                        sub = indices[
                            micro_start : micro_start + micro_batch_size
                        ]
                        new_log_probs = self._recompute_cps_path_stats(
                            actor_obs[sub],
                            latent_path[sub],
                        )
                        self._ensure_finite(
                            "actor/new_log_probability",
                            new_log_probs,
                        )
                        delta = new_log_probs - old_log_probs[sub]
                        advantage = advantages[sub]
                        mask = valid[sub].to(dtype=delta.dtype)
                        log_ratio = delta.sum(dim=1)
                        ratio = torch.exp(log_ratio)
                        for tensor_name, tensor in (
                            ("log_probability_delta", delta),
                            ("log_ratio", log_ratio),
                            ("ratio", ratio),
                            ("minibatch_advantage", advantage),
                        ):
                            self._ensure_finite(
                                f"actor/{tensor_name}",
                                tensor,
                            )
                        unclipped = -advantage * ratio
                        clipped = -advantage * torch.clamp(
                            ratio,
                            clip_low,
                            clip_high,
                        )
                        policy_sum = (
                            torch.maximum(unclipped, clipped) * mask
                        ).sum()
                        self._ensure_finite(
                            "actor/policy_loss_sum",
                            policy_sum,
                        )
                        scaled_loss = (
                            policy_sum
                            * (objective_weight / valid_denominator)
                        )
                        self._ensure_finite(
                            "actor/scaled_loss",
                            scaled_loss,
                        )
                        scaled_loss.backward()

                        with torch.no_grad():
                            kl = 0.5 * log_ratio.square()
                            clipped_flag = (
                                (ratio < clip_low) | (ratio > clip_high)
                            ).to(ratio.dtype)
                            factor_mask = mask.unsqueeze(1).expand_as(delta)
                            full_chunk_log_ratio = (
                                log_ratio * mask
                            ).sum(dim=1)
                            chunk_active = (
                                mask.sum(dim=1) > 0
                            ).to(delta.dtype)
                            stream_sums["policy_loss"] += float(
                                policy_sum.item()
                            )
                            stream_sums["kl"] += float(
                                (kl * mask).sum().item()
                            )
                            stream_sums["per_factor_kl"] += float(
                                (
                                    0.5
                                    * delta.square()
                                    * factor_mask
                                ).sum().item()
                            )
                            stream_sums[
                                "full_chunk_path_kl"
                            ] += float(
                                (
                                    0.5
                                    * full_chunk_log_ratio.square()
                                    * chunk_active
                                ).sum().item()
                            )
                            stream_sums["ratio"] += float(
                                (ratio * mask).sum().item()
                            )
                            stream_sums["clip"] += float(
                                (clipped_flag * mask).sum().item()
                            )
                            totals["joint_log_ratio_abs_max"] = max(
                                totals["joint_log_ratio_abs_max"],
                                float(log_ratio.abs().max().item()),
                            )
                            frame_totals[name]["kl"] += (
                                kl * mask
                            ).sum(dim=0)
                            frame_totals[name]["ratio"] += (
                                ratio * mask
                            ).sum(dim=0)
                            frame_totals[name]["clip"] += (
                                clipped_flag * mask
                            ).sum(dim=0)
                            frame_totals[name]["count"] += mask.sum(
                                dim=0
                            )
                    stream_means = {
                        "policy_loss": (
                            stream_sums["policy_loss"]
                            / valid_denominator
                        ),
                        "kl": stream_sums["kl"] / valid_denominator,
                        "per_factor_kl": (
                            stream_sums["per_factor_kl"]
                            / max(factor_denominator, 1.0)
                        ),
                        "full_chunk_path_kl": (
                            stream_sums["full_chunk_path_kl"]
                            / max(chunk_denominator, 1.0)
                        ),
                        "ratio": (
                            stream_sums["ratio"]
                            / valid_denominator
                        ),
                        "clip": (
                            stream_sums["clip"]
                            / valid_denominator
                        ),
                    }
                    for key, value in stream_means.items():
                        combined_metrics[key] += (
                            objective_weight * value
                        )
                        stream_totals[name][key] += value
                    stream_steps[name] += 1

                observed_kl = combined_metrics["kl"]
                if not math.isfinite(observed_kl):
                    raise FloatingPointError(
                        "actor KL is non-finite before optimizer step"
                    )
                previous_lr = float(self.learning_rate)
                self._update_adaptive_learning_rates(observed_kl)
                if self.learning_rate < previous_lr:
                    lr_decrease_steps += 1
                elif self.learning_rate > previous_lr:
                    lr_increase_steps += 1
                else:
                    lr_hold_steps += 1
                grad_norm = nn.utils.clip_grad_norm_(
                    self._policy.parameters(),
                    float(self.cfg.max_grad_norm),
                    error_if_nonfinite=True,
                )
                self._ensure_gradients_finite(
                    "actor",
                    self._policy,
                    grad_norm,
                )
                self.actor_optimizer.step()
                self._ensure_parameters_finite("actor/after", self._policy)
                self._ensure_optimizer_finite(
                    "actor_optimizer",
                    self.actor_optimizer,
                )
                for key in metric_names:
                    totals[key] += combined_metrics[key]
                totals["grad_norm"] += float(grad_norm)
                steps += 1
                epoch_kl_sum += observed_kl
                epoch_steps += 1
            if (
                float(self.cfg.desired_kl) > 0.0
                and epoch_steps > 0
                and epoch_kl_sum / epoch_steps
                > float(self.cfg.kl_early_stop_factor)
                * float(self.cfg.desired_kl)
            ):
                early_stop_epoch = epoch + 1
                break

        if steps <= 0:
            raise RuntimeError("fixed_reward actor performed no optimizer step")
        parameter_delta = self._parameter_delta_l2(self._policy, before)
        if not math.isfinite(parameter_delta) or parameter_delta <= 0.0:
            raise RuntimeError("fixed_reward actor parameters did not update")
        backbone_delta = self._named_parameter_delta_l2(
            self._policy,
            backbone_before,
        )
        covariance_delta = self._named_parameter_delta_l2(
            self._policy,
            covariance_before,
        )
        eta_delta = float(
            (
                self._policy.cps_eta_raw.detach() - eta_before
            ).abs().item()
        )
        for name, value in (
            ("backbone", backbone_delta),
            ("covariance", covariance_delta),
            ("eta", eta_delta),
        ):
            if not math.isfinite(value):
                raise FloatingPointError(
                    f"fixed_reward actor {name} parameter delta is non-finite"
                )
        denominator = float(steps)
        metrics = {
            "flow_cps/policy_loss": totals["policy_loss"] / denominator,
            "flow_cps/kl": totals["kl"] / denominator,
            "flow_cps/per_factor_kl": (
                totals["per_factor_kl"] / denominator
            ),
            "flow_cps/full_chunk_path_kl": (
                totals["full_chunk_path_kl"] / denominator
            ),
            "flow_cps/ratio": totals["ratio"] / denominator,
            "flow_cps/clip_fraction": totals["clip"] / denominator,
            "flow_cps/actor_grad_norm": (
                totals["grad_norm"] / denominator
            ),
            "flow_cps/actor_parameter_delta_l2": parameter_delta,
            "flow_cps/backbone_parameter_delta_l2": backbone_delta,
            "flow_cps/covariance_parameter_delta_l2": covariance_delta,
            "flow_cps/eta_parameter_delta_abs": eta_delta,
            "flow_cps/actor_lr_start": actor_lr_start,
            "flow_cps/actor_lr": float(self.learning_rate),
            "flow_cps/kl_target_per_step": float(self.cfg.desired_kl),
            "flow_cps/kl_units": float(self.kl_units),
            "flow_cps/lr_decrease_steps": float(lr_decrease_steps),
            "flow_cps/lr_increase_steps": float(lr_increase_steps),
            "flow_cps/lr_hold_steps": float(lr_hold_steps),
            "flow_cps/actor_optimizer_steps": float(steps),
            "flow_cps/actor_early_stop_epoch": float(early_stop_epoch),
            "flow_cps/joint_log_ratio_abs_max": totals[
                "joint_log_ratio_abs_max"
            ],
        }
        objective_weights = {
            name: weight
            for name, _, weight, _ in stream_specs
        }
        for name, _, objective_weight, _ in stream_specs:
            stream_denominator = max(stream_steps[name], 1)
            metrics[
                f"stream/{name}/actor_objective_weight"
            ] = objective_weight
            for key in metric_names:
                metrics[f"stream/{name}/actor_{key}"] = (
                    stream_totals[name][key]
                    / stream_denominator
                )
        for frame_idx in range(horizon):
            frame_values = {
                "kl": 0.0,
                "ratio": 0.0,
                "clip": 0.0,
            }
            for name, objective_weight in objective_weights.items():
                count = float(
                    frame_totals[name]["count"][frame_idx].item()
                )
                if count <= 0.0:
                    continue
                for key in frame_values:
                    frame_values[key] += objective_weight * float(
                        (
                            frame_totals[name][key][frame_idx]
                            / count
                        ).item()
                    )
            metrics[f"flow_cps/frame_{frame_idx}_kl"] = frame_values[
                "kl"
            ]
            metrics[
                f"flow_cps/frame_{frame_idx}_ratio"
            ] = frame_values["ratio"]
            metrics[f"flow_cps/frame_{frame_idx}_clip"] = frame_values[
                "clip"
            ]
        return metrics

    def _critic_update(self, rollout: dict) -> dict[str, float]:
        chunks, n_envs, horizon = rollout["valid"].shape
        contexts = rollout["contexts"].reshape(
            -1,
            self.prefix_context_dim,
        )
        targets = rollout["value_targets"].reshape(-1)
        valid = rollout["valid"].reshape(-1).bool()
        self._ensure_finite("critic/input/context", contexts)
        self._ensure_finite("critic/input/return", targets)
        valid_indices = valid.nonzero(as_tuple=False).squeeze(-1)
        if valid_indices.numel() == 0:
            raise RuntimeError(
                "fixed_reward rollout has no valid critic samples"
            )
        stream_labels = rollout["stream_ids"].reshape(
            1,
            n_envs,
            1,
        ).expand(chunks, n_envs, horizon).reshape(-1)
        local_specs = self._stream_specs(stream_labels[valid])
        stream_specs = [
            (
                name,
                stream_id,
                objective_weight,
                valid_indices.index_select(0, local_indices),
            )
            for (
                name,
                stream_id,
                objective_weight,
                local_indices,
            ) in local_specs
        ]
        num_mini_batches = max(1, int(self.cfg.num_mini_batches))
        micro_batch_size = self._policy_micro_batch_size(
            self._policy_mini_batch_size(
                int(valid_indices.numel())
            )
        )
        total_loss = 0.0
        total_grad = 0.0
        stream_totals = {
            name: 0.0
            for name, _, _, _ in stream_specs
        }
        stream_steps = {
            name: 0
            for name, _, _, _ in stream_specs
        }
        steps = 0
        before = self._snapshot_parameters(self.critic)
        self._ensure_parameters_finite("critic/before", self.critic)
        for _ in range(int(self.cfg.policy_epochs)):
            stream_splits: dict[
                str,
                tuple[float, tuple[torch.Tensor, ...]],
            ] = {}
            for name, _, objective_weight, indices in stream_specs:
                shuffled = indices.index_select(
                    0,
                    torch.randperm(
                        indices.numel(),
                        device=indices.device,
                    ),
                )
                stream_splits[name] = (
                    objective_weight,
                    torch.tensor_split(
                        shuffled,
                        num_mini_batches,
                    ),
                )
            for mini_batch_index in range(num_mini_batches):
                parts = [
                    (
                        name,
                        objective_weight,
                        splits[mini_batch_index],
                    )
                    for name, (
                        objective_weight,
                        splits,
                    ) in stream_splits.items()
                    if splits[mini_batch_index].numel() > 0
                ]
                if not parts:
                    continue
                self.critic_optimizer.zero_grad(set_to_none=True)
                combined_loss = 0.0
                for name, objective_weight, indices in parts:
                    denominator = float(indices.numel())
                    stream_loss_sum = 0.0
                    for micro_start in range(
                        0,
                        indices.numel(),
                        micro_batch_size,
                    ):
                        sub = indices[
                            micro_start : micro_start + micro_batch_size
                        ]
                        losses = self.critic.flow_matching_loss(
                            contexts[sub],
                            targets[sub],
                            fm_samples=self.FLOW_CRITIC_FM_SAMPLES,
                        )
                        self._ensure_finite(
                            "critic/flow_matching_loss",
                            losses,
                        )
                        loss_sum = losses.sum()
                        scaled_loss = (
                            objective_weight
                            * loss_sum
                            / denominator
                        )
                        self._ensure_finite(
                            "critic/scaled_loss",
                            scaled_loss,
                        )
                        scaled_loss.backward()
                        stream_loss_sum += float(loss_sum.item())
                    stream_mean = stream_loss_sum / denominator
                    combined_loss += objective_weight * stream_mean
                    stream_totals[name] += stream_mean
                    stream_steps[name] += 1
                if not math.isfinite(combined_loss):
                    raise FloatingPointError(
                        "critic loss is non-finite before optimizer step"
                    )
                grad_norm = nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    float(self.cfg.max_grad_norm),
                    error_if_nonfinite=True,
                )
                self._ensure_gradients_finite(
                    "critic",
                    self.critic,
                    grad_norm,
                )
                self.critic_optimizer.step()
                self._ensure_parameters_finite("critic/after", self.critic)
                self._ensure_optimizer_finite(
                    "critic_optimizer",
                    self.critic_optimizer,
                )
                total_loss += combined_loss
                total_grad += float(grad_norm)
                steps += 1

        if steps <= 0:
            raise RuntimeError("fixed_reward critic performed no optimizer step")
        parameter_delta = self._parameter_delta_l2(self.critic, before)
        if not math.isfinite(parameter_delta) or parameter_delta <= 0.0:
            raise RuntimeError("fixed_reward critic parameters did not update")
        metrics = {
            "critic/flow_loss": total_loss / float(steps),
            "critic/grad_norm": total_grad / float(steps),
            "critic/parameter_delta_l2": parameter_delta,
            "critic/lr": float(self.critic_learning_rate),
            "critic/optimizer_steps": float(steps),
            "critic/valid_count": float(valid.sum().item()),
        }
        for name, _, objective_weight, _ in stream_specs:
            metrics[
                f"stream/{name}/critic_objective_weight"
            ] = objective_weight
            metrics[f"stream/{name}/critic_flow_loss"] = (
                stream_totals[name]
                / max(stream_steps[name], 1)
            )
        return metrics

    # ------------------------------------------------------------------ #
    # Metrics and console reporting
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _stream_rollout_metrics(
        self,
        rollout: dict,
    ) -> dict[str, float]:
        stream_ids = rollout["stream_ids"]
        valid = rollout["valid"]
        done = rollout["done"]
        failure = rollout["failure"]
        timeout = rollout["timeout"]
        complete = rollout["motion_complete"]
        total_valid = max(int(valid.sum().item()), 1)
        configured_weights = {
            "phase0": float(self.cfg.streams.phase0_fraction),
            "curriculum": (
                1.0 - float(self.cfg.streams.phase0_fraction)
            ),
        }
        metrics: dict[str, float] = {}
        for name, stream_id in (
            ("phase0", PHASE0_STREAM),
            ("curriculum", CURRICULUM_STREAM),
        ):
            env_mask = stream_ids == stream_id
            transition_mask = env_mask.reshape(
                1,
                -1,
                1,
            ).expand_as(valid)
            stream_valid = valid & transition_mask
            stream_done = done & transition_mask
            stream_failure = failure & transition_mask
            stream_timeout = timeout & transition_mask
            stream_complete = complete & transition_mask
            env_count = int(env_mask.sum().item())
            valid_count = int(stream_valid.sum().item())
            terminal_count = int(stream_done.sum().item())
            failure_count = int(stream_failure.sum().item())
            timeout_count = int(stream_timeout.sum().item())
            complete_count = int(stream_complete.sum().item())
            possible = max(
                env_count * valid.shape[0] * valid.shape[2],
                1,
            )
            returns = self._stream_return_buffers[stream_id]
            lengths = self._stream_length_buffers[stream_id]
            prefix = f"stream/{name}"
            metrics.update(
                {
                    f"{prefix}/env_count": float(env_count),
                    f"{prefix}/env_fraction": float(
                        env_count / max(stream_ids.numel(), 1)
                    ),
                    f"{prefix}/configured_objective_weight": (
                        configured_weights[name]
                    ),
                    f"{prefix}/valid_transition_count": float(valid_count),
                    f"{prefix}/valid_transition_fraction": float(
                        valid_count / possible
                    ),
                    f"{prefix}/valid_transition_share": float(
                        valid_count / total_valid
                    ),
                    f"{prefix}/episode_count": float(len(returns)),
                    f"{prefix}/episode_return_mean": float(
                        sum(returns) / len(returns)
                        if returns
                        else 0.0
                    ),
                    f"{prefix}/episode_length_mean": float(
                        sum(lengths) / len(lengths)
                        if lengths
                        else 0.0
                    ),
                    f"{prefix}/terminal_count": float(terminal_count),
                    f"{prefix}/failure_count": float(failure_count),
                    f"{prefix}/timeout_count": float(timeout_count),
                    f"{prefix}/motion_complete_count": float(
                        complete_count
                    ),
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
            if start_phases.numel() > 0:
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
            failure_phases = rollout["terminal_phase"][
                stream_failure
            ].float()
            if failure_phases.numel() > 0:
                quantiles = torch.quantile(
                    failure_phases,
                    torch.tensor(
                        [0.5, 0.95],
                        device=failure_phases.device,
                    ),
                )
                metrics.update(
                    {
                        f"{prefix}/failure_phase_mean": float(
                            failure_phases.mean().item()
                        ),
                        f"{prefix}/failure_phase_p50": float(
                            quantiles[0].item()
                        ),
                        f"{prefix}/failure_phase_p95": float(
                            quantiles[1].item()
                        ),
                    }
                )
            else:
                metrics[f"{prefix}/failure_phase_mean"] = -1.0
                metrics[f"{prefix}/failure_phase_p50"] = -1.0
                metrics[f"{prefix}/failure_phase_p95"] = -1.0
        return metrics

    @torch.no_grad()
    def _cps_health_metrics(self) -> dict[str, float]:
        steps = int(self.cfg.flow_steps)
        sigma_schedule = torch.linspace(
            1.0,
            0.0,
            steps + 1,
            device=self.env.device,
            dtype=self._policy.cps_diag_raw.dtype,
        )
        eta = self._cps_eta_value()
        self._ensure_finite("cps/eta", eta)
        eta_value = float(eta.item())
        if not 0.0 < eta_value < 1.0:
            raise RuntimeError("CPS global eta left (0, 1)")
        metrics: dict[str, float] = {
            "cps/eta": eta_value,
            "cps/eta_raw": float(self._policy.cps_eta_raw.item()),
            "health/covariance_psd": 1.0,
        }
        for step_index in range(steps):
            (
                _,
                _,
                covariance,
                chol,
                _,
                average_variance,
            ) = self._cps_covariance_factors(
                step_index,
                device=torch.device(self.env.device),
                dtype=self._policy.cps_diag_raw.dtype,
            )
            predicted_coeff, noise_coeff, step_eta = (
                self._cps_step_coeffs(
                    step_index,
                    sigma_schedule,
                    self._policy.cps_diag_raw,
                )
            )
            for name, tensor in (
                ("covariance", covariance),
                ("cholesky", chol),
                ("average_variance", average_variance),
                ("eta", step_eta),
                ("noise_coeff", noise_coeff),
                ("predicted_coeff", predicted_coeff),
            ):
                self._ensure_finite(
                    f"cps/step_{step_index}/{name}",
                    tensor,
                )
            chol_diag = torch.diagonal(chol)
            if bool((chol_diag <= 0.0).any()):
                raise RuntimeError(
                    f"CPS covariance step {step_index} is not positive definite"
                )
            average_variance_value = float(average_variance.item())
            if abs(average_variance_value - 1.0) > 1.0e-5:
                raise RuntimeError(
                    "CPS covariance shape normalization drifted: "
                    f"step={step_index}, average_variance="
                    f"{average_variance_value:.9g}"
                )
            prefix = f"cps/step_{step_index}"
            metrics.update(
                {
                    f"{prefix}/predicted_coeff": float(
                        predicted_coeff.item()
                    ),
                    f"{prefix}/noise_coeff": float(
                        noise_coeff.item()
                    ),
                    f"{prefix}/covariance_trace": float(
                        torch.trace(covariance).item()
                    ),
                    f"{prefix}/shape_average_variance": (
                        average_variance_value
                    ),
                    f"{prefix}/innovation_average_variance": float(
                        noise_coeff.square().item()
                        * average_variance_value
                    ),
                    f"{prefix}/cholesky_diag_min": float(
                        chol_diag.min().item()
                    ),
                    f"{prefix}/cholesky_diag_max": float(
                        chol_diag.max().item()
                    ),
                    f"{prefix}/covariance_psd": 1.0,
                }
            )
        return metrics

    def _soft_health_metrics(
        self,
        metrics: dict[str, float],
    ) -> dict[str, float]:
        kl = float(metrics["flow_cps/kl"])
        reward_mean = float(metrics["reward/total/mean"])
        self._high_kl_streak = (
            self._high_kl_streak + 1
            if kl > 0.04
            else 0
        )
        self._nonpositive_reward_streak = (
            self._nonpositive_reward_streak + 1
            if reward_mean <= 0.0
            else 0
        )
        ratio = float(metrics["flow_cps/ratio"])
        clip_fraction = float(metrics["flow_cps/clip_fraction"])
        return {
            "health/high_kl_streak": float(self._high_kl_streak),
            "health/high_kl_warning": float(self._high_kl_streak >= 3),
            "health/high_clip_warning": float(clip_fraction > 0.5),
            "health/ratio_warning": float(
                ratio < 0.8 or ratio > 1.2
            ),
            "health/nonpositive_reward_streak": float(
                self._nonpositive_reward_streak
            ),
            "health/nonpositive_reward_warning": float(
                self._nonpositive_reward_streak >= 5
            ),
        }

    def update(self, rollout: dict, collect_time: float) -> dict[str, float]:
        update_start = time.perf_counter()
        legacy = self._legacy_checkpoint_keys(rollout)
        if legacy:
            raise RuntimeError(
                "fixed_reward rollout contains disabled state: "
                + ", ".join(sorted(legacy))
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
        if not bool(rollout["valid"].any()):
            raise RuntimeError("fixed_reward rollout has no valid samples")
        if (
            float(rollout["reward_decomposition_abs_max"])
            > 1.0e-6
        ):
            raise RuntimeError(
                "fixed reward decomposition identity exceeded 1e-6"
            )
        if float(rollout["action_bound_violation_max"]) > 1.0e-6:
            raise RuntimeError("policy action-bound contract failed")
        valid_reward = rollout["reward"][rollout["valid"]]
        if (
            valid_reward.numel() == 0
            or float(valid_reward.max().item()) > 0.100001
        ):
            raise RuntimeError("fixed reward maximum contract failed")

        actor_start = time.perf_counter()
        actor_metrics = self._actor_update(rollout)
        actor_time = time.perf_counter() - actor_start
        critic_start = time.perf_counter()
        critic_metrics = self._critic_update(rollout)
        critic_time = time.perf_counter() - critic_start

        with torch.no_grad():
            self._update_empirical_normalizer_chunked(
                self.actor_obs_normalizer,
                rollout["actor_obs_raw"],
            )
            self._update_empirical_normalizer_chunked(
                self.prefix_context_normalizer,
                rollout["contexts_raw"],
                rollout["valid"],
            )
            self._update_empirical_normalizer_chunked(
                self.prefix_context_normalizer,
                rollout["next_contexts_raw"],
                rollout["valid"],
            )
            self._ensure_normalizer_finite(
                "actor_observation_normalizer",
                self.actor_obs_normalizer,
            )
            self._ensure_normalizer_finite(
                "critic_context_normalizer",
                self.prefix_context_normalizer,
            )

        valid = rollout["valid"]
        metrics: dict[str, float] = {}
        metrics.update(actor_metrics)
        metrics.update(critic_metrics)
        metrics.update(self._stream_rollout_metrics(rollout))
        metrics.update(self.phase0_attempts.metrics())
        metrics.update(
            _masked_stats(
                "reward/total",
                rollout["reward"],
                valid,
            )
        )
        metrics.update(
            _masked_stats(
                "credit/raw_advantage",
                rollout["raw_advantages"],
                valid,
            )
        )
        metrics.update(
            _masked_stats(
                "credit/actor_advantage",
                rollout["advantages"],
                valid,
            )
        )
        metrics.update(
            _masked_stats(
                "critic/value",
                rollout["values"],
                valid,
            )
        )
        metrics.update(
            _masked_stats(
                "critic/return_target",
                rollout["value_targets"],
                valid,
            )
        )
        for key in RAW_POSE_TERMS:
            metrics.update(
                _masked_stats(
                    "reward/raw/"
                    + REWARD_METRIC_NAMES[key],
                    rollout["reward_raw_terms"][key],
                    valid,
                )
            )
        for key in REWARD_TERM_WEIGHTS:
            metrics.update(
                _masked_stats(
                    "reward/contribution/"
                    + REWARD_METRIC_NAMES[key],
                    rollout["reward_contributions"][key],
                    valid,
                )
            )
        metrics.update(
            {
                "reward/decomposition_identity_abs_max": float(
                    rollout["reward_decomposition_abs_max"]
                ),
                "rollout/valid_fraction": float(
                    valid.float().mean().item()
                ),
                "rollout/done_fraction": float(
                    rollout["done"].float().mean().item()
                ),
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
                    rollout[
                        "collection_start_phases"
                    ].float().mean().item()
                ),
                "phase/start_min": float(
                    rollout["collection_start_phases"].min().item()
                ),
                "phase/start_max": float(
                    rollout["collection_start_phases"].max().item()
                ),
                "act/abs_max": float(rollout["action_abs_max"]),
                "act/policy_bound_violation_max": float(
                    rollout["action_bound_violation_max"]
                ),
                "intervention/edge_count": float(
                    rollout["intervention_edge"].sum().item()
                ),
                "train/mean_reward": float(
                    sum(self._train_reward_buffer)
                    / len(self._train_reward_buffer)
                    if self._train_reward_buffer
                    else 0.0
                ),
                "train/mean_episode_length": float(
                    sum(self._train_length_buffer)
                    / len(self._train_length_buffer)
                    if self._train_length_buffer
                    else 0.0
                ),
                "timing/collect_s": float(collect_time),
                "timing/actor_update_s": float(actor_time),
                "timing/critic_update_s": float(critic_time),
                "timing/update_s": float(
                    time.perf_counter() - update_start
                ),
                "system/primitive_steps": float(
                    self._update_index
                    * int(self.cfg.rollout_env_steps)
                    * self.env.num_envs
                ),
                "system/cuda_peak_allocated_gib": float(
                    torch.cuda.max_memory_allocated(self.env.device)
                    / (1024**3)
                    if torch.cuda.is_available()
                    else 0.0
                ),
            }
        )
        self._add_sampler_metrics(metrics)
        innovation_rms = rollout["cps_offset_innovation_rms"]
        self._ensure_finite("cps/offset_innovation_rms", innovation_rms)
        for offset, value in enumerate(innovation_rms):
            metrics[f"cps/offset_{offset}_innovation_rms"] = float(
                value.item()
            )
        innovation_min = float(innovation_rms.min().item())
        innovation_max = float(innovation_rms.max().item())
        metrics["cps/offset_innovation_rms_ratio"] = (
            innovation_max / max(innovation_min, 1.0e-12)
        )
        metrics.update(self._cps_health_metrics())
        self._ensure_parameters_finite("actor/final", self._policy)
        self._ensure_parameters_finite("critic/final", self.critic)
        metrics["system/parameters_finite"] = 1.0
        metrics.update(self._soft_health_metrics(metrics))
        return metrics

    def log(
        self,
        update_idx: int,
        max_updates: int,
        metrics: dict,
    ) -> None:
        print(
            f"[FLOW_CPS] update={update_idx}/{max_updates} "
            f"loss={metrics.get('flow_cps/policy_loss', 0.0):.5f} "
            f"kl={metrics.get('flow_cps/kl', 0.0):.6f} "
            f"ratio={metrics.get('flow_cps/ratio', 0.0):.4f} "
            f"clip={metrics.get('flow_cps/clip_fraction', 0.0):.4f} "
            f"grad={metrics.get('flow_cps/actor_grad_norm', 0.0):.4f} "
            f"delta={metrics.get('flow_cps/actor_parameter_delta_l2', 0.0):.3e} "
            f"backbone_delta={metrics.get('flow_cps/backbone_parameter_delta_l2', 0.0):.3e} "
            f"cov_delta={metrics.get('flow_cps/covariance_parameter_delta_l2', 0.0):.3e} "
            f"eta={metrics.get('cps/eta', 0.0):.6f} "
            f"eta_delta={metrics.get('flow_cps/eta_parameter_delta_abs', 0.0):.3e}",
            flush=True,
        )
        print(
            f"[CRITIC] loss={metrics.get('critic/flow_loss', 0.0):.5f} "
            f"value={metrics.get('critic/value/mean', 0.0):.5f} "
            f"return={metrics.get('critic/return_target/mean', 0.0):.5f} "
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
            f"[HEALTH] finite={metrics.get('system/parameters_finite', 0.0):.0f} "
            f"action_violation={metrics.get('act/policy_bound_violation_max', 0.0):.3e} "
            f"phase0_return={metrics.get('stream/phase0/episode_return_mean', 0.0):.4f} "
            f"phase0_length={metrics.get('stream/phase0/episode_length_mean', 0.0):.1f} "
            f"phase0_fail={metrics.get('stream/phase0/failure_rate', 0.0):.4f} "
            f"phase0_complete={metrics.get('stream/phase0/completion_rate', 0.0):.4f} "
            f"curr_return={metrics.get('stream/curriculum/episode_return_mean', 0.0):.4f} "
            f"curr_length={metrics.get('stream/curriculum/episode_length_mean', 0.0):.1f} "
            f"curr_fail={metrics.get('stream/curriculum/failure_rate', 0.0):.4f} "
            f"curr_complete={metrics.get('stream/curriculum/completion_rate', 0.0):.4f}",
            flush=True,
        )

    def log_banner(self) -> None:
        print(
            "[FLOW_CPS] method=fixed_reward actor=causal_flow_cps "
            "decoder=cumulative_residual CPS=direct_residual "
            "covariance=shared_offset_shape scale=learned_global_eta "
            "ppo=per_frame_conditional_factor reward=pose_only "
            "streams=phase0_10pct,curriculum_90pct",
            flush=True,
        )
        print(
            f"[CRITIC] type=single_task_flow context={self.prefix_context_dim} "
            f"flow_steps={self.cfg.flow_steps}",
            flush=True,
        )
        print(
            "[REWARD] formula=(0.5*anchor_pos+0.5*anchor_ori+"
            "2*body_pos+2*body_ori-0.1*action_rate-10*joint_limit-"
            "0.1*undesired_contacts)*dt",
            flush=True,
        )
        schema = FIXED_REWARD_CHECKPOINT_CONTRACT[
            "fixed_reward_schema_version"
        ]
        print(
            "[HEALTH] finite_guards=state,observation,action,reward,value,"
            "return,advantage,loss,gradient,parameter "
            f"fixed_reward_schema={schema}",
            flush=True,
        )
