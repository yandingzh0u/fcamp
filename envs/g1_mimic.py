from __future__ import annotations

import math

import torch

from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
from components.rollout.reset_diagnostics import ResetPhaseRecorder
from engine.config import EnvironmentConfig

from .adaptive_sampling import AdaptiveTimestepsSampler
from .spec import (
    CRITIC_OBS_DIM,
    OBS_DIM,
    RESET_JOINT_POSITION_RANGE,
    RESET_ROOT_POSE_RANGE,
    VELOCITY_RANGE,
    CONTACT_ALLOWED_SUBSTRINGS,
    MIMIC_ANCHOR_BODY_NAME,
    MIMIC_BODY_NAMES,
    MIMIC_EE_BODY_NAMES,
    MIMIC_FOOT_BODY_NAMES,
    MIMIC_TERMINATION_BODY_NAMES,
)
from .motion import (
    MimicMotionReference,
    mimickit_frame_delta,
    mimickit_full_motion_steps,
)
from .observation import MimicObservationMixin
from .reward import MimicRewardMixin
from .robot import G1Env
from .robots.g1 import G1_29DOF_ACTION_NAMES
from .step import MimicStepMixin
from .terminal import MimicTerminationMixin
from .tasks import resolve_task


class G1MimicEnv(
    MimicStepMixin,
    MimicTerminationMixin,
    MimicRewardMixin,
    MimicObservationMixin,
    G1Env,
):
    def __init__(
        self,
        cfg: EnvironmentConfig,
        *,
        render: bool = False,
        render_every: int = 1,
        contact_debug_vis: bool = False,
    ):
        self.config = cfg
        self.task = resolve_task(cfg.task)
        self.observation_noise = cfg.observation_noise
        super().__init__(
            cfg,
            self.task,
            render=render,
            render_every=render_every,
            contact_debug_vis=contact_debug_vis,
        )
        track_body_ids, track_body_names = self.robot.find_bodies(list(MIMIC_BODY_NAMES), preserve_order=True)
        self.track_body_ids = torch.tensor(track_body_ids, dtype=torch.long, device=self.device)
        self.track_body_names = list(track_body_names)
        self.anchor_body_id = self.robot.body_names.index(MIMIC_ANCHOR_BODY_NAME)
        self.ee_body_names = list(MIMIC_EE_BODY_NAMES)
        self.ee_body_indices = [self.track_body_names.index(name) for name in MIMIC_EE_BODY_NAMES]
        termination_names = MIMIC_TERMINATION_BODY_NAMES
        self.termination_body_indices = [self.track_body_names.index(name) for name in termination_names]
        self.contact_sensor = self.scene["contact_forces"]
        if self.uses_ground_contact_filter:
            force_matrix = self.contact_sensor.data.force_matrix_w
            expected_shape = (self.num_envs, len(self.contact_sensor.body_names), 1, 3)
            if not torch.is_tensor(force_matrix) or tuple(force_matrix.shape) != expected_shape:
                actual_shape = None if force_matrix is None else tuple(force_matrix.shape)
                raise RuntimeError(
                    f"Ground-filter contact matrix shape mismatch: expected {expected_shape}, got {actual_shape}"
                )
        def _contact_allowed(body_name: str) -> bool:
            return any(token in body_name for token in CONTACT_ALLOWED_SUBSTRINGS)

        self.undesired_contact_body_ids = torch.tensor(
            [
                self.contact_sensor.body_names.index(body_name)
                for body_name in self.contact_sensor.body_names
                if not _contact_allowed(body_name)
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.foot_contact_body_ids = torch.tensor(
            [
                self.contact_sensor.body_names.index(name)
                for name in MIMIC_FOOT_BODY_NAMES
            ],
            dtype=torch.long,
            device=self.device,
        )


        self.termination_contact_body_ids = torch.tensor(
            [self.contact_sensor.body_names.index(name) for name in termination_names],
            dtype=torch.long,
            device=self.device,
        )
        self.motion = MimicMotionReference(
            str(self.task.motion_file),
            self.track_body_ids,
            self.anchor_body_id,
            self.device,
            robot_body_names=list(self.robot.body_names),
            action_joint_names=list(G1_29DOF_ACTION_NAMES),
            root_body_name="pelvis",
        )
        self._init_adaptive_motion_sampling()


        self.max_episode_steps = (
            int(cfg.max_episode_steps) if int(cfg.max_episode_steps) > 0 else int(self.motion.num_frames)
        )

        self.last_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.phase_steps = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_ids = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._next_episode_id = 0
        self.motion_start_phase = max(0, min(int(cfg.motion_start_phase), self.motion.num_frames - 1))
        if cfg.motion_end_phase < 0:
            self.motion_end_phase = self.motion.num_frames - 1
        else:
            self.motion_end_phase = max(self.motion_start_phase, min(int(cfg.motion_end_phase), self.motion.num_frames - 1))

        self.reset_phase_recorder = ResetPhaseRecorder(
            self.motion.num_frames,
            start_phase=self.motion_start_phase,
            device=self.device,
        )
        self.reset()

    @property
    def observation_dim(self) -> int:
        return OBS_DIM

    @property
    def critic_observation_dim(self) -> int:
        return CRITIC_OBS_DIM

    @property
    def push_interval_step_range(self) -> tuple[int, int]:
        return self._push_interval_step_range

    @property
    def motion_frame_delta(self) -> float:
        return mimickit_frame_delta(self.motion.fps, self.dt)

    def full_motion_control_steps(self) -> int:
        return mimickit_full_motion_steps(
            self.motion_start_phase,
            self.motion_end_phase,
            self.motion.fps,
            self.dt,
        )

    def _adaptive_phase_range(self, horizon: int) -> tuple[int, int]:

        horizon = max(1, int(math.ceil(float(horizon) * self.motion_frame_delta)))
        min_phase = self.motion_start_phase
        max_phase = min(
            self.motion_end_phase,
            max(0, self.motion_end_phase + 1 - max(horizon, 2)),
        )
        return int(min_phase), int(max_phase)

    def sample_phase_indices(self, num_samples: int, horizon: int) -> torch.Tensor:
        if num_samples < 0:
            raise ValueError(f"num_samples must be >= 0, got {num_samples}")
        if num_samples == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        min_phase, max_phase = self._adaptive_phase_range(horizon)
        if max_phase < min_phase:
            return torch.full((num_samples,), min_phase, dtype=torch.long, device=self.device)
        if self.reset_phase_sampling == "zero":
            return torch.full((num_samples,), min_phase, dtype=torch.long, device=self.device)
        if self.reset_phase_sampling == "rsi":
            keyframes = self._rsi_keyframe_phases(min_phase, max_phase)
            indices = torch.randint(0, keyframes.numel(), (num_samples,), device=self.device)
            return keyframes.index_select(0, indices)
        if self.reset_phase_sampling == "uniform":
            return torch.randint(min_phase, max_phase + 1, (num_samples,), dtype=torch.long, device=self.device)
        return self.adaptive_sampler.sample_frames(num_samples, min_phase, max_phase)

    def _rsi_keyframe_phases(self, min_phase: int, max_phase: int) -> torch.Tensor:
        count = min(int(self.rsi_keyframe_count), max_phase - min_phase + 1)
        if count <= 1:
            return torch.full((1,), min_phase, dtype=torch.long, device=self.device)
        phases = torch.linspace(
            float(min_phase),
            float(max_phase),
            steps=count,
            device=self.device,
        ).round().to(dtype=torch.long)
        return torch.unique_consecutive(phases)

    def reset(
        self,
        phase_indices: torch.Tensor | None = None,
        reset_stream_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        return self.reset_envs(
            env_ids=env_ids,
            phase_indices=phase_indices,
            reset_stream_ids=reset_stream_ids,
        )

    def reset_envs(
        self,
        env_ids: torch.Tensor,
        phase_indices: torch.Tensor | None = None,
        reset_stream_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._reset_env_state(
            env_ids,
            phase_indices=phase_indices,
            reset_stream_ids=reset_stream_ids,
        )
        if env_ids.numel() == 0:
            return torch.empty(0, self.observation_dim, device=self.device)
        observation = self.get_observation()
        return observation.index_select(0, env_ids)

    def _reference_policy_action(
        self,
        env_ids: torch.Tensor,
        reference_joint_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a clean reference pose into the matching policy command."""

        default_joint_pos = self.default_action_joint_pos.index_select(0, env_ids)
        reference_action = (
            reference_joint_pos - default_joint_pos
        ) / self.action_scale
        if not bool(torch.isfinite(reference_action).all()):
            raise RuntimeError(
                "Reference pose produced a non-finite reset policy command"
            )
        if self._policy_action_low is not None:
            self.validate_policy_actions(reference_action)
        return reference_action

    def _reset_env_state(
        self,
        env_ids: torch.Tensor,
        phase_indices: torch.Tensor | None = None,
        reset_stream_ids: torch.Tensor | None = None,
    ) -> None:
        if env_ids.ndim != 1:
            raise ValueError(f"env_ids must have shape (N,), got {tuple(env_ids.shape)}")
        if env_ids.numel() == 0:
            return
        if phase_indices is None:
            phase_indices = self.sample_phase_indices(env_ids.numel(), horizon=1)
        if phase_indices.shape != (env_ids.numel(),):
            raise ValueError(
                f"phase_indices must have shape {(env_ids.numel(),)}, got {tuple(phase_indices.shape)}"
            )

        phase_indices = torch.clamp(
            self.motion.clamp_time_steps(phase_indices),
            min=self.motion_start_phase,
            max=self.motion_end_phase,
        )
        if reset_stream_ids is not None and reset_stream_ids.shape != phase_indices.shape:
            raise ValueError("reset_stream_ids must match phase_indices")
        self.reset_phase_recorder.record(phase_indices, reset_stream_ids)
        self.phase_steps[env_ids] = phase_indices.to(dtype=self.phase_steps.dtype)
        self.episode_steps[env_ids] = 0
        new_episode_ids = torch.arange(
            self._next_episode_id,
            self._next_episode_id + int(env_ids.numel()),
            dtype=torch.long,
            device=self.device,
        )
        self.episode_ids[env_ids] = new_episode_ids
        self._next_episode_id += int(env_ids.numel())
        self._failure_recorded[env_ids] = False
        self._reset_interval_push_schedule(env_ids)

        self.scene.reset(env_ids=env_ids)
        reference = self.motion.get_frame(phase_indices)
        reference_action = self._reference_policy_action(
            env_ids,
            reference["joint_pos"],
        )
        root_pos = reference["root_pos_w"].clone()
        root_quat = reference["root_quat_w"].clone()
        root_lin_vel = reference["root_lin_vel_w"].clone()
        root_ang_vel = reference["root_ang_vel_w"].clone()
        joint_pos = reference["joint_pos"].clone()
        joint_vel = reference["joint_vel"].clone()
        if self.reset_noise:
            self._apply_official_reset_noise(env_ids, root_pos, root_quat, root_lin_vel, root_ang_vel, joint_pos)
        self._write_robot_state(
            root_pos=root_pos,
            root_quat=root_quat,
            root_lin_vel=root_lin_vel,
            root_ang_vel=root_ang_vel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            env_ids=env_ids,
        )
        # The reset pose and the actor's continuation anchor are one atomic
        # controller state.  Use the clean reference target, not the noised
        # plant pose written above.
        self.last_action[env_ids] = reference_action
        self.scene.update(self.physics_dt)

    def begin_reset_phase_diagnostics(self) -> None:
        self.reset_phase_recorder.begin()

    def finish_reset_phase_diagnostics(self) -> dict[str, float]:
        return self.reset_phase_recorder.finish(self.adaptive_sampler)

    def _uniform(self, ranges: tuple[tuple[float, float], ...], shape: tuple[int, int]) -> torch.Tensor:
        range_tensor = torch.tensor(ranges, dtype=torch.float32, device=self.device)
        low = range_tensor[:, 0].unsqueeze(0)
        high = range_tensor[:, 1].unsqueeze(0)
        return low + (high - low) * torch.rand(shape, device=self.device)

    def _apply_official_reset_noise(
        self,
        env_ids: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        joint_pos: torch.Tensor,
    ) -> None:
        num_resets = int(env_ids.numel())
        pose_noise = self._uniform(RESET_ROOT_POSE_RANGE, (num_resets, 6))
        root_pos += pose_noise[:, :3]
        quat_delta = quat_from_euler_xyz(pose_noise[:, 3], pose_noise[:, 4], pose_noise[:, 5])
        root_quat[:] = quat_mul(quat_delta, root_quat)

        velocity_noise = self._uniform(VELOCITY_RANGE, (num_resets, 6))
        root_lin_vel += velocity_noise[:, :3]
        root_ang_vel += velocity_noise[:, 3:]

        joint_low, joint_high = RESET_JOINT_POSITION_RANGE
        joint_pos += joint_low + (joint_high - joint_low) * torch.rand_like(
            joint_pos
        )
        soft_limits = self.robot.data.soft_joint_pos_limits.index_select(0, env_ids)
        joint_pos[:] = torch.clamp(joint_pos, soft_limits[:, self.action_joint_ids, 0], soft_limits[:, self.action_joint_ids, 1])

    def _apply_action_targets(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"Expected action shape {(self.num_envs, self.action_dim)}, got {tuple(actions.shape)}")
        if self._policy_action_low is None or self._policy_action_high is None:
            raise RuntimeError("No policy action contract is installed")
        # HOLOSOMA stores and scores the raw Gaussian action, while its action
        # manager clips only the command sent to the PD controller.
        applied_actions = torch.maximum(
            torch.minimum(actions, self._policy_action_high),
            self._policy_action_low,
        )
        self.validate_policy_actions(applied_actions)
        action_targets = (
            self.default_action_joint_pos
            + self.action_scale * applied_actions
        )
        self.robot.set_joint_position_target(action_targets, joint_ids=self.action_joint_ids)
        return applied_actions

    def _init_adaptive_motion_sampling(self) -> None:
        self.reset_phase_sampling = str(self.config.reset_phase_sampling)
        self.rsi_keyframe_count = int(self.config.rsi_keyframe_count)
        env_fps = int(round(1.0 / float(self.config.sim_dt)))
        self.adaptive_sampler = AdaptiveTimestepsSampler(
            motion_time_step_total=int(self.motion.num_frames),
            device=self.device,
            num_bins=int(self.config.adaptive_num_bins),
            env_fps=env_fps,
            adaptive_alpha=float(self.config.adaptive_alpha),
            adaptive_predecessor_ratio=float(
                self.config.adaptive_predecessor_ratio
            ),
            adaptive_predecessor_lookback_bins=int(
                self.config.adaptive_predecessor_lookback_bins
            ),
        )
        self._failure_recorded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.adaptive_failure_eligibility_mask = torch.ones(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self.record_motion_failures = True
        self.reset_noise = bool(self.config.reset_noise)
        self.interval_pushes = bool(self.config.interval_pushes)

    def set_adaptive_failure_eligibility(self, mask: torch.Tensor) -> None:
        mask = mask.to(device=self.device, dtype=torch.bool)
        if mask.shape != (self.num_envs,):
            raise ValueError(
                "adaptive failure eligibility must have shape "
                f"{(self.num_envs,)}, got {tuple(mask.shape)}"
            )
        self.adaptive_failure_eligibility_mask.copy_(mask)

    def _record_adaptive_failures(
        self,
        tracking_failure: torch.Tensor,
        death_phase_steps: torch.Tensor,
    ) -> None:

        if self.reset_phase_sampling != "adaptive" or not self.record_motion_failures:
            return
        failure = (
            tracking_failure
            & self.adaptive_failure_eligibility_mask
            & (~self._failure_recorded)
        )
        if bool(failure.any()):
            self.adaptive_sampler.update_current_failure_count(death_phase_steps[failure])
            self._failure_recorded |= failure

    def _fold_adaptive_sampler(self) -> None:

        if self.reset_phase_sampling != "adaptive" or not self.record_motion_failures:
            return
        self.adaptive_sampler.update_failure_ema()

    def adaptive_sampling_stats(self) -> dict[str, float]:
        min_phase, max_phase = self._adaptive_phase_range(horizon=1)
        if self.reset_phase_sampling == "adaptive":
            stats = self.adaptive_sampler.stats()
            stats["mode"] = 0.0
            return stats
        if self.reset_phase_sampling == "rsi":
            keyframes = self._rsi_keyframe_phases(min_phase, max_phase)
            count = max(1, int(keyframes.numel()))
            return {
                "mode": 1.0,
                "top_bin": -1.0,
                "top_prob": 1.0 / float(count),
                "failed_sum": 0.0,
                "entropy": 1.0 if count > 1 else 0.0,
                "peak_bin": -1.0,
                "rsi_keyframe_count": float(count),
            }
        if self.reset_phase_sampling == "zero":
            return {
                "mode": 2.0,
                "top_bin": 0.0,
                "top_prob": 1.0,
                "failed_sum": 0.0,
                "entropy": 0.0,
                "peak_bin": 0.0,
                "rsi_keyframe_count": 1.0,
            }
        total = max(1, max_phase - min_phase + 1)
        return {
            "mode": 3.0,
            "top_bin": -1.0,
            "top_prob": 1.0 / float(total),
            "failed_sum": 0.0,
            "entropy": 1.0 if total > 1 else 0.0,
            "peak_bin": -1.0,
        }
