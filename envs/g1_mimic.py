from __future__ import annotations

import torch

from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
from engine.config import EnvironmentConfig

from .adaptive_sampling import AdaptiveTimestepsSampler
from .imitation_data import G1_IMITATION_FRAME_DIM, G1_IMITATION_KEY_BODY_NAMES
from .spec import (
    CRITIC_OBS_DIM,
    OBS_DIM,
    PUSH_INTERVAL_STEP_RANGE,
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
from .motion import MimicMotionReference
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
        observation_group_size: int,
        *,
        render: bool = False,
        render_every: int = 1,
        contact_debug_vis: bool = False,
    ):
        self.config = cfg
        self.task = resolve_task(cfg.task)
        self.observation_group_size = observation_group_size
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
        missing_imitation_bodies = [name for name in G1_IMITATION_KEY_BODY_NAMES if name not in self.robot.body_names]
        if missing_imitation_bodies:
            raise ValueError(f"imitation key bodies are missing from the robot asset: {missing_imitation_bodies}")
        self.imitation_key_body_ids = torch.tensor(
            [self.robot.body_names.index(name) for name in G1_IMITATION_KEY_BODY_NAMES],
            dtype=torch.long,
            device=self.device,
        )
        self.anchor_body_id = self.robot.body_names.index(MIMIC_ANCHOR_BODY_NAME)
        self.ee_body_names = list(MIMIC_EE_BODY_NAMES)
        self.ee_body_indices = [self.track_body_names.index(name) for name in MIMIC_EE_BODY_NAMES]
        termination_names = MIMIC_TERMINATION_BODY_NAMES
        self.termination_body_indices = [self.track_body_names.index(name) for name in termination_names]
        self.contact_sensor = self.scene["contact_forces"]

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
        self.foot_body_names = list(MIMIC_FOOT_BODY_NAMES)
        self.foot_contact_body_ids = torch.tensor(
            [self.contact_sensor.body_names.index(name) for name in self.foot_body_names],
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

        self.reset()

    @property
    def observation_dim(self) -> int:
        return OBS_DIM

    @property
    def critic_observation_dim(self) -> int:
        return CRITIC_OBS_DIM

    @property
    def imitation_frame_dim(self) -> int:
        return G1_IMITATION_FRAME_DIM

    def get_imitation_demo_history(
        self,
        phase_indices: torch.Tensor,
        window_size: int,
        *,
        flatten: bool = False,
    ) -> torch.Tensor:
        return self.motion.get_imitation_demo_history(
            phase_indices,
            window_size,
            flatten=flatten,
        )

    def sample_imitation_demo_windows(
        self,
        num_samples: int,
        window_size: int = 16,
        *,
        flatten: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return self.motion.sample_imitation_demo_windows(
            num_samples,
            window_size,
            flatten=flatten,
            generator=generator,
        )

    def get_imitation_demo_windows_at_end_indices(
        self,
        end_indices: torch.Tensor,
        window_size: int = 16,
        *,
        flatten: bool = True,
    ) -> torch.Tensor:
        return self.motion.get_imitation_demo_windows_at_end_indices(
            end_indices,
            window_size,
            flatten=flatten,
        )

    def _adaptive_phase_range(self, horizon: int) -> tuple[int, int]:

        horizon = max(1, int(horizon))
        min_phase = self.motion_start_phase
        max_phase = min(
            self.motion_end_phase,
            max(0, self.motion.num_frames - max(horizon, 2)),
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
        if self.reset_phase_sampling == "uniform" or not self.adaptive_motion_sampling:
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

    def reset(self, phase_indices: torch.Tensor | None = None) -> torch.Tensor:
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        return self.reset_envs(env_ids=env_ids, phase_indices=phase_indices)

    def reset_envs(self, env_ids: torch.Tensor, phase_indices: torch.Tensor | None = None) -> torch.Tensor:
        if env_ids.ndim != 1:
            raise ValueError(f"env_ids must have shape (N,), got {tuple(env_ids.shape)}")
        if env_ids.numel() == 0:
            return torch.empty(0, self.observation_dim, device=self.device)
        if phase_indices is None:
            phase_indices = self.sample_phase_indices(env_ids.numel(), horizon=1)
        if phase_indices.shape != (env_ids.numel(),):
            raise ValueError(
                f"phase_indices must have shape {(env_ids.numel(),)}, got {tuple(phase_indices.shape)}"
            )

        phase_indices = self.motion.clamp_time_steps(phase_indices)
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
        self.last_action[env_ids] = 0.0


        self._failure_recorded[env_ids] = False
        min_push, max_push = PUSH_INTERVAL_STEP_RANGE
        self.next_push_step[env_ids] = torch.randint(
            min_push,
            max_push + 1,
            (env_ids.numel(),),
            dtype=torch.long,
            device=self.device,
        )
        self.first_push_step[env_ids] = -1

        self.scene.reset(env_ids=env_ids)
        reference = self.motion.get_frame(phase_indices)
        root_pos = reference["root_pos_w"].clone()
        root_quat = reference["root_quat_w"].clone()
        root_lin_vel = reference["root_lin_vel_w"].clone()
        root_ang_vel = reference["root_ang_vel_w"].clone()
        joint_pos = reference["joint_pos"].clone()
        joint_vel = reference["joint_vel"].clone()
        if self.config.reset_noise:
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
        self.scene.update(self.physics_dt)
        observation = self.get_observation()
        return observation.index_select(0, env_ids)

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
        joint_pos += joint_low + (joint_high - joint_low) * torch.rand_like(joint_pos)
        soft_limits = self.robot.data.soft_joint_pos_limits.index_select(0, env_ids)
        joint_pos[:] = torch.clamp(joint_pos, soft_limits[:, self.action_joint_ids, 0], soft_limits[:, self.action_joint_ids, 1])

    def _apply_action_targets(self, actions: torch.Tensor) -> None:


        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"Expected action shape {(self.num_envs, self.action_dim)}, got {tuple(actions.shape)}")

        clipped_actions = torch.clamp(actions, -100.0, 100.0)
        action_targets = self.default_action_joint_pos + self.action_scale * clipped_actions
        self.robot.set_joint_position_target(action_targets, joint_ids=self.action_joint_ids)

    def _init_adaptive_motion_sampling(self) -> None:
        self.reset_phase_sampling = str(self.config.reset_phase_sampling)
        self.rsi_keyframe_count = int(self.config.rsi_keyframe_count)
        self.adaptive_motion_sampling = bool(
            self.config.adaptive_motion_sampling and self.reset_phase_sampling == "adaptive"
        )
        self.adaptive_sampler = AdaptiveTimestepsSampler(
            motion_time_step_total=int(self.motion.num_frames),
            device=self.device,
            num_bins=int(self.config.adaptive_num_bins),
            env_fps=int(round(1.0 / float(self.config.sim_dt))) if float(self.config.sim_dt) > 0 else 50,
            adaptive_alpha=float(self.config.adaptive_alpha),
            adaptive_predecessor_ratio=float(self.config.adaptive_predecessor_ratio),
            adaptive_predecessor_lookback_bins=int(self.config.adaptive_predecessor_lookback_bins),
        )
        self._failure_recorded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.record_motion_failures = True
        # Preserve Flow-CPS's historical default.  FCAMP opts in through its own
        # environment config without requiring this module to import FCAMP.
        self.terminate_on_motion_end = bool(getattr(self.config, "terminate_on_motion_end", False))

    def _record_adaptive_failures(
        self,
        tracking_failure: torch.Tensor,
        death_phase_steps: torch.Tensor,
    ) -> None:

        if not self.adaptive_motion_sampling or not self.record_motion_failures:
            return
        failure = tracking_failure & (~self._failure_recorded)
        if bool(failure.any()):
            self.adaptive_sampler.update_current_failure_count(death_phase_steps[failure])
            self._failure_recorded |= failure

    def _fold_adaptive_sampler(self) -> None:

        if not self.adaptive_motion_sampling or not self.record_motion_failures:
            return
        self.adaptive_sampler.update_failure_ema()

    def adaptive_sampling_stats(self) -> dict[str, float]:
        min_phase, max_phase = self._adaptive_phase_range(horizon=1)
        if self.reset_phase_sampling == "adaptive":
            stats = self.adaptive_sampler.stats(min_phase, max_phase)
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

    def _resample_finished_motions(self) -> tuple[torch.Tensor, torch.Tensor]:

        env_ids = torch.where(self.phase_steps >= self.motion.num_frames)[0]
        if env_ids.numel() == 0:
            return env_ids, torch.empty(0, dtype=torch.long, device=self.device)
        phase_indices = self.sample_phase_indices(env_ids.numel(), horizon=1)
        self.phase_steps[env_ids] = phase_indices.to(dtype=self.phase_steps.dtype)
        self._failure_recorded[env_ids] = False
        reference = self.motion.get_frame(phase_indices)
        root_pos = reference["root_pos_w"].clone()
        root_quat = reference["root_quat_w"].clone()
        root_lin_vel = reference["root_lin_vel_w"].clone()
        root_ang_vel = reference["root_ang_vel_w"].clone()
        joint_pos = reference["joint_pos"].clone()
        joint_vel = reference["joint_vel"].clone()
        if self.config.reset_noise:
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
        self.scene.update(self.physics_dt)
        return env_ids, phase_indices
