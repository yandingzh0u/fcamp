from __future__ import annotations

import torch

from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from .config import (
    CRITIC_OBS_DIM,
    FUTURE_REF_FRAME_DIM,
    OBS_DIM,
    PUSH_INTERVAL_STEP_RANGE,
    RESET_JOINT_POSITION_RANGE,
    RESET_ROOT_POSE_RANGE,
    MimicEnvConfig,
    VELOCITY_RANGE,
)
from .motion import MimicMotionReference
from .observation import MimicObservationMixin
from .reward import MimicRewardMixin
from .robot import G1Env
from .step import MimicStepMixin
from .terminal import MimicTerminationMixin


class G1MimicEnv(
    MimicStepMixin,
    MimicTerminationMixin,
    MimicRewardMixin,
    MimicObservationMixin,
    G1Env,
):
    def __init__(self, cfg: MimicEnvConfig):
        self.task_cfg = cfg
        super().__init__(cfg)
        track_body_ids, track_body_names = self.robot.find_bodies(list(cfg.track_body_names), preserve_order=True)
        self.track_body_ids = torch.tensor(track_body_ids, dtype=torch.long, device=self.device)
        self.track_body_names = list(track_body_names)
        self.anchor_body_id = self.robot.body_names.index(cfg.anchor_body_name)
        self.ee_body_names = list(cfg.ee_body_names)
        self.ee_body_indices = [self.track_body_names.index(name) for name in cfg.ee_body_names]
        self.termination_body_indices = list(self.ee_body_indices)
        self.contact_sensor = self.scene["contact_forces"]
        allowed_contact_names = set(cfg.ee_body_names)
        self.undesired_contact_body_ids = torch.tensor(
            [
                self.contact_sensor.body_names.index(body_name)
                for body_name in self.contact_sensor.body_names
                if body_name not in allowed_contact_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        self.foot_body_names = list(cfg.foot_body_names)
        self.foot_contact_body_ids = torch.tensor(
            [self.contact_sensor.body_names.index(name) for name in self.foot_body_names],
            dtype=torch.long,
            device=self.device,
        )
        self.motion = MimicMotionReference(
            cfg.motion_file,
            self.track_body_ids,
            self.anchor_body_id,
            self.device,
        )
        self._init_adaptive_motion_sampling()

        self.last_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.phase_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_start_phase = max(0, min(int(cfg.motion_start_phase), self.motion.num_frames - 1))
        if cfg.motion_end_phase < 0:
            self.motion_end_phase = self.motion.num_frames - 1
        else:
            self.motion_end_phase = max(self.motion_start_phase, min(int(cfg.motion_end_phase), self.motion.num_frames - 1))

        self.reset()

    @property
    def observation_dim(self) -> int:
        return OBS_DIM + int(self.task_cfg.future_ref_steps) * FUTURE_REF_FRAME_DIM

    @property
    def critic_observation_dim(self) -> int:
        return CRITIC_OBS_DIM

    def sample_phase_indices(self, num_samples: int, horizon: int) -> torch.Tensor:
        if num_samples < 0:
            raise ValueError(f"num_samples must be >= 0, got {num_samples}")
        if num_samples == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        horizon = max(1, int(horizon))
        sampling_probabilities = self.bin_failed_count + self.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.adaptive_kernel_size - 1),
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(
            sampling_probabilities,
            self.adaptive_kernel.view(1, 1, -1),
        ).view(-1)
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()
        sampled_bins = torch.multinomial(sampling_probabilities, num_samples, replacement=True)
        phase_indices = (
            (sampled_bins + torch.rand(num_samples, device=self.device))
            / self.bin_count
            * (self.motion.num_frames - 1)
        ).long()
        max_phase = min(
            self.motion_end_phase,
            max(0, self.motion.num_frames - horizon),
        )
        if max_phase < self.motion_start_phase:
            return torch.full((num_samples,), self.motion_start_phase, dtype=torch.long, device=self.device)
        return torch.clamp(phase_indices, min=self.motion_start_phase, max=max_phase)

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
        self.phase_steps[env_ids] = phase_indices
        self.episode_steps[env_ids] = 0
        self.last_action[env_ids] = 0.0
        min_push, max_push = PUSH_INTERVAL_STEP_RANGE
        self.next_push_step[env_ids] = torch.randint(
            min_push,
            max_push + 1,
            (env_ids.numel(),),
            dtype=torch.long,
            device=self.device,
        )

        self.scene.reset(env_ids=env_ids)
        reference = self.motion.get_frame(phase_indices)
        root_pos = reference["root_pos_w"].clone()
        root_quat = reference["root_quat_w"].clone()
        root_lin_vel = reference["root_lin_vel_w"].clone()
        root_ang_vel = reference["root_ang_vel_w"].clone()
        joint_pos = reference["joint_pos"].clone()
        joint_vel = reference["joint_vel"].clone()
        if self.task_cfg.reset_noise:
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

    def _apply_action_targets(self, action_offsets: torch.Tensor) -> None:
        if action_offsets.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"Expected action shape {(self.num_envs, self.action_dim)}, got {tuple(action_offsets.shape)}")

        next_phase = self.motion.clamp_time_steps(self.phase_steps + 1)
        ref_joint_pos = self.motion.joint_pos.index_select(0, next_phase)
        action_targets = ref_joint_pos + self.action_scale * action_offsets
        self.robot.set_joint_position_target(action_targets, joint_ids=self.action_joint_ids)

    def _init_adaptive_motion_sampling(self) -> None:
        self.bin_count = int(self.motion.num_frames // (1.0 / self.dt)) + 1
        self.bin_count = max(self.bin_count, 1)
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float32, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float32, device=self.device)
        self.adaptive_kernel_size = 1
        self.adaptive_uniform_ratio = 0.1
        self.adaptive_alpha = 0.001
        kernel = torch.tensor([0.8**i for i in range(self.adaptive_kernel_size)], dtype=torch.float32, device=self.device)
        self.adaptive_kernel = kernel / kernel.sum()

    def _record_adaptive_motion_failures(self, failed_env_ids: torch.Tensor, failure_phase_steps: torch.Tensor) -> None:
        if failed_env_ids.numel() == 0:
            return
        current_bin_index = torch.clamp(
            (failure_phase_steps * self.bin_count) // max(self.motion.num_frames, 1),
            0,
            self.bin_count - 1,
        )
        fail_bins = current_bin_index.index_select(0, failed_env_ids)
        self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count).to(self._current_bin_failed)

    def _update_adaptive_motion_sampling(self) -> None:
        self.bin_failed_count = (
            self.adaptive_alpha * self._current_bin_failed
            + (1.0 - self.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _resample_finished_motions(self) -> None:
        env_ids = torch.where(self.phase_steps >= self.motion.num_frames)[0]
        if env_ids.numel() == 0:
            return
        phase_indices = self.sample_phase_indices(env_ids.numel(), horizon=1)
        self.phase_steps[env_ids] = phase_indices
        reference = self.motion.get_frame(phase_indices)
        root_pos = reference["root_pos_w"].clone()
        root_quat = reference["root_quat_w"].clone()
        root_lin_vel = reference["root_lin_vel_w"].clone()
        root_ang_vel = reference["root_ang_vel_w"].clone()
        joint_pos = reference["joint_pos"].clone()
        joint_vel = reference["joint_vel"].clone()
        if self.task_cfg.reset_noise:
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
