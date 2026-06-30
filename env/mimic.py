from __future__ import annotations

import torch

from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from .adaptive_sampling import AdaptiveTimestepsSampler
from .config import (
    CRITIC_OBS_DIM,
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
from .robots.g1 import G1_29DOF_ACTION_NAMES
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
        termination_names = getattr(cfg, "termination_body_names", None) or cfg.ee_body_names
        self.termination_body_indices = [self.track_body_names.index(name) for name in termination_names]
        self.contact_sensor = self.scene["contact_forces"]
        from .config import CONTACT_ALLOWED_SUBSTRINGS

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
        self.foot_body_names = list(cfg.foot_body_names)
        self.foot_contact_body_ids = torch.tensor(
            [self.contact_sensor.body_names.index(name) for name in self.foot_body_names],
            dtype=torch.long,
            device=self.device,
        )
        # Contact-sensor ids for the termination bodies (ankles + wrists). The actor needs to
        # see the current contact state of the exact bodies whose z-error kills the episode,
        # otherwise its chunk-start observation is blind to how close the wrists/ankles are to
        # the termination gate (the dominant death cause in crawl).
        self.termination_contact_body_ids = torch.tensor(
            [self.contact_sensor.body_names.index(name) for name in termination_names],
            dtype=torch.long,
            device=self.device,
        )
        self.motion = MimicMotionReference(
            cfg.motion_file,
            self.track_body_ids,
            self.anchor_body_id,
            self.device,
            robot_body_names=list(self.robot.body_names),
            action_joint_names=list(G1_29DOF_ACTION_NAMES),
            root_body_name="pelvis",
        )
        self._init_adaptive_motion_sampling()

        # Adaptive episode cap: when max_episode_steps <= 0, the time-out follows the motion
        # length so "survive the whole clip" is the real success bar instead of an arbitrary
        # fixed horizon. The motion-end termination already caps episodes at num_frames; this
        # keeps the explicit time_out consistent with the clip and robust to clip-length changes.
        if int(cfg.max_episode_steps) <= 0:
            self.task_cfg.max_episode_steps = int(self.motion.num_frames)

        self.last_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self.prev_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
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
        return OBS_DIM

    @property
    def critic_observation_dim(self) -> int:
        return CRITIC_OBS_DIM

    def sample_phase_indices(self, num_samples: int, horizon: int) -> torch.Tensor:
        if num_samples < 0:
            raise ValueError(f"num_samples must be >= 0, got {num_samples}")
        if num_samples == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        horizon = max(1, int(horizon))
        min_phase = self.motion_start_phase
        # Never start on the final frame: the last valid reference frame is the terminal frame,
        # so an env spawned there motion-times-out on its very first step. Hold back at least 2
        # frames (Holosoma retreats the start to the second-to-last frame) so a fresh episode
        # always has at least one real tracking step.
        max_phase = min(
            self.motion_end_phase,
            max(0, self.motion.num_frames - max(horizon, 2)),
        )
        if max_phase < min_phase:
            return torch.full((num_samples,), min_phase, dtype=torch.long, device=self.device)
        if not self.adaptive_motion_sampling:
            return torch.randint(min_phase, max_phase + 1, (num_samples,), dtype=torch.long, device=self.device)

        # Holosoma death-frame sampler: bins are weighted by the EMA of the frames the robot
        # actually dies at (plus a uniform floor). The sampler is fed only by env.step().
        return self.adaptive_sampler.sample_frames(num_samples, min_phase, max_phase)

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
        self.prev_action[env_ids] = 0.0
        # New episode for these envs: clear the per-episode death-record guard so their next
        # termination is counted by the adaptive sampler exactly once.
        if hasattr(self, "_failure_recorded"):
            self._failure_recorded[env_ids] = False
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

    def _apply_action_targets(self, actions: torch.Tensor) -> None:
        # Official WBT action semantics (use_default_offset=True): the PD target is the default
        # joint pose plus the scaled policy action,  q_target = q_default + S * a_t.
        # The raw action is clipped to +-100 before forming the PD target (official); the
        # unclipped action is what reaches the action-rate reward (handled in step.py).
        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"Expected action shape {(self.num_envs, self.action_dim)}, got {tuple(actions.shape)}")

        clipped_actions = torch.clamp(actions, -100.0, 100.0)
        action_targets = self.default_action_joint_pos + self.action_scale * clipped_actions
        self.robot.set_joint_position_target(action_targets, joint_ids=self.action_joint_ids)

    def _init_adaptive_motion_sampling(self) -> None:
        self.adaptive_motion_sampling = bool(self.task_cfg.adaptive_motion_sampling)
        # ~1-second bins over the global motion-frame axis (env_fps = round(1/dt)).
        env_fps = max(1, int(round(1.0 / self.dt)))
        num_bins = max(1, int(self.motion.num_frames // env_fps) + 1)
        self.adaptive_sampler = AdaptiveTimestepsSampler(
            motion_time_step_total=int(self.motion.num_frames),
            device=self.device,
            num_bins=num_bins,
            adaptive_kernel_size=max(1, int(self.task_cfg.adaptive_kernel_size)),
            adaptive_uniform_ratio=float(self.task_cfg.adaptive_uniform_ratio),
            adaptive_alpha=float(self.task_cfg.adaptive_alpha),
        )
        # Per-env guard so the same termination is recorded by the sampler exactly once even
        # when the env is not auto-reset (validation / GRPO branch rollouts replay dead envs).
        self._failure_recorded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Validation / probes flip this off so eval deaths do not feed the training sampler.
        self.record_motion_failures = True

    def _record_adaptive_failures(
        self,
        tracking_failure: torch.Tensor,
        death_phase_steps: torch.Tensor,
    ) -> None:
        """Record this step's tracking-failure death frames into the per-step accumulator.

        ``tracking_failure`` is the union of the real tracking terminations
        (anchor_pos_bad | anchor_ori_bad | ee_body_bad). A pure motion-end/episode-cap timeout is
        NOT a failure, but a tracking failure that happens to coincide with a timeout on the same
        step IS still counted (Holosoma records any non-timeout termination cause).

        Called BEFORE reset/phase-advance. Only touches ``current_bin_failed_count`` -- it does
        NOT fold the EMA, so the reset that follows still samples from the OLD EMA. Guarded so a
        single termination is counted once per episode (matters when auto_reset=False)."""
        if not self.adaptive_motion_sampling or not self.record_motion_failures:
            return
        failure = tracking_failure & (~self._failure_recorded)
        if bool(failure.any()):
            self.adaptive_sampler.update_current_bin_failed_count(death_phase_steps[failure])
            self._failure_recorded |= failure

    def _fold_adaptive_sampler(self) -> None:
        """Fold the per-step failure accumulator into the EMA, then zero it.

        Called at the END of the step, AFTER reset/phase-advance (official order: record death
        -> reset using the old EMA -> update the EMA). Folding here avoids an instantaneous jump
        in the sampling distribution when many envs die on the same step and are reset against it."""
        if not self.adaptive_motion_sampling or not self.record_motion_failures:
            return
        self.adaptive_sampler.update_bin_failed_count()

    def adaptive_sampling_stats(self) -> dict[str, float]:
        return self.adaptive_sampler.stats()

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
