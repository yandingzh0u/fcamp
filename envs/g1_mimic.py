from __future__ import annotations

import torch

from components.rollout.reset_diagnostics import ResetPhaseRecorder
from engine.config import EnvironmentConfig

from .imitation_data import G1_IMITATION_FRAME_DIM, G1_IMITATION_KEY_BODY_NAMES
from .spec import (
    CRITIC_OBS_DIM,
    OBS_DIM,
    MIMIC_ANCHOR_BODY_NAME,
    MIMIC_BODY_NAMES,
    MIMIC_EE_BODY_NAMES,
    MIMIC_TERMINATION_BODY_NAMES,
)
from .motion import (
    MimicMotionReference,
    mimickit_frame_delta,
    mimickit_full_motion_steps,
)
from .observation import MimicObservationMixin
from .robot import G1Env, RootVelocityFrame
from .robots.g1 import G1_29DOF_ACTION_NAMES, G1_LOCAL_URDF_PATH
from .step import MimicStepMixin
from .terminal import MimicTerminationMixin
from .tasks import resolve_task


class G1MimicEnv(
    MimicStepMixin,
    MimicTerminationMixin,
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
        self.robot_asset_path = G1_LOCAL_URDF_PATH.resolve()
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
        missing_contact_bodies = [
            name
            for name in self.task.allowed_contact_bodies
            if name not in self.contact_sensor.body_names
        ]
        if missing_contact_bodies:
            raise ValueError(
                "AMP allowed-contact bodies are missing from the contact "
                f"sensor: {missing_contact_bodies}"
            )
        self.allowed_contact_sensor_ids = torch.tensor(
            [
                self.contact_sensor.body_names.index(name)
                for name in self.task.allowed_contact_bodies
            ],
            dtype=torch.long,
            device=self.device,
        )
        if self.uses_ground_contact_filter:
            force_matrix = self.contact_sensor.data.force_matrix_w
            expected_shape = (self.num_envs, len(self.contact_sensor.body_names), 1, 3)
            if not torch.is_tensor(force_matrix) or tuple(force_matrix.shape) != expected_shape:
                actual_shape = None if force_matrix is None else tuple(force_matrix.shape)
                raise RuntimeError(
                    f"Ground-filter contact matrix shape mismatch: expected {expected_shape}, got {actual_shape}"
                )
        self.motion = MimicMotionReference(
            str(self.task.motion_file),
            self.track_body_ids,
            self.anchor_body_id,
            self.device,
            robot_body_names=list(self.robot.body_names),
            action_joint_names=list(G1_29DOF_ACTION_NAMES),
            root_body_name="pelvis",
            kinematic_urdf_file=self.robot_asset_path,
        )
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
            end_phase=self.motion_end_phase,
            device=self.device,
            log_num_bins=(
                self.motion.num_frames
                // max(1, int(round(1.0 / float(self.dt))))
                + 1
            ),
        )
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

    def sample_phase_indices(self, num_samples: int, horizon: int) -> torch.Tensor:
        if num_samples < 0:
            raise ValueError(f"num_samples must be >= 0, got {num_samples}")
        if num_samples == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        # Standard AMP samples reset states over the complete demonstration.
        # ``horizon`` is intentionally irrelevant: reference motion completion
        # is not terminal, so a reset near the clip endpoint is fully valid.
        del horizon
        min_phase = int(self.motion_start_phase)
        max_phase = int(self.motion_end_phase)
        if max_phase < min_phase:
            return torch.full((num_samples,), min_phase, dtype=torch.long, device=self.device)
        if max_phase == min_phase:
            return torch.full(
                (num_samples,),
                float(min_phase),
                dtype=torch.float32,
                device=self.device,
            )
        # Official AMP random-state initialization samples a continuous motion
        # time. With the native 50 Hz clip and 50 Hz controller, each episode
        # keeps that random fractional phase offset while advancing by one
        # motion frame per simulator step.
        return (
            torch.rand(
                num_samples,
                dtype=torch.float32,
                device=self.device,
            )
            * float(max_phase - min_phase)
            + float(min_phase)
        )

    def reset(
        self,
        phase_indices: torch.Tensor | None = None,
        *,
        root_velocity_frame: RootVelocityFrame | None = None,
    ) -> torch.Tensor:
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        return self.reset_envs(
            env_ids=env_ids,
            phase_indices=phase_indices,
            root_velocity_frame=root_velocity_frame,
        )

    def reset_envs(
        self,
        env_ids: torch.Tensor,
        phase_indices: torch.Tensor | None = None,
        *,
        root_velocity_frame: RootVelocityFrame | None = None,
    ) -> torch.Tensor:
        self._reset_env_state(
            env_ids,
            phase_indices=phase_indices,
            root_velocity_frame=root_velocity_frame,
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
        """Normalize an absolute reference joint target like MimicKit."""

        reference_action = reference_joint_pos / self.action_scale
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
        *,
        root_velocity_frame: RootVelocityFrame | None = None,
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

        phase_indices = self.motion.clamp_time_steps(phase_indices)
        self.reset_phase_recorder.record(phase_indices)
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
        self._write_robot_state(
            root_pos=root_pos,
            root_quat=root_quat,
            root_lin_vel=root_lin_vel,
            root_ang_vel=root_ang_vel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            env_ids=env_ids,
            root_velocity_frame=root_velocity_frame,
        )
        # ``last_action`` is observation/debug state only in standard AMP; it is
        # never used as the mathematical origin of the next policy action.
        self.last_action[env_ids] = reference_action
        self.scene.update(self.physics_dt)

    def begin_reset_phase_diagnostics(self) -> None:
        self.reset_phase_recorder.begin()

    def finish_reset_phase_diagnostics(self) -> dict[str, float]:
        return self.reset_phase_recorder.finish()

    def _apply_action_targets(self, actions: torch.Tensor) -> torch.Tensor:


        if actions.shape != (self.num_envs, self.action_dim):
            raise ValueError(f"Expected action shape {(self.num_envs, self.action_dim)}, got {tuple(actions.shape)}")

        if not bool(torch.isfinite(actions).all()):
            raise RuntimeError("Policy produced a non-finite normalized action")
        if self._policy_action_low is None or self._policy_action_high is None:
            raise RuntimeError("No normalized AMP action contract is installed")
        # MimicKit computes the Gaussian density in normalized action space,
        # unnormalizes linearly, and clips only at the environment boundary.
        # Clipping here is exactly equivalent because the physical action
        # interval is the affine image of [-1, 1].
        clipped_actions = torch.maximum(
            torch.minimum(actions, self._policy_action_high),
            self._policy_action_low,
        )
        action_targets = self.action_scale * clipped_actions
        self.robot.set_joint_position_target(action_targets, joint_ids=self.action_joint_ids)
        return clipped_actions
