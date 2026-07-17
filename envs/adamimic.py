from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


ADAMIMIC_NUM_DOFS = 29
ADAMIMIC_FULL_ACTION_DIM = ADAMIMIC_NUM_DOFS + 1
ADAMIMIC_ONE_STEP_OBSERVATION_DIM = 100
ADAMIMIC_OBSERVATION_DIM = 5 * ADAMIMIC_ONE_STEP_OBSERVATION_DIM
ADAMIMIC_CRITIC_OBSERVATION_DIM = 136
ADAMIMIC_REWARD_GROUPS = ("dense", "sparse")

# The official G1 asset exposes seventeen kinematic key-frame bodies.  These
# are the name-equivalent links on the shared 29-DoF G1 asset.
ADAMIMIC_BODY_NAMES = (
    "pelvis",
    "head_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
)
_UPPER_BODY_INDICES = tuple(range(11))
_LOWER_BODY_INDICES = tuple(range(11, 17))
_FEET_BODY_INDICES = (13, 16)
_LIDAR_PARENT_BODY_NAME = "torso_link"
# ``mid360_link`` is fixed to the torso and is merged by the shared IsaacLab
# URDF importer because it has no inertial body.  Keep its exact URDF transform
# so AdaMimic still observes the official lidar-frame odometry.
_LIDAR_PARENT_POS = (0.0002835, 0.00003, 0.41618)
_LIDAR_PARENT_RPY = (0.0, 3.101, 3.1415)


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat[..., :1], -quat[..., 1:]), dim=-1)


def _quat_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs.unbind(dim=-1)
    rw, rx, ry, rz = rhs.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quat_rotate_inverse(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., 1:]
    t = 2.0 * torch.cross(-xyz, vector, dim=-1)
    return vector + quat[..., :1] * t + torch.cross(-xyz, t, dim=-1)


def _quat_rotate(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return _quat_rotate_inverse(_quat_conjugate(quat), vector)


def _quat_angle(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    delta = _quat_mul(lhs, _quat_conjugate(rhs))
    xyz_norm = torch.linalg.vector_norm(delta[..., 1:], dim=-1)
    return 2.0 * torch.atan2(xyz_norm, delta[..., 0].abs().clamp_min(1.0e-9))


def _quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cr, sr = torch.cos(0.5 * roll), torch.sin(0.5 * roll)
    cp, sp = torch.cos(0.5 * pitch), torch.sin(0.5 * pitch)
    cy, sy = torch.cos(0.5 * yaw), torch.sin(0.5 * yaw)
    return torch.stack(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dim=-1,
    )


def _as_vector(value: torch.Tensor | float, count: int, device: torch.device) -> torch.Tensor:
    result = torch.as_tensor(value, dtype=torch.float32, device=device)
    if result.ndim == 0:
        result = result.expand(count)
    return result.reshape(count)


@dataclass(slots=True)
class _Curriculum:
    stage: str
    termination_enabled: bool
    termination_threshold: float
    termination_min: float
    termination_max: float
    termination_degree: float
    termination_down: float
    termination_up: float
    reverse_termination: bool
    reverse_iteration: int
    limit_enabled: bool
    soft_pos: float
    soft_vel: float
    soft_torque: float
    limit_min: float
    limit_max: float
    limit_degree: float
    limit_down: float
    limit_up: float
    penalty_enabled: bool
    penalty_scale: float
    penalty_min: float
    penalty_max: float
    penalty_degree: float
    penalty_down: float
    penalty_up: float
    average_episode_length: float = 0.0
    update_index: int = 0

    @classmethod
    def from_config(cls, cfg: Any) -> _Curriculum:
        stage = str(cfg.stage)
        stage1 = stage == "stage1"
        if stage not in {"stage1", "stage2"}:
            raise ValueError(f"AdaMimic stage must be stage1 or stage2, got {stage!r}")
        return cls(
            stage=stage,
            termination_enabled=bool(cfg.termination_curriculum),
            termination_threshold=float(cfg.termination_initial_threshold),
            termination_min=float(cfg.termination_min_threshold),
            termination_max=float(cfg.termination_max_threshold),
            termination_degree=float(cfg.termination_curriculum_degree),
            termination_down=float(cfg.termination_level_down_threshold),
            termination_up=float(cfg.termination_level_up_threshold),
            reverse_termination=bool(cfg.reverse_term_curriculum),
            reverse_iteration=int(cfg.reverse_term_curriculum_iter),
            limit_enabled=bool(cfg.limit_curriculum),
            soft_pos=float(cfg.limit_initial_soft_factor),
            soft_vel=float(cfg.limit_initial_soft_factor),
            soft_torque=float(cfg.limit_initial_soft_factor),
            limit_min=float(cfg.limit_min_soft_factor),
            limit_max=float(cfg.limit_max_soft_factor),
            limit_degree=float(cfg.limit_curriculum_degree),
            limit_down=float(cfg.limit_level_down_threshold),
            limit_up=float(cfg.limit_level_up_threshold),
            penalty_enabled=bool(cfg.penalty_curriculum),
            penalty_scale=float(cfg.penalty_initial_scale),
            penalty_min=float(cfg.penalty_min_scale),
            penalty_max=float(cfg.penalty_max_scale),
            penalty_degree=float(cfg.penalty_curriculum_degree),
            penalty_down=float(cfg.penalty_level_down_threshold),
            penalty_up=float(cfg.penalty_level_up_threshold),
        )

    def set_update(self, update_index: int) -> None:
        self.update_index = int(update_index)

    def update(self, episode_lengths: torch.Tensor) -> None:
        if episode_lengths.numel() == 0:
            return
        count = int(episode_lengths.numel())
        weight = min(float(count) / 10_000.0, 1.0)
        current = float(episode_lengths.float().mean().item())
        self.average_episode_length = (1.0 - weight) * self.average_episode_length + weight * current

        if self.termination_enabled:
            if self.reverse_termination and self.update_index >= self.reverse_iteration:
                self.termination_threshold = 2.0
            elif self.average_episode_length < self.termination_down:
                self.termination_threshold *= 1.0 + self.termination_degree
            elif self.average_episode_length > self.termination_up:
                self.termination_threshold *= 1.0 - self.termination_degree
            self.termination_threshold = min(
                self.termination_max, max(self.termination_min, self.termination_threshold)
            )

        if self.penalty_enabled:
            if self.average_episode_length < self.penalty_down:
                self.penalty_scale *= 1.0 - self.penalty_degree
            elif self.average_episode_length > self.penalty_up:
                self.penalty_scale *= 1.0 + self.penalty_degree
            self.penalty_scale = min(self.penalty_max, max(self.penalty_min, self.penalty_scale))

        if self.limit_enabled:
            factor = 1.0
            if self.average_episode_length < self.limit_down:
                factor += self.limit_degree
            elif self.average_episode_length > self.limit_up:
                factor -= self.limit_degree
            self.soft_pos = min(self.limit_max, max(self.limit_min, self.soft_pos * factor))
            self.soft_vel = min(self.limit_max, max(self.limit_min, self.soft_vel * factor))
            self.soft_torque = min(self.limit_max, max(self.limit_min, self.soft_torque * factor))


class AdaMimicEnvironment:
    """Official AdaMimic environment semantics on the shared 29-DoF G1.

    Training uses the paper's observation history, two reward groups and
    key-frame termination.  ``step_evaluation`` deliberately delegates reward
    and termination to the shared benchmark environment.
    """

    observation_dim = ADAMIMIC_OBSERVATION_DIM
    critic_observation_dim = ADAMIMIC_CRITIC_OBSERVATION_DIM

    def __init__(self, base_env: Any, cfg: Any):
        self.base_env = base_env
        self.cfg = cfg
        if int(base_env.action_dim) != ADAMIMIC_NUM_DOFS:
            raise ValueError(
                f"AdaMimic requires {ADAMIMIC_NUM_DOFS} controlled joints, got {base_env.action_dim}"
            )
        self.stage = str(cfg.stage)
        self.curriculum = _Curriculum.from_config(cfg)
        if int(cfg.actor_observation_history) != 5:
            raise ValueError("Official AdaMimic requires actor_observation_history=5")
        if tuple(tuple(float(v) for v in row) for row in cfg.reward_group_weights) != (
            (0.5, 1.0),
            (0.5, 1.0),
        ):
            raise ValueError("Official AdaMimic reward_group_weights must be [[0.5,1],[0.5,1]]")
        self.apply_reward_scale = bool(cfg.apply_reward_scale)
        self.sparse_global = bool(cfg.sparse_global)
        self.sparse_local = bool(cfg.sparse_local)
        self.special_scale = bool(cfg.special_scale)
        self.special_scale_size = float(cfg.special_scale_size)

        phases = tuple(float(x) for x in base_env.config.adamimic_keyframe_phases)
        if not phases:
            raise ValueError("AdaMimic requires explicit semantic adamimic_keyframe_phases")
        if any(right <= left for left, right in zip(phases, phases[1:])):
            raise ValueError("adamimic_keyframe_phases must be strictly increasing")
        if phases[0] <= 0.0 or phases[-1] > float(base_env.motion.num_frames - 1):
            raise ValueError(
                "adamimic_keyframe_phases must lie in (0, motion.num_frames - 1]"
            )
        self.keyframe_phases = torch.tensor(phases, dtype=torch.float32, device=self.device)
        special = tuple(int(x) for x in base_env.config.adamimic_special_keyframe_indices)
        self.special_keyframe_indices = torch.tensor(special, dtype=torch.long, device=self.device)

        missing = [name for name in ADAMIMIC_BODY_NAMES if name not in base_env.robot.body_names]
        if missing:
            raise ValueError(f"AdaMimic key-frame bodies missing from G1 asset: {missing}")
        self.body_ids = torch.tensor(
            [base_env.robot.body_names.index(name) for name in ADAMIMIC_BODY_NAMES],
            dtype=torch.long,
            device=self.device,
        )
        self.lidar_body_id = (
            base_env.robot.body_names.index("mid360_link")
            if "mid360_link" in base_env.robot.body_names
            else None
        )
        self.lidar_parent_body_id = base_env.robot.body_names.index(_LIDAR_PARENT_BODY_NAME)
        self._lidar_parent_pos = torch.tensor(
            _LIDAR_PARENT_POS, dtype=torch.float32, device=self.device
        )
        lidar_rpy = torch.tensor(_LIDAR_PARENT_RPY, dtype=torch.float32, device=self.device)
        self._lidar_parent_quat = _quat_from_euler_xyz(
            lidar_rpy[0:1], lidar_rpy[1:2], lidar_rpy[2:3]
        )[0]
        self.upper_body_indices = torch.tensor(_UPPER_BODY_INDICES, dtype=torch.long, device=self.device)
        self.lower_body_indices = torch.tensor(_LOWER_BODY_INDICES, dtype=torch.long, device=self.device)
        self.feet_body_indices = torch.tensor(_FEET_BODY_INDICES, dtype=torch.long, device=self.device)

        self._history = torch.zeros(
            self.num_envs,
            5,
            ADAMIMIC_ONE_STEP_OBSERVATION_DIM,
            dtype=torch.float32,
            device=self.device,
        )
        self._actor_observation = self._history.flatten(1)
        self._critic_observation = torch.zeros(
            self.num_envs, ADAMIMIC_CRITIC_OBSERVATION_DIM, device=self.device
        )
        self._last_full_action = torch.zeros(
            self.num_envs, ADAMIMIC_FULL_ACTION_DIM, device=self.device
        )
        self._odometry = torch.zeros(self.num_envs, 3, device=self.device)
        self._odometry_noise = torch.zeros_like(self._odometry)
        self._initial_lidar_pos = torch.zeros_like(self._odometry)
        self._initial_lidar_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self._initial_root_quat = torch.zeros_like(self._initial_lidar_quat)
        angle = torch.zeros(self.num_envs, 3, device=self.device)
        if base_env.observation_noise:
            angle[:, :2].uniform_(-15.0 * torch.pi / 180.0, 15.0 * torch.pi / 180.0)
        self._initial_lidar_orientation_noise = _quat_from_euler_xyz(
            angle[:, 0], angle[:, 1], angle[:, 2]
        )
        self._odometry_initialized = False
        self._training_step_count = 0
        self._first_reset = True

        self.push_robots = True
        push_interval_s = float(cfg.push_interval_seconds)
        self.push_interval_steps = max(1, int(torch.ceil(torch.tensor(push_interval_s / base_env.dt)).item()))
        self.max_push_vel_xy = float(cfg.max_push_velocity_xy)

        self.domain_randomization = bool(cfg.domain_randomization)
        self._delay_buffer = torch.zeros(5, self.num_envs, self.action_dim, device=self.device)
        self._delay_index = torch.full(
            (self.num_envs,), 4, dtype=torch.long, device=self.device
        )
        self._actuation_offset = torch.zeros(
            self.num_envs, self.action_dim, device=self.device
        )
        self._kp_factor = torch.ones_like(self._actuation_offset)
        self._kd_factor = torch.ones_like(self._actuation_offset)
        self._motor_strength = torch.ones_like(self._actuation_offset)
        self._actuator_groups = self._collect_actuator_groups()
        self._nominal_stiffness = self._collect_nominal_stiffness()
        self._nominal_physical_properties = self._capture_nominal_physical_properties()
        self._training_physical_properties = {
            name: value.clone()
            for name, value in self._nominal_physical_properties.items()
        }
        self._using_nominal_physics = True
        if self.domain_randomization:
            self._training_physical_properties = self._randomize_physics_once()
        self._apply_physical_properties(self._training_physical_properties)
        self._using_nominal_physics = not self.domain_randomization
        all_env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._resample_control_randomization(all_env_ids)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base_env, name)

    @property
    def device(self) -> torch.device:
        return self.base_env.device

    @property
    def num_envs(self) -> int:
        return int(self.base_env.num_envs)

    @property
    def action_dim(self) -> int:
        return int(self.base_env.action_dim)

    def set_update(self, update_index: int) -> None:
        self.curriculum.set_update(update_index)

    def sample_rsi(self, count: int) -> torch.Tensor:
        if count == 0:
            return torch.empty(0, dtype=torch.float32, device=self.device)
        if self.stage == "stage2":
            return torch.zeros(count, dtype=torch.float32, device=self.device)
        choices = torch.cat((torch.zeros(1, device=self.device), self.keyframe_phases))
        indices = torch.randint(0, choices.numel(), (count,), device=self.device)
        # Official RSI starts 0.2 ms before the annotated key frame.
        return (choices.index_select(0, indices) - 0.0002 * float(self.base_env.motion.fps)).clamp_min(0.0)

    def reset(
        self,
        phase_indices: torch.Tensor | None = None,
        *,
        warmup: bool = True,
    ) -> torch.Tensor:
        if phase_indices is None:
            # Official construction performs the first reset at motion time 0;
            # RSI becomes active only after that zero-action warmup.
            if self._first_reset:
                phase_indices = torch.zeros(
                    self.num_envs, dtype=torch.float32, device=self.device
                )
            else:
                phase_indices = self.sample_rsi(self.num_envs)
        else:
            phase_indices = phase_indices.to(device=self.device, dtype=torch.float32)
        self._first_reset = False
        if warmup:
            self.curriculum.update(self.base_env.episode_steps.clone())
        # The official physical randomization is sampled once at construction
        # and remains active across training resets.  Clean benchmark
        # validation temporarily switches to the nominal robot instead.
        self._apply_physical_properties(self._training_physical_properties)
        self._using_nominal_physics = not self.domain_randomization
        self.base_env.reset(phase_indices=phase_indices)
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._zero_reset_root_angular_velocity(env_ids)
        self._reset_buffers(env_ids)
        zero_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        zero_time = torch.zeros(self.num_envs, device=self.device)
        if warmup:
            # Official env.reset performs one zero-action physics step.  Its
            # time action is zero, hence the near-zero reference advance.
            self._base_step(zero_action, torch.full_like(zero_time, 1.0e-6))
            self._training_step_count += 1
        self._last_full_action.zero_()
        self._observe(zero_action, zero_time, update_history=True)
        return self.get_observation()

    def get_observation(self) -> torch.Tensor:
        return self._actor_observation

    def get_critic_observation(self) -> torch.Tensor:
        return self._critic_observation

    def reset_evaluation(self, phase_indices: torch.Tensor) -> torch.Tensor:
        """Reset into the benchmark's clean evaluation protocol.

        AdaMimic's zero-angular-velocity reset, zero-action warm-up and
        per-episode control randomization are training semantics.  The shared
        leaderboard reset must retain the reference root velocity and start
        from the same unperturbed actuator state as every other method.
        """
        phase_indices = phase_indices.to(device=self.device, dtype=torch.float32)
        if phase_indices.shape != (self.num_envs,):
            raise ValueError(
                f"Expected evaluation phases {(self.num_envs,)}, got {tuple(phase_indices.shape)}"
            )

        self._apply_physical_properties(self._nominal_physical_properties)
        self._using_nominal_physics = True
        self.base_env.reset(phase_indices=phase_indices)
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._apply_nominal_actuator_gains(env_ids)
        self.base_env.robot.set_joint_effort_target(
            torch.zeros_like(self._actuation_offset),
            joint_ids=self.base_env.action_joint_ids,
        )

        # Evaluation observations are rebuilt from the clean reset state: four
        # zero history frames followed by the current, noise-free actor frame.
        self._history.zero_()
        self._actor_observation = self._history.flatten(1)
        self._critic_observation.zero_()
        self._last_full_action.zero_()
        self._odometry.zero_()
        self._odometry_noise.zero_()
        self._initial_lidar_orientation_noise.zero_()
        self._initial_lidar_orientation_noise[:, 0] = 1.0
        self._odometry_initialized = False

        zero_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        zero_time = torch.zeros(self.num_envs, device=self.device)
        self._observe(
            zero_action,
            zero_time,
            update_history=True,
            add_actor_noise=False,
        )
        print(
            "[ADAMIMIC_EVAL] clean=1 physics=nominal gains=nominal "
            "effort_offset=0 action_delay=0 wrapper_push=0",
            flush=True,
        )
        return self.get_observation()

    def step_training(
        self,
        control_action: torch.Tensor,
        action_time: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        control_action, action_time = self._validate_action(control_action, action_time)
        previous_phase = self.base_env.phase_steps.clone()
        _, _, _, base_info = self._base_step(control_action, action_time)
        self._training_step_count += 1
        self._apply_official_push()

        reward_low, reward_high, done, done_terms, reward_terms = self._evaluate_transition(
            previous_phase, self.base_env.phase_steps, control_action, action_time
        )
        termination_phase_steps = self.base_env.phase_steps.clone()
        final_actor, final_critic = self._observe(
            control_action, action_time, update_history=False, add_actor_noise=False
        )

        reset_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
        reset_phases = torch.empty(0, dtype=torch.float32, device=self.device)
        if reset_env_ids.numel() > 0:
            self.curriculum.update(self.base_env.episode_steps.index_select(0, reset_env_ids))
            reset_phases = self.sample_rsi(int(reset_env_ids.numel()))
            self.base_env._reset_env_state(reset_env_ids, phase_indices=reset_phases)
            self._zero_reset_root_angular_velocity(reset_env_ids)
            self._reset_buffers(reset_env_ids)

        self._last_full_action.copy_(torch.cat((control_action, action_time[:, None]), dim=-1))
        actor_observation, critic_observation = self._observe(
            control_action, action_time, update_history=True
        )
        info = {
            **base_info,
            "reward_terms": reward_terms,
            "done_terms": done_terms,
            "adamimic_reward_low": reward_low,
            "adamimic_reward_high": reward_high,
            "final_adamimic_observation": final_actor,
            "final_adamimic_critic_observation": final_critic,
            "reset_env_ids": reset_env_ids,
            "reset_phase_indices": reset_phases,
            "termination_phase_steps": termination_phase_steps,
            "adamimic_critic_observation": critic_observation,
        }
        return actor_observation, reward_low, reward_high, done, info

    def step_evaluation(
        self,
        control_action: torch.Tensor,
        action_time: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        control_action, action_time = self._validate_action(control_action, action_time)
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        self._apply_nominal_actuator_gains(env_ids)
        self.base_env.robot.set_joint_effort_target(
            torch.zeros_like(self._actuation_offset),
            joint_ids=self.base_env.action_joint_ids,
        )
        observation_noise = self.base_env.observation_noise
        interval_pushes = self.base_env.interval_pushes
        self.base_env.observation_noise = False
        self.base_env.interval_pushes = False
        try:
            # Do not route validation through AdaMimic's delayed-action,
            # randomized-gain/offset or push path.  Reward and termination are
            # produced directly by the shared clean largebox environment.
            _, reward, done, info = self.base_env.step(
                control_action,
                auto_reset=False,
                reference_dt=action_time,
            )
        finally:
            self.base_env.observation_noise = observation_noise
            self.base_env.interval_pushes = interval_pushes
        self._last_full_action.copy_(torch.cat((control_action, action_time[:, None]), dim=-1))
        actor_observation, critic_observation = self._observe(
            control_action,
            action_time,
            update_history=True,
            add_actor_noise=False,
        )
        info = {
            **info,
            "adamimic_critic_observation": critic_observation,
        }
        return actor_observation, reward, done, info

    def snapshot_runtime_state(self) -> dict[str, Any]:
        return {
            "history": self._history.clone(),
            "actor_observation": self._actor_observation.clone(),
            "critic_observation": self._critic_observation.clone(),
            "last_full_action": self._last_full_action.clone(),
            "odometry": self._odometry.clone(),
            "odometry_noise": self._odometry_noise.clone(),
            "initial_lidar_pos": self._initial_lidar_pos.clone(),
            "initial_lidar_quat": self._initial_lidar_quat.clone(),
            "initial_root_quat": self._initial_root_quat.clone(),
            "initial_lidar_orientation_noise": self._initial_lidar_orientation_noise.clone(),
            "odometry_initialized": self._odometry_initialized,
            "training_step_count": self._training_step_count,
            "first_reset": self._first_reset,
            "delay_buffer": self._delay_buffer.clone(),
            "delay_index": self._delay_index.clone(),
            "actuation_offset": self._actuation_offset.clone(),
            "kp_factor": self._kp_factor.clone(),
            "kd_factor": self._kd_factor.clone(),
            "motor_strength": self._motor_strength.clone(),
            "using_nominal_physics": self._using_nominal_physics,
            "curriculum": {
                name: getattr(self.curriculum, name)
                for name in self.curriculum.__dataclass_fields__
            },
        }

    def restore_runtime_state(self, state: dict[str, Any]) -> None:
        self._history.copy_(state["history"])
        self._actor_observation = state["actor_observation"].clone()
        self._critic_observation.copy_(state["critic_observation"])
        self._last_full_action.copy_(state["last_full_action"])
        self._odometry.copy_(state["odometry"])
        self._odometry_noise.copy_(state["odometry_noise"])
        self._initial_lidar_pos.copy_(state["initial_lidar_pos"])
        self._initial_lidar_quat.copy_(state["initial_lidar_quat"])
        self._initial_root_quat.copy_(state["initial_root_quat"])
        self._initial_lidar_orientation_noise.copy_(state["initial_lidar_orientation_noise"])
        self._odometry_initialized = bool(state["odometry_initialized"])
        self._training_step_count = int(state["training_step_count"])
        self._first_reset = bool(state["first_reset"])
        self._delay_buffer.copy_(state["delay_buffer"])
        self._delay_index.copy_(state["delay_index"])
        self._actuation_offset.copy_(state["actuation_offset"])
        self._kp_factor.copy_(state["kp_factor"])
        self._kd_factor.copy_(state["kd_factor"])
        self._motor_strength.copy_(state["motor_strength"])
        self._apply_actuator_gains(
            torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        )
        self.base_env.robot.set_joint_effort_target(
            self._actuation_offset,
            joint_ids=self.base_env.action_joint_ids,
        )
        if bool(state.get("using_nominal_physics", False)):
            self._apply_physical_properties(self._nominal_physical_properties)
            self._using_nominal_physics = True
        else:
            self._apply_physical_properties(self._training_physical_properties)
            self._using_nominal_physics = not self.domain_randomization
        for name, value in state["curriculum"].items():
            setattr(self.curriculum, name, value)

    def state_dict(self) -> dict[str, Any]:
        """Checkpoint the persistent curricula and global perturbation clock."""
        return {
            "training_step_count": self._training_step_count,
            "first_reset": self._first_reset,
            "curriculum": {
                name: getattr(self.curriculum, name)
                for name in self.curriculum.__dataclass_fields__
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._training_step_count = int(state["training_step_count"])
        self._first_reset = bool(state.get("first_reset", False))
        for name, value in state["curriculum"].items():
            setattr(self.curriculum, name, value)

    def _validate_action(
        self, control_action: torch.Tensor, action_time: torch.Tensor | float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if control_action.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"Expected control action {(self.num_envs, self.action_dim)}, got {tuple(control_action.shape)}"
            )
        return control_action, _as_vector(action_time, self.num_envs, self.device)

    def _base_step(
        self, control_action: torch.Tensor, action_time: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        physics_substep_actions = self._prepare_control_substep_actions(control_action)
        self.base_env.robot.set_joint_effort_target(
            self._actuation_offset,
            joint_ids=self.base_env.action_joint_ids,
        )
        observation_noise = self.base_env.observation_noise
        interval_pushes = self.base_env.interval_pushes
        self.base_env.observation_noise = False
        self.base_env.interval_pushes = False
        try:
            return self.base_env.step(
                control_action,
                auto_reset=False,
                reference_dt=action_time,
                physics_substep_actions=physics_substep_actions,
            )
        finally:
            self.base_env.observation_noise = observation_noise
            self.base_env.interval_pushes = interval_pushes

    def _reset_buffers(self, env_ids: torch.Tensor) -> None:
        self._history[env_ids] = 0.0
        self._actor_observation = self._history.flatten(1)
        self._critic_observation[env_ids] = 0.0
        self._last_full_action[env_ids] = 0.0
        self._odometry[env_ids] = 0.0
        self._odometry_noise[env_ids].uniform_(-0.01, 0.01)
        self._resample_control_randomization(env_ids)

    def _zero_reset_root_angular_velocity(self, env_ids: torch.Tensor) -> None:
        velocity = self.base_env.get_mimic_root_velocity_w().index_select(0, env_ids).clone()
        velocity[:, 3:] = 0.0
        if self.base_env.config.root_velocity_mode == "link":
            self.base_env.robot.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
        else:
            self.base_env.robot.write_root_velocity_to_sim(velocity, env_ids=env_ids)

    def _collect_actuator_groups(self) -> list[tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]]:
        action_ids = self.base_env.action_joint_ids.detach().cpu().tolist()
        action_index = {int(joint_id): idx for idx, joint_id in enumerate(action_ids)}
        groups: list[tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for actuator in self.base_env.robot.actuators.values():
            joint_ids = actuator.joint_indices
            if isinstance(joint_ids, slice):
                joint_ids = torch.arange(
                    self.base_env.robot.num_joints, device=self.device
                )[joint_ids]
            joint_ids = torch.as_tensor(joint_ids, dtype=torch.long, device=self.device)
            positions = [action_index[int(j)] for j in joint_ids.detach().cpu().tolist() if int(j) in action_index]
            if len(positions) != int(joint_ids.numel()):
                raise ValueError("AdaMimic actuator contains a joint outside the 29-DoF action set")
            groups.append(
                (
                    actuator,
                    torch.tensor(positions, dtype=torch.long, device=self.device),
                    actuator.stiffness.clone(),
                    actuator.damping.clone(),
                )
            )
        return groups

    def _collect_nominal_stiffness(self) -> torch.Tensor:
        result = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        for _, positions, stiffness, _ in self._actuator_groups:
            result[:, positions] = stiffness
        if bool((result <= 0.0).any()):
            raise ValueError("AdaMimic requires positive nominal PD stiffness for every action joint")
        return result

    def _resample_control_randomization(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._delay_buffer[:, env_ids] = 0.0
        if not self.domain_randomization:
            self._delay_index[env_ids] = 4
            self._actuation_offset[env_ids] = 0.0
            self._kp_factor[env_ids] = 1.0
            self._kd_factor[env_ids] = 1.0
            self._motor_strength[env_ids] = 1.0
            self._apply_actuator_gains(env_ids)
            return
        count = int(env_ids.numel())
        shape = (count, self.action_dim)
        self._delay_index[env_ids] = torch.randint(0, 5, (count,), device=self.device)
        self._kp_factor[env_ids] = 0.85 + 0.30 * torch.rand(shape, device=self.device)
        self._kd_factor[env_ids] = 0.85 + 0.30 * torch.rand(shape, device=self.device)
        self._motor_strength[env_ids] = 0.90 + 0.20 * torch.rand(shape, device=self.device)
        effort_limits = self.base_env.robot.data.joint_effort_limits.index_select(
            1, self.base_env.action_joint_ids
        ).index_select(0, env_ids)
        self._actuation_offset[env_ids] = (
            -0.03 + 0.06 * torch.rand(shape, device=self.device)
        ) * effort_limits
        self._apply_actuator_gains(env_ids)

    def _apply_actuator_gains(self, env_ids: torch.Tensor) -> None:
        for actuator, positions, nominal_stiffness, nominal_damping in self._actuator_groups:
            motor = self._motor_strength.index_select(0, env_ids).index_select(1, positions)
            kp = self._kp_factor.index_select(0, env_ids).index_select(1, positions)
            kd = self._kd_factor.index_select(0, env_ids).index_select(1, positions)
            actuator.stiffness[env_ids] = nominal_stiffness.index_select(0, env_ids) * kp * motor
            actuator.damping[env_ids] = nominal_damping.index_select(0, env_ids) * kd * motor

    def _apply_nominal_actuator_gains(self, env_ids: torch.Tensor) -> None:
        for actuator, _, nominal_stiffness, nominal_damping in self._actuator_groups:
            actuator.stiffness[env_ids] = nominal_stiffness.index_select(0, env_ids)
            actuator.damping[env_ids] = nominal_damping.index_select(0, env_ids)

    def _prepare_control_substep_actions(self, control_action: torch.Tensor) -> torch.Tensor:
        """Reproduce AdaMimic's delay buffer at the 5 ms physics rate."""
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        substep_actions = []
        for _ in range(int(self.base_env.decimation)):
            self._delay_buffer[:-1].copy_(self._delay_buffer[1:].clone())
            self._delay_buffer[-1].copy_(control_action)
            substep_actions.append(self._delay_buffer[self._delay_index, env_ids].clone())
        return torch.stack(substep_actions, dim=0)

    def _capture_nominal_physical_properties(self) -> dict[str, torch.Tensor]:
        robot = self.base_env.robot
        return {
            "masses": robot.data.default_mass.detach().cpu().clone(),
            "inertias": robot.data.default_inertia.detach().cpu().clone(),
            "coms": robot.root_physx_view.get_coms().clone(),
            "materials": robot.root_physx_view.get_material_properties().clone(),
        }

    def _apply_physical_properties(self, properties: dict[str, torch.Tensor]) -> None:
        view = self.base_env.robot.root_physx_view
        env_ids = torch.arange(self.num_envs, dtype=torch.long, device="cpu")
        view.set_masses(properties["masses"], env_ids)
        view.set_inertias(properties["inertias"], env_ids)
        view.set_coms(properties["coms"], env_ids)
        view.set_material_properties(properties["materials"], env_ids)

    def _randomize_physics_once(self) -> dict[str, torch.Tensor]:
        """Apply the active one-time rigid-body/material randomization in train.yaml."""
        robot = self.base_env.robot
        root_body_id = robot.body_names.index("pelvis")

        default_mass = self._nominal_physical_properties["masses"]
        masses = default_mass.clone()
        masses[:, root_body_id] += -2.0 + 7.0 * torch.rand(self.num_envs)
        link_ids = [idx for idx in range(robot.num_bodies) if idx != root_body_id]
        if link_ids:
            link_ids_cpu = torch.tensor(link_ids, dtype=torch.long)
            masses[:, link_ids_cpu] *= 0.9 + 0.2 * torch.rand(self.num_envs, len(link_ids))
        if bool((masses <= 0.0).any()):
            raise ValueError("AdaMimic payload randomization produced a non-positive body mass")
        inertias = self._nominal_physical_properties["inertias"].clone()
        inertias *= (masses / default_mass).unsqueeze(-1)

        coms = self._nominal_physical_properties["coms"].clone()
        coms[:, root_body_id, :3] += -0.1 + 0.2 * torch.rand(self.num_envs, 3)

        materials = self._nominal_physical_properties["materials"].clone()
        friction = 0.1 + torch.rand(self.num_envs, 1)
        restitution = 0.1 * torch.rand(self.num_envs, 1)
        materials[..., 0] = friction
        materials[..., 1] = friction
        materials[..., 2] = restitution
        return {
            "masses": masses,
            "inertias": inertias,
            "coms": coms,
            "materials": materials,
        }

    def _initialize_odometry(self) -> None:
        if self._odometry_initialized:
            return
        lidar_pos, lidar_quat = self._lidar_pose()
        self._initial_lidar_pos.copy_(lidar_pos)
        self._initial_lidar_quat.copy_(
            _quat_mul(
                lidar_quat,
                self._initial_lidar_orientation_noise,
            )
        )
        self._initial_root_quat.copy_(self.base_env.robot.data.root_quat_w)
        self._odometry_initialized = True

    def _lidar_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        robot = self.base_env.robot
        if self.lidar_body_id is not None:
            return (
                robot.data.body_pos_w[:, self.lidar_body_id],
                robot.data.body_quat_w[:, self.lidar_body_id],
            )
        parent_pos = robot.data.body_pos_w[:, self.lidar_parent_body_id]
        parent_quat = robot.data.body_quat_w[:, self.lidar_parent_body_id]
        offset = self._lidar_parent_pos.expand(self.num_envs, -1)
        fixed_quat = self._lidar_parent_quat.expand(self.num_envs, -1)
        return (
            parent_pos + _quat_rotate(parent_quat, offset),
            _quat_mul(parent_quat, fixed_quat),
        )

    def _observe(
        self,
        control_action: torch.Tensor,
        action_time: torch.Tensor,
        *,
        update_history: bool,
        add_actor_noise: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._initialize_odometry()
        robot = self.base_env.robot
        joint_pos, joint_vel = self.base_env.get_action_joint_state()
        full_action = torch.cat((control_action, action_time[:, None]), dim=-1)
        gravity = robot.data.GRAVITY_VEC_W
        if gravity.ndim == 1:
            gravity = gravity.expand(self.num_envs, 3)
        projected_gravity = _quat_rotate_inverse(robot.data.root_quat_w, gravity)

        update_odometry = (self.base_env.episode_steps % 5) == 0
        if bool(update_odometry.any()):
            lidar_pos, _ = self._lidar_pose()
            lidar_delta = (
                lidar_pos
                - self._initial_lidar_pos
                + self._odometry_noise
            )
            value = _quat_rotate_inverse(self._initial_lidar_quat, lidar_delta)
            self._odometry[update_odometry] = value[update_odometry]

        phase = self.base_env.phase_steps.float()
        norm_time = (phase / max(1, self.base_env.motion.num_frames - 1)).clamp(0.0, 1.0)
        zeros = torch.zeros_like(norm_time)
        one_step_clean = torch.cat(
            (
                robot.data.root_ang_vel_b * 0.25,
                projected_gravity,
                joint_pos,
                joint_vel * 0.05,
                full_action,
                norm_time[:, None],
                zeros[:, None],
                zeros[:, None],
                self._odometry,
            ),
            dim=-1,
        )
        if one_step_clean.shape[-1] != ADAMIMIC_ONE_STEP_OBSERVATION_DIM:
            raise RuntimeError(
                f"Expected AdaMimic one-step observation {ADAMIMIC_ONE_STEP_OBSERVATION_DIM}, "
                f"got {one_step_clean.shape[-1]}"
            )

        one_step_actor = one_step_clean.clone()
        if self.base_env.observation_noise and add_actor_noise:
            noise = torch.zeros_like(one_step_actor)
            noise[:, 0:3].uniform_(-0.05, 0.05)
            noise[:, 3:6].uniform_(-0.05, 0.05)
            noise[:, 6:35].uniform_(-0.01, 0.01)
            noise[:, 35:64].uniform_(-0.075, 0.075)
            noise[:, 64:93].uniform_(-0.1, 0.1)
            one_step_actor += noise

        if update_history:
            self._history[:, :-1].copy_(self._history[:, 1:].clone())
            self._history[:, -1].copy_(one_step_actor)
            actor = self._history.flatten(1).clamp(-100.0, 100.0)
            self._actor_observation = actor
        else:
            actor = torch.cat((self._history[:, 1:], one_step_actor[:, None]), dim=1).flatten(1)
            actor = actor.clamp(-100.0, 100.0)

        imu_quat = _quat_mul(_quat_conjugate(self._initial_root_quat), robot.data.root_quat_w)
        reference_joint_pos = self.base_env.motion.get_frame(phase)["joint_pos"]
        critic = torch.cat(
            (
                one_step_clean,
                imu_quat,
                robot.data.root_lin_vel_b * 2.0,
                reference_joint_pos,
            ),
            dim=-1,
        ).clamp(-100.0, 100.0)
        if critic.shape[-1] != ADAMIMIC_CRITIC_OBSERVATION_DIM:
            raise RuntimeError(
                f"Expected AdaMimic critic observation {ADAMIMIC_CRITIC_OBSERVATION_DIM}, got {critic.shape[-1]}"
            )
        if update_history:
            self._critic_observation.copy_(critic)
        return actor, critic

    def _reference_state(self, phase: torch.Tensor) -> dict[str, torch.Tensor]:
        full = self.base_env.motion.get_full_body_state(phase)
        frame = self.base_env.motion.get_frame(phase)
        origins = self.base_env.scene.env_origins
        return {
            "body_pos": full["body_pos_w"].index_select(1, self.body_ids) + origins[:, None, :],
            "body_quat": full["body_quat_w"].index_select(1, self.body_ids),
            "root_pos": frame["root_pos_w"] + origins,
            "root_quat": frame["root_quat_w"],
            "joint_pos": frame["joint_pos"],
        }

    def _keyframe_state(
        self, previous_phase: torch.Tensor, current_phase: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        crossed = (previous_phase[:, None] < self.keyframe_phases) & (
            current_phase[:, None] >= self.keyframe_phases
        )
        transition = crossed.any(dim=-1)
        stage = torch.searchsorted(self.keyframe_phases, current_phase.contiguous(), right=True) - 1
        special = torch.zeros_like(transition)
        if self.special_keyframe_indices.numel() > 0:
            special = (stage[:, None] == self.special_keyframe_indices[None]).any(dim=-1)
        return transition, stage, special

    def _evaluate_transition(
        self,
        previous_phase: torch.Tensor,
        current_phase: torch.Tensor,
        control_action: torch.Tensor,
        action_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        reference = self._reference_state(current_phase)
        robot = self.base_env.robot
        robot_body_pos = robot.data.body_pos_w.index_select(1, self.body_ids)
        robot_body_quat = robot.data.body_quat_w.index_select(1, self.body_ids)
        robot_joint_pos, robot_joint_vel = self.base_env.get_action_joint_state()
        global_pos_diff = robot_body_pos - reference["body_pos"]

        root_pos = robot.data.root_pos_w
        root_quat = robot.data.root_quat_w
        robot_local_pos = _quat_rotate_inverse(
            root_quat[:, None].expand(-1, len(ADAMIMIC_BODY_NAMES), -1),
            robot_body_pos - root_pos[:, None],
        )
        reference_local_pos = _quat_rotate_inverse(
            reference["root_quat"][:, None].expand(-1, len(ADAMIMIC_BODY_NAMES), -1),
            reference["body_pos"] - reference["root_pos"][:, None],
        )
        local_pos_diff = robot_local_pos - reference_local_pos
        robot_local_quat = _quat_mul(
            _quat_conjugate(root_quat)[:, None], robot_body_quat
        )
        reference_local_quat = _quat_mul(
            _quat_conjugate(reference["root_quat"])[:, None], reference["body_quat"]
        )

        transition, stage, special = self._keyframe_state(previous_phase, current_phase)
        threshold = float(self.curriculum.termination_threshold)
        body_deviation = torch.linalg.vector_norm(global_pos_diff, dim=-1)
        keyframe_body_bad = (body_deviation > threshold).any(dim=-1)
        feet_z_bad = (
            global_pos_diff.index_select(1, self.feet_body_indices)[..., 2].abs()
            > threshold / 3.0
        ).any(dim=-1)
        keyframe_bad = transition & (keyframe_body_bad | feet_z_bad)

        gravity = robot.data.GRAVITY_VEC_W
        if gravity.ndim == 1:
            gravity = gravity.expand(self.num_envs, 3)
        projected_gravity = _quat_rotate_inverse(root_quat, gravity)
        root_rot_bad = (projected_gravity[:, :2].abs() > 0.8).any(dim=-1)
        root_height_bad = root_pos[:, 2] < 0.4
        motion_complete = torch.ceil(current_phase) >= float(
            self.base_env.motion.num_frames - 1
        )
        failure = keyframe_bad | root_rot_bad | root_height_bad
        done = motion_complete | failure
        penalized_failure = failure & ~motion_complete

        time_ratio = action_time / float(self.base_env.dt)
        if not self.apply_reward_scale:
            time_ratio = torch.ones_like(time_ratio)
        dt = float(self.base_env.dt)

        upper_pos = global_pos_diff.index_select(1, self.upper_body_indices).square().mean((1, 2))
        lower_pos = global_pos_diff.index_select(1, self.lower_body_indices).square().mean((1, 2))
        sparse_body_pos = 0.5 * torch.exp(-upper_pos / 0.03) + 0.5 * torch.exp(-lower_pos / 0.1)
        global_rot_angle = _quat_angle(reference["body_quat"], robot_body_quat)
        sparse_body_rot = torch.exp(-global_rot_angle.square().mean(dim=-1) / 1.0)
        feet_pos = global_pos_diff.index_select(1, self.feet_body_indices).square().mean((1, 2))
        sparse_feet = torch.exp(-feet_pos / 0.03)
        sparse_mask = transition if self.sparse_global else torch.ones_like(transition)
        # The official source's special-scale temporary is returned only by
        # the feet term; body-position and body-rotation return the unscaled
        # tensor.  Preserve that source behavior exactly.
        feet_special_scale = torch.where(
            self.special_scale & special,
            torch.full_like(sparse_feet, self.special_scale_size),
            torch.ones_like(sparse_feet),
        )

        local_upper = local_pos_diff.index_select(1, self.upper_body_indices).square().mean((1, 2))
        local_lower = local_pos_diff.index_select(1, self.lower_body_indices).square().mean((1, 2))
        dense_body_pos = 0.5 * torch.exp(-local_upper / 0.03) + 0.5 * torch.exp(-local_lower / 0.1)
        local_rot_angle = _quat_angle(reference_local_quat, robot_local_quat)
        dense_body_rot = torch.exp(-local_rot_angle.square().mean(dim=-1) / 1.0)
        joint_error = (reference["joint_pos"] - robot_joint_pos).square().mean(dim=-1)
        dense_joint = torch.exp(-joint_error / 1.0)
        if self.sparse_local:
            dense_gate = transition.to(dtype=dense_joint.dtype)
            dense_body_pos *= dense_gate
            dense_body_rot *= dense_gate
            dense_joint *= dense_gate

        joint_ids = self.base_env.action_joint_ids
        hard_pos_limits = robot.data.joint_pos_limits.index_select(1, joint_ids)
        # Official source computes the position curriculum limits but then
        # reads its fixed 0.98 limits.  Reproduce the committed implementation.
        soft_pos_limits = hard_pos_limits * 0.98
        pos_limit = (
            -(robot_joint_pos - soft_pos_limits[..., 0]).clamp(max=0.0)
            + (robot_joint_pos - soft_pos_limits[..., 1]).clamp(min=0.0)
        ).sum(dim=-1)
        vel_limits = robot.data.joint_vel_limits.index_select(1, joint_ids)
        vel_limit = (
            robot_joint_vel.abs() - vel_limits * float(self.curriculum.soft_vel)
        ).clamp_min(0.0).sum(dim=-1)
        torques = robot.data.applied_torque.index_select(1, joint_ids)
        torque_limits = robot.data.joint_effort_limits.index_select(1, joint_ids)
        torque_limit = (
            torques.abs() - torque_limits * float(self.curriculum.soft_torque)
        ).clamp_min(0.0).sum(dim=-1)
        gains = self._nominal_stiffness
        torque_cost = (torques / gains).abs().sum(dim=-1)
        full_action = torch.cat((control_action, action_time[:, None]), dim=-1)
        action_rate = (full_action - self._last_full_action).abs().sum(dim=-1)
        penalty_scale = float(self.curriculum.penalty_scale)

        dense = (
            0.75 * dense_joint
            + 0.75 * dense_body_pos
            + 0.50 * dense_body_rot
            + penalty_scale
            * (-10.0 * pos_limit - 5.0 * vel_limit - 5.0 * torque_limit - 2.0e-7 * torque_cost - 0.5 * action_rate)
        ) * dt * time_ratio
        termination_penalty = -200.0 * dt * penalized_failure.to(dtype=dense.dtype)
        sparse = (
            10.0 * sparse_body_pos * sparse_mask
            + 5.0 * sparse_body_rot * sparse_mask
            + 10.0 * sparse_feet * feet_special_scale * sparse_mask
        ) * dt + termination_penalty
        reward_low = torch.stack((dense, sparse), dim=-1)

        high_dense = -torch.linalg.vector_norm(local_pos_diff, dim=-1).mean(dim=-1) * time_ratio
        high_sparse = -torch.linalg.vector_norm(global_pos_diff, dim=-1).mean(dim=-1) * transition
        high_dense += termination_penalty
        high_sparse += termination_penalty
        reward_high = torch.stack((high_dense, high_sparse), dim=-1)

        zeros = torch.zeros_like(done)
        done_terms = {
            "time_out": motion_complete,
            "motion_complete": motion_complete,
            "anchor_pos_bad": root_height_bad,
            "anchor_ori_bad": root_rot_bad,
            "ee_body_bad": keyframe_bad,
            "fall_contact": zeros,
            "keyframe_bad": keyframe_bad,
            "keyframe_transition": transition,
        }
        reward_terms = {
            "dense_tracking_dof_pos": dense_joint,
            "dense_tracking_body_position_local": dense_body_pos,
            "dense_tracking_body_rot_local": dense_body_rot,
            "dense_dof_pos_limits": pos_limit,
            "dense_dof_vel_limits": vel_limit,
            "dense_torque_limits": torque_limit,
            "dense_torques": torque_cost,
            "dense_action_rate": action_rate,
            "sparse_tracking_body_position": sparse_body_pos * sparse_mask,
            "sparse_tracking_body_rot": sparse_body_rot * sparse_mask,
            "sparse_tracking_body_position_feet": sparse_feet * feet_special_scale * sparse_mask,
            "sparse_termination": penalized_failure.to(dtype=dense.dtype),
            "high_local_body_error": torch.linalg.vector_norm(local_pos_diff, dim=-1).mean(dim=-1),
            "high_global_body_error": torch.linalg.vector_norm(global_pos_diff, dim=-1).mean(dim=-1),
            "keyframe_stage": stage.to(dtype=dense.dtype),
            "termination_threshold": torch.full_like(dense, threshold),
            "penalty_scale": torch.full_like(dense, penalty_scale),
            "reference_dt": action_time,
            "reference_frame_delta": action_time * float(self.base_env.motion.fps),
        }
        return reward_low, reward_high, done, done_terms, reward_terms

    def _apply_official_push(self) -> None:
        if not self.push_robots or self._training_step_count % self.push_interval_steps != 0:
            return
        root_velocity = self.base_env.get_mimic_root_velocity_w().clone()
        root_velocity[:, :2].uniform_(-self.max_push_vel_xy, self.max_push_vel_xy)
        if self.base_env.config.root_velocity_mode == "link":
            self.base_env.robot.write_root_link_velocity_to_sim(root_velocity)
        else:
            self.base_env.robot.write_root_velocity_to_sim(root_velocity)
