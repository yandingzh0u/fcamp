from __future__ import annotations

import torch

from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_inv,
    quat_mul,
    subtract_frame_transforms,
    yaw_quat,
)

from .config import CRITIC_OBS_DIM, FUTURE_REF_FRAME_DIM, OBS_DIM


class MimicObservationMixin:
    def get_reference_state(self) -> dict[str, torch.Tensor]:
        reference = dict(self.motion.get_frame(self.phase_steps))
        env_origins = self.scene.env_origins
        reference["body_pos_w"] = reference["body_pos_w"] + env_origins[:, None, :]
        reference["anchor_pos_w"] = reference["anchor_pos_w"] + env_origins
        reference["root_pos_w"] = reference["root_pos_w"] + env_origins
        return reference

    def _compute_relative_reference_bodies(
        self,
        reference: dict[str, torch.Tensor],
        robot_anchor_pos_w: torch.Tensor,
        robot_anchor_quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_bodies = len(self.track_body_names)
        anchor_pos_w_repeat = reference["anchor_pos_w"][:, None, :].repeat(1, num_bodies, 1)
        anchor_quat_w_repeat = reference["anchor_quat_w"][:, None, :].repeat(1, num_bodies, 1)
        robot_anchor_pos_w_repeat = robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1)
        robot_anchor_quat_w_repeat = robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1)

        delta_pos_w = robot_anchor_pos_w_repeat.clone()
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        body_quat_relative_w = quat_mul(delta_ori_w, reference["body_quat_w"])
        body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, reference["body_pos_w"] - anchor_pos_w_repeat)
        return body_pos_relative_w, body_quat_relative_w

    def get_tracking_context(self) -> dict[str, torch.Tensor]:
        reference = self.get_reference_state()
        robot_joint_pos, robot_joint_vel = self.get_action_joint_state()
        robot_body_pos_w = self.robot.data.body_pos_w[:, self.track_body_ids]
        robot_body_quat_w = self.robot.data.body_quat_w[:, self.track_body_ids]
        robot_body_lin_vel_w = self.robot.data.body_lin_vel_w[:, self.track_body_ids]
        robot_body_ang_vel_w = self.robot.data.body_ang_vel_w[:, self.track_body_ids]
        robot_anchor_pos_w = self.robot.data.body_pos_w[:, self.anchor_body_id]
        robot_anchor_quat_w = self.robot.data.body_quat_w[:, self.anchor_body_id]

        body_pos_relative_w, body_quat_relative_w = self._compute_relative_reference_bodies(
            reference,
            robot_anchor_pos_w,
            robot_anchor_quat_w,
        )
        return {
            "reference": reference,
            "robot_joint_pos": robot_joint_pos,
            "robot_joint_vel": robot_joint_vel,
            "robot_body_pos_w": robot_body_pos_w,
            "robot_body_quat_w": robot_body_quat_w,
            "robot_body_lin_vel_w": robot_body_lin_vel_w,
            "robot_body_ang_vel_w": robot_body_ang_vel_w,
            "robot_anchor_pos_w": robot_anchor_pos_w,
            "robot_anchor_quat_w": robot_anchor_quat_w,
            "body_pos_relative_w": body_pos_relative_w,
            "body_quat_relative_w": body_quat_relative_w,
        }

    def _motion_anchor_observation_terms(
        self,
        robot_anchor_pos_w: torch.Tensor,
        robot_anchor_quat_w: torch.Tensor,
        reference: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        motion_anchor_pos_b, motion_anchor_ori = subtract_frame_transforms(
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            reference["anchor_pos_w"],
            reference["anchor_quat_w"],
        )
        motion_anchor_ori_b = matrix_from_quat(motion_anchor_ori)[..., :2].reshape(self.num_envs, -1)
        return motion_anchor_pos_b, motion_anchor_ori_b

    def get_observation(self) -> torch.Tensor:
        return self.build_observation()

    def get_critic_observation(self) -> torch.Tensor:
        return self.build_critic_observation()

    def build_observation(self) -> torch.Tensor:
        context = self.get_tracking_context()
        reference = context["reference"]
        reference_joint_state = torch.cat([reference["joint_pos"], reference["joint_vel"]], dim=-1)
        motion_anchor_pos_b, motion_anchor_ori_b = self._motion_anchor_observation_terms(
            context["robot_anchor_pos_w"],
            context["robot_anchor_quat_w"],
            reference,
        )
        anchor_z_err = (reference["anchor_pos_w"][:, 2] - context["robot_anchor_pos_w"][:, 2]).unsqueeze(-1)
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        foot_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self.foot_contact_body_ids], dim=-1), dim=1)[0]
            > 1.0
        ).to(dtype=motion_anchor_ori_b.dtype)
        joint_pos_rel = context["robot_joint_pos"] - self.default_action_joint_pos
        joint_vel_rel = context["robot_joint_vel"] - self.default_action_joint_vel
        motion_anchor_pos_b = self._add_uniform_noise(motion_anchor_pos_b, -0.02, 0.02)
        motion_anchor_ori_b = self._add_uniform_noise(motion_anchor_ori_b, -0.05, 0.05)
        anchor_z_err = self._add_uniform_noise(anchor_z_err, -0.01, 0.01)
        root_lin_vel_b = self._add_uniform_noise(self.robot.data.root_lin_vel_b, -0.1, 0.1)
        base_ang_vel = self._add_uniform_noise(self.robot.data.root_ang_vel_b, -0.2, 0.2)
        joint_pos_rel = self._add_uniform_noise(joint_pos_rel, -0.01, 0.01)
        joint_vel_rel = self._add_uniform_noise(joint_vel_rel, -0.5, 0.5)
        observation = torch.cat(
            [
                reference_joint_state,
                motion_anchor_pos_b,
                motion_anchor_ori_b,
                anchor_z_err,
                root_lin_vel_b,
                foot_contact,
                base_ang_vel,
                joint_pos_rel,
                joint_vel_rel,
                self.last_action,
            ],
            dim=-1,
        )
        future_ref_steps = int(self.task_cfg.future_ref_steps)
        if future_ref_steps > 0:
            future_terms = []
            for k in range(1, future_ref_steps + 1):
                future_phase = self.motion.clamp_time_steps(self.phase_steps + k)
                future_frame = self.motion.get_frame(future_phase)
                future_joint = torch.cat([future_frame["joint_pos"], future_frame["joint_vel"]], dim=-1)
                future_terms.append(future_joint)
            observation = torch.cat([observation, *future_terms], dim=-1)
        expected_dim = OBS_DIM + future_ref_steps * FUTURE_REF_FRAME_DIM
        if observation.shape[-1] != expected_dim:
            raise RuntimeError(f"Expected observation dim {expected_dim}, got {observation.shape[-1]}")
        return observation

    def build_critic_observation(self) -> torch.Tensor:
        context = self.get_tracking_context()
        reference = context["reference"]
        reference_joint_state = torch.cat([reference["joint_pos"], reference["joint_vel"]], dim=-1)
        motion_anchor_pos_b, motion_anchor_ori_b = self._motion_anchor_observation_terms(
            context["robot_anchor_pos_w"],
            context["robot_anchor_quat_w"],
            reference,
        )
        num_bodies = len(self.track_body_names)
        robot_anchor_pos_repeat = context["robot_anchor_pos_w"][:, None, :].repeat(1, num_bodies, 1)
        robot_anchor_quat_repeat = context["robot_anchor_quat_w"][:, None, :].repeat(1, num_bodies, 1)
        robot_body_pos_b, robot_body_ori_b = subtract_frame_transforms(
            robot_anchor_pos_repeat,
            robot_anchor_quat_repeat,
            context["robot_body_pos_w"],
            context["robot_body_quat_w"],
        )
        robot_body_ori_b = matrix_from_quat(robot_body_ori_b)[..., :2].reshape(self.num_envs, -1)
        joint_pos_rel = context["robot_joint_pos"] - self.default_action_joint_pos
        joint_vel_rel = context["robot_joint_vel"] - self.default_action_joint_vel
        observation = torch.cat(
            [
                reference_joint_state,
                motion_anchor_pos_b,
                motion_anchor_ori_b,
                robot_body_pos_b.reshape(self.num_envs, -1),
                robot_body_ori_b,
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                joint_pos_rel,
                joint_vel_rel,
                self.last_action,
            ],
            dim=-1,
        )
        if observation.shape[-1] != CRITIC_OBS_DIM:
            raise RuntimeError(f"Expected critic observation dim {CRITIC_OBS_DIM}, got {observation.shape[-1]}")
        return observation

    def _add_uniform_noise(self, value: torch.Tensor, n_min: float, n_max: float) -> torch.Tensor:
        return value + torch.empty_like(value).uniform_(n_min, n_max)
