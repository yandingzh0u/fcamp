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

from .spec import CRITIC_OBS_DIM, OBS_DIM
from .contracts import require_finite_tensors


class MimicObservationMixin:
    def get_foot_contact_mask(self) -> torch.Tensor:
        """Return the two measured foot-contact flags in left/right order."""

        foot_body_ids = self.foot_contact_body_ids
        if not torch.is_tensor(foot_body_ids) or int(foot_body_ids.numel()) != 2:
            raise RuntimeError(
                "Contact-mode diagnostics require exactly two ordered foot body ids"
            )
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        require_finite_tensors(
            {"net_forces_w_history": net_contact_forces},
            context="Contact sensor",
        )
        if net_contact_forces.ndim != 4 or net_contact_forces.shape[0] != self.num_envs:
            raise RuntimeError(
                "Contact force history must have shape [num_envs, history, bodies, 3], "
                f"got {tuple(net_contact_forces.shape)}"
            )
        if net_contact_forces.shape[-1] != 3:
            raise RuntimeError(
                "Contact force vectors must have three components, "
                f"got {net_contact_forces.shape[-1]}"
            )
        return (
            torch.linalg.vector_norm(
                net_contact_forces[:, :, foot_body_ids],
                dim=-1,
            ).amax(dim=1)
            > 1.0
        )

    def get_contact_mode_masks(self) -> dict[str, torch.Tensor]:
        """Classify measured support for diagnostics without policy gating."""

        foot_contact = self.get_foot_contact_mask()
        left = foot_contact[:, 0]
        right = foot_contact[:, 1]
        return {
            "flight": ~left & ~right,
            "left_only": left & ~right,
            "right_only": ~left & right,
            "double_support": left & right,
        }

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
        robot_anchor_pos_w = self.robot.data.body_pos_w[:, self.anchor_body_id]
        robot_anchor_quat_w = self.robot.data.body_quat_w[:, self.anchor_body_id]

        body_pos_relative_w, body_quat_relative_w = self._compute_relative_reference_bodies(
            reference,
            robot_anchor_pos_w,
            robot_anchor_quat_w,
        )
        context = {
            "reference": reference,
            "robot_joint_pos": robot_joint_pos,
            "robot_joint_vel": robot_joint_vel,
            "robot_body_pos_w": robot_body_pos_w,
            "robot_body_quat_w": robot_body_quat_w,
            "robot_anchor_pos_w": robot_anchor_pos_w,
            "robot_anchor_quat_w": robot_anchor_quat_w,
            "body_pos_relative_w": body_pos_relative_w,
            "body_quat_relative_w": body_quat_relative_w,
        }
        require_finite_tensors(
            {
                **{
                    f"reference.{name}": value
                    for name, value in reference.items()
                },
                **{
                    name: value
                    for name, value in context.items()
                    if name != "reference"
                },
            },
            context="Tracking state",
        )
        return context

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
        motion_anchor_ori_b = matrix_from_quat(motion_anchor_ori)[..., :2].reshape(
            robot_anchor_pos_w.shape[0], -1
        )
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


        termination_z_err = (
            context["body_pos_relative_w"][:, self.termination_body_indices, 2]
            - context["robot_body_pos_w"][:, self.termination_body_indices, 2]
        )
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        foot_contact = self.get_foot_contact_mask().to(
            dtype=motion_anchor_ori_b.dtype
        )


        termination_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self.termination_contact_body_ids], dim=-1), dim=1)[0]
            > 1.0
        ).to(dtype=motion_anchor_ori_b.dtype)
        joint_pos_rel = context["robot_joint_pos"] - self.default_action_joint_pos
        joint_vel_rel = context["robot_joint_vel"] - self.default_action_joint_vel
        motion_anchor_pos_b = self._add_uniform_noise(motion_anchor_pos_b, -0.02, 0.02)
        motion_anchor_ori_b = self._add_uniform_noise(motion_anchor_ori_b, -0.05, 0.05)
        anchor_z_err = self._add_uniform_noise(anchor_z_err, -0.01, 0.01)
        termination_z_err = self._add_uniform_noise(termination_z_err, -0.01, 0.01)
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
                termination_z_err,
                root_lin_vel_b,
                foot_contact,
                termination_contact,
                base_ang_vel,
                joint_pos_rel,
                joint_vel_rel,
                self.last_action,
            ],
            dim=-1,
        )
        if observation.shape[-1] != OBS_DIM:
            raise RuntimeError(f"Expected observation dim {OBS_DIM}, got {observation.shape[-1]}")
        require_finite_tensors(
            {"observation": observation},
            context="Policy observation",
        )
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
        require_finite_tensors(
            {"observation": observation},
            context="Critic observation",
        )
        return observation

    def _add_uniform_noise(self, value: torch.Tensor, n_min: float, n_max: float) -> torch.Tensor:
        if not self.observation_noise:
            return value
        return value + torch.empty_like(value).uniform_(n_min, n_max)
