from __future__ import annotations

import torch

from isaaclab.utils.math import quat_error_magnitude

from .config import UNDESIRED_CONTACT_THRESHOLD


class MimicRewardMixin:
    def compute_reward(
        self,
        action_offsets: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        context = self.get_tracking_context()
        reference = context["reference"]
        joint_acc = torch.sum(torch.square(self.robot.data.joint_acc[:, self.action_joint_ids]), dim=-1)
        joint_torque = torch.sum(torch.square(self.robot.data.applied_torque[:, self.action_joint_ids]), dim=-1)
        action_rate = torch.sum(torch.square(action_offsets - previous_action), dim=-1)
        out_of_limits = -(
            self.robot.data.joint_pos[:, self.action_joint_ids]
            - self.robot.data.soft_joint_pos_limits[:, self.action_joint_ids, 0]
        ).clip(max=0.0)
        out_of_limits += (
            self.robot.data.joint_pos[:, self.action_joint_ids]
            - self.robot.data.soft_joint_pos_limits[:, self.action_joint_ids, 1]
        ).clip(min=0.0)
        joint_limit = torch.sum(out_of_limits, dim=-1)

        anchor_pos_error = torch.sum(
            torch.square(reference["anchor_pos_w"] - context["robot_anchor_pos_w"]),
            dim=-1,
        )
        anchor_pos_reward = torch.exp(-anchor_pos_error / (0.3**2))
        anchor_ori_error = quat_error_magnitude(reference["anchor_quat_w"], context["robot_anchor_quat_w"]) ** 2
        anchor_ori_reward = torch.exp(-anchor_ori_error / (0.4**2))

        body_pos_error = torch.sum(
            torch.square(context["body_pos_relative_w"] - context["robot_body_pos_w"]),
            dim=-1,
        )
        body_pos_reward = torch.exp(-body_pos_error.mean(-1) / (0.3**2))
        body_ori_error = quat_error_magnitude(
            context["body_quat_relative_w"],
            context["robot_body_quat_w"],
        ) ** 2
        body_ori_reward = torch.exp(-body_ori_error.mean(-1) / (0.4**2))
        body_lin_vel_error = torch.sum(
            torch.square(reference["body_lin_vel_w"] - context["robot_body_lin_vel_w"]),
            dim=-1,
        )
        body_lin_vel_reward = torch.exp(-body_lin_vel_error.mean(-1) / (1.0**2))
        body_ang_vel_error = torch.sum(
            torch.square(reference["body_ang_vel_w"] - context["robot_body_ang_vel_w"]),
            dim=-1,
        )
        body_ang_vel_reward = torch.exp(-body_ang_vel_error.mean(-1) / (3.14**2))
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        undesired_contact_mask = (
            torch.max(torch.norm(net_contact_forces[:, :, self.undesired_contact_body_ids], dim=-1), dim=1)[0]
            > UNDESIRED_CONTACT_THRESHOLD
        )
        undesired_contacts = torch.sum(undesired_contact_mask.to(dtype=torch.float32), dim=-1)

        reward = (
            -2.5e-7 * joint_acc
            - 1.0e-5 * joint_torque
            - 1.0e-1 * action_rate
            - 10.0 * joint_limit
            + 0.5 * anchor_pos_reward
            + 0.5 * anchor_ori_reward
            + 1.0 * body_pos_reward
            + 1.0 * body_ori_reward
            + 1.0 * body_lin_vel_reward
            + 1.0 * body_ang_vel_reward
            - 0.1 * undesired_contacts
        ) * self.dt
        return reward, {
            "joint_acc": joint_acc,
            "joint_torque": joint_torque,
            "action_rate": action_rate,
            "joint_limit": joint_limit,
            "anchor_pos_reward": anchor_pos_reward,
            "anchor_ori_reward": anchor_ori_reward,
            "body_pos_reward": body_pos_reward,
            "body_ori_reward": body_ori_reward,
            "body_lin_vel_reward": body_lin_vel_reward,
            "body_ang_vel_reward": body_ang_vel_reward,
            "undesired_contacts": undesired_contacts,
            "diag_torso_ori_deg": body_ori_error[:, 7].sqrt() * (180.0 / 3.14159),
            "diag_left_wrist_ori_deg": body_ori_error[:, 10].sqrt() * (180.0 / 3.14159),
            "diag_right_wrist_ori_deg": body_ori_error[:, 13].sqrt() * (180.0 / 3.14159),
            "diag_left_shoulder_ori_deg": body_ori_error[:, 8].sqrt() * (180.0 / 3.14159),
            "diag_right_shoulder_ori_deg": body_ori_error[:, 11].sqrt() * (180.0 / 3.14159),
            "diag_left_elbow_ori_deg": body_ori_error[:, 9].sqrt() * (180.0 / 3.14159),
            "diag_right_elbow_ori_deg": body_ori_error[:, 12].sqrt() * (180.0 / 3.14159),
            "diag_torso_ang_vel": body_ang_vel_error[:, 7].sqrt(),
            "diag_left_wrist_ang_vel": body_ang_vel_error[:, 10].sqrt(),
            "diag_right_wrist_ang_vel": body_ang_vel_error[:, 13].sqrt(),
            "diag_left_shoulder_ang_vel": body_ang_vel_error[:, 8].sqrt(),
            "diag_right_shoulder_ang_vel": body_ang_vel_error[:, 11].sqrt(),
            "diag_left_elbow_ang_vel": body_ang_vel_error[:, 9].sqrt(),
            "diag_right_elbow_ang_vel": body_ang_vel_error[:, 12].sqrt(),
        }
