from __future__ import annotations

import torch

from isaaclab.utils.math import quat_error_magnitude

from .config import UNDESIRED_CONTACT_THRESHOLD


class MimicRewardMixin:
    def compute_reward(
        self,
        action_offsets: torch.Tensor,
        previous_action: torch.Tensor,
        previous_previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        context = self.get_tracking_context()
        reference = context["reference"]
        joint_acc = torch.sum(torch.square(self.robot.data.joint_acc[:, self.action_joint_ids]), dim=-1)
        joint_torque = torch.sum(torch.square(self.robot.data.applied_torque[:, self.action_joint_ids]), dim=-1)
        action_regularization_scale = float(getattr(self.task_cfg, "action_scale_multiplier", 1.0))
        regularized_action = action_offsets * action_regularization_scale
        regularized_previous_action = previous_action * action_regularization_scale
        regularized_previous_previous_action = previous_previous_action * action_regularization_scale
        action_rate = torch.sum(torch.square(regularized_action - regularized_previous_action), dim=-1)
        action_accel = torch.sum(
            torch.square(regularized_action - 2.0 * regularized_previous_action + regularized_previous_previous_action),
            dim=-1,
        )
        action_l2 = torch.sum(torch.square(regularized_action), dim=-1)
        out_of_limits = -(
            self.robot.data.joint_pos[:, self.action_joint_ids]
            - self.robot.data.soft_joint_pos_limits[:, self.action_joint_ids, 0]
        ).clip(max=0.0)
        out_of_limits += (
            self.robot.data.joint_pos[:, self.action_joint_ids]
            - self.robot.data.soft_joint_pos_limits[:, self.action_joint_ids, 1]
        ).clip(min=0.0)
        joint_limit = torch.sum(out_of_limits, dim=-1)

        anchor_pos_diff = reference["anchor_pos_w"] - context["robot_anchor_pos_w"]
        anchor_xy_error = torch.sum(torch.square(anchor_pos_diff[..., :2]), dim=-1)
        anchor_z_error_sq = torch.square(anchor_pos_diff[..., 2])
        # z-axis sigma 0.1 keeps the gradient meaningful across the whole survival window:
        # 5cm drift -> reward 0.78 (gentle warning), 10cm drift -> 0.37 (strong push), 12cm
        # death threshold -> 0.24 (clearly distinguishable from healthy). The xy term keeps
        # 0.3 sigma so lateral tracking stays gentle.
        anchor_pos_reward = torch.exp(-anchor_xy_error / (0.3**2)) * torch.exp(-anchor_z_error_sq / (0.1**2))
        anchor_ori_error = quat_error_magnitude(reference["anchor_quat_w"], context["robot_anchor_quat_w"]) ** 2
        anchor_ori_reward = torch.exp(-anchor_ori_error / (0.4**2))

        body_pos_error = torch.sum(
            torch.square(context["body_pos_relative_w"] - context["robot_body_pos_w"]),
            dim=-1,
        )
        body_pos_reward = torch.exp(-body_pos_error.mean(-1) / (0.3**2))
        # Dense reward on the SAME quantity the hard termination gate checks: the MAX z-error
        # over the termination bodies (ankles + wrists). termination kills on
        #   any(|ref_z - robot_z| > EE_Z_TERMINATION_THRESHOLD)  over these bodies,
        # which is equivalent to  max(z_err) > threshold. Optimizing the mean over 14 bodies
        # lets a single wrist breach the gate while the mean still looks great, so the policy
        # gets a contradictory signal (dense: "fine", done: "dead"). This term aligns the
        # objective with the success condition by rewarding exp(-(max_term_z_err/sigma)^2).
        term_z_err = torch.abs(
            context["body_pos_relative_w"][:, self.termination_body_indices, 2]
            - context["robot_body_pos_w"][:, self.termination_body_indices, 2]
        )
        max_term_z_err = term_z_err.max(dim=-1).values
        term_z_sigma = float(getattr(self.task_cfg, "term_z_sigma", 0.12))
        term_z_reward = torch.exp(-(max_term_z_err / term_z_sigma) ** 2)
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
        joint_pos_error = torch.sum(
            torch.square(reference["joint_pos"] - context["robot_joint_pos"]),
            dim=-1,
        )
        joint_pos_reward = torch.exp(-joint_pos_error / (0.5**2))
        joint_vel_error = torch.sum(
            torch.square(reference["joint_vel"] - context["robot_joint_vel"]),
            dim=-1,
        )
        joint_vel_reward = torch.exp(-joint_vel_error / (10.0**2))
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        undesired_contact_mask = (
            torch.max(torch.norm(net_contact_forces[:, :, self.undesired_contact_body_ids], dim=-1), dim=1)[0]
            > UNDESIRED_CONTACT_THRESHOLD
        )
        undesired_contacts = torch.sum(undesired_contact_mask.to(dtype=torch.float32), dim=-1)

        joint_acc_weight = float(getattr(self.task_cfg, "joint_acc_weight", 2.5e-7))
        joint_torque_weight = float(getattr(self.task_cfg, "joint_torque_weight", 1.0e-5))
        action_rate_weight = float(getattr(self.task_cfg, "action_rate_weight", 1.0e-1))
        action_accel_weight = float(getattr(self.task_cfg, "action_accel_weight", 0.0))
        action_l2_weight = float(getattr(self.task_cfg, "action_l2_weight", 0.0))
        term_z_weight = float(getattr(self.task_cfg, "term_z_weight", 3.0))

        reward = (
            -joint_acc_weight * joint_acc
            - joint_torque_weight * joint_torque
            - action_rate_weight * action_rate
            - action_accel_weight * action_accel
            - action_l2_weight * action_l2
            - 10.0 * joint_limit
            + 2.0 * anchor_pos_reward
            + 2.0 * anchor_ori_reward
            + 1.0 * body_pos_reward
            + term_z_weight * term_z_reward
            + 1.0 * body_ori_reward
            + 1.0 * body_lin_vel_reward
            + 1.0 * body_ang_vel_reward
            + 1.0 * joint_pos_reward
            + 0.5 * joint_vel_reward
            - 0.1 * undesired_contacts
        ) * self.dt
        return reward, {
            "joint_acc": joint_acc,
            "joint_torque": joint_torque,
            "action_rate": action_rate,
            "action_accel": action_accel,
            "action_l2": action_l2,
            "joint_limit": joint_limit,
            "anchor_pos_reward": anchor_pos_reward,
            "anchor_ori_reward": anchor_ori_reward,
            "body_pos_reward": body_pos_reward,
            "term_z_reward": term_z_reward,
            "max_term_z_err": max_term_z_err,
            "body_ori_reward": body_ori_reward,
            "body_lin_vel_reward": body_lin_vel_reward,
            "body_ang_vel_reward": body_ang_vel_reward,
            "joint_pos_reward": joint_pos_reward,
            "joint_vel_reward": joint_vel_reward,
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
