from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply_inverse, quat_error_magnitude

from .config import TILT_REWARD_SIGMA, UNDESIRED_CONTACT_THRESHOLD


class MimicRewardMixin:
    def compute_reward(
        self,
        action_offsets: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        context = self.get_tracking_context()
        reference = context["reference"]
        action_regularization_scale = float(getattr(self.task_cfg, "action_scale_multiplier", 1.0))
        regularized_action = action_offsets * action_regularization_scale
        regularized_previous_action = previous_action * action_regularization_scale
        action_rate = torch.sum(torch.square(regularized_action - regularized_previous_action), dim=-1)
        # Intensive (per-joint mean) version of the action-rate, used by the bounded quality
        # term so it does NOT scale with the number of joints (the extensive sum is ~28 and
        # makes exp(-sum) underflow to 0, killing the gradient). Mean keeps it O(sigma^2).
        action_rate_mse = torch.mean(torch.square(regularized_action - regularized_previous_action), dim=-1)
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
        # Official anchor-position contract: single isotropic sigma=0.3 over the full 3-D
        # position error (no separate tightened z term — sigma=0.1 on z zeroed the reward as
        # soon as the robot drifted and stopped giving a usable gradient).
        anchor_pos_error = anchor_xy_error + anchor_z_error_sq
        anchor_pos_reward = torch.exp(-anchor_pos_error / (0.3**2))
        anchor_ori_error = quat_error_magnitude(reference["anchor_quat_w"], context["robot_anchor_quat_w"]) ** 2
        anchor_ori_reward = torch.exp(-anchor_ori_error / (0.4**2))

        # Relative anchor tilt angle (radians) — the SAME quantity the termination uses, so the
        # reward gives a continuous gradient toward upright as the robot approaches the tilt
        # death line. (Full-orientation quat reward is kept above for yaw etc.)
        gravity_vec = self.robot.data.GRAVITY_VEC_W
        motion_grav_b = quat_apply_inverse(reference["anchor_quat_w"], gravity_vec)
        robot_grav_b = quat_apply_inverse(context["robot_anchor_quat_w"], gravity_vec)
        cos_tilt = torch.sum(motion_grav_b * robot_grav_b, dim=-1) / (
            motion_grav_b.norm(dim=-1).clamp(min=1e-6) * robot_grav_b.norm(dim=-1).clamp(min=1e-6)
        )
        anchor_tilt_error = torch.acos(torch.clamp(cos_tilt, -1.0, 1.0))
        tilt_quality = torch.exp(-((anchor_tilt_error / TILT_REWARD_SIGMA) ** 2))

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
        # Intensive (fraction of monitored bodies in undesired contact) version for the quality
        # term, so it is independent of how many bodies are monitored.
        num_undesired_bodies = max(1, int(self.undesired_contact_body_ids.numel()))
        undesired_contact_frac = undesired_contacts / float(num_undesired_bodies)
        # Per-joint mean limit violation (intensive), for the quality term.
        joint_limit_mean = torch.mean(out_of_limits, dim=-1)

        action_rate_weight = float(getattr(self.task_cfg, "action_rate_weight", 1.0e-1))

        # Official whole-body-tracking reward, retained ONLY for logging/diagnostics. It is
        # signed (can be negative), so GRPO must NOT optimize it directly: a signed per-step
        # reward lets the policy raise its return by dying early to truncate future negative
        # reward. The actual GRPO objective is the bounded non-negative grpo_score below.
        official_reward = (
            -action_rate_weight * action_rate
            - 10.0 * joint_limit
            + 0.5 * anchor_pos_reward
            + 0.5 * anchor_ori_reward
            + 1.0 * body_pos_reward
            + 1.0 * body_ori_reward
            + 1.0 * body_lin_vel_reward
            + 1.0 * body_ang_vel_reward
            - 0.1 * undesired_contacts
        ) * self.dt

        # Bounded non-negative GRPO score in [0, 1]: tracking is the ONLY way to earn score;
        # the regularization costs can only MODULATE it down through a bounded gate, never earn
        # score on their own. (The old additive form let action/joint/contact quality contribute
        # ~23% even when tracking was ~0, so a smooth low-contact but badly-tracking policy could
        # still score — exactly the failure seen in the logs.) tracking==0 => score==0.
        tracking_score = (
            0.5 * anchor_pos_reward
            + 0.5 * anchor_ori_reward
            + tilt_quality
            + body_pos_reward
            + body_ori_reward
            + body_lin_vel_reward
            + body_ang_vel_reward
        ) / 5.5
        # Normalized intensive costs (no exp(-sum) underflow); each is O(1) and >= 0.
        normalized_action_cost = action_rate_mse / (1.0**2)
        normalized_joint_cost = joint_limit_mean / 0.1
        normalized_contact_cost = undesired_contact_frac / 0.5
        w_action, w_joint, w_contact = 0.1, 0.1, 0.1
        cost = (
            w_action * normalized_action_cost
            + w_joint * normalized_joint_cost
            + w_contact * normalized_contact_cost
        )
        gate = 1.0 / (1.0 + cost)  # in (0, 1]
        grpo_score = tracking_score * gate
        return grpo_score, {
            "official_reward": official_reward,
            "tracking": tracking_score,
            "tilt_quality": tilt_quality,
            "gate": gate,
            "grpo_score": grpo_score,
            "anchor_tilt_error_deg": anchor_tilt_error * (180.0 / 3.14159265),
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
