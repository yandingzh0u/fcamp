from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply_inverse

from .spec import (
    ANCHOR_ORI_TERMINATION_THRESHOLD,
    ANCHOR_Z_TERMINATION_THRESHOLD,
    EE_Z_TERMINATION_THRESHOLD,
)


class MimicTerminationMixin:
    def compute_termination(self) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        context = self.get_tracking_context()
        reference = context["reference"]
        gravity_vec = self.robot.data.GRAVITY_VEC_W
        motion_projected_gravity_b = quat_apply_inverse(reference["anchor_quat_w"], gravity_vec)
        robot_projected_gravity_b = quat_apply_inverse(context["robot_anchor_quat_w"], gravity_vec)

        anchor_z_error = torch.abs(reference["anchor_pos_w"][:, 2] - context["robot_anchor_pos_w"][:, 2])
        anchor_gravity_z_error = (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs()

        robot_anchor_height = context["robot_anchor_pos_w"][:, 2]
        robot_anchor_tilt = torch.acos(torch.clamp(-robot_projected_gravity_b[:, 2], -1.0, 1.0)).abs()
        anchor_pos_bad = anchor_z_error > ANCHOR_Z_TERMINATION_THRESHOLD
        anchor_ori_bad = anchor_gravity_z_error > ANCHOR_ORI_TERMINATION_THRESHOLD
        ee_z_error = torch.abs(
            context["body_pos_relative_w"][:, self.ee_body_indices, 2]
            - context["robot_body_pos_w"][:, self.ee_body_indices, 2]
        )
        termination_z_error = torch.abs(
            context["body_pos_relative_w"][:, self.termination_body_indices, 2]
            - context["robot_body_pos_w"][:, self.termination_body_indices, 2]
        )
        ee_body_bad = torch.any(termination_z_error > EE_Z_TERMINATION_THRESHOLD, dim=-1)
        ee_z_error_max = ee_z_error.max(dim=-1).values
        ee_z_error_mean = ee_z_error.mean(dim=-1)
        time_out = self.episode_steps >= self.max_episode_steps

        motion_complete = self._motion_end_mask
        done = time_out | anchor_pos_bad | anchor_ori_bad | ee_body_bad | motion_complete
        return done, {
            "time_out": time_out,
            "motion_complete": motion_complete,
            "anchor_pos_bad": anchor_pos_bad,
            "anchor_ori_bad": anchor_ori_bad,
            "ee_body_bad": ee_body_bad,
        }, {
            "anchor_z_error": anchor_z_error,
            "anchor_gravity_z_error": anchor_gravity_z_error,
            "robot_anchor_height": robot_anchor_height,
            "robot_anchor_tilt": robot_anchor_tilt,
            "ee_z_error_max": ee_z_error_max,
            "ee_z_error_mean": ee_z_error_mean,
            "ee_z_error_by_body": ee_z_error,
        }
