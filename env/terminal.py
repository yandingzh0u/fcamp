from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply_inverse

from .config import (
    ANCHOR_TILT_TERMINATION_THRESHOLD,
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
        # Unified relative-tilt angle (radians) between the reference and robot anchor
        # orientations, measured through their projected-gravity directions. Both vectors are
        # unit (gravity is unit), so dot = cos(relative tilt) and acos(dot) is the exact angle.
        # This SINGLE physical quantity is used by the termination, the reward (tilt_quality)
        # and the validation log, so all three agree (the old code terminated on the full
        # gravity-vector NORM with a threshold copied from the official z-only check, a unit
        # mismatch that made ~47 deg lethal while logging a different z-only number).
        cos_tilt = torch.sum(motion_projected_gravity_b * robot_projected_gravity_b, dim=-1)
        cos_tilt = cos_tilt / (
            motion_projected_gravity_b.norm(dim=-1).clamp(min=1e-6)
            * robot_projected_gravity_b.norm(dim=-1).clamp(min=1e-6)
        )
        anchor_tilt_error = torch.acos(torch.clamp(cos_tilt, -1.0, 1.0))
        robot_anchor_height = context["robot_anchor_pos_w"][:, 2]
        anchor_pos_bad = anchor_z_error > ANCHOR_Z_TERMINATION_THRESHOLD
        anchor_ori_bad = anchor_tilt_error > ANCHOR_TILT_TERMINATION_THRESHOLD
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
        termination_z_error_max = termination_z_error.max(dim=-1).values
        termination_z_error_mean = termination_z_error.mean(dim=-1)
        time_out = self.episode_steps >= self.task_cfg.max_episode_steps
        motion_end = getattr(self, "_motion_end_mask", None)
        if motion_end is not None:
            time_out = time_out | motion_end
        done = time_out | anchor_pos_bad | anchor_ori_bad | ee_body_bad
        return done, {
            "time_out": time_out,
            "anchor_pos_bad": anchor_pos_bad,
            "anchor_ori_bad": anchor_ori_bad,
            "ee_body_bad": ee_body_bad,
        }, {
            "anchor_z_error": anchor_z_error,
            "anchor_tilt_error": anchor_tilt_error,
            "anchor_tilt_error_deg": anchor_tilt_error * (180.0 / 3.14159265),
            "robot_anchor_height": robot_anchor_height,
            "ee_z_error_max": ee_z_error_max,
            "ee_z_error_mean": ee_z_error_mean,
            "ee_z_error_by_body": ee_z_error,
            "termination_z_error_max": termination_z_error_max,
            "termination_z_error_mean": termination_z_error_mean,
        }
