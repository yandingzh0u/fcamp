from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply_inverse

from .spec import (
    ANCHOR_ORI_TERMINATION_THRESHOLD,
    ANCHOR_Z_TERMINATION_THRESHOLD,
    EE_Z_TERMINATION_THRESHOLD,
)


class MimicTerminationMixin:
    def _amp_ground_contact_forces(self) -> torch.Tensor:
        """Return one world-frame ground-contact force per sensed body."""

        force_matrix = self.contact_sensor.data.force_matrix_w
        if torch.is_tensor(force_matrix):
            if force_matrix.ndim != 4 or force_matrix.shape[-2:] != (1, 3):
                raise RuntimeError(
                    "Ground-filter contact force matrix must have shape "
                    f"[N,B,1,3], got {tuple(force_matrix.shape)}"
                )
            return force_matrix[..., 0, :]
        forces = self.contact_sensor.data.net_forces_w
        if not torch.is_tensor(forces) or forces.ndim != 3 or forces.shape[-1] != 3:
            raise RuntimeError("Contact sensor did not expose finite body forces")
        return forces

    def _amp_numerical_failure(self) -> torch.Tensor:
        finite = (
            torch.isfinite(self.robot.data.root_link_pos_w).all(dim=-1)
            & torch.isfinite(self.robot.data.root_link_quat_w).all(dim=-1)
            & torch.isfinite(self.robot.data.root_link_vel_w).all(dim=-1)
            & torch.isfinite(self.robot.data.body_pos_w).all(dim=(-1, -2))
            & torch.isfinite(self.robot.data.body_quat_w).all(dim=(-1, -2))
            & torch.isfinite(self.robot.data.joint_pos).all(dim=-1)
            & torch.isfinite(self.robot.data.joint_vel).all(dim=-1)
            & torch.isfinite(self.last_action).all(dim=-1)
        )
        return ~finite

    def compute_termination(self) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # Reference-relative quantities remain available as counterfactual
        # diagnostics, but standard pure AMP never uses them as terminal
        # conditions.
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
        contact_forces = self._amp_ground_contact_forces()
        contact_force_abs = contact_forces.abs()
        contact_force_max_by_body = contact_force_abs.max(dim=-1).values
        if self.task.contact_termination:
            illegal_forces = contact_forces.detach().clone()
            if self.allowed_contact_sensor_ids.numel() > 0:
                illegal_forces[:, self.allowed_contact_sensor_ids, :] = 0.0
            illegal_contact_by_body = illegal_forces.abs().gt(0.1).any(dim=-1)
            illegal_contact = illegal_contact_by_body.any(dim=-1)
        else:
            illegal_contact_by_body = torch.zeros(
                contact_forces.shape[:2],
                dtype=torch.bool,
                device=self.device,
            )
            illegal_contact = torch.zeros(
                self.num_envs,
                dtype=torch.bool,
                device=self.device,
            )
        numerical_failure = self._amp_numerical_failure()
        physical_failure = illegal_contact | numerical_failure
        tracking_failure = anchor_pos_bad | anchor_ori_bad | ee_body_bad
        # MimicKit AMP: timeout plus motion-specific illegal physical contact.
        # Motion end and reference pose deviation are deliberately nonterminal.
        done = time_out | physical_failure
        return done, {
            "time_out": time_out,
            "motion_complete": motion_complete,
            "anchor_pos_bad": anchor_pos_bad,
            "anchor_ori_bad": anchor_ori_bad,
            "ee_body_bad": ee_body_bad,
            "tracking_failure": tracking_failure,
            "illegal_contact": illegal_contact,
            "numerical_failure": numerical_failure,
            "physical_failure": physical_failure,
        }, {
            "anchor_z_error": anchor_z_error,
            "anchor_gravity_z_error": anchor_gravity_z_error,
            "robot_anchor_height": robot_anchor_height,
            "robot_anchor_tilt": robot_anchor_tilt,
            "ee_z_error_max": ee_z_error_max,
            "ee_z_error_mean": ee_z_error_mean,
            "ee_z_error_by_body": ee_z_error,
            "contact_force_max": contact_force_max_by_body.max(dim=-1).values,
            "contact_force_max_by_body": contact_force_max_by_body,
            "illegal_contact_body_count": illegal_contact_by_body.sum(dim=-1),
        }
