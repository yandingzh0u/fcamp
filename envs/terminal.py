from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply_inverse

from .spec import (
    ANCHOR_ORI_TERMINATION_THRESHOLD,
    ANCHOR_Z_TERMINATION_THRESHOLD,
    EE_Z_TERMINATION_THRESHOLD,
)


BEYONDMIMIC_ANCHOR_Z_TERMINATION_THRESHOLD = 0.25
BEYONDMIMIC_ANCHOR_ORI_TERMINATION_THRESHOLD = 0.8
BEYONDMIMIC_EE_Z_TERMINATION_THRESHOLD = 0.25


class MimicTerminationMixin:
    def _amp_ground_contact_forces_w(self) -> torch.Tensor:
        """Match MimicKit's IsaacLab ground-filtered contact forces."""
        force_matrix_w = getattr(self.contact_sensor.data, "force_matrix_w", None)
        if torch.is_tensor(force_matrix_w) and force_matrix_w.numel() > 0:
            return force_matrix_w.sum(dim=-2)
        net_forces_w = getattr(self.contact_sensor.data, "net_forces_w", None)
        if not torch.is_tensor(net_forces_w):
            net_forces_w = self.contact_sensor.data.net_forces_w_history[:, -1]
        ground_forces = net_forces_w.clone()
        contact_robot_body_ids = getattr(self, "contact_robot_body_ids", None)
        if torch.is_tensor(contact_robot_body_ids):
            body_height = torch.full(
                ground_forces.shape[:2],
                float("inf"),
                dtype=ground_forces.dtype,
                device=ground_forces.device,
            )
            known = contact_robot_body_ids >= 0
            if bool(known.any()):
                robot_body_ids = contact_robot_body_ids[known]
                body_height[:, known] = self.robot.data.body_pos_w[:, robot_body_ids, 2]
            ground_forces[body_height > 0.3] = 0.0
        return ground_forces

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

        if getattr(self, "termination_mode", "tracking") == "beyondmimic":
            # Official BeyondMimic uses only these three tracking failures.
            # In particular, undesired contact contributes a reward penalty
            # but is never a termination condition.
            anchor_pos_bad = (
                anchor_z_error > BEYONDMIMIC_ANCHOR_Z_TERMINATION_THRESHOLD
            )
            anchor_ori_bad = (
                anchor_gravity_z_error
                > BEYONDMIMIC_ANCHOR_ORI_TERMINATION_THRESHOLD
            )
            ee_body_bad = torch.any(
                termination_z_error
                > BEYONDMIMIC_EE_Z_TERMINATION_THRESHOLD,
                dim=-1,
            )
            zeros = torch.zeros_like(time_out)
            done = time_out | anchor_pos_bad | anchor_ori_bad | ee_body_bad
            return done, {
                "time_out": time_out,
                "motion_complete": zeros,
                "anchor_pos_bad": anchor_pos_bad,
                "anchor_ori_bad": anchor_ori_bad,
                "ee_body_bad": ee_body_bad,
                "fall_contact": zeros,
            }, {
                "anchor_z_error": anchor_z_error,
                "anchor_gravity_z_error": anchor_gravity_z_error,
                "robot_anchor_height": robot_anchor_height,
                "robot_anchor_tilt": robot_anchor_tilt,
                "ee_z_error_max": ee_z_error_max,
                "ee_z_error_mean": ee_z_error_mean,
                "ee_z_error_by_body": ee_z_error,
            }

        if getattr(self, "termination_mode", "tracking") == "add":
            ground_contact_forces = self._amp_ground_contact_forces_w()
            add_allowed_contact_body_ids = getattr(
                self,
                "add_allowed_contact_body_ids",
                torch.empty(0, dtype=torch.long, device=self.device),
            )
            add_undesired_contact_body_ids = getattr(
                self,
                "add_undesired_contact_body_ids",
                self.undesired_contact_body_ids,
            )
            masked_contact = ground_contact_forces.detach().clone()
            if add_allowed_contact_body_ids.numel() > 0:
                masked_contact[:, add_allowed_contact_body_ids] = 0.0
            add_contact_force_by_body = torch.zeros(
                self.num_envs,
                int(add_undesired_contact_body_ids.numel()),
                device=self.device,
            )
            if add_undesired_contact_body_ids.numel() > 0:
                add_contact_force_by_body = torch.amax(
                    masked_contact[:, add_undesired_contact_body_ids].abs(),
                    dim=-1,
                )
                fall_contact = torch.any(add_contact_force_by_body > 0.1, dim=-1)
            else:
                fall_contact = torch.zeros_like(time_out)

            add_body_ids = getattr(
                self,
                "add_disc_body_ids",
                torch.arange(len(self.robot.body_names), dtype=torch.long, device=self.device),
            )
            reference_full = self.motion.get_add_body_pos(self.phase_steps, body_ids=add_body_ids)
            reference_full = reference_full + self.scene.env_origins[:, None, :]
            robot_add_body_pos = self.robot.data.body_pos_w.index_select(1, add_body_ids)
            body_pos_diff = reference_full - robot_add_body_pos
            body_pos_dist_sq = body_pos_diff.square().sum(dim=-1)
            pose_fail = body_pos_dist_sq.max(dim=-1).values > 1.0
            root_pos_dist_sq = (
                reference["root_pos_w"] - self.robot.data.root_pos_w
            ).square().sum(dim=-1)
            pose_fail = pose_fail | (root_pos_dist_sq > 1.0)

            failed = (fall_contact | pose_fail) & (self.episode_steps > 0)
            motion_complete = self._motion_end_mask if self.terminate_on_motion_end else torch.zeros_like(time_out)
            done = time_out | failed | motion_complete
            zeros = torch.zeros_like(time_out)
            return done, {
                "time_out": time_out,
                "motion_complete": motion_complete,
                "anchor_pos_bad": zeros,
                "anchor_ori_bad": zeros,
                "ee_body_bad": failed,
                "fall_contact": fall_contact & (self.episode_steps > 0),
                "pose_fail": pose_fail & (self.episode_steps > 0),
            }, {
                "anchor_z_error": anchor_z_error,
                "anchor_gravity_z_error": anchor_gravity_z_error,
                "robot_anchor_height": robot_anchor_height,
                "robot_anchor_tilt": robot_anchor_tilt,
                "ee_z_error_max": ee_z_error_max,
                "ee_z_error_mean": ee_z_error_mean,
                "ee_z_error_by_body": ee_z_error,
                "add_undesired_contact_force_by_body": add_contact_force_by_body,
                "add_pose_body_dist": torch.sqrt(body_pos_dist_sq.max(dim=-1).values),
            }

        if getattr(self, "termination_mode", "tracking") == "amp":
            ground_contact_forces = self._amp_ground_contact_forces_w()
            amp_undesired_contact_body_ids = getattr(
                self,
                "amp_undesired_contact_body_ids",
                self.undesired_contact_body_ids,
            )
            amp_contact_force_by_body = torch.zeros(
                self.num_envs,
                int(amp_undesired_contact_body_ids.numel()),
                device=self.device,
            )
            if amp_undesired_contact_body_ids.numel() > 0:
                amp_contact_force_by_body = torch.norm(
                    ground_contact_forces[:, amp_undesired_contact_body_ids],
                    dim=-1,
                )
                fall_contact = torch.any(amp_contact_force_by_body > 0.1, dim=-1)
            else:
                fall_contact = torch.zeros_like(time_out)
            fall_contact = fall_contact & (self.episode_steps > 0)
            motion_complete = torch.zeros_like(time_out)
            if self.terminate_on_motion_end:
                motion_complete = self._motion_end_mask
            done = time_out | fall_contact | motion_complete
            zeros = torch.zeros_like(time_out)
            return done, {
                "time_out": time_out,
                "motion_complete": motion_complete,
                "anchor_pos_bad": zeros,
                "anchor_ori_bad": zeros,
                "ee_body_bad": fall_contact,
                "fall_contact": fall_contact,
            }, {
                "anchor_z_error": anchor_z_error,
                "anchor_gravity_z_error": anchor_gravity_z_error,
                "robot_anchor_height": robot_anchor_height,
                "robot_anchor_tilt": robot_anchor_tilt,
                "ee_z_error_max": ee_z_error_max,
                "ee_z_error_mean": ee_z_error_mean,
                "ee_z_error_by_body": ee_z_error,
                "amp_undesired_contact_force_by_body": amp_contact_force_by_body,
            }

        motion_complete = torch.zeros_like(time_out)
        if self.terminate_on_motion_end:
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
