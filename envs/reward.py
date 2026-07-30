from __future__ import annotations

import torch

from isaaclab.utils.math import quat_error_magnitude

from .contracts import require_finite_tensors
from .spec import UNDESIRED_CONTACT_THRESHOLD


class MimicRewardMixin:
    def compute_reward(
        self,
        actions: torch.Tensor,
        previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        require_finite_tensors(
            {"actions": actions, "previous_action": previous_action},
            context="Reward inputs",
        )
        context = self.get_tracking_context()
        reference = context["reference"]

        action_rate = torch.sum(torch.square(actions - previous_action), dim=-1)

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
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        require_finite_tensors(
            {"net_forces_w_history": net_contact_forces},
            context="Contact sensor",
        )
        undesired_contact_mask = (
            torch.max(torch.norm(net_contact_forces[:, :, self.undesired_contact_body_ids], dim=-1), dim=1)[0]
            > UNDESIRED_CONTACT_THRESHOLD
        )
        undesired_contacts = torch.sum(undesired_contact_mask.to(dtype=torch.float32), dim=-1)

        action_rate_weight = float(self.config.action_rate_weight)
        if action_rate_weight != 0.1:
            raise RuntimeError(
                "Fixed pose reward requires environment.action_rate_weight=0.1, "
                f"got {action_rate_weight}"
            )
        contributions = {
            "anchor_pos_contribution": 0.5 * anchor_pos_reward * self.dt,
            "anchor_ori_contribution": 0.5 * anchor_ori_reward * self.dt,
            "body_pos_contribution": 2.0 * body_pos_reward * self.dt,
            "body_ori_contribution": 2.0 * body_ori_reward * self.dt,
            "action_rate_contribution": -action_rate_weight * action_rate * self.dt,
            "joint_limit_contribution": -10.0 * joint_limit * self.dt,
            "undesired_contacts_contribution": -0.1 * undesired_contacts * self.dt,
        }
        reward = torch.stack(tuple(contributions.values()), dim=0).sum(dim=0)
        reconstructed_reward = sum(contributions.values())
        decomposition_error = torch.abs(reward - reconstructed_reward)
        tensors_to_validate = {
            "reward": reward,
            "anchor_pos_reward": anchor_pos_reward,
            "anchor_ori_reward": anchor_ori_reward,
            "body_pos_reward": body_pos_reward,
            "body_ori_reward": body_ori_reward,
            "action_rate": action_rate,
            "joint_limit": joint_limit,
            "undesired_contacts": undesired_contacts,
            **contributions,
        }
        require_finite_tensors(tensors_to_validate, context="Reward tensors")
        similarity_names = (
            "anchor_pos_reward",
            "anchor_ori_reward",
            "body_pos_reward",
            "body_ori_reward",
        )
        invalid_similarity = torch.stack(
            tuple(
                ((tensors_to_validate[name] < 0.0) | (tensors_to_validate[name] > 1.0)).any()
                for name in similarity_names
            )
        )
        if bool(invalid_similarity.any()):
            invalid_cpu = invalid_similarity.detach().cpu().tolist()
            bad_names = [
                name
                for name, is_bad in zip(similarity_names, invalid_cpu)
                if is_bad
            ]
            raise RuntimeError(
                f"Raw pose similarities lie outside [0, 1]: {bad_names}"
            )
        max_decomposition_error = float(decomposition_error.max().item())
        if max_decomposition_error > 1.0e-6:
            raise RuntimeError(
                "Reward decomposition identity violated; "
                f"max error={max_decomposition_error:.6g}"
            )
        max_reward = float(reward.max().item())
        theoretical_max = 5.0 * float(self.dt)
        if max_reward > theoretical_max + 1.0e-6:
            raise RuntimeError(
                "Fixed pose reward exceeds its theoretical per-step maximum; "
                f"max={max_reward:.6g}, limit={theoretical_max + 1.0e-6:.6g}"
            )
        return reward, {
            "action_rate": action_rate,
            "joint_limit": joint_limit,
            "anchor_pos_reward": anchor_pos_reward,
            "anchor_ori_reward": anchor_ori_reward,
            "body_pos_reward": body_pos_reward,
            "body_ori_reward": body_ori_reward,
            "undesired_contacts": undesired_contacts,
            **contributions,
            "reward_decomposition_error": decomposition_error,
            "diag_torso_ori_deg": body_ori_error[:, 7].sqrt() * (180.0 / 3.14159),
            "diag_left_wrist_ori_deg": body_ori_error[:, 10].sqrt() * (180.0 / 3.14159),
            "diag_right_wrist_ori_deg": body_ori_error[:, 13].sqrt() * (180.0 / 3.14159),
            "diag_left_shoulder_ori_deg": body_ori_error[:, 8].sqrt() * (180.0 / 3.14159),
            "diag_right_shoulder_ori_deg": body_ori_error[:, 11].sqrt() * (180.0 / 3.14159),
            "diag_left_elbow_ori_deg": body_ori_error[:, 9].sqrt() * (180.0 / 3.14159),
            "diag_right_elbow_ori_deg": body_ori_error[:, 12].sqrt() * (180.0 / 3.14159),
        }
