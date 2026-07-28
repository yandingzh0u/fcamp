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
from .imitation_data import build_g1_imitation_frame


class MimicObservationMixin:
    def get_imitation_policy_frame(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Return the native method-specific post-action imitation frame."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if env_ids.ndim != 1:
            raise ValueError(f"env_ids must be 1-D, got {tuple(env_ids.shape)}")
        joint_pos, joint_vel = self.get_action_joint_state()
        root_pos = self.robot.data.root_link_pos_w.index_select(0, env_ids)
        env_origins = self.scene.env_origins.index_select(0, env_ids)
        key_body_pos = self.robot.data.body_pos_w.index_select(0, env_ids)[..., self.imitation_key_body_ids, :]
        # Scene origins are translations only.  Subtracting them from both root
        # and key bodies keeps root-relative key features invariant and removes
        # the vectorized-environment grid from the discriminator input.
        root_pos_local = root_pos - env_origins
        key_body_pos_local = key_body_pos - env_origins.unsqueeze(-2)
        root_velocity = self.robot.data.root_link_vel_w.index_select(0, env_ids)
        return build_g1_imitation_frame(
            root_pos=root_pos_local,
            root_quat_wxyz=self.robot.data.root_link_quat_w.index_select(0, env_ids),
            joint_pos=joint_pos.index_select(0, env_ids),
            key_body_pos=key_body_pos_local,
            root_lin_vel=root_velocity[:, :3],
            root_ang_vel=root_velocity[:, 3:],
            joint_vel=joint_vel.index_select(0, env_ids),
        )

    def get_fcamp_fk_aligned_policy_frame(
        self, env_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Rebuild the same simulator state through FCAMP's expert FK path."""

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if env_ids.ndim != 1:
            raise ValueError(f"env_ids must be 1-D, got {tuple(env_ids.shape)}")
        joint_pos, joint_vel = self.get_action_joint_state()
        origins = self.scene.env_origins.index_select(0, env_ids)
        return self.motion.build_fcamp_frame_from_robot_state(
            root_pos=self.robot.data.root_link_pos_w.index_select(0, env_ids) - origins,
            root_quat=self.robot.data.root_link_quat_w.index_select(0, env_ids),
            joint_pos=joint_pos.index_select(0, env_ids),
            root_link_velocity=self.robot.data.root_link_vel_w.index_select(0, env_ids),
            joint_vel=joint_vel.index_select(0, env_ids),
        )

    def get_evaluator_imitation_policy_frame(
        self,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the method-independent policy frame used only by validation.

        Training imitation features intentionally preserve each paper's native
        root-velocity convention. Cross-method evaluation must not: it always
        uses the physical root-link pose/velocity exposed by Isaac Lab, so ADD's
        link convention and the other methods' COM convention cannot change the
        external metric for an identical simulator state.
        """

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if env_ids.ndim != 1:
            raise ValueError(f"env_ids must be 1-D, got {tuple(env_ids.shape)}")
        joint_pos, joint_vel = self.get_action_joint_state()
        root_pos = self.robot.data.root_link_pos_w.index_select(0, env_ids)
        root_quat = self.robot.data.root_link_quat_w.index_select(0, env_ids)
        root_velocity = self.robot.data.root_link_vel_w.index_select(0, env_ids)
        env_origins = self.scene.env_origins.index_select(0, env_ids)
        key_body_pos = self.robot.data.body_pos_w.index_select(0, env_ids)[
            ..., self.imitation_key_body_ids, :
        ]
        return build_g1_imitation_frame(
            root_pos=root_pos - env_origins,
            root_quat_wxyz=root_quat,
            joint_pos=joint_pos.index_select(0, env_ids),
            key_body_pos=key_body_pos - env_origins.unsqueeze(-2),
            root_lin_vel=root_velocity[:, :3],
            root_ang_vel=root_velocity[:, 3:],
            joint_vel=joint_vel.index_select(0, env_ids),
        )

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
        motion_anchor_ori_b = matrix_from_quat(motion_anchor_ori)[..., :2].reshape(
            robot_anchor_pos_w.shape[0], -1
        )
        return motion_anchor_pos_b, motion_anchor_ori_b

    def get_observation(self) -> torch.Tensor:
        return self.build_observation()

    def get_critic_observation(self) -> torch.Tensor:
        return self.build_critic_observation()

    def build_observation(self) -> torch.Tensor:
        observation = torch.cat(
            [
                self.get_imitation_policy_frame()[..., 2:],
                self.last_action,
            ],
            dim=-1,
        )
        if observation.shape[-1] != OBS_DIM:
            raise RuntimeError(f"Expected observation dim {OBS_DIM}, got {observation.shape[-1]}")
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
