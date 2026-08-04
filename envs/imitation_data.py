from __future__ import annotations

import torch

from components.imitation.motion_features import (
    ImitationFeatureSchema,
    build_imitation_frame,
    revolute_dof_to_quat,
)


G1_IMITATION_KEY_BODY_NAMES = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "head_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)

# Joint order is exactly G1_29DOF_ACTION_NAMES.  Axes come from the checked-in
# g1_29dof.urdf and are kept explicit so demo features do not depend on Isaac.
G1_IMITATION_JOINT_AXES = (
    (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0),
    (0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
)

G1_IMITATION_NUM_JOINTS = len(G1_IMITATION_JOINT_AXES)
G1_IMITATION_NUM_KINEMATIC_JOINTS = G1_IMITATION_NUM_JOINTS + 1
G1_IMITATION_NUM_KEY_BODIES = len(G1_IMITATION_KEY_BODY_NAMES)
G1_IMITATION_SCHEMA = ImitationFeatureSchema(
    num_joint_rotations=G1_IMITATION_NUM_KINEMATIC_JOINTS,
    num_dofs=G1_IMITATION_NUM_JOINTS,
    num_key_bodies=G1_IMITATION_NUM_KEY_BODIES,
)
G1_IMITATION_FRAME_DIM = G1_IMITATION_SCHEMA.frame_dim


def build_g1_amp_actor_observation(
    imitation_frame: torch.Tensor,
) -> torch.Tensor:
    """Reorder one discriminator frame into MimicKit's G1 actor schema."""

    if imitation_frame.shape[-1] != G1_IMITATION_FRAME_DIM:
        raise ValueError(
            "G1 imitation frame must end in dimension "
            f"{G1_IMITATION_FRAME_DIM}, got {tuple(imitation_frame.shape)}"
        )
    joint_rot_start = 9
    joint_rot_stop = joint_rot_start + 6 * G1_IMITATION_NUM_KINEMATIC_JOINTS
    key_pos_stop = joint_rot_stop + 3 * G1_IMITATION_NUM_KEY_BODIES
    root_lin_vel_stop = key_pos_stop + 3
    root_ang_vel_stop = root_lin_vel_stop + 3
    return torch.cat(
        (
            imitation_frame[..., 2:3],
            imitation_frame[..., 3:9],
            imitation_frame[..., key_pos_stop:root_lin_vel_stop],
            imitation_frame[..., root_lin_vel_stop:root_ang_vel_stop],
            imitation_frame[..., joint_rot_start:joint_rot_stop],
            imitation_frame[..., root_ang_vel_stop:],
            imitation_frame[..., joint_rot_stop:key_pos_stop],
        ),
        dim=-1,
    )


def build_g1_imitation_frame(
    *,
    root_pos: torch.Tensor,
    root_quat_wxyz: torch.Tensor,
    joint_pos: torch.Tensor,
    key_body_pos: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    joint_vel: torch.Tensor,
) -> torch.Tensor:
    """Build one clean, reference-free imitation frame."""

    axes = joint_pos.new_tensor(G1_IMITATION_JOINT_AXES)
    actuated_joint_quat = revolute_dof_to_quat(joint_pos, axes)
    fixed_head = torch.zeros(
        actuated_joint_quat.shape[:-2] + (1, 4),
        device=actuated_joint_quat.device,
        dtype=actuated_joint_quat.dtype,
    )
    fixed_head[..., 0] = 1.0
    joint_quat = torch.cat(
        (
            actuated_joint_quat[..., :15, :],
            fixed_head,
            actuated_joint_quat[..., 15:, :],
        ),
        dim=-2,
    )
    return build_imitation_frame(
        root_pos=root_pos,
        root_quat=root_quat_wxyz,
        joint_rotation=joint_quat,
        key_body_pos=key_body_pos,
        root_lin_vel=root_lin_vel,
        root_ang_vel=root_ang_vel,
        dof_vel=joint_vel,
        schema=G1_IMITATION_SCHEMA,
    )


def history_indices(
    end_indices: torch.Tensor,
    window_size: int,
    num_frames: int,
) -> torch.Tensor:
    """Return chronological history indices ending at each requested frame."""

    if end_indices.ndim != 1:
        raise ValueError(f"end_indices must be 1-D, got {tuple(end_indices.shape)}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    end_indices = end_indices.long()
    if bool((end_indices < 0).any()) or bool((end_indices >= num_frames).any()):
        raise ValueError(f"end_indices must lie in [0, {num_frames - 1}]")
    offsets = torch.arange(1 - window_size, 1, device=end_indices.device)
    indices = end_indices.unsqueeze(-1) + offsets
    return indices.clamp_min(0)
