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
G1_IMITATION_NUM_KEY_BODIES = len(G1_IMITATION_KEY_BODY_NAMES)
G1_IMITATION_SCHEMA = ImitationFeatureSchema(
    num_joints=G1_IMITATION_NUM_JOINTS,
    num_key_bodies=G1_IMITATION_NUM_KEY_BODIES,
)
G1_IMITATION_FRAME_DIM = G1_IMITATION_SCHEMA.frame_dim


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
    """Build one clean, reference-free imitation frame.

    ``root_pos`` must already be expressed in the per-environment scene frame.
    ``key_body_pos`` uses the same frame and is converted to root-relative
    positions here.  No phase, reference target, action, contact, or privileged
    tracking error is included.
    """
    axes = joint_pos.new_tensor(G1_IMITATION_JOINT_AXES)
    joint_quat = revolute_dof_to_quat(joint_pos, axes)
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
    """Return chronological history indices ending at each requested frame.

    Negative indices are replaced by frame 0 for reset-history semantics.
    Motion-end wrapping is never performed.
    """
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
