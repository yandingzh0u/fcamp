from __future__ import annotations

import torch

from components.imitation.motion_features import ImitationFeatureSchema, build_imitation_frame, quat_to_rot6d, revolute_dof_to_quat


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
    history_len=16,
)
G1_IMITATION_FRAME_DIM = G1_IMITATION_SCHEMA.frame_dim


def quat_wxyz_to_tan_norm(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """MimicKit-compatible 6D tangent/normal rotation from wxyz quaternions.

    MimicKit stores quaternions as xyzw; this project and Isaac Lab use wxyz.
    Computing the rotated x and z basis vectors directly makes that convention
    boundary explicit and avoids a silent component-order bug.
    """
    return quat_to_rot6d(quat_wxyz)


def joint_positions_to_tan_norm(joint_pos: torch.Tensor) -> torch.Tensor:
    """Convert the 29 one-DoF G1 joint angles to local 6D rotations."""
    axes = joint_pos.new_tensor(G1_IMITATION_JOINT_AXES)
    quat = revolute_dof_to_quat(joint_pos, axes)
    return quat_to_rot6d(quat)


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
        joint_rotation_format="quat",
    )


def history_indices(
    end_indices: torch.Tensor,
    window_size: int,
    num_frames: int,
    *,
    clamp_start: bool,
) -> torch.Tensor:
    """Return chronological history indices ending at each requested frame.

    With ``clamp_start=True`` (reset-history semantics), negative indices are
    replaced by frame 0.  With it false, every window must be strictly in range.
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
    if clamp_start:
        return indices.clamp_min(0)
    if bool((indices < 0).any()):
        raise ValueError("Strict contiguous history extends before motion frame 0")
    return indices


def sample_contiguous_window_indices(
    num_samples: int,
    window_size: int,
    num_frames: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample chronological windows whose adjacent frame indices differ by one."""
    if num_samples < 0:
        raise ValueError(f"num_samples must be non-negative, got {num_samples}")
    if window_size <= 0 or window_size > num_frames:
        raise ValueError(f"window_size must be in [1, {num_frames}], got {window_size}")
    if num_samples == 0:
        return torch.empty((0, window_size), dtype=torch.long, device=device)
    max_start = num_frames - window_size
    starts = torch.randint(0, max_start + 1, (num_samples,), device=device, generator=generator)
    offsets = torch.arange(window_size, device=device)
    return starts.unsqueeze(-1) + offsets
