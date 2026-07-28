"""Pure-PyTorch construction of MimicKit-style imitation observations.

The builder deliberately accepts the same physical fields for policy and demo
data.  Reference targets, contacts, actions, phase and motion IDs are not part
of the schema, preventing accidental information leakage into the discriminator.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ImitationFeatureSchema:
    """Description of one imitation state."""

    num_joints: int = 29
    num_key_bodies: int = 5

    def __post_init__(self) -> None:
        if self.num_joints <= 0:
            raise ValueError("num_joints must be positive")
        if self.num_key_bodies < 0:
            raise ValueError("num_key_bodies must be non-negative")

    @property
    def frame_dim(self) -> int:
        # root xyz + root rot6d + joint rot6d + key xyz + root v/w + dof v
        return 3 + 6 + 6 * self.num_joints + 3 * self.num_key_bodies + 3 + 3 + self.num_joints

def canonicalize_imitation_window(frames: torch.Tensor) -> torch.Tensor:
    """Apply MimicKit's G1 imitation root-translation convention to a window.

    ``frames`` is chronological with shape ``[..., W, F]`` and stores root xyz
    in features ``0:3``. MimicKit keeps global orientations and velocities for
    G1, but expresses every root x/y relative to the newest root position.
    Root z remains the absolute height (``root_height_obs=True``).

    This belongs at the window boundary: the causal ring must retain raw root
    positions so every appended frame can re-anchor all preceding frames.
    """

    if frames.ndim < 2:
        raise ValueError(f"imitation windows must have shape [..., W, F], got {tuple(frames.shape)}")
    if frames.shape[-2] <= 0:
        raise ValueError("imitation windows must contain at least one frame")
    if frames.shape[-1] < 3:
        raise ValueError("imitation frames must store root xyz in features 0:3")
    canonical = frames.clone()
    canonical[..., :2] = frames[..., :2] - frames[..., -1:, :2]
    return canonical


def _check_last_dim(name: str, value: torch.Tensor, size: int) -> None:
    if value.shape[-1] != size:
        raise ValueError(f"{name} must end in dimension {size}, got {tuple(value.shape)}")


def quat_to_matrix(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """Convert scalar-first unit quaternions to rotation matrices.

    Inputs are normalized here so policy simulator state and interpolated demo
    state follow exactly the same numerical path.
    """

    _check_last_dim("quat_wxyz", quat_wxyz, 4)
    q = torch.nn.functional.normalize(quat_wxyz, dim=-1, eps=1.0e-8)
    w, x, y, z = q.unbind(dim=-1)
    two = 2.0
    matrix = torch.stack(
        (
            1.0 - two * (y * y + z * z),
            two * (x * y - z * w),
            two * (x * z + y * w),
            two * (x * y + z * w),
            1.0 - two * (x * x + z * z),
            two * (y * z - x * w),
            two * (x * z - y * w),
            two * (y * z + x * w),
            1.0 - two * (x * x + y * y),
        ),
        dim=-1,
    )
    return matrix.reshape(q.shape[:-1] + (3, 3))


def quat_to_rot6d(quat_wxyz: torch.Tensor) -> torch.Tensor:
    """MimicKit 6-D tangent/normal rotation (matrix columns x and z)."""

    matrix = quat_to_matrix(quat_wxyz)
    # MimicKit's quat_to_tan_norm concatenates the rotated x (tangent) and z
    # (normal) basis vectors. Keep each vector contiguous.
    tangent = matrix[..., :, 0]
    normal = matrix[..., :, 2]
    return torch.cat((tangent, normal), dim=-1)


def revolute_dof_to_quat(dof_pos: torch.Tensor, joint_axes: torch.Tensor) -> torch.Tensor:
    """Convert revolute DOF angles and fixed xyz axes to scalar-first quats.

    ``dof_pos`` has shape ``[..., J]``. ``joint_axes`` may be ``[J, 3]`` or
    broadcastable to ``[..., J, 3]``. This is useful when the motion file stores
    joint angles rather than joint quaternions.
    """

    _check_last_dim("joint_axes", joint_axes, 3)
    if joint_axes.shape[-2] != dof_pos.shape[-1]:
        raise ValueError(
            f"joint_axes has {joint_axes.shape[-2]} joints, dof_pos has {dof_pos.shape[-1]}"
        )
    axes = torch.nn.functional.normalize(joint_axes, dim=-1, eps=1.0e-8)
    half = 0.5 * dof_pos
    xyz = axes * torch.sin(half).unsqueeze(-1)
    return torch.cat((torch.cos(half).unsqueeze(-1), xyz), dim=-1)


def build_imitation_frame(
    *,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    joint_rotation: torch.Tensor,
    key_body_pos: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    dof_vel: torch.Tensor,
    schema: ImitationFeatureSchema,
) -> torch.Tensor:
    """Build one or more imitation frames with a common policy/demo code path.

    All tensors share leading dimensions (for example ``[N]`` or ``[B, W]``).
    Quaternion inputs use Isaac Lab's scalar-first ``wxyz`` convention.  World
    positions should already have per-environment scene origins removed.  Key
    positions are root-relative by default, matching MimicKit's G1 imitation feature.
    """

    _check_last_dim("root_pos", root_pos, 3)
    _check_last_dim("root_quat", root_quat, 4)
    _check_last_dim("root_lin_vel", root_lin_vel, 3)
    _check_last_dim("root_ang_vel", root_ang_vel, 3)
    _check_last_dim("dof_vel", dof_vel, schema.num_joints)
    if key_body_pos.shape[-2:] != (schema.num_key_bodies, 3):
        raise ValueError(
            "key_body_pos must end in "
            f"({schema.num_key_bodies}, 3), got {tuple(key_body_pos.shape)}"
        )

    if joint_rotation.shape[-2:] != (schema.num_joints, 4):
        raise ValueError(
            "joint_rotation quaternion input must end in "
            f"({schema.num_joints}, 4), got {tuple(joint_rotation.shape)}"
        )
    joint_rot6d = quat_to_rot6d(joint_rotation)

    leading = root_pos.shape[:-1]
    named = {
        "root_quat": root_quat.shape[:-1],
        "joint_rotation": joint_rotation.shape[:-2],
        "key_body_pos": key_body_pos.shape[:-2],
        "root_lin_vel": root_lin_vel.shape[:-1],
        "root_ang_vel": root_ang_vel.shape[:-1],
        "dof_vel": dof_vel.shape[:-1],
    }
    mismatched = {name: shape for name, shape in named.items() if shape != leading}
    if mismatched:
        raise ValueError(f"imitation feature leading dimensions must match {leading}; got {mismatched}")

    key_body_pos = key_body_pos - root_pos.unsqueeze(-2)

    pieces = (
        root_pos,
        quat_to_rot6d(root_quat),
        joint_rot6d.reshape(leading + (6 * schema.num_joints,)),
        key_body_pos.reshape(leading + (3 * schema.num_key_bodies,)),
        root_lin_vel,
        root_ang_vel,
        dof_vel,
    )
    frame = torch.cat(pieces, dim=-1)
    if frame.shape[-1] != schema.frame_dim:
        raise RuntimeError(f"internal imitation feature size error: {frame.shape[-1]} != {schema.frame_dim}")
    return frame
