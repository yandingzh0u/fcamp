from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from .contracts import require_finite_tensors


# Holosoma and this repo's Isaac asset share the same moving G1 body chain, but
# a few fixed bodies are named differently or omitted by the exported motion.
_G1_MOTION_BODY_ALIASES = {
    "left_rubber_hand": "left_rubber_hand_link",
    "right_rubber_hand": "right_rubber_hand_link",
}
_G1_FIXED_BODY_OFFSETS = {
    "head_link": ("torso_link", np.array([0.0039635, 0.0, -0.044], dtype=np.float32)),
    "logo_link": ("torso_link", np.array([0.0039635, 0.0, -0.044], dtype=np.float32)),
    "left_foot_contact_point": (
        "left_ankle_roll_link",
        np.array([0.0, 0.0, -0.037], dtype=np.float32),
    ),
    "right_foot_contact_point": (
        "right_ankle_roll_link",
        np.array([0.0, 0.0, -0.037], dtype=np.float32),
    ),
    "LL_FOOT": (
        "left_ankle_roll_link",
        np.array([0.04, 0.0, -0.037], dtype=np.float32),
    ),
    "LR_FOOT": (
        "right_ankle_roll_link",
        np.array([0.04, 0.0, -0.037], dtype=np.float32),
    ),
}

_QUATERNION_NORM_TOLERANCE = 1.0e-3
_ROOT_POSE_TOLERANCE = 1.0e-5
_ROOT_VELOCITY_TOLERANCE = 1.0e-3


def mimickit_frame_delta(fps: float, control_dt: float) -> float:
    return float(fps) * float(control_dt)


def mimickit_full_motion_steps(
    start_phase: float,
    end_phase: float,
    fps: float,
    control_dt: float,
) -> int:
    span = max(0.0, float(end_phase) - float(start_phase))
    return max(1, int(math.ceil(span / mimickit_frame_delta(fps, control_dt))))


def _quat_rotate_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quat[..., 1:]
    twice_cross = 2.0 * np.cross(xyz, vector)
    return vector + quat[..., :1] * twice_cross + np.cross(xyz, twice_cross)


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def _central_angular_velocity_w(root_quat_wxyz: np.ndarray, fps: float) -> np.ndarray:
    previous = root_quat_wxyz[:-2]
    following = root_quat_wxyz[2:]
    previous_inverse = previous.copy()
    previous_inverse[..., 1:] *= -1.0
    delta = _quat_mul_wxyz(following, previous_inverse)
    delta /= np.linalg.norm(delta, axis=-1, keepdims=True)
    delta *= np.where(delta[..., :1] < 0.0, -1.0, 1.0)
    vector_norm = np.linalg.norm(delta[..., 1:], axis=-1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(delta[..., 0], -1.0, 1.0))
    axis = delta[..., 1:] / np.maximum(vector_norm[..., None], 1.0e-12)
    return axis * angle[..., None] * (float(fps) / 2.0)


def _validate_motion_contract(
    *,
    joint_pos_raw: np.ndarray,
    joint_vel_raw: np.ndarray,
    body_pos_raw: np.ndarray,
    body_quat_raw: np.ndarray,
    motion_joint_names: list[str],
    motion_body_names: list[str],
    root_body_name: str,
    fps: float,
) -> int:
    num_frames = int(joint_pos_raw.shape[0]) if joint_pos_raw.ndim > 0 else 0
    num_joints = len(motion_joint_names)
    num_bodies = len(motion_body_names)
    expected_shapes = {
        "joint_pos": (num_frames, num_joints + 7),
        "joint_vel": (num_frames, num_joints + 6),
        "body_pos_w": (num_frames, num_bodies, 3),
        "body_quat_w": (num_frames, num_bodies, 4),
    }
    actual = {
        "joint_pos": joint_pos_raw.shape,
        "joint_vel": joint_vel_raw.shape,
        "body_pos_w": body_pos_raw.shape,
        "body_quat_w": body_quat_raw.shape,
    }
    if num_frames < 3:
        raise ValueError(
            f"Motion must contain at least three frames for velocity validation, got {num_frames}"
        )
    for name, expected in expected_shapes.items():
        if actual[name] != expected:
            raise ValueError(
                f"Motion array {name!r} must have shape {expected}, got {actual[name]}"
            )
    for name, value in (
        ("joint_pos", joint_pos_raw),
        ("joint_vel", joint_vel_raw),
        ("body_pos_w", body_pos_raw),
        ("body_quat_w", body_quat_raw),
    ):
        if not np.isfinite(value).all():
            raise ValueError(f"Motion array {name!r} contains non-finite values")

    quat_norm_error = float(
        np.max(np.abs(np.linalg.norm(body_quat_raw, axis=-1) - 1.0))
    )
    if quat_norm_error > _QUATERNION_NORM_TOLERANCE:
        raise ValueError(
            "Motion body quaternions are not unit length; "
            f"max norm error={quat_norm_error:.6g}"
        )

    root_body_index = motion_body_names.index(root_body_name)
    root_pos = body_pos_raw[:, root_body_index]
    root_quat = body_quat_raw[:, root_body_index]
    root_pos_error = float(np.max(np.abs(joint_pos_raw[:, :3] - root_pos)))
    quat_direct_error = np.max(np.abs(joint_pos_raw[:, 3:7] - root_quat), axis=-1)
    quat_negated_error = np.max(np.abs(joint_pos_raw[:, 3:7] + root_quat), axis=-1)
    root_quat_error = float(np.max(np.minimum(quat_direct_error, quat_negated_error)))
    if max(root_pos_error, root_quat_error) > _ROOT_POSE_TOLERANCE:
        raise ValueError(
            "Motion root DOF pose is inconsistent with the pelvis body pose; "
            f"position error={root_pos_error:.6g}, quaternion error={root_quat_error:.6g}"
        )

    # The Holosoma joint_vel prefix is the authoritative root-link velocity:
    # [world linear xyz, world angular xyz]. It must never be rotated at load,
    # reset, or evaluation time.
    expected_linear = (root_pos[2:] - root_pos[:-2]) * (float(fps) / 2.0)
    expected_angular = _central_angular_velocity_w(root_quat, fps)
    linear_error = float(
        np.max(np.abs(joint_vel_raw[1:-1, :3] - expected_linear))
    )
    angular_error = float(
        np.max(np.abs(joint_vel_raw[1:-1, 3:6] - expected_angular))
    )
    if max(linear_error, angular_error) > _ROOT_VELOCITY_TOLERANCE:
        raise ValueError(
            "Motion joint_vel root-link world velocity is inconsistent with "
            "the centered pelvis pose difference; "
            f"linear error={linear_error:.6g}, angular error={angular_error:.6g}, "
            f"tolerance={_ROOT_VELOCITY_TOLERANCE:.6g}"
        )
    return root_body_index


class MimicMotionReference:
    def __init__(
        self,
        motion_file: str | Path,
        track_body_ids: torch.Tensor,
        anchor_body_id: int,
        device: torch.device,
        robot_body_names: list[str],
        action_joint_names: list[str],
        root_body_name: str,
    ) -> None:
        motion_file = Path(motion_file)
        if not motion_file.is_file():
            raise FileNotFoundError(f"Motion file not found: {motion_file}")

        # body_lin_vel_w/body_ang_vel_w are intentionally absent from this
        # contract. They are neither required nor read, so missing or NaN legacy
        # arrays cannot affect fixed-reward training.
        required_keys = {
            "fps",
            "joint_names",
            "body_names",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
        }
        with np.load(motion_file, allow_pickle=True) as data:
            missing_keys = required_keys - set(data.files)
            if missing_keys:
                raise ValueError(
                    f"Motion is missing required arrays: {sorted(missing_keys)}"
                )
            fps_values = np.asarray(data["fps"], dtype=np.float64).reshape(-1)
            if (
                fps_values.size != 1
                or not np.isfinite(fps_values[0])
                or fps_values[0] <= 0.0
            ):
                raise ValueError(
                    f"Motion fps must be one positive finite scalar, got {fps_values}"
                )
            self.fps = float(fps_values[0])
            motion_joint_names = [str(name) for name in np.asarray(data["joint_names"])]
            motion_body_names = [str(name) for name in np.asarray(data["body_names"])]
            if len(set(motion_joint_names)) != len(motion_joint_names):
                raise ValueError("Motion joint_names contains duplicates")
            if len(set(motion_body_names)) != len(motion_body_names):
                raise ValueError("Motion body_names contains duplicates")
            missing_joints = sorted(set(action_joint_names) - set(motion_joint_names))
            if missing_joints:
                raise ValueError(f"Motion is missing action joints: {missing_joints}")
            if root_body_name not in motion_body_names:
                raise ValueError(f"Motion is missing root body {root_body_name!r}")

            joint_pos_raw = np.asarray(data["joint_pos"], dtype=np.float32)
            joint_vel_raw = np.asarray(data["joint_vel"], dtype=np.float32)
            body_pos_raw = np.asarray(data["body_pos_w"], dtype=np.float32)
            body_quat_raw = np.asarray(data["body_quat_w"], dtype=np.float32)
            motion_root_body_index = _validate_motion_contract(
                joint_pos_raw=joint_pos_raw,
                joint_vel_raw=joint_vel_raw,
                body_pos_raw=body_pos_raw,
                body_quat_raw=body_quat_raw,
                motion_joint_names=motion_joint_names,
                motion_body_names=motion_body_names,
                root_body_name=root_body_name,
                fps=self.fps,
            )

            action_indices = [
                motion_joint_names.index(name) for name in action_joint_names
            ]
            joint_pos = joint_pos_raw[:, 7:][:, action_indices]
            joint_vel = joint_vel_raw[:, 6:][:, action_indices]
            root_link_velocity = joint_vel_raw[:, :6].copy()
            body_pos, body_quat, available_motion_bodies = self._map_bodies(
                body_pos_raw=body_pos_raw,
                body_quat_raw=body_quat_raw,
                motion_body_names=motion_body_names,
                robot_body_names=robot_body_names,
            )

        self.device = device
        self.joint_pos = torch.as_tensor(
            joint_pos, dtype=torch.float32, device=device
        )
        self.joint_vel = torch.as_tensor(
            joint_vel, dtype=torch.float32, device=device
        )
        self.root_link_velocity_w = torch.as_tensor(
            root_link_velocity, dtype=torch.float32, device=device
        )
        self.body_pos_full_w = torch.as_tensor(
            body_pos, dtype=torch.float32, device=device
        )
        self.body_quat_full_w = torch.as_tensor(
            body_quat, dtype=torch.float32, device=device
        )
        self.track_body_ids = track_body_ids.to(device=device, dtype=torch.long)
        self.anchor_body_id = int(anchor_body_id)
        self.root_body_id = robot_body_names.index(root_body_name)
        required_motion_bodies = {
            robot_body_names[int(index)]
            for index in self.track_body_ids.detach().cpu().tolist()
        }
        required_motion_bodies.add(root_body_name)
        missing_motion_bodies = sorted(
            required_motion_bodies - available_motion_bodies
        )
        if missing_motion_bodies:
            raise ValueError(
                "Required tracking bodies are missing from the motion and cannot "
                f"be reconstructed: {missing_motion_bodies}"
            )
        # The mapped root must be the exact motion pelvis validated above.
        mapped_root_pos = body_pos[:, self.root_body_id]
        if not np.array_equal(
            mapped_root_pos, body_pos_raw[:, motion_root_body_index]
        ):
            raise RuntimeError("Mapped root body no longer matches validated pelvis")
        self.num_frames = int(self.joint_pos.shape[0])

    @staticmethod
    def _map_bodies(
        *,
        body_pos_raw: np.ndarray,
        body_quat_raw: np.ndarray,
        motion_body_names: list[str],
        robot_body_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray, set[str]]:
        num_frames = int(body_pos_raw.shape[0])
        num_robot_bodies = len(robot_body_names)
        body_pos = np.zeros(
            (num_frames, num_robot_bodies, 3), dtype=np.float32
        )
        body_quat = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (num_frames, num_robot_bodies, 1),
        )
        available_motion_bodies: set[str] = set()
        for robot_index, name in enumerate(robot_body_names):
            motion_name = (
                name if name in motion_body_names else _G1_MOTION_BODY_ALIASES.get(name)
            )
            if motion_name in motion_body_names:
                motion_index = motion_body_names.index(motion_name)
                body_pos[:, robot_index] = body_pos_raw[:, motion_index]
                body_quat[:, robot_index] = body_quat_raw[:, motion_index]
                available_motion_bodies.add(name)

        for child_name, (parent_name, offset_parent) in _G1_FIXED_BODY_OFFSETS.items():
            if (
                child_name not in robot_body_names
                or child_name in available_motion_bodies
                or parent_name not in available_motion_bodies
            ):
                continue
            parent_index = robot_body_names.index(parent_name)
            child_index = robot_body_names.index(child_name)
            parent_quat = body_quat[:, parent_index]
            local_offset = np.broadcast_to(offset_parent, (num_frames, 3))
            body_pos[:, child_index] = (
                body_pos[:, parent_index]
                + _quat_rotate_wxyz(parent_quat, local_offset)
            )
            body_quat[:, child_index] = parent_quat
            available_motion_bodies.add(child_name)
        return body_pos, body_quat, available_motion_bodies

    def clamp_time_steps(self, time_steps: torch.Tensor) -> torch.Tensor:
        if torch.is_floating_point(time_steps) and not bool(
            torch.isfinite(time_steps).all()
        ):
            raise ValueError("Motion time steps contain non-finite values")
        return torch.clamp(time_steps, min=0, max=self.num_frames - 1)

    def _interpolate(
        self, values: torch.Tensor, time_steps: torch.Tensor
    ) -> torch.Tensor:
        if not torch.is_floating_point(time_steps):
            return values[time_steps]
        time = self.clamp_time_steps(
            time_steps.to(device=self.device, dtype=torch.float32)
        )
        lower = torch.floor(time).to(dtype=torch.long)
        upper = torch.clamp(lower + 1, max=self.num_frames - 1)
        weight = (time - lower.to(dtype=time.dtype)).view(
            -1, *([1] * (values.ndim - 1))
        )
        return (
            values.index_select(0, lower) * (1.0 - weight)
            + values.index_select(0, upper) * weight
        )

    def _interpolate_quat(
        self, values: torch.Tensor, time_steps: torch.Tensor
    ) -> torch.Tensor:
        if not torch.is_floating_point(time_steps):
            return values[time_steps]
        time = self.clamp_time_steps(
            time_steps.to(device=self.device, dtype=torch.float32)
        )
        lower = torch.floor(time).to(dtype=torch.long)
        upper = torch.clamp(lower + 1, max=self.num_frames - 1)
        q0 = values.index_select(0, lower)
        q1 = values.index_select(0, upper)
        q1 = q1 * torch.where(
            (q0 * q1).sum(dim=-1, keepdim=True) < 0.0, -1.0, 1.0
        )
        weight = (time - lower.to(dtype=time.dtype)).view(
            -1, *([1] * (values.ndim - 1))
        )
        return torch.nn.functional.normalize(
            q0 * (1.0 - weight) + q1 * weight, dim=-1
        )

    def get_frame(self, time_steps: torch.Tensor) -> dict[str, torch.Tensor]:
        time_steps = self.clamp_time_steps(time_steps.to(device=self.device))
        body_pos = self._interpolate(self.body_pos_full_w, time_steps)
        body_quat = self._interpolate_quat(self.body_quat_full_w, time_steps)
        root_velocity = self._interpolate(self.root_link_velocity_w, time_steps)
        frame = {
            "joint_pos": self._interpolate(self.joint_pos, time_steps),
            "joint_vel": self._interpolate(self.joint_vel, time_steps),
            "body_pos_w": body_pos[:, self.track_body_ids],
            "body_quat_w": body_quat[:, self.track_body_ids],
            "anchor_pos_w": body_pos[:, self.anchor_body_id],
            "anchor_quat_w": body_quat[:, self.anchor_body_id],
            "root_pos_w": body_pos[:, self.root_body_id],
            "root_quat_w": body_quat[:, self.root_body_id],
            "root_lin_vel_w": root_velocity[:, :3],
            "root_ang_vel_w": root_velocity[:, 3:],
        }
        require_finite_tensors(frame, context="Interpolated motion frame")
        return frame
