"""Exact AMP frame adapter for source commit 6901e302.

This module intentionally lives in diagnostics instead of changing the
current fixed-reward environment.  It restores the three scientifically
distinct code paths required by the discovery suite:

* the standard AMP policy negative built from PhysX body readback;
* the expert positive built with the checked-in URDF FK model; and
* a same-state URDF-FK reconstruction used only to audit the domain gap.

All returned frames are raw scene-local 239-D frames.  Root-x/y
canonicalization belongs at a downstream temporal-window boundary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from envs.imitation_data import (
    G1_IMITATION_FRAME_DIM,
    G1_IMITATION_JOINT_AXES,
    G1_IMITATION_KEY_BODY_NAMES,
    G1_IMITATION_NUM_JOINTS,
    build_g1_imitation_frame,
)


IMITATION_SOURCE_COMMIT = "6901e302499711e2207687e1342348a4078330f8"
IMITATION_FRAME_SCHEMA = "g1_amp_imitation_frame_v1"
IMITATION_ROOT_XY_CANONICALIZATION = "raw_until_window_newest_anchor"
IMITATION_FRAME_DIM = G1_IMITATION_FRAME_DIM
IMITATION_KEY_BODY_NAMES = G1_IMITATION_KEY_BODY_NAMES

# This is exactly the action order used by commit 6901 and by the current G1
# asset.  Keeping it local avoids importing Isaac-only robot configuration in
# pure unit tests.
G1_ACTION_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

_FRAME_SCHEMA_PAYLOAD = {
    "schema": IMITATION_FRAME_SCHEMA,
    "frame_dim": IMITATION_FRAME_DIM,
    "feature_order": [
        "root_pos_xyz",
        "root_rot6d_tangent_normal",
        "30_joint_rot6d_tangent_normal",
        "5_key_body_root_relative_xyz",
        "root_linear_velocity_world",
        "root_angular_velocity_world",
        "29_joint_velocity",
    ],
    "joint_axes": G1_IMITATION_JOINT_AXES,
    "fixed_head_joint_rotation_index": 15,
    "key_body_names": IMITATION_KEY_BODY_NAMES,
    "root_xy_canonicalization": IMITATION_ROOT_XY_CANONICALIZATION,
}
IMITATION_FRAME_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(_FRAME_SCHEMA_PAYLOAD, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
).hexdigest()


def imitation_contract_metadata() -> dict[str, Any]:
    return {
        "imitation_source_commit": IMITATION_SOURCE_COMMIT,
        "imitation_frame_schema": IMITATION_FRAME_SCHEMA,
        "imitation_frame_schema_sha256": IMITATION_FRAME_SCHEMA_SHA256,
        "imitation_frame_dim": IMITATION_FRAME_DIM,
        "imitation_key_body_names": list(IMITATION_KEY_BODY_NAMES),
        "imitation_root_xy_canonicalization": IMITATION_ROOT_XY_CANONICALIZATION,
        "imitation_agent_negative_field": "agent_physx_raw_frame",
        "imitation_reference_positive_field": "reference_expert_raw_frame",
        "imitation_fk_aligned_role": "same_state_domain_gap_audit_only",
        "imitation_phase_field": "phase_normalized_pre_step",
    }


def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(dim=-1)
    bw, bx, by, bz = b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def _quat_rotate_wxyz_torch(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., 1:]
    twice_cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + quat[..., :1] * twice_cross + torch.cross(
        xyz, twice_cross, dim=-1
    )


def _axis_angle_quat_wxyz(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    half = 0.5 * angle
    sin_half = torch.sin(half).unsqueeze(-1)
    axis = F.normalize(axis.to(device=angle.device, dtype=angle.dtype), dim=-1)
    return torch.cat(
        (torch.cos(half).unsqueeze(-1), axis.unsqueeze(0) * sin_half), dim=-1
    )


def _euler_xyz_to_quat_np(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = (0.5 * rpy).tolist()
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ),
        dtype=np.float32,
    )


class _URDFFKModel:
    """Torch FK model copied from commit 6901's expert-data path."""

    def __init__(
        self,
        urdf_file: str | Path,
        *,
        robot_body_names: list[str],
        action_joint_names: list[str],
        device: torch.device,
    ) -> None:
        self.device = device
        root = ET.parse(urdf_file).getroot()
        joints = root.findall("joint")
        links = root.findall("link")
        child_names = {joint.find("child").attrib["link"] for joint in joints}
        root_name = next(
            (link.attrib["name"] for link in links if link.attrib["name"] not in child_names),
            None,
        )
        if root_name is None:
            raise ValueError(f"URDF has no root link: {urdf_file}")

        joint_by_child = {joint.find("child").attrib["link"]: joint for joint in joints}
        children_by_parent: dict[str, list[str]] = {}
        for joint in joints:
            parent = joint.find("parent").attrib["link"]
            child = joint.find("child").attrib["link"]
            children_by_parent.setdefault(parent, []).append(child)

        body_names: list[str] = []
        parent_indices: list[int] = []
        local_pos: list[np.ndarray] = []
        local_quat: list[np.ndarray] = []
        joint_indices: list[int] = []
        joint_axes: list[np.ndarray] = []
        name_to_index: dict[str, int] = {}
        action_index = {name: idx for idx, name in enumerate(action_joint_names)}

        def add_link(name: str, parent_index: int) -> None:
            index = len(body_names)
            name_to_index[name] = index
            body_names.append(name)
            parent_indices.append(parent_index)
            if parent_index < 0:
                local_pos.append(np.zeros(3, dtype=np.float32))
                local_quat.append(
                    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                )
                joint_indices.append(-1)
                joint_axes.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))
            else:
                joint = joint_by_child[name]
                origin = joint.find("origin")
                xyz = np.zeros(3, dtype=np.float32)
                rpy = np.zeros(3, dtype=np.float32)
                if origin is not None:
                    if origin.attrib.get("xyz") is not None:
                        xyz = np.fromstring(origin.attrib["xyz"], dtype=np.float32, sep=" ")
                    if origin.attrib.get("rpy") is not None:
                        rpy = np.fromstring(origin.attrib["rpy"], dtype=np.float32, sep=" ")
                local_pos.append(xyz)
                local_quat.append(_euler_xyz_to_quat_np(rpy))
                if joint.attrib.get("type") == "fixed":
                    joint_indices.append(-1)
                    joint_axes.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))
                else:
                    joint_name = joint.attrib["name"]
                    if joint_name not in action_index:
                        raise ValueError(
                            f"URDF joint {joint_name} is not in the action joint list"
                        )
                    axis_node = joint.find("axis")
                    axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    if axis_node is not None and axis_node.attrib.get("xyz") is not None:
                        axis = np.fromstring(
                            axis_node.attrib["xyz"], dtype=np.float32, sep=" "
                        )
                    joint_indices.append(action_index[joint_name])
                    joint_axes.append(axis)
            for child in children_by_parent.get(name, ()):
                add_link(child, index)

        add_link(root_name, -1)
        missing = [name for name in robot_body_names if name not in name_to_index]
        if missing:
            raise ValueError(f"URDF FK model is missing robot bodies: {missing}")

        self.robot_body_gather = torch.tensor(
            [name_to_index[name] for name in robot_body_names],
            dtype=torch.long,
            device=device,
        )
        self.parent_indices = parent_indices
        self.local_pos = torch.tensor(
            np.asarray(local_pos), dtype=torch.float32, device=device
        )
        self.local_quat = torch.tensor(
            np.asarray(local_quat), dtype=torch.float32, device=device
        )
        self.joint_indices = joint_indices
        self.joint_axes = torch.tensor(
            np.asarray(joint_axes), dtype=torch.float32, device=device
        )

    def body_pos(
        self,
        *,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        joint_pos: torch.Tensor,
    ) -> torch.Tensor:
        count = int(root_pos.shape[0])
        link_pos: list[torch.Tensor] = []
        link_quat: list[torch.Tensor] = []
        for link_id, parent_id in enumerate(self.parent_indices):
            if parent_id < 0:
                link_pos.append(root_pos)
                link_quat.append(F.normalize(root_quat, dim=-1))
                continue
            parent_pos = link_pos[parent_id]
            parent_quat = link_quat[parent_id]
            offset = self.local_pos[link_id].expand(count, 3)
            origin_quat = self.local_quat[link_id].expand(count, 4)
            joint_idx = self.joint_indices[link_id]
            if joint_idx >= 0:
                joint_quat = _axis_angle_quat_wxyz(
                    self.joint_axes[link_id], joint_pos[:, joint_idx]
                )
                local_quat = _quat_mul_wxyz(origin_quat, joint_quat)
            else:
                local_quat = origin_quat
            link_pos.append(
                parent_pos + _quat_rotate_wxyz_torch(parent_quat, offset)
            )
            link_quat.append(
                F.normalize(_quat_mul_wxyz(parent_quat, local_quat), dim=-1)
            )
        all_pos = torch.stack(link_pos, dim=1)
        return all_pos.index_select(1, self.robot_body_gather)


def _interpolate(values: torch.Tensor, phases: torch.Tensor, num_frames: int) -> torch.Tensor:
    lower = torch.floor(phases).to(torch.long)
    upper = torch.clamp(lower + 1, max=num_frames - 1)
    weight = (phases - lower.to(phases.dtype)).view(
        -1, *([1] * (values.ndim - 1))
    )
    return values.index_select(0, lower) * (1.0 - weight) + values.index_select(
        0, upper
    ) * weight


def _interpolate_quat_shortest(
    values: torch.Tensor, phases: torch.Tensor, num_frames: int
) -> torch.Tensor:
    lower = torch.floor(phases).to(torch.long)
    upper = torch.clamp(lower + 1, max=num_frames - 1)
    q0 = F.normalize(values.index_select(0, lower), dim=-1)
    q1 = F.normalize(values.index_select(0, upper), dim=-1)
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    weight = (phases - lower.to(phases.dtype)).view(
        -1, *([1] * (values.ndim - 1))
    )
    slerp = (
        torch.sin((1.0 - weight) * theta) / sin_theta.clamp_min(1.0e-8) * q0
        + torch.sin(weight * theta) / sin_theta.clamp_min(1.0e-8) * q1
    )
    linear = q0 * (1.0 - weight) + q1 * weight
    return F.normalize(torch.where(sin_theta > 1.0e-6, slerp, linear), dim=-1)


class Commit6901ImitationAdapter:
    """Build commit-6901 AMP fields from a current G1 environment."""

    def __init__(self, env: Any, *, repo_root: str | Path) -> None:
        self.env = env
        if int(env.action_dim) != G1_IMITATION_NUM_JOINTS:
            raise ValueError(
                f"commit-6901 AMP requires {G1_IMITATION_NUM_JOINTS} action joints"
            )
        body_names = list(env.robot.body_names)
        missing = [name for name in IMITATION_KEY_BODY_NAMES if name not in body_names]
        if missing:
            raise ValueError(f"G1 robot is missing AMP key bodies: {missing}")
        self.key_body_ids = torch.tensor(
            [body_names.index(name) for name in IMITATION_KEY_BODY_NAMES],
            dtype=torch.long,
            device=env.device,
        )
        urdf = (
            Path(repo_root).expanduser().resolve()
            / "assets/robots/holosoma_g1/g1_29dof.urdf"
        )
        if not urdf.is_file():
            raise FileNotFoundError(f"commit-6901 FK URDF is missing: {urdf}")
        self.fk = _URDFFKModel(
            urdf,
            robot_body_names=body_names,
            action_joint_names=list(G1_ACTION_JOINT_NAMES),
            device=torch.device(env.device),
        )

    @staticmethod
    def _validate_frame(name: str, frame: torch.Tensor) -> torch.Tensor:
        if frame.ndim != 2 or frame.shape[-1] != IMITATION_FRAME_DIM:
            raise RuntimeError(
                f"{name} must have shape [N,{IMITATION_FRAME_DIM}], got {tuple(frame.shape)}"
            )
        if not bool(torch.isfinite(frame).all()):
            raise FloatingPointError(f"{name} contains NaN or Inf")
        return frame

    def _build_fk_frame(
        self,
        *,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        joint_pos: torch.Tensor,
        root_velocity: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> torch.Tensor:
        fk_body_pos = self.fk.body_pos(
            root_pos=root_pos,
            root_quat=root_quat,
            joint_pos=joint_pos,
        )
        return build_g1_imitation_frame(
            root_pos=root_pos,
            root_quat_wxyz=root_quat,
            joint_pos=joint_pos,
            key_body_pos=fk_body_pos.index_select(1, self.key_body_ids),
            root_lin_vel=root_velocity[:, :3],
            root_ang_vel=root_velocity[:, 3:],
            joint_vel=joint_vel,
        )

    def agent_physx_raw_frame(self) -> torch.Tensor:
        env = self.env
        joint_pos, joint_vel = env.get_action_joint_state()
        origins = env.scene.env_origins
        root_pos = env.robot.data.root_link_pos_w - origins
        key_body_pos = env.robot.data.body_pos_w.index_select(1, self.key_body_ids)
        key_body_pos = key_body_pos - origins.unsqueeze(1)
        frame = build_g1_imitation_frame(
            root_pos=root_pos,
            root_quat_wxyz=env.robot.data.root_link_quat_w,
            joint_pos=joint_pos,
            key_body_pos=key_body_pos,
            root_lin_vel=env.robot.data.root_link_vel_w[:, :3],
            root_ang_vel=env.robot.data.root_link_vel_w[:, 3:],
            joint_vel=joint_vel,
        )
        return self._validate_frame("agent_physx_raw_frame", frame)

    def agent_fk_aligned_raw_frame(self) -> torch.Tensor:
        env = self.env
        joint_pos, joint_vel = env.get_action_joint_state()
        frame = self._build_fk_frame(
            root_pos=env.robot.data.root_link_pos_w - env.scene.env_origins,
            root_quat=env.robot.data.root_link_quat_w,
            joint_pos=joint_pos,
            root_velocity=env.robot.data.root_link_vel_w,
            joint_vel=joint_vel,
        )
        return self._validate_frame("agent_fk_aligned_raw_frame", frame)

    def reference_expert_raw_frame(self, phases: torch.Tensor) -> torch.Tensor:
        motion = self.env.motion
        phases = phases.to(device=self.env.device, dtype=torch.float32).reshape(-1)
        if not bool(torch.isfinite(phases).all()):
            raise ValueError("expert phases contain NaN or Inf")
        if bool((phases < 0).any()) or bool((phases > motion.num_frames - 1).any()):
            raise ValueError(
                f"expert phases must lie in [0,{motion.num_frames - 1}]"
            )
        lower = torch.floor(phases).to(torch.long)
        root_pos = _interpolate(
            motion.body_pos_full_w[:, motion.root_body_id], phases, motion.num_frames
        )
        root_quat = _interpolate_quat_shortest(
            motion.body_quat_full_w[:, motion.root_body_id], phases, motion.num_frames
        )
        joint_pos = _interpolate(motion.joint_pos, phases, motion.num_frames)
        frame = self._build_fk_frame(
            root_pos=root_pos,
            root_quat=root_quat,
            joint_pos=joint_pos,
            root_velocity=motion.root_link_velocity_w.index_select(0, lower),
            joint_vel=motion.joint_vel.index_select(0, lower),
        )
        return self._validate_frame("reference_expert_raw_frame", frame)

    def phase_normalized_pre_step(self, phases: torch.Tensor) -> torch.Tensor:
        start = float(self.env.motion_start_phase)
        end = float(self.env.motion_end_phase)
        if end <= start:
            raise ValueError("motion phase interval cannot be normalized")
        normalized = (phases.to(torch.float32) - start) / (end - start)
        if bool((normalized < -1.0e-6).any() or (normalized > 1.0 + 1.0e-6).any()):
            raise ValueError("pre-step phase lies outside configured motion interval")
        return normalized.clamp(0.0, 1.0)

    def record(self, phases: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "agent_physx_raw_frame": self.agent_physx_raw_frame(),
            "agent_fk_aligned_raw_frame": self.agent_fk_aligned_raw_frame(),
            "reference_expert_raw_frame": self.reference_expert_raw_frame(phases),
            "phase_normalized_pre_step": self.phase_normalized_pre_step(phases),
        }


__all__ = [
    "Commit6901ImitationAdapter",
    "G1_ACTION_JOINT_NAMES",
    "IMITATION_FRAME_DIM",
    "IMITATION_FRAME_SCHEMA",
    "IMITATION_FRAME_SCHEMA_SHA256",
    "IMITATION_KEY_BODY_NAMES",
    "IMITATION_ROOT_XY_CANONICALIZATION",
    "IMITATION_SOURCE_COMMIT",
    "imitation_contract_metadata",
]
