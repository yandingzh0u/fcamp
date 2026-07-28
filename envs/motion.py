from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
from torch.nn import functional as F

from .imitation_data import (
    G1_IMITATION_FRAME_DIM,
    G1_IMITATION_KEY_BODY_NAMES,
    G1_IMITATION_NUM_JOINTS,
    build_g1_imitation_frame,
    history_indices,
)


# Holosoma and this repo's Isaac asset share the same moving G1 body chain, but
# a few fixed bodies are named differently or omitted by the exported motion.
# These fixed transforms are copied from the checked-in G1 URDF.
_G1_MOTION_BODY_ALIASES = {
    "left_rubber_hand": "left_rubber_hand_link",
    "right_rubber_hand": "right_rubber_hand_link",
}
_G1_FIXED_BODY_OFFSETS = {
    "head_link": ("torso_link", np.array([0.0039635, 0.0, -0.044], dtype=np.float32)),
    "logo_link": ("torso_link", np.array([0.0039635, 0.0, -0.044], dtype=np.float32)),
    "left_foot_contact_point": ("left_ankle_roll_link", np.array([0.0, 0.0, -0.037], dtype=np.float32)),
    "right_foot_contact_point": ("right_ankle_roll_link", np.array([0.0, 0.0, -0.037], dtype=np.float32)),
    "LL_FOOT": ("left_ankle_roll_link", np.array([0.04, 0.0, -0.037], dtype=np.float32)),
    "LR_FOOT": ("right_ankle_roll_link", np.array([0.04, 0.0, -0.037], dtype=np.float32)),
}

def mimickit_frame_delta(fps: float, control_dt: float) -> float:
    return float(fps) * float(control_dt)


def mimickit_full_motion_steps(start_phase: float, end_phase: float, fps: float, control_dt: float) -> int:
    span = max(0.0, float(end_phase) - float(start_phase))
    return max(1, int(math.ceil(span / mimickit_frame_delta(fps, control_dt))))


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
    t = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + quat[..., :1] * t + torch.cross(xyz, t, dim=-1)


def _axis_angle_quat_wxyz(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    half = 0.5 * angle
    sin_half = torch.sin(half).unsqueeze(-1)
    axis = F.normalize(axis.to(device=angle.device, dtype=angle.dtype), dim=-1)
    return torch.cat((torch.cos(half).unsqueeze(-1), axis.unsqueeze(0) * sin_half), dim=-1)


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


def _quat_rotate_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Vectorized quaternion rotation for ``quat[...,4]`` and ``vector[...,3]``."""

    xyz = quat[..., 1:]
    t = 2.0 * np.cross(xyz, vector)
    return vector + quat[..., :1] * t + np.cross(xyz, t)


class _URDFFKModel:
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
        root_name = next((link.attrib["name"] for link in links if link.attrib["name"] not in child_names), None)
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
            idx = len(body_names)
            name_to_index[name] = idx
            body_names.append(name)
            parent_indices.append(parent_index)
            if parent_index < 0:
                local_pos.append(np.zeros(3, dtype=np.float32))
                local_quat.append(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
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
                        raise ValueError(f"URDF joint {joint_name} is not in the action joint list")
                    axis_node = joint.find("axis")
                    axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                    if axis_node is not None and axis_node.attrib.get("xyz") is not None:
                        axis = np.fromstring(axis_node.attrib["xyz"], dtype=np.float32, sep=" ")
                    joint_indices.append(action_index[joint_name])
                    joint_axes.append(axis)
            for child in children_by_parent.get(name, ()):
                add_link(child, idx)

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
        self.local_pos = torch.tensor(np.asarray(local_pos), dtype=torch.float32, device=device)
        self.local_quat = torch.tensor(np.asarray(local_quat), dtype=torch.float32, device=device)
        self.joint_indices = joint_indices
        self.joint_axes = torch.tensor(np.asarray(joint_axes), dtype=torch.float32, device=device)

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
                joint_quat = _axis_angle_quat_wxyz(self.joint_axes[link_id], joint_pos[:, joint_idx])
                local_quat = _quat_mul_wxyz(origin_quat, joint_quat)
            else:
                local_quat = origin_quat
            link_pos.append(parent_pos + _quat_rotate_wxyz_torch(parent_quat, offset))
            link_quat.append(F.normalize(_quat_mul_wxyz(parent_quat, local_quat), dim=-1))
        all_pos = torch.stack(link_pos, dim=1)
        return all_pos.index_select(1, self.robot_body_gather)


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
        kinematic_urdf_file: str | Path,
        imitation_key_body_names: tuple[str, ...] = G1_IMITATION_KEY_BODY_NAMES,
    ):
        motion_file = Path(motion_file)
        if not motion_file.is_file():
            raise FileNotFoundError(f"Motion file not found: {motion_file}")

        data = np.load(motion_file, allow_pickle=True)
        self.device = device
        required_keys = {
            "fps",
            "joint_names",
            "body_names",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        }
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
        frame, available_motion_bodies = self._load_holosoma(
            data,
            robot_body_names,
            action_joint_names,
        )

        joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w = frame
        self.joint_pos = torch.tensor(joint_pos, dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(joint_vel, dtype=torch.float32, device=device)
        self.body_pos_full_w = torch.tensor(body_pos_w, dtype=torch.float32, device=device)
        self.body_quat_full_w = torch.tensor(body_quat_w, dtype=torch.float32, device=device)
        self.body_lin_vel_full_w = torch.tensor(body_lin_vel_w, dtype=torch.float32, device=device)
        self.body_ang_vel_full_w = torch.tensor(body_ang_vel_w, dtype=torch.float32, device=device)

        self.track_body_ids = track_body_ids.to(device=device, dtype=torch.long)
        self.anchor_body_id = int(anchor_body_id)
        self.root_body_id = robot_body_names.index(root_body_name)
        missing_imitation_bodies = [name for name in imitation_key_body_names if name not in robot_body_names]
        if missing_imitation_bodies:
            raise ValueError(f"imitation key bodies are missing from the robot asset: {missing_imitation_bodies}")
        required_motion_bodies = {
            *(robot_body_names[int(i)] for i in track_body_ids.detach().cpu().tolist()),
            *(imitation_key_body_names),
        }
        required_motion_bodies.add(root_body_name)
        missing_motion_bodies = sorted(required_motion_bodies - available_motion_bodies)
        if missing_motion_bodies:
            raise ValueError(
                "Required policy/imitation bodies are missing from the motion and cannot be "
                f"reconstructed: {missing_motion_bodies}"
            )
        self.imitation_key_body_ids = torch.tensor(
            [robot_body_names.index(name) for name in imitation_key_body_names],
            dtype=torch.long,
            device=device,
        )
        self._fk_model = _URDFFKModel(
            kinematic_urdf_file,
            robot_body_names=robot_body_names,
            action_joint_names=action_joint_names,
            device=device,
        )
        self._fcamp_expert_integer_frame_cache: torch.Tensor | None = None
        self.num_frames = int(self.joint_pos.shape[0])

    def _load_holosoma(self, data, robot_body_names, action_joint_names):
        motion_joint_names = [str(n) for n in data["joint_names"]]
        motion_body_names = [str(n) for n in data["body_names"]]


        joint_pos_raw = np.asarray(data["joint_pos"], dtype=np.float32)
        joint_vel_raw = np.asarray(data["joint_vel"], dtype=np.float32)
        num_joints = len(motion_joint_names)
        joint_pos_joints = joint_pos_raw[:, joint_pos_raw.shape[1] - num_joints:]
        joint_vel_joints = joint_vel_raw[:, joint_vel_raw.shape[1] - num_joints:]
        j_idx = [motion_joint_names.index(n) for n in action_joint_names]
        joint_pos = joint_pos_joints[:, j_idx]
        joint_vel = joint_vel_joints[:, j_idx]


        body_pos_raw = np.asarray(data["body_pos_w"], dtype=np.float32)
        # Holosoma motion ``.npz`` files store raw body quaternions as wxyz.
        # Holosoma converts them to xyzw only at its simulator boundary; IsaacLab
        # and MimicKit-facing features here both stay in wxyz.
        body_quat_raw = np.asarray(data["body_quat_w"], dtype=np.float32)
        body_lin_raw = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
        body_ang_raw = np.asarray(data["body_ang_vel_w"], dtype=np.float32)
        num_frames = body_pos_raw.shape[0]
        num_robot_bodies = len(robot_body_names)

        body_pos = np.zeros((num_frames, num_robot_bodies, 3), dtype=np.float32)
        body_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_frames, num_robot_bodies, 1))
        body_lin = np.zeros((num_frames, num_robot_bodies, 3), dtype=np.float32)
        body_ang = np.zeros((num_frames, num_robot_bodies, 3), dtype=np.float32)
        available_motion_bodies: set[str] = set()
        for robot_idx, name in enumerate(robot_body_names):
            motion_name = name if name in motion_body_names else _G1_MOTION_BODY_ALIASES.get(name)
            if motion_name in motion_body_names:
                m = motion_body_names.index(motion_name)
                body_pos[:, robot_idx] = body_pos_raw[:, m]
                body_quat[:, robot_idx] = body_quat_raw[:, m]
                body_lin[:, robot_idx] = body_lin_raw[:, m]
                body_ang[:, robot_idx] = body_ang_raw[:, m]
                available_motion_bodies.add(name)

        for child_name, (parent_name, offset_parent) in _G1_FIXED_BODY_OFFSETS.items():
            if child_name not in robot_body_names or child_name in available_motion_bodies:
                continue
            if parent_name not in available_motion_bodies:
                continue
            parent_idx = robot_body_names.index(parent_name)
            child_idx = robot_body_names.index(child_name)
            parent_quat = body_quat[:, parent_idx]
            local_offset = np.broadcast_to(offset_parent, (num_frames, 3))
            world_offset = _quat_rotate_wxyz(parent_quat, local_offset)
            body_pos[:, child_idx] = body_pos[:, parent_idx] + world_offset
            body_quat[:, child_idx] = parent_quat
            body_ang[:, child_idx] = body_ang[:, parent_idx]
            body_lin[:, child_idx] = body_lin[:, parent_idx] + np.cross(body_ang[:, parent_idx], world_offset)
            available_motion_bodies.add(child_name)

        return (
            (joint_pos, joint_vel, body_pos, body_quat, body_lin, body_ang),
            available_motion_bodies,
        )

    def clamp_time_steps(self, time_steps: torch.Tensor) -> torch.Tensor:
        return torch.clamp(time_steps, min=0, max=self.num_frames - 1)

    def _interpolate(self, values: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(time_steps):
            return values[time_steps]
        t = self.clamp_time_steps(time_steps.to(device=self.device, dtype=torch.float32))
        lo = torch.floor(t).to(dtype=torch.long)
        hi = torch.clamp(lo + 1, max=self.num_frames - 1)
        weight = (t - lo.to(dtype=t.dtype)).view(-1, *([1] * (values.ndim - 1)))
        return values.index_select(0, lo) * (1.0 - weight) + values.index_select(0, hi) * weight

    def _interpolate_quat(self, values: torch.Tensor, time_steps: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(time_steps):
            return values[time_steps]
        t = self.clamp_time_steps(time_steps.to(device=self.device, dtype=torch.float32))
        lo = torch.floor(t).to(dtype=torch.long)
        hi = torch.clamp(lo + 1, max=self.num_frames - 1)
        q0 = values.index_select(0, lo)
        q1 = values.index_select(0, hi)
        sign = torch.where((q0 * q1).sum(dim=-1, keepdim=True) < 0.0, -1.0, 1.0)
        q1 = q1 * sign
        weight = (t - lo.to(dtype=t.dtype)).view(-1, *([1] * (values.ndim - 1)))
        return torch.nn.functional.normalize(q0 * (1.0 - weight) + q1 * weight, dim=-1)

    def _interpolate_quat_shortest(
        self,
        values: torch.Tensor,
        time_steps: torch.Tensor,
    ) -> torch.Tensor:
        """Mode-independent shortest-path SLERP for external evaluation."""

        t = self.clamp_time_steps(time_steps.to(device=self.device, dtype=torch.float32))
        lo = torch.floor(t).to(dtype=torch.long)
        hi = torch.clamp(lo + 1, max=self.num_frames - 1)
        q0 = F.normalize(values.index_select(0, lo), dim=-1)
        q1 = F.normalize(values.index_select(0, hi), dim=-1)
        dot = (q0 * q1).sum(dim=-1, keepdim=True)
        q1 = torch.where(dot < 0.0, -q1, q1)
        dot = dot.abs().clamp(max=1.0)
        theta = torch.acos(dot)
        sin_theta = torch.sin(theta)
        weight = (t - lo.to(dtype=t.dtype)).view(-1, *([1] * (values.ndim - 1)))
        slerp = (
            torch.sin((1.0 - weight) * theta) / sin_theta.clamp_min(1.0e-8) * q0
            + torch.sin(weight * theta) / sin_theta.clamp_min(1.0e-8) * q1
        )
        linear = q0 * (1.0 - weight) + q1 * weight
        return F.normalize(torch.where(sin_theta > 1.0e-6, slerp, linear), dim=-1)

    def _root_link_velocity(self, time_steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._interpolate(self.body_lin_vel_full_w[:, self.root_body_id], time_steps),
            self._interpolate(self.body_ang_vel_full_w[:, self.root_body_id], time_steps),
        )

    def get_frame(self, time_steps: torch.Tensor) -> dict[str, torch.Tensor]:
        time_steps = self.clamp_time_steps(time_steps.to(device=self.device))
        body_pos = self._interpolate(self.body_pos_full_w, time_steps)
        body_quat = self._interpolate_quat(self.body_quat_full_w, time_steps)
        body_lin_vel = self._interpolate(self.body_lin_vel_full_w, time_steps)
        body_ang_vel = self._interpolate(self.body_ang_vel_full_w, time_steps)
        root_lin_vel, root_ang_vel = self._root_link_velocity(time_steps)
        return {
            "joint_pos": self._interpolate(self.joint_pos, time_steps),
            "joint_vel": self._interpolate(self.joint_vel, time_steps),
            "body_pos_w": body_pos[:, self.track_body_ids],
            "body_quat_w": body_quat[:, self.track_body_ids],
            "body_lin_vel_w": body_lin_vel[:, self.track_body_ids],
            "body_ang_vel_w": body_ang_vel[:, self.track_body_ids],
            "anchor_pos_w": body_pos[:, self.anchor_body_id],
            "anchor_quat_w": body_quat[:, self.anchor_body_id],
            "root_pos_w": body_pos[:, self.root_body_id],
            "root_quat_w": body_quat[:, self.root_body_id],
            "root_lin_vel_w": root_lin_vel,
            "root_ang_vel_w": root_ang_vel,
        }

    def get_fcamp_fk_body_positions(
        self,
        *,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        joint_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the checked-in runtime URDF at caller-provided states."""

        count = int(root_pos.shape[0])
        if tuple(root_pos.shape) != (count, 3):
            raise ValueError("root_pos must have shape [B,3]")
        if tuple(root_quat.shape) != (count, 4):
            raise ValueError("root_quat must have shape [B,4]")
        if tuple(joint_pos.shape) != (count, G1_IMITATION_NUM_JOINTS):
            raise ValueError(
                f"joint_pos must have shape [B,{G1_IMITATION_NUM_JOINTS}]"
            )
        return self._fk_model.body_pos(
            root_pos=root_pos.to(device=self.device, dtype=torch.float32),
            root_quat=root_quat.to(device=self.device, dtype=torch.float32),
            joint_pos=joint_pos.to(device=self.device, dtype=torch.float32),
        )

    def build_fcamp_frame_from_robot_state(
        self,
        *,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        joint_pos: torch.Tensor,
        root_link_velocity: torch.Tensor,
        joint_vel: torch.Tensor,
    ) -> torch.Tensor:
        """Build the FCAMP 233-D frame through the expert FK code path.

        This is used as a same-state runtime invariant.  All physical fields,
        including root velocity, are supplied by the simulator; only key-body
        origins are reconstructed by the exact URDF used for expert data.
        """

        if root_link_velocity.shape != (root_pos.shape[0], 6):
            raise ValueError("root_link_velocity must have shape [B,6]")
        fk_body_pos = self.get_fcamp_fk_body_positions(
            root_pos=root_pos,
            root_quat=root_quat,
            joint_pos=joint_pos,
        )
        return build_g1_imitation_frame(
            root_pos=root_pos,
            root_quat_wxyz=root_quat,
            joint_pos=joint_pos,
            key_body_pos=fk_body_pos.index_select(1, self.imitation_key_body_ids),
            root_lin_vel=root_link_velocity[:, :3],
            root_ang_vel=root_link_velocity[:, 3:],
            joint_vel=joint_vel,
        )

    def get_fcamp_expert_frame_at_times(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Return expert frames in the runtime URDF/root-link feature domain."""

        if not torch.is_tensor(time_steps):
            raise TypeError("time_steps must be a torch.Tensor")
        requested_shape = tuple(time_steps.shape)
        phases = time_steps.to(device=self.device, dtype=torch.float32).reshape(-1)
        if not bool(torch.isfinite(phases).all()):
            raise ValueError("FCAMP expert frame times contain non-finite values")
        if bool((phases < 0).any()) or bool((phases > self.num_frames - 1).any()):
            raise ValueError(
                f"FCAMP expert frame times must lie in [0, {self.num_frames - 1}]"
            )
        root_pos = self._interpolate(self.body_pos_full_w[:, self.root_body_id], phases)
        root_quat = self._interpolate_quat_shortest(
            self.body_quat_full_w[:, self.root_body_id], phases
        )
        joint_pos = self._interpolate(self.joint_pos, phases)
        root_link_velocity = torch.cat(
            (
                self._interpolate(
                    self.body_lin_vel_full_w[:, self.root_body_id], phases
                ),
                self._interpolate(
                    self.body_ang_vel_full_w[:, self.root_body_id], phases
                ),
            ),
            dim=-1,
        )
        frame = self.build_fcamp_frame_from_robot_state(
            root_pos=root_pos,
            root_quat=root_quat,
            joint_pos=joint_pos,
            root_link_velocity=root_link_velocity,
            joint_vel=self._interpolate(self.joint_vel, phases),
        )
        return frame.reshape(requested_shape + (G1_IMITATION_FRAME_DIM,))

    @torch.no_grad()
    def _fcamp_integer_expert_frames(self) -> torch.Tensor:
        cached = self._fcamp_expert_integer_frame_cache
        if cached is None:
            cached = self.get_fcamp_expert_frame_at_times(
                torch.arange(self.num_frames, device=self.device, dtype=torch.float32)
            ).detach()
            self._fcamp_expert_integer_frame_cache = cached
        return cached

    def get_fcamp_demo_history(
        self,
        phase_indices: torch.Tensor,
        window_size: int,
    ) -> torch.Tensor:
        """Build chronological fixed-W FCAMP reset seeds ending at each phase."""

        if not torch.is_tensor(phase_indices):
            raise TypeError("phase_indices must be a torch.Tensor")
        phases = phase_indices.to(device=self.device)
        if phases.ndim != 1:
            raise ValueError(f"phase_indices must be 1-D, got {tuple(phases.shape)}")
        if torch.is_floating_point(phases):
            if not bool(torch.isfinite(phases).all()):
                raise ValueError("phase_indices contain non-finite values")
            rounded = torch.round(phases)
            if not torch.equal(phases, rounded):
                raise ValueError("FCAMP reset history endpoints must be integer phases")
            phases = rounded
        indices = history_indices(
            phases.to(dtype=torch.long), window_size, self.num_frames
        )
        frames = self._fcamp_integer_expert_frames().index_select(
            0, indices.reshape(-1)
        ).reshape(indices.shape + (G1_IMITATION_FRAME_DIM,))
        return frames

    def get_fcamp_demo_windows_at_end_indices(
        self,
        end_indices: torch.Tensor,
        window_size: int,
    ) -> torch.Tensor:
        return self.get_fcamp_demo_history(end_indices, window_size)

    def get_imitation_frame_at_times(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Return evaluator-only frames at exact fractional motion phases.

        This method always uses the dataset's raw root-link body velocities and
        joint velocities plus one shortest-path quaternion interpolation rule.
        """

        if not torch.is_tensor(time_steps):
            raise TypeError("time_steps must be a torch.Tensor")
        phases = time_steps.to(device=self.device)
        if phases.ndim != 1:
            raise ValueError(f"time_steps must be 1-D, got {tuple(phases.shape)}")
        if not bool(torch.isfinite(phases).all()):
            raise ValueError("time_steps contain non-finite values")
        if bool((phases < 0).any()) or bool((phases > self.num_frames - 1).any()):
            raise ValueError(f"imitation frame times must lie in [0, {self.num_frames - 1}]")
        phases = phases.to(dtype=torch.float32)
        root_pos = self._interpolate(self.body_pos_full_w[:, self.root_body_id], phases)
        return build_g1_imitation_frame(
            root_pos=root_pos,
            root_quat_wxyz=self._interpolate_quat_shortest(
                self.body_quat_full_w[:, self.root_body_id], phases
            ),
            joint_pos=self._interpolate(self.joint_pos, phases),
            key_body_pos=self._interpolate(
                self.body_pos_full_w[:, self.imitation_key_body_ids], phases
            ),
            root_lin_vel=self._interpolate(
                self.body_lin_vel_full_w[:, self.root_body_id], phases
            ),
            root_ang_vel=self._interpolate(
                self.body_ang_vel_full_w[:, self.root_body_id], phases
            ),
            joint_vel=self._interpolate(self.joint_vel, phases),
        )
