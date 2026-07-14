from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from amp.features import canonicalize_amp_window

from .amp_data import (
    G1_AMP_FRAME_DIM,
    G1_AMP_KEY_BODY_NAMES,
    build_g1_amp_frame,
    history_indices,
)


# ``head_link`` is a fixed child omitted by Holosoma's exported rigid-body
# trajectory.  This transform is copied from the checked-in G1 URDF.
_G1_HEAD_PARENT = "torso_link"
_G1_HEAD_OFFSET_PARENT = np.array([0.0039635, 0.0, -0.044], dtype=np.float32)


def _quat_rotate_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Vectorized quaternion rotation for ``quat[...,4]`` and ``vector[...,3]``."""

    xyz = quat[..., 1:]
    t = 2.0 * np.cross(xyz, vector)
    return vector + quat[..., :1] * t + np.cross(xyz, t)


class MimicMotionReference:


    def __init__(
        self,
        motion_file: str | Path,
        track_body_ids: torch.Tensor,
        anchor_body_id: int,
        device: torch.device,
        robot_body_names: list[str] | None = None,
        action_joint_names: list[str] | None = None,
        root_body_name: str | None = None,
        amp_key_body_names: tuple[str, ...] = G1_AMP_KEY_BODY_NAMES,
    ):
        motion_file = Path(motion_file)
        if not motion_file.is_file():
            raise FileNotFoundError(f"Motion file not found: {motion_file}")

        data = np.load(motion_file, allow_pickle=True)
        self.device = device

        if "joint_names" in data.files:
            frame, available_motion_bodies = self._load_holosoma(
                data, robot_body_names, action_joint_names
            )
        else:
            frame = self._load_legacy(data)
            available_motion_bodies = set(robot_body_names or ())

        joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w = frame
        self.joint_pos = torch.tensor(joint_pos, dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(joint_vel, dtype=torch.float32, device=device)
        self.body_pos_full_w = torch.tensor(body_pos_w, dtype=torch.float32, device=device)
        self.body_quat_full_w = torch.tensor(body_quat_w, dtype=torch.float32, device=device)
        self.body_lin_vel_full_w = torch.tensor(body_lin_vel_w, dtype=torch.float32, device=device)
        self.body_ang_vel_full_w = torch.tensor(body_ang_vel_w, dtype=torch.float32, device=device)

        self.track_body_ids = track_body_ids.to(device=device, dtype=torch.long)
        self.anchor_body_id = int(anchor_body_id)
        if root_body_name is not None and robot_body_names is not None:
            self.root_body_id = robot_body_names.index(root_body_name)
        else:
            self.root_body_id = 0
        if robot_body_names is None:
            raise ValueError("robot_body_names is required to construct AMP demo features")
        missing_amp_bodies = [name for name in amp_key_body_names if name not in robot_body_names]
        if missing_amp_bodies:
            raise ValueError(f"AMP key bodies are missing from the robot asset: {missing_amp_bodies}")
        required_motion_bodies = {
            *(robot_body_names[int(i)] for i in track_body_ids.detach().cpu().tolist()),
            *(amp_key_body_names),
        }
        if root_body_name is not None:
            required_motion_bodies.add(root_body_name)
        missing_motion_bodies = sorted(required_motion_bodies - available_motion_bodies)
        if missing_motion_bodies:
            raise ValueError(
                "Required policy/AMP bodies are missing from the motion and cannot be "
                f"reconstructed: {missing_motion_bodies}"
            )
        self.amp_key_body_ids = torch.tensor(
            [robot_body_names.index(name) for name in amp_key_body_names],
            dtype=torch.long,
            device=device,
        )
        self.num_frames = int(self.joint_pos.shape[0])

    def _load_legacy(self, data):
        joint_pos = data["joint_pos"]
        joint_vel = data["joint_vel"]
        body_pos_w = data["body_pos_w"]
        body_quat_w = data["body_quat_w"]
        body_lin_vel_w = data["body_lin_vel_w"]
        body_ang_vel_w = data["body_ang_vel_w"]
        return joint_pos, joint_vel, body_pos_w, body_quat_w, body_lin_vel_w, body_ang_vel_w

    def _load_holosoma(self, data, robot_body_names, action_joint_names):
        if robot_body_names is None or action_joint_names is None:
            raise ValueError(
                "Holosoma-format motion requires robot_body_names and action_joint_names for "
                "name-based reordering."
            )
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
            if name in motion_body_names:
                m = motion_body_names.index(name)
                body_pos[:, robot_idx] = body_pos_raw[:, m]
                body_quat[:, robot_idx] = body_quat_raw[:, m]
                body_lin[:, robot_idx] = body_lin_raw[:, m]
                body_ang[:, robot_idx] = body_ang_raw[:, m]
                available_motion_bodies.add(name)

        if "head_link" in robot_body_names and "head_link" not in available_motion_bodies:
            if _G1_HEAD_PARENT in available_motion_bodies:
                parent_idx = robot_body_names.index(_G1_HEAD_PARENT)
                head_idx = robot_body_names.index("head_link")
                parent_quat = body_quat[:, parent_idx]
                local_offset = np.broadcast_to(
                    _G1_HEAD_OFFSET_PARENT, (num_frames, 3)
                )
                world_offset = _quat_rotate_wxyz(parent_quat, local_offset)
                body_pos[:, head_idx] = body_pos[:, parent_idx] + world_offset
                body_quat[:, head_idx] = parent_quat
                body_ang[:, head_idx] = body_ang[:, parent_idx]
                body_lin[:, head_idx] = (
                    body_lin[:, parent_idx]
                    + np.cross(body_ang[:, parent_idx], world_offset)
                )
                available_motion_bodies.add("head_link")

        return (
            (joint_pos, joint_vel, body_pos, body_quat, body_lin, body_ang),
            available_motion_bodies,
        )

    def clamp_time_steps(self, time_steps: torch.Tensor) -> torch.Tensor:
        return torch.clamp(time_steps, min=0, max=self.num_frames - 1)

    def get_frame(self, time_steps: torch.Tensor) -> dict[str, torch.Tensor]:
        time_steps = self.clamp_time_steps(time_steps)
        return {
            "joint_pos": self.joint_pos[time_steps],
            "joint_vel": self.joint_vel[time_steps],
            "body_pos_w": self.body_pos_full_w[time_steps][:, self.track_body_ids],
            "body_quat_w": self.body_quat_full_w[time_steps][:, self.track_body_ids],
            "body_lin_vel_w": self.body_lin_vel_full_w[time_steps][:, self.track_body_ids],
            "body_ang_vel_w": self.body_ang_vel_full_w[time_steps][:, self.track_body_ids],
            "anchor_pos_w": self.body_pos_full_w[time_steps, self.anchor_body_id],
            "anchor_quat_w": self.body_quat_full_w[time_steps, self.anchor_body_id],
            "root_pos_w": self.body_pos_full_w[time_steps, self.root_body_id],
            "root_quat_w": self.body_quat_full_w[time_steps, self.root_body_id],
            "root_lin_vel_w": self.body_lin_vel_full_w[time_steps, self.root_body_id],
            "root_ang_vel_w": self.body_ang_vel_full_w[time_steps, self.root_body_id],
        }

    @property
    def amp_frame_dim(self) -> int:
        return G1_AMP_FRAME_DIM

    def get_amp_frame(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Return clean reference frames in the same 233-D schema as policy frames."""
        if not torch.is_tensor(time_steps):
            raise TypeError("time_steps must be a torch.Tensor")
        time_steps = time_steps.to(device=self.device, dtype=torch.long)
        if bool((time_steps < 0).any()) or bool((time_steps >= self.num_frames).any()):
            raise ValueError(f"AMP frame indices must lie in [0, {self.num_frames - 1}]")
        root_pos = self.body_pos_full_w[time_steps, self.root_body_id]
        return build_g1_amp_frame(
            root_pos=root_pos,
            root_quat_wxyz=self.body_quat_full_w[time_steps, self.root_body_id],
            joint_pos=self.joint_pos[time_steps],
            key_body_pos=self.body_pos_full_w[time_steps][..., self.amp_key_body_ids, :],
            root_lin_vel=self.body_lin_vel_full_w[time_steps, self.root_body_id],
            root_ang_vel=self.body_ang_vel_full_w[time_steps, self.root_body_id],
            joint_vel=self.joint_vel[time_steps],
        )

    def get_amp_demo_history(
        self,
        phase_indices: torch.Tensor,
        window_size: int,
        *,
        flatten: bool = False,
    ) -> torch.Tensor:
        """Build reset-aligned histories ending at ``phase_indices``.

        At the beginning of a motion, missing predecessor frames are filled by
        repeating frame zero.  This mirrors a stationary left boundary and,
        importantly, never wraps the end of the motion into its beginning.
        """
        phase_indices = phase_indices.to(device=self.device, dtype=torch.long)
        indices = history_indices(
            phase_indices,
            window_size,
            self.num_frames,
            clamp_start=True,
        )
        # Reset seeds remain raw: CausalAMPHistory must re-anchor the whole
        # window again after every subsequently appended policy frame.
        frames = self.get_amp_frame(indices)
        return frames.flatten(start_dim=-2) if flatten else frames

    def sample_amp_demo_windows(
        self,
        num_samples: int,
        window_size: int,
        *,
        flatten: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample MimicKit-style expert windows over the full motion timeline.

        The newest frame is uniform over all motion frames. Missing predecessors
        at the left boundary repeat frame zero, matching MimicKit's clipped
        negative demo times. Motion-end wrapping is never used.
        """
        if num_samples < 0:
            raise ValueError(f"num_samples must be non-negative, got {num_samples}")
        if window_size <= 0:
            raise ValueError(f"window_size must be positive, got {window_size}")
        end_indices = torch.randint(
            0,
            self.num_frames,
            (num_samples,),
            device=self.device,
            generator=generator,
        )
        return self.get_amp_demo_windows_at_end_indices(
            end_indices,
            window_size,
            flatten=flatten,
        )

    def get_amp_demo_windows_at_end_indices(
        self,
        end_indices: torch.Tensor,
        window_size: int,
        *,
        flatten: bool = True,
    ) -> torch.Tensor:
        """Build expert AMP windows with caller-specified endpoint frames."""

        end_indices = end_indices.to(device=self.device, dtype=torch.long)
        indices = history_indices(
            end_indices,
            window_size,
            self.num_frames,
            clamp_start=True,
        )
        frames = self.get_amp_frame(indices)
        if not flatten:
            return frames
        return canonicalize_amp_window(frames).flatten(start_dim=-2)
