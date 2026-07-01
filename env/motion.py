from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


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
    ):
        motion_file = Path(motion_file)
        if not motion_file.is_file():
            raise FileNotFoundError(f"Motion file not found: {motion_file}")

        data = np.load(motion_file, allow_pickle=True)
        self.device = device

        if "joint_names" in data.files:
            frame = self._load_holosoma(data, robot_body_names, action_joint_names)
        else:
            frame = self._load_legacy(data)

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
        for robot_idx, name in enumerate(robot_body_names):
            if name in motion_body_names:
                m = motion_body_names.index(name)
                body_pos[:, robot_idx] = body_pos_raw[:, m]
                body_quat[:, robot_idx] = body_quat_raw[:, m]
                body_lin[:, robot_idx] = body_lin_raw[:, m]
                body_ang[:, robot_idx] = body_ang_raw[:, m]
        return joint_pos, joint_vel, body_pos, body_quat, body_lin, body_ang

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
