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
    ):
        motion_file = Path(motion_file)
        if not motion_file.is_file():
            raise FileNotFoundError(f"Motion file not found: {motion_file}")

        data = np.load(motion_file)
        self.motion_file = motion_file
        self.device = device
        self.fps = int(np.asarray(data["fps"]).reshape(-1)[0])
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self.body_pos_full_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self.body_quat_full_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self.body_lin_vel_full_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self.body_ang_vel_full_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self.track_body_ids = track_body_ids.to(device=device, dtype=torch.long)
        self.anchor_body_id = int(anchor_body_id)
        self.root_body_id = 0
        self.num_frames = int(self.joint_pos.shape[0])

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
