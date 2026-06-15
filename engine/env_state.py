from __future__ import annotations

import torch


class EnvStateMixin:
    def _contact_history_buffers(self) -> dict[str, torch.Tensor]:
        """Return the contact sensor's rolling buffers keyed by attribute name (or {})."""
        sensor = getattr(self.env, "contact_sensor", None)
        if sensor is None:
            return {}
        data = getattr(sensor, "_data", None) or getattr(sensor, "data", None)
        if data is None:
            return {}
        buffers: dict[str, torch.Tensor] = {}
        for attr in ("net_forces_w", "net_forces_w_history", "force_matrix_w", "force_matrix_w_history"):
            buffer = getattr(data, attr, None)
            if buffer is not None:
                buffers[attr] = buffer
        return buffers

    def _snapshot_env_state(self) -> dict[str, torch.Tensor]:
        robot = self.env.robot
        snapshot = {
            "root_state_w": robot.data.root_state_w.clone(),
            "joint_pos": robot.data.joint_pos.clone(),
            "joint_vel": robot.data.joint_vel.clone(),
            "default_root_state": self.env.default_root_state.clone(),
            "default_joint_pos": self.env.default_joint_pos.clone(),
            "default_joint_vel": self.env.default_joint_vel.clone(),
            "default_action_joint_pos": self.env.default_action_joint_pos.clone(),
            "default_action_joint_vel": self.env.default_action_joint_vel.clone(),
            "phase_steps": self.env.phase_steps.clone(),
            "episode_steps": self.env.episode_steps.clone(),
            "last_action": self.env.last_action.clone(),
            "prev_action": self.env.prev_action.clone(),
            "next_push_step": self.env.next_push_step.clone(),
        }
        snapshot["contact_history"] = {
            attr: buffer.clone() for attr, buffer in self._contact_history_buffers().items()
        }
        return snapshot

    def _restore_env_state(self, snapshot: dict[str, torch.Tensor]) -> None:
        env_ids = torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        root_state = snapshot["root_state_w"]
        root_pos_local = root_state[:, :3] - self.env.scene.env_origins
        self.env.scene.reset(env_ids=env_ids)
        self.env._write_robot_state(
            root_pos=root_pos_local,
            root_quat=root_state[:, 3:7],
            root_lin_vel=root_state[:, 7:10],
            root_ang_vel=root_state[:, 10:13],
            joint_pos=snapshot["joint_pos"][:, self.env.action_joint_ids],
            joint_vel=snapshot["joint_vel"][:, self.env.action_joint_ids],
            env_ids=env_ids,
        )
        self.env.phase_steps = snapshot["phase_steps"].clone()
        self.env.episode_steps = snapshot["episode_steps"].clone()
        self.env.last_action = snapshot["last_action"].clone()
        self.env.prev_action = snapshot.get("prev_action", torch.zeros_like(self.env.last_action)).clone()
        self.env.next_push_step = snapshot["next_push_step"].clone()
        self.env.default_root_state = snapshot["default_root_state"].clone()
        self.env.default_joint_pos = snapshot["default_joint_pos"].clone()
        self.env.default_joint_vel = snapshot["default_joint_vel"].clone()
        self.env.default_action_joint_pos = snapshot["default_action_joint_pos"].clone()
        self.env.default_action_joint_vel = snapshot["default_action_joint_vel"].clone()
        contact_history = snapshot.get("contact_history", {})
        if contact_history:
            live_buffers = self._contact_history_buffers()
            for attr, saved in contact_history.items():
                buffer = live_buffers.get(attr)
                if buffer is not None:
                    buffer.copy_(saved)
        self.env.scene.update(self.env.physics_dt)
