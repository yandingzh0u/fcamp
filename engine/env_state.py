from __future__ import annotations

import torch


def snapshot_env_state(env) -> dict[str, torch.Tensor]:
    robot = env.robot
    snapshot = {
        "root_pose_w": robot.data.root_link_pose_w.clone(),
        "root_velocity_w": env.get_mimic_root_velocity_w().clone(),
        "joint_pos": robot.data.joint_pos.clone(),
        "joint_vel": robot.data.joint_vel.clone(),
        "phase_steps": env.phase_steps.clone(),
        "episode_steps": env.episode_steps.clone(),
        "episode_ids": env.episode_ids.clone(),
        "next_episode_id": torch.tensor(
            int(env._next_episode_id), dtype=torch.long, device=env.device
        ),
        "last_action": env.last_action.clone(),
        "command_rate": env.command_rate.clone(),
        "next_push_step": env.next_push_step.clone(),
        "push_time_left": env.push_time_left.clone(),
        "first_push_step": env.first_push_step.clone(),
        "adaptive_bin_failed_count": env.adaptive_sampler.bin_failed_count.clone(),
        "adaptive_current_bin_failed_count": env.adaptive_sampler.current_bin_failed_count.clone(),
        "failure_recorded": env._failure_recorded.clone(),
    }
    sensor = env.contact_sensor
    snapshot["contact_timestamp"] = sensor._timestamp.clone()
    snapshot["contact_timestamp_last_update"] = sensor._timestamp_last_update.clone()
    snapshot["contact_is_outdated"] = sensor._is_outdated.clone()
    for name in (
        "net_forces_w",
        "net_forces_w_history",
        "force_matrix_w",
        "force_matrix_w_history",
        "last_air_time",
        "current_air_time",
        "last_contact_time",
        "current_contact_time",
    ):
        value = getattr(sensor.data, name, None)
        if torch.is_tensor(value):
            snapshot[f"contact_{name}"] = value.clone()
    return snapshot


def restore_env_state(env, snapshot: dict[str, torch.Tensor]) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    root_pose = snapshot["root_pose_w"]
    root_velocity = snapshot["root_velocity_w"]
    root_pos_local = root_pose[:, :3] - env.scene.env_origins
    env.scene.reset(env_ids=env_ids)
    env._write_robot_state(
        root_pos=root_pos_local,
        root_quat=root_pose[:, 3:7],
        root_lin_vel=root_velocity[:, :3],
        root_ang_vel=root_velocity[:, 3:],
        joint_pos=snapshot["joint_pos"][:, env.action_joint_ids],
        joint_vel=snapshot["joint_vel"][:, env.action_joint_ids],
        env_ids=env_ids,
    )
    env.phase_steps.copy_(snapshot["phase_steps"])
    env.episode_steps.copy_(snapshot["episode_steps"])
    if "episode_ids" in snapshot:
        env.episode_ids.copy_(snapshot["episode_ids"])
        env._next_episode_id = int(snapshot["next_episode_id"].item())
    else:
        env.episode_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        env._next_episode_id = int(env.num_envs)
    env.last_action.copy_(snapshot["last_action"])
    env.command_rate.copy_(snapshot["command_rate"])
    env.next_push_step.copy_(snapshot["next_push_step"])
    env.push_time_left.copy_(snapshot["push_time_left"])
    env.first_push_step.copy_(snapshot["first_push_step"])
    sampler = env.adaptive_sampler
    sampler.bin_failed_count.copy_(snapshot["adaptive_bin_failed_count"].to(sampler.bin_failed_count))
    sampler.current_bin_failed_count.copy_(
        snapshot["adaptive_current_bin_failed_count"].to(sampler.current_bin_failed_count)
    )
    env._failure_recorded.copy_(snapshot["failure_recorded"])
    env.scene.update(env.physics_dt)
    sensor = env.contact_sensor
    for name in (
        "net_forces_w",
        "net_forces_w_history",
        "force_matrix_w",
        "force_matrix_w_history",
        "last_air_time",
        "current_air_time",
        "last_contact_time",
        "current_contact_time",
    ):
        saved = snapshot.get(f"contact_{name}")
        current = getattr(sensor.data, name, None)
        if torch.is_tensor(saved) and torch.is_tensor(current):
            current.copy_(saved)
    sensor._timestamp.copy_(snapshot["contact_timestamp"])
    sensor._timestamp_last_update.copy_(snapshot["contact_timestamp_last_update"])
    sensor._is_outdated.copy_(snapshot["contact_is_outdated"])
