from __future__ import annotations

import torch


def snapshot_env_state(env) -> dict[str, torch.Tensor]:
    robot = env.robot
    return {
        "root_state_w": robot.data.root_state_w.clone(),
        "joint_pos": robot.data.joint_pos.clone(),
        "joint_vel": robot.data.joint_vel.clone(),
        "phase_steps": env.phase_steps.clone(),
        "episode_steps": env.episode_steps.clone(),
        "last_action": env.last_action.clone(),
        "next_push_step": env.next_push_step.clone(),
        "adaptive_bin_failed_count": env.adaptive_sampler.bin_failed_count.clone(),
        "adaptive_current_bin_failed_count": env.adaptive_sampler.current_bin_failed_count.clone(),
        "failure_recorded": env._failure_recorded.clone(),
    }


def restore_env_state(env, snapshot: dict[str, torch.Tensor]) -> None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    root_state = snapshot["root_state_w"]
    root_pos_local = root_state[:, :3] - env.scene.env_origins
    env.scene.reset(env_ids=env_ids)
    env._write_robot_state(
        root_pos=root_pos_local,
        root_quat=root_state[:, 3:7],
        root_lin_vel=root_state[:, 7:10],
        root_ang_vel=root_state[:, 10:13],
        joint_pos=snapshot["joint_pos"][:, env.action_joint_ids],
        joint_vel=snapshot["joint_vel"][:, env.action_joint_ids],
        env_ids=env_ids,
    )
    env.phase_steps = snapshot["phase_steps"].clone()
    env.episode_steps = snapshot["episode_steps"].clone()
    env.last_action = snapshot["last_action"].clone()
    env.next_push_step = snapshot["next_push_step"].clone()
    sampler = env.adaptive_sampler
    sampler.bin_failed_count.copy_(snapshot["adaptive_bin_failed_count"].to(sampler.bin_failed_count))
    sampler.current_bin_failed_count.copy_(
        snapshot["adaptive_current_bin_failed_count"].to(sampler.current_bin_failed_count)
    )
    env._failure_recorded = snapshot["failure_recorded"].clone()
    env.scene.update(env.physics_dt)
