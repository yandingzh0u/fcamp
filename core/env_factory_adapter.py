"""Build the task env from the typed Config. The EnvCfg is the single source of truth; this
maps it 1:1 into the env's internal MimicEnvConfig (no getattr-with-default shadowing)."""
from __future__ import annotations

from env.config import DEFAULT_MOTION_FILE, MimicEnvConfig
from env.mimic import G1MimicEnv


def build_env(cfg):
    e = cfg.env
    return G1MimicEnv(
        MimicEnvConfig(
            device=e.device,
            num_envs=e.num_envs,
            sim_dt=e.sim_dt,
            render=e.render,
            render_every=e.render_every,
            fix_root_link=e.fix_root_link,
            action_scale_multiplier=e.action_scale_multiplier,
            startup_randomization=e.startup_randomization,
            motion_start_phase=e.motion_start_phase,
            motion_end_phase=e.motion_end_phase,
            adaptive_motion_sampling=e.adaptive_motion_sampling,
            adaptive_uniform_ratio=e.adaptive_uniform_ratio,
            motion_start_phase_ratio=e.motion_start_phase_ratio,
            adaptive_alpha=e.adaptive_alpha,
            adaptive_kernel_size=e.adaptive_kernel_size,
            motion_file=e.motion_file or str(DEFAULT_MOTION_FILE),
            max_episode_steps=e.max_episode_steps,
            reset_noise=e.reset_noise,
            interval_pushes=e.interval_pushes,
            observation_noise=e.observation_noise,
            num_generations=e.num_generations,
            joint_acc_weight=e.joint_acc_weight,
            joint_torque_weight=e.joint_torque_weight,
            action_rate_weight=e.action_rate_weight,
            action_accel_weight=e.action_accel_weight,
            action_l2_weight=e.action_l2_weight,
            term_z_weight=e.term_z_weight,
            term_z_sigma=e.term_z_sigma,
        )
    )
