from __future__ import annotations

from env.config import MimicEnvConfig
from env.mimic import G1MimicEnv


def make_mimic_env(cfg) -> G1MimicEnv:
    future_ref_steps = 0
    return G1MimicEnv(
        MimicEnvConfig(
            device=cfg.device,
            num_envs=cfg.num_envs,
            sim_dt=cfg.sim_dt,
            render=getattr(cfg, "render", False),
            render_every=getattr(cfg, "render_every", 1),
            fix_root_link=cfg.fix_root_link,
            action_scale_multiplier=getattr(cfg, "action_scale_multiplier", 1.0),
            startup_randomization=cfg.startup_randomization,
            motion_start_phase=cfg.motion_start_phase,
            motion_end_phase=cfg.motion_end_phase,
            adaptive_motion_sampling=getattr(cfg, "adaptive_motion_sampling", True),
            adaptive_uniform_ratio=getattr(cfg, "adaptive_uniform_ratio", 0.1),
            motion_start_phase_ratio=getattr(cfg, "motion_start_phase_ratio", 0.25),
            adaptive_alpha=getattr(cfg, "adaptive_alpha", 0.001),
            adaptive_kernel_size=getattr(cfg, "adaptive_kernel_size", 1),
            motion_file=cfg.motion_file,
            max_episode_steps=cfg.max_episode_steps,
            reset_noise=cfg.reset_noise,
            interval_pushes=cfg.interval_pushes,
            observation_noise=getattr(cfg, "observation_noise", True),
            num_generations=int(getattr(cfg, "num_generations", 1)),
            joint_acc_weight=getattr(cfg, "joint_acc_weight", 2.5e-7),
            joint_torque_weight=getattr(cfg, "joint_torque_weight", 1.0e-5),
            action_rate_weight=getattr(cfg, "action_rate_weight", 1.0e-1),
            action_accel_weight=getattr(cfg, "action_accel_weight", 0.0),
            action_l2_weight=getattr(cfg, "action_l2_weight", 0.0),
            future_ref_steps=future_ref_steps,
        )
    )
