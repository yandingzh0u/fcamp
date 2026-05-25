from __future__ import annotations

from env.config import MimicEnvConfig
from env.mimic import G1MimicEnv


def make_mimic_env(cfg) -> G1MimicEnv:
    return G1MimicEnv(
        MimicEnvConfig(
            device=cfg.device,
            num_envs=cfg.num_envs,
            sim_dt=cfg.sim_dt,
            fix_root_link=cfg.fix_root_link,
            startup_randomization=cfg.startup_randomization,
            motion_start_phase=cfg.motion_start_phase,
            motion_end_phase=cfg.motion_end_phase,
            motion_file=cfg.motion_file,
            max_episode_steps=cfg.max_episode_steps,
            reset_noise=cfg.reset_noise,
            interval_pushes=cfg.interval_pushes,
        )
    )
