from __future__ import annotations

from dataclasses import dataclass

from env import DEFAULT_MOTION_FILE


@dataclass(slots=True)
class MixGRPOConfig:
    device: str = "cuda:0"
    num_envs: int = 4096
    sim_dt: float = 0.02
    fix_root_link: bool = False
    max_episode_steps: int = 1500
    motion_start_phase: int = 0
    motion_end_phase: int = -1
    motion_file: str = str(DEFAULT_MOTION_FILE)

    action_dim: int = 29
    policy_obs_dim: int = 0
    horizon: int = 1
    hidden_dim: int = 512
    time_embed_dim: int = 64
    depth: int = 4
    flow_steps: int = 4
    action_limit: float = 1.0

    cps_eta: float = 0.7
    group_size: int = 4
    chunks_per_rollout: int = 24
    discount_gamma: float = 0.99
    clip_range: float = 1e-2
    adv_clip_max: float = 5.0
    latent_reg_coeff: float = 0.01
    latent_soft_limit: float = 2.0
    action_saturation_coeff: float = 0.05
    action_saturation_threshold: float = 0.85
    policy_epochs: int = 1
    mini_batch_size: int = 1024
    max_grad_norm: float = 1.0

    lr: float = 3e-4
    seed: int = 0
    max_updates: int = 1000

    log_every: int = 1
    save_every: int = 50
    checkpoint_dir: str = ""
    resume: str = ""

    validation_every: int = 25
    validation_max_steps: int = 500
    validation_start_phase: int = 0
    target_validation_steps: int = 0
    success_checkpoint_name: str = "success_10s.pt"
