from __future__ import annotations

from dataclasses import dataclass

from env import DEFAULT_MOTION_FILE


@dataclass(slots=True)
class OfficialPPOConfig:
    device: str = "cuda:0"
    num_envs: int = 4096
    sim_dt: float = 0.02
    fix_root_link: bool = False
    max_episode_steps: int = 1500
    motion_start_phase: int = 0
    motion_end_phase: int = -1
    motion_file: str = str(DEFAULT_MOTION_FILE)
    startup_randomization: bool = True
    reset_noise: bool = True
    interval_pushes: bool = True

    action_dim: int = 29
    policy_obs_dim: int = 0
    critic_obs_dim: int = 0
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    init_noise_std: float = 1.0
    empirical_normalization: bool = False
    horizon: int = 1

    num_steps_per_env: int = 24
    policy_epochs: int = 5
    num_mini_batches: int = 4
    clip_range: float = 0.2
    discount_gamma: float = 0.99
    gae_lambda: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.005
    lr: float = 1.0e-3
    schedule: str = "adaptive"
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    max_updates: int = 30000
    seed: int = 0

    log_every: int = 1
    save_every: int = 500
    checkpoint_dir: str = ""
    resume: str = ""
    reset_optimizer_on_resume: bool = False

    validation_every: int = 0
    validation_max_steps: int = 500
    validation_start_phase: int = 0
    validation_fixed_seed: int = -1
    validation_preserve_state: bool = True
    target_validation_steps: int = 0
    success_checkpoint_name: str = "success_10s.pt"
