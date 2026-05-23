from __future__ import annotations

from dataclasses import dataclass

from env import DEFAULT_MOTION_FILE


@dataclass(slots=True)
class MixGRPOConfig:
    device: str = "cuda:0"
    num_envs: int = 8192
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
    horizon: int = 1
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    flow_steps: int = 4
    action_squash_scale: float = 5.0

    init_noise_std: float = 0.8
    init_same_noise: bool = False
    eval_initial_noise: str = "zero"
    sde_eta: float = 0.7
    num_generations: int = 4
    chunks_per_rollout: int = 4
    rollout_segments_per_update: int = 32
    # Number of extra deterministic-policy steps rolled out *after* the main GRPO window
    # to estimate a Monte-Carlo tail bootstrap value. 0 disables (legacy behavior). The
    # tail_return is injected as `last_values` for GAE and as the terminal RTG seed for
    # the GRPO score, so the policy can credit/blame in-window actions for failures that
    # occur shortly after the window closes — without adding a critic.
    tail_bootstrap_steps: int = 0
    terminal_penalty: float = 50.0
    discount_gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.3
    adv_clip_max: float = 5.0
    desired_kl: float = 0.03
    entropy_coef: float = 0.005
    value_loss_coef: float = 0.0
    use_clipped_value_loss: bool = True
    policy_epochs: int = 5
    num_mini_batches: int = 4
    mini_batch_size: int = 0
    micro_batch_size: int = 8192
    max_grad_norm: float = 1.0

    lr: float = 1e-3
    policy_lr: float = 1e-3
    seed: int = 0
    max_updates: int = 30000

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

    debug_probe: bool = False
    debug_probe_every: int = 1
