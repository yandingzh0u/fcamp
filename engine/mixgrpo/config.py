from __future__ import annotations

from dataclasses import dataclass

from env.config import DEFAULT_MOTION_FILE


@dataclass(slots=True)
class MixGRPOConfig:
    device: str = "cuda:0"
    num_envs: int = 8192
    sim_dt: float = 0.02
    fix_root_link: bool = False
    action_scale_multiplier: float = 1.0
    max_episode_steps: int = 1500
    motion_start_phase: int = 0
    motion_end_phase: int = -1
    adaptive_motion_sampling: bool = True
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    adaptive_kernel_size: int = 1
    motion_file: str = str(DEFAULT_MOTION_FILE)
    startup_randomization: bool = True
    reset_noise: bool = True
    interval_pushes: bool = True
    observation_noise: bool = True
    joint_acc_weight: float = 2.5e-7
    joint_torque_weight: float = 1.0e-5
    action_rate_weight: float = 1.0e-1
    action_accel_weight: float = 0.0
    action_l2_weight: float = 0.0

    action_dim: int = 29
    policy_obs_dim: int = 0
    horizon: int = 1
    # Deprecated compatibility field. Actor observations use the verified legacy input:
    # current reference only, no future reference frames.
    future_ref_steps: int = 0
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    flow_steps: int = 4
    action_squash_scale: float = 5.0

    # DEPRECATED for the loss path. The flow policy is a JOINT policy pi(a0..a_{h-1}|s);
    # per-frame PPO is a biased gradient estimator (FPO/DPPO use chunk-level PPO: one joint
    # log-ratio + one chunk advantage). The training objective is ALWAYS chunk-level now.
    # This flag is retained only so per-frame DIAGNOSTICS can be logged; it does NOT change
    # the loss/advantage/log-prob path regardless of value.
    frame_factorized: bool = False
    # joint_kl_guard: currently inert (the guard lived in the removed frame-level loss path).
    joint_kl_guard: bool = False
    # Temporal trajectory prior for h>1 action chunks. The flow/log_prob operate in a
    # COEFFICIENT latent of `basis_count` low-frequency modes per joint; a fixed temporal
    # basis expands them to the horizon-frame action chunk, constraining executed chunks to
    # the smooth-trajectory manifold (fixes in_chunk_delta -> 1 / rising action_rate at large
    # horizon). 0 -> basis_count = horizon = legacy flat per-frame parametrization (no-op).
    basis_count: int = 0
    # Front-of-chunk residual stitching. When >0, decoded chunks are blended from the
    # previous executed residual action (the last_action observation term) into the raw
    # decoded chunk over this many frames. This fixes cross-chunk target discontinuities
    # without adding future reference context.
    chunk_stitch_frames: int = 0
    chunk_stitch_mode: str = "smoothstep"

    init_noise_std: float = 0.8
    init_same_noise: bool = False
    first_generation_zero_noise: bool = False
    eval_initial_noise: str = "zero"
    sde_eta: float = 0.7
    num_generations: int = 4
    # Fixed environment frames per GRPO update. The effective number of policy
    # chunks is derived as `rollout_env_steps // horizon`; rollout_env_steps
    # must divide horizon exactly. This matches FPO-style data collection where
    # the physical rollout window stays fixed while the executed action chunk
    # length changes. Set <= 0 to use chunks_per_rollout directly.
    rollout_env_steps: int = 24
    # Fallback number of policy chunks per GRPO update when rollout_env_steps <= 0.
    chunks_per_rollout: int = 24
    # Number of extra deterministic-policy steps rolled out *after* the main GRPO window
    # to estimate a Monte-Carlo tail bootstrap value. 0 disables (legacy behavior). The
    # tail_return is injected as `last_values` for GAE and as the terminal RTG seed for
    # the GRPO score, so the policy can credit/blame in-window actions for failures that
    # occur shortly after the window closes — without adding a critic.
    tail_bootstrap_steps: int = 0
    terminal_penalty: float = 50.0
    discount_gamma: float = 0.99
    clip_range: float = 0.3
    adv_clip_max: float = 5.0
    desired_kl: float = 0.03
    # If > 0, add a KL-penalty term `kl_penalty_coef * KL(old || new)` to the policy loss.
    # Useful as an alternative to clip when the chunk-level ratio is high-dimensional and
    # exp(log_ratio) blows up; with kl_penalty_coef > 0 you typically want clip_range very
    # large so it does not dominate.
    kl_penalty_coef: float = 0.0
    entropy_coef: float = 0.005
    value_loss_coef: float = 0.0
    policy_epochs: int = 5
    num_mini_batches: int = 4
    mini_batch_size: int = 0
    micro_batch_size: int = 8192
    max_grad_norm: float = 1.0

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
    validation_observation_noise: bool = False
    target_validation_steps: int = 0
    success_checkpoint_name: str = "success_10s.pt"

    debug_probe: bool = False
    debug_probe_every: int = 1
