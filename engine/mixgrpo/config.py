from __future__ import annotations

from dataclasses import dataclass

from env.config import DEFAULT_MOTION_FILE


@dataclass(slots=True)
class MixGRPOConfig:
    device: str = "cuda:0"
    num_envs: int = 8192
    sim_dt: float = 0.02
    render: bool = False
    render_every: int = 1
    fix_root_link: bool = False
    action_scale_multiplier: float = 1.0
    max_episode_steps: int = -1
    motion_start_phase: int = 0
    motion_end_phase: int = -1
    motion_start_phase_ratio: float = 0.25
    motion_file: str = str(DEFAULT_MOTION_FILE)
    startup_randomization: bool = True
    reset_noise: bool = True
    interval_pushes: bool = True
    observation_noise: bool = True
    action_rate_weight: float = 1.0e-1

    action_dim: int = 29
    policy_obs_dim: int = 0
    # Defaults are kept in sync with the train_mixgrpo.py CLI defaults so programmatic
    # construction (without the CLI) reproduces the same configuration.
    horizon: int = 12
    # Deprecated compatibility field. Actor observations use the verified legacy input:
    # current reference only, no future reference frames.
    future_ref_steps: int = 0
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    flow_steps: int = 4
    action_squash_scale: float = 5.0

    # Anchored incremental-trajectory parametrization for h>1 action chunks. The flow/log_prob
    # operate in a COEFFICIENT latent of `basis_count` low-frequency VELOCITY modes per joint;
    # these are integrated into a displacement trajectory (first frame == 0) and added to the
    # previous executed residual in atanh space before tanh, so chunk[:,0] == previous chunk's
    # last frame exactly (cross-chunk continuity is intrinsic, no execution-time stitch).
    # basis_count is clamped to [1, horizon-1]. 0 -> horizon-1.
    basis_count: int = 4

    init_noise_std: float = 0.8
    init_same_noise: bool = False
    first_generation_zero_noise: bool = False
    eval_initial_noise: str = "zero"
    sde_eta: float = 0.7
    num_generations: int = 4
    # Fixed environment frames per GRPO update. The effective number of policy chunks is
    # `rollout_env_steps // horizon` (must divide exactly). 120 = 10 chunks of horizon 12;
    # every one of these frames is trained on. Set <= 0 to use chunks_per_rollout directly.
    rollout_env_steps: int = 120
    # Fallback number of policy chunks per GRPO update when rollout_env_steps <= 0.
    chunks_per_rollout: int = 24
    discount_gamma: float = 0.99
    clip_range: float = 0.3
    adv_clip_max: float = 5.0
    desired_kl: float = 0.06
    # If > 0, add a KL-penalty term `kl_penalty_coef * KL(old || new)` to the policy loss.
    # Useful as an alternative to clip when the chunk-level ratio is high-dimensional and
    # exp(log_ratio) blows up; with kl_penalty_coef > 0 you typically want clip_range very
    # large so it does not dominate.
    kl_penalty_coef: float = 0.0
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
    validation_done_frac_early_stop: float = 0.98
    target_validation_steps: int = 0
    success_checkpoint_name: str = "success_10s.pt"

    debug_probe: bool = False
    debug_probe_every: int = 1
