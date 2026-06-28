"""Single-source typed configuration.

One YAML file is the only place defaults live. We load it, apply `--set a.b=c` dotted
overrides, then build three frozen-ish dataclasses:

    EnvCfg   -> environment / task / reward / termination
    AlgoCfg  -> algorithm + network + optimization (mixgrpo today, ppo/fpo later)
    TrainCfg -> training loop / io / validation

No argparse defaults, no getattr copies between layers. Change a value in the YAML (or via
--set) and it takes effect, with no second default silently shadowing it.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvCfg:
    device: str = "cuda:0"
    num_envs: int = 8192
    sim_dt: float = 0.02
    fix_root_link: bool = False
    startup_randomization: bool = True
    action_scale_multiplier: float = 1.0
    motion_file: str = ""
    max_episode_steps: int = -1
    motion_start_phase: int = 0
    motion_end_phase: int = -1
    reset_noise: bool = True
    interval_pushes: bool = True
    observation_noise: bool = True
    adaptive_motion_sampling: bool = True
    adaptive_uniform_ratio: float = 0.1
    motion_start_phase_ratio: float = 0.25
    adaptive_alpha: float = 0.001
    adaptive_kernel_size: int = 1
    joint_acc_weight: float = 2.5e-7
    joint_torque_weight: float = 1.0e-5
    action_rate_weight: float = 0.1
    action_accel_weight: float = 0.0
    action_l2_weight: float = 0.0
    term_z_weight: float = 3.0
    term_z_sigma: float = 0.12
    # GRPO observation-noise sharing: set by the algorithm (num_generations) at build time.
    num_generations: int = 1
    render: bool = False
    render_every: int = 1


@dataclass
class AlgoCfg:
    # network
    action_dim: int = 29
    horizon: int = 12
    actor_hidden_dims: tuple[int, ...] = (512, 256, 128)
    activation: str = "elu"
    action_squash_scale: float = 5.0
    # flow / SDE exploration
    flow_steps: int = 4
    sde_eta: float = 0.7
    init_noise_std: float = 0.8
    init_same_noise: bool = False
    first_generation_zero_noise: bool = False
    eval_initial_noise: str = "zero"
    num_generations: int = 4
    # rollout
    rollout_env_steps: int = 48
    chunks_per_rollout: int = 24
    tail_bootstrap_steps: int = 80
    terminal_penalty: float = 50.0
    discount_gamma: float = 0.99
    # on-policy state bank
    onpolicy_state_bank: bool = False
    onpolicy_state_ratio: float = 0.5
    onpolicy_refresh_every: int = 25
    onpolicy_bank_rollout_steps: int = 0
    onpolicy_bank_min_phase: int = 80
    onpolicy_bank_capacity: int = 16384
    onpolicy_bank_min_size: int = 256
    onpolicy_bank_hard_ratio: float = 0.5
    onpolicy_bank_hard_window: int = 96
    # optimization
    clip_range: float = 0.3
    adv_clip_max: float = 5.0
    desired_kl: float = 0.06
    entropy_coef: float = 0.005
    policy_epochs: int = 5
    num_mini_batches: int = 4
    micro_batch_size: int = 8192
    max_grad_norm: float = 1.0
    policy_lr: float = 1.0e-3
    value_lr: float = 1.0e-3   # critic LR. Decoupled from actor (adaptive) LR so the value head's
                               # large early gradient (esp. with terminal_penalty) cannot drag the
                               # actor's adaptive schedule. <=0 -> follow policy_lr.

    # --- PPO (actor-critic) specific. Unused by MixGRPO. ---
    num_steps_per_env: int = 24
    num_learning_epochs: int = 5
    gae_lambda: float = 0.95
    value_loss_coef: float = 1.0
    actor_learning_rate: float = 1.0e-3
    critic_learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    empirical_normalization: bool = True
    init_at_random_ep_len: bool = True

    # --- FPO++ specific. Unused by MixGRPO / PPO. ---
    fpo_num_mc: int = 16            # Monte-Carlo (eps, t) samples per action for the CFM ratio (paper Eq. 10).
    fpo_delta_clip: float = 3.0     # STE clamp on log-ratio (l_old - l_new) before exp() (official cfm_diff_clamp_max).
    fpo_cfm_loss_clamp: float = 3.0  # Symmetric clamp on old/new CFM loss before the ratio diff (official cfm_loss_clamp).
    # Official-aligned single-step Flow actor (amazon-far/fpo-control G1 motion tracking).
    actor_scale: float = 1.0                      # action = actor_scale * x_t (linear, NO tanh).
    mlp_output_scale: float = 1.0                 # scale on the raw velocity-net output.
    timestep_embed_dim: int = 8                   # sinusoidal cos/sin timestep embedding width.
    cfm_loss_reduction: str = "mean"              # reduction over the action dim (tracking: mean).
    action_perturb_std: float = 0.1               # Gaussian noise added to the action in training (entropy reg).
    cfm_loss_t_inverse_cdf_beta: float = 1.0      # Beta(1, beta) inverse-CDF shaping of CFM timesteps.
    schedule: str = "adaptive"                    # "adaptive" (KL-driven LR) or "fixed".
    fpo_adv_clamp: float = 5.0                    # symmetric advantage clamp before the surrogate.
    cfm_loss_clamp_neg_adv: bool = True           # clamp the new CFM loss where advantage < 0.
    cfm_loss_clamp_neg_adv_max: float = 20.0      # cap for that negative-advantage CFM clamp.
    trust_region_mode: str = "aspo"              # ppo | spo | aspo.
    num_micro_batches: int = 1                    # gradient-accum microbatches per logical minibatch.
    storage_action_noise_std: float = 0.0         # extra noise added to stored actions (off by default).


@dataclass
class TrainCfg:
    seed: int = 0
    max_updates: int = 1000
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


@dataclass
class Config:
    algo_name: str = "mixgrpo"
    env: EnvCfg = field(default_factory=EnvCfg)
    algo: AlgoCfg = field(default_factory=AlgoCfg)
    train: TrainCfg = field(default_factory=TrainCfg)


def _coerce(value: Any, default: Any) -> Any:
    """Coerce a YAML/CLI value to the type of the dataclass default."""
    if isinstance(default, tuple):
        if isinstance(value, str):
            value = ast.literal_eval(value)
        return tuple(value)
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


def _fill(cls, data: dict[str, Any]):
    base = cls()  # all defaults materialized (handles default_factory)
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise KeyError(f"{cls.__name__} got unknown keys: {sorted(unknown)}")
    kwargs = {}
    for name in known:
        if name in data and data[name] is not None:
            kwargs[name] = _coerce(data[name], getattr(base, name))
    return cls(**kwargs)


def _apply_overrides(tree: dict[str, Any], overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        keys = dotted.split(".")
        node = tree
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = raw


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> Config:
    with open(config_path, "r") as handle:
        tree = yaml.safe_load(handle) or {}
    if overrides:
        _apply_overrides(tree, overrides)
    cfg = Config(
        algo_name=str(tree.get("algo_name", "mixgrpo")),
        env=_fill(EnvCfg, tree.get("env", {}) or {}),
        algo=_fill(AlgoCfg, tree.get("algo", {}) or {}),
        train=_fill(TrainCfg, tree.get("train", {}) or {}),
    )
    # The env shares the algorithm's group size for observation-noise replication.
    cfg.env.num_generations = cfg.algo.num_generations
    return cfg
