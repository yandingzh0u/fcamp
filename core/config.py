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
    terrain_type: str = "slope"
    motion_file: str = ""
    max_episode_steps: int = -1
    motion_start_phase: int = 0
    motion_end_phase: int = -1
    reset_noise: bool = True
    interval_pushes: bool = True
    observation_noise: bool = True
    adaptive_motion_sampling: bool = True
    # Failure-predecessor sampler: death mass in bin b is shifted to b-lookback before reset,
    # then mixed with exact global-uniform exploration.
    adaptive_num_bins: int = 0          # 0 -> auto ⌊num_frames/fps⌋+1 (~1s bins)
    adaptive_alpha: float = 0.001
    adaptive_predecessor_ratio: float = 0.8
    adaptive_predecessor_lookback_bins: int = 1
    action_rate_weight: float = 0.1
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
    # Critic network. Decoupled from the actor so PPO and FPO can share an IDENTICAL value head
    # ([512,256,128]) while FPO keeps a wider flow actor ([1024,512,256]). The value estimation
    # problem is the same for both algorithms, so the critic architecture is shared.
    critic_hidden_dims: tuple[int, ...] = (512, 256, 128)
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
    # Independent value-function clip range for the clipped value loss (PPO-style). Kept separate
    # from the policy clip_range so FPO's tiny policy clip (0.01) never clips the critic.
    value_clip_range: float = 0.2
    adv_clip_max: float = 5.0
    desired_kl: float = 0.06
    entropy_coef: float = 0.005
    policy_epochs: int = 5
    num_mini_batches: int = 4
    micro_batch_size: int = 8192
    max_grad_norm: float = 1.0
    policy_lr: float = 1.0e-3
    value_lr: float = 1.0e-3   # FPO/MixGRPO critic LR. For FPO it follows the same adaptive
                               # multiplier as the actor, matching PPO's scheduler semantics.

    # --- PPO (actor-critic) specific. Unused by MixGRPO. ---
    num_steps_per_env: int = 24
    num_learning_epochs: int = 5
    gae_lambda: float = 0.95
    value_loss_coef: float = 1.0
    actor_learning_rate: float = 1.0e-3
    critic_learning_rate: float = 1.0e-3
    weight_decay: float = 0.0
    # Critic-only weight decay. Decoupled from the actor's weight_decay so the shared value head
    # can use 0 (PPO/FPO parity) while the actor keeps its own regularization.
    critic_weight_decay: float = 0.0
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
    num_micro_batches: int = 1                    # gradient-accum microbatches per logical minibatch.


@dataclass
class TrainCfg:
    seed: int = 0
    max_updates: int = 1000
    log_every: int = 1
    save_every: int = 500
    checkpoint_dir: str = ""
    resume: str = ""
    reset_optimizer_on_resume: bool = False
    reset_sampler_on_resume: bool = False
    validation_every: int = 0
    validation_max_steps: int = 500
    validation_start_phase: int = 0
    # Directional validation: an extra eval rollout started deep in the clip (e.g. phase 800)
    # that measures feasibility of the hard stand-up segment in isolation, separately from the
    # full phase-0 trajectory. Set < 0 to disable.
    validation_directional_start_phase: int = 800
    validation_fixed_seed: int = -1
    validation_preserve_state: bool = True
    validation_observation_noise: bool = False
    validation_done_frac_early_stop: float = 0.98
    target_validation_steps: int = 0
    success_checkpoint_name: str = "success_10s.pt"


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
