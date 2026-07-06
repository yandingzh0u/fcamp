from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, TypeAlias

import yaml


AlgorithmName: TypeAlias = Literal["ppo", "fpo", "mixgrpo", "sfpo"]


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    task: str
    device: str
    num_envs: int
    sim_dt: float
    decimation: int
    fix_root_link: bool
    startup_randomization: bool
    max_episode_steps: int
    motion_start_phase: int
    motion_end_phase: int
    reset_noise: bool
    interval_pushes: bool
    observation_noise: bool
    adaptive_motion_sampling: bool
    adaptive_num_bins: int
    adaptive_alpha: float
    adaptive_predecessor_ratio: float
    adaptive_predecessor_lookback_bins: int
    action_rate_weight: float


@dataclass(frozen=True, slots=True)
class PPOConfig:
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    activation: str
    init_noise_std: float
    discount_gamma: float
    num_steps_per_env: int
    num_learning_epochs: int
    num_mini_batches: int
    gae_lambda: float
    clip_range: float
    value_clip_range: float
    entropy_coef: float
    value_loss_coef: float
    desired_kl: float
    actor_learning_rate: float
    critic_learning_rate: float
    weight_decay: float
    critic_weight_decay: float
    empirical_normalization: bool
    init_at_random_ep_len: bool
    max_grad_norm: float

    @property
    def horizon(self) -> int:
        return 1


@dataclass(frozen=True, slots=True)
class FPOConfig:
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    activation: str
    flow_steps: int
    actor_scale: float
    mlp_output_scale: float
    timestep_embed_dim: int
    cfm_loss_reduction: str
    action_perturb_std: float
    cfm_loss_t_inverse_cdf_beta: float
    discount_gamma: float
    num_steps_per_env: int
    fpo_num_mc: int
    fpo_delta_clip: float
    fpo_cfm_loss_clamp: float
    cfm_loss_clamp_neg_adv: bool
    cfm_loss_clamp_neg_adv_max: float
    fpo_adv_clamp: float
    clip_range: float
    value_clip_range: float
    use_clipped_value_loss: bool
    schedule: str
    desired_kl: float
    num_learning_epochs: int
    num_mini_batches: int
    num_micro_batches: int
    gae_lambda: float
    value_loss_coef: float
    policy_lr: float
    value_lr: float
    weight_decay: float
    critic_weight_decay: float
    max_grad_norm: float
    empirical_normalization: bool
    init_at_random_ep_len: bool

    @property
    def horizon(self) -> int:
        return 1


@dataclass(frozen=True, slots=True)
class MixGRPOConfig:
    horizon: int
    actor_hidden_dims: tuple[int, ...]
    activation: str
    action_squash_scale: float
    flow_steps: int
    sde_eta: float
    init_noise_std: float
    init_same_noise: bool
    first_generation_zero_noise: bool
    eval_initial_noise: str
    num_generations: int
    rollout_env_steps: int
    tail_bootstrap_steps: int
    terminal_penalty: float
    discount_gamma: float
    clip_range: float
    adv_clip_max: float
    desired_kl: float
    entropy_coef: float
    policy_epochs: int
    num_mini_batches: int
    micro_batch_size: int
    max_grad_norm: float
    policy_lr: float


@dataclass(frozen=True, slots=True)
class SFPOConfig:
    """Causal smooth-chunk SFPO (root-cause redesign).

    Fixes the PPO/GRPO-to-chunk-flow unit mismatches:

      * action chunk is a previous-action-conditioned smooth trajectory
        (bounded delta from the last executed action), NOT absolute actions --
        matches the environment's action-rate penalty contract so open-loop
        chunk execution stays continuous;
      * causal flow velocity over horizon: ``v_k`` only sees ``z_0..z_k`` so
        the per-frame flow log-prob is a valid conditional density for PPO;
      * state-only V critic (the action-conditioned Q prefix is dropped: it
        was trained but never wired into the actor advantage);
      * per-frame GAE advantage (frame-j unit, lambda smoothing, cross-chunk
        propagation) instead of the multi-prefix objective
        ``A_k = T_{k+1} - V(s_0)`` which mixed early-reward credit;
      * per-frame PPO ratio + flat clip over the causal conditional factors;
      * terminal failure cost: an immediate per-step penalty added to the
        failure frame's reward with bootstrap=0 (true terminal), NOT an
        absorbing -10 bootstrap that saturated the value distribution;
      * KL controller updated before each minibatch optimizer step
        (PPO-aligned) with actor epoch early-stop on prefix KL.
    """

    horizon: int
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    activation: str
    action_squash_scale: float
    flow_steps: int
    sde_eta: float
    init_noise_std: float
    eval_initial_noise: str
    rollout_env_steps: int
    discount_gamma: float
    gae_lambda: float
    clip_range: float
    desired_kl: float
    policy_epochs: int
    num_mini_batches: int
    micro_batch_size: int
    value_loss_coef: float
    value_clip_range: float
    use_clipped_value_loss: bool
    policy_lr: float
    value_lr: float
    weight_decay: float
    critic_weight_decay: float
    empirical_normalization: bool
    init_at_random_ep_len: bool
    max_grad_norm: float
    # causal flow policy: v_k depends only on z_0..z_k (prefix-cumsum), so
    # logp_k is a genuine conditional density and per-frame PPO ratio is valid.
    causal_velocity: bool
    causal_arch: str
    # smooth action chunk
    action_max_delta: float
    # terminal failure cost
    failure_penalty: float
    # KL controller
    kl_early_stop_factor: float
    advantage_normalization: str


AlgorithmConfig: TypeAlias = PPOConfig | FPOConfig | MixGRPOConfig | SFPOConfig


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int
    max_updates: int
    log_every: int
    save_every: int
    resume: str
    reset_optimizer_on_resume: bool
    reset_sampler_on_resume: bool
    validation_every: int
    validation_max_steps: int
    validation_start_phase: int
    validation_directional_start_phase: int
    validation_fixed_seed: int
    validation_done_frac_early_stop: float
    target_validation_steps: int


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    algorithm: AlgorithmName
    environment: EnvironmentConfig
    parameters: AlgorithmConfig
    training: TrainingConfig

    @property
    def observation_group_size(self) -> int:
        if isinstance(self.parameters, MixGRPOConfig):
            return self.parameters.num_generations
        return 1


ALGORITHM_CONFIGS = {
    "ppo": PPOConfig,
    "fpo": FPOConfig,
    "mixgrpo": MixGRPOConfig,
    "sfpo": SFPOConfig,
}


def _construct(cls, values: dict[str, Any]):
    names = {field.name for field in fields(cls)}
    missing = names - values.keys()
    unknown = values.keys() - names
    if missing:
        raise KeyError(f"{cls.__name__} missing keys: {sorted(missing)}")
    if unknown:
        raise KeyError(f"{cls.__name__} unknown keys: {sorted(unknown)}")
    converted = dict(values)
    for name in ("actor_hidden_dims", "critic_hidden_dims"):
        if name in converted:
            converted[name] = tuple(int(value) for value in converted[name])
    return cls(**converted)


def _apply_overrides(tree: dict[str, Any], overrides: list[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        keys = dotted.split(".")
        node: Any = tree
        for key in keys[:-1]:
            if not isinstance(node, dict) or key not in node:
                raise KeyError(f"Unknown override path: {dotted}")
            node = node[key]
        if not isinstance(node, dict) or keys[-1] not in node:
            raise KeyError(f"Unknown override path: {dotted}")
        node[keys[-1]] = yaml.safe_load(raw)


def _resolve_path(value: str, config_path: Path) -> str:
    if not value:
        return ""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_path.parent / path
    return str(path.resolve())


def config_from_dict(tree: dict[str, Any], source: str | Path = ".") -> ExperimentConfig:
    required = {"algorithm", "environment", "parameters", "training"}
    missing = required - tree.keys()
    unknown = tree.keys() - required
    if missing:
        raise KeyError(f"ExperimentConfig missing keys: {sorted(missing)}")
    if unknown:
        raise KeyError(f"ExperimentConfig unknown keys: {sorted(unknown)}")
    algorithm = str(tree["algorithm"])
    if algorithm not in ALGORITHM_CONFIGS:
        raise ValueError(f"algorithm must be one of {sorted(ALGORITHM_CONFIGS)}, got {algorithm!r}")
    source_path = Path(source).expanduser().resolve()
    training_values = dict(tree["training"])
    training_values["resume"] = _resolve_path(str(training_values["resume"]), source_path)
    parameter_values = dict(tree["parameters"])
    if algorithm == "fpo":
        parameter_values.setdefault("use_clipped_value_loss", False)
    config = ExperimentConfig(
        algorithm=algorithm,
        environment=_construct(EnvironmentConfig, dict(tree["environment"])),
        parameters=_construct(ALGORITHM_CONFIGS[algorithm], parameter_values),
        training=_construct(TrainingConfig, training_values),
    )
    _validate(config)
    return config


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> ExperimentConfig:
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        tree = yaml.safe_load(handle) or {}
    if overrides:
        _apply_overrides(tree, overrides)
    return config_from_dict(tree, path)


def _validate(config: ExperimentConfig) -> None:
    from env.tasks import resolve_task

    env = config.environment
    train = config.training
    if env.num_envs < 1:
        raise ValueError("environment.num_envs must be positive")
    if env.sim_dt <= 0.0:
        raise ValueError("environment.sim_dt must be positive")
    if env.decimation < 1:
        raise ValueError("environment.decimation must be positive")
    if train.max_updates < 1:
        raise ValueError("training.max_updates must be positive")
    if train.log_every < 1:
        raise ValueError("training.log_every must be positive")
    resolve_task(env.task)
    if isinstance(config.parameters, MixGRPOConfig):
        if config.parameters.num_generations < 2:
            raise ValueError("MixGRPO requires parameters.num_generations >= 2")
        if env.num_envs % config.parameters.num_generations:
            raise ValueError("environment.num_envs must be divisible by parameters.num_generations")
    if isinstance(config.parameters, SFPOConfig):
        if config.parameters.horizon < 1:
            raise ValueError("SFPO requires parameters.horizon >= 1")
        if config.parameters.flow_steps < 1:
            raise ValueError("SFPO requires parameters.flow_steps >= 1")
        if config.parameters.rollout_env_steps <= 0:
            raise ValueError("SFPO requires parameters.rollout_env_steps > 0")
        if config.parameters.rollout_env_steps % config.parameters.horizon:
            raise ValueError("parameters.rollout_env_steps must be divisible by parameters.horizon")
        if config.parameters.failure_penalty < 0.0:
            raise ValueError("SFPO requires parameters.failure_penalty >= 0")
        norm = str(config.parameters.advantage_normalization).lower()
        if norm not in {"per_prefix", "global", "none"}:
            raise ValueError(
                "SFPO parameters.advantage_normalization must be one of "
                f"'per_prefix', 'global', 'none'; got {config.parameters.advantage_normalization!r}"
            )
