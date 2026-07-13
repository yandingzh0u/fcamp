from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, TypeAlias

import yaml


AlgorithmName: TypeAlias = Literal["ppo", "sfpo", "fcamp"]


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
class SFPOConfig:
    """SFPO's single supported training path.

    SFPO uses one actor-density path: a causal residual flow policy with
    action-chunk coefficients-preserving exploration, a chunk-start flow value critic,
    and no hand-written failure penalty.
    """

    horizon: int
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    activation: str
    action_squash_scale: float
    flow_steps: int
    cps_noise_level: float
    cps_trainable: bool
    cps_cov_rank: int
    rollout_env_steps: int
    discount_gamma: float
    gae_lambda: float
    clip_range: float
    desired_kl: float
    policy_epochs: int
    num_mini_batches: int
    micro_batch_size: int
    value_loss_coef: float
    policy_lr: float
    value_lr: float
    weight_decay: float
    critic_weight_decay: float
    empirical_normalization: bool
    init_at_random_ep_len: bool
    max_grad_norm: float
    # KL controller
    kl_early_stop_factor: float
    advantage_normalization: str


@dataclass(frozen=True, slots=True)
class AMPConfig:
    """Standard AMP discriminator and replay configuration.

    ``obs_steps`` counts states, not transitions.  FC-AMP uses 16 states at
    50 Hz to match the roughly 0.3 s receptive field of MimicKit's 10-state,
    30 Hz G1 setup.
    """

    enabled: bool
    obs_steps: int
    hidden_dims: tuple[int, ...]
    reward_scale: float
    reward_epsilon: float
    optimizer: str
    learning_rate: float
    weight_decay: float
    epochs: int
    batch_size: int
    current_buffer_size: int
    replay_size: int
    replay_samples: int
    replay_dtype: str
    replay_device: str
    grad_penalty: float
    logit_reg: float
    normalizer_clip: float
    reward_eval_batch_size: int
    max_updates_per_iteration: int


@dataclass(frozen=True, slots=True)
class FCAMPCreditConfig:
    mode: str
    task_weight: float
    amp_weight: float
    integrate_amp_reward_dt: bool
    advantage_normalization: str
    ratio_mode: str


@dataclass(frozen=True, slots=True)
class FCAMPCriticConfig:
    encoder_hidden_dims: tuple[int, ...]
    head_hidden_dims: tuple[int, ...]
    sharing: str
    task_loss_weight: float
    amp_loss_weight: float


@dataclass(frozen=True, slots=True)
class FCAMPConfig(SFPOConfig):
    """Full causal Flow-chunk AMP training path.

    The actor/CPS fields are inherited from :class:`SFPOConfig`.  FC-AMP adds
    a standard independent AMP discriminator, primitive causal credit, and a
    shared prefix encoder with task/AMP Flow-value heads.
    """

    amp: AMPConfig
    credit: FCAMPCreditConfig
    critics: FCAMPCriticConfig


AlgorithmConfig: TypeAlias = PPOConfig | SFPOConfig | FCAMPConfig


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
        return 1


ALGORITHM_CONFIGS = {
    "ppo": PPOConfig,
    "sfpo": SFPOConfig,
    "fcamp": FCAMPConfig,
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
    for name in (
        "actor_hidden_dims",
        "critic_hidden_dims",
        "hidden_dims",
        "encoder_hidden_dims",
        "head_hidden_dims",
    ):
        if name in converted:
            converted[name] = tuple(int(value) for value in converted[name])
    return cls(**converted)


def _construct_algorithm_config(algorithm: str, values: dict[str, Any]) -> AlgorithmConfig:
    cls = ALGORITHM_CONFIGS[algorithm]
    if cls is not FCAMPConfig:
        return _construct(cls, values)
    nested = dict(values)
    try:
        nested["amp"] = _construct(AMPConfig, dict(nested["amp"]))
        nested["credit"] = _construct(FCAMPCreditConfig, dict(nested["credit"]))
        nested["critics"] = _construct(FCAMPCriticConfig, dict(nested["critics"]))
    except KeyError as exc:
        raise KeyError(f"FCAMPConfig missing nested section: {exc.args[0]}") from exc
    return _construct(FCAMPConfig, nested)


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
    config = ExperimentConfig(
        algorithm=algorithm,
        environment=_construct(EnvironmentConfig, dict(tree["environment"])),
        parameters=_construct_algorithm_config(algorithm, parameter_values),
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
    if isinstance(config.parameters, SFPOConfig):
        if config.parameters.horizon < 1:
            raise ValueError("SFPO requires parameters.horizon >= 1")
        if config.parameters.flow_steps < 1:
            raise ValueError("SFPO requires parameters.flow_steps >= 1")
        if config.parameters.rollout_env_steps <= 0:
            raise ValueError("SFPO requires parameters.rollout_env_steps > 0")
        if config.parameters.rollout_env_steps % config.parameters.horizon:
            raise ValueError("parameters.rollout_env_steps must be divisible by parameters.horizon")
        if not (0.0 < config.parameters.cps_noise_level < 1.0):
            raise ValueError("SFPO requires parameters.cps_noise_level in (0, 1)")
        if config.parameters.cps_cov_rank < 0:
            raise ValueError("SFPO requires parameters.cps_cov_rank >= 0")
        norm = str(config.parameters.advantage_normalization).lower()
        if norm not in {"per_prefix", "global", "none"}:
            raise ValueError(
                "SFPO parameters.advantage_normalization must be one of "
                f"'per_prefix', 'global', 'none'; got {config.parameters.advantage_normalization!r}"
            )
    if isinstance(config.parameters, FCAMPConfig):
        amp = config.parameters.amp
        credit = config.parameters.credit
        critics = config.parameters.critics
        if not amp.enabled:
            raise ValueError("FC-AMP requires parameters.amp.enabled=true")
        if amp.obs_steps < 2:
            raise ValueError("FC-AMP requires amp.obs_steps >= 2")
        if not amp.hidden_dims:
            raise ValueError("FC-AMP discriminator hidden_dims cannot be empty")
        if amp.reward_scale <= 0.0 or not (0.0 < amp.reward_epsilon < 1.0):
            raise ValueError("FC-AMP AMP reward scale/epsilon are invalid")
        if amp.optimizer.lower() not in {"sgd", "adam", "adamw"}:
            raise ValueError("FC-AMP amp.optimizer must be sgd, adam, or adamw")
        if amp.learning_rate <= 0.0 or amp.batch_size < 1 or amp.epochs < 1:
            raise ValueError("FC-AMP discriminator optimizer/batch/epoch settings are invalid")
        if amp.max_updates_per_iteration < 1:
            raise ValueError("FC-AMP max_updates_per_iteration must be positive")
        if amp.current_buffer_size < amp.batch_size:
            raise ValueError("FC-AMP amp.current_buffer_size must be >= amp.batch_size")
        if amp.replay_size < amp.batch_size or amp.replay_samples < 0:
            raise ValueError("FC-AMP replay settings are invalid")
        if amp.replay_dtype.lower() not in {"float16", "float32"}:
            raise ValueError("FC-AMP replay_dtype must be float16 or float32")
        if amp.replay_device.lower() not in {"cpu", "cuda"}:
            raise ValueError("FC-AMP replay_device must be cpu or cuda")
        if credit.mode not in {"causal_frame", "chunk_shared"}:
            raise ValueError("FC-AMP credit.mode must be causal_frame or chunk_shared")
        if credit.advantage_normalization not in {
            "per_channel_per_offset", "per_channel_global", "none"
        }:
            raise ValueError("Unsupported FC-AMP advantage normalization")
        if credit.ratio_mode not in {"factorized", "mean_log", "joint_path"}:
            raise ValueError("FC-AMP ratio_mode must be factorized, mean_log, or joint_path")
        if credit.task_weight < 0.0 or credit.amp_weight < 0.0:
            raise ValueError("FC-AMP reward weights must be non-negative")
        if credit.task_weight == 0.0 and credit.amp_weight == 0.0:
            raise ValueError("FC-AMP requires at least one non-zero reward weight")
        if critics.sharing != "encoder":
            raise ValueError("Full FC-AMP requires critics.sharing=encoder")
        if not critics.encoder_hidden_dims or not critics.head_hidden_dims:
            raise ValueError("FC-AMP critic encoder/head dimensions cannot be empty")
