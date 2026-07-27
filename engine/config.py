from __future__ import annotations

from dataclasses import dataclass, fields
import math
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    platform_profile: str
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
    reset_phase_sampling: str
    rsi_keyframe_count: int
    adaptive_motion_sampling: bool
    adaptive_num_bins: int
    adaptive_alpha: float
    adaptive_predecessor_ratio: float
    adaptive_predecessor_lookback_bins: int
    action_rate_weight: float
    policy_action_bound: float
    command_rate_limit: tuple[float, ...]
    rate_half_life_seconds: float
    root_velocity_mode: str
    physics_material_combine_mode: str
    contact_sensor_update_period: str


@dataclass(frozen=True, slots=True)
class StylePriorConfig:
    """Temporal discriminator prior configuration used by FCAMP."""

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
    grad_penalty: float
    logit_reg: float
    normalizer_clip: float
    reward_eval_batch_size: int
    max_updates_per_iteration: int
    discriminator_warmup_rollouts: int = 0


@dataclass(frozen=True, slots=True)
class FCAMPCreditConfig:
    task_weight: float
    amp_weight: float


@dataclass(frozen=True, slots=True)
class FCAMPCriticConfig:
    encoder_hidden_dims: tuple[int, ...]
    head_hidden_dims: tuple[int, ...]
    task_loss_weight: float
    amp_loss_weight: float


@dataclass(frozen=True, slots=True)
class FCAMPStreamsConfig:
    """Fixed phase-zero trajectory-attempt/curriculum mixture used by FCAMP."""

    phase0_fraction: float


@dataclass(frozen=True, slots=True)
class FCAMPConfig:
    """Causal Flow-CPS policy with a temporal discriminator prior."""

    horizon: int
    actor_hidden_dims: tuple[int, ...]
    activation: str
    flow_steps: int
    cps_physical_rms: float
    rollout_env_steps: int
    discount_gamma: float
    gae_lambda: float
    clip_range: float
    desired_kl: float
    policy_epochs: int
    num_mini_batches: int
    micro_batch_size: int
    policy_lr: float
    value_lr: float
    weight_decay: float
    critic_weight_decay: float
    init_at_random_ep_len: bool
    max_grad_norm: float
    kl_acceptance_factor: float
    style_prior: StylePriorConfig
    credit: FCAMPCreditConfig
    critics: FCAMPCriticConfig
    streams: FCAMPStreamsConfig


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int
    max_updates: int
    log_every: int
    save_every: int
    official_reset_every: int
    resume: str
    reset_optimizer_on_resume: bool
    reset_sampler_on_resume: bool
    validation_every: int
    validation_max_steps: int
    validation_start_phase: int
    validation_directional_start_phase: int
    validation_fixed_seed: int
    target_validation_steps: int


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    method: str
    environment: EnvironmentConfig
    parameters: FCAMPConfig
    training: TrainingConfig

    @property
    def observation_group_size(self) -> int:
        return 1


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
        "disc_hidden_dims",
        "hidden_dims",
        "encoder_hidden_dims",
        "head_hidden_dims",
        "command_rate_limit",
    ):
        if name in converted:
            converter = float if name == "command_rate_limit" else int
            converted[name] = tuple(converter(value) for value in converted[name])
    return cls(**converted)


def _construct_fcamp_config(values: dict[str, Any]) -> FCAMPConfig:
    nested = dict(values)
    try:
        nested["style_prior"] = _construct(
            StylePriorConfig, dict(nested["style_prior"])
        )
        nested["credit"] = _construct(FCAMPCreditConfig, dict(nested["credit"]))
        nested["critics"] = _construct(FCAMPCriticConfig, dict(nested["critics"]))
        nested["streams"] = _construct(FCAMPStreamsConfig, dict(nested["streams"]))
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
    normalized = dict(tree)
    required = {"method", "environment", "parameters", "training"}
    missing = required - normalized.keys()
    unknown = normalized.keys() - required
    if missing:
        raise KeyError(f"ExperimentConfig missing keys: {sorted(missing)}")
    if unknown:
        raise KeyError(f"ExperimentConfig unknown keys: {sorted(unknown)}")
    method = str(normalized["method"])
    if method != "fcamp":
        raise ValueError(f"Only method='fcamp' is supported, got {method!r}")
    source_path = Path(source).expanduser().resolve()
    environment_values = dict(normalized["environment"])
    environment_values.setdefault("platform_profile", "custom")
    environment_values.setdefault("root_velocity_mode", "com")
    environment_values.setdefault("physics_material_combine_mode", "average")
    environment_values.setdefault("contact_sensor_update_period", "control")
    training_values = dict(normalized["training"])
    training_values.setdefault("official_reset_every", 0)
    training_values["resume"] = _resolve_path(str(training_values["resume"]), source_path)
    config = ExperimentConfig(
        method=method,
        environment=_construct(EnvironmentConfig, environment_values),
        parameters=_construct_fcamp_config(dict(normalized["parameters"])),
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
    from envs.tasks import resolve_task

    env = config.environment
    train = config.training
    if env.num_envs < 1:
        raise ValueError("environment.num_envs must be positive")
    if env.sim_dt <= 0.0:
        raise ValueError("environment.sim_dt must be positive")
    if env.decimation < 1:
        raise ValueError("environment.decimation must be positive")
    if (
        not math.isfinite(env.policy_action_bound)
        or env.policy_action_bound <= 0.0
    ):
        raise ValueError(
            "environment.policy_action_bound must be finite and positive"
        )
    if len(env.command_rate_limit) != 29 or any(
        not math.isfinite(limit) or limit <= 0.0
        for limit in env.command_rate_limit
    ):
        raise ValueError(
            "environment.command_rate_limit must contain 29 finite positive values"
        )
    if (
        not math.isfinite(env.rate_half_life_seconds)
        or env.rate_half_life_seconds <= 0.0
    ):
        raise ValueError(
            "environment.rate_half_life_seconds must be finite and positive"
        )
    if env.platform_profile not in {"custom", "g1_largebox_50hz"}:
        raise ValueError("environment.platform_profile must be custom or g1_largebox_50hz")
    if env.reset_phase_sampling not in {
        "adaptive",
        "uniform",
        "continuous_uniform",
        "rsi",
        "zero",
    }:
        raise ValueError(
            "environment.reset_phase_sampling must be one of "
            "adaptive/uniform/continuous_uniform/rsi/zero"
        )
    if env.rsi_keyframe_count < 1:
        raise ValueError("environment.rsi_keyframe_count must be positive")
    if env.adaptive_motion_sampling != (env.reset_phase_sampling == "adaptive"):
        raise ValueError(
            "environment.adaptive_motion_sampling must match adaptive reset_phase_sampling"
        )
    if env.root_velocity_mode not in {"com", "link"}:
        raise ValueError("environment.root_velocity_mode must be com or link")
    if env.physics_material_combine_mode not in {"average", "multiply"}:
        raise ValueError("environment.physics_material_combine_mode must be average or multiply")
    if env.contact_sensor_update_period not in {"control", "physics"}:
        raise ValueError("environment.contact_sensor_update_period must be control or physics")
    if env.platform_profile == "g1_largebox_50hz":
        if env.task != "largebox_plane":
            raise ValueError("g1_largebox_50hz requires environment.task=largebox_plane")
        if abs(float(env.sim_dt) - 0.02) > 1.0e-12 or env.decimation != 4:
            raise ValueError("g1_largebox_50hz requires 50 Hz control / 200 Hz simulation")
        if env.fix_root_link:
            raise ValueError("g1_largebox_50hz requires a free root link")
        if int(env.max_episode_steps) != 500:
            raise ValueError("g1_largebox_50hz uses a 10 second, 500-step episode limit")
    if train.max_updates < 1:
        raise ValueError("training.max_updates must be positive")
    if train.log_every < 1:
        raise ValueError("training.log_every must be positive")
    if train.official_reset_every < 0:
        raise ValueError("training.official_reset_every must be non-negative")
    resolve_task(env.task)
    if env.num_envs < 2:
        raise ValueError("FCAMP requires at least two environments for two streams")
    if env.reset_phase_sampling == "continuous_uniform":
        raise ValueError("FCAMP fixed-window reset history requires integer phases")
    _validate_fcamp(config.parameters)

def _validate_fcamp(params: FCAMPConfig) -> None:
    if params.horizon < 1:
        raise ValueError("Flow-CPS requires parameters.horizon >= 1")
    if params.flow_steps < 1:
        raise ValueError("Flow-CPS requires parameters.flow_steps >= 1")
    if params.rollout_env_steps <= 0:
        raise ValueError("Flow-CPS requires parameters.rollout_env_steps > 0")
    if params.rollout_env_steps % params.horizon:
        raise ValueError("parameters.rollout_env_steps must be divisible by parameters.horizon")
    if not math.isfinite(params.cps_physical_rms) or params.cps_physical_rms <= 0.0:
        raise ValueError(
            "Flow-CPS requires parameters.cps_physical_rms to be finite and positive"
        )
    if (
        not math.isfinite(params.desired_kl)
        or params.desired_kl <= 0.0
        or not math.isfinite(params.kl_acceptance_factor)
        or params.kl_acceptance_factor <= 0.0
    ):
        raise ValueError(
            "FCAMP requires positive desired_kl and kl_acceptance_factor"
        )
    style = params.style_prior
    credit = params.credit
    critics = params.critics
    streams = params.streams
    if not (0.0 < streams.phase0_fraction < 1.0):
        raise ValueError("FCAMP streams.phase0_fraction must be in (0, 1)")
    if style.obs_steps < 2:
        raise ValueError("FCAMP requires style_prior.obs_steps >= 2")
    if not style.hidden_dims:
        raise ValueError("FCstyle discriminator hidden_dims cannot be empty")
    if style.reward_scale <= 0.0 or not (0.0 < style.reward_epsilon < 1.0):
        raise ValueError("FCAMP style reward scale/epsilon are invalid")
    if style.optimizer.lower() not in {"sgd", "adam", "adamw"}:
        raise ValueError("FCAMP style_prior.optimizer must be sgd, adam, or adamw")
    if style.learning_rate <= 0.0 or style.batch_size < 2 or style.epochs < 1:
        raise ValueError("FCstyle discriminator optimizer/batch/epoch settings are invalid")
    if style.max_updates_per_iteration < 1:
        raise ValueError("FCAMP max_updates_per_iteration must be positive")
    if style.discriminator_warmup_rollouts not in {0, 1}:
        raise ValueError("FCAMP discriminator_warmup_rollouts must be 0 or 1")
    if style.current_buffer_size < style.batch_size:
        raise ValueError("FCAMP style_prior.current_buffer_size must be >= batch_size")
    current_phase0 = int(
        round(style.current_buffer_size * streams.phase0_fraction)
    )
    if not 0 < current_phase0 < style.current_buffer_size:
        raise ValueError(
            "FCAMP current discriminator buffer cannot realize both streams"
        )
    if (
        style.replay_size < style.batch_size
        or style.replay_samples <= 0
        or style.replay_samples > style.replay_size
    ):
        raise ValueError("FCAMP complete-window replay settings are invalid")
    phase0_capacity = int(round(style.replay_size * streams.phase0_fraction))
    phase0_replace = int(round(style.replay_samples * streams.phase0_fraction))
    if not (
        0 < phase0_capacity < style.replay_size
        and 0 < phase0_replace < style.replay_samples
        and phase0_replace <= phase0_capacity
        and style.replay_samples - phase0_replace
        <= style.replay_size - phase0_capacity
    ):
        raise ValueError(
            "FCAMP replay size/replacement quotas cannot realize both streams"
        )
    if credit.task_weight < 0.0 or credit.amp_weight < 0.0:
        raise ValueError("FCstyle reward weights must be non-negative")
    if credit.task_weight == 0.0 and credit.amp_weight == 0.0:
        raise ValueError("FCAMP requires at least one non-zero reward weight")
    if not critics.encoder_hidden_dims or not critics.head_hidden_dims:
        raise ValueError("FCAMP critic encoder/head dimensions cannot be empty")
