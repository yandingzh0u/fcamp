from __future__ import annotations

from dataclasses import dataclass, fields
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
    adaptive_num_bins: int
    adaptive_alpha: float
    adaptive_predecessor_ratio: float
    adaptive_predecessor_lookback_bins: int
    action_rate_weight: float
    root_velocity_mode: str


@dataclass(frozen=True, slots=True)
class FlowCPSConfig:
    """Shared Flow-CPS actor and optimizer settings."""

    horizon: int
    actor_hidden_dims: tuple[int, ...]
    activation: str
    action_squash_scale: float
    flow_steps: int
    cps_noise_init: float
    cps_cov_rank: int
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
    kl_early_stop_factor: float


@dataclass(frozen=True, slots=True)
class FixedRewardCriticConfig:
    encoder_hidden_dims: tuple[int, ...]
    head_hidden_dims: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FixedRewardStreamsConfig:
    """Fixed phase-zero trajectory-attempt/curriculum mixture."""

    phase0_fraction: float


@dataclass(frozen=True, slots=True)
class FixedRewardConfig(FlowCPSConfig):
    """Task-only fixed-pose-reward Flow-CPS training configuration."""

    critic: FixedRewardCriticConfig
    streams: FixedRewardStreamsConfig


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
    target_validation_steps: int


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    method: str
    environment: EnvironmentConfig
    parameters: FixedRewardConfig
    training: TrainingConfig


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
        "encoder_hidden_dims",
        "head_hidden_dims",
    ):
        if name in converted:
            converted[name] = tuple(int(value) for value in converted[name])
    return cls(**converted)


def _construct_fixed_reward(values: dict[str, Any]) -> FixedRewardConfig:
    nested = dict(values)
    try:
        nested["critic"] = _construct(
            FixedRewardCriticConfig,
            dict(nested["critic"]),
        )
        nested["streams"] = _construct(
            FixedRewardStreamsConfig,
            dict(nested["streams"]),
        )
    except KeyError as exc:
        raise KeyError(
            f"FixedRewardConfig missing nested section: {exc.args[0]}"
        ) from exc
    return _construct(FixedRewardConfig, nested)


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


def config_from_dict(
    tree: dict[str, Any],
    source: str | Path = ".",
) -> ExperimentConfig:
    normalized = dict(tree)
    required = {"method", "environment", "parameters", "training"}
    missing = required - normalized.keys()
    unknown = normalized.keys() - required
    if missing:
        raise KeyError(f"ExperimentConfig missing keys: {sorted(missing)}")
    if unknown:
        raise KeyError(f"ExperimentConfig unknown keys: {sorted(unknown)}")

    method = str(normalized["method"])
    if method != "fixed_reward":
        raise ValueError(
            "Only method='fixed_reward' is supported; legacy checkpoints and "
            "configurations must start a fresh run."
        )

    source_path = Path(source).expanduser().resolve()
    training_values = dict(normalized["training"])
    training_values["resume"] = _resolve_path(
        str(training_values["resume"]),
        source_path,
    )
    config = ExperimentConfig(
        method=method,
        environment=_construct(
            EnvironmentConfig,
            dict(normalized["environment"]),
        ),
        parameters=_construct_fixed_reward(dict(normalized["parameters"])),
        training=_construct(TrainingConfig, training_values),
    )
    _validate(config)
    return config


def config_from_checkpoint_dict(
    tree: dict[str, Any],
    source: str | Path = ".",
) -> ExperimentConfig:
    """Load a fixed-reward checkpoint configuration without legacy coercion."""

    if not isinstance(tree, dict):
        raise ValueError("Checkpoint config must be a mapping.")
    if tree.get("method") != "fixed_reward":
        raise ValueError(
            "Legacy checkpoints are incompatible with fixed_reward; "
            "start a fresh run."
        )
    return config_from_dict(tree, source)


def load_config(
    config_path: str | Path,
    overrides: list[str] | None = None,
) -> ExperimentConfig:
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
    params = config.parameters

    if env.num_envs < 2:
        raise ValueError(
            "fixed_reward requires at least two environments for two streams"
        )
    if env.sim_dt <= 0.0:
        raise ValueError("environment.sim_dt must be positive")
    if env.decimation < 1:
        raise ValueError("environment.decimation must be positive")
    if env.platform_profile not in {"custom", "g1_largebox_50hz"}:
        raise ValueError(
            "environment.platform_profile must be custom or g1_largebox_50hz"
        )
    if env.reset_phase_sampling not in {
        "adaptive",
        "uniform",
        "rsi",
        "zero",
    }:
        raise ValueError(
            "environment.reset_phase_sampling must be one of "
            "adaptive/uniform/rsi/zero"
        )
    if env.rsi_keyframe_count < 1:
        raise ValueError("environment.rsi_keyframe_count must be positive")
    if abs(float(env.action_rate_weight) - 0.1) > 1.0e-12:
        raise ValueError(
            "fixed_reward requires environment.action_rate_weight=0.1"
        )
    if env.root_velocity_mode != "link":
        raise ValueError("fixed_reward requires environment.root_velocity_mode=link")
    if env.platform_profile == "g1_largebox_50hz":
        if env.task != "largebox_plane":
            raise ValueError(
                "g1_largebox_50hz requires environment.task=largebox_plane"
            )
        if abs(float(env.sim_dt) - 0.02) > 1.0e-12 or env.decimation != 4:
            raise ValueError(
                "g1_largebox_50hz requires 50 Hz control / 200 Hz simulation"
            )
        if env.fix_root_link:
            raise ValueError("g1_largebox_50hz requires a free root link")
        if int(env.max_episode_steps) != 500:
            raise ValueError(
                "g1_largebox_50hz uses a 10 second, 500-step episode limit"
            )

    if train.max_updates < 1:
        raise ValueError("training.max_updates must be positive")
    if train.log_every < 1:
        raise ValueError("training.log_every must be positive")
    if train.save_every < 0:
        raise ValueError("training.save_every must be non-negative")
    if train.validation_every < 0:
        raise ValueError("training.validation_every must be non-negative")
    if train.validation_max_steps < 1:
        raise ValueError("training.validation_max_steps must be positive")

    resolve_task(env.task)
    _validate_fixed_reward(params)


def _validate_fixed_reward(params: FixedRewardConfig) -> None:
    if params.horizon < 1:
        raise ValueError("Flow-CPS requires parameters.horizon >= 1")
    if params.flow_steps < 1:
        raise ValueError("Flow-CPS requires parameters.flow_steps >= 1")
    if params.rollout_env_steps <= 0:
        raise ValueError("Flow-CPS requires parameters.rollout_env_steps > 0")
    if params.rollout_env_steps % params.horizon:
        raise ValueError(
            "parameters.rollout_env_steps must be divisible by parameters.horizon"
        )
    if not (1.0e-4 < params.cps_noise_init < 1.0 - 1.0e-4):
        raise ValueError(
            "Flow-CPS requires parameters.cps_noise_init in "
            "(1e-4, 1-1e-4)"
        )
    if params.cps_cov_rank < 0:
        raise ValueError("Flow-CPS requires parameters.cps_cov_rank >= 0")
    if params.policy_lr <= 0.0:
        raise ValueError("Flow-CPS requires parameters.policy_lr > 0")
    if params.value_lr <= 0.0:
        raise ValueError("Flow-CPS requires parameters.value_lr > 0")
    if abs(float(params.streams.phase0_fraction) - 0.10) > 1.0e-12:
        raise ValueError(
            "fixed_reward requires streams.phase0_fraction=0.10"
        )
    critic = params.critic
    if (
        not critic.encoder_hidden_dims
        or not critic.head_hidden_dims
        or any(width < 1 for width in critic.encoder_hidden_dims)
        or any(width < 1 for width in critic.head_hidden_dims)
    ):
        raise ValueError(
            "fixed_reward critic encoder/head dimensions must be positive"
        )
