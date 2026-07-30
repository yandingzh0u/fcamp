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
    max_episode_steps: int
    motion_start_phase: int
    motion_end_phase: int
    root_velocity_mode: str


@dataclass(frozen=True, slots=True)
class AMPPolicyConfig:
    """Standard AMP Gaussian actor and PPO/value settings.

    The policy variable is a direct normalized absolute action, or an
    ``[H, action_dim]`` chunk of direct actions.
    """

    horizon: int
    actor_hidden_dims: tuple[int, ...]
    rollout_env_steps: int
    discount_gamma: float
    gae_lambda: float
    clip_range: float
    advantage_clip: float
    policy_epochs: int
    critic_epochs: int
    # MimicKit expresses logical optimizer batch sizes as multiples of the
    # environment count (4*N for actor, 2*N for critic).
    actor_batch_size: int
    critic_batch_size: int
    # Device-memory subdivision only.  Splitting a logical batch must
    # accumulate one equivalent gradient and perform exactly one optimizer
    # step; it must not create additional PPO updates.
    micro_batch_size: int
    policy_lr: float
    value_lr: float
    action_bound_weight: float


@dataclass(frozen=True, slots=True)
class AMPDiscriminatorConfig:
    """Unconditional temporal AMP discriminator settings."""

    obs_steps: int
    hidden_dims: tuple[int, ...]
    reward_scale: float
    reward_epsilon: float
    learning_rate: float
    weight_decay: float
    epochs: int
    # Logical discriminator batch multiplier: B = batch_size * num_envs.
    batch_size: int
    # Device-memory subdivision for an exactly equivalent accumulated logical
    # batch. It never changes the number of discriminator optimizer steps.
    micro_batch_size: int
    replay_size: int
    replay_samples: int
    grad_penalty: float
    logit_reg: float
    normalizer_clip: float
    reward_eval_batch_size: int


@dataclass(frozen=True, slots=True)
class AMPValueConfig:
    hidden_dims: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AMPConfig(AMPPolicyConfig):
    """MimicKit-style pure AMP adapted to the configured robot and motion."""

    style_prior: AMPDiscriminatorConfig
    critic: AMPValueConfig


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int
    max_updates: int
    log_every: int
    save_every: int
    resume: str
    reset_optimizer_on_resume: bool
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
    parameters: AMPConfig
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
        "hidden_dims",
    ):
        if name in converted:
            converted[name] = tuple(int(value) for value in converted[name])
    return cls(**converted)


def _construct_amp(values: dict[str, Any]) -> AMPConfig:
    nested = dict(values)
    try:
        nested["style_prior"] = _construct(
            AMPDiscriminatorConfig,
            dict(nested["style_prior"]),
        )
        nested["critic"] = _construct(
            AMPValueConfig,
            dict(nested["critic"]),
        )
    except KeyError as exc:
        raise KeyError(f"AMPConfig missing nested section: {exc.args[0]}") from exc
    return _construct(AMPConfig, nested)


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
    if method != "amp":
        raise ValueError(
            f"method must be 'amp', got {method!r}"
        )
    source_path = Path(source).expanduser().resolve()
    environment_values = dict(normalized["environment"])
    training_values = dict(normalized["training"])
    training_values["resume"] = _resolve_path(
        str(training_values["resume"]),
        source_path,
    )
    config = ExperimentConfig(
        method=method,
        environment=_construct(EnvironmentConfig, environment_values),
        parameters=_construct_amp(dict(normalized["parameters"])),
        training=_construct(TrainingConfig, training_values),
    )
    _validate(config)
    return config


def config_from_checkpoint_dict(
    tree: dict[str, Any],
    source: str | Path = ".",
) -> ExperimentConfig:
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


def _finite_positive(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


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
    if env.platform_profile not in {"custom", "g1_largebox_50hz"}:
        raise ValueError(
            "environment.platform_profile must be custom or g1_largebox_50hz"
        )
    if env.root_velocity_mode != "link":
        raise ValueError(
            "AMP requires environment.root_velocity_mode=link so reset, "
            "policy, expert, validation, and snapshot velocities share one "
            "root-link world-frame contract"
        )
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
    if train.save_every < 1:
        raise ValueError("training.save_every must be positive")
    if train.validation_every < 1:
        raise ValueError("training.validation_every must be positive")
    resolve_task(env.task)
    _validate_amp(config.parameters)


def _validate_amp(params: AMPConfig) -> None:
    if params.horizon < 1:
        raise ValueError("AMP Gaussian actor requires parameters.horizon >= 1")
    if params.rollout_env_steps <= 0:
        raise ValueError("AMP requires parameters.rollout_env_steps > 0")
    if (
        not params.actor_hidden_dims
        or any(width < 1 for width in params.actor_hidden_dims)
    ):
        raise ValueError("AMP actor_hidden_dims must be positive")
    if not 0.0 < params.discount_gamma <= 1.0:
        raise ValueError("parameters.discount_gamma must be in (0, 1]")
    if not 0.0 <= params.gae_lambda <= 1.0:
        raise ValueError("parameters.gae_lambda must be in [0, 1]")
    if not 0.0 < params.clip_range < 1.0:
        raise ValueError("parameters.clip_range must be in (0, 1)")
    if not _finite_positive(params.advantage_clip):
        raise ValueError("parameters.advantage_clip must be finite and positive")
    if (
        params.policy_epochs < 1
        or params.critic_epochs < 1
        or params.actor_batch_size < 1
        or params.critic_batch_size < 1
        or params.micro_batch_size < 1
    ):
        raise ValueError("AMP optimizer epoch/batch settings must be positive")
    if not _finite_positive(params.policy_lr):
        raise ValueError("AMP requires parameters.policy_lr > 0")
    if not _finite_positive(params.value_lr):
        raise ValueError("AMP requires parameters.value_lr > 0")
    if params.action_bound_weight < 0.0:
        raise ValueError("parameters.action_bound_weight cannot be negative")

    style = params.style_prior
    critic = params.critic
    if (
        not critic.hidden_dims
        or any(width < 1 for width in critic.hidden_dims)
    ):
        raise ValueError("AMP critic hidden_dims must be positive")
    if style.obs_steps < 2:
        raise ValueError("AMP requires style_prior.obs_steps >= 2")
    if not style.hidden_dims or any(width < 1 for width in style.hidden_dims):
        raise ValueError("AMP discriminator hidden_dims must be positive")
    if (
        style.reward_scale <= 0.0
        or not 0.0 < style.reward_epsilon < 1.0
    ):
        raise ValueError("AMP style reward scale/epsilon are invalid")
    if (
        not _finite_positive(style.learning_rate)
        or style.batch_size < 1
        or style.epochs < 1
    ):
        raise ValueError(
            "AMP discriminator optimizer/batch/epoch settings are invalid"
        )
    if (
        style.replay_size < style.batch_size
        or style.replay_samples <= 0
        or style.replay_samples > style.replay_size
    ):
        raise ValueError("AMP global replay settings are invalid")
    if style.grad_penalty < 0.0 or style.logit_reg < 0.0:
        raise ValueError(
            "AMP gradient penalty and logit regularization cannot be negative"
        )
    if not _finite_positive(style.normalizer_clip):
        raise ValueError(
            "style_prior.normalizer_clip must be finite and positive"
        )
    if style.reward_eval_batch_size < 1:
        raise ValueError(
            "style_prior.reward_eval_batch_size must be positive"
        )
