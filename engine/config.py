from __future__ import annotations

from dataclasses import dataclass, fields
import math
from pathlib import Path
from typing import Any, TypeAlias

import yaml


MethodName: TypeAlias = str


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
    command_servo_omega: float
    terminate_on_motion_end: bool
    motion_reference_mode: str
    root_velocity_mode: str
    policy_observation_mode: str
    motion_end_behavior: str
    physics_material_combine_mode: str
    contact_sensor_update_period: str


@dataclass(frozen=True, slots=True)
class FlowCPSConfig:
    """Shared Flow-CPS actor settings used by FCAMP."""

    horizon: int
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    activation: str
    flow_steps: int
    cps_physical_rms: float
    cps_trainable: bool
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
    kl_acceptance_factor: float
    advantage_normalization: str


@dataclass(frozen=True, slots=True)
class StylePriorConfig:
    """Temporal discriminator prior configuration used by FCAMP."""

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
    discriminator_warmup_rollouts: int = 0


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
class FCAMPStreamsConfig:
    """Fixed phase-zero trajectory-attempt/curriculum mixture used by FCAMP."""

    phase0_fraction: float


@dataclass(frozen=True, slots=True)
class FCAMPConfig(FlowCPSConfig):
    """Full causal Flow-CPS + temporal discriminator training path."""

    style_prior: StylePriorConfig
    credit: FCAMPCreditConfig
    critics: FCAMPCriticConfig
    streams: FCAMPStreamsConfig

    @property
    def amp(self) -> StylePriorConfig:
        """Compatibility alias for older FCAMP internals."""
        return self.style_prior


MethodConfig: TypeAlias = FCAMPConfig


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
    method: MethodName
    environment: EnvironmentConfig
    parameters: MethodConfig
    training: TrainingConfig

    @property
    def algorithm(self) -> str:
        """Legacy checkpoint/log alias."""
        return self.method

    @property
    def observation_group_size(self) -> int:
        return 1


METHOD_CONFIGS = {
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


def _construct_method_config(method: str, values: dict[str, Any], source_path: Path) -> MethodConfig:
    if method != "fcamp":
        raise ValueError(f"Unknown method {method!r}. Only FCAMP is supported.")
    nested = dict(values)
    if "style_prior" not in nested and "amp" in nested:
        nested["style_prior"] = nested.pop("amp")
    try:
        style_prior = dict(nested["style_prior"])
        # Preserve compatibility with configs written before the explicit
        # discriminator warm-up lifecycle was introduced.
        style_prior.setdefault("discriminator_warmup_rollouts", 0)
        nested["style_prior"] = _construct(StylePriorConfig, style_prior)
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
    if "method" not in normalized and "algorithm" in normalized:
        normalized["method"] = normalized.pop("algorithm")
    required = {"method", "environment", "parameters", "training"}
    missing = required - normalized.keys()
    unknown = normalized.keys() - required
    if missing:
        raise KeyError(f"ExperimentConfig missing keys: {sorted(missing)}")
    if unknown:
        raise KeyError(f"ExperimentConfig unknown keys: {sorted(unknown)}")
    method = str(normalized["method"])
    if method not in METHOD_CONFIGS:
        raise ValueError(f"method must be one of {sorted(METHOD_CONFIGS)}, got {method!r}")
    source_path = Path(source).expanduser().resolve()
    environment_values = dict(normalized["environment"])
    environment_values.setdefault("platform_profile", "custom")
    environment_values.setdefault("terminate_on_motion_end", False)
    environment_values.setdefault("motion_reference_mode", "frame")
    environment_values.setdefault("root_velocity_mode", "com")
    environment_values.setdefault("policy_observation_mode", "tracking")
    environment_values.setdefault("motion_end_behavior", "hold_last")
    environment_values.setdefault("physics_material_combine_mode", "average")
    environment_values.setdefault("contact_sensor_update_period", "control")
    training_values = dict(normalized["training"])
    training_values.setdefault("official_reset_every", 0)
    training_values["resume"] = _resolve_path(str(training_values["resume"]), source_path)
    parameters_values = dict(normalized["parameters"])
    # Reference Flow-CPS / throwaway base critic only; FCAMP uses critics.*.
    parameters_values.setdefault("critic_hidden_dims", (512, 256, 128))
    parameters_values.setdefault("value_loss_coef", 1.0)
    parameters_values.setdefault("advantage_normalization", "global")
    config = ExperimentConfig(
        method=method,
        environment=_construct(EnvironmentConfig, environment_values),
        parameters=_construct_method_config(method, parameters_values, source_path),
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
    if (
        not math.isfinite(env.command_servo_omega)
        or env.command_servo_omega <= 0.0
    ):
        raise ValueError(
            "environment.command_servo_omega must be finite and positive"
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
    if env.motion_reference_mode != "frame":
        raise ValueError("environment.motion_reference_mode must be frame")
    if env.root_velocity_mode not in {"com", "link"}:
        raise ValueError("environment.root_velocity_mode must be com or link")
    if env.policy_observation_mode != "tracking":
        raise ValueError("environment.policy_observation_mode must be tracking")
    if env.motion_end_behavior not in {"hold_last", "resample_command"}:
        raise ValueError("environment.motion_end_behavior must be hold_last or resample_command")
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
    if config.method == "fcamp":
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
    norm = str(params.advantage_normalization).lower()
    if norm not in {"per_prefix", "global", "none"}:
        raise ValueError(
            "parameters.advantage_normalization must be one of "
            f"'per_prefix', 'global', 'none'; got {params.advantage_normalization!r}"
        )

    style = params.style_prior
    credit = params.credit
    critics = params.critics
    streams = params.streams
    if not (0.0 < streams.phase0_fraction < 1.0):
        raise ValueError("FCAMP streams.phase0_fraction must be in (0, 1)")
    if not style.enabled:
        raise ValueError("FCAMP requires parameters.style_prior.enabled=true")
    if style.obs_steps < 2:
        raise ValueError("FCAMP requires style_prior.obs_steps >= 2")
    if not style.hidden_dims:
        raise ValueError("FCstyle discriminator hidden_dims cannot be empty")
    if style.reward_scale <= 0.0 or not (0.0 < style.reward_epsilon < 1.0):
        raise ValueError("FCAMP style reward scale/epsilon are invalid")
    if style.optimizer.lower() != "sgd":
        raise ValueError("FCAMP style_prior.optimizer must be sgd")
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
    if style.replay_dtype.lower() != "float32":
        raise ValueError("FCAMP requires style_prior.replay_dtype=float32")
    if style.replay_device.lower() != "cpu":
        raise ValueError("FCAMP complete-window replay must use replay_device=cpu")
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
    if credit.mode != "causal_frame":
        raise ValueError("FCAMP credit.mode must be causal_frame")
    if credit.advantage_normalization != "global":
        raise ValueError(
            "FCAMP actor advantage must normalize the weighted reward mixture "
            "once globally"
        )
    if not credit.integrate_amp_reward_dt:
        raise ValueError(
            "FCAMP requires credit.integrate_amp_reward_dt=true so the style "
            "reward has the same control-frequency semantics as task reward"
        )
    if credit.ratio_mode != "conditional_frame":
        raise ValueError(
            "FCAMP requires credit.ratio_mode=conditional_frame"
        )
    if credit.task_weight < 0.0 or credit.amp_weight < 0.0:
        raise ValueError("FCstyle reward weights must be non-negative")
    if credit.task_weight == 0.0 and credit.amp_weight == 0.0:
        raise ValueError("FCAMP requires at least one non-zero reward weight")
    if critics.sharing != "encoder":
        raise ValueError("Full FCAMP requires critics.sharing=encoder")
    if not critics.encoder_hidden_dims or not critics.head_hidden_dims:
        raise ValueError("FCAMP critic encoder/head dimensions cannot be empty")
