from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeAlias

import yaml


MethodName: TypeAlias = str


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
    reset_phase_sampling: str
    rsi_keyframe_count: int
    adaptive_motion_sampling: bool
    adaptive_num_bins: int
    adaptive_alpha: float
    adaptive_predecessor_ratio: float
    adaptive_predecessor_lookback_bins: int
    action_rate_weight: float
    termination_mode: str
    terminate_on_motion_end: bool
    motion_reference_mode: str


@dataclass(frozen=True, slots=True)
class FlowCPSConfig:
    """Shared Flow-CPS actor settings used by FCAMP."""

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
    kl_early_stop_factor: float
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
class FCAMPConfig(FlowCPSConfig):
    """Full causal Flow-CPS + temporal discriminator training path."""

    style_prior: StylePriorConfig
    credit: FCAMPCreditConfig
    critics: FCAMPCriticConfig

    @property
    def amp(self) -> StylePriorConfig:
        """Compatibility alias for older FCAMP internals."""
        return self.style_prior


@dataclass(frozen=True, slots=True)
class AdaMimicConfig:
    """AdaMimic two-level adaptive-time PPO configuration."""

    stage: str
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    activation: str
    rollout_env_steps: int
    discount_gamma: float
    time_discount_gamma: float
    gae_lambda: float
    clip_range: float
    desired_kl: float
    policy_epochs: int
    num_mini_batches: int
    micro_batch_size: int
    value_loss_coef: float
    entropy_coef: float
    policy_lr: float
    weight_decay: float
    empirical_normalization: bool
    init_at_random_ep_len: bool
    max_grad_norm: float
    use_clipped_value_loss: bool
    infer_keyframe_time: bool
    actor_time_scale_range: tuple[float, float]
    fixed_dt: float
    time_min_std: float
    init_noise_std: float
    train_time: bool
    time_reward_scale: float
    use_timeout_bootstrap: bool
    use_smooth: bool
    smoothness_upper_bound: float
    smoothness_lower_bound: float
    value_smoothness_coef: float
    residual_delta: bool
    checkpoint_path: str
    freeze_base: bool
    residual_time_threshold: float


@dataclass(frozen=True, slots=True)
class AMPConfig:
    """MimicKit-style AMP PPO configuration."""

    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    disc_hidden_dims: tuple[int, ...]
    activation: str
    rollout_env_steps: int
    discount_gamma: float
    gae_lambda: float
    clip_range: float
    norm_adv_clip: float
    actor_epochs: int
    actor_batch_size: int
    critic_epochs: int
    critic_batch_size: int
    disc_epochs: int
    disc_batch_size: int
    actor_lr: float
    critic_lr: float
    disc_lr: float
    disc_weight_decay: float
    actor_init_output_scale: float
    action_std: float
    action_bound_weight: float
    action_entropy_weight: float
    action_reg_weight: float
    task_reward_weight: float
    disc_reward_weight: float
    disc_reward_scale: float
    disc_reward_epsilon: float
    disc_buffer_size: int
    disc_replay_samples: int
    disc_logit_reg: float
    disc_grad_penalty: float
    disc_obs_steps: int
    disc_normalizer_clip: float
    disc_eval_batch_size: int
    empirical_normalization: bool
    init_at_random_ep_len: bool
    normalizer_samples: int


@dataclass(frozen=True, slots=True)
class ADDConfig(AMPConfig):
    """MimicKit ADD uses AMP's PPO knobs with a diff discriminator."""


MethodConfig: TypeAlias = FCAMPConfig | AdaMimicConfig | AMPConfig | ADDConfig


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
    validation_done_frac_early_stop: float
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
    "adamimic": AdaMimicConfig,
    "amp": AMPConfig,
    "add": ADDConfig,
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
        "disc_hidden_dims",
        "hidden_dims",
        "encoder_hidden_dims",
        "head_hidden_dims",
    ):
        if name in converted:
            converted[name] = tuple(int(value) for value in converted[name])
    if "actor_time_scale_range" in converted:
        converted["actor_time_scale_range"] = tuple(float(value) for value in converted["actor_time_scale_range"])
    return cls(**converted)


def _construct_method_config(method: str, values: dict[str, Any], source_path: Path) -> MethodConfig:
    if method == "amp":
        nested = dict(values)
        nested.setdefault("normalizer_samples", 100_000_000)
        return _construct(AMPConfig, nested)
    if method == "add":
        nested = dict(values)
        nested.setdefault("normalizer_samples", 100_000_000)
        return _construct(ADDConfig, nested)
    if method == "adamimic":
        nested = dict(values)
        nested["checkpoint_path"] = _resolve_path(str(nested.get("checkpoint_path", "")), source_path)
        return _construct(AdaMimicConfig, nested)
    if method != "fcamp":
        raise ValueError(f"Unknown method {method!r}. Add method/{method}.py and register its config schema.")
    nested = dict(values)
    if "style_prior" not in nested and "amp" in nested:
        nested["style_prior"] = nested.pop("amp")
    try:
        nested["style_prior"] = _construct(StylePriorConfig, dict(nested["style_prior"]))
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
    environment_values.setdefault("termination_mode", "tracking")
    environment_values.setdefault("terminate_on_motion_end", False)
    environment_values.setdefault("motion_reference_mode", "frame")
    training_values = dict(normalized["training"])
    training_values.setdefault("official_reset_every", 0)
    training_values["resume"] = _resolve_path(str(training_values["resume"]), source_path)
    config = ExperimentConfig(
        method=method,
        environment=_construct(EnvironmentConfig, environment_values),
        parameters=_construct_method_config(method, dict(normalized["parameters"]), source_path),
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
    if env.reset_phase_sampling not in {"adaptive", "uniform", "rsi", "zero"}:
        raise ValueError("environment.reset_phase_sampling must be one of adaptive/uniform/rsi/zero")
    if env.rsi_keyframe_count < 1:
        raise ValueError("environment.rsi_keyframe_count must be positive")
    if env.adaptive_motion_sampling != (env.reset_phase_sampling == "adaptive"):
        raise ValueError("environment.adaptive_motion_sampling must match reset_phase_sampling=adaptive")
    if env.termination_mode not in {"tracking", "amp", "add"}:
        raise ValueError("environment.termination_mode must be one of tracking/amp/add")
    if env.motion_reference_mode not in {"frame", "mimickit_add"}:
        raise ValueError("environment.motion_reference_mode must be frame or mimickit_add")
    if train.max_updates < 1:
        raise ValueError("training.max_updates must be positive")
    if train.log_every < 1:
        raise ValueError("training.log_every must be positive")
    if train.official_reset_every < 0:
        raise ValueError("training.official_reset_every must be non-negative")
    resolve_task(env.task)
    if config.method == "fcamp":
        _validate_fcamp(config.parameters)
    elif config.method == "adamimic":
        if config.parameters.stage == "stage1" and env.reset_phase_sampling != "rsi":
            raise ValueError("AdaMimic stage1 follows official RSI reset sampling")
        if config.parameters.stage == "stage2" and env.reset_phase_sampling != "zero":
            raise ValueError("AdaMimic stage2 follows official rsi=false zero reset sampling")
        _validate_adamimic(config.parameters)
    elif config.method == "amp":
        _validate_amp(config.parameters)
    elif config.method == "add":
        if env.motion_reference_mode != "mimickit_add":
            raise ValueError("ADD requires environment.motion_reference_mode=mimickit_add")
        if abs(float(env.sim_dt) - 1.0 / 30.0) > 1.0e-12 or env.decimation != 4:
            raise ValueError("ADD follows MimicKit's 30 Hz control / 120 Hz simulation")
        if env.reset_phase_sampling != "uniform":
            raise ValueError("ADD follows MimicKit's continuous uniform motion-time reset")
        _validate_add(config.parameters)
    else:
        raise ValueError(f"Unsupported method {config.method!r}")


def _validate_fcamp(params: FCAMPConfig) -> None:
    if params.horizon < 1:
        raise ValueError("Flow-CPS requires parameters.horizon >= 1")
    if params.flow_steps < 1:
        raise ValueError("Flow-CPS requires parameters.flow_steps >= 1")
    if params.rollout_env_steps <= 0:
        raise ValueError("Flow-CPS requires parameters.rollout_env_steps > 0")
    if params.rollout_env_steps % params.horizon:
        raise ValueError("parameters.rollout_env_steps must be divisible by parameters.horizon")
    if not (0.0 < params.cps_noise_level < 1.0):
        raise ValueError("Flow-CPS requires parameters.cps_noise_level in (0, 1)")
    if params.cps_cov_rank < 0:
        raise ValueError("Flow-CPS requires parameters.cps_cov_rank >= 0")
    norm = str(params.advantage_normalization).lower()
    if norm not in {"per_prefix", "global", "none"}:
        raise ValueError(
            "parameters.advantage_normalization must be one of "
            f"'per_prefix', 'global', 'none'; got {params.advantage_normalization!r}"
        )

    style = params.style_prior
    credit = params.credit
    critics = params.critics
    if not style.enabled:
        raise ValueError("FCAMP requires parameters.style_prior.enabled=true")
    if style.obs_steps < 2:
        raise ValueError("FCAMP requires style_prior.obs_steps >= 2")
    if not style.hidden_dims:
        raise ValueError("FCstyle discriminator hidden_dims cannot be empty")
    if style.reward_scale <= 0.0 or not (0.0 < style.reward_epsilon < 1.0):
        raise ValueError("FCAMP style reward scale/epsilon are invalid")
    if style.optimizer.lower() not in {"sgd", "adam", "adamw"}:
        raise ValueError("FCAMP style_prior.optimizer must be sgd, adam, or adamw")
    if style.learning_rate <= 0.0 or style.batch_size < 1 or style.epochs < 1:
        raise ValueError("FCstyle discriminator optimizer/batch/epoch settings are invalid")
    if style.max_updates_per_iteration < 1:
        raise ValueError("FCAMP max_updates_per_iteration must be positive")
    if style.current_buffer_size < style.batch_size:
        raise ValueError("FCAMP style_prior.current_buffer_size must be >= batch_size")
    if style.replay_size < style.batch_size or style.replay_samples < 0:
        raise ValueError("FCframe replay settings are invalid")
    if style.replay_dtype.lower() != "float32":
        raise ValueError("FCAMP requires style_prior.replay_dtype=float32")
    if style.replay_device.lower() not in {"cpu", "cuda"}:
        raise ValueError("FCAMP style_prior.replay_device must be cpu or cuda")
    if credit.mode not in {"causal_frame", "chunk_shared"}:
        raise ValueError("FCAMP credit.mode must be causal_frame or chunk_shared")
    if credit.advantage_normalization not in {
        "per_offset",
        "global",
        "per_channel_per_offset",
        "per_channel_global",
        "none",
    }:
        raise ValueError("Unsupported FCAMP advantage normalization")
    if credit.ratio_mode != "joint_path":
        raise ValueError("FCAMP requires credit.ratio_mode=joint_path")
    if credit.task_weight < 0.0 or credit.amp_weight < 0.0:
        raise ValueError("FCstyle reward weights must be non-negative")
    if credit.task_weight == 0.0 and credit.amp_weight == 0.0:
        raise ValueError("FCAMP requires at least one non-zero reward weight")
    if critics.sharing != "encoder":
        raise ValueError("Full FCAMP requires critics.sharing=encoder")
    if not critics.encoder_hidden_dims or not critics.head_hidden_dims:
        raise ValueError("FCAMP critic encoder/head dimensions cannot be empty")


def _validate_adamimic(params: AdaMimicConfig) -> None:
    if params.stage not in {"stage1", "stage2"}:
        raise ValueError("AdaMimic parameters.stage must be 'stage1' or 'stage2'")
    if not params.actor_hidden_dims or not params.critic_hidden_dims:
        raise ValueError("AdaMimic actor/critic hidden dims cannot be empty")
    if params.rollout_env_steps <= 1:
        raise ValueError("AdaMimic requires parameters.rollout_env_steps > 1")
    if not (0.0 < params.discount_gamma <= 1.0):
        raise ValueError("AdaMimic discount_gamma must be in (0, 1]")
    if not (0.0 < params.time_discount_gamma <= 1.0):
        raise ValueError("AdaMimic time_discount_gamma must be in (0, 1]")
    if not (0.0 <= params.gae_lambda <= 1.0):
        raise ValueError("AdaMimic gae_lambda must be in [0, 1]")
    if not (0.0 < params.clip_range < 1.0):
        raise ValueError("AdaMimic clip_range must be in (0, 1)")
    if params.policy_epochs < 1 or params.num_mini_batches < 1:
        raise ValueError("AdaMimic policy_epochs/num_mini_batches must be positive")
    if params.policy_lr <= 0.0 or params.max_grad_norm <= 0.0:
        raise ValueError("AdaMimic policy_lr/max_grad_norm must be positive")
    if params.value_loss_coef < 0.0 or params.entropy_coef < 0.0:
        raise ValueError("AdaMimic loss coefficients must be non-negative")
    if len(params.actor_time_scale_range) != 2:
        raise ValueError("AdaMimic actor_time_scale_range must have two values")
    low, high = params.actor_time_scale_range
    if high < low:
        raise ValueError("AdaMimic actor_time_scale_range must be [low, high]")
    if params.fixed_dt <= 0.0 or params.time_min_std <= 0.0 or params.init_noise_std <= 0.0:
        raise ValueError("AdaMimic fixed_dt/time_min_std/init_noise_std must be positive")
    if params.time_reward_scale < 0.0:
        raise ValueError("AdaMimic time_reward_scale must be non-negative")
    if params.smoothness_upper_bound <= params.smoothness_lower_bound or params.smoothness_lower_bound <= 0.0:
        raise ValueError("AdaMimic smoothness bounds must satisfy upper > lower > 0")
    if params.value_smoothness_coef < 0.0:
        raise ValueError("AdaMimic smoothness coefficients must be non-negative")
    is_fixed_time = low == 0.0 and high == 0.0
    if params.stage == "stage1":
        if params.train_time:
            raise ValueError("AdaMimic stage1 follows official train_high=false; set parameters.train_time=false")
        if not is_fixed_time:
            raise ValueError("AdaMimic stage1 follows official fixed time; set actor_time_scale_range=[0.0, 0.0]")
        if params.residual_delta or params.freeze_base:
            raise ValueError("AdaMimic stage1 must not enable residual_delta/freeze_base")
    if params.stage == "stage2" and not params.residual_delta:
        raise ValueError("AdaMimic stage2 requires residual_delta=true")
    if params.residual_delta and not params.checkpoint_path:
        raise ValueError("AdaMimic residual_delta requires checkpoint_path")
    if params.stage == "stage2":
        if not params.train_time:
            raise ValueError("AdaMimic stage2 follows official train_high=true; set parameters.train_time=true")
        if is_fixed_time:
            raise ValueError("AdaMimic stage2 requires a non-zero actor_time_scale_range")
        if not params.freeze_base:
            raise ValueError("AdaMimic stage2 follows official freeze=true; set parameters.freeze_base=true")
        if params.use_smooth:
            raise ValueError("AdaMimic stage2 follows official use_smooth=false")


def _validate_amp(params: AMPConfig, *, method_label: str = "AMP", min_disc_obs_steps: int = 2) -> None:
    if not params.actor_hidden_dims or not params.critic_hidden_dims or not params.disc_hidden_dims:
        raise ValueError(f"{method_label} actor/critic/discriminator hidden dims cannot be empty")
    if params.rollout_env_steps < 1:
        raise ValueError(f"{method_label} rollout_env_steps must be positive")
    if not (0.0 < params.discount_gamma <= 1.0):
        raise ValueError(f"{method_label} discount_gamma must be in (0, 1]")
    if not (0.0 <= params.gae_lambda <= 1.0):
        raise ValueError(f"{method_label} gae_lambda must be in [0, 1]")
    if not (0.0 < params.clip_range < 1.0):
        raise ValueError(f"{method_label} clip_range must be in (0, 1)")
    for name in (
        "actor_epochs",
        "actor_batch_size",
        "critic_epochs",
        "critic_batch_size",
        "disc_epochs",
        "disc_batch_size",
    ):
        if int(getattr(params, name)) < 1:
            raise ValueError(f"{method_label} {name} must be positive")
    for name in ("actor_lr", "critic_lr", "disc_lr", "action_std", "actor_init_output_scale"):
        if float(getattr(params, name)) <= 0.0:
            raise ValueError(f"{method_label} {name} must be positive")
    if params.disc_weight_decay < 0.0:
        raise ValueError(f"{method_label} disc_weight_decay must be non-negative")
    if params.action_bound_weight < 0.0 or params.action_entropy_weight < 0.0 or params.action_reg_weight < 0.0:
        raise ValueError(f"{method_label} action regularization weights must be non-negative")
    if params.task_reward_weight < 0.0 or params.disc_reward_weight < 0.0:
        raise ValueError(f"{method_label} reward weights must be non-negative")
    if params.task_reward_weight == 0.0 and params.disc_reward_weight == 0.0:
        raise ValueError(f"{method_label} requires at least one non-zero reward weight")
    if params.disc_reward_scale <= 0.0 or not (0.0 < params.disc_reward_epsilon < 1.0):
        raise ValueError(f"{method_label} discriminator reward scale/epsilon are invalid")
    if params.disc_buffer_size < 1 or params.disc_replay_samples < 0:
        raise ValueError(f"{method_label} discriminator replay settings are invalid")
    if params.disc_logit_reg < 0.0 or params.disc_grad_penalty < 0.0:
        raise ValueError(f"{method_label} discriminator regularization weights must be non-negative")
    if params.disc_obs_steps < int(min_disc_obs_steps):
        raise ValueError(f"{method_label} disc_obs_steps must be at least {int(min_disc_obs_steps)}")
    if params.disc_normalizer_clip <= 0.0 or params.disc_eval_batch_size < 1:
        raise ValueError(f"{method_label} discriminator normalizer/eval batch settings are invalid")
    if params.normalizer_samples < 1:
        raise ValueError(f"{method_label} normalizer_samples must be positive")


def _validate_add(params: ADDConfig) -> None:
    _validate_amp(params, method_label="ADD", min_disc_obs_steps=1)
    if params.disc_obs_steps != 1:
        raise ValueError("ADD follows MimicKit add_g1_env.yaml: parameters.disc_obs_steps must be 1")
