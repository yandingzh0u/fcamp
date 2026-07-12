from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, TypeAlias

import yaml


AlgorithmName: TypeAlias = Literal[
    "ppo", "fpo", "fpo++", "flowrl", "reinflow", "fql", "sfpo", "sfpo-gaussian", "chunk-ppo"
]


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
class FPOPlusPlusConfig:
    horizon: int
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


@dataclass(frozen=True, slots=True)
class OriginalFPOConfig(FPOPlusPlusConfig):
    average_losses_before_exp: bool


@dataclass(frozen=True, slots=True)
class FlowRLConfig:
    horizon: int
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    activation: str
    flow_steps: int
    action_scale: float
    rollout_env_steps: int
    discount_gamma: float
    target_tau: float
    expectile: float
    w2_lambda: float
    cfm_weight_min: float
    cfm_weight_max: float
    replay_capacity: int
    replay_batch_size: int
    gradient_steps_per_update: int
    policy_delay: int
    warmup_env_steps: int
    recent_fraction: float
    recent_window: int
    exploration_noise: float
    policy_lr: float
    critic_lr: float
    value_lr: float
    weight_decay: float
    critic_weight_decay: float
    empirical_normalization: bool
    init_at_random_ep_len: bool
    max_grad_norm: float


@dataclass(frozen=True, slots=True)
class ReinFlowConfig:
    horizon: int
    actor_hidden_dims: tuple[int, ...]
    noise_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    activation: str
    flow_steps: int
    timestep_embed_dim: int
    action_scale: float
    min_denoising_std: float
    max_denoising_std: float
    randn_clip_value: float
    logprob_min: float
    logprob_max: float
    account_for_initial_stochasticity: bool
    normalize_denoising_horizon: bool
    normalize_action_dimension: bool
    rollout_env_steps: int
    discount_gamma: float
    gae_lambda: float
    clip_range: float
    target_kl: float
    policy_epochs: int
    num_mini_batches: int
    entropy_coef: float
    value_loss_coef: float
    policy_lr: float
    value_lr: float
    weight_decay: float
    critic_weight_decay: float
    empirical_normalization: bool
    init_at_random_ep_len: bool
    max_grad_norm: float
    pretrained_actor_path: str


@dataclass(frozen=True, slots=True)
class FQLConfig:
    horizon: int
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    activation: str
    actor_layer_norm: bool
    critic_layer_norm: bool
    flow_steps: int
    action_scale: float
    environment_action_scale: float
    rollout_env_steps: int
    discount_gamma: float
    target_tau: float
    q_aggregation: str
    alpha: float
    normalize_q_loss: bool
    offline_dataset_path: str
    offline_pretrain_gradient_steps: int
    offline_pretrain_log_every: int
    replay_capacity: int
    replay_batch_size: int
    gradient_steps_per_update: int
    warmup_env_steps: int
    recent_fraction: float
    recent_window: int
    flow_lr: float
    policy_lr: float
    critic_lr: float
    weight_decay: float
    critic_weight_decay: float
    empirical_normalization: bool
    init_at_random_ep_len: bool
    max_grad_norm: float


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
class SFPOGaussianConfig:
    """SFPO ablation with final action-space diagonal Gaussian exploration."""

    horizon: int
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    activation: str
    action_squash_scale: float
    flow_steps: int
    init_noise_std: float
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
class ChunkPPOConfig:
    """PPO-style policy over fixed-length action chunks."""

    horizon: int
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


AlgorithmConfig: TypeAlias = (
    PPOConfig
    | OriginalFPOConfig
    | FPOPlusPlusConfig
    | FlowRLConfig
    | ReinFlowConfig
    | FQLConfig
    | SFPOConfig
    | SFPOGaussianConfig
    | ChunkPPOConfig
)


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
    "fpo": OriginalFPOConfig,
    "fpo++": FPOPlusPlusConfig,
    "flowrl": FlowRLConfig,
    "reinflow": ReinFlowConfig,
    "fql": FQLConfig,
    "sfpo": SFPOConfig,
    "sfpo-gaussian": SFPOGaussianConfig,
    "chunk-ppo": ChunkPPOConfig,
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
    for name in ("actor_hidden_dims", "critic_hidden_dims", "noise_hidden_dims"):
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
    if algorithm in {"fpo", "fpo++"}:
        parameter_values.setdefault("use_clipped_value_loss", False)
    if algorithm == "reinflow":
        parameter_values["pretrained_actor_path"] = _resolve_path(
            str(parameter_values.get("pretrained_actor_path", "")), source_path
        )
    if algorithm == "fql":
        parameter_values["offline_dataset_path"] = _resolve_path(
            str(parameter_values.get("offline_dataset_path", "")), source_path
        )
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
    if isinstance(config.parameters, FPOPlusPlusConfig):
        if config.parameters.horizon != 1:
            raise ValueError("FPO++ currently implements the official action horizon h=1")
        if config.parameters.flow_steps < 1:
            raise ValueError("FPO++ requires parameters.flow_steps >= 1")
        if config.parameters.num_steps_per_env <= 0:
            raise ValueError("FPO++ requires parameters.num_steps_per_env > 0")
        if config.parameters.fpo_num_mc < 1:
            raise ValueError("FPO++ requires parameters.fpo_num_mc >= 1")
    if isinstance(config.parameters, OriginalFPOConfig):
        if not config.parameters.average_losses_before_exp:
            raise ValueError("Original FPO requires average_losses_before_exp=true")
        if config.parameters.schedule != "fixed":
            raise ValueError("Original FPO comparison uses its fixed learning-rate schedule")
    if isinstance(config.parameters, FlowRLConfig):
        parameters = config.parameters
        if parameters.horizon != 1:
            raise ValueError("FlowRL currently implements the official action horizon h=1")
        if parameters.flow_steps < 1 or parameters.rollout_env_steps < 1:
            raise ValueError("FlowRL requires positive flow_steps and rollout_env_steps")
        if not (0.0 < parameters.expectile < 1.0):
            raise ValueError("FlowRL expectile must be in (0, 1)")
        if not (0.0 < parameters.target_tau <= 1.0):
            raise ValueError("FlowRL target_tau must be in (0, 1]")
        if parameters.replay_capacity < 1 or parameters.replay_batch_size < 1:
            raise ValueError("FlowRL replay capacity and batch size must be positive")
        if parameters.gradient_steps_per_update < 1 or parameters.policy_delay < 1:
            raise ValueError("FlowRL gradient steps and policy delay must be positive")
        if not (0.0 <= parameters.recent_fraction <= 1.0):
            raise ValueError("FlowRL recent_fraction must be in [0, 1]")
    if isinstance(config.parameters, ReinFlowConfig):
        parameters = config.parameters
        if parameters.horizon != 4:
            raise ValueError("ReinFlow comparison requires the official horizon h=4")
        if parameters.rollout_env_steps < 1 or parameters.rollout_env_steps % parameters.horizon:
            raise ValueError("ReinFlow rollout_env_steps must be positive and divisible by horizon")
        if parameters.flow_steps < 1:
            raise ValueError("ReinFlow flow_steps must be positive")
        if not (0.0 < parameters.min_denoising_std <= parameters.max_denoising_std):
            raise ValueError("ReinFlow denoising std range is invalid")
        if parameters.logprob_min >= parameters.logprob_max:
            raise ValueError("ReinFlow logprob_min must be below logprob_max")
        if parameters.policy_epochs < 1 or parameters.num_mini_batches < 1:
            raise ValueError("ReinFlow policy_epochs and num_mini_batches must be positive")
        if not (
            parameters.account_for_initial_stochasticity
            and parameters.normalize_denoising_horizon
            and parameters.normalize_action_dimension
        ):
            raise ValueError(
                "ReinFlow comparison fixes official initial-noise accounting and likelihood normalization"
            )
    if isinstance(config.parameters, FQLConfig):
        parameters = config.parameters
        if parameters.horizon != 1:
            raise ValueError("FQL comparison uses the official one-step action horizon h=1")
        if parameters.rollout_env_steps < 1 or parameters.flow_steps < 1:
            raise ValueError("FQL requires positive rollout_env_steps and flow_steps")
        if parameters.action_scale != 1.0:
            raise ValueError("Official FQL clips actions to [-1, 1], so action_scale must be 1.0")
        if parameters.environment_action_scale <= 0.0:
            raise ValueError("FQL environment_action_scale must be positive")
        if not (0.0 < parameters.target_tau <= 1.0):
            raise ValueError("FQL target_tau must be in (0, 1]")
        if parameters.q_aggregation not in {"mean", "min"}:
            raise ValueError("FQL q_aggregation must be 'mean' or 'min'")
        if parameters.alpha < 0.0:
            raise ValueError("FQL alpha must be non-negative")
        if parameters.offline_pretrain_gradient_steps < 0:
            raise ValueError("FQL offline_pretrain_gradient_steps must be non-negative")
        if parameters.offline_pretrain_log_every < 1:
            raise ValueError("FQL offline_pretrain_log_every must be positive")
        if parameters.offline_pretrain_gradient_steps and not parameters.offline_dataset_path:
            raise ValueError(
                "FQL offline pretraining requires parameters.offline_dataset_path"
            )
        if parameters.replay_capacity < 1 or parameters.replay_batch_size < 1:
            raise ValueError("FQL replay capacity and batch size must be positive")
        if parameters.gradient_steps_per_update < 1 or parameters.warmup_env_steps < 0:
            raise ValueError("FQL gradient steps must be positive and warmup must be non-negative")
        if not (0.0 <= parameters.recent_fraction <= 1.0):
            raise ValueError("FQL recent_fraction must be in [0, 1]")
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
    if isinstance(config.parameters, SFPOGaussianConfig):
        if config.parameters.horizon < 1:
            raise ValueError("SFPO-Gaussian requires parameters.horizon >= 1")
        if config.parameters.flow_steps < 1:
            raise ValueError("SFPO-Gaussian requires parameters.flow_steps >= 1")
        if config.parameters.init_noise_std <= 0.0:
            raise ValueError("SFPO-Gaussian requires parameters.init_noise_std > 0")
        if config.parameters.rollout_env_steps <= 0:
            raise ValueError("SFPO-Gaussian requires parameters.rollout_env_steps > 0")
        if config.parameters.rollout_env_steps % config.parameters.horizon:
            raise ValueError("parameters.rollout_env_steps must be divisible by parameters.horizon")
        norm = str(config.parameters.advantage_normalization).lower()
        if norm not in {"per_prefix", "global", "none"}:
            raise ValueError(
                "SFPO-Gaussian parameters.advantage_normalization must be one of "
                f"'per_prefix', 'global', 'none'; got {config.parameters.advantage_normalization!r}"
            )
    if isinstance(config.parameters, ChunkPPOConfig):
        if config.parameters.horizon < 1:
            raise ValueError("ChunkPPO requires parameters.horizon >= 1")
        if config.parameters.init_noise_std <= 0.0:
            raise ValueError("ChunkPPO requires parameters.init_noise_std > 0")
        if config.parameters.num_steps_per_env <= 0:
            raise ValueError("ChunkPPO requires parameters.num_steps_per_env > 0")
        if config.parameters.num_steps_per_env % config.parameters.horizon:
            raise ValueError("parameters.num_steps_per_env must be divisible by parameters.horizon")
