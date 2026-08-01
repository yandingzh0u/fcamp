from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from engine.config import FixedRewardConfig, load_config


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "fixed_reward_largebox.yaml"


def test_default_config_is_exact_holosoma_g1_wbt_ppo() -> None:
    cfg = load_config(CONFIG)
    assert cfg.environment.num_envs == 4096
    assert cfg.training.seed == 42
    assert cfg.training.max_updates == 500
    assert cfg.training.log_every == 1
    params = cfg.parameters
    assert params.actor_hidden_dims == (512, 256, 128)
    assert params.critic_hidden_dims == (512, 256, 128)
    assert params.activation == "ELU"
    assert params.action_clip_value == 100.0
    assert params.num_steps_per_env == 24
    assert params.num_learning_epochs == 5
    assert params.num_mini_batches == 4
    assert params.clip_param == 0.2
    assert params.gamma == 0.99
    assert params.lam == 0.95
    assert params.value_loss_coef == 1.0
    assert params.entropy_coef == 0.005
    assert params.actor_learning_rate == 0.001
    assert params.critic_learning_rate == 0.001
    assert params.actor_weight_decay == 0.0
    assert params.critic_weight_decay == 0.0
    assert params.max_grad_norm == 1.0
    assert params.schedule == "adaptive"
    assert params.desired_kl == 0.01
    assert params.init_noise_std == 1.0
    assert params.init_at_random_ep_len is True
    assert params.empirical_normalization is True
    assert params.use_symmetry is False
    assert cfg.environment.num_envs * params.num_steps_per_env == 98_304
    assert 500 * cfg.environment.num_envs * params.num_steps_per_env == 49_152_000


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("actor_hidden_dims", "[256,128]"),
        ("critic_hidden_dims", "[256,128]"),
        ("activation", "ReLU"),
        ("action_clip_value", "5.0"),
        ("num_steps_per_env", "23"),
        ("num_learning_epochs", "4"),
        ("num_mini_batches", "8"),
        ("clip_param", "0.1"),
        ("gamma", "0.98"),
        ("lam", "0.9"),
        ("value_loss_coef", "0.5"),
        ("entropy_coef", "0.0"),
        ("actor_learning_rate", "0.0001"),
        ("critic_learning_rate", "0.0003"),
        ("actor_weight_decay", "0.001"),
        ("critic_weight_decay", "0.001"),
        ("max_grad_norm", "0.5"),
        ("schedule", "fixed"),
        ("desired_kl", "0.02"),
        ("init_noise_std", "0.8"),
        ("init_at_random_ep_len", "false"),
        ("empirical_normalization", "false"),
        ("use_symmetry", "true"),
    ],
)
def test_holosoma_algorithm_parameters_cannot_silently_drift(
    key: str, value: str
) -> None:
    with pytest.raises(ValueError, match=key):
        load_config(CONFIG, [f"parameters.{key}={value}"])


def test_config_has_no_flow_cps_or_custom_ppo_knobs() -> None:
    names = {field.name for field in fields(FixedRewardConfig)}
    for removed in (
        "flow_steps",
        "exploration_scale_init",
        "exploration_cov_rank",
        "cps_noise_init",
        "cps_cov_rank",
        "micro_batch_size",
        "kl_early_stop_factor",
        "policy_lr",
        "value_lr",
        "rollout_env_steps",
    ):
        assert removed not in names


def test_smoke_can_override_only_runtime_dimensions() -> None:
    cfg = load_config(
        CONFIG,
        [
            "environment.num_envs=128",
            "training.max_updates=3",
            "training.save_every=1",
            "training.validation_every=1",
        ],
    )
    assert cfg.environment.num_envs == 128
    assert cfg.training.max_updates == 3
    assert cfg.parameters.actor_learning_rate == 0.001
    assert cfg.parameters.init_noise_std == 1.0
