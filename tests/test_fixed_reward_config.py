from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from engine.config import (
    FixedRewardConfig,
    FlowCPSConfig,
    TrainingConfig,
    config_from_checkpoint_dict,
    config_from_dict,
    load_config,
)
from envs.tasks import TASKS


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "fixed_reward_largebox.yaml"


def test_fixed_reward_config_is_production_pose_only_recipe() -> None:
    cfg = load_config(CONFIG)

    assert cfg.method == "fixed_reward"
    assert isinstance(cfg.parameters, FixedRewardConfig)
    assert cfg.environment.root_velocity_mode == "link"
    assert cfg.environment.num_envs == 8192
    assert cfg.parameters.horizon == 4
    assert cfg.parameters.rollout_env_steps == 24
    assert cfg.parameters.rollout_env_steps % cfg.parameters.horizon == 0
    assert cfg.parameters.flow_steps == 4
    assert cfg.parameters.action_squash_scale == 5.0
    assert cfg.parameters.cps_noise_init == 0.36
    assert cfg.parameters.cps_cov_rank == 8
    assert cfg.parameters.desired_kl == 0.01
    assert cfg.parameters.policy_lr == 0.0003
    assert cfg.parameters.value_lr == 0.0003
    assert cfg.parameters.streams.phase0_fraction == pytest.approx(0.10)
    assert cfg.training.seed == 0
    assert cfg.training.max_updates == 500
    assert cfg.training.log_every == 10
    assert cfg.training.validation_every == 50
    assert cfg.training.save_every == 50
    transitions_per_update = (
        cfg.environment.num_envs * cfg.parameters.rollout_env_steps
    )
    assert transitions_per_update == 196_608
    assert (
        transitions_per_update * cfg.training.max_updates
        == 98_304_000
    )


def test_fixed_reward_has_one_critic_and_no_removed_sections() -> None:
    tree = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert "critic" in tree["parameters"]
    assert "critics" not in tree["parameters"]
    assert "style_prior" not in tree["parameters"]
    assert "credit" not in tree["parameters"]
    assert set(tree["parameters"]["critic"]) == {
        "encoder_hidden_dims",
        "head_hidden_dims",
    }


def test_fixed_reward_requires_two_streams_supported_reset_and_link_velocity() -> None:
    with pytest.raises(ValueError, match="at least two environments"):
        load_config(CONFIG, ["environment.num_envs=1"])
    with pytest.raises(ValueError, match="reset_phase_sampling must be"):
        load_config(
            CONFIG,
            ["environment.reset_phase_sampling=continuous_uniform"],
        )
    with pytest.raises(ValueError, match="root_velocity_mode=link"):
        load_config(CONFIG, ["environment.root_velocity_mode=com"])
    with pytest.raises(ValueError, match="action_rate_weight=0.1"):
        load_config(CONFIG, ["environment.action_rate_weight=0.2"])


def test_smoke_overrides_preserve_the_same_recipe() -> None:
    cfg = load_config(
        CONFIG,
        [
            "environment.num_envs=128",
            "training.max_updates=2",
            "training.log_every=1",
            "training.validation_every=1",
            "training.save_every=1",
        ],
    )
    assert cfg.environment.num_envs == 128
    assert cfg.training.max_updates == 2
    assert cfg.training.validation_every == 1
    assert cfg.parameters.critic.encoder_hidden_dims == (512, 256, 128)


def test_validation_has_no_fractional_early_stop() -> None:
    assert "validation_done_frac_early_stop" not in {
        field.name for field in fields(TrainingConfig)
    }


def test_flow_cps_config_has_no_legacy_algorithm_fields() -> None:
    fields_by_name = {field.name for field in fields(FlowCPSConfig)}
    assert "cps_noise_init" in fields_by_name
    assert "cps_noise_level" not in fields_by_name
    assert "init_noise_std" not in fields_by_name
    assert "num_generations" not in fields_by_name
    assert "tail_bootstrap_steps" not in fields_by_name
    assert "actor_density" not in fields_by_name
    assert "failure_penalty" not in fields_by_name


@pytest.mark.parametrize(
    "value",
    [0.0, 1.0e-4, 0.9999, 1.0],
)
def test_cps_noise_init_respects_bounded_eta_parameterization(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="cps_noise_init"):
        load_config(
            CONFIG,
            [f"parameters.cps_noise_init={value}"],
        )


def test_task_binds_motion_and_terrain() -> None:
    assert TASKS["largebox_plane"].terrain == "plane"
    assert TASKS["crawl_slope"].terrain == "slope"
    assert TASKS["largebox_plane"].motion_file.name == "sub3_largebox_003_mj.npz"
    assert TASKS["crawl_slope"].motion_file.name == "motion_crawl_slope.npz"


def test_override_cannot_create_a_second_config_entry() -> None:
    with pytest.raises(KeyError):
        load_config(CONFIG, ["environment.terrain=plane"])


def test_config_rejects_removed_sections_instead_of_silently_ignoring_them() -> None:
    tree = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    tree["parameters"]["credit"] = {"task_weight": 1.0}

    with pytest.raises(KeyError, match="unknown keys"):
        config_from_dict(tree)


def test_checkpoint_loader_rejects_legacy_method_before_coercion() -> None:
    tree = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    tree["method"] = "fcamp"
    tree["parameters"]["style_prior"] = {}

    with pytest.raises(ValueError, match="Legacy checkpoints"):
        config_from_checkpoint_dict(tree)


def test_checkpoint_config_round_trip_is_strict() -> None:
    tree = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg = config_from_checkpoint_dict(tree)

    assert cfg.method == "fixed_reward"
    assert cfg.parameters.critic.head_hidden_dims == (128, 64)


@pytest.mark.parametrize(
    "override",
    [
        "parameters.policy_lr=0.0",
        "parameters.value_lr=0.0",
        "parameters.critic.encoder_hidden_dims=[512,0]",
        "parameters.critic.head_hidden_dims=[0]",
        "parameters.streams.phase0_fraction=0.0",
        "parameters.streams.phase0_fraction=0.2",
        "parameters.streams.phase0_fraction=1.0",
    ],
)
def test_fixed_reward_rejects_degenerate_optimizers_and_critic(
    override: str,
) -> None:
    with pytest.raises(ValueError):
        load_config(CONFIG, [override])
