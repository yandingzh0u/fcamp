from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path

import pytest
import yaml

from engine.config import (
    AMPConfig,
    AMPPolicyConfig,
    TrainingConfig,
    config_from_checkpoint_dict,
    config_from_dict,
    load_config,
)
from envs.tasks import TASKS


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "amp_largebox.yaml"


def _raw_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_amp_config_matches_standard_mimickit_defaults() -> None:
    cfg = load_config(CONFIG_PATH)

    assert cfg.method == "amp"
    assert isinstance(cfg.parameters, AMPConfig)
    assert cfg.environment.num_envs == 4096
    assert cfg.environment.root_velocity_mode == "link"

    params = cfg.parameters
    assert params.horizon == 1
    assert params.actor_hidden_dims == (1024, 512)
    assert params.critic.hidden_dims == (1024, 512)
    assert params.rollout_env_steps == 32
    assert params.discount_gamma == pytest.approx(0.99)
    assert params.gae_lambda == pytest.approx(0.95)
    assert params.clip_range == pytest.approx(0.2)
    assert params.advantage_clip == pytest.approx(4.0)
    assert params.policy_epochs == 5
    assert params.critic_epochs == 2
    assert params.actor_batch_size == 4
    assert params.critic_batch_size == 2
    assert params.micro_batch_size == 4096
    assert params.policy_lr == pytest.approx(1.0e-4)
    assert params.value_lr == pytest.approx(1.0e-4)
    assert params.action_bound_weight == pytest.approx(10.0)

    style = params.style_prior
    assert style.obs_steps == 10
    assert style.hidden_dims == (1024, 512)
    assert style.learning_rate == pytest.approx(2.5e-4)
    assert style.weight_decay == pytest.approx(1.0e-4)
    assert style.epochs == 2
    assert style.batch_size == 2
    assert style.micro_batch_size == 1024
    assert style.replay_size == 200_000
    assert style.replay_samples == 1_000
    assert style.grad_penalty == pytest.approx(10.0)
    assert style.logit_reg == pytest.approx(0.01)


@pytest.mark.parametrize("horizon", [1, 4])
def test_amp_horizon_is_configurable_without_changing_action_contract(
    horizon: int,
) -> None:
    cfg = load_config(CONFIG_PATH, [f"parameters.horizon={horizon}"])

    assert cfg.parameters.horizon == horizon


def test_amp_preserves_requested_logging_checkpoint_validation_cadence() -> None:
    cfg = load_config(CONFIG_PATH)

    assert cfg.training.max_updates == 500
    assert cfg.training.log_every == 10
    assert cfg.training.save_every == 50
    assert cfg.training.validation_every == 50


def test_amp_config_checkpoint_round_trip_is_exact() -> None:
    cfg = load_config(CONFIG_PATH)
    restored = config_from_checkpoint_dict(asdict(cfg), CONFIG_PATH)

    assert restored == cfg


def test_unknown_configuration_fields_are_rejected() -> None:
    tree = _raw_config()
    tree["parameters"]["unknown_parameter"] = 4
    with pytest.raises(KeyError, match="unknown_parameter"):
        config_from_dict(tree, CONFIG_PATH)

    tree = _raw_config()
    tree["environment"]["unknown_environment_field"] = True
    with pytest.raises(KeyError, match="unknown_environment_field"):
        config_from_dict(tree, CONFIG_PATH)

def test_amp_allows_one_environment_because_there_are_no_fixed_streams() -> None:
    cfg = load_config(CONFIG_PATH, ["environment.num_envs=1"])
    assert cfg.environment.num_envs == 1


def test_amp_rejects_com_root_velocity_contract() -> None:
    with pytest.raises(ValueError, match="root_velocity_mode=link"):
        load_config(CONFIG_PATH, ["environment.root_velocity_mode=com"])


def test_amp_discriminator_batch_and_global_replay_are_validated() -> None:
    with pytest.raises(ValueError, match="optimizer/batch/epoch"):
        load_config(
            CONFIG_PATH,
            ["parameters.style_prior.batch_size=0"],
        )
    with pytest.raises(ValueError, match="global replay"):
        load_config(
            CONFIG_PATH,
            ["parameters.style_prior.replay_samples=200001"],
        )


def test_validation_has_no_fractional_early_stop() -> None:
    assert "validation_done_frac_early_stop" not in {
        field.name for field in fields(TrainingConfig)
    }
    load_config(CONFIG_PATH)


def test_amp_policy_config_has_only_the_direct_gaussian_contract_fields() -> None:
    fields_by_name = {field.name for field in fields(AMPPolicyConfig)}
    assert fields_by_name == {
        "horizon",
        "actor_hidden_dims",
        "rollout_env_steps",
        "discount_gamma",
        "gae_lambda",
        "clip_range",
        "advantage_clip",
        "policy_epochs",
        "critic_epochs",
        "actor_batch_size",
        "critic_batch_size",
        "micro_batch_size",
        "policy_lr",
        "value_lr",
        "action_bound_weight",
    }


def test_task_binds_motion_and_terrain() -> None:
    assert TASKS["largebox_plane"].terrain == "plane"
    assert TASKS["crawl_slope"].terrain == "slope"
    assert TASKS["largebox_plane"].motion_file.name == (
        "sub3_largebox_003_mj.npz"
    )
    assert TASKS["crawl_slope"].motion_file.name == (
        "motion_crawl_slope.npz"
    )


def test_override_cannot_create_a_second_config_entry() -> None:
    with pytest.raises(KeyError):
        load_config(CONFIG_PATH, ["environment.terrain=plane"])


@pytest.mark.parametrize(
    "override",
    [
        "parameters.horizon=0",
        "parameters.policy_lr=0.0",
        "parameters.value_lr=0.0",
        "parameters.actor_hidden_dims=[1024,0]",
        "parameters.critic.hidden_dims=[512,0]",
        "parameters.style_prior.hidden_dims=[1024,0]",
    ],
)
def test_amp_rejects_degenerate_network_and_optimizer_values(
    override: str,
) -> None:
    with pytest.raises(ValueError):
        load_config(CONFIG_PATH, [override])
