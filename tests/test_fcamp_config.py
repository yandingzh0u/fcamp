from dataclasses import fields
from pathlib import Path

import pytest
import yaml

from engine.config import (
    FCAMPConfig,
    FlowCPSConfig,
    TrainingConfig,
    config_from_checkpoint_dict,
    config_from_dict,
    load_config,
)
from envs.tasks import TASKS


ROOT = Path(__file__).resolve().parents[1]


def test_fcamp_config_is_h4_w16_flow_cps() -> None:
    cfg = load_config(ROOT / "configs" / "fcamp_largebox.yaml")
    assert cfg.method == "fcamp"
    assert isinstance(cfg.parameters, FCAMPConfig)
    assert cfg.parameters.horizon == 4
    assert cfg.parameters.rollout_env_steps == 24
    assert cfg.parameters.rollout_env_steps % cfg.parameters.horizon == 0
    assert cfg.parameters.flow_steps == 4
    assert cfg.parameters.action_squash_scale == 5.0
    assert cfg.parameters.cps_noise_level == 0.8
    assert cfg.parameters.cps_cov_rank == 8
    assert cfg.parameters.desired_kl == 0.01
    assert cfg.parameters.policy_lr == 0.0003
    assert cfg.parameters.value_lr == 0.0003
    assert cfg.parameters.style_prior.obs_steps == 16
    assert cfg.parameters.streams.phase0_fraction == 0.10
    assert cfg.training.max_updates == 500


def test_fcamp_discriminator_batch_must_realize_both_streams() -> None:
    with pytest.raises(
        ValueError,
        match="discriminator optimizer/batch/epoch settings are invalid",
    ):
        load_config(
            ROOT / "configs" / "fcamp_largebox.yaml",
            ["parameters.style_prior.batch_size=1"],
        )


def test_fcamp_requires_two_streams_and_supported_reset_phases() -> None:
    with pytest.raises(ValueError, match="at least two environments"):
        load_config(
            ROOT / "configs" / "fcamp_largebox.yaml",
            ["environment.num_envs=1"],
        )
    with pytest.raises(ValueError, match="reset_phase_sampling must be"):
        load_config(
            ROOT / "configs" / "fcamp_largebox.yaml",
            ["environment.reset_phase_sampling=continuous_uniform"],
        )


def test_validation_has_no_fractional_early_stop() -> None:
    assert "validation_done_frac_early_stop" not in {
        field.name for field in fields(TrainingConfig)
    }
    for name in (
        "fcamp_largebox.yaml",
    ):
        load_config(ROOT / "configs" / name)


def test_flow_cps_config_has_no_legacy_algorithm_fields() -> None:
    fields_by_name = {field.name for field in fields(FlowCPSConfig)}
    assert "init_noise_std" not in fields_by_name
    assert "num_generations" not in fields_by_name
    assert "tail_bootstrap_steps" not in fields_by_name
    assert "actor_density" not in fields_by_name
    assert "failure_penalty" not in fields_by_name


def test_task_binds_motion_and_terrain() -> None:
    assert TASKS["largebox_plane"].terrain == "plane"
    assert TASKS["crawl_slope"].terrain == "slope"
    assert TASKS["largebox_plane"].motion_file.name == "sub3_largebox_003_mj.npz"
    assert TASKS["crawl_slope"].motion_file.name == "motion_crawl_slope.npz"


def test_override_cannot_create_a_second_config_entry() -> None:
    with pytest.raises(KeyError):
        load_config(ROOT / "configs" / "fcamp_largebox.yaml", ["environment.terrain=plane"])


def test_checkpoint_loader_ignores_only_removed_noop_fields() -> None:
    tree = yaml.safe_load(
        (ROOT / "configs" / "fcamp_largebox.yaml").read_text(encoding="utf-8")
    )
    tree["environment"]["motion_reference_mode"] = "frame"
    tree["parameters"]["critic_hidden_dims"] = [512, 256, 128]
    tree["parameters"]["style_prior"]["enabled"] = True
    tree["parameters"]["credit"]["mode"] = "causal_frame"
    tree["parameters"]["critics"]["sharing"] = "encoder"
    tree["training"]["official_reset_every"] = 0

    with pytest.raises(KeyError, match="unknown keys"):
        config_from_dict(tree)

    cfg = config_from_checkpoint_dict(tree)
    assert cfg.method == "fcamp"
    assert cfg.parameters.horizon == 4


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("environment.adaptive_motion_sampling", False),
        ("environment.physics_material_combine_mode", "multiply"),
        ("environment.contact_sensor_update_period", "physics"),
        ("parameters.cps_trainable", False),
        ("parameters.style_prior.optimizer", "adam"),
        ("parameters.style_prior.discriminator_warmup_rollouts", 0),
        ("parameters.credit.mode", "chunk_shared"),
        ("parameters.critics.sharing", "separate"),
        ("training.official_reset_every", 10),
    ],
)
def test_checkpoint_loader_rejects_removed_semantic_changes(
    path: str,
    value,
) -> None:
    tree = yaml.safe_load(
        (ROOT / "configs" / "fcamp_largebox.yaml").read_text(encoding="utf-8")
    )
    node = tree
    keys = path.split(".")
    for key in keys[:-1]:
        node = node[key]
    node[keys[-1]] = value

    with pytest.raises(ValueError, match="fixed FCAMP value"):
        config_from_checkpoint_dict(tree)


@pytest.mark.parametrize(
    "override",
    [
        "parameters.policy_lr=0.0",
        "parameters.value_lr=0.0",
        "parameters.critics.encoder_hidden_dims=[512,0]",
        "parameters.critics.head_hidden_dims=[0]",
    ],
)
def test_fcamp_rejects_degenerate_optimizers_and_critics(override: str) -> None:
    with pytest.raises(ValueError):
        load_config(ROOT / "configs" / "fcamp_largebox.yaml", [override])
