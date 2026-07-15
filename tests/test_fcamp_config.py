from dataclasses import asdict, fields
from pathlib import Path

import pytest

from engine.config import FCAMPConfig, FlowCPSConfig, load_config, config_from_dict
from envs.tasks import TASKS


ROOT = Path(__file__).resolve().parents[1]


def test_fcamp_config_is_h4_w16_flow_cps() -> None:
    cfg = load_config(ROOT / "configs" / "fcamp_largebox.yaml")
    assert cfg.method == "fcamp"
    assert isinstance(cfg.parameters, FCAMPConfig)
    assert cfg.observation_group_size == 1
    assert cfg.parameters.horizon == 4
    assert cfg.parameters.rollout_env_steps == 24
    assert cfg.parameters.rollout_env_steps % cfg.parameters.horizon == 0
    assert cfg.parameters.flow_steps == 4
    assert cfg.parameters.action_squash_scale == 5.0
    assert cfg.parameters.cps_noise_level == 0.8
    assert cfg.parameters.cps_trainable is True
    assert cfg.parameters.cps_cov_rank == 8
    assert cfg.parameters.desired_kl == 0.01
    assert cfg.parameters.policy_lr == 0.0003
    assert cfg.parameters.value_lr == 0.0003
    assert cfg.parameters.style_prior.obs_steps == 16
    assert cfg.training.max_updates == 500


def test_flow_cps_config_has_no_legacy_algorithm_fields() -> None:
    fields_by_name = {field.name for field in fields(FlowCPSConfig)}
    assert "init_noise_std" not in fields_by_name
    assert "num_generations" not in fields_by_name
    assert "tail_bootstrap_steps" not in fields_by_name
    assert "actor_density" not in fields_by_name
    assert "failure_penalty" not in fields_by_name


def test_legacy_algorithm_key_is_normalized_to_method() -> None:
    cfg = load_config(ROOT / "configs" / "fcamp_largebox.yaml")
    tree = asdict(cfg)
    tree["algorithm"] = tree.pop("method")
    rebuilt = config_from_dict(tree, ROOT / "configs" / "legacy.yaml")
    assert rebuilt.method == "fcamp"
    assert rebuilt.algorithm == "fcamp"


def test_task_binds_motion_and_terrain() -> None:
    assert TASKS["largebox_plane"].terrain == "plane"
    assert TASKS["crawl_slope"].terrain == "slope"
    assert TASKS["largebox_plane"].motion_file.name == "sub3_largebox_003_mj.npz"
    assert TASKS["crawl_slope"].motion_file.name == "motion_crawl_slope.npz"


def test_override_cannot_create_a_second_config_entry() -> None:
    with pytest.raises(KeyError):
        load_config(ROOT / "configs" / "fcamp_largebox.yaml", ["environment.terrain=plane"])
