from dataclasses import asdict, fields
from pathlib import Path

import pytest

from core.config import FPOConfig, MixGRPOConfig, PPOConfig, config_from_dict, load_config
from env.tasks import TASKS


ROOT = Path(__file__).resolve().parents[1]


def test_algorithm_configs_are_disjoint() -> None:
    ppo = load_config(ROOT / "configs" / "ppo.yaml")
    fpo = load_config(ROOT / "configs" / "fpo.yaml")
    mixgrpo = load_config(ROOT / "configs" / "mixgrpo.yaml")
    assert isinstance(ppo.parameters, PPOConfig)
    assert isinstance(fpo.parameters, FPOConfig)
    assert isinstance(mixgrpo.parameters, MixGRPOConfig)
    assert "flow_steps" not in {field.name for field in fields(PPOConfig)}
    assert "entropy_coef" not in {field.name for field in fields(FPOConfig)}
    assert "critic_hidden_dims" not in {field.name for field in fields(MixGRPOConfig)}


def test_task_binds_motion_and_terrain() -> None:
    assert TASKS["largebox_plane"].terrain == "plane"
    assert TASKS["crawl_slope"].terrain == "slope"
    assert TASKS["largebox_plane"].motion_file.name == "sub3_largebox_003_mj.npz"
    assert TASKS["crawl_slope"].motion_file.name == "motion_crawl_slope.npz"


def test_override_cannot_create_a_second_config_entry() -> None:
    with pytest.raises(KeyError):
        load_config(ROOT / "configs" / "ppo.yaml", ["environment.terrain=plane"])


def test_fpo_legacy_config_defaults_unclipped_value_loss() -> None:
    fpo = load_config(ROOT / "configs" / "fpo.yaml")
    tree = asdict(fpo)
    del tree["parameters"]["use_clipped_value_loss"]
    rebuilt = config_from_dict(tree)
    assert isinstance(rebuilt.parameters, FPOConfig)
    assert rebuilt.parameters.use_clipped_value_loss is False
