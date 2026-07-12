from dataclasses import fields
from pathlib import Path

import pytest

from core.config import (
    PPOConfig,
    SFPOConfig,
    load_config,
)
from env.tasks import TASKS


ROOT = Path(__file__).resolve().parents[1]


def test_algorithm_configs_are_disjoint() -> None:
    ppo = load_config(ROOT / "configs" / "ppo.yaml")
    sfpo = load_config(ROOT / "configs" / "sfpo.yaml")
    assert isinstance(ppo.parameters, PPOConfig)
    assert isinstance(sfpo.parameters, SFPOConfig)
    assert "flow_steps" not in {field.name for field in fields(PPOConfig)}
    sfpo_fields = {field.name for field in fields(SFPOConfig)}
    assert "num_generations" not in sfpo_fields
    assert "tail_bootstrap_steps" not in sfpo_fields
    assert "actor_density" not in sfpo_fields
    assert "failure_penalty" not in sfpo_fields

def test_sfpo_config_is_lowrank_cps_h4() -> None:
    sfpo = load_config(ROOT / "configs" / "sfpo.yaml")
    assert sfpo.algorithm == "sfpo"
    assert isinstance(sfpo.parameters, SFPOConfig)
    assert sfpo.observation_group_size == 1
    assert sfpo.parameters.horizon == 4
    assert sfpo.parameters.rollout_env_steps == 24
    assert sfpo.parameters.rollout_env_steps % sfpo.parameters.horizon == 0
    assert sfpo.parameters.flow_steps == 4
    assert sfpo.parameters.action_squash_scale == 5.0
    assert sfpo.parameters.cps_noise_level == 0.8
    assert sfpo.parameters.cps_trainable is True
    assert sfpo.parameters.cps_cov_rank == 8
    assert sfpo.parameters.desired_kl == 0.01
    assert sfpo.parameters.policy_lr == 0.0003
    assert sfpo.parameters.value_lr == 0.0003
    assert sfpo.parameters.init_at_random_ep_len is True
    sfpo_fields = {field.name for field in fields(SFPOConfig)}
    assert "init_noise_std" not in sfpo_fields
    assert sfpo.parameters.gae_lambda == 0.95
    assert sfpo.parameters.kl_early_stop_factor == 4.0
    assert sfpo.parameters.advantage_normalization == "global"
    assert sfpo.training.max_updates == 1000

def test_desired_kl_is_unified_per_step_budget() -> None:
    # desired_kl is a per-env-control-step KL budget shared across PPO/SFPO.
    #
    #   PPO: per-step KL
    #   SFPO: per-frame KL → kl_units = 1
    #
    # The raw target internally is desired_kl * kl_units; for SFPO kl_units=1
    # so the raw target equals desired_kl directly.
    ppo = load_config(ROOT / "configs" / "ppo.yaml")
    sfpo = load_config(ROOT / "configs" / "sfpo.yaml")
    assert ppo.parameters.desired_kl == 0.01
    assert sfpo.parameters.desired_kl == 0.01
    # SFPO kl_units=1 (per-frame), not horizon-scaled.
    assert sfpo.parameters.desired_kl * 1 == 0.01


def test_task_binds_motion_and_terrain() -> None:
    assert TASKS["largebox_plane"].terrain == "plane"
    assert TASKS["crawl_slope"].terrain == "slope"
    assert TASKS["largebox_plane"].motion_file.name == "sub3_largebox_003_mj.npz"
    assert TASKS["crawl_slope"].motion_file.name == "motion_crawl_slope.npz"


def test_override_cannot_create_a_second_config_entry() -> None:
    with pytest.raises(KeyError):
        load_config(ROOT / "configs" / "ppo.yaml", ["environment.terrain=plane"])
