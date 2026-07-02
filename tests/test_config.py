from dataclasses import asdict, fields
from pathlib import Path

import pytest

from core.config import FPOConfig, MixGRPOConfig, PPOConfig, SFPOConfig, config_from_dict, load_config
from env.tasks import TASKS


ROOT = Path(__file__).resolve().parents[1]


def test_algorithm_configs_are_disjoint() -> None:
    ppo = load_config(ROOT / "configs" / "ppo.yaml")
    fpo = load_config(ROOT / "configs" / "fpo.yaml")
    mixgrpo = load_config(ROOT / "configs" / "mixgrpo.yaml")
    sfpo = load_config(ROOT / "configs" / "sfpo.yaml")
    assert isinstance(ppo.parameters, PPOConfig)
    assert isinstance(fpo.parameters, FPOConfig)
    assert isinstance(mixgrpo.parameters, MixGRPOConfig)
    assert isinstance(sfpo.parameters, SFPOConfig)
    assert "flow_steps" not in {field.name for field in fields(PPOConfig)}
    assert "entropy_coef" not in {field.name for field in fields(FPOConfig)}
    assert "critic_hidden_dims" not in {field.name for field in fields(MixGRPOConfig)}
    sfpo_fields = {field.name for field in fields(SFPOConfig)}
    assert "num_generations" not in sfpo_fields
    assert "tail_bootstrap_steps" not in sfpo_fields


def test_sfpo_config_is_ppo_aligned_h4() -> None:
    sfpo = load_config(ROOT / "configs" / "sfpo.yaml")
    assert sfpo.algorithm == "sfpo"
    assert isinstance(sfpo.parameters, SFPOConfig)
    assert sfpo.observation_group_size == 1
    # PPO-aligned shell with h=4 chunk policy: 6 chunks/update, chunk-internal
    # done mask, chunk-end reset, no entropy bonus, no first-life truncation.
    assert sfpo.parameters.horizon == 4
    assert sfpo.parameters.rollout_env_steps == 24
    assert sfpo.parameters.rollout_env_steps % sfpo.parameters.horizon == 0
    assert sfpo.parameters.desired_kl == 0.01
    assert sfpo.parameters.policy_lr == 0.001
    assert sfpo.parameters.value_lr == 0.001
    assert sfpo.parameters.use_clipped_value_loss is True
    assert sfpo.parameters.init_at_random_ep_len is True
    assert sfpo.parameters.terminal_penalty == 0.0
    assert sfpo.training.max_updates == 1000
    # SFPO-only flow/SDE head is preserved.
    assert sfpo.parameters.flow_steps >= 1
    assert sfpo.parameters.sde_eta > 0.0


def test_desired_kl_is_unified_per_step_budget() -> None:
    # desired_kl is a per-env-control-step KL budget, shared across PPO/SFPO/
    # MixGRPO so it never needs per-horizon hand-tuning. Raw targets are
    # derived internally as desired_kl * kl_units (kl_units == horizon for
    # chunk policies, 1 for PPO).
    ppo = load_config(ROOT / "configs" / "ppo.yaml")
    sfpo = load_config(ROOT / "configs" / "sfpo.yaml")
    mixgrpo = load_config(ROOT / "configs" / "mixgrpo.yaml")
    assert ppo.parameters.desired_kl == 0.01
    assert sfpo.parameters.desired_kl == 0.01
    assert mixgrpo.parameters.desired_kl == 0.01
    # Derived raw targets scale with horizon (sanity, not equality of value):
    assert ppo.parameters.desired_kl * 1 == 0.01
    assert sfpo.parameters.desired_kl * sfpo.parameters.horizon == 0.01 * 4
    assert mixgrpo.parameters.desired_kl * mixgrpo.parameters.horizon == 0.01 * 12


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
