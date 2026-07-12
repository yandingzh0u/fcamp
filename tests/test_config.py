from dataclasses import asdict, fields
from pathlib import Path

import pytest

from core.config import (
    ChunkPPOConfig,
    FPOPlusPlusConfig,
    FQLConfig,
    FlowRLConfig,
    OriginalFPOConfig,
    PolicyFlowConfig,
    ReinFlowConfig,
    SACFlowConfig,
    SEARConfig,
    PPOConfig,
    SFPOConfig,
    SFPOGaussianConfig,
    config_from_dict,
    load_config,
)
from env.tasks import TASKS


ROOT = Path(__file__).resolve().parents[1]


def test_algorithm_configs_are_disjoint() -> None:
    ppo = load_config(ROOT / "configs" / "ppo.yaml")
    fpo = load_config(ROOT / "configs" / "fpo_plus_plus.yaml")
    original_fpo = load_config(ROOT / "configs" / "fpo.yaml")
    flowrl = load_config(ROOT / "configs" / "flowrl.yaml")
    fql = load_config(ROOT / "configs" / "fql.yaml")
    reinflow = load_config(ROOT / "configs" / "reinflow.yaml")
    policyflow = load_config(ROOT / "configs" / "policyflow.yaml")
    sac_flow = load_config(ROOT / "configs" / "sac_flow.yaml")
    sear = load_config(ROOT / "configs" / "sear.yaml")
    sfpo = load_config(ROOT / "configs" / "sfpo.yaml")
    sfpo_gaussian = load_config(ROOT / "configs" / "sfpo_gaussian.yaml")
    chunk_ppo = load_config(ROOT / "configs" / "chunk_ppo.yaml")
    assert isinstance(ppo.parameters, PPOConfig)
    assert isinstance(fpo.parameters, FPOPlusPlusConfig)
    assert isinstance(original_fpo.parameters, OriginalFPOConfig)
    assert isinstance(flowrl.parameters, FlowRLConfig)
    assert isinstance(fql.parameters, FQLConfig)
    assert isinstance(reinflow.parameters, ReinFlowConfig)
    assert isinstance(policyflow.parameters, PolicyFlowConfig)
    assert isinstance(sac_flow.parameters, SACFlowConfig)
    assert isinstance(sear.parameters, SEARConfig)
    assert isinstance(sfpo.parameters, SFPOConfig)
    assert isinstance(sfpo_gaussian.parameters, SFPOGaussianConfig)
    assert isinstance(chunk_ppo.parameters, ChunkPPOConfig)
    assert "flow_steps" not in {field.name for field in fields(PPOConfig)}
    assert "entropy_coef" not in {field.name for field in fields(FPOPlusPlusConfig)}
    sfpo_fields = {field.name for field in fields(SFPOConfig)}
    assert "num_generations" not in sfpo_fields
    assert "tail_bootstrap_steps" not in sfpo_fields
    assert "actor_density" not in sfpo_fields
    assert "failure_penalty" not in sfpo_fields


def test_flowrl_config_preserves_official_core_and_common_budget() -> None:
    flowrl = load_config(ROOT / "configs" / "flowrl.yaml")
    assert flowrl.algorithm == "flowrl"
    assert isinstance(flowrl.parameters, FlowRLConfig)
    assert flowrl.parameters.horizon == 1
    assert flowrl.parameters.rollout_env_steps == 24
    assert flowrl.parameters.flow_steps == 4
    assert flowrl.parameters.gradient_steps_per_update == 24
    assert flowrl.parameters.policy_delay == 2
    assert flowrl.parameters.expectile == 0.9
    assert flowrl.parameters.target_tau == 0.95


def test_policyflow_config_preserves_official_core_and_common_budget() -> None:
    policyflow = load_config(ROOT / "configs" / "policyflow.yaml")
    assert policyflow.algorithm == "policyflow"
    assert isinstance(policyflow.parameters, PolicyFlowConfig)
    assert policyflow.parameters.horizon == 1
    assert policyflow.parameters.rollout_env_steps == 24
    assert policyflow.parameters.flow_steps == 4
    assert policyflow.parameters.clip_range == 0.2
    assert policyflow.parameters.gaussian_entropy_coef == 0.002
    assert policyflow.parameters.brownian_reg_coef == 0.006


def test_sac_flow_config_preserves_official_core_and_common_budget() -> None:
    sac_flow = load_config(ROOT / "configs" / "sac_flow.yaml")
    assert sac_flow.algorithm == "sac-flow"
    assert isinstance(sac_flow.parameters, SACFlowConfig)
    assert sac_flow.parameters.horizon == 1
    assert sac_flow.parameters.rollout_env_steps == 24
    assert sac_flow.parameters.flow_steps == 4
    assert sac_flow.parameters.gradient_steps_per_update == 24
    assert sac_flow.parameters.policy_delay == 3
    assert sac_flow.parameters.target_entropy == 0.0
    assert sac_flow.parameters.use_batch_renorm is True


def test_sear_config_preserves_official_core_and_frame_budget() -> None:
    sear = load_config(ROOT / "configs" / "sear.yaml")
    assert sear.algorithm == "sear"
    assert isinstance(sear.parameters, SEARConfig)
    assert sear.parameters.horizon == 4
    assert sear.parameters.rollout_env_steps == 24
    assert sear.parameters.critic_num_heads == 16
    assert sear.parameters.critic_num_blocks == 2
    assert sear.parameters.num_value_bins == 101
    assert sear.parameters.target_tau == 0.05
    assert sear.parameters.gradient_steps_per_update == 6


def test_reinflow_config_uses_official_h4_chain_likelihood() -> None:
    reinflow = load_config(ROOT / "configs" / "reinflow.yaml")
    assert reinflow.algorithm == "reinflow"
    assert isinstance(reinflow.parameters, ReinFlowConfig)
    assert reinflow.parameters.horizon == 4
    assert reinflow.parameters.rollout_env_steps == 24
    assert reinflow.parameters.flow_steps == 4
    assert reinflow.parameters.min_denoising_std == 0.1
    assert reinflow.parameters.max_denoising_std == 0.24
    assert reinflow.parameters.normalize_denoising_horizon is True
    assert reinflow.parameters.normalize_action_dimension is True


def test_fql_config_preserves_official_objective_and_common_budget() -> None:
    fql = load_config(ROOT / "configs" / "fql.yaml")
    assert fql.algorithm == "fql"
    assert isinstance(fql.parameters, FQLConfig)
    assert fql.parameters.horizon == 1
    assert fql.parameters.rollout_env_steps == 24
    assert fql.parameters.flow_steps == 4
    assert fql.parameters.environment_action_scale == 1.0
    assert fql.parameters.gradient_steps_per_update == 24
    assert fql.parameters.target_tau == 0.005
    assert fql.parameters.q_aggregation == "mean"
    assert fql.parameters.alpha == 10.0
    assert fql.parameters.normalize_q_loss is True
    assert fql.parameters.offline_dataset_path == ""
    assert fql.parameters.offline_pretrain_gradient_steps == 0
    assert fql.parameters.recent_fraction == 0.0


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


def test_sfpo_gaussian_config_is_registered_h4() -> None:
    sfpo_gaussian = load_config(ROOT / "configs" / "sfpo_gaussian.yaml")
    assert sfpo_gaussian.algorithm == "sfpo-gaussian"
    assert isinstance(sfpo_gaussian.parameters, SFPOGaussianConfig)
    assert sfpo_gaussian.parameters.horizon == 4
    assert sfpo_gaussian.parameters.rollout_env_steps == 24
    assert sfpo_gaussian.parameters.rollout_env_steps % sfpo_gaussian.parameters.horizon == 0
    assert sfpo_gaussian.parameters.flow_steps == 4
    assert sfpo_gaussian.parameters.action_squash_scale == 5.0
    assert sfpo_gaussian.parameters.init_noise_std == 0.5
    gaussian_fields = {field.name for field in fields(SFPOGaussianConfig)}
    assert "cps_noise_level" not in gaussian_fields
    assert "cps_trainable" not in gaussian_fields
    assert "cps_cov_rank" not in gaussian_fields


def test_chunk_ppo_config_is_registered_h4() -> None:
    chunk_ppo = load_config(ROOT / "configs" / "chunk_ppo.yaml")
    assert chunk_ppo.algorithm == "chunk-ppo"
    assert isinstance(chunk_ppo.parameters, ChunkPPOConfig)
    assert chunk_ppo.parameters.horizon == 4
    assert chunk_ppo.parameters.num_steps_per_env == 24
    assert chunk_ppo.parameters.num_steps_per_env % chunk_ppo.parameters.horizon == 0
    assert chunk_ppo.parameters.desired_kl == 0.01
    assert chunk_ppo.parameters.actor_learning_rate == 0.0003
    assert chunk_ppo.parameters.critic_learning_rate == 0.001


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


def test_fpo_plus_plus_defaults_unclipped_value_loss() -> None:
    fpo = load_config(ROOT / "configs" / "fpo_plus_plus.yaml")
    tree = asdict(fpo)
    del tree["parameters"]["use_clipped_value_loss"]
    rebuilt = config_from_dict(tree)
    assert isinstance(rebuilt.parameters, FPOPlusPlusConfig)
    assert rebuilt.parameters.use_clipped_value_loss is False


def test_original_fpo_is_separate_from_fpo_plus_plus() -> None:
    original = load_config(ROOT / "configs" / "fpo.yaml")
    plus_plus = load_config(ROOT / "configs" / "fpo_plus_plus.yaml")
    assert original.algorithm == "fpo"
    assert plus_plus.algorithm == "fpo++"
    assert isinstance(original.parameters, OriginalFPOConfig)
    assert isinstance(plus_plus.parameters, FPOPlusPlusConfig)
    assert original.parameters.average_losses_before_exp is True
    assert original.parameters.flow_steps == plus_plus.parameters.flow_steps == 4
    assert original.parameters.fpo_num_mc == plus_plus.parameters.fpo_num_mc == 16
    assert original.parameters.num_steps_per_env == plus_plus.parameters.num_steps_per_env == 24
