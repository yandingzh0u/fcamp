from pathlib import Path

from engine.config import BeyondMimicConfig, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_beyondmimic_config_matches_official_recipe_and_common_platform() -> None:
    cfg = load_config(ROOT / "configs" / "beyondmimic_largebox.yaml")
    env = cfg.environment
    params = cfg.parameters
    assert isinstance(params, BeyondMimicConfig)
    assert env.platform_profile == "g1_largebox_50hz"
    assert (env.sim_dt, env.decimation, env.max_episode_steps) == (0.02, 4, 500)
    assert env.policy_observation_mode == "beyondmimic"
    assert env.motion_end_behavior == "resample_command"
    assert env.termination_mode == "beyondmimic"
    assert env.terminate_on_motion_end is False
    assert env.reset_phase_sampling == "beyondmimic"
    assert env.adaptive_motion_sampling is True
    assert env.adaptive_num_bins == 0
    assert env.adaptive_alpha == 0.001
    assert env.adaptive_uniform_ratio == 0.1
    assert env.adaptive_kernel_size == 1
    assert env.adaptive_lambda == 0.8
    assert env.startup_randomization and env.reset_noise and env.interval_pushes and env.observation_noise
    assert env.physics_material_combine_mode == "multiply"
    assert env.contact_sensor_update_period == "physics"
    assert params.actor_hidden_dims == (512, 256, 128)
    assert params.critic_hidden_dims == (512, 256, 128)
    assert params.activation == "elu"
    assert params.rollout_env_steps == 24
    assert (params.num_learning_epochs, params.num_mini_batches) == (5, 4)
    assert (params.clip_param, params.gamma, params.lam) == (0.2, 0.99, 0.95)
    assert (params.value_loss_coef, params.entropy_coef) == (1.0, 0.005)
    assert params.learning_rate == 0.001
    assert params.schedule == "adaptive"
    assert params.desired_kl == 0.01
    assert params.max_grad_norm == 1.0
    assert params.empirical_normalization is True
    assert params.init_noise_std == 1.0
    assert params.noise_std_type == "scalar"
    assert params.state_dependent_std is False
    assert params.normalize_advantage_per_mini_batch is False
