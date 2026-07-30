"""Strict checkpoint semantics for the local-asset pure AMP implementation."""

AMP_CHECKPOINT_CONTRACT = {
    "amp_schema_version": 5,
    "action_contract": "normalized_absolute_direct_h_configurable_v1",
    "actor_architecture_contract": "mimickit_mlp_1024_512_relu_zero_bias_v2",
    "actor_observation_contract": "g1_self_state_official_field_order_v3",
    "exploration_contract": "fixed_diagonal_gaussian_std_0p05_v1",
    "reward_contract": "pure_amp_no_dt_scaling_v2",
    "critic_contract": "scalar_state_value_mlp_1024_512_relu_v1",
    "asset_contract": "local_g1_urdf_and_user_motion_v1",
    "reset_contract": "uniform_continuous_pose_slerp_velocity_left_frame_v3",
    "policy_history_contract": "demo_predecessor_seeded_w10_v1",
    "credit_contract": "official_unbiased_adv_immediate_amp_chunk_gae_v2",
    "expert_velocity_contract": "mujoco_root_ang_world_velocity_left_frame_v3",
    "expert_sampling_contract": "independent_uniform_full_motion_v1",
    "validation_contract": "physical_survival_no_tracking_terminal_v2",
    "ppo_contract": "direct_absolute_gaussian_chunk_joint_ratio_v1",
    "discriminator_contract": "unconditional_amp_window_bce_gp_v1",
    "replay_contract": "single_global_circular_capacity_permutation_v2",
    "termination_contract": "timeout_illegal_contact_numerical_only_v1",
}

__all__ = ["AMP_CHECKPOINT_CONTRACT"]
