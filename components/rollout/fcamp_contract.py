FCAMP_CHECKPOINT_CONTRACT = {
    "fcamp_schema_version": 19,
    "action_contract": "residual_absolute_v1",
    "actor_observation_contract": "self_state_last_action_v1",
    "reward_contract": "pure_amp_dt_v1",
    "critic_contract": "scalar_amp_flow_v1",
    "reset_contract": "phase_reference_link_velocity_v2",
    "expert_velocity_contract": "npz_root_link_world_v1",
    "expert_sampling_contract": "independent_uniform_integer_disc_and_norm_v1",
    "validation_contract": "tracking_primary_v1",
    "ppo_contract": "fcamp_joint_flow_path_sum_v1",
}
