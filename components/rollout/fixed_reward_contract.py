FIXED_REWARD_CHECKPOINT_CONTRACT = {
    "fixed_reward_schema_version": 17,
    "control_semantics": "closed_loop_h1_v1",
    "policy_semantics": "holosoma_g1_wbt_ppo_v1",
    "action_semantics": "unsquashed_diagonal_normal_env_clip_100_v1",
    "exploration_semantics": "state_independent_trainable_diagonal_std_v1",
    "critic_semantics": "holosoma_asymmetric_scalar_mlp_v1",
    "gae_semantics": "holosoma_timeout_reward_bootstrap_global_norm_v1",
    "optimizer_semantics": "holosoma_adamw_adaptive_kl_v1",
    "minibatch_semantics": "single_permutation_reused_across_epochs_v1",
    "upstream_commit": "c5c836c68f423ac4565f57801ff4ff47ea56e5ac",
}
