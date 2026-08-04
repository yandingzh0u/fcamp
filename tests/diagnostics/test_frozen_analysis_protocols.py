import json
from pathlib import Path


SPEC_PATH = (
    Path(__file__).resolve().parents[2]
    / "diagnostics"
    / "specs"
    / "largebox_discovery_v1.yaml"
)


def _spec() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def test_stage2_protocol_is_frozen_before_collection() -> None:
    protocol = _spec()["analysis_protocols"]["policy_class"]
    assert protocol["primary_teacher_checkpoint_update"] == 500
    assert protocol["seeds"] == [20260803, 20260804, 20260805]
    assert protocol["history_steps_for_aliasing"] == [1, 2, 4, 8, 16, 32]
    assert protocol["history_steps_for_bc"] == [1, 4, 8, 16, 32]
    assert protocol["probe_models"]["checkpoint_selection"] == (
        "best_validation_loss_test_untouched"
    )
    assert protocol["open_loop_replay"][
        "initial_joint_position_perturbation_fraction_of_joint_range"
    ] == [0.001, 0.005, 0.01]


def test_stage3_protocol_uses_original_amp_window_and_five_seeds() -> None:
    protocols = _spec()["analysis_protocols"]
    seeds = [20260803, 20260804, 20260805, 20260806, 20260807]
    assert protocols["source_classifier"]["seeds"] == seeds
    assert protocols["temporal_windows"]["steps"] == 10
    assert protocols["temporal_windows"]["allow_reset_crossing"] is False
    assert protocols["temporal_windows"]["allow_wrap_crossing"] is False
    assert protocols["offline_amp_critic"]["seeds"] == seeds
    assert protocols["offline_amp_critic"]["checkpoint_selection"] == (
        "final_epoch_no_validation_selection"
    )
    reliability = protocols["reward_reliability"]
    assert reliability["critic_family"] == "standard_commit_6901_amp_discriminator"
    assert reliability["domain_edges"].startswith("all_ordered_pairs_without_self")
    assert reliability["seeds"] == seeds
    assert reliability["source_classifier_scores_for_reward_icc"] == "forbidden"
    assert reliability["reverse_edge_reuse"].startswith("forbidden")


def test_stage3_primary_domain_catalog_is_preregistered_and_condition_matched() -> None:
    protocol = _spec()["analysis_protocols"]["domain_triangle_domains"]
    assert protocol["primary_collector_mode"] == "controlled_environment"
    catalog = protocol["catalog"]
    assert [entry["name"] for entry in catalog] == [
        "K", "T_u200", "T_u500", "A_amp", "B"
    ]
    assert [entry["family"] for entry in catalog] == [
        "K", "T_early", "T", "A_amp", "B"
    ]
    assert catalog[0]["frame_role"] == "reference_expert"
    assert all(entry["frame_role"] == "agent_physx" for entry in catalog[1:])
    assert all(entry["failure_label"] == "trajectory_eventual" for entry in catalog)
    assert all(
        entry["filters"]["collector_mode"] == "controlled_environment"
        and entry["filters"]["common_sigma"] == 0.0
        and entry["filters"]["eligible_for_primary_overlap"] is True
        for entry in catalog
    )
    assert catalog[-1]["filters"]["checkpoint_id"] == "GRU-H32-seed20260803"
    assert catalog[-1]["filters"]["checkpoint_lineage_id"] == "diag24-bc-v1"


def test_validation_protocols_share_the_frozen_u500_checkpoint() -> None:
    collection = _spec()["collection"]
    validation = collection["validation_protocols"]
    assert validation["summary_checkpoint_update"] == 500
    assert validation["controlled_noise_sigma"] in collection[
        "common_action_noise_scales"
    ]
    assert collection["protocol_details"]["after_done"] == (
        "do_not_reset_and_crop_each_trajectory_at_first_done"
    )
    assert "independent of rollout horizon" in collection["protocol_details"][
        "phase_grid"
    ]
    imitation = collection["imitation_contract"]
    assert imitation["frame_dim"] == 239
    assert imitation["window_steps"] == 10
    assert imitation["key_body_names"][2] == "head_link"
    assert "forbidden" in imitation["fk_aligned_agent_frame_role"]


def test_local_edge_budget_and_controls_are_frozen_before_online_results() -> None:
    protocol = _spec()["analysis_protocols"]["local_edge_causality"]
    assert protocol["ppo_seeds"] == [20260803, 20260804, 20260805]
    assert protocol["ppo_updates_per_edge"] == 50
    assert protocol["evaluation_every_updates"] == 5
    assert protocol["positive_rollout_mode"] == "controlled_environment"
    assert protocol["edge_selection"]["transition_completion_levels"] == [0.10, 0.50, 0.90]
    assert protocol["controls"] == [
        "direct_final_amp", "fixed_checkpoint_schedule", "bc_action_distillation"
    ]
    assert protocol["full_curriculum_forbidden_until_abort_guard_pass"] is True


def test_offline_family_screeners_are_frozen_and_do_not_enter_ppo() -> None:
    screeners = _spec()["analysis_protocols"]["family_screeners"]
    paired = screeners["paired_common_coordinate"]
    assert paired["paired_domains"] == ["K", "T_u500"]
    assert paired["linear_models"] == ["CCA", "PLS"]
    assert paired["seeds"] == [20260803, 20260804, 20260805]
    assert "PPO_reward" in paired["forbidden_uses"]
    prior = screeners["frozen_robot_domain_prior"]
    assert prior["raw_human_reference_positive"] is False
    assert prior["PPO"].startswith("forbidden")
    executable = screeners["closed_loop_executability"]
    assert executable["teacher_checkpoint_update"] == 500
    assert executable["candidate_time_scales"] == [0.75, 1.0, 1.25]
    assert "not_physical_reachability" in executable["name_guard"]
