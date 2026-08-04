from __future__ import annotations

import json
from pathlib import Path

import pytest

from diagnostics.common.reporting import (
    DECISION_VARIABLES,
    build_suite_result,
    load_structured_file,
    sha256_file,
    topological_diagnostics,
    validate_spec,
)
from diagnostics.common.status import (
    DiagnosticResult,
    DiagnosticStatus,
    aggregate_status,
)
from tools.run_discovery_suite import _dependency_blockers, run
from tools.diag_64_render_discovery_report import _completed_suite_status


ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = ROOT / "diagnostics" / "specs" / "largebox_discovery_v1.yaml"
RESULT_SCHEMA_PATH = ROOT / "diagnostics" / "schemas" / "result_schema.json"
ROLLOUT_SCHEMA_PATH = ROOT / "diagnostics" / "schemas" / "rollout_schema.json"


def test_frozen_spec_is_complete_and_topological() -> None:
    spec = load_structured_file(SPEC_PATH)
    diagnostics = validate_spec(spec)
    ordered = topological_diagnostics(spec)

    assert spec["suite_id"] == "largebox_discovery_v1"
    assert spec["fail_closed"] is True
    assert spec["status_values"] == [status.value for status in DiagnosticStatus]
    assert tuple(spec["decision_variables"]) == DECISION_VARIABLES
    assert len(spec["decision_matrix"]) == 6
    assert [row["priority"] for row in spec["decision_matrix"]] == [1, 2, 3, 4, 5, 6]

    position = {item["id"]: index for index, item in enumerate(ordered)}
    assert set(position) == {item["id"] for item in diagnostics}
    for item in ordered:
        assert all(position[dependency] < position[item["id"]] for dependency in item["dependencies"])


def test_all_frozen_numeric_gates_are_exact() -> None:
    thresholds = load_structured_file(SPEC_PATH)["thresholds"]

    assert thresholds["reference_free_bc_green"] == {
        "motion_completion_relative_to_teacher_min": 0.80,
        "failure_rate_increase_max": 0.10,
        "action_nrmse_max": 0.20,
        "minimum_passing_seeds": 2,
        "total_seeds": 3,
        "history_steps": [1, 4, 8, 16, 32],
    }
    overlap = thresholds["empirical_effective_overlap_edge"]
    assert overlap["source_auc_max"] == 0.90
    assert overlap["posterior_overlap_tau"] == 0.10
    assert overlap["posterior_overlap_fraction_min"] == 0.10
    assert overlap["forward_knn_coverage_min"] == 0.20
    assert overlap["reverse_knn_coverage_min"] == 0.20
    assert overlap["forward_ess_over_n_min"] == 0.10
    assert overlap["reverse_ess_over_n_min"] == 0.10
    assert overlap["reward_icc_min"] == 0.75

    reward = thresholds["reward_validity_green"]
    assert reward["strict_pareto_pairwise_accuracy_min"] == 0.65
    assert reward["same_state_branch_pairwise_accuracy_min"] == 0.65
    assert reward["seed_icc_min"] == 0.75
    assert reward["clearly_failed_branch_in_reward_top_decile"] is False
    assert reward["cem_significant_reward_hacking"] is False

    edge = thresholds["single_edge_pass"]
    assert edge["minimum_passing_ppo_seeds"] == 2
    assert edge["total_ppo_seeds"] == 3
    assert edge["target_distance_reduction_min"] == 0.30
    assert edge["failure_rate_increase_max"] == 0.10
    assert edge["task_or_progress_drop_max"] == 0.05

    latent = thresholds["paired_common_coordinate_viable"]
    assert latent["latent_source_auc_max"] == 0.65
    assert latent["cross_domain_retrieval_min"] == 0.70
    assert latent["contact_future_prediction_relative_to_raw_min"] == 0.80
    assert latent["latent_pairwise_spearman_min"] == 0.75

    frozen = thresholds["frozen_robot_domain_prior_viable"]
    assert frozen["same_state_strict_pair_accuracy_min"] == 0.65
    assert frozen["seed_icc_min"] == 0.80
    assert frozen["cem_significant_reward_exploit"] is False

    executable = thresholds["closed_loop_executability_viable"]
    assert executable["positive_fraction_min"] == 0.15
    assert executable["negative_fraction_min"] == 0.15
    assert executable["held_out_auroc_min"] == 0.80
    assert executable["ece_max"] == 0.08
    assert executable["same_state_candidate_pair_accuracy_min"] == 0.70


def test_result_schema_requires_every_status_and_decision_field() -> None:
    schema = json.loads(RESULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    rollout_schema = json.loads(ROLLOUT_SCHEMA_PATH.read_text(encoding="utf-8"))

    required = set(schema["required"])
    assert {
        "execution",
        "overall_status",
        "status_counts",
        "diagnostics",
        "decision_variables",
        "decision",
        "artifacts",
    } <= required
    assert set(schema["properties"]["decision_variables"]["required"]) == set(
        DECISION_VARIABLES
    )
    assert schema["$defs"]["diagnosticStatus"]["enum"] == [
        status.value for status in DiagnosticStatus
    ]
    assert set(schema["properties"]["status_counts"]["required"]) == {
        status.value for status in DiagnosticStatus
    }
    assert "diagnostic_results" in schema["properties"]["artifacts"]["required"]
    final_stage = load_structured_file(SPEC_PATH)["stages"][-1]["diagnostics"][-1]
    assert "diagnostic_results.txt" in final_stage["expected_outputs"]

    assert set(rollout_schema["required"]) == {
        "metadata",
        "trajectory",
        "observation",
        "state",
        "action",
        "reference",
        "outcome",
        "imitation",
    }
    metadata_required = set(rollout_schema["properties"]["metadata"]["required"])
    assert {
        "source_snapshot_sha256",
        "resolved_config_sha256",
        "checkpoint_sha256",
        "checkpoint_lineage_id",
        "robot_asset_sha256",
        "motion_sha256",
        "action_schema_sha256",
        "collector_mode",
        "collector_seed",
    } <= metadata_required


def test_schema_complete_result_skeleton_is_fail_closed() -> None:
    spec = load_structured_file(SPEC_PATH)
    diagnostic = DiagnosticResult(
        diagnostic_id="00",
        name="freeze_manifest",
        stage="0_identity_lineage",
        status=DiagnosticStatus.SKIPPED_DEPENDENCY,
        reason="test",
    )
    payload = build_suite_result(
        spec=spec,
        spec_sha256=sha256_file(SPEC_PATH),
        spec_path=str(SPEC_PATH),
        output_dir="/tmp/unused",
        dry_run=True,
        results=[diagnostic],
        started_at="2026-08-04T00:00:00Z",
        finished_at="2026-08-04T00:00:01Z",
    )

    schema = json.loads(RESULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    assert set(payload) == set(schema["required"])
    assert payload["overall_status"] == "SKIPPED_DEPENDENCY"
    assert payload["decision"]["status"] == "DEFERRED"
    assert payload["decision"]["selected_family"] is None
    assert all(
        payload["decision_variables"][name]["value"] == "UNKNOWN"
        for name in DECISION_VARIABLES
    )
    assert sum(payload["status_counts"].values()) == len(payload["diagnostics"])


def test_status_aggregation_never_fails_open() -> None:
    def result(status: DiagnosticStatus) -> DiagnosticResult:
        return DiagnosticResult("00", "x", "stage", status)

    assert aggregate_status([result(DiagnosticStatus.PASS)]) is DiagnosticStatus.PASS
    assert aggregate_status([]) is DiagnosticStatus.INVALID_PROTOCOL
    assert (
        aggregate_status(
            [result(DiagnosticStatus.PASS), result(DiagnosticStatus.SKIPPED_DEPENDENCY)]
        )
        is DiagnosticStatus.SKIPPED_DEPENDENCY
    )
    assert (
        aggregate_status([result(DiagnosticStatus.FAIL), result(DiagnosticStatus.INVALID_PROTOCOL)])
        is DiagnosticStatus.INVALID_PROTOCOL
    )


def test_runner_dry_run_is_complete_and_has_no_filesystem_side_effect(tmp_path: Path) -> None:
    output_dir = tmp_path / "must_not_be_created"
    payload, exit_code = run(
        [
            "--spec",
            str(SPEC_PATH),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    spec = load_structured_file(SPEC_PATH)
    expected_count = len(topological_diagnostics(spec))
    assert exit_code == 0
    assert not output_dir.exists()
    assert len(payload["diagnostics"]) == expected_count
    assert payload["status_counts"]["SKIPPED_DEPENDENCY"] == expected_count
    assert payload["overall_status"] == "SKIPPED_DEPENDENCY"
    assert all(item["reason"] == "dry_run_not_executed" for item in payload["diagnostics"])


def test_spec_rejects_decision_matrix_drift() -> None:
    spec = load_structured_file(SPEC_PATH)
    spec["decision_matrix"] = spec["decision_matrix"][:-1]
    with pytest.raises(ValueError, match="exactly six"):
        validate_spec(spec)


def test_recorded_valid_allows_unknown_summary_but_not_invalid_protocol() -> None:
    diagnostic = {"id": "99", "dependency_policy": "recorded_valid", "dependencies": ["01"]}
    skipped = DiagnosticResult(
        "01", "upstream", "stage", DiagnosticStatus.SKIPPED_DEPENDENCY
    )
    invalid = DiagnosticResult(
        "01", "upstream", "stage", DiagnosticStatus.INVALID_PROTOCOL
    )
    assert _dependency_blockers(diagnostic, {"01": skipped}) == ()
    assert _dependency_blockers(diagnostic, {"01": invalid}) == ("01",)


def test_final_txt_status_count_includes_diag64_itself() -> None:
    rows = [
        {"status": "PASS"},
        {"status": "FAIL"},
        {"status": "SKIPPED_DEPENDENCY"},
        {"status": "PASS"},
    ]
    completed = _completed_suite_status(
        {"overall_status": "SKIPPED_DEPENDENCY", "status_counts": {}}, rows
    )
    assert sum(completed["status_counts"].values()) == len(rows)
    assert completed["status_counts"]["PASS"] == 2
    assert completed["overall_status"] == "FAIL"
