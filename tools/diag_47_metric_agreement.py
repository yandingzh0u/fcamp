#!/usr/bin/env python3
"""Aggregate Stage-4 evidence into the frozen reward-validity decision T."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.manifest import (
    FAIL,
    INVALID_PROTOCOL,
    PASS,
    SKIPPED_DEPENDENCY,
    DependencyUnavailable,
    ProtocolError,
    diagnostic_result,
    load_spec,
    read_json,
    write_json_exclusive,
)
from diagnostics.common.policy_class_probe import output_dir_from_args
from diagnostics.common.reward_stage4 import RewardValidityProtocol


STATUS_PATHS = {
    "41": "score_outcome.json",
    "42": "branching.json",
    "43": "local_rank.json",
    "44": "exploitability.json",
    "45": "intervention.json",
    "46": "videos/blind_pair_manifest.status.json",
}


def _load_all(output_dir: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    artifacts: dict[str, dict[str, Any]] = {}
    unavailable: list[str] = []
    for diagnostic_id, relative in STATUS_PATHS.items():
        path = output_dir / relative
        if not path.is_file():
            unavailable.append(f"diag_{diagnostic_id}:missing")
            continue
        payload = read_json(path)
        if not isinstance(payload, dict):
            raise ProtocolError(f"diag_{diagnostic_id} status artifact is not a mapping")
        status = str(payload.get("status", ""))
        if status not in {PASS, FAIL, INVALID_PROTOCOL, SKIPPED_DEPENDENCY}:
            raise ProtocolError(f"diag_{diagnostic_id} has unknown status {status!r}")
        artifacts[diagnostic_id] = payload
        if status in {INVALID_PROTOCOL, SKIPPED_DEPENDENCY}:
            unavailable.append(f"diag_{diagnostic_id}:{status}")
    return artifacts, unavailable


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    node: Any = payload
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            raise ProtocolError(f"Stage-4 artifact lacks {'.'.join(keys)}")
        node = node[key]
    return node


def _offline_agreement(artifacts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if "41" not in artifacts or artifacts["41"].get("status") not in {PASS, FAIL}:
        return {"status": "UNKNOWN", "reason": "diag_41 unavailable"}
    rewards = _nested(artifacts["41"], "evidence", "reward_results")
    return {
        family: {
            "strict_physical_outcome_pairwise_accuracy": _nested(
                values, "strict_same_snapshot_pareto", "ensemble", "accuracy"
            ),
            "seed_icc": _nested(values, "seed_agreement", "icc_consistency"),
            "per_primary_outcome_spearman": values.get("primary_outcome_spearman", {}),
            "per_tracking_metric_spearman": values.get("secondary_metric_spearman", {}),
            "failed_in_top_decile_fraction": values.get("failed_in_top_decile_fraction"),
        }
        for family, values in rewards.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(root, spec, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agreement_path = output_dir / "agreement.json"
    gate_path = output_dir / "reward_validity_gate.json"
    try:
        protocol = RewardValidityProtocol.from_spec(spec)
        thresholds = spec.get("thresholds", {}).get("reward_validity_green")
        if not isinstance(thresholds, dict):
            raise ProtocolError("thresholds.reward_validity_green is absent")
        if tuple(int(v) for v in thresholds.get("branch_horizons", ())) != protocol.branch_horizons:
            raise ProtocolError("reward-validity threshold horizons differ from Stage-4 protocol")
        artifacts, unavailable = _load_all(output_dir)
        offline = _offline_agreement(artifacts)
        agreement_evidence: dict[str, Any] = {
            "offline_reward_vs_outcomes": offline,
            "same_state_branching": artifacts.get("42", {}).get("evidence", {}),
            "local_action_ranking": artifacts.get("43", {}).get("evidence", {}),
            "cem_exploitability": artifacts.get("44", {}).get("evidence", {}),
            "intervention_sensitivity": artifacts.get("45", {}).get("evidence", {}),
            "blind_video": artifacts.get("46", {}).get("evidence", {}),
            "external_video_preferences": {
                "status": "NOT_COLLECTED_OPTIONAL",
                "claim": "no human/video preference was imputed from physical metrics",
            },
            "metric_combination": "none; every agreement statistic remains separate",
            "primary_candidate": "T_u500-positive persisted AMP critic",
            "standard_AMP_control": "K-positive persisted AMP critic",
            "A_mix": "legacy_quarantined",
            "unavailable_evidence": unavailable,
        }
        if unavailable:
            agreement = diagnostic_result(
                "47",
                SKIPPED_DEPENDENCY,
                summary="metric agreement is partial because real Stage-4 evidence is unavailable",
                evidence=agreement_evidence,
                errors=unavailable,
            )
            gate = diagnostic_result(
                "47",
                SKIPPED_DEPENDENCY,
                summary="reward validity remains UNKNOWN until every frozen real-evidence diagnostic completes",
                evidence={
                    "decision_variable_T": "UNKNOWN",
                    "decision_reason": "missing or skipped real branch/CEM/intervention/video evidence",
                    "unavailable_evidence": unavailable,
                    "fail_closed": True,
                    "primary_candidate": "T_u500",
                    "A_mix": "legacy_quarantined",
                },
                errors=unavailable,
            )
        else:
            candidate = "T_u500"
            global_accuracy = float(
                _nested(
                    artifacts["41"],
                    "evidence",
                    "reward_results",
                    candidate,
                    "strict_same_snapshot_pareto",
                    "ensemble",
                    "accuracy",
                )
            )
            global_icc = float(
                _nested(
                    artifacts["41"],
                    "evidence",
                    "reward_results",
                    candidate,
                    "seed_agreement",
                    "icc_consistency",
                )
            )
            branch_accuracy = float(
                _nested(
                    artifacts["42"],
                    "evidence",
                    "reward_accuracy",
                    candidate,
                    "ensemble",
                    "accuracy",
                )
            )
            branch_icc = float(
                _nested(
                    artifacts["42"],
                    "evidence",
                    "reward_accuracy",
                    candidate,
                    "seed_agreement",
                    "icc_consistency",
                )
            )
            branch_horizons = tuple(
                int(v) for v in _nested(artifacts["42"], "evidence", "horizons")
            )
            if branch_horizons != protocol.branch_horizons:
                raise ProtocolError("diag_42 branch horizons changed before the gate")
            failed_top = bool(
                _nested(
                    artifacts["42"],
                    "evidence",
                    "failed_top_decile",
                    candidate,
                    "clearly_failed_branch_in_top_decile",
                )
            )
            cem_hacking = bool(
                _nested(artifacts["44"], "evidence", "significant_reward_hacking")
            )
            tests = {
                "strict_pareto_pairwise_accuracy": {
                    "value": global_accuracy,
                    "threshold": float(thresholds["strict_pareto_pairwise_accuracy_min"]),
                    "passed": global_accuracy >= float(thresholds["strict_pareto_pairwise_accuracy_min"]),
                },
                "same_state_branch_pairwise_accuracy": {
                    "value": branch_accuracy,
                    "threshold": float(thresholds["same_state_branch_pairwise_accuracy_min"]),
                    "passed": branch_accuracy >= float(thresholds["same_state_branch_pairwise_accuracy_min"]),
                },
                "seed_icc": {
                    "value": min(global_icc, branch_icc),
                    "global_value": global_icc,
                    "same_state_value": branch_icc,
                    "threshold": float(thresholds["seed_icc_min"]),
                    "passed": min(global_icc, branch_icc) >= float(thresholds["seed_icc_min"]),
                },
                "clearly_failed_branch_in_reward_top_decile": {
                    "value": failed_top,
                    "required": bool(thresholds["clearly_failed_branch_in_reward_top_decile"]),
                    "passed": failed_top is bool(thresholds["clearly_failed_branch_in_reward_top_decile"]),
                },
                "cem_significant_reward_hacking": {
                    "value": cem_hacking,
                    "required": bool(thresholds["cem_significant_reward_hacking"]),
                    "passed": cem_hacking is bool(thresholds["cem_significant_reward_hacking"]),
                },
            }
            green = all(bool(value["passed"]) for value in tests.values())
            agreement_evidence["registered_gate_tests"] = tests
            agreement = diagnostic_result(
                "47",
                PASS if green else FAIL,
                summary=(
                    "T_u500-positive AMP reward agrees with every frozen validity criterion"
                    if green
                    else "T_u500-positive AMP reward fails at least one frozen validity criterion"
                ),
                evidence=agreement_evidence,
            )
            gate = diagnostic_result(
                "47",
                PASS if green else FAIL,
                summary=agreement["summary"],
                evidence={
                    "decision_variable_T": "+" if green else "-",
                    "decision_reason": agreement["summary"],
                    "primary_candidate": candidate,
                    "registered_gate_tests": tests,
                    "standard_AMP_K_control": offline.get("K", {}),
                    "fail_closed": True,
                    "A_mix": "legacy_quarantined",
                },
            )
    except DependencyUnavailable as exc:
        agreement = diagnostic_result(
            "47", SKIPPED_DEPENDENCY, summary="metric agreement evidence is unavailable", errors=[str(exc)]
        )
        gate = diagnostic_result(
            "47",
            SKIPPED_DEPENDENCY,
            summary="reward validity is UNKNOWN",
            evidence={"decision_variable_T": "UNKNOWN", "fail_closed": True},
            errors=[str(exc)],
        )
    except (FileNotFoundError, ProtocolError, ValueError, KeyError, TypeError, FloatingPointError, RuntimeError) as exc:
        agreement = diagnostic_result(
            "47", INVALID_PROTOCOL, summary="metric-agreement protocol failed closed", errors=[str(exc)]
        )
        gate = diagnostic_result(
            "47",
            INVALID_PROTOCOL,
            summary="reward-validity gate failed closed",
            evidence={"decision_variable_T": "UNKNOWN", "fail_closed": True},
            errors=[str(exc)],
        )
    write_json_exclusive(agreement_path, agreement)
    write_json_exclusive(gate_path, gate)
    print(f"[diag_47] {gate['status']} {gate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
