#!/usr/bin/env python3
"""Apply the preregistered Green/Yellow/Red reference-free policy-class gate."""

from __future__ import annotations

import argparse
import csv
import json
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
    ProtocolError,
    diagnostic_result,
    load_spec,
    read_json,
    write_json_exclusive,
)
from diagnostics.common.policy_class_probe import PolicyClassProtocol, output_dir_from_args


ARTIFACTS = {
    "21": "reference_sensitivity.json",
    "22": "aliasing.json",
    "23": "predictability.json",
    "24": "bc_probe.json",
    "25": "open_loop.json",
    "26": "reference_ablation.json",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def _thresholds(spec: dict[str, Any]) -> dict[str, Any]:
    try:
        value = spec["thresholds"]["reference_free_bc_green"]
    except KeyError as exc:
        raise ProtocolError("spec lacks thresholds.reference_free_bc_green") from exc
    required = {
        "motion_completion_relative_to_teacher_min",
        "failure_rate_increase_max",
        "action_nrmse_max",
        "minimum_passing_seeds",
        "total_seeds",
    }
    if not isinstance(value, dict) or not required <= set(value):
        raise ProtocolError(
            f"reference-free Green thresholds are incomplete: {sorted(required - set(value or {}))}"
        )
    return value


def _read_table(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value: Any, name: str) -> float:
    if value in (None, "", "None"):
        raise ProtocolError(f"BC table is missing numeric field {name}")
    result = float(value)
    if not (-float("inf") < result < float("inf")):
        raise ProtocolError(f"BC table field {name} is non-finite")
    return result


def _teacher_baseline(bc: dict[str, Any]) -> tuple[float, float]:
    closed = bc.get("evidence", {}).get("closed_loop", {})
    metrics = closed.get("metrics", {}) if isinstance(closed, dict) else {}
    teacher = metrics.get("teacher_baseline", {}) if isinstance(metrics, dict) else {}
    if not isinstance(teacher, dict):
        raise ProtocolError("BC closed-loop result lacks teacher_baseline")
    completion = teacher.get("motion_completion", teacher.get("motion_complete_frac"))
    failure = teacher.get("failure_rate", teacher.get("failure_frac"))
    return _float(completion, "teacher motion completion"), _float(failure, "teacher failure rate")


def _predictability_metrics(
    predictability: dict[str, Any],
    protocol: PolicyClassProtocol,
    minimum_seeds: int,
) -> dict[str, Any]:
    models = predictability["evidence"]["models"]
    h1 = models["no_reference_mlp_h1"]["seeds"]
    h32 = models["no_reference_gru_h32"]["seeds"]
    if len(h1) != len(protocol.seeds) or len(h32) != len(protocol.seeds):
        raise ProtocolError("predictability matrix does not contain every frozen seed")
    h1_nrmse = [float(item["heldout_trajectory"]["action"]["nrmse"]) for item in h1]
    h32_nrmse = [float(item["heldout_trajectory"]["action"]["nrmse"]) for item in h32]
    phase_relative = [
        float(item["heldout_trajectory"]["phase"]["relative_to_constant_baseline"])
        for item in h32
    ]
    phase_relative_max = 1.0 - protocol.phase_recovery_improvement
    return {
        "memoryless_action_nrmse_by_seed": h1_nrmse,
        "history32_action_nrmse_by_seed": h32_nrmse,
        "history32_phase_relative_to_constant_by_seed": phase_relative,
        "memoryless_action_predictable_at_0.20": sum(value <= 0.20 for value in h1_nrmse) >= minimum_seeds,
        "history32_action_predictable_at_0.20": sum(value <= 0.20 for value in h32_nrmse) >= minimum_seeds,
        "history32_phase_recoverable": sum(value <= phase_relative_max for value in phase_relative) >= minimum_seeds,
        "phase_relative_to_constant_max": phase_relative_max,
        "minimum_passing_seeds": minimum_seeds,
    }


def main() -> int:
    args = _arguments()
    repo_root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(repo_root, spec, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "policy_class_gate.json"
    gate = "UNKNOWN"
    try:
        protocol = PolicyClassProtocol.from_spec(spec)
        thresholds = _thresholds(spec)
        artifacts: dict[str, dict[str, Any]] = {}
        missing: list[str] = []
        for diagnostic_id, filename in ARTIFACTS.items():
            path = output_dir / filename
            if not path.is_file():
                missing.append(diagnostic_id)
                continue
            value = read_json(path)
            if not isinstance(value, dict) or value.get("diagnostic_id") != diagnostic_id:
                raise ProtocolError(f"artifact {filename} has the wrong diagnostic identity")
            artifacts[diagnostic_id] = value
        if missing:
            result = diagnostic_result(
                "27", SKIPPED_DEPENDENCY,
                summary="policy-class gate remains UNKNOWN because prerequisite artifacts are missing",
                evidence={
                    "gate": gate,
                    "decision_variable_R": "UNKNOWN",
                    "missing_diagnostics": missing,
                    "observed_statuses": {key: value.get("status") for key, value in artifacts.items()},
                },
                warnings=["missing assets are not classified as a Red policy-class result"],
            )
        else:
            statuses = {key: str(value.get("status")) for key, value in artifacts.items()}
            invalid = [key for key, value in statuses.items() if value == INVALID_PROTOCOL]
            skipped = [key for key, value in statuses.items() if value == SKIPPED_DEPENDENCY]
            failed = [key for key, value in statuses.items() if value == FAIL]
            if invalid:
                result = diagnostic_result(
                    "27", INVALID_PROTOCOL,
                    summary="policy-class gate is invalid because prerequisite protocols failed",
                    evidence={
                        "gate": gate,
                        "decision_variable_R": "UNKNOWN",
                        "invalid_diagnostics": invalid,
                        "observed_statuses": statuses,
                    },
                )
            elif skipped:
                result = diagnostic_result(
                    "27", SKIPPED_DEPENDENCY,
                    summary="policy-class gate remains UNKNOWN because real evidence is incomplete",
                    evidence={
                        "gate": gate,
                        "decision_variable_R": "UNKNOWN",
                        "skipped_diagnostics": skipped,
                        "observed_statuses": statuses,
                    },
                    warnings=["dependency gaps are never promoted to Red"],
                )
            elif failed:
                # In current stage 2 the only valid FAIL before gate derivation
                # is the full-observation sanity control in diag_23.  That is
                # not evidence that reference-free control is impossible.
                result = diagnostic_result(
                    "27", FAIL,
                    summary="policy-class gate remains UNKNOWN because a prerequisite sanity control failed",
                    evidence={
                        "gate": gate,
                        "decision_variable_R": "UNKNOWN",
                        "failed_diagnostics": failed,
                        "observed_statuses": statuses,
                    },
                    warnings=["sanity-control failure must not be relabelled as policy-class Red"],
                )
            elif set(statuses.values()) != {PASS}:
                raise ProtocolError(f"unknown prerequisite statuses: {statuses}")
            else:
                predictability_metrics = _predictability_metrics(
                    artifacts["23"], protocol, int(thresholds["minimum_passing_seeds"])
                )
                teacher_completion, teacher_failure = _teacher_baseline(artifacts["24"])
                rows = [
                    row for row in _read_table(output_dir / "tables" / "reference_free_bc.csv")
                    if row.get("model") == "GRU-H32"
                ]
                total_seeds = int(thresholds["total_seeds"])
                if len(rows) != total_seeds or len({row.get("seed") for row in rows}) != total_seeds:
                    raise ProtocolError(
                        f"GRU-H32 table must contain exactly {total_seeds} distinct seeds"
                    )
                seed_results: list[dict[str, Any]] = []
                for row in rows:
                    nrmse = _float(row.get("action_nrmse"), "action_nrmse")
                    completion = _float(row.get("motion_completion"), "motion_completion")
                    failure = _float(row.get("failure_rate"), "failure_rate")
                    criteria = {
                        "action_nrmse": nrmse <= float(thresholds["action_nrmse_max"]),
                        "motion_completion": completion >= (
                            float(thresholds["motion_completion_relative_to_teacher_min"])
                            * teacher_completion
                        ),
                        "failure_rate": failure <= (
                            teacher_failure + float(thresholds["failure_rate_increase_max"])
                        ),
                    }
                    seed_results.append(
                        {
                            "seed": int(row["seed"]),
                            "action_nrmse": nrmse,
                            "motion_completion": completion,
                            "failure_rate": failure,
                            "criteria": criteria,
                            "passes_green": all(criteria.values()),
                        }
                    )
                passing = sum(bool(item["passes_green"]) for item in seed_results)
                minimum = int(thresholds["minimum_passing_seeds"])
                pointwise_good = sum(
                    item["action_nrmse"] <= float(thresholds["action_nrmse_max"])
                    for item in seed_results
                ) >= minimum
                closure_bad = passing < minimum
                red_conjunction = (
                    not bool(predictability_metrics["history32_action_predictable_at_0.20"])
                    and not bool(predictability_metrics["history32_phase_recoverable"])
                    and closure_bad
                )
                if passing >= minimum:
                    gate = "GREEN"
                    decision_r = "+"
                    explanation = "at least two seeds satisfy pointwise and matched closed-loop criteria"
                elif pointwise_good and closure_bad:
                    gate = "YELLOW"
                    decision_r = "CONDITIONAL"
                    explanation = "pointwise BC is adequate but matched closed-loop rollouts drift"
                elif red_conjunction:
                    gate = "RED"
                    decision_r = "-"
                    explanation = "H32 cannot predict actions or recover phase and closed-loop BC fails"
                else:
                    gate = "YELLOW"
                    decision_r = "CONDITIONAL"
                    explanation = "complete evidence is mixed and does not satisfy either Green or strict Red"
                result = diagnostic_result(
                    "27", PASS,
                    summary=f"reference-free policy-class gate: {gate}",
                    evidence={
                        "gate": gate,
                        "decision_variable_R": decision_r,
                        "explanation": explanation,
                        "thresholds": thresholds,
                        "teacher_baseline": {
                            "motion_completion": teacher_completion,
                            "failure_rate": teacher_failure,
                        },
                        "predictability": predictability_metrics,
                        "gru_h32_seeds": seed_results,
                        "passing_seed_count": passing,
                        "observed_statuses": statuses,
                        "strict_red_conjunction": red_conjunction,
                    },
                )
    except (FileNotFoundError, ProtocolError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        result = diagnostic_result(
            "27", INVALID_PROTOCOL,
            summary="policy-class summary failed closed",
            evidence={"gate": gate, "decision_variable_R": "UNKNOWN"},
            errors=[str(exc)],
        )
    write_json_exclusive(target, result)
    print(f"[diag_27] {result['status']} gate={gate} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
