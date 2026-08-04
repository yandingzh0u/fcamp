#!/usr/bin/env python3
"""Fail-closed Stage-3 gate summarizing gap source and comparable domain edges."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.manifest import (
    FAIL, INVALID_PROTOCOL, PASS, SKIPPED_DEPENDENCY, DependencyUnavailable,
    ProtocolError, diagnostic_result, load_spec, read_json, write_json_exclusive,
)
from diagnostics.common.policy_class_probe import output_dir_from_args


def _require_pass(path: Path, diagnostic_id: str):
    if not path.is_file():
        raise DependencyUnavailable(f"diag_{diagnostic_id} artifact is missing: {path}")
    payload = read_json(path)
    status = payload.get("status")
    if status in (INVALID_PROTOCOL, SKIPPED_DEPENDENCY):
        raise DependencyUnavailable(f"diag_{diagnostic_id} is {status}")
    if status not in (PASS, FAIL):
        raise ProtocolError(f"diag_{diagnostic_id} has unknown status {status!r}")
    return payload


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
    target = output_dir / "domain_triangle_gate.json"
    try:
        source = _require_pass(output_dir / "pairwise_source_matrix.status.json", "31")
        gap = _require_pass(output_dir / "gap_decomposition.json", "32")
        causal = _require_pass(output_dir / "causal_sources.json", "33")
        overlap = _require_pass(output_dir / "effective_overlap_matrix.status.json", "34")
        reliability = _require_pass(output_dir / "ratio_reliability.json", "35")
        substitution = _require_pass(output_dir / "positive_substitution.json", "36")
        conditional = _require_pass(output_dir / "conditional_overlap_matrices.json", "37")
        if spec.get("thresholds", {}).get("empirical_effective_overlap_edge", {}).get("ess_gate_uses") != "unclipped_density_ratio":
            raise ProtocolError("Stage-3 gate is not frozen to unclipped ESS")
        pairs = reliability.get("evidence", {}).get("pairs")
        if not isinstance(pairs, list):
            raise ProtocolError("diag_35 pair gate records are missing")
        comparable = [
            {
                "source_negative": record["source_negative"],
                "destination_positive": record["destination_positive"],
            }
            for record in pairs if record.get("empirically_comparable") is True
        ]
        gaps = gap.get("evidence", {}).get("gaps", {})
        incremental = {
            name: float(gaps[name]["rms"])
            for name in (
                "motion_to_reset_readback",
                "reset_readback_to_teacher_one_step",
                "teacher_one_step_to_closed_loop",
            )
            if name in gaps
        }
        if len(incremental) != 3:
            raise ProtocolError("diag_32 did not report all three incremental gaps")
        dominant = max(incremental, key=incremental.get)
        scientific_pass = bool(comparable)
        result = diagnostic_result(
            "38", PASS if scientific_pass else FAIL,
            summary=(
                "Stage-3 found preregistered empirically comparable domain edges"
                if scientific_pass else "no domain edge passed every preregistered comparability criterion"
            ),
            evidence={
                "gate_pass": scientific_pass,
                "empirically_comparable_edges": comparable,
                "dominant_incremental_gap": dominant,
                "incremental_gap_rms": incremental,
                "positive_substitution_status": substitution["status"],
                "conditional_eligible_cells": conditional.get("evidence", {}).get("eligible_cell_count"),
                "source_matrix_status": source["status"],
                "causal_decomposition_status": causal["status"],
                "overlap_status": overlap["status"],
                "claims_guard": (
                    "comparability is an empirical edge-selection diagnostic only; "
                    "it is not mathematical support, reachability, or proof that PPO can cross the edge"
                ),
                "decision_variable_T": "UNKNOWN_until_independent_reward_validity",
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "38", SKIPPED_DEPENDENCY,
            summary="domain-triangle summary awaits every valid Stage-3 artifact",
            errors=[str(exc)],
        )
    except (FileNotFoundError, ProtocolError, ValueError, KeyError) as exc:
        result = diagnostic_result("38", INVALID_PROTOCOL, summary="domain-triangle gate failed closed", errors=[str(exc)])
    write_json_exclusive(target, result)
    print(f"[diag_38] {result['status']} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
