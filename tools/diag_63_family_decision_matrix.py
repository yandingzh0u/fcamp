#!/usr/bin/env python3
"""Derive R/T/E/L/F/X and apply the preregistered family decision matrix."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.family_decision import derive_variables, select_frozen_family
from diagnostics.common.manifest import (
    INVALID_PROTOCOL,
    PASS,
    ProtocolError,
    diagnostic_result,
    load_spec,
    read_json,
    write_json_exclusive,
)


ARTIFACTS = {
    "27": "policy_class_gate.json",
    "38": "domain_triangle_gate.json",
    "47": "reward_validity_gate.json",
    "58": "edge_gate.json",
    "60": "paired_common_latent_probe.json",
    "61": "teacher_energy_prior_probe.json",
    "62": "closed_loop_executability_map.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "family_decision_matrix.json"
    try:
        spec = load_spec(args.spec)
        artifacts = {}
        hashes = {}
        from diagnostics.common.manifest import sha256_file

        for diagnostic_id, filename in ARTIFACTS.items():
            path = output_dir / filename
            if not path.is_file():
                raise ProtocolError(f"decision input is absent: {filename}")
            payload = read_json(path)
            if str(payload.get("diagnostic_id")) != diagnostic_id:
                raise ProtocolError(f"decision input has wrong identity: {filename}")
            if payload.get("status") not in {"PASS", "FAIL", "SKIPPED_DEPENDENCY"}:
                raise ProtocolError(
                    f"decision input {diagnostic_id} is protocol-invalid or has an unknown status"
                )
            artifacts[diagnostic_id] = payload
            hashes[filename] = sha256_file(path)
        variables, auxiliaries = derive_variables(artifacts)
        decision = select_frozen_family(spec.get("decision_matrix"), variables, auxiliaries)
        result = diagnostic_result(
            "63",
            PASS,
            summary=(
                f"frozen decision result: {decision['selected_family']}"
                if decision["status"] == "SELECTED"
                else "no preregistered family is currently eligible"
            ),
            evidence={
                "decision_variables": {
                    name: value.to_dict() for name, value in variables.items()
                },
                "auxiliary_conditions": auxiliaries,
                "decision": decision,
                "decision_matrix": spec["decision_matrix"],
                "input_artifact_sha256": hashes,
                "claims_guard": (
                    "the selected family, if any, is only the frozen diagnostic decision; "
                    "this script proposes or implements no new method"
                ),
            },
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, ProtocolError) as exc:
        result = diagnostic_result(
            "63", INVALID_PROTOCOL, summary="family decision matrix failed closed", errors=[str(exc)]
        )
    write_json_exclusive(target, result)
    print(f"[diag_63] {result['status']} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
