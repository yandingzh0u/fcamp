#!/usr/bin/env python3
"""Measure AMP reward ordering on local, real-PhysX action candidates."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.manifest import (
    INVALID_PROTOCOL,
    PASS,
    SKIPPED_DEPENDENCY,
    DependencyUnavailable,
    ProtocolError,
    diagnostic_result,
    load_spec,
    read_json,
    sha256_file,
    write_json_exclusive,
)
from diagnostics.common.policy_class_probe import output_dir_from_args
from diagnostics.common.reward_stage4 import (
    PRIMARY_REWARD_FAMILIES,
    RewardValidityProtocol,
    branch_pair_rows,
    load_branch_bank,
    load_stage3_critics,
    read_parquet_rows,
    reward_accuracy_by_family,
    validate_branch_rows,
)


def _is_local(row: dict[str, object]) -> bool:
    if "local_action_candidate" in row:
        return bool(row["local_action_candidate"])
    category = str(row["branch_category"]).lower().replace("-", "_")
    return "interpol" in category or "perturb" in category


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
    target = output_dir / "local_rank.json"
    try:
        protocol = RewardValidityProtocol.from_spec(spec)
        branching_path = output_dir / "branching.json"
        if not branching_path.is_file():
            raise DependencyUnavailable("diag_42 branching status is absent")
        branching = read_json(branching_path)
        if branching.get("status") != PASS:
            raise DependencyUnavailable("diag_42 has no valid real PhysX branch bank")
        evidence = branching.get("evidence", {})
        bank_path = Path(str(evidence["branch_bank"])).expanduser().resolve()
        row_path = Path(str(evidence["branch_rows"])).expanduser().resolve()
        if sha256_file(bank_path) != evidence.get("branch_bank_sha256"):
            raise ProtocolError("branch bank changed after diag_42")
        if sha256_file(row_path) != evidence.get("branch_rows_sha256"):
            raise ProtocolError("branch quality table changed after diag_42")
        load_branch_bank(bank_path)
        rows = read_parquet_rows(row_path)
        critics = load_stage3_critics(output_dir, spec, device="cpu")
        validate_branch_rows(rows, protocol=protocol, seeds=critics.seeds)
        pairs = branch_pair_rows(rows, metrics=protocol.primary_metrics)
        local_ids = {str(row["branch_id"]) for row in rows if _is_local(row)}
        local_pairs = [
            pair
            for pair in pairs
            if str(pair["winner_branch_id"]) in local_ids
            or str(pair["loser_branch_id"]) in local_ids
        ]
        if not local_ids:
            raise DependencyUnavailable("real branch bank contains no interpolation/perturbation candidates")
        if not local_pairs:
            raise DependencyUnavailable("local candidates yield no strict Pareto comparison")
        accuracy = reward_accuracy_by_family(rows, local_pairs, seeds=critics.seeds)
        by_horizon: dict[str, dict[str, object]] = {}
        for horizon in protocol.branch_horizons:
            subset = [pair for pair in local_pairs if int(pair["horizon"]) == horizon]
            by_horizon[str(horizon)] = {
                "pair_count": len(subset) // len(PRIMARY_REWARD_FAMILIES),
                "ensemble_accuracy": {
                    family: (
                        float(np.mean([bool(pair["reward_correct"]) for pair in subset if pair["reward_family"] == family]))
                        if any(pair["reward_family"] == family for pair in subset)
                        else None
                    )
                    for family in PRIMARY_REWARD_FAMILIES
                },
            }
        result = diagnostic_result(
            "43",
            PASS,
            summary="persisted AMP critics were tested on real local action branches",
            evidence={
                "branch_bank": str(bank_path),
                "branch_bank_sha256": sha256_file(bank_path),
                "local_branch_ids": sorted(local_ids),
                "strict_local_pair_count": len(local_pairs) // len(PRIMARY_REWARD_FAMILIES),
                "reward_accuracy": accuracy,
                "by_horizon": by_horizon,
                "same_snapshot": True,
                "real_physx_rollouts": True,
                "source_classifier_used_as_reward": False,
                "A_mix": "legacy_quarantined",
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "43",
            SKIPPED_DEPENDENCY,
            summary="local action ranking awaits real same-state action branches",
            errors=[str(exc)],
        )
    except (FileNotFoundError, ProtocolError, ValueError, KeyError, FloatingPointError, RuntimeError) as exc:
        result = diagnostic_result(
            "43", INVALID_PROTOCOL, summary="local action-ranking protocol failed closed", errors=[str(exc)]
        )
    write_json_exclusive(target, result)
    print(f"[diag_43] {result['status']} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
