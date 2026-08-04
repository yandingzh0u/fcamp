#!/usr/bin/env python3
"""Validate and analyze real same-snapshot PhysX short-horizon branches."""

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
    write_csv_exclusive,
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


def _failed_top_decile(rows, family: str) -> dict[str, object]:
    values = np.asarray([float(row[f"reward_{family}_mean"]) for row in rows])
    cutoff = float(np.quantile(values, 0.9))
    selected = values >= cutoff
    failed = np.asarray([bool(float(row["failure"])) for row in rows])
    return {
        "cutoff": cutoff,
        "top_decile_count": int(selected.sum()),
        "failed_count": int(np.sum(selected & failed)),
        "clearly_failed_branch_in_top_decile": bool(np.any(selected & failed)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--branch-bank", type=Path, default=None)
    parser.add_argument("--branch-rows", type=Path, default=None)
    parser.set_defaults(execute_real=True)
    parser.add_argument("--execute-real", dest="execute_real", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-execute-real",
        dest="execute_real",
        action="store_false",
        help="Audit only existing assets; do not launch the real Isaac collector.",
    )
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(root, spec, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "branching.json"
    table_path = output_dir / "tables" / "branch_pairwise.csv"
    try:
        protocol = RewardValidityProtocol.from_spec(spec)
        bank_path = (args.branch_bank or output_dir / "branch_bank.pt").expanduser().resolve()
        row_path = (
            args.branch_rows or output_dir / "tables" / "branch_quality_panel.parquet"
        ).expanduser().resolve()
        if args.execute_real and (not bank_path.is_file() or not row_path.is_file()):
            try:
                from diagnostics.common.stage4_sim import run_same_state_branching_real
            except (ImportError, ModuleNotFoundError) as exc:
                raise DependencyUnavailable(
                    "real Stage-4 Isaac branch executor is not installed"
                ) from exc
            produced = run_same_state_branching_real(
                repo_root=root,
                output_dir=output_dir,
                spec=spec,
                bank_path=bank_path,
                row_path=row_path,
            )
            if produced is not True:
                raise ProtocolError("real branch executor did not confirm artifact completion")
        if not bank_path.is_file() or not row_path.is_file():
            raise DependencyUnavailable(
                "real same-snapshot PhysX branch_bank.pt/branch_quality_panel.parquet are absent; "
                "no offline or fabricated substitute is permitted"
            )
        bank = load_branch_bank(bank_path)
        rows = read_parquet_rows(row_path)
        metadata = bank["metadata"]
        if tuple(int(value) for value in metadata.get("checkpoint_updates", ())) != protocol.checkpoint_updates:
            raise ProtocolError("real branch bank lacks the exact frozen checkpoint updates")
        if tuple(float(value) for value in metadata.get("interpolation_alphas", ())) != protocol.interpolation_alphas:
            raise ProtocolError("real branch bank lacks the exact frozen interpolation alphas")
        if tuple(float(value) for value in metadata.get("perturbation_scales", ())) != protocol.perturbation_scales:
            raise ProtocolError("real branch bank lacks the exact frozen perturbation scales")
        expected_scale_basis = spec["analysis_protocols"]["reward_validity"].get(
            "action_perturbation_scale_basis"
        )
        if metadata.get("perturbation_scale_basis") != expected_scale_basis:
            raise ProtocolError("real branch perturbation scaling is not the frozen train-IQR basis")
        branches = tuple(str(value) for value in metadata["branch_ids"])
        snapshots = tuple(str(value) for value in metadata["snapshot_ids"])
        horizons = tuple(int(value) for value in metadata["horizons"])
        expected_count = len(branches) * len(snapshots) * len(horizons)
        if len(rows) != expected_count:
            raise ProtocolError(
                f"branch row product is incomplete: expected={expected_count}, actual={len(rows)}"
            )
        keys = {
            (str(row["branch_id"]), str(row["snapshot_id"]), int(row["horizon"]))
            for row in rows
        }
        expected_keys = {
            (branch, snapshot, horizon)
            for branch in branches
            for snapshot in snapshots
            for horizon in horizons
        }
        if keys != expected_keys or len(keys) != len(rows):
            raise ProtocolError("branch rows do not cover the exact branch x snapshot x horizon product")
        categories = {
            str(row["branch_category"])
            for row in rows
            if str(row["branch_category"]) != "legacy_quarantined"
        }
        if len(categories) < protocol.minimum_distinct_branch_sources:
            raise DependencyUnavailable(
                f"only {len(categories)} real branch source categories are available"
            )
        if any(str(row["branch_category"]) == "A_mix" for row in rows):
            raise ProtocolError("A_mix/FCAMP entered a formal Stage-4 branch")
        critics = load_stage3_critics(output_dir, spec, device="cpu")
        validate_branch_rows(rows, protocol=protocol, seeds=critics.seeds)
        pairs = branch_pair_rows(rows, metrics=protocol.primary_metrics)
        if not pairs:
            raise DependencyUnavailable("real branches produced no strict Pareto pairs")
        write_csv_exclusive(table_path, pairs)
        accuracy = reward_accuracy_by_family(rows, pairs, seeds=critics.seeds)
        by_horizon = {}
        for horizon in horizons:
            subset_rows = [row for row in rows if int(row["horizon"]) == horizon]
            subset_pairs = [row for row in pairs if int(row["horizon"]) == horizon]
            by_horizon[str(horizon)] = {
                "row_count": len(subset_rows),
                "strict_pair_count": len(subset_pairs) // len(PRIMARY_REWARD_FAMILIES),
                "accuracy": {
                    family: float(
                        np.mean(
                            [bool(row["reward_correct"]) for row in subset_pairs if row["reward_family"] == family]
                        )
                    )
                    for family in PRIMARY_REWARD_FAMILIES
                },
            }
        result = diagnostic_result(
            "42",
            PASS,
            summary="real condition-matched PhysX branches were scored by persisted AMP critics",
            evidence={
                "branch_bank": str(bank_path),
                "branch_bank_sha256": sha256_file(bank_path),
                "branch_rows": str(row_path),
                "branch_rows_sha256": sha256_file(row_path),
                "branch_pairwise": str(table_path),
                "branch_count": len(branches),
                "snapshot_count": len(snapshots),
                "horizons": list(horizons),
                "branch_source_categories": sorted(categories),
                "reward_accuracy": accuracy,
                "by_horizon": by_horizon,
                "failed_top_decile": {
                    family: _failed_top_decile(rows, family)
                    for family in PRIMARY_REWARD_FAMILIES
                },
                "same_snapshot_replay_verified": True,
                "real_physx_rollouts": True,
                "source_classifier_used_as_reward": False,
                "A_mix": {
                    "status": "SKIPPED_DEPENDENCY",
                    "reason": "legacy FCAMP/A_mix is quarantined and was never executed",
                },
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "42",
            SKIPPED_DEPENDENCY,
            summary="same-state branching awaits the real Isaac executor/assets",
            evidence={
                "executor_seam": "diagnostics.common.stage4_sim.run_same_state_branching_real",
                "required_bank_schema": "largebox_same_snapshot_branch_bank_v1",
                "A_mix": "legacy_quarantined",
            },
            errors=[str(exc)],
        )
    except (
        FileNotFoundError,
        ProtocolError,
        ValueError,
        KeyError,
        FloatingPointError,
        RuntimeError,
    ) as exc:
        replay_audit_path = output_dir / "stage4_replay_audit.json"
        result = diagnostic_result(
            "42",
            INVALID_PROTOCOL,
            summary="same-state branch protocol failed closed",
            evidence={
                "replay_audit": (
                    read_json(replay_audit_path) if replay_audit_path.is_file() else None
                )
            },
            errors=[str(exc)],
        )
    write_json_exclusive(target, result)
    print(f"[diag_42] {result['status']} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
