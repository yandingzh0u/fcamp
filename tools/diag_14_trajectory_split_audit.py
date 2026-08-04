#!/usr/bin/env python3
"""Create the inner snapshot split and separately audit lineage holdout capacity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.manifest import (
    DependencyUnavailable,
    INVALID_PROTOCOL,
    PASS,
    SKIPPED_DEPENDENCY,
    ProtocolError,
    diagnostic_result,
    load_spec,
    read_json,
    write_json_exclusive,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def _outer_lineage_audit(
    lineage_ids: list[str],
    *,
    minimum: int,
    seed: int,
) -> dict[str, Any]:
    unique = sorted(set(lineage_ids))
    if len(unique) < minimum:
        return {
            "status": SKIPPED_DEPENDENCY,
            "summary": (
                f"outer checkpoint-lineage holdout needs {minimum} lineages; "
                f"only {len(unique)} is available"
            ),
            "lineage_count": len(unique),
            "lineage_ids": unique,
            "assignment": {},
        }
    ordered = sorted(
        unique,
        key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest(),
    )
    # Outer lineage is a separate generalization axis, not a second row split.
    return {
        "status": PASS,
        "summary": "independent checkpoint lineages are available for an outer holdout",
        "lineage_count": len(unique),
        "lineage_ids": unique,
        "assignment": {
            lineage: ("outer_test" if index == len(ordered) - 1 else "outer_train")
            for index, lineage in enumerate(ordered)
        },
    }


def main() -> int:
    args = _arguments()
    output_dir = args.output_dir.expanduser().resolve()
    result_path = output_dir / "split_audit.json"
    try:
        from diagnostics.common.canonical_collection import (
            CanonicalCollectionProtocol,
            load_rollout_index,
        )
        from diagnostics.common.noise_bank import CollectorMode
        from diagnostics.common.trajectory_split import (
            TrajectoryRecord,
            assign_trajectory_splits,
            audit_trajectory_splits,
        )

        spec = load_spec(args.spec)
        protocol = CanonicalCollectionProtocol.from_spec(spec)
        split_contract = spec.get("split_contract")
        if not isinstance(split_contract, dict):
            raise ProtocolError("spec.split_contract is missing")
        if split_contract.get("inner_split_atomic_units") != [
            "trajectory_id",
            "snapshot_id",
        ]:
            raise ProtocolError("inner split atomic units are not frozen correctly")
        if split_contract.get("outer_holdout_axis") != "checkpoint_lineage":
            raise ProtocolError("checkpoint lineage must be a separate outer holdout axis")
        fractions_raw = spec["collection"].get("trajectory_split_fractions")
        if not isinstance(fractions_raw, list) or len(fractions_raw) != 3:
            raise ProtocolError("trajectory_split_fractions must contain train/validation/test")
        fractions = {
            "train": float(fractions_raw[0]),
            "validation": float(fractions_raw[1]),
            "test": float(fractions_raw[2]),
        }
        seed = int(spec["collection"]["trajectory_split_seed"])
        collection_status_path = output_dir / "canonical_rollout_index.status.json"
        index_path = output_dir / "canonical_rollout_index.parquet"
        if not collection_status_path.is_file() or not index_path.is_file():
            raise DependencyUnavailable("diag_12 canonical collection is absent")
        if read_json(collection_status_path).get("status") != PASS:
            raise DependencyUnavailable("diag_12 did not PASS")
        rows = load_rollout_index(index_path)
        records = [
            TrajectoryRecord(
                sample_id=str(row["sample_id"]),
                trajectory_id=str(row["trajectory_id"]),
                snapshot_id=str(row["snapshot_id"]),
                checkpoint_lineage_id=str(row["checkpoint_lineage_id"]),
                checkpoint_id=str(row["checkpoint_id"]),
                branch_id=str(row["trajectory_id"]),
                collector_mode=str(row["collector_mode"]),
            )
            for row in rows
        ]
        assignment = assign_trajectory_splits(
            records,
            fractions=fractions,
            seed=seed,
            group_fields=("trajectory_id", "snapshot_id"),
        )
        audit = audit_trajectory_splits(
            records,
            assignment,
            group_fields=("trajectory_id", "snapshot_id"),
        )
        audit.require_valid()

        # The fully crossed checkpoint-by-snapshot design makes simultaneous
        # row-level lineage grouping impossible.  It is intentionally audited
        # as a separate outer axis instead of being smuggled into the inner
        # union-find grouping.
        outer = _outer_lineage_audit(
            [record.checkpoint_lineage_id for record in records],
            minimum=int(split_contract.get("outer_holdout_minimum_lineages", 2)),
            seed=seed,
        )
        native_samples = [
            record.sample_id
            for record in records
            if record.collector_mode == CollectorMode.NATIVE_STOCHASTIC.value
        ]
        native_eligible = [
            str(row["sample_id"])
            for row in rows
            if row["collector_mode"] == CollectorMode.NATIVE_STOCHASTIC.value
            and bool(row["eligible_for_primary_overlap"])
        ]
        if native_eligible:
            raise ProtocolError(
                "native stochastic samples entered primary overlap: "
                f"{native_eligible[:5]}"
            )
        result = diagnostic_result(
            "14",
            PASS,
            summary=(
                "inner trajectory/snapshot split is leakage-free; checkpoint-lineage "
                "generalization is reported as a separate outer audit"
            ),
            evidence={
                "inner_snapshot_split": {
                    "status": PASS,
                    "group_fields": ["trajectory_id", "snapshot_id"],
                    "seed": seed,
                    "fractions": fractions,
                    "split_counts": dict(audit.split_counts),
                    "leaks": dict(audit.leaks),
                    "sample_to_split": dict(assignment.sample_to_split),
                },
                "outer_lineage_split": outer,
                "crossed_design_resolution": split_contract.get(
                    "crossed_design_resolution"
                ),
                "native_stochastic_sample_count": len(native_samples),
                "native_stochastic_primary_overlap_count": 0,
                "sample_count": len(records),
                "snapshot_count": len({record.snapshot_id for record in records}),
                "checkpoint_count": len({record.checkpoint_id for record in records}),
            },
            warnings=(
                [outer["summary"]]
                if outer["status"] == SKIPPED_DEPENDENCY
                else []
            ),
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "14", SKIPPED_DEPENDENCY, summary=str(exc), errors=[str(exc)]
        )
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, ProtocolError, json.JSONDecodeError) as exc:
        result = diagnostic_result(
            "14",
            INVALID_PROTOCOL,
            summary="trajectory split protocol is invalid",
            errors=[str(exc)],
        )
    write_json_exclusive(result_path, result)
    print(f"[diag_14] {result['status']} {result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
