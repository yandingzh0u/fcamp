#!/usr/bin/env python3
"""Build the independent, uncombined trajectory outcome panel and Pareto pairs."""

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
from diagnostics.common.quality_panel import (
    build_quality_panel,
    grouped_strict_pareto_pairs,
    outcome_metrics_from_spec,
)
from diagnostics.common.reward_stage4 import (
    RewardValidityProtocol,
    write_parquet_exclusive,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--canonical-index", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(root, spec, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "quality_panel.status.json"
    panel_path = output_dir / "quality_panel.parquet"
    pairs_path = output_dir / "tables" / "quality_pareto_pairs.parquet"
    try:
        RewardValidityProtocol.from_spec(spec)
        metrics = outcome_metrics_from_spec(spec)
        collection_status = read_json(output_dir / "canonical_rollout_index.status.json")
        if collection_status.get("status") != PASS:
            raise DependencyUnavailable("diag_12 canonical collection is not PASS")
        index_path = (
            args.canonical_index or output_dir / "canonical_rollout_index.parquet"
        ).expanduser().resolve()
        expected_hash = (
            collection_status.get("evidence", {}).get("canonical_rollout_index_sha256")
        )
        if not index_path.is_file() or sha256_file(index_path) != expected_hash:
            raise ProtocolError("canonical rollout index is missing or changed after diag_12")
        panel = build_quality_panel(index_path)
        split_audit = read_json(output_dir / "split_audit.json")
        if split_audit.get("status") != PASS:
            raise DependencyUnavailable("diag_14 split audit is not PASS")
        sample_to_split = (
            split_audit.get("evidence", {})
            .get("inner_snapshot_split", {})
            .get("sample_to_split")
        )
        if not isinstance(sample_to_split, dict):
            raise ProtocolError("split audit lacks the frozen sample_to_split mapping")
        missing_splits = [row["sample_id"] for row in panel if row["sample_id"] not in sample_to_split]
        if missing_splits:
            raise ProtocolError(f"quality rows lack frozen splits: {missing_splits[:5]}")
        for row in panel:
            row["split"] = str(sample_to_split[row["sample_id"]])
        # Only same-snapshot comparisons enter the registered partial order.
        # This holds reset state/randomness fixed without combining outcomes.
        pairs = grouped_strict_pareto_pairs(
            panel, group_field="snapshot_id", metrics=metrics
        )
        pair_rows = [
            {
                "winner_sample_id": panel[winner]["sample_id"],
                "loser_sample_id": panel[loser]["sample_id"],
                "snapshot_id": panel[winner]["snapshot_id"],
                "winner_checkpoint_update": panel[winner]["checkpoint_update"],
                "loser_checkpoint_update": panel[loser]["checkpoint_update"],
                "split": panel[winner]["split"],
            }
            for winner, loser in pairs
        ]
        if any(
            panel[winner]["snapshot_id"] != panel[loser]["snapshot_id"]
            or panel[winner]["split"] != panel[loser]["split"]
            for winner, loser in pairs
        ):
            raise ProtocolError("quality Pareto pair crosses snapshot/split")
        mandatory = tuple(metric.name for metric in metrics)
        if any(
            not all(np.isfinite(float(row[name])) for name in mandatory)
            for row in panel
        ):
            raise ProtocolError("quality panel contains non-finite mandatory outcomes")
        write_parquet_exclusive(panel_path, panel)
        pair_artifact = None
        if pair_rows:
            write_parquet_exclusive(pairs_path, pair_rows)
            pair_artifact = str(pairs_path)
        result = diagnostic_result(
            "40",
            PASS,
            summary="canonical PhysX trajectories were reduced to an independent outcome panel",
            evidence={
                "quality_panel": str(panel_path),
                "quality_panel_sha256": sha256_file(panel_path),
                "row_count": len(panel),
                "strict_same_snapshot_pareto_pair_count": len(pair_rows),
                "pareto_pairs": pair_artifact,
                "collector_mode": "controlled_environment",
                "common_sigma": 0.0,
                "primary_outcomes": [
                    {
                        "name": metric.name,
                        "higher_is_better": metric.higher_is_better,
                        "tolerance": metric.tolerance,
                    }
                    for metric in metrics
                ],
                "metric_combination": "none; strict tolerant Pareto partial order only",
                "optional_unavailable_fields": [
                    "contact_mode_agreement (reference contact label absent)",
                    "torque_saturation (actuator effort bound absent)",
                ],
                "source_classifier_used_as_reward": False,
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "40",
            SKIPPED_DEPENDENCY,
            summary="independent quality panel awaits canonical PhysX trajectories",
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
        result = diagnostic_result(
            "40",
            INVALID_PROTOCOL,
            summary="independent quality-panel protocol failed closed",
            errors=[str(exc)],
        )
    write_json_exclusive(target, result)
    print(f"[diag_40] {result['status']} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
