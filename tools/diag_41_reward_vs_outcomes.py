#!/usr/bin/env python3
"""Relate persisted standard-AMP rewards to independent trajectory outcomes."""

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
from diagnostics.common.quality_panel import grouped_strict_pareto_pairs
from diagnostics.common.reward_stage4 import (
    PRIMARY_REWARD_FAMILIES,
    RewardValidityProtocol,
    load_stage3_critics,
    per_outcome_spearman,
    read_parquet_rows,
    score_canonical_trajectories_streaming,
    write_parquet_exclusive,
)
from diagnostics.common.reward_validity import (
    failed_top_decile_fraction,
    reward_seed_agreement,
    strict_pairwise_accuracy,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--canonical-index", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(root, spec, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "score_outcome.json"
    scored_path = output_dir / "tables" / "score_outcome_rows.parquet"
    try:
        protocol = RewardValidityProtocol.from_spec(spec)
        quality_status = read_json(output_dir / "quality_panel.status.json")
        if quality_status.get("status") != PASS:
            raise DependencyUnavailable("diag_40 quality panel is not PASS")
        panel_path = output_dir / "quality_panel.parquet"
        if sha256_file(panel_path) != quality_status.get("evidence", {}).get(
            "quality_panel_sha256"
        ):
            raise ProtocolError("quality panel changed after diag_40")
        panel = read_parquet_rows(panel_path)
        device = "cpu"
        if args.device != "auto":
            device = args.device
        else:
            try:
                import torch

                device = "cuda:0" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        critics = load_stage3_critics(output_dir, spec, device=device)
        index_path = (
            args.canonical_index or output_dir / "canonical_rollout_index.parquet"
        ).expanduser().resolve()
        scores, unscored = score_canonical_trajectories_streaming(
            index_path, panel, critics, batch_trajectories=16
        )
        scored_indices = [index for index in range(len(panel)) if index not in set(unscored)]
        if len(scored_indices) < 10:
            raise DependencyUnavailable(
                "fewer than ten quality trajectories have complete alive AMP windows"
            )
        scored_rows = [dict(panel[index]) for index in scored_indices]
        for local, global_index in enumerate(scored_indices):
            for family in PRIMARY_REWARD_FAMILIES:
                for seed_index, seed in enumerate(critics.seeds):
                    scored_rows[local][f"reward_{family}_seed_{seed}"] = float(
                        scores[family][seed_index, global_index]
                    )
                scored_rows[local][f"reward_{family}_mean"] = float(
                    np.mean(scores[family][:, global_index])
                )
        test_positions = [
            index for index, row in enumerate(scored_rows) if str(row["split"]) == "test"
        ]
        if len(test_positions) < 10:
            raise DependencyUnavailable("held-out test quality bank has fewer than ten rows")
        test_rows = [scored_rows[index] for index in test_positions]
        pairs = grouped_strict_pareto_pairs(
            test_rows, group_field="snapshot_id", metrics=protocol.primary_metrics
        )
        if not pairs:
            raise DependencyUnavailable("held-out quality bank has no strict same-snapshot pairs")
        outcome_names = tuple(metric.name for metric in protocol.primary_metrics)
        secondary_numeric = (
            "anchor_error",
            "body_error",
            "joint_error",
            "action_jerk",
        )
        by_family = {}
        for family in PRIMARY_REWARD_FAMILIES:
            seed_banks = [
                np.asarray(
                    [float(row[f"reward_{family}_seed_{seed}"]) for row in test_rows],
                    dtype=np.float64,
                )
                for seed in critics.seeds
            ]
            ensemble = np.mean(np.stack(seed_banks), axis=0)
            failed = np.asarray([bool(float(row["failure"])) for row in test_rows])
            by_family[family] = {
                "critic_training": {
                    "source_negative": "A_amp",
                    "destination_positive": family,
                    "source_commit": critics.by_positive[family][0].source_commit,
                    "seeds": list(critics.seeds),
                },
                "strict_same_snapshot_pareto": {
                    "ensemble": strict_pairwise_accuracy(ensemble, pairs),
                    "by_seed": [
                        {"seed": seed, **strict_pairwise_accuracy(values, pairs)}
                        for seed, values in zip(critics.seeds, seed_banks)
                    ],
                },
                "seed_agreement": reward_seed_agreement(seed_banks),
                "failed_in_top_decile_fraction": failed_top_decile_fraction(
                    ensemble, failed
                ),
                "primary_outcome_spearman": per_outcome_spearman(
                    test_rows, ensemble, outcome_names
                ),
                "secondary_metric_spearman": per_outcome_spearman(
                    test_rows, ensemble, secondary_numeric
                ),
            }
        write_parquet_exclusive(scored_path, scored_rows)
        result = diagnostic_result(
            "41",
            PASS,
            summary="real Stage-3 AMP critics were evaluated against held-out independent outcomes",
            evidence={
                "reward_results": by_family,
                "scored_rows": str(scored_path),
                "scored_rows_sha256": sha256_file(scored_path),
                "panel_row_count": len(panel),
                "scored_row_count": len(scored_rows),
                "unscored_short_trajectory_count": len(unscored),
                "unscored_short_trajectory_sample_ids": [
                    panel[index]["sample_id"] for index in unscored
                ],
                "test_row_count": len(test_rows),
                "strict_test_pair_count": len(pairs),
                "offline_amp_artifact_index": str(critics.index_path),
                "offline_amp_artifact_index_sha256": critics.index_sha256,
                "feature_contract_sha256": critics.feature_contract_sha256,
                "source_classifier_used_as_reward": False,
                "A_mix": {
                    "status": "SKIPPED_DEPENDENCY",
                    "reason": "legacy FCAMP/A_mix was user-quarantined and is not loaded by Stage 4",
                },
                "metric_combination": "none; correlations are reported per outcome",
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "41",
            SKIPPED_DEPENDENCY,
            summary="reward/outcome analysis awaits valid quality and AMP-critic assets",
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
            "41",
            INVALID_PROTOCOL,
            summary="reward/outcome protocol failed closed",
            errors=[str(exc)],
        )
    write_json_exclusive(target, result)
    print(f"[diag_41] {result['status']} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
