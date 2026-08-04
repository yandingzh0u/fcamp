#!/usr/bin/env python3
"""Quantify reference-free state/action aliasing versus history length."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.manifest import (
    INVALID_PROTOCOL,
    PASS,
    DependencyUnavailable,
    ProtocolError,
    diagnostic_result,
    load_spec,
    write_json_exclusive,
)
from diagnostics.common.policy_class import knn_action_aliasing
from diagnostics.common.policy_class_probe import (
    CANONICAL_INDEX_NAME,
    PolicyClassProtocol,
    canonical_index,
    deterministic_group_splits,
    deterministic_subsample,
    history_windows,
    load_canonical_arrays,
    load_observation_partition,
    output_dir_from_args,
    phase_period,
    require_primary_teacher_quality,
    select_policy_class_rows,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--canonical-index", type=Path, default=None)
    return parser.parse_args()


def _adjacent_thresholds(actions: np.ndarray, trajectories: np.ndarray) -> dict[str, float]:
    same = trajectories[1:] == trajectories[:-1]
    if not np.any(same):
        raise DependencyUnavailable("no adjacent within-trajectory actions exist")
    distance = np.linalg.norm(actions[1:][same] - actions[:-1][same], axis=1) / np.sqrt(actions.shape[1])
    return {
        "adjacent_p50": float(np.quantile(distance, 0.50)),
        "adjacent_p90": float(np.quantile(distance, 0.90)),
        "adjacent_p95": float(np.quantile(distance, 0.95)),
    }


def main() -> int:
    args = _arguments()
    repo_root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(repo_root, spec, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "aliasing.json"
    try:
        protocol = PolicyClassProtocol.from_spec(spec)
        if protocol.max_knn_samples <= protocol.knn_k:
            raise ProtocolError("frozen maximum_knn_samples must exceed knn_k")
        load_observation_partition(output_dir / "observation_spec.json")
        index_path = (args.canonical_index or output_dir / CANONICAL_INDEX_NAME).expanduser().resolve()
        rows = select_policy_class_rows(
            canonical_index(index_path), checkpoint_update=protocol.primary_update
        )
        teacher_quality = require_primary_teacher_quality(
            output_dir, protocol, checkpoint_path=rows.iloc[0]["checkpoint_path"]
        )
        arrays = load_canonical_arrays(index_path, rows)
        splits = deterministic_group_splits(
            arrays.trajectory_ids,
            arrays.snapshot_ids,
            seed=protocol.split_seed,
            fractions=protocol.split_fractions,
        )
        period = phase_period(arrays.phases)
        train_mask = splits == "train"
        thresholds = _adjacent_thresholds(
            arrays.action_mean[train_mask], arrays.trajectory_ids[train_mask]
        )
        by_history: list[dict[str, object]] = []
        for history in protocol.alias_histories:
            windows, ends = history_windows(
                arrays.actor_no_reference, arrays.trajectory_ids, history
            )
            candidate = np.flatnonzero(splits[ends] == "test")
            if candidate.size <= protocol.knn_k:
                raise DependencyUnavailable(
                    f"history {history} has only {candidate.size} held-out samples"
                )
            selected = deterministic_subsample(
                candidate,
                int(protocol.max_knn_samples),
                seed=int(protocol.seeds[0]) + history,
            )
            features = windows[selected].reshape(selected.size, -1)
            actions = arrays.action_mean[ends[selected]]
            phases = arrays.phases[ends[selected]]
            contacts = arrays.contact_modes[ends[selected]]
            threshold_panels: dict[str, object] = {}
            for name, threshold in thresholds.items():
                threshold_panels[name] = knn_action_aliasing(
                    features,
                    actions,
                    phases,
                    contacts,
                    k=int(protocol.knn_k),
                    action_delta=max(float(threshold), 1.0e-8),
                    phase_period=period,
                )
            principal = threshold_panels["adjacent_p95"]
            by_history.append(
                {
                    "history_steps": history,
                    "feature_dimension": int(features.shape[1]),
                    "heldout_sample_count_before_cap": int(candidate.size),
                    "sample_count": int(selected.size),
                    "principal_threshold": "adjacent_p95",
                    "principal": principal,
                    "threshold_panels": threshold_panels,
                }
            )
        result = diagnostic_result(
            "22", PASS,
            summary="held-out no-reference kNN aliasing was measured for histories 1--32",
            evidence={
                "checkpoint_sha256": arrays.metadata["checkpoint_sha256"],
                "checkpoint_update": arrays.metadata["checkpoint_update"],
                "collector_mode": "clean_mean",
                "histories": list(protocol.alias_histories),
                "k": int(protocol.knn_k),
                "phase_period": period,
                "action_delta_thresholds": thresholds,
                "threshold_selection_guard": (
                    "the principal delta is the held-in trajectory adjacent-action p95, "
                    "not a threshold selected on aliasing results"
                ),
                "split": "complete snapshot/trajectory groups; metrics on test only",
                "by_history": by_history,
                "primary_teacher_quality": teacher_quality,
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "22", "SKIPPED_DEPENDENCY",
            summary="state aliasing awaits sufficient canonical trajectories",
            errors=[str(exc)],
        )
    except (FileNotFoundError, ProtocolError, ValueError, KeyError, json.JSONDecodeError) as exc:
        result = diagnostic_result(
            "22", INVALID_PROTOCOL,
            summary="state-aliasing protocol failed closed",
            errors=[str(exc)],
        )
    write_json_exclusive(target, result)
    print(f"[diag_22] {result['status']} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
