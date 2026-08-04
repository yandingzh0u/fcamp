#!/usr/bin/env python3
"""Measure teacher action dependence on phase-shifted reference slices."""

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
from diagnostics.common.policy_class_probe import (
    CANONICAL_INDEX_NAME,
    FrozenCheckpointPolicy,
    PolicyClassProtocol,
    canonical_index,
    deterministic_subsample,
    load_canonical_arrays,
    load_observation_partition,
    named_term_indices,
    output_dir_from_args,
    resolve_checkpoint,
    require_primary_teacher_quality,
    select_policy_class_rows,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--canonical-index", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _paired_indices(trajectory_ids: np.ndarray, delta: int) -> tuple[np.ndarray, np.ndarray]:
    base: list[np.ndarray] = []
    shifted: list[np.ndarray] = []
    ids = np.asarray(trajectory_ids)
    for trajectory in dict.fromkeys(ids.tolist()):
        group = np.flatnonzero(ids == trajectory)
        if group.size <= abs(delta):
            continue
        if delta >= 0:
            base.append(group[: group.size - delta] if delta else group)
            shifted.append(group[delta:])
        else:
            amount = -delta
            base.append(group[amount:])
            shifted.append(group[: group.size - amount])
    if not base:
        raise DependencyUnavailable(f"no trajectory supports phase offset delta={delta}")
    return np.concatenate(base), np.concatenate(shifted)


def _distribution(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def main() -> int:
    args = _arguments()
    repo_root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(repo_root, spec, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "reference_sensitivity.json"
    try:
        index_path = (args.canonical_index or output_dir / CANONICAL_INDEX_NAME).expanduser().resolve()
        protocol = PolicyClassProtocol.from_spec(spec)
        frame = canonical_index(index_path)
        rows = select_policy_class_rows(frame, checkpoint_update=protocol.primary_update)
        teacher_quality = require_primary_teacher_quality(
            output_dir, protocol, checkpoint_path=rows.iloc[0]["checkpoint_path"]
        )
        arrays = load_canonical_arrays(index_path, rows)
        partition = load_observation_partition(output_dir / "observation_spec.json")
        reference_indices = named_term_indices(partition, role="reference")
        if reference_indices.size != arrays.actor_reference_terms.shape[1]:
            raise ProtocolError(
                "canonical actor_reference_terms width disagrees with AST-derived named slices"
            )
        first_row = rows.iloc[0].to_dict()
        checkpoint = resolve_checkpoint(
            repo_root, output_dir, first_row, explicit=args.checkpoint
        )
        policy = FrozenCheckpointPolicy(checkpoint, device=args.device)
        if policy.observation_dim != arrays.actor_full.shape[1]:
            raise ProtocolError("checkpoint actor dimension disagrees with canonical observations")

        # The exact delta=0 equality closes a critical loophole: the policy
        # invocation here must be the same actor/normalizer used by stage 1.
        baseline_prediction = policy.mean(arrays.actor_full)
        baseline_error = np.abs(baseline_prediction - arrays.action_mean)
        max_equivalence_error = float(baseline_error.max())
        if max_equivalence_error > 1.0e-5:
            raise ProtocolError(
                "reconstructed checkpoint policy is not equivalent to canonical action.mean; "
                f"max_abs_error={max_equivalence_error}"
            )

        action_dim = arrays.action_mean.shape[1]
        empirical_action_rms = float(np.sqrt(np.mean(np.square(arrays.action_mean))))
        policy_std_rms = float(np.sqrt(np.mean(np.square(policy.std))))
        adjacent_base, adjacent_shifted = _paired_indices(arrays.trajectory_ids, 1)
        adjacent_distance = np.linalg.norm(
            arrays.action_mean[adjacent_shifted] - arrays.action_mean[adjacent_base], axis=1
        ) / np.sqrt(action_dim)
        adjacent_change_rms = float(np.sqrt(np.mean(np.square(adjacent_distance))))
        denominators = {
            "empirical_action_rms": empirical_action_rms,
            "checkpoint_policy_std_rms": policy_std_rms,
            "normal_adjacent_action_change_rms": adjacent_change_rms,
        }
        delta_results: list[dict[str, object]] = []
        for delta in protocol.reference_offsets:
            base, shifted = _paired_indices(arrays.trajectory_ids, delta)
            selected_positions = deterministic_subsample(
                np.arange(base.size, dtype=np.int64),
                int(protocol.max_probe_samples),
                seed=int(protocol.seeds[0]) + delta + 16,
            )
            base = base[selected_positions]
            shifted = shifted[selected_positions]
            counterfactual = arrays.actor_full[base].copy()
            counterfactual[:, reference_indices] = arrays.actor_full[shifted][:, reference_indices]
            prediction = policy.mean(counterfactual)
            distance = np.linalg.norm(prediction - baseline_prediction[base], axis=1) / np.sqrt(action_dim)
            record: dict[str, object] = {
                "delta_control_steps": delta,
                "sample_count": int(base.size),
                "phase_delta_mean": float(np.mean(arrays.phases[shifted] - arrays.phases[base])),
                "sensitivity_raw": _distribution(distance),
                "per_joint_mean_abs_change": np.mean(
                    np.abs(prediction - baseline_prediction[base]), axis=0
                ).tolist(),
            }
            for name, denominator in denominators.items():
                record[f"sensitivity_over_{name}"] = (
                    _distribution(distance / denominator)
                    if denominator > 1.0e-12
                    else None
                )
            delta_results.append(record)
        result = diagnostic_result(
            "21", PASS,
            summary="teacher action sensitivity was measured under reference-slice counterfactuals",
            evidence={
                "protocol": "fixed actor proprio/state slices; replace only AST-named reference slices",
                "deltas_control_steps": list(protocol.reference_offsets),
                "checkpoint_path": str(checkpoint),
                "checkpoint_sha256": policy.checkpoint_sha256,
                "checkpoint_update": policy.update,
                "collector_mode": "clean_mean",
                "canonical_sample_count": int(arrays.actor_full.shape[0]),
                "reference_dimension": int(reference_indices.size),
                "action_dimension": int(action_dim),
                "action_path_max_abs_error": max_equivalence_error,
                "normalizers": denominators,
                "primary_teacher_quality": teacher_quality,
                "by_delta": delta_results,
                "causal_scope_guard": (
                    "slice substitution diagnoses checkpoint action dependence; it is not a "
                    "closed-loop reachability or motion-quality result"
                ),
            },
            warnings=[
                "composite reference-error slices are substituted as recorded; diag_26 is the real-PhysX closed-loop causal check"
            ],
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "21", "SKIPPED_DEPENDENCY",
            summary="counterfactual reference dependence awaits canonical assets",
            errors=[str(exc)],
        )
    except (FileNotFoundError, ProtocolError, ValueError, KeyError, json.JSONDecodeError) as exc:
        result = diagnostic_result(
            "21", INVALID_PROTOCOL,
            summary="counterfactual reference protocol failed closed",
            errors=[str(exc)],
        )
    write_json_exclusive(target, result)
    print(f"[diag_21] {result['status']} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
