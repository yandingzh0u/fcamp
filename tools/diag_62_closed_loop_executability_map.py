#!/usr/bin/env python3
"""Policy-relative closed-loop executability screener from real PhysX outcomes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.family_screeners import (
    deterministic_group_split,
    executability_failure_cause,
    load_executability_dataset,
    same_state_binary_pair_accuracy,
    train_executability_classifier,
)
from diagnostics.common.manifest import (
    FAIL,
    INVALID_PROTOCOL,
    PASS,
    SKIPPED_DEPENDENCY,
    DependencyUnavailable,
    ProtocolError,
    canonical_sha256,
    diagnostic_result,
    load_spec,
    sha256_file,
    write_json_exclusive,
)
from diagnostics.common.policy_class_probe import output_dir_from_args
from diagnostics.common.reward_validity import reward_seed_agreement
from diagnostics.common.isaac_exit import finish_isaac_entrypoint

_ALLOW_HARD_EXIT = __name__ == "__main__"


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--outcome-bank", type=Path, default=None)
    parser.add_argument(
        "--execute-real",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect the frozen candidate grid in Isaac/PhysX when the bank is absent.",
    )
    return parser


def _protocol(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = spec.get("analysis_protocols", {}).get("family_screeners", {}).get(
        "closed_loop_executability"
    )
    threshold = spec.get("thresholds", {}).get("closed_loop_executability_viable")
    if not isinstance(raw, dict) or not isinstance(threshold, dict):
        raise ProtocolError("diag_62 protocol/thresholds are not frozen")
    exact = {
        "teacher_checkpoint_update": 500,
        "candidate_phase_offsets": [-64, -32, -16, 0, 16, 32, 64],
        "candidate_time_scales": [0.75, 1.0, 1.25],
        "segment_horizon_control_steps": 50,
        "snapshot_count": 64,
        "environment_randomness": "same_snapshot_shared_NoiseBank_stream_across_candidates",
        "label_source": "closed_loop_PhysX_outcomes_only",
        "model": "two_layer_MLP_binary_classifier",
        "hidden_dims": [256, 256],
        "seeds": [20260803, 20260804, 20260805],
        "calibration": "temperature_scaling_validation_only",
        "held_out_axes": ["phase", "segment", "snapshot"],
        "name_guard": "policy-relative_closed-loop_executability_not_physical_reachability",
    }
    changed = {key: {"expected": value, "actual": raw.get(key)} for key, value in exact.items() if raw.get(key) != value}
    if changed:
        raise ProtocolError(f"diag_62 frozen protocol changed: {changed}")
    if "reference" in str(raw.get("success_label", "")).lower() or "rmse" in str(raw.get("success_label", "")).lower():
        raise ProtocolError("diag_62 success label contains a reference surrogate")
    return dict(raw), dict(threshold)


def _event_incidence(dataset) -> dict[str, float]:
    return {
        name: float(np.mean(np.asarray(getattr(dataset, name), dtype=bool)))
        for name in (
            "success", "segment_complete", "failure", "tracking_loss",
            "joint_limit_event", "undesired_contact_event",
        )
    }


def _scientific_fail(
    *,
    protocol: dict[str, Any],
    thresholds: dict[str, Any],
    dataset,
    bank_path: Path,
    reason: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cause = executability_failure_cause(dataset)
    return diagnostic_result(
        "62", FAIL,
        summary=reason,
        evidence={
            "decision_variable_X": "-",
            "explanation": reason,
            "failure_cause": cause["failure_cause"],
            "failure_cause_evidence": cause,
            "protocol_sha256": canonical_sha256(protocol),
            "outcome_bank": str(bank_path),
            "outcome_bank_sha256": sha256_file(bank_path),
            "sample_count": int(len(dataset.success)),
            "event_incidence": _event_incidence(dataset),
            "thresholds": thresholds,
            "label_source": "closed_loop_PhysX_outcomes_only",
            "reference_RMSE_used_as_label": False,
            "real_physx_outcomes": True,
            "PPO_updates": 0,
            "A_mix": "legacy_quarantined",
            **(extra or {}),
        },
    )


def main() -> int:
    app_launcher = None
    simulation_app = None
    try:
        try:
            from isaaclab.app import AppLauncher
        except (ImportError, ModuleNotFoundError):
            AppLauncher = None
        parser = _base_parser()
        if AppLauncher is not None:
            AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args()
        root = args.repo_root.expanduser().resolve()
        spec = load_spec(args.spec)
        output_dir = output_dir_from_args(root, spec, args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "closed_loop_executability_map.json"
        bank_path = (args.outcome_bank or output_dir / "executability_physx_outcomes.npz").expanduser().resolve()
        try:
            protocol, thresholds = _protocol(spec)
            if args.execute_real and not bank_path.is_file():
                if AppLauncher is None:
                    raise DependencyUnavailable("Isaac Lab is unavailable for the real diag_62 collector")
                from engine.config import load_config

                cfg = load_config(
                    root / "configs" / "fixed_reward_largebox.yaml",
                    [f"environment.num_envs={int(protocol['snapshot_count'])}"],
                )
                args.headless = True
                args.device = cfg.environment.device
                app_launcher = AppLauncher(args)
                simulation_app = app_launcher.app
                from diagnostics.common.stage6_sim import run_closed_loop_executability_real

                completed = run_closed_loop_executability_real(
                    simulation_app=simulation_app,
                    repo_root=root,
                    output_dir=output_dir,
                    spec=spec,
                    result_path=bank_path,
                )
                if completed is not True:
                    raise ProtocolError("real diag_62 collector did not confirm completion")
            dataset = load_executability_dataset(bank_path)
            expected_segments = len(protocol["candidate_phase_offsets"]) * len(protocol["candidate_time_scales"])
            dataset.validate(
                expected_snapshots=int(protocol["snapshot_count"]),
                expected_segments=expected_segments,
            )
            positive_fraction = float(np.mean(dataset.success))
            negative_fraction = 1.0 - positive_fraction
            if (
                positive_fraction < float(thresholds["positive_fraction_min"])
                or negative_fraction < float(thresholds["negative_fraction_min"])
            ):
                result = _scientific_fail(
                    protocol=protocol,
                    thresholds=thresholds,
                    dataset=dataset,
                    bank_path=bank_path,
                    reason="real PhysX candidate grid has no preregistered non-trivial success/failure boundary",
                    extra={
                        "positive_fraction": positive_fraction,
                        "negative_fraction": negative_fraction,
                        "held_out_models": [],
                    },
                )
            else:
                axis_groups = {
                    "phase": np.asarray(dataset.phase_bins),
                    "segment": np.asarray(dataset.segment_ids),
                    "snapshot": np.asarray(dataset.snapshot_ids),
                }
                split_seed = int(protocol["seeds"][0])
                axis_records: dict[str, Any] = {}
                probability_by_axis: dict[str, list[np.ndarray]] = {}
                analysis_gap: str | None = None
                try:
                    for axis in protocol["held_out_axes"]:
                        split = deterministic_group_split(axis_groups[axis], seed=split_seed)
                        probabilities = []
                        seed_records = []
                        for seed in protocol["seeds"]:
                            probability, metrics = train_executability_classifier(
                                dataset.features,
                                dataset.success,
                                split,
                                hidden_dims=protocol["hidden_dims"],
                                batch_size=int(protocol["batch_size"]),
                                epochs_max=int(protocol["epochs_max"]),
                                patience=int(protocol["early_stopping_patience"]),
                                learning_rate=float(protocol["learning_rate"]),
                                seed=int(seed),
                            )
                            probabilities.append(probability)
                            seed_records.append(metrics)
                        probability_by_axis[axis] = probabilities
                        axis_records[axis] = {
                            "group_count": int(len(set(str(value) for value in axis_groups[axis]))),
                            "split_group_counts": {
                                name: int(len(set(str(value) for value in axis_groups[axis][split == name])))
                                for name in ("train", "validation", "test")
                            },
                            "seeds": seed_records,
                        }
                except DependencyUnavailable as exc:
                    analysis_gap = str(exc)
                if analysis_gap is not None:
                    result = _scientific_fail(
                        protocol=protocol,
                        thresholds=thresholds,
                        dataset=dataset,
                        bank_path=bank_path,
                        reason="real outcome boundary exists but cannot generalize across every preregistered held-out axis",
                        extra={
                            "positive_fraction": positive_fraction,
                            "negative_fraction": negative_fraction,
                            "held_out_models": axis_records,
                            "held_out_evaluation_gap": analysis_gap,
                        },
                    )
                else:
                    all_metrics = [
                        record
                        for axis in axis_records.values()
                        for record in axis["seeds"]
                    ]
                    minimum_auc = min(float(record["test_auroc"]) for record in all_metrics)
                    maximum_ece = max(float(record["test_ece"]) for record in all_metrics)
                    snapshot_split = deterministic_group_split(
                        dataset.snapshot_ids, seed=split_seed
                    )
                    snapshot_test = snapshot_split == "test"
                    snapshot_scores = probability_by_axis["snapshot"]
                    pair_by_seed = [
                        same_state_binary_pair_accuracy(
                            score,
                            dataset.success,
                            dataset.snapshot_ids,
                            eligible=snapshot_test,
                        )
                        for score in snapshot_scores
                    ]
                    ensemble = np.mean(np.stack(snapshot_scores), axis=0)
                    pair_ensemble = same_state_binary_pair_accuracy(
                        ensemble,
                        dataset.success,
                        dataset.snapshot_ids,
                        eligible=snapshot_test,
                    )
                    seed_agreement = reward_seed_agreement(
                        [score[snapshot_test] for score in snapshot_scores]
                    )
                    direction_consistent = bool(
                        all(float(record["accuracy"]) > 0.5 for record in pair_by_seed)
                        and float(seed_agreement["pairwise_spearman_min"]) >= 0.75
                    )
                    criteria = {
                        "positive_fraction": positive_fraction >= float(thresholds["positive_fraction_min"]),
                        "negative_fraction": negative_fraction >= float(thresholds["negative_fraction_min"]),
                        "held_out_auroc": minimum_auc >= float(thresholds["held_out_auroc_min"]),
                        "calibration": maximum_ece <= float(thresholds["ece_max"]),
                        "same_state_pairs": float(pair_ensemble["accuracy"]) >= float(thresholds["same_state_candidate_pair_accuracy_min"]),
                        "seed_direction_consistent": direction_consistent == bool(thresholds["seed_direction_consistent"]),
                    }
                    viable = all(criteria.values())
                    cause = executability_failure_cause(dataset)
                    result = diagnostic_result(
                        "62", PASS if viable else FAIL,
                        summary=(
                            "policy-relative closed-loop executability passed every preregistered criterion"
                            if viable else "policy-relative closed-loop executability failed at least one preregistered criterion"
                        ),
                        evidence={
                            "decision_variable_X": "+" if viable else "-",
                            "explanation": "X uses only event labels measured after same-snapshot real PhysX branches",
                            "failure_cause": cause["failure_cause"],
                            "failure_cause_evidence": cause,
                            "protocol_sha256": canonical_sha256(protocol),
                            "outcome_bank": str(bank_path),
                            "outcome_bank_sha256": sha256_file(bank_path),
                            "sample_count": int(len(dataset.success)),
                            "snapshot_count": int(len(set(str(value) for value in dataset.snapshot_ids))),
                            "candidate_segment_count": int(len(set(str(value) for value in dataset.segment_ids))),
                            "positive_fraction": positive_fraction,
                            "negative_fraction": negative_fraction,
                            "event_incidence": _event_incidence(dataset),
                            "held_out_models": axis_records,
                            "minimum_held_out_auroc": minimum_auc,
                            "maximum_held_out_ece": maximum_ece,
                            "same_state_candidate_pairs": {
                                "by_seed": pair_by_seed,
                                "ensemble": pair_ensemble,
                            },
                            "seed_direction": {
                                "consistent": direction_consistent,
                                "agreement": seed_agreement,
                                "operational_rule": "all seed pair accuracies >0.5 and pairwise Spearman min >=0.75",
                            },
                            "criteria": criteria,
                            "thresholds": thresholds,
                            "label_source": "closed_loop_PhysX_outcomes_only",
                            "reference_RMSE_used_as_label": False,
                            "real_physx_outcomes": True,
                            "claim_name": "policy-relative closed-loop executability",
                            "physical_reachability_claimed": False,
                            "PPO_updates": 0,
                            "A_mix": "legacy_quarantined",
                        },
                    )
        except DependencyUnavailable as exc:
            result = diagnostic_result(
                "62", SKIPPED_DEPENDENCY,
                summary="closed-loop executability lacks its own required real PhysX outcome bank",
                evidence={
                    "decision_variable_X": "UNKNOWN",
                    "executor_seam": "diagnostics.common.stage6_sim.run_closed_loop_executability_real",
                    "label_source": "closed_loop_PhysX_outcomes_only",
                    "reference_RMSE_used_as_label": False,
                    "PPO_updates": 0,
                    "A_mix": "legacy_quarantined",
                },
                errors=[str(exc)],
            )
        except (FileNotFoundError, ProtocolError, ValueError, KeyError, FloatingPointError, RuntimeError) as exc:
            result = diagnostic_result(
                "62", INVALID_PROTOCOL,
                summary="closed-loop executability protocol failed closed",
                evidence={
                    "decision_variable_X": "UNKNOWN",
                    "label_source": "closed_loop_PhysX_outcomes_only",
                    "reference_RMSE_used_as_label": False,
                    "PPO_updates": 0,
                    "A_mix": "legacy_quarantined",
                },
                errors=[str(exc)],
            )
        write_json_exclusive(target, result)
        print(f"[diag_62] {result['status']} {target}")
        return finish_isaac_entrypoint(
            0,
            isaac_launched=app_launcher is not None,
            allow_hard_exit=_ALLOW_HARD_EXIT,
        )
    finally:
        # An imported/unit-tested invocation returns normally.  The real CLI
        # has already hard-exited above after the durable status write.
        if simulation_app is not None and not _ALLOW_HARD_EXIT:
            try:
                simulation_app.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
