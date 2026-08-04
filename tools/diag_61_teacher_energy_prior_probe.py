#!/usr/bin/env python3
"""Offline five-seed G1-only frozen denoising-prior screener (no PPO)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.domain_data import load_domain_index
from diagnostics.common.family_screeners import (
    AMP_WINDOW_DIM,
    build_robot_teacher_archive,
    train_denoising_prior,
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
    read_json,
    sha256_file,
    write_json_exclusive,
)
from diagnostics.common.policy_class_probe import output_dir_from_args
from diagnostics.common.quality_panel import pareto_preference
from diagnostics.common.reward_stage4 import (
    CEM_BANK_SCHEMA,
    RewardValidityProtocol,
    classify_cem_hacking,
    load_branch_bank,
    read_parquet_rows,
)
from diagnostics.common.reward_validity import reward_seed_agreement, strict_pairwise_accuracy


def _protocol(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = spec.get("analysis_protocols", {}).get("family_screeners", {}).get(
        "frozen_robot_domain_prior"
    )
    threshold = spec.get("thresholds", {}).get("frozen_robot_domain_prior_viable")
    if not isinstance(raw, dict) or not isinstance(threshold, dict):
        raise ProtocolError("diag_61 protocol/thresholds are not frozen")
    exact = {
        "raw_human_reference_positive": False,
        "model": "denoising_MLP",
        "hidden_dims": [512, 256],
        "noise_scales_standardized": [0.01, 0.05, 0.10],
        "seeds": [20260803, 20260804, 20260805, 20260806, 20260807],
        "epochs": 20,
        "optimizer": "AdamW",
        "learning_rate": 0.0003,
        "score": "negative_multiscale_denoising_error",
        "normalizer": "successful_teacher_train_split_only_then_frozen",
    }
    changed = {key: {"expected": value, "actual": raw.get(key)} for key, value in exact.items() if raw.get(key) != value}
    if changed:
        raise ProtocolError(f"diag_61 frozen protocol changed: {changed}")
    if not str(raw.get("PPO", "")).startswith("forbidden"):
        raise ProtocolError("diag_61 no longer forbids PPO")
    return dict(raw), dict(threshold)


def _branch_windows_and_pairs(
    output_dir: Path,
    reward_protocol: RewardValidityProtocol,
) -> tuple[np.ndarray, list[dict[str, Any]], list[tuple[int, int]], dict[str, Any]]:
    status = read_json(output_dir / "branching.json")
    if status.get("status") != PASS:
        raise DependencyUnavailable("diag_61 requires the real same-state PhysX branch bank")
    evidence = status.get("evidence", {})
    bank_path = Path(str(evidence.get("branch_bank", ""))).expanduser().resolve()
    rows_path = Path(str(evidence.get("branch_rows", ""))).expanduser().resolve()
    if not bank_path.is_file() or sha256_file(bank_path) != str(evidence.get("branch_bank_sha256", "")):
        raise ProtocolError("diag_42 branch bank is missing or changed")
    if not rows_path.is_file() or sha256_file(rows_path) != str(evidence.get("branch_rows_sha256", "")):
        raise ProtocolError("diag_42 branch rows are missing or changed")
    bank = load_branch_bank(bank_path)
    rows = read_parquet_rows(rows_path)
    metadata = bank["metadata"]
    branch_lookup = {str(value): index for index, value in enumerate(metadata["branch_ids"])}
    horizon_lookup = {int(value): index for index, value in enumerate(metadata["horizons"])}
    snapshot_lookup = {str(value): index for index, value in enumerate(metadata["snapshot_ids"])}
    endpoint = bank["endpoint_windows"].detach().cpu().numpy()
    windows: list[np.ndarray] = []
    for row in rows:
        if "A_mix" in str(row) or "FCAMP" in str(row):
            raise ProtocolError("retired A_mix/FCAMP entered diag_61 branches")
        for metric in reward_protocol.primary_metrics:
            if metric.name not in row or not np.isfinite(float(row[metric.name])):
                raise ProtocolError(f"branch row lacks real finite outcome {metric.name}")
        key = (
            branch_lookup[str(row["branch_id"])],
            horizon_lookup[int(row["horizon"])],
            snapshot_lookup[str(row["snapshot_id"])],
        )
        windows.append(np.asarray(endpoint[key], dtype=np.float32))
    values = np.stack(windows)
    if values.shape != (len(rows), AMP_WINDOW_DIM) or not np.isfinite(values).all():
        raise ProtocolError("same-state endpoint windows are not aligned [N,2390]")
    groups: dict[tuple[str, int], list[int]] = {}
    for index, row in enumerate(rows):
        groups.setdefault((str(row["snapshot_id"]), int(row["horizon"])), []).append(index)
    pairs: list[tuple[int, int]] = []
    for indices in groups.values():
        for left_position, left in enumerate(indices):
            for right in indices[left_position + 1 :]:
                preference = pareto_preference(rows[left], rows[right], metrics=reward_protocol.primary_metrics)
                if preference > 0:
                    pairs.append((left, right))
                elif preference < 0:
                    pairs.append((right, left))
    if not pairs:
        raise DependencyUnavailable("real same-state branches contain no strict outcome pairs")
    return values, rows, pairs, {
        "path": str(bank_path),
        "sha256": sha256_file(bank_path),
        "row_path": str(rows_path),
        "row_sha256": sha256_file(rows_path),
    }


def _tensor_windows(value: Any, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    if array.ndim == 3 and array.shape[1:] == (10, 239):
        array = array.reshape(array.shape[0], -1)
    if array.ndim != 2 or array.shape[1] != AMP_WINDOW_DIM or not np.isfinite(array).all():
        raise ProtocolError(f"CEM {name} must contain finite [N,2390] real endpoint windows")
    return np.asarray(array, dtype=np.float32)


def _cem_windows_and_records(
    output_dir: Path,
    reward_protocol: RewardValidityProtocol,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    status = read_json(output_dir / "exploitability.json")
    if status.get("status") != PASS:
        raise DependencyUnavailable("diag_61 requires the existing real PhysX CEM bank")
    evidence = status.get("evidence", {})
    bank_path = Path(str(evidence.get("cem_bank", ""))).expanduser().resolve()
    if not bank_path.is_file() or sha256_file(bank_path) != str(evidence.get("cem_bank_sha256", "")):
        raise ProtocolError("CEM bank is missing or changed after diag_44")
    payload = read_json(bank_path)
    if payload.get("schema") != CEM_BANK_SCHEMA:
        raise ProtocolError("CEM bank schema differs from Stage-4")
    for guarantee in ("real_physx_rollouts", "same_snapshot_replay_verified", "audit_critic_independent"):
        if payload.get(guarantee) is not True:
            raise ProtocolError(f"CEM bank lacks {guarantee}")
    if payload.get("A_mix") not in (None, "legacy_quarantined"):
        raise ProtocolError("retired A_mix/FCAMP entered the CEM bank")
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ProtocolError("CEM bank has no outcome records")

    baseline: Any = None
    optimized: Any = None
    window_bank_value = payload.get("endpoint_window_bank")
    if window_bank_value:
        window_path = Path(str(window_bank_value)).expanduser()
        if not window_path.is_absolute():
            window_path = bank_path.parent / window_path
        if not window_path.is_file():
            raise DependencyUnavailable("CEM endpoint-window bank referenced by diag_44 is absent")
        try:
            window_payload = torch.load(window_path, map_location="cpu", weights_only=False)
        except TypeError:
            window_payload = torch.load(window_path, map_location="cpu")
        if not isinstance(window_payload, Mapping):
            raise ProtocolError("CEM endpoint-window bank is not a mapping")
        baseline = window_payload.get("baseline_windows")
        optimized = window_payload.get("optimized_windows")
    elif all("baseline_endpoint_window" in row and "optimized_endpoint_window" in row for row in records):
        baseline = [row["baseline_endpoint_window"] for row in records]
        optimized = [row["optimized_endpoint_window"] for row in records]
    else:
        raise DependencyUnavailable(
            "existing CEM bank has real outcomes but no exact before/after 10x239 endpoint windows; "
            "the frozen prior cannot be audited on it without fabricating inputs"
        )
    before = _tensor_windows(baseline, "baseline_windows")
    after = _tensor_windows(optimized, "optimized_windows")
    if before.shape != after.shape or before.shape[0] != len(records):
        raise ProtocolError("CEM before/after windows do not align with outcome records")
    clean_records: list[dict[str, Any]] = []
    for record in records:
        baseline_outcome = record.get("baseline_outcomes")
        optimized_outcome = record.get("optimized_outcomes")
        if not isinstance(baseline_outcome, Mapping) or not isinstance(optimized_outcome, Mapping):
            raise ProtocolError("CEM record lacks independent before/after outcomes")
        for metric in reward_protocol.primary_metrics:
            if any(metric.name not in panel or not np.isfinite(float(panel[metric.name])) for panel in (baseline_outcome, optimized_outcome)):
                raise ProtocolError(f"CEM record lacks finite outcome {metric.name}")
        clean_records.append(dict(record))
    return before, after, clean_records, {"path": str(bank_path), "sha256": sha256_file(bank_path)}


def _teacher_maximum_test(
    ensemble_scores: np.ndarray,
    updates: np.ndarray,
    successful: np.ndarray,
) -> dict[str, Any]:
    teacher = ensemble_scores[successful]
    if teacher.size < 2:
        raise DependencyUnavailable("held-out successful teacher distribution has fewer than two windows")
    landscape = {}
    for update in sorted(set(np.asarray(updates, dtype=np.int64).tolist())):
        values = ensemble_scores[np.asarray(updates) == update]
        landscape[str(int(update))] = {
            "count": int(values.size),
            "mean": float(values.mean()),
            "standard_error": float(values.std(ddof=1) / np.sqrt(max(values.size, 1))) if values.size > 1 else 0.0,
        }
    teacher_mean = float(teacher.mean())
    teacher_se = float(teacher.std(ddof=1) / np.sqrt(teacher.size))
    maximum_update = max(landscape, key=lambda key: landscape[key]["mean"])
    competitor = landscape[maximum_update]
    difference = teacher_mean - float(competitor["mean"])
    difference_se = math.sqrt(teacher_se ** 2 + float(competitor["standard_error"]) ** 2)
    tied_or_maximum = bool(difference >= 0.0 or difference + 1.96 * difference_se >= 0.0)
    return {
        "checkpoint_landscape": landscape,
        "successful_teacher_distribution": {
            "count": int(teacher.size), "mean": teacher_mean, "standard_error": teacher_se,
        },
        "maximum_checkpoint_update": int(maximum_update),
        "teacher_minus_maximum_mean": difference,
        "teacher_or_maximum_95pct_normal_bound": difference + 1.96 * difference_se,
        "teacher_distribution_is_unique_or_statistically_tied_maximum": tied_or_maximum,
    }


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
    target = output_dir / "teacher_energy_prior_probe.json"
    try:
        protocol, thresholds = _protocol(spec)
        reward_protocol = RewardValidityProtocol.from_spec(spec)
        index_path = output_dir / "canonical_rollout_index.parquet"
        split_path = output_dir / "split_audit.json"
        if not index_path.is_file() or not split_path.is_file():
            raise DependencyUnavailable("canonical rollout index/split audit is absent")
        archive = build_robot_teacher_archive(index_path, split_path)
        branch_windows, branch_rows, branch_pairs, branch_provenance = _branch_windows_and_pairs(
            output_dir, reward_protocol
        )
        cem_before, cem_after, cem_records, cem_provenance = _cem_windows_and_records(
            output_dir, reward_protocol
        )
        _, bundles = load_domain_index(output_dir / "domain_triangle_index.json")
        early = bundles["T_u200"]
        if early.name != "T_u200" or "A_mix" in str(early.metadata):
            raise ProtocolError("early-agent gradient bank is not current-lineage T_u200")

        successful_train = archive.successful & (np.asarray(archive.split) == "train")
        successful_validation = archive.successful & (np.asarray(archive.split) == "validation")
        heldout = np.asarray(archive.split) == "test"
        if successful_train.sum() < 32 or successful_validation.sum() < 16 or heldout.sum() < 16:
            raise DependencyUnavailable("successful teacher archive lacks frozen train/validation/test windows")

        training_records = []
        landscape_seed_scores = []
        branch_seed_scores = []
        cem_gain_by_seed = []
        gradient_by_seed = []
        for seed in protocol["seeds"]:
            prior, training = train_denoising_prior(
                archive.features[successful_train],
                archive.features[successful_validation],
                hidden_dims=protocol["hidden_dims"],
                noise_scales=protocol["noise_scales_standardized"],
                epochs=int(protocol["epochs"]),
                batch_size=int(protocol["batch_size"]),
                learning_rate=float(protocol["learning_rate"]),
                seed=int(seed),
            )
            training_records.append(training)
            landscape_seed_scores.append(prior.score(archive.features[heldout]))
            branch_seed_scores.append(prior.score(branch_windows))
            cem_gain_by_seed.append(prior.score(cem_after) - prior.score(cem_before))
            early_train = np.asarray(early.split) == "test"
            gradient_by_seed.append({
                "seed": int(seed),
                "mean_input_gradient_norm_T_u200": prior.input_gradient_norm(early.features[early_train]),
            })
            del prior

        landscape_scores = np.mean(np.stack(landscape_seed_scores), axis=0)
        maximum = _teacher_maximum_test(
            landscape_scores,
            np.asarray(archive.checkpoint_update)[heldout],
            np.asarray(archive.successful)[heldout],
        )
        branch_ensemble = np.mean(np.stack(branch_seed_scores), axis=0)
        pair_accuracy = strict_pairwise_accuracy(branch_ensemble, branch_pairs)
        seed_agreement = reward_seed_agreement(branch_seed_scores)
        cem_ensemble_gain = np.mean(np.stack(cem_gain_by_seed), axis=0)
        exploit_records = []
        for index, record in enumerate(cem_records):
            baseline = record["baseline_outcomes"]
            optimized = record["optimized_outcomes"]
            exploit_records.append({
                "snapshot_id": str(record["snapshot_id"]),
                "seed": int(record["seed"]),
                "search_reward_gain": float(cem_ensemble_gain[index]),
                "strict_pareto_regression": bool(
                    pareto_preference(baseline, optimized, metrics=reward_protocol.primary_metrics) > 0
                ),
                "clear_failure": bool(float(optimized["failure"])) and not bool(float(baseline["failure"])),
            })
        hacking = classify_cem_hacking(
            exploit_records,
            alpha=float(reward_protocol.cem["significance_alpha"]),
            bootstrap_replicates=int(reward_protocol.cem["paired_bootstrap_replicates"]),
            seed=int(protocol["seeds"][0]),
        )
        criteria = {
            "teacher_maximum": bool(maximum["teacher_distribution_is_unique_or_statistically_tied_maximum"]),
            "same_state_strict_pairs": float(pair_accuracy["accuracy"]) >= float(thresholds["same_state_strict_pair_accuracy_min"]),
            "seed_icc": float(seed_agreement["icc_consistency"]) >= float(thresholds["seed_icc_min"]),
            "no_cem_exploit": bool(hacking["significant_reward_hacking"]) == bool(thresholds["cem_significant_reward_exploit"]),
        }
        viable = all(criteria.values())
        result = diagnostic_result(
            "61", PASS if viable else FAIL,
            summary=(
                "G1-only frozen denoising prior passed every preregistered offline criterion"
                if viable else "G1-only frozen denoising prior failed at least one preregistered offline criterion"
            ),
            evidence={
                "decision_variable_F": "+" if viable else "-",
                "explanation": "F is based only on held-out robot-domain executions and real PhysX branch/CEM outcomes",
                "protocol_sha256": canonical_sha256(protocol),
                "training": training_records,
                "successful_teacher_updates": list(archive.selected_successful_updates),
                "clean_completion_by_update": {str(key): value for key, value in archive.clean_completion_by_update.items()},
                "archive_window_counts": {
                    "all_landscape": int(len(archive.features)),
                    "successful_train": int(successful_train.sum()),
                    "successful_validation": int(successful_validation.sum()),
                    "heldout_landscape": int(heldout.sum()),
                },
                "checkpoint_reward_landscape": maximum,
                "same_state_strict_pairs": {**pair_accuracy, "real_physx": True},
                "seed_agreement": seed_agreement,
                "existing_cem_candidate_audit": {
                    "test": hacking,
                    "candidate_bank_was_not_optimized_against_this_prior": True,
                    "interpretation": "conservative reuse check only; absence of exploit here is not a proof of global robustness",
                },
                "early_agent_gradient": gradient_by_seed,
                "criteria": criteria,
                "thresholds": thresholds,
                "provenance": {
                    "canonical_index": str(index_path),
                    "canonical_index_sha256": sha256_file(index_path),
                    "split_audit_sha256": sha256_file(split_path),
                    "branch": branch_provenance,
                    "cem": cem_provenance,
                },
                "raw_human_reference_positive": False,
                "PPO_used": False,
                "A_mix": "legacy_quarantined",
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "61", SKIPPED_DEPENDENCY,
            summary="frozen robot-domain prior lacks one of its own required real evidence banks",
            evidence={
                "decision_variable_F": "UNKNOWN",
                "raw_human_reference_positive": False,
                "PPO_used": False,
                "A_mix": "legacy_quarantined",
            },
            errors=[str(exc)],
        )
    except (FileNotFoundError, ProtocolError, ValueError, KeyError, FloatingPointError, RuntimeError) as exc:
        result = diagnostic_result(
            "61", INVALID_PROTOCOL,
            summary="frozen robot-domain prior protocol failed closed",
            evidence={
                "decision_variable_F": "UNKNOWN",
                "raw_human_reference_positive": False,
                "PPO_used": False,
                "A_mix": "legacy_quarantined",
            },
            errors=[str(exc)],
        )
    write_json_exclusive(target, result)
    print(f"[diag_61] {result['status']} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
