#!/usr/bin/env python3
"""Offline K<->T_u500 exact-pair common-coordinate screener (no PPO)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.domain_data import load_domain_index
from diagnostics.common.family_screeners import (
    FrozenStandardizer,
    binary_auc,
    cross_domain_retrieval,
    exact_paired_amp_windows,
    latent_distance_stability,
    representation_preservation,
    train_paired_contrastive,
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


def _protocol(spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = spec.get("analysis_protocols", {}).get("family_screeners", {}).get(
        "paired_common_coordinate"
    )
    threshold = spec.get("thresholds", {}).get("paired_common_coordinate_viable")
    if not isinstance(raw, dict) or not isinstance(threshold, dict):
        raise ProtocolError("diag_60 protocol/thresholds are not frozen")
    exact = {
        "paired_domains": ["K", "T_u500"],
        "pair_key": ["snapshot_id", "phase", "window_endpoint"],
        "input": "exact_chronological_10x239_raw_amp_windows",
        "linear_models": ["CCA", "PLS"],
        "linear_latent_dim": 32,
        "contrastive_encoder_hidden_dims": [256, 256],
        "contrastive_latent_dim": 32,
        "contrastive_temperature": 0.07,
        "seeds": [20260803, 20260804, 20260805],
        "model_selection": "validation_cross_domain_retrieval_test_untouched",
        "phase_bins": 16,
        "raw_baseline": "standardized_unprojected_same_physical_window",
    }
    changed = {key: {"expected": value, "actual": raw.get(key)} for key, value in exact.items() if raw.get(key) != value}
    if changed:
        raise ProtocolError(f"diag_60 frozen protocol changed: {changed}")
    if "PPO_reward" not in raw.get("forbidden_uses", []):
        raise ProtocolError("diag_60 no longer explicitly forbids PPO reward use")
    return dict(raw), dict(threshold)


def _split(values: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    return {name: np.asarray(values)[labels == name] for name in ("train", "validation", "test")}


def _source_auc(
    reference: dict[str, np.ndarray], execution: dict[str, np.ndarray]
) -> float:
    train_x = np.concatenate((reference["train"], execution["train"]), axis=0)
    train_y = np.concatenate((np.zeros(len(reference["train"])), np.ones(len(execution["train"]))))
    test_x = np.concatenate((reference["test"], execution["test"]), axis=0)
    test_y = np.concatenate((np.zeros(len(reference["test"])), np.ones(len(execution["test"]))))
    return binary_auc(train_x, train_y, test_x, test_y)


def _linear_models(
    reference: dict[str, np.ndarray],
    execution: dict[str, np.ndarray],
    *,
    latent_dim: int,
) -> dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]]:
    from sklearn.cross_decomposition import CCA, PLSCanonical

    if reference["train"].shape[0] <= int(latent_dim):
        raise DependencyUnavailable("diag_60 train pairs are insufficient for fixed 32-D CCA/PLS")
    result = {}
    for name, model in (
        ("CCA", CCA(n_components=int(latent_dim), scale=False, max_iter=500, tol=1.0e-6)),
        ("PLS", PLSCanonical(n_components=int(latent_dim), scale=False, max_iter=500, tol=1.0e-6)),
    ):
        model.fit(reference["train"], execution["train"])
        ref_latent: dict[str, np.ndarray] = {}
        exe_latent: dict[str, np.ndarray] = {}
        for split in ("train", "validation", "test"):
            transformed = model.transform(reference[split], execution[split])
            if not isinstance(transformed, tuple) or len(transformed) != 2:
                raise ProtocolError(f"{name} did not return paired coordinates")
            ref_latent[split] = np.asarray(transformed[0], dtype=np.float32)
            exe_latent[split] = np.asarray(transformed[1], dtype=np.float32)
        result[name] = (
            ref_latent,
            exe_latent,
            {
                "fixed_latent_dim": int(latent_dim),
                "fitted_train_pairs": int(reference["train"].shape[0]),
                "test_untouched_during_fit": True,
            },
        )
    return result


def _evaluate_representation(
    data,
    reference_latent: dict[str, np.ndarray],
    execution_latent: dict[str, np.ndarray],
    raw_execution: dict[str, np.ndarray],
    *,
    phase_bins: int,
) -> dict[str, Any]:
    test = np.asarray(data.split) == "test"
    retrieval = cross_domain_retrieval(
        reference_latent["test"], execution_latent["test"],
        snapshot_ids=data.snapshot_ids[test], phase=data.phase[test], phase_bins=phase_bins,
    )
    preservation = representation_preservation(data, execution_latent, raw_execution)
    outcome: dict[str, Any]
    try:
        train = np.asarray(data.split) == "train"
        outcome = {
            "status": PASS,
            "failure_auroc": binary_auc(
                execution_latent["train"], data.failure[train],
                execution_latent["test"], data.failure[test],
            ),
        }
    except DependencyUnavailable as exc:
        outcome = {"status": SKIPPED_DEPENDENCY, "reason": str(exc)}
    return {
        "latent_source_auc": _source_auc(reference_latent, execution_latent),
        "cross_domain_retrieval": retrieval,
        "contact_future_preservation": preservation,
        "outcome_discrimination": outcome,
        "checkpoint_discrimination": {
            "status": SKIPPED_DEPENDENCY,
            "reason": "exact K/T_u500 pairs share the same u500 execution checkpoint; no checkpoint label exists in this paired dataset",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--domain-index", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(root, spec, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "paired_common_latent_probe.json"
    decision = "UNKNOWN"
    try:
        protocol, thresholds = _protocol(spec)
        index_path = (args.domain_index or output_dir / "domain_triangle_index.json").expanduser().resolve()
        index, bundles = load_domain_index(index_path)
        data = exact_paired_amp_windows(bundles["K"], bundles["T_u500"])
        split_labels = np.asarray(data.split)
        train = split_labels == "train"
        normalizer = FrozenStandardizer.fit(
            np.concatenate((data.reference[train], data.execution[train]), axis=0)
        )
        reference = _split(normalizer.transform(data.reference), split_labels)
        execution = _split(normalizer.transform(data.execution), split_labels)
        raw_baseline = {
            "source_auc": _source_auc(reference, execution),
            "cross_domain_retrieval": cross_domain_retrieval(
                reference["test"], execution["test"],
                snapshot_ids=data.snapshot_ids[split_labels == "test"],
                phase=data.phase[split_labels == "test"],
                phase_bins=int(protocol["phase_bins"]),
            ),
        }
        linear_results = {}
        for name, (ref_latent, exe_latent, metadata) in _linear_models(
            reference, execution, latent_dim=int(protocol["linear_latent_dim"])
        ).items():
            linear_results[name] = {
                **metadata,
                **_evaluate_representation(
                    data, ref_latent, exe_latent, execution,
                    phase_bins=int(protocol["phase_bins"]),
                ),
            }

        contrastive_results = []
        execution_test_latents = []
        for seed in protocol["seeds"]:
            validation_mask = split_labels == "validation"
            model, training = train_paired_contrastive(
                reference,
                execution,
                hidden_dims=protocol["contrastive_encoder_hidden_dims"],
                latent_dim=int(protocol["contrastive_latent_dim"]),
                temperature=float(protocol["contrastive_temperature"]),
                batch_size=int(protocol["batch_size"]),
                epochs_max=int(protocol["epochs_max"]),
                patience=int(protocol["early_stopping_patience"]),
                seed=int(seed),
                validation_metadata={
                    "snapshot_ids": data.snapshot_ids[validation_mask],
                    "phase": data.phase[validation_mask],
                },
                phase_bins=int(protocol["phase_bins"]),
            )
            from diagnostics.common.family_screeners import _encode_batches

            ref_latent = {name: _encode_batches(model, values, reference=True) for name, values in reference.items()}
            exe_latent = {name: _encode_batches(model, values, reference=False) for name, values in execution.items()}
            metrics = _evaluate_representation(
                data, ref_latent, exe_latent, execution,
                phase_bins=int(protocol["phase_bins"]),
            )
            contrastive_results.append({"seed": int(seed), "training": training, **metrics})
            execution_test_latents.append(exe_latent["test"])
        stability = latent_distance_stability(execution_test_latents)
        seed_criteria = []
        for record in contrastive_results:
            criteria = {
                "source_auc": float(record["latent_source_auc"]) <= float(thresholds["latent_source_auc_max"]),
                "retrieval": float(record["cross_domain_retrieval"]["mean"]) >= float(thresholds["cross_domain_retrieval_min"]),
                "contact_future": float(record["contact_future_preservation"]["joint_min_relative_to_raw"]) >= float(thresholds["contact_future_prediction_relative_to_raw_min"]),
            }
            seed_criteria.append({"seed": record["seed"], "criteria": criteria, "passes": all(criteria.values())})
        stability_pass = float(stability["pairwise_spearman_min"]) >= float(thresholds["latent_pairwise_spearman_min"])
        viable = all(record["passes"] for record in seed_criteria) and stability_pass
        decision = "+" if viable else "-"
        result = diagnostic_result(
            "60", PASS if viable else FAIL,
            summary=(
                "paired common coordinates passed every preregistered offline criterion"
                if viable else "paired common coordinates failed at least one preregistered offline criterion"
            ),
            evidence={
                "decision_variable_L": decision,
                "explanation": "L is positive only if all three frozen contrastive seeds and cross-seed stability pass",
                "protocol_sha256": canonical_sha256(protocol),
                "domain_index": str(index_path),
                "domain_index_sha256": sha256_file(index_path),
                "feature_schema_sha256": index["feature_schema_sha256"],
                "exact_pair_count": int(len(data.phase)),
                "split_pair_counts": {name: int(np.sum(split_labels == name)) for name in ("train", "validation", "test")},
                "pair_key": protocol["pair_key"],
                "approximate_or_synthetic_pairs": 0,
                "normalizer_sha256": normalizer.sha256,
                "raw_baseline": raw_baseline,
                "linear_models": linear_results,
                "paired_contrastive": {
                    "seeds": contrastive_results,
                    "seed_criteria": seed_criteria,
                    "latent_distance_stability": stability,
                    "stability_pass": stability_pass,
                },
                "thresholds": thresholds,
                "PPO_used": False,
                "A_mix": "legacy_quarantined",
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "60", SKIPPED_DEPENDENCY,
            summary="paired common-coordinate probe lacks its own exact real pair evidence",
            evidence={"decision_variable_L": "UNKNOWN", "A_mix": "legacy_quarantined", "PPO_used": False},
            errors=[str(exc)],
        )
    except (FileNotFoundError, ProtocolError, ValueError, KeyError, FloatingPointError, RuntimeError) as exc:
        result = diagnostic_result(
            "60", INVALID_PROTOCOL,
            summary="paired common-coordinate protocol failed closed",
            evidence={"decision_variable_L": "UNKNOWN", "A_mix": "legacy_quarantined", "PPO_used": False},
            errors=[str(exc)],
        )
    write_json_exclusive(target, result)
    print(f"[diag_60] {result['status']} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
