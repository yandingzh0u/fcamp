#!/usr/bin/env python3
"""Audit ratio ESS and directed five-seed standard-AMP reward reliability."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.domain_data import amp_domain_feature_contract, load_domain_index
from diagnostics.common.domain_triangle import reward_ordering_stability
from diagnostics.common.manifest import (
    INVALID_PROTOCOL, PASS, SKIPPED_DEPENDENCY, DependencyUnavailable,
    ProtocolError, canonical_sha256, diagnostic_result, load_spec, read_json,
    sha256_file,
    write_csv_exclusive, write_json_exclusive,
)
from diagnostics.common.offline_amp import (
    SOURCE_COMMIT, directed_offline_amp_protocol, held_out_amp_metrics,
    load_offline_amp_fit, save_offline_amp_fit, train_offline_amp_critic,
)
from diagnostics.common.policy_class_probe import output_dir_from_args


HELD_OUT_REWARD_BANK_SEED = 0x524557415244
FROZEN_REWARD_FORMULA = "2*(-log(max(1-sigmoid(logit),1e-4)))"


def _balanced_union(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    count = min(first.shape[0], second.shape[0])
    if count < 2:
        raise DependencyUnavailable("directed reward edge lacks a balanced held-out bank")
    rng = np.random.default_rng(HELD_OUT_REWARD_BANK_SEED)
    left = first[rng.choice(first.shape[0], count, replace=False)]
    right = second[rng.choice(second.shape[0], count, replace=False)]
    return np.concatenate((left, right), axis=0)


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _overlap_for_order(overlap_by_pair, source: str, destination: str):
    direct = overlap_by_pair.get((source, destination))
    if direct is not None:
        return direct, False
    reverse = overlap_by_pair.get((destination, source))
    if reverse is None:
        raise ProtocolError(f"missing effective-overlap pair for {source}->{destination}")
    return reverse, True


def _require_finite_mapping(name: str, values) -> None:
    if isinstance(values, dict):
        for key, value in values.items():
            _require_finite_mapping(f"{name}.{key}", value)
    elif isinstance(values, (list, tuple)):
        for index, value in enumerate(values):
            _require_finite_mapping(f"{name}[{index}]", value)
    elif isinstance(values, (float, np.floating)) and not np.isfinite(float(values)):
        raise FloatingPointError(f"{name} is non-finite")


def _load_or_train_fit(
    path: Path,
    *,
    negative_train: np.ndarray,
    positive_train: np.ndarray,
    seed: int,
    protocol,
    device: str,
    feature_contract,
    training_provenance,
):
    protocol_sha256 = canonical_sha256(dict(protocol))
    feature_contract_sha256 = canonical_sha256(dict(feature_contract))
    if path.is_file():
        fit, metadata = load_offline_amp_fit(
            path,
            device=device,
            expected_protocol_sha256=protocol_sha256,
            expected_feature_contract_sha256=feature_contract_sha256,
        )
        if int(fit.seed) != int(seed):
            raise ProtocolError(f"cached critic seed mismatch: {path}")
        if metadata.get("training_provenance") != dict(training_provenance):
            raise ProtocolError(f"cached critic training provenance mismatch: {path}")
        return fit, True
    fit = train_offline_amp_critic(
        negative_train,
        positive_train,
        seed=seed,
        protocol=protocol,
        device=device,
    )
    save_offline_amp_fit(
        path,
        fit,
        protocol=protocol,
        feature_contract=feature_contract,
        training_provenance=training_provenance,
    )
    return fit, False


def _write_or_validate_index(path: Path, payload: dict) -> None:
    if path.is_file():
        existing = read_json(path)
        if existing != payload:
            raise ProtocolError("existing diag35 critic index differs from this exact run")
        return
    write_json_exclusive(path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--domain-index", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(root, spec, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "ratio_reliability.json"
    try:
        analysis = spec.get("analysis_protocols", {})
        protocol = analysis.get("reward_reliability", {})
        critic_protocol = analysis.get("offline_amp_critic", {})
        seeds = tuple(int(value) for value in protocol.get("seeds", ()))
        expected_domains = ("K", "T_u200", "T_u500", "A_amp", "B")
        frozen = {
            "critic_family": "standard_commit_6901_amp_discriminator",
            "domain_edges": "all_ordered_pairs_without_self_among_K_T_u200_T_u500_A_amp_B",
            "edge_semantics": "source_domain_is_negative_and_destination_domain_is_positive",
            "normalizer_fit": "balanced_source_and_destination_train_split_only_then_frozen",
            "training_split": "train",
            "held_out_state_bank": "one_fixed_balanced_test_split_union_per_ordered_edge_shared_by_all_five_seeds",
            "source_classifier_scores_for_reward_icc": "forbidden",
            "reverse_edge_reuse": "forbidden_train_each_direction_explicitly",
        }
        if any(protocol.get(key) != value for key, value in frozen.items()):
            raise ProtocolError("reward-reliability directed AMP protocol is not frozen exactly")
        if protocol.get("reward_formula") != FROZEN_REWARD_FORMULA:
            raise ProtocolError("reward-reliability formula differs from standard AMP softplus")
        required_metrics = set(protocol.get("required_seed_metrics", ()))
        if required_metrics != {
            "pairwise_spearman", "icc_consistency", "sign_disagreement",
            "top_decile_overlap", "bottom_decile_overlap",
        }:
            raise ProtocolError("reward reliability seed metrics are not frozen exactly")
        if len(seeds) != 5 or tuple(int(value) for value in critic_protocol.get("seeds", ())) != seeds:
            raise ProtocolError("reward reliability and offline AMP seeds differ")
        if critic_protocol.get("source_commit") != SOURCE_COMMIT:
            raise ProtocolError("reward reliability does not reference standard commit 6901")
        threshold = spec.get("thresholds", {}).get("empirical_effective_overlap_edge", {})
        if threshold.get("ess_gate_uses") != "unclipped_density_ratio":
            raise ProtocolError("ESS gate must use the frozen unclipped density ratio")
        index_path = (args.domain_index or output_dir / "domain_triangle_index.json").expanduser().resolve()
        domain_index, bundles = load_domain_index(index_path)
        if tuple(sorted(bundles)) != tuple(sorted(expected_domains)):
            raise ProtocolError(f"reward reliability requires exactly {expected_domains}")
        overlap_path = output_dir / "effective_overlap_details.json"
        overlap = read_json(overlap_path)
        if overlap.get("status") != PASS:
            raise DependencyUnavailable("effective-overlap details are not PASS")
        overlap_by_pair = {
            (str(record["source"]), str(record["target"])): record
            for record in overlap["pairs"]
        }
        device = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else (
            "cpu" if args.device == "auto" else args.device
        )
        feature_contract = amp_domain_feature_contract(bundles[expected_domains[0]])
        if any(
            amp_domain_feature_contract(bundles[name]) != feature_contract
            for name in expected_domains[1:]
        ):
            raise ProtocolError("directed critic domains do not share one exact AMP feature contract")
        base_protocol_sha256 = canonical_sha256(dict(critic_protocol))
        feature_contract_sha256 = canonical_sha256(feature_contract)
        domain_records = {
            str(record["name"]): dict(record) for record in domain_index["domains"]
        }
        model_root = output_dir / "models" / "diag35"
        artifact_records = []
        reused_model_count = 0
        records = []
        icc_matrix = {
            source: {
                destination: (1.0 if source == destination else float("nan"))
                for destination in expected_domains
            }
            for source in expected_domains
        }
        for source_name in expected_domains:
            for destination_name in expected_domains:
                if source_name == destination_name:
                    continue
                source = bundles[source_name]
                destination = bundles[destination_name]
                held_out_bank = _balanced_union(
                    source.split_features("test"), destination.split_features("test")
                )
                held_out_bank_sha256 = _array_sha256(held_out_bank)
                source_train = source.split_features("train")
                destination_train = destination.split_features("train")
                source_test = source.split_features("test")
                destination_test = destination.split_features("test")
                edge_protocol = directed_offline_amp_protocol(
                    critic_protocol,
                    negative_domain=source_name,
                    positive_domain=destination_name,
                )
                edge_protocol_sha256 = canonical_sha256(edge_protocol)
                rewards = []
                critic_metrics = []
                edge_artifacts = []
                for seed in seeds:
                    training_provenance = {
                        "source_negative": source_name,
                        "destination_positive": destination_name,
                        "directed_edge_id": edge_protocol["directed_edge_id"],
                        "edge_semantics": frozen["edge_semantics"],
                        "training_split": "train",
                        "source_bundle_sha256": domain_records[source_name]["bundle_sha256"],
                        "destination_bundle_sha256": domain_records[destination_name]["bundle_sha256"],
                        "source_train_count": int(source_train.shape[0]),
                        "destination_train_count": int(destination_train.shape[0]),
                        "held_out_split": "test",
                        "held_out_bank_sha256": held_out_bank_sha256,
                    }
                    model_path = (
                        model_root / f"{source_name}_to_{destination_name}" / f"seed_{seed}.pt"
                    )
                    fit, reused = _load_or_train_fit(
                        model_path,
                        negative_train=source_train,
                        positive_train=destination_train,
                        seed=seed,
                        protocol=edge_protocol,
                        device=device,
                        feature_contract=feature_contract,
                        training_provenance=training_provenance,
                    )
                    reused_model_count += int(reused)
                    reward = fit.rewards(held_out_bank)
                    if not np.isfinite(reward).all():
                        raise FloatingPointError("directed critic reward contains NaN or Inf")
                    rewards.append(reward)
                    metrics = held_out_amp_metrics(fit, source_test, destination_test)
                    _require_finite_mapping(
                        f"critic_metrics.{source_name}_to_{destination_name}.seed_{seed}",
                        metrics,
                    )
                    critic_metrics.append({"seed": seed, **metrics})
                    artifact = {
                        "source_negative": source_name,
                        "destination_positive": destination_name,
                        "directed_edge_id": edge_protocol["directed_edge_id"],
                        "seed": int(seed),
                        "path": str(model_path),
                        "sha256": sha256_file(model_path),
                        "protocol_sha256": edge_protocol_sha256,
                        "base_protocol_sha256": base_protocol_sha256,
                        "feature_contract_sha256": feature_contract_sha256,
                    }
                    artifact_records.append(artifact)
                    edge_artifacts.append(artifact)
                    del fit
                    if str(device).startswith("cuda"):
                        torch.cuda.empty_cache()
                reliability = reward_ordering_stability(rewards)
                _require_finite_mapping(
                    f"reward_reliability.{source_name}_to_{destination_name}",
                    reliability,
                )
                overlap_record, reversed_record = _overlap_for_order(
                    overlap_by_pair, source_name, destination_name
                )
                forward_direction = "reverse" if reversed_record else "forward"
                reverse_direction = "forward" if reversed_record else "reverse"
                forward_coverage_key = (
                    "reverse_knn_coverage" if reversed_record else "forward_knn_coverage"
                )
                reverse_coverage_key = (
                    "forward_knn_coverage" if reversed_record else "reverse_knn_coverage"
                )
                tau_key = str(float(threshold["posterior_overlap_tau"]))
                forward_ess = overlap_record["ratio_ess"][forward_direction]["unclipped"]["ess_fraction_mean"]
                reverse_ess = overlap_record["ratio_ess"][reverse_direction]["unclipped"]["ess_fraction_mean"]
                gates = {
                    "source_auc": float(overlap_record["mean_auc"]) <= float(threshold["source_auc_max"]),
                    "posterior_overlap": float(overlap_record["posterior_overlap"][tau_key]) >= float(threshold["posterior_overlap_fraction_min"]),
                    "forward_knn_coverage": float(overlap_record[forward_coverage_key]) >= float(threshold["forward_knn_coverage_min"]),
                    "reverse_knn_coverage": float(overlap_record[reverse_coverage_key]) >= float(threshold["reverse_knn_coverage_min"]),
                    "forward_unclipped_ess": float(forward_ess) >= float(threshold["forward_ess_over_n_min"]),
                    "reverse_unclipped_ess": float(reverse_ess) >= float(threshold["reverse_ess_over_n_min"]),
                    "reward_icc": float(reliability["icc_consistency"]) >= float(threshold["reward_icc_min"]),
                }
                records.append({
                    "source_negative": source_name,
                    "destination_positive": destination_name,
                    "direction_trained_explicitly": True,
                    "held_out_bank_count": int(held_out_bank.shape[0]),
                    "held_out_bank_sha256": held_out_bank_sha256,
                    "reward_ordering_stability": reliability,
                    "critic_by_seed": critic_metrics,
                    "critic_artifacts": edge_artifacts,
                    "density_ratio": {
                        "forward": overlap_record["ratio_ess"][forward_direction],
                        "reverse": overlap_record["ratio_ess"][reverse_direction],
                    },
                    "comparability_checks": gates,
                    "empirically_comparable": all(gates.values()),
                })
                icc_matrix[source_name][destination_name] = reliability["icc_consistency"]
        artifact_index_path = model_root / "index.json"
        artifact_index = {
            "status": PASS,
            "artifact_schema": "largebox_diag35_directed_offline_amp_index_v1",
            "source_commit": SOURCE_COMMIT,
            "edge_semantics": frozen["edge_semantics"],
            "base_protocol": dict(critic_protocol),
            "base_protocol_sha256": base_protocol_sha256,
            "directed_protocol_rule": (
                "base_protocol with negative_domain=source, positive_domains=[destination], "
                "directed_edge_id and protocol_role bound per ordered edge"
            ),
            "feature_contract": feature_contract,
            "feature_contract_sha256": feature_contract_sha256,
            "domain_bundle_sha256": {
                name: domain_records[name]["bundle_sha256"] for name in expected_domains
            },
            "models": artifact_records,
        }
        _write_or_validate_index(artifact_index_path, artifact_index)
        table_path = output_dir / "tables" / "reward_icc_matrix.csv"
        write_csv_exclusive(
            table_path,
            [{"source_negative": name, **icc_matrix[name]} for name in expected_domains],
            fieldnames=("source_negative", *expected_domains),
        )
        result = diagnostic_result(
            "35", PASS,
            summary="all 20 directed edges received independent five-seed standard-AMP critics",
            evidence={
                "pairs": records,
                "reward_icc_matrix": str(table_path),
                "device": device,
                "critic_source_commit": SOURCE_COMMIT,
                "directed_critic_count": len(records) * len(seeds),
                "offline_amp_artifact_index": str(artifact_index_path),
                "offline_amp_artifact_index_sha256": sha256_file(artifact_index_path),
                "loadable_model_count": len(artifact_records),
                "reused_model_count": reused_model_count,
                "feature_contract_sha256": feature_contract_sha256,
                "base_protocol_sha256": base_protocol_sha256,
                "source_classifier_used_as_reward": False,
                "ess_gate_uses": "unclipped_density_ratio",
                "clipped_ess_role": "sensitivity report only; never selected for the gate",
                "native_stochastic_excluded": True,
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result("35", SKIPPED_DEPENDENCY, summary="ratio/reward reliability awaits overlap/domain assets", errors=[str(exc)])
    except (FileNotFoundError, ProtocolError, ValueError, KeyError, FloatingPointError, RuntimeError) as exc:
        result = diagnostic_result("35", INVALID_PROTOCOL, summary="directed standard-AMP reliability protocol failed closed", errors=[str(exc)])
    write_json_exclusive(target, result)
    print(f"[diag_35] {result['status']} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
