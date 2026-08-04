#!/usr/bin/env python3
"""Compare frozen standard AMP critics with K-positive versus T_u500-positive."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.stats import spearmanr
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
    write_json_exclusive,
)
from diagnostics.common.offline_amp import (
    SOURCE_COMMIT, directed_offline_amp_protocol, held_out_amp_metrics,
    load_offline_amp_fit,
)
from diagnostics.common.imitation_6901 import imitation_contract_metadata
from diagnostics.common.policy_class_probe import output_dir_from_args


def _jaccard_top_decile(first: np.ndarray, second: np.ndarray) -> float:
    count = max(1, int(np.ceil(0.1 * first.size)))
    left = set(np.argsort(first)[-count:].tolist())
    right = set(np.argsort(second)[-count:].tolist())
    return len(left & right) / max(1, len(left | right))


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _save_score_bank(
    path: Path,
    *,
    negative,
    seeds: tuple[int, ...],
    logits: dict[str, list[np.ndarray]],
    rewards: dict[str, list[np.ndarray]],
    metadata: dict,
) -> Path:
    mask = np.asarray(negative.split).astype(str) == "test"
    count = int(np.sum(mask))
    for positive_name in ("K", "T_u500"):
        if len(logits[positive_name]) != len(seeds) or len(rewards[positive_name]) != len(seeds):
            raise ProtocolError("offline AMP score bank lacks a frozen seed")
        if any(np.asarray(values).shape != (count,) for values in logits[positive_name]):
            raise ProtocolError("offline AMP held-out logits are not aligned")
        if any(np.asarray(values).shape != (count,) for values in rewards[positive_name]):
            raise ProtocolError("offline AMP held-out rewards are not aligned")
    payload = {
        "seeds": np.asarray(seeds, dtype=np.int64),
        "sample_ids": np.asarray(negative.sample_ids[mask]).astype(str),
        "trajectory_ids": np.asarray(negative.trajectory_ids[mask]).astype(str),
        "snapshot_ids": np.asarray(negative.snapshot_ids[mask]).astype(str),
        "phase": np.asarray(negative.phase[mask], dtype=np.float64),
        "contact_mode": np.asarray(negative.contact_mode[mask]).astype(str),
        "failure": np.asarray(negative.failure[mask], dtype=bool),
        "K_logits": np.stack(logits["K"]),
        "K_rewards": np.stack(rewards["K"]),
        "T_u500_logits": np.stack(logits["T_u500"]),
        "T_u500_rewards": np.stack(rewards["T_u500"]),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
    }
    if any(
        not np.isfinite(np.asarray(payload[key], dtype=np.float64)).all()
        for key in ("phase", "K_logits", "K_rewards", "T_u500_logits", "T_u500_rewards")
    ):
        raise FloatingPointError("offline AMP held-out score bank contains NaN or Inf")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **payload)
    return path


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
    target = output_dir / "positive_substitution.json"
    try:
        device = "cuda:0" if args.device == "auto" and torch.cuda.is_available() else (
            "cpu" if args.device == "auto" else args.device
        )
        protocol = spec.get("analysis_protocols", {}).get("offline_amp_critic", {})
        if protocol.get("source_commit") != SOURCE_COMMIT:
            raise ProtocolError("offline AMP source commit differs from the frozen protocol")
        if protocol.get("positive_domains") != ["K", "T_u500"] or protocol.get("negative_domain") != "A_amp":
            raise ProtocolError("positive-substitution domains are not frozen as K/T_u500 versus A_amp")
        subprocess.run(
            ["git", "cat-file", "-e", f"{SOURCE_COMMIT}^{{commit}}"], cwd=root,
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        index_path = (args.domain_index or output_dir / "domain_triangle_index.json").expanduser().resolve()
        domain_index, bundles = load_domain_index(index_path)
        missing = [name for name in ("K", "T_u500", "A_amp") if name not in bundles]
        if missing:
            raise DependencyUnavailable(f"positive substitution awaits exact domains: {missing}")
        required_representation = {
            "amp_source_commit": SOURCE_COMMIT,
            "amp_representation": "mimickit_g1_chronological_window",
            "amp_frame_dim": 239,
            "amp_window_steps": 10,
            "amp_root_xy_anchor": "newest_frame",
        }
        for name in ("K", "T_u500", "A_amp"):
            metadata = bundles[name].metadata
            absent = [key for key in required_representation if key not in metadata]
            if absent:
                raise DependencyUnavailable(
                    f"domain {name} lacks exact standard-AMP window evidence: {absent}"
                )
            mismatched = {
                key: {"expected": value, "actual": metadata.get(key)}
                for key, value in required_representation.items()
                if metadata.get(key) != value
            }
            if mismatched or bundles[name].features.shape[1] != 2390:
                raise ProtocolError(
                    f"domain {name} is not the frozen 10x239 AMP representation: {mismatched}"
                )
            if bundles[name].metadata.get("imitation_contract") != imitation_contract_metadata():
                raise ProtocolError(
                    f"domain {name} raw frame schema/hash differs from the exact 6901 contract"
                )
        negative = bundles["A_amp"]
        negative_train = negative.split_features("train")
        negative_test = negative.split_features("test")
        seeds = tuple(int(value) for value in protocol["seeds"])
        if len(seeds) != 5:
            raise ProtocolError("offline AMP requires exactly five frozen seeds")
        feature_contract = amp_domain_feature_contract(negative)
        if any(
            amp_domain_feature_contract(bundles[name]) != feature_contract
            for name in ("K", "T_u500")
        ):
            raise ProtocolError("K/T/A_amp critic domains do not share one exact feature contract")
        feature_contract_sha256 = canonical_sha256(feature_contract)
        base_protocol_sha256 = canonical_sha256(dict(protocol))
        index_records = {
            str(record["name"]): dict(record) for record in domain_index["domains"]
        }
        diag35_index_path = output_dir / "models" / "diag35" / "index.json"
        if not diag35_index_path.is_file():
            raise DependencyUnavailable("diag36 requires the frozen diag35 critic index")
        diag35_index = read_json(diag35_index_path)
        if (
            diag35_index.get("status") != PASS
            or diag35_index.get("artifact_schema")
            != "largebox_diag35_directed_offline_amp_index_v1"
            or diag35_index.get("source_commit") != SOURCE_COMMIT
            or diag35_index.get("base_protocol_sha256") != base_protocol_sha256
            or diag35_index.get("feature_contract_sha256") != feature_contract_sha256
            or diag35_index.get("base_protocol") != dict(protocol)
            or diag35_index.get("feature_contract") != feature_contract
        ):
            raise ProtocolError("diag35 critic index does not match the exact substitution contract")
        diag35_records = {
            (
                str(record.get("source_negative")),
                str(record.get("destination_positive")),
                int(record.get("seed", -1)),
            ): dict(record)
            for record in diag35_index.get("models", ())
            if isinstance(record, dict)
        }
        by_positive: dict[str, dict] = {}
        score_banks: dict[str, list[np.ndarray]] = {}
        logit_banks: dict[str, list[np.ndarray]] = {}
        artifact_records: list[dict] = []
        model_root = output_dir / "models" / "offline_amp"
        for positive_name in ("K", "T_u500"):
            positive = bundles[positive_name]
            edge_protocol = directed_offline_amp_protocol(
                protocol,
                negative_domain="A_amp",
                positive_domain=positive_name,
            )
            edge_protocol_sha256 = canonical_sha256(edge_protocol)
            negative_rewards: list[np.ndarray] = []
            negative_logits: list[np.ndarray] = []
            seed_metrics: list[dict] = []
            for seed in seeds:
                key = ("A_amp", positive_name, int(seed))
                if key not in diag35_records:
                    raise DependencyUnavailable(f"diag35 critic index lacks required edge/seed {key}")
                source_record = diag35_records[key]
                model_path = Path(str(source_record["path"])).expanduser().resolve()
                if (
                    source_record.get("protocol_sha256") != edge_protocol_sha256
                    or source_record.get("base_protocol_sha256")
                    != base_protocol_sha256
                    or source_record.get("feature_contract_sha256")
                    != feature_contract_sha256
                    or sha256_file(model_path) != source_record.get("sha256")
                ):
                    raise ProtocolError(f"diag35 critic artifact hash/contract mismatch: {model_path}")
                fit, fit_metadata = load_offline_amp_fit(
                    model_path,
                    device=device,
                    expected_protocol_sha256=edge_protocol_sha256,
                    expected_feature_contract_sha256=feature_contract_sha256,
                )
                provenance = fit_metadata.get("training_provenance")
                if (
                    int(fit.seed) != int(seed)
                    or not isinstance(provenance, dict)
                    or provenance.get("source_negative") != "A_amp"
                    or provenance.get("destination_positive") != positive_name
                    or provenance.get("source_bundle_sha256")
                    != index_records["A_amp"]["bundle_sha256"]
                    or provenance.get("destination_bundle_sha256")
                    != index_records[positive_name]["bundle_sha256"]
                ):
                    raise ProtocolError(f"diag35 critic provenance mismatch: {model_path}")
                held_out_logits = fit.logits(negative_test)
                held_out_rewards = fit.rewards(negative_test)
                negative_logits.append(held_out_logits)
                negative_rewards.append(held_out_rewards)
                artifact_records.append({
                    "positive_domain": positive_name,
                    "negative_domain": "A_amp",
                    "seed": int(seed),
                    "path": str(model_path),
                    "sha256": sha256_file(model_path),
                    "protocol_sha256": edge_protocol_sha256,
                    "base_protocol_sha256": base_protocol_sha256,
                    "feature_contract_sha256": feature_contract_sha256,
                    "reused_from_diag35": True,
                })
                metrics = held_out_amp_metrics(
                    fit, negative_test, positive.split_features("test")
                )
                if not all(
                    np.isfinite(float(metrics[key]))
                    for key in ("auc", "negative_reward_mean", "positive_reward_mean")
                ):
                    raise FloatingPointError("offline AMP held-out metrics are non-finite")
                seed_metrics.append({"seed": seed, **metrics})
                del fit
                if str(device).startswith("cuda"):
                    torch.cuda.empty_cache()
            score_banks[positive_name] = negative_rewards
            logit_banks[positive_name] = negative_logits
            by_positive[positive_name] = {
                "by_seed": seed_metrics,
                "negative_bank_reward_stability": reward_ordering_stability(negative_rewards),
            }
        cross = []
        for seed_index, seed in enumerate(seeds):
            k_reward = score_banks["K"][seed_index]
            t_reward = score_banks["T_u500"][seed_index]
            record = {
                "seed": seed,
                "negative_bank_spearman": float(spearmanr(k_reward, t_reward).statistic),
                "negative_bank_top_decile_jaccard": _jaccard_top_decile(k_reward, t_reward),
                "mean_reward_shift_T_minus_K": float(np.mean(t_reward - k_reward)),
            }
            if not all(np.isfinite(float(value)) for key, value in record.items() if key != "seed"):
                raise FloatingPointError("K/T positive comparison produced a non-finite statistic")
            cross.append(record)
        held_out_mask = np.asarray(negative.split).astype(str) == "test"
        score_bank_metadata = {
            "artifact_schema": "largebox_offline_amp_held_out_score_bank_v1",
            "negative_domain": "A_amp",
            "split": "test",
            "count": int(negative_test.shape[0]),
            "sample_id_sha256": _array_sha256(
                np.asarray(negative.sample_ids[held_out_mask]).astype("S")
            ),
            "feature_sha256": _array_sha256(negative_test),
            "seeds": list(seeds),
            "positive_domains": ["K", "T_u500"],
            "base_protocol_sha256": base_protocol_sha256,
            "feature_contract_sha256": feature_contract_sha256,
        }
        score_bank_path = _save_score_bank(
            model_root / "held_out_A_amp_test_score_bank.npz",
            negative=negative,
            seeds=seeds,
            logits=logit_banks,
            rewards=score_banks,
            metadata=score_bank_metadata,
        )
        artifact_index_path = model_root / "index.json"
        artifact_index = {
            "status": PASS,
            "artifact_schema": "largebox_offline_amp_positive_substitution_index_v1",
            "source_commit": SOURCE_COMMIT,
            "source_diag35_index": str(diag35_index_path),
            "source_diag35_index_sha256": sha256_file(diag35_index_path),
            "base_protocol": dict(protocol),
            "base_protocol_sha256": base_protocol_sha256,
            "feature_contract": feature_contract,
            "feature_contract_sha256": feature_contract_sha256,
            "models": artifact_records,
            "model_storage": "references_exact_A_amp_to_K_and_A_amp_to_T_u500_diag35_artifacts_without_duplication",
            "held_out_score_bank": {
                "path": str(score_bank_path),
                "sha256": sha256_file(score_bank_path),
                **score_bank_metadata,
            },
        }
        write_json_exclusive(artifact_index_path, artifact_index)
        result = diagnostic_result(
            "36", PASS,
            summary="K-positive and T_u500-positive critics were compared on the exact same A_amp bank",
            evidence={
                "source_commit": SOURCE_COMMIT, "protocol": dict(protocol),
                "device": device,
                "positive_results": by_positive, "cross_positive_comparison": cross,
                "negative_bank_identity": "same A_amp test rows for both positives and every seed",
                "offline_amp_artifact_index": str(artifact_index_path),
                "offline_amp_artifact_index_sha256": sha256_file(artifact_index_path),
                "held_out_score_bank": str(score_bank_path),
                "held_out_score_bank_sha256": sha256_file(score_bank_path),
                "loadable_model_count": len(artifact_records),
                "model_storage": "reused_exact_diag35_artifacts_without_retraining",
                "retrained": False,
                "source_diag35_index": str(diag35_index_path),
                "source_diag35_index_sha256": sha256_file(diag35_index_path),
                "feature_contract_sha256": feature_contract_sha256,
                "base_protocol_sha256": base_protocol_sha256,
                "native_stochastic_excluded": True,
            },
        )
    except (DependencyUnavailable, subprocess.CalledProcessError) as exc:
        result = diagnostic_result(
            "36", SKIPPED_DEPENDENCY,
            summary="teacher-positive substitution awaits exact AMP domains/source commit",
            errors=[str(exc)],
        )
    except (FileNotFoundError, ProtocolError, ValueError, KeyError, FloatingPointError) as exc:
        result = diagnostic_result("36", INVALID_PROTOCOL, summary="offline AMP substitution protocol failed closed", errors=[str(exc)])
    write_json_exclusive(target, result)
    print(f"[diag_36] {result['status']} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
