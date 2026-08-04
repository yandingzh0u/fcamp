#!/usr/bin/env python3
"""Audit checkpoint fingerprints without admitting native stochastic data to overlap."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
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


def _maximum(value) -> float:
    return float(value.detach().abs().max().item()) if value.numel() else 0.0


def _prefix_error(first, second) -> float:
    if first.ndim < 1 or second.ndim < 1 or first.shape[1:] != second.shape[1:]:
        raise ProtocolError("fingerprint tensors have incompatible non-time shapes")
    count = min(int(first.shape[0]), int(second.shape[0]))
    if count < 1:
        raise ProtocolError("fingerprint tensors have no common time prefix")
    return _maximum(first[:count] - second[:count])


def main() -> int:
    args = _arguments()
    output_dir = args.output_dir.expanduser().resolve()
    status_path = output_dir / "fingerprints.json"
    try:
        import torch

        from diagnostics.common.canonical_collection import (
            CanonicalCollectionProtocol,
            load_rollout_index,
            load_rollout_trajectory,
            policy_only_payload,
            validate_frozen_collection_semantics,
        )
        from diagnostics.common.checkpoint_io import state_digest
        from diagnostics.common.noise_bank import CollectorMode, is_overlap_eligible

        spec = load_spec(args.spec)
        protocol = CanonicalCollectionProtocol.from_spec(spec)
        validate_frozen_collection_semantics(spec, protocol)
        collection_status_path = output_dir / "canonical_rollout_index.status.json"
        index_path = output_dir / "canonical_rollout_index.parquet"
        if not collection_status_path.is_file() or not index_path.is_file():
            raise DependencyUnavailable("diag_12 canonical collection is absent")
        collection_status = read_json(collection_status_path)
        if collection_status.get("status") != PASS:
            raise DependencyUnavailable("diag_12 did not PASS")
        rows = load_rollout_index(index_path)
        expected_modes = {mode.value for mode, _ in protocol.branch_variants}
        actual_modes = {str(row["collector_mode"]) for row in rows}
        if actual_modes != expected_modes:
            raise ProtocolError(
                f"collector modes differ from frozen protocol: {sorted(actual_modes)}"
            )
        native = [
            row
            for row in rows
            if row["collector_mode"] == CollectorMode.NATIVE_STOCHASTIC.value
        ]
        if not native:
            raise ProtocolError("native_stochastic fingerprint rows are missing")
        if any(bool(row["eligible_for_primary_overlap"]) for row in native):
            raise ProtocolError("native_stochastic is marked primary-overlap eligible")
        for row in rows:
            expected = is_overlap_eligible(str(row["collector_mode"]))
            if bool(row["eligible_for_primary_overlap"]) != expected:
                raise ProtocolError("index overlap eligibility contradicts collector mode")

        # One snapshot is sufficient to audit algebraic action paths and the
        # checkpoint-independent initial condition.  Every row still remains
        # available for later scientific analyses.
        first_snapshot = sorted({str(row["snapshot_id"]) for row in rows})[0]
        probes = [row for row in rows if str(row["snapshot_id"]) == first_snapshot]
        path_errors: dict[str, float] = defaultdict(float)
        initial_by_condition: dict[tuple[str, float], list[tuple[int, torch.Tensor]]] = defaultdict(list)
        epsilon_by_condition: dict[tuple[str, float], list[tuple[int, torch.Tensor]]] = defaultdict(list)
        native_std_by_update: dict[int, torch.Tensor] = {}
        for row in probes:
            # Exactly one probe row is read from each shard.  Do not retain the
            # 90 full [T,N,...] shards in a process-wide cache.
            trajectory = load_rollout_trajectory(index_path, row)
            action = trajectory["action"]
            mode = str(row["collector_mode"])
            sigma = float(row["common_sigma"])
            mean = action["mean"]
            sampled = action["sampled"]
            epsilon = action["common_epsilon"]
            std = action["std"]
            if mode in {
                CollectorMode.CLEAN_MEAN.value,
                CollectorMode.CONTROLLED_ENVIRONMENT.value,
            }:
                reconstructed = mean
            elif mode == CollectorMode.COMMON_ACTION_NOISE.value:
                # Match construct_action_record's exact float32 expression.
                # ``(sampled - mean) - noise`` is algebraically equivalent but
                # not bit-equivalent because subtraction is not associative.
                reconstructed = mean + sigma * epsilon
            elif mode == CollectorMode.NATIVE_STOCHASTIC.value:
                reconstructed = mean + std * epsilon
                native_std_by_update[int(row["checkpoint_update"])] = std[0].clone()
            else:  # pragma: no cover - mode set checked above
                raise ProtocolError(f"unknown mode: {mode}")
            residual = sampled - reconstructed
            path_errors[mode] = max(path_errors[mode], _maximum(residual))
            clip = 100.0
            applied_expected = torch.clamp(sampled, -clip, clip)
            path_errors[f"{mode}/applied"] = max(
                path_errors[f"{mode}/applied"],
                _maximum(action["applied"] - applied_expected),
            )
            condition = (mode, sigma)
            update = int(row["checkpoint_update"])
            initial_by_condition[condition].append(
                (update, trajectory["observation"]["actor_full"][0].clone())
            )
            epsilon_by_condition[condition].append((update, epsilon.clone()))

        if any(value != 0.0 for value in path_errors.values()):
            raise ProtocolError(f"collector action algebra is not exact: {dict(path_errors)}")
        initial_errors: dict[str, float] = {}
        epsilon_errors: dict[str, float] = {}
        for condition, values in initial_by_condition.items():
            values.sort(key=lambda pair: pair[0])
            baseline = values[0][1]
            key = f"{condition[0]}@{condition[1]:g}"
            initial_errors[key] = max(_maximum(value - baseline) for _, value in values)
        for condition, values in epsilon_by_condition.items():
            values.sort(key=lambda pair: pair[0])
            baseline = values[0][1]
            key = f"{condition[0]}@{condition[1]:g}"
            epsilon_errors[key] = max(_prefix_error(value, baseline) for _, value in values)
        # This diagnostic is explicitly tasked with *measuring* reset/collector
        # fingerprints.  A finite nonzero initial-observation difference is a
        # scientific finding, not malformed evidence.  In the present bank it
        # is isolated to the first clean branch after Kit startup (the
        # controlled/NoiseBank branches remain bit-exact); preserve and report
        # it so downstream analyses can avoid that source cue.
        initial_observation_fingerprint = any(
            value != 0.0 for value in initial_errors.values()
        )
        if any(value != 0.0 for value in epsilon_errors.values()):
            raise ProtocolError(
                f"NoiseBank epsilon varies by checkpoint: {epsilon_errors}"
            )

        # Checkpoint normalizers and learned std are intentionally measured as
        # fingerprints.  Finding them is a valid diagnostic result, not a
        # protocol failure; native rows remain quarantined from overlap.
        checkpoint_paths: dict[int, Path] = {}
        for row in rows:
            checkpoint_paths[int(row["checkpoint_update"])] = Path(str(row["checkpoint_path"]))
        normalizer_digests: dict[int, str] = {}
        payload_std_digests: dict[int, str] = {}
        for update, path in sorted(checkpoint_paths.items()):
            payload = policy_only_payload(path)
            policy = payload["policy"]
            normalizer = {
                key: value
                for key, value in policy.items()
                if str(key).startswith("actor_obs_normalizer.")
            }
            std_state = {
                key: value
                for key, value in policy.items()
                if str(key) == "actor.std"
            }
            if not normalizer or not std_state:
                raise ProtocolError(f"checkpoint {update} lacks normalizer/std state")
            normalizer_digests[update] = state_digest(normalizer)
            payload_std_digests[update] = state_digest(std_state)
        if set(native_std_by_update) != set(checkpoint_paths):
            raise ProtocolError("native rollout std coverage differs from checkpoint coverage")
        native_std_mean = {
            update: float(value.float().mean().item())
            for update, value in sorted(native_std_by_update.items())
        }
        if any(not math.isfinite(value) or value <= 0.0 for value in native_std_mean.values()):
            raise ProtocolError("native policy std is non-positive or non-finite")

        evidence = {
            "probe_snapshot_id": first_snapshot,
            "action_path_max_abs_error": dict(path_errors),
            "same_snapshot_initial_observation_max_abs_error": initial_errors,
            "common_epsilon_cross_checkpoint_max_abs_error": epsilon_errors,
            "normalizer_sha256_by_update": normalizer_digests,
            "normalizer_unique_count": len(set(normalizer_digests.values())),
            "native_std_state_sha256_by_update": payload_std_digests,
            "native_std_unique_count": len(set(payload_std_digests.values())),
            "native_std_mean_by_update": native_std_mean,
            "native_noise_scale_is_checkpoint_fingerprint": len(set(payload_std_digests.values())) > 1,
            "normalizer_is_checkpoint_fingerprint": len(set(normalizer_digests.values())) > 1,
            "reset_phase_or_collector_path_is_checkpoint_fingerprint": (
                initial_observation_fingerprint
            ),
            "initial_observation_bit_exact": not initial_observation_fingerprint,
            "native_stochastic_primary_overlap_rows": 0,
            "primary_overlap_modes": sorted(
                mode.value
                for mode in CollectorMode
                if is_overlap_eligible(mode)
            ),
        }
        result = diagnostic_result(
            "13",
            PASS,
            summary=(
                "action/NoiseBank algebra is bit-exact; a measured initial-state, "
                "normalizer, or learned-std fingerprint is reported rather than hidden"
                if initial_observation_fingerprint
                else "collector/reset paths are condition-matched; learned std and normalizer "
                "fingerprints are measured and native stochastic data is quarantined"
            ),
            evidence=evidence,
            warnings=(
                [
                    "a finite initial-observation collector-order fingerprint was measured; "
                    "use the condition-matched controlled branches for cross-checkpoint comparisons"
                ]
                if initial_observation_fingerprint
                else []
            ),
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "13", SKIPPED_DEPENDENCY, summary=str(exc), errors=[str(exc)]
        )
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, ProtocolError, json.JSONDecodeError) as exc:
        result = diagnostic_result(
            "13",
            INVALID_PROTOCOL,
            summary="collection fingerprint protocol is invalid",
            errors=[str(exc)],
        )
    write_json_exclusive(status_path, result)
    print(f"[diag_13] {result['status']} {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
