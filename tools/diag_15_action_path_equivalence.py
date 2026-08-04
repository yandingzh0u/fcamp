#!/usr/bin/env python3
"""Prove canonical policy means equal the production deterministic action path."""

from __future__ import annotations

import argparse
import json
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
from diagnostics.common.isaac_exit import finish_isaac_entrypoint

_ALLOW_HARD_EXIT = __name__ == "__main__"


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser


def _write_status(path: Path, result: dict[str, Any]) -> int:
    write_json_exclusive(path, result)
    print(f"[diag_15] {result['status']} {path}")
    return 0


def main() -> int:
    try:
        from isaaclab.app import AppLauncher
    except (ImportError, ModuleNotFoundError) as exc:
        args = _base_parser().parse_args()
        output_dir = args.output_dir.expanduser().resolve()
        result = diagnostic_result(
            "15",
            SKIPPED_DEPENDENCY,
            summary="Isaac Lab is unavailable; production action path cannot be instantiated",
            errors=[str(exc)],
        )
        return _write_status(output_dir / "action_equivalence.json", result)

    parser = _base_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    result_path = output_dir / "action_equivalence.json"
    app_launcher = None
    trainer = None
    try:
        import torch

        from diagnostics.common.canonical_collection import (
            CanonicalCollectionProtocol,
            assert_startup_randomization_matches,
            close_collection_trainer,
            load_rollout_index,
            load_rollout_shard,
            make_collection_trainer,
            read_dense_checkpoint_records,
            resolve_shard_path,
            switch_policy_state,
            validate_frozen_collection_semantics,
        )
        from diagnostics.common.checkpoint_io import state_digest
        from diagnostics.common.noise_bank import CollectorMode
        from diagnostics.common.rollout_collector import (
            deterministic_action_equivalence,
            policy_mean_and_std,
        )
        from diagnostics.common.snapshot_bank import SnapshotBank

        spec = load_spec(args.spec)
        protocol = CanonicalCollectionProtocol.from_spec(spec)
        validate_frozen_collection_semantics(spec, protocol)
        required = {
            "manifest": output_dir / "manifest.json",
            "snapshot_status": output_dir / "snapshot_bank.status.json",
            "collection_status": output_dir / "canonical_rollout_index.status.json",
            "snapshot_bank": output_dir / "snapshot_bank.pt",
            "index": output_dir / "canonical_rollout_index.parquet",
            "dense": output_dir / "checkpoints/dense_checkpoint_inventory.csv",
        }
        absent = [name for name, path in required.items() if not path.is_file()]
        if absent:
            raise DependencyUnavailable(f"action-equivalence dependencies are missing: {absent}")
        if read_json(required["snapshot_status"]).get("status") != PASS:
            raise DependencyUnavailable("diag_11 did not PASS")
        if read_json(required["collection_status"]).get("status") != PASS:
            raise DependencyUnavailable("diag_12 did not PASS")
        manifest = read_json(required["manifest"])
        bank = SnapshotBank.load(required["snapshot_bank"])
        rows = load_rollout_index(required["index"])
        checkpoints = read_dense_checkpoint_records(required["dense"])

        from engine.config import load_config

        cfg = load_config(
            repo_root / "configs/fixed_reward_largebox.yaml",
            [f"environment.num_envs={protocol.num_envs}"],
        )
        args.headless = True
        args.device = cfg.environment.device
        app_launcher = AppLauncher(args)
        trainer = make_collection_trainer(
            app_launcher.app,
            repo_root=repo_root,
            protocol=protocol,
            runtime_dir=output_dir / "runtime/diag_15",
        )
        startup_digest = assert_startup_randomization_matches(trainer.env, bank)
        expected_platform = {
            "dataset_sha256": str(manifest["motion_sha256"]),
            "robot_asset_sha256": str(manifest["robot_asset_sha256"]),
            "action_schema_sha256": str(manifest["action_schema_sha256"]),
        }
        clean_by_update: dict[int, dict[str, Any]] = {}
        for row in rows:
            if row["collector_mode"] == CollectorMode.CLEAN_MEAN.value:
                clean_by_update.setdefault(int(row["checkpoint_update"]), row)
        if set(clean_by_update) != {record.update for record in checkpoints}:
            raise ProtocolError("clean rollout coverage differs from dense checkpoint coverage")

        checkpoint_results: dict[int, dict[str, Any]] = {}
        global_stored_error = 0.0
        global_internal_error = 0.0
        for checkpoint in checkpoints:
            switch_policy_state(
                trainer,
                checkpoint,
                expected_platform=expected_platform,
            )
            row = clean_by_update[checkpoint.update]
            shard = load_rollout_shard(resolve_shard_path(required["index"], row))
            observation = shard["observation"]["actor_full"]
            stored_mean = shard["action"]["mean"]
            if observation.shape[:2] != stored_mean.shape[:2]:
                raise ProtocolError("stored observation/action time axes differ")
            normalizer_before = state_digest(trainer.algo.actor_obs_normalizer.state_dict())
            stored_error = 0.0
            internal_error = 0.0
            count = 0
            # The production collector evaluates exactly one [num_envs, D]
            # batch per control step.  Flattening [T,N,D] into a much larger
            # batch changes CUDA GEMM reduction order and can introduce an
            # irrelevant ~1e-5 float32 difference.  Replay the original call
            # shape so this remains a true bit-exact path audit rather than a
            # numerical-tolerance test.
            for step in range(int(observation.shape[0])):
                batch = observation[step].to(trainer.env.device)
                expected = stored_mean[step].to(trainer.env.device)
                action = trainer.algo.deterministic_action(batch)
                stored_error = max(
                    stored_error,
                    float(torch.max(torch.abs(action - expected)).item()),
                )
                internal = deterministic_action_equivalence(
                    trainer.algo,
                    batch,
                    atol=0.0,
                )
                internal_error = max(internal_error, float(internal["max_abs_error"]))
                helper_mean, _ = policy_mean_and_std(trainer.algo, batch)
                internal_error = max(
                    internal_error,
                    float(torch.max(torch.abs(helper_mean - action)).item()),
                )
                count += int(batch.shape[0])
            normalizer_after = state_digest(trainer.algo.actor_obs_normalizer.state_dict())
            if normalizer_before != normalizer_after:
                raise ProtocolError("deterministic inference updated actor normalizer state")
            global_stored_error = max(global_stored_error, stored_error)
            global_internal_error = max(global_internal_error, internal_error)
            checkpoint_results[checkpoint.update] = {
                "sample_count": count,
                "stored_mean_vs_deterministic_max_abs_error": stored_error,
                "act_inference_path_internal_max_abs_error": internal_error,
                "normalizer_unchanged": True,
            }
        if global_stored_error != 0.0 or global_internal_error != 0.0:
            raise ProtocolError(
                "deterministic action equivalence is not exact: "
                f"stored={global_stored_error}, internal={global_internal_error}"
            )
        result = diagnostic_result(
            "15",
            PASS,
            summary="stored canonical means are bit-exact with production deterministic_action/act_inference",
            evidence={
                "atol": 0.0,
                "stored_mean_global_max_abs_error": global_stored_error,
                "internal_path_global_max_abs_error": global_internal_error,
                "startup_randomization_sha256": startup_digest,
                "single_persistent_environment": True,
                "checkpoint_switch": "policy_state_only",
                "checkpoint_results": checkpoint_results,
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "15", SKIPPED_DEPENDENCY, summary=str(exc), errors=[str(exc)]
        )
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, ProtocolError, json.JSONDecodeError) as exc:
        result = diagnostic_result(
            "15",
            INVALID_PROTOCOL,
            summary="deterministic action-path protocol is invalid",
            errors=[str(exc)],
        )
    finally:
        if trainer is not None:
            try:
                from diagnostics.common.canonical_collection import close_collection_trainer

                close_collection_trainer(trainer)
            except Exception:
                pass
    code = _write_status(result_path, result)
    return finish_isaac_entrypoint(
        code, isaac_launched=app_launcher is not None, allow_hard_exit=_ALLOW_HARD_EXIT
    )


if __name__ == "__main__":
    raise SystemExit(main())
