#!/usr/bin/env python3
"""Collect all four canonical modes for every frozen teacher checkpoint."""

from __future__ import annotations

import argparse
from collections import Counter
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
    sha256_file,
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
    _write_json_idempotent(path, result)
    print(f"[diag_12] {result['status']} {path}")
    return 0


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(nested) for nested in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json_idempotent(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        existing = read_json(path)
        if _plain(existing) != _plain(payload):
            raise ProtocolError(f"existing JSON artifact differs on resume: {path}")
        return
    write_json_exclusive(path, payload)


def _write_index_idempotent(rows: list[dict[str, Any]], path: Path) -> None:
    from diagnostics.common.canonical_collection import (
        load_rollout_index,
        write_rollout_index,
    )

    if path.is_file():
        actual = sorted(load_rollout_index(path), key=lambda row: str(row["sample_id"]))
        expected = sorted(rows, key=lambda row: str(row["sample_id"]))
        if _plain(actual) != _plain(expected):
            raise ProtocolError(f"existing canonical index differs on resume: {path}")
        return
    write_rollout_index(rows, path)


def _validation_payload(checkpoint: Any, metrics: dict[str, float], protocol: str, **extra):
    return {
        "checkpoint_sha256": checkpoint.sha256,
        "checkpoint_update": checkpoint.update,
        "protocol": protocol,
        **extra,
        "metrics": metrics,
    }


def _collect_legacy_a_amp(
    *,
    trainer: Any,
    bank: Any,
    protocol: Any,
    manifest: dict[str, Any],
    spec: dict[str, Any],
    repo_root: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from diagnostics.common.canonical_collection import (
        branch_shard_relative,
        collect_canonical_branch,
        index_rows_from_tree,
        load_rollout_shard,
        save_branch_shard,
        validate_canonical_rollout_tree,
    )
    from diagnostics.common.legacy_policy import AAMPPolicyAdapter

    index_path = output_dir / "canonical_A_amp_rollout_index.parquet"
    status_path = output_dir / "canonical_A_amp_rollout_index.status.json"
    try:
        checkpoint, adapter = AAMPPolicyAdapter.load(
            repo_root=repo_root,
            spec=spec,
            device=trainer.env.device,
        )
        rows_all: list[dict[str, Any]] = []
        generated = 0
        reused = 0
        domain_root = output_dir / "legacy/A_amp"
        for mode, sigma in protocol.branch_variants:
            relative_inside = branch_shard_relative(checkpoint, mode, sigma)
            destination = domain_root / relative_inside
            if destination.is_file():
                tree = load_rollout_shard(destination)
                validate_canonical_rollout_tree(tree)
                metadata = tree["metadata"]
                expected = {
                    "checkpoint_sha256": checkpoint.sha256,
                    "checkpoint_update": checkpoint.update,
                    "policy_domain": checkpoint.policy_domain,
                    "collector_mode": mode.value,
                    "collector_seed": protocol.collector_seed,
                }
                mismatches = {
                    key: {"expected": value, "actual": metadata.get(key)}
                    for key, value in expected.items()
                    if metadata.get(key) != value
                }
                if mismatches:
                    raise ProtocolError(
                        f"A_amp partial shard cannot be resumed: {mismatches}"
                    )
                rows = index_rows_from_tree(
                    tree,
                    checkpoint=checkpoint,
                    mode=mode,
                    common_sigma=sigma,
                    horizon=protocol.horizon,
                )
                reused += 1
            else:
                tree, rows, _ = collect_canonical_branch(
                    trainer,
                    bank,
                    checkpoint,
                    protocol=protocol,
                    manifest=manifest,
                    spec=spec,
                    repo_root=repo_root,
                    mode=mode,
                    common_sigma=sigma,
                    policy_adapter=adapter,
                )
                save_branch_shard(
                    tree,
                    output_dir=domain_root,
                    checkpoint=checkpoint,
                    mode=mode,
                    common_sigma=sigma,
                )
                generated += 1
            shard_relative_root = (
                Path("legacy/A_amp") / relative_inside
            ).as_posix()
            for row in rows:
                row["shard_path"] = shard_relative_root
            rows_all.extend(rows)
        expected_shards = len(protocol.branch_variants)
        if generated + reused != expected_shards:
            raise ProtocolError("A_amp branch cartesian product is incomplete")
        _write_index_idempotent(rows_all, index_path)
        result = diagnostic_result(
            "12.A_amp",
            PASS,
            summary="official H=1 AMP actor was recollected in the shared canonical environment",
            evidence={
                "policy_domain": checkpoint.policy_domain,
                "canonical_rollout_index": str(index_path),
                "canonical_rollout_index_sha256": sha256_file(index_path),
                "trajectory_count": len(rows_all),
                "shard_count": expected_shards,
                "collector_mode_counts": dict(
                    Counter(row["collector_mode"] for row in rows_all)
                ),
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "12.A_amp", SKIPPED_DEPENDENCY, summary=str(exc), errors=[str(exc)]
        )
        rows_all = []
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, ProtocolError) as exc:
        result = diagnostic_result(
            "12.A_amp",
            INVALID_PROTOCOL,
            summary="official A_amp policy could not be loaded exactly",
            errors=[str(exc)],
        )
        rows_all = []
    _write_json_idempotent(status_path, result)
    return result, rows_all


def _quarantine_legacy_a_mix(
    *, output_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Record the frozen exclusion without resolving any A_mix asset."""

    result = diagnostic_result(
        "12.A_mix",
        SKIPPED_DEPENDENCY,
        summary="obsolete FCAMP/H4 domain is quarantined from the current protocol",
        evidence={
            "policy_domain": "A_mix_tracking",
            "canonical_rollout_index": None,
            "approximation_used": False,
            "legacy_quarantined": True,
            "checkpoint_loaded": False,
            "reason": "obsolete FCAMP/H4 excluded by the current protocol",
        },
    )
    _write_json_idempotent(
        output_dir / "canonical_A_mix_rollout_index.status.json", result
    )
    return result, []


def main() -> int:
    try:
        from isaaclab.app import AppLauncher
    except (ImportError, ModuleNotFoundError) as exc:
        args = _base_parser().parse_args()
        output_dir = args.output_dir.expanduser().resolve()
        result = diagnostic_result(
            "12",
            SKIPPED_DEPENDENCY,
            summary="Isaac Lab is unavailable; canonical rollout tensors were not fabricated",
            errors=[str(exc)],
        )
        return _write_status(output_dir / "canonical_rollout_index.status.json", result)

    parser = _base_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    status_path = output_dir / "canonical_rollout_index.status.json"
    index_path = output_dir / "canonical_rollout_index.parquet"
    app_launcher = None
    trainer = None
    try:
        from diagnostics.common.canonical_collection import (
            CanonicalCollectionProtocol,
            assert_startup_randomization_matches,
            branch_shard_relative,
            close_collection_trainer,
            collect_canonical_branch,
            index_rows_from_tree,
            load_rollout_shard,
            make_collection_trainer,
            read_dense_checkpoint_records,
            save_branch_shard,
            summarize_rollout_tree,
            validate_canonical_rollout_tree,
            validate_frozen_collection_semantics,
        )
        from diagnostics.common.noise_bank import CollectorMode
        from diagnostics.common.snapshot_bank import SnapshotBank

        spec = load_spec(args.spec)
        protocol = CanonicalCollectionProtocol.from_spec(spec)
        frozen = validate_frozen_collection_semantics(spec, protocol)
        prerequisites = {
            "manifest": output_dir / "manifest.json",
            "dense": output_dir / "checkpoints/dense_checkpoint_inventory.status.json",
            "snapshot": output_dir / "snapshot_bank.status.json",
        }
        missing = [name for name, path in prerequisites.items() if not path.is_file()]
        if missing:
            raise DependencyUnavailable(f"stage-1 prerequisite artifacts are missing: {missing}")
        manifest = read_json(prerequisites["manifest"])
        dense_status = read_json(prerequisites["dense"])
        snapshot_status = read_json(prerequisites["snapshot"])
        if any(value.get("status") != PASS for value in (manifest, dense_status, snapshot_status)):
            raise DependencyUnavailable("diag_00, diag_10, and diag_11 must PASS")
        bank_path = output_dir / "snapshot_bank.pt"
        if not bank_path.is_file():
            raise DependencyUnavailable("snapshot_bank.pt is missing")
        if sha256_file(bank_path) != snapshot_status["evidence"]["snapshot_bank_sha256"]:
            raise ProtocolError("snapshot bank hash differs from diag_11")
        bank = SnapshotBank.load(bank_path)
        checkpoints = read_dense_checkpoint_records(
            output_dir / "checkpoints/dense_checkpoint_inventory.csv"
        )
        if tuple(record.update for record in checkpoints) != protocol.canonical_checkpoint_updates:
            raise ProtocolError("dense inventory does not exactly cover canonical updates")
        if len({record.lineage_id for record in checkpoints}) != 1:
            raise ProtocolError("canonical checkpoints do not share one verified lineage")

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
            runtime_dir=output_dir / "runtime/diag_12",
        )
        startup_digest = assert_startup_randomization_matches(trainer.env, bank)

        all_rows: list[dict[str, Any]] = []
        summaries: dict[tuple[int, str, float], dict[str, float]] = {}
        reused_shards = 0
        generated_shards = 0
        for checkpoint in checkpoints:
            for mode, sigma in protocol.branch_variants:
                relative = branch_shard_relative(checkpoint, mode, sigma)
                destination = output_dir / relative
                if destination.is_file():
                    tree = load_rollout_shard(destination)
                    validate_canonical_rollout_tree(tree)
                    metadata = tree["metadata"]
                    expected = {
                        "checkpoint_sha256": checkpoint.sha256,
                        "checkpoint_update": checkpoint.update,
                        "checkpoint_lineage_id": checkpoint.lineage_id,
                        "collector_mode": mode.value,
                        "collector_seed": protocol.collector_seed,
                        "source_snapshot_sha256": manifest["source_snapshot_sha256"],
                    }
                    mismatches = {
                        key: {"expected": value, "actual": metadata.get(key)}
                        for key, value in expected.items()
                        if metadata.get(key) != value
                    }
                    if mismatches:
                        raise ProtocolError(
                            f"partial rollout shard cannot be resumed: {relative}: {mismatches}"
                        )
                    rows = index_rows_from_tree(
                        tree,
                        checkpoint=checkpoint,
                        mode=mode,
                        common_sigma=sigma,
                        horizon=protocol.horizon,
                    )
                    summary = summarize_rollout_tree(tree)
                    reused_shards += 1
                else:
                    tree, rows, summary = collect_canonical_branch(
                        trainer,
                        bank,
                        checkpoint,
                        protocol=protocol,
                        manifest=manifest,
                        spec=spec,
                        repo_root=repo_root,
                        mode=mode,
                        common_sigma=sigma,
                    )
                    _, saved_relative = save_branch_shard(
                        tree,
                        output_dir=output_dir,
                        checkpoint=checkpoint,
                        mode=mode,
                        common_sigma=sigma,
                    )
                    if saved_relative != relative.as_posix():
                        raise ProtocolError("canonical shard naming changed during save")
                    generated_shards += 1
                for row in rows:
                    row["shard_path"] = relative.as_posix()
                all_rows.extend(rows)
                summaries[(checkpoint.update, mode.value, float(sigma))] = summary
                print(
                    f"[diag_12] checkpoint={checkpoint.update} mode={mode.value} "
                    f"sigma={sigma:g} trajectories={len(rows)}",
                    flush=True,
                )

        expected_shards = len(checkpoints) * len(protocol.branch_variants)
        if generated_shards + reused_shards != expected_shards:
            raise ProtocolError("canonical shard count differs from the frozen cartesian product")
        native_rows = [
            row for row in all_rows if row["collector_mode"] == CollectorMode.NATIVE_STOCHASTIC.value
        ]
        if any(bool(row["eligible_for_primary_overlap"]) for row in native_rows):
            raise ProtocolError("native_stochastic entered primary-overlap eligibility")
        _write_index_idempotent(all_rows, index_path)

        validation = spec["collection"]["validation_protocols"]
        summary_update = int(validation["summary_checkpoint_update"])
        primary = next(record for record in checkpoints if record.update == summary_update)
        controlled_sigma = float(validation["controlled_noise_sigma"])
        validation_dir = output_dir / "validation_protocols"
        _write_json_idempotent(
            validation_dir / "clean_mean.json",
            _validation_payload(
                primary,
                summaries[(summary_update, CollectorMode.CLEAN_MEAN.value, 0.0)],
                "clean_mean",
            ),
        )
        _write_json_idempotent(
            validation_dir / "training_like.json",
            _validation_payload(
                primary,
                summaries[(summary_update, CollectorMode.NATIVE_STOCHASTIC.value, 0.0)],
                "training_like",
                source_collector_mode=CollectorMode.NATIVE_STOCHASTIC.value,
                stochastic_coupling="shared_epsilon_times_checkpoint_learned_std",
            ),
        )
        _write_json_idempotent(
            validation_dir / "controlled_noise.json",
            _validation_payload(
                primary,
                summaries[
                    (summary_update, CollectorMode.COMMON_ACTION_NOISE.value, controlled_sigma)
                ],
                "controlled_noise",
                source_collector_mode=CollectorMode.COMMON_ACTION_NOISE.value,
                common_sigma=controlled_sigma,
            ),
        )

        legacy_a_amp_result, legacy_a_amp_rows = _collect_legacy_a_amp(
            trainer=trainer,
            bank=bank,
            protocol=protocol,
            manifest=manifest,
            spec=spec,
            repo_root=repo_root,
            output_dir=output_dir,
        )
        legacy_a_mix_result, legacy_a_mix_rows = _quarantine_legacy_a_mix(
            output_dir=output_dir
        )

        counts = Counter(row["collector_mode"] for row in all_rows)
        result = diagnostic_result(
            "12",
            PASS,
            summary="condition-matched canonical teacher rollouts were collected in one environment",
            evidence={
                "canonical_rollout_index": str(index_path),
                "canonical_rollout_index_sha256": sha256_file(index_path),
                "checkpoint_count": len(checkpoints),
                "snapshot_count": len(bank),
                "trajectory_count": len(all_rows),
                "shard_count": expected_shards,
                "generated_shards": generated_shards,
                "reused_verified_shards": reused_shards,
                "collector_mode_counts": dict(counts),
                "native_stochastic_primary_overlap_rows": 0,
                "startup_randomization_sha256": startup_digest,
                "frozen_semantics": frozen,
                "validation_summary_checkpoint_update": summary_update,
                "legacy_domains": {
                    "A_amp_official": {
                        "status": legacy_a_amp_result["status"],
                        "trajectory_count": len(legacy_a_amp_rows),
                        "index": (
                            str(output_dir / "canonical_A_amp_rollout_index.parquet")
                            if legacy_a_amp_result["status"] == PASS
                            else None
                        ),
                    },
                    "A_mix_tracking": {
                        "status": legacy_a_mix_result["status"],
                        "trajectory_count": len(legacy_a_mix_rows),
                        "index": None,
                        "legacy_quarantined": True,
                        "checkpoint_loaded": False,
                    },
                },
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "12", SKIPPED_DEPENDENCY, summary=str(exc), errors=[str(exc)]
        )
    except (FileNotFoundError, KeyError, ValueError, RuntimeError, ProtocolError, json.JSONDecodeError) as exc:
        result = diagnostic_result(
            "12",
            INVALID_PROTOCOL,
            summary="canonical rollout collection protocol is invalid",
            errors=[str(exc)],
        )
    finally:
        if trainer is not None:
            try:
                from diagnostics.common.canonical_collection import close_collection_trainer

                close_collection_trainer(trainer)
            except Exception:
                pass
    code = _write_status(status_path, result)
    return finish_isaac_entrypoint(
        code, isaac_launched=app_launcher is not None, allow_hard_exit=_ALLOW_HARD_EXIT
    )


if __name__ == "__main__":
    raise SystemExit(main())
