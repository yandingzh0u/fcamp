#!/usr/bin/env python3
"""Audit the dense teacher archive and emit exact deterministic recovery commands."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.checkpoint_io import discover_checkpoints, inventory_checkpoints
from diagnostics.common.manifest import (
    DependencyUnavailable,
    INVALID_PROTOCOL,
    PASS,
    SKIPPED_DEPENDENCY,
    ProtocolError,
    diagnostic_result,
    load_spec,
    read_json,
    resolve_path,
    spec_value,
    write_csv_exclusive,
    write_json_exclusive,
)


CSV_FIELDS = (
    "required_update",
    "present",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_lineage_id",
    "checkpoint_branch_id",
    "lineage_coherent",
    "source_run_dir",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--checkpoint-glob", action="append", default=[])
    return parser.parse_args()


def _recovery_commands(repo_root: Path) -> list[list[str]]:
    original = (
        repo_root
        / "runs/fixed_reward_holosoma_geometry_ckpts_u200_s42_20260802_020905/"
        "checkpoints/update_0200.pt"
    )
    first_run = "largebox_discovery_dense_u200_u300_s42_20260804"
    first_300 = repo_root / "runs" / first_run / "checkpoints/update_0300.pt"
    return [
        [
            sys.executable,
            "train.py",
            "--config",
            "configs/fixed_reward_largebox.yaml",
            "--run_name",
            first_run,
            "--set",
            f"training.resume={original}",
            "--set",
            "training.max_updates=300",
            "--set",
            "training.save_every=2",
            "--set",
            "training.validation_every=5",
            "--set",
            "training.reset_optimizer_on_resume=false",
            "--set",
            "training.reset_sampler_on_resume=false",
        ],
        [
            sys.executable,
            "train.py",
            "--config",
            "configs/fixed_reward_largebox.yaml",
            "--run_name",
            "largebox_discovery_dense_u300_u500_s42_20260804",
            "--set",
            f"training.resume={first_300}",
            "--set",
            "training.max_updates=500",
            "--set",
            "training.save_every=5",
            "--set",
            "training.validation_every=5",
            "--set",
            "training.reset_optimizer_on_resume=false",
            "--set",
            "training.reset_sampler_on_resume=false",
        ],
    ]


def _select_records(
    records: list[dict[str, Any]],
    protocol: Any,
    manifest: dict[str, Any],
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    unique = [row for row in records if row.get("content_alias_of") is None]
    platform = {
        "dataset_sha256": manifest["motion_sha256"],
        "robot_asset_sha256": manifest["robot_asset_sha256"],
        "action_schema_sha256": manifest["action_schema_sha256"],
    }
    collection = spec.get("collection")
    if not isinstance(collection, dict):
        raise ProtocolError("spec.collection is missing")
    preferences = collection.get("canonical_run_preferences")
    audit_patterns = collection.get("audit_only_run_patterns")
    if not isinstance(preferences, list) or not preferences:
        raise ProtocolError("canonical_run_preferences must be a nonempty frozen list")
    if not isinstance(audit_patterns, list):
        raise ProtocolError("audit_only_run_patterns must be a frozen list")

    def run_name(row: dict[str, Any]) -> str:
        return Path(str(row["run_dir"])).name

    def audit_only(row: dict[str, Any]) -> bool:
        name = run_name(row)
        return any(str(pattern) in name for pattern in audit_patterns)

    compatible = [
        row
        for row in unique
        if row.get("method") == "fixed_reward"
        and all(row.get(key) == value for key, value in platform.items())
        and run_name(row) in {str(value) for value in preferences}
        and not audit_only(row)
    ]
    output: list[dict[str, Any]] = []
    errors: list[str] = []
    selected_lineage: str | None = None
    for update in protocol.canonical_checkpoint_updates:
        candidates = [row for row in compatible if row.get("update_payload") == update]
        candidates.sort(key=lambda row: preferences.index(run_name(row)))
        by_hash = {str(row["checkpoint_sha256"]): row for row in candidates}
        if len(by_hash) > 1:
            errors.append(
                f"update {update} has multiple different compatible payloads: "
                f"{sorted(by_hash)}"
            )
            selected = None
        elif by_hash:
            selected = candidates[0]
            lineage = str(selected["checkpoint_lineage_id"])
            if selected_lineage is None:
                selected_lineage = lineage
            elif lineage != selected_lineage:
                errors.append(
                    f"canonical archive crosses lineage at update {update}: "
                    f"{selected_lineage} vs {lineage}"
                )
            if not bool(selected.get("lineage_coherent")):
                errors.append(f"update {update} has incoherent lineage provenance")
        else:
            selected = None
        output.append(
            {
                "required_update": update,
                "present": selected is not None,
                "checkpoint_path": "" if selected is None else selected["path"],
                "checkpoint_sha256": "" if selected is None else selected["checkpoint_sha256"],
                "checkpoint_lineage_id": "" if selected is None else selected["checkpoint_lineage_id"],
                "checkpoint_branch_id": "" if selected is None else selected.get("checkpoint_branch_id", ""),
                "lineage_coherent": False if selected is None else selected["lineage_coherent"],
                "source_run_dir": "" if selected is None else selected["run_dir"],
            }
        )
    return output, errors


def main() -> int:
    args = _arguments()
    repo_root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = resolve_path(args.output_dir, base=repo_root)
    assert output_dir is not None
    csv_path = output_dir / "checkpoints/dense_checkpoint_inventory.csv"
    status_path = output_dir / "checkpoints/dense_checkpoint_inventory.status.json"
    rows: list[dict[str, Any]] = []
    try:
        try:
            from diagnostics.common.canonical_collection import CanonicalCollectionProtocol
        except ImportError as exc:
            raise DependencyUnavailable(
                "PyTorch is required to audit physical checkpoint payloads"
            ) from exc
        protocol = CanonicalCollectionProtocol.from_spec(spec)
        auto_execute = bool(spec_value(spec, "collection.dense_training_auto_execute", default=False))
        if auto_execute:
            raise ProtocolError(
                "stage-1 implementation is audit-only: dense training must be launched as "
                "a separately logged CoreTrainer job, never hidden inside an audit entrypoint"
            )
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.is_file():
            raise DependencyUnavailable("diag_00 manifest is not available")
        manifest = read_json(manifest_path)
        if manifest.get("status") != PASS:
            raise DependencyUnavailable("diag_00 did not PASS")
        checkpoint_globs = list(args.checkpoint_glob)
        if not checkpoint_globs:
            frozen_globs = (spec.get("inputs") or {}).get("checkpoint_globs")
            if not isinstance(frozen_globs, list) or not frozen_globs:
                raise ProtocolError(
                    "inputs.checkpoint_globs must freeze the teacher-only inventory"
                )
            checkpoint_globs = [str(value) for value in frozen_globs]
        paths = discover_checkpoints(repo_root, checkpoint_globs)
        if not paths:
            raise DependencyUnavailable("no checkpoint files are available")
        inventory = inventory_checkpoints(paths)
        rows, errors = _select_records(inventory, protocol, manifest, spec)
        missing = [row["required_update"] for row in rows if not row["present"]]
        write_csv_exclusive(csv_path, rows, fieldnames=CSV_FIELDS)
        if errors:
            result = diagnostic_result(
                "10",
                INVALID_PROTOCOL,
                summary="dense checkpoint archive violates lineage or uniqueness protocol",
                evidence={
                    "required_updates": list(protocol.canonical_checkpoint_updates),
                    "missing_updates": missing,
                    "recovery_commands": _recovery_commands(repo_root),
                },
                errors=errors,
            )
        elif missing:
            result = diagnostic_result(
                "10",
                SKIPPED_DEPENDENCY,
                summary="dense teacher recovery is still running or required checkpoints are absent",
                evidence={
                    "required_updates": list(protocol.canonical_checkpoint_updates),
                    "present_updates": [row["required_update"] for row in rows if row["present"]],
                    "missing_updates": missing,
                    "auto_execute": False,
                    "recovery_commands": _recovery_commands(repo_root),
                },
                errors=[f"missing physical checkpoint updates: {missing}"],
            )
        else:
            result = diagnostic_result(
                "10",
                PASS,
                summary="all frozen canonical teacher checkpoints exist in one coherent lineage",
                evidence={
                    "required_updates": list(protocol.canonical_checkpoint_updates),
                    "checkpoint_count": len(rows),
                    "checkpoint_lineage_id": rows[0]["checkpoint_lineage_id"],
                    "inventory_csv": str(csv_path),
                },
            )
    except DependencyUnavailable as exc:
        write_csv_exclusive(csv_path, rows, fieldnames=CSV_FIELDS)
        result = diagnostic_result(
            "10", SKIPPED_DEPENDENCY, summary=str(exc), errors=[str(exc)]
        )
    except (FileNotFoundError, KeyError, ValueError, ProtocolError) as exc:
        write_csv_exclusive(csv_path, rows, fieldnames=CSV_FIELDS)
        result = diagnostic_result(
            "10",
            INVALID_PROTOCOL,
            summary="dense checkpoint audit protocol is invalid",
            errors=[str(exc)],
        )
    write_json_exclusive(status_path, result)
    print(f"[diag_10] {result['status']} {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
