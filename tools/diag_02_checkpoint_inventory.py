#!/usr/bin/env python3
"""Inventory physical checkpoints and audit their recoverable lineage state."""

from __future__ import annotations

import argparse
import json
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
    diagnostic_result,
    load_spec,
    read_json,
    resolve_path,
    spec_value,
    write_csv_exclusive,
    write_json_exclusive,
)


CSV_FIELDS = (
    "path",
    "run_dir",
    "checkpoint_sha256",
    "size_bytes",
    "update_filename",
    "update_payload",
    "update_match",
    "hardlink_alias_of",
    "content_alias_of",
    "method",
    "seed",
    "schema_version",
    "checkpoint_lineage_id",
    "checkpoint_branch_id",
    "lineage_coherent",
    "lineage_parent_resolved",
    "lineage_evidence",
    "duplicate_update",
    "semantic_config_sha256",
    "source_snapshot_sha256",
    "policy_state_sha256",
    "optimizer_present",
    "optimizer_state_count",
    "optimizer_state_sha256",
    "critic_optimizer_present",
    "normalizer_actor_present",
    "normalizer_critic_present",
    "normalizer_actor_count",
    "normalizer_critic_count",
    "torch_rng_present",
    "torch_rng_sha256",
    "cuda_rng_present",
    "cuda_rng_sha256",
    "sampler_present",
    "sampler_version",
    "sampler_state_sha256",
    "env_transitions_total",
    "actor_optimizer_steps_total",
    "critic_optimizer_steps_total",
    "dataset_sha256",
    "robot_asset_sha256",
    "action_schema_sha256",
    "resume_path",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--checkpoint-glob", action="append", default=[])
    return parser.parse_args()


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def main() -> int:
    args = _arguments()
    repo_root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = resolve_path(
        args.output_dir or spec_value(spec, "output_dir", default="output/largebox_discovery_v1"),
        base=repo_root,
    )
    assert output_dir is not None
    csv_path = output_dir / "checkpoint_inventory.csv"
    status_path = output_dir / "checkpoint_inventory.status.json"
    try:
        manifest = read_json(output_dir / "manifest.json")
        identity = read_json(output_dir / "task_identity.json")
        if manifest.get("status") != PASS or identity.get("status") != PASS:
            raise ValueError("diag_00 and diag_01 must both PASS before lineage audit")
        configured_globs = args.checkpoint_glob or list(
            spec_value(spec, "inputs.checkpoint_globs", default=[])
        )
        paths = discover_checkpoints(repo_root, configured_globs)
        if not paths:
            raise DependencyUnavailable("no physical checkpoint files were found")
        records = inventory_checkpoints(paths)
        unique_records = [row for row in records if row.get("content_alias_of") is None]
        expected_platform = {
            "dataset_sha256": manifest["motion_sha256"],
            "robot_asset_sha256": manifest["robot_asset_sha256"],
            "action_schema_sha256": manifest["action_schema_sha256"],
        }
        errors: list[str] = []
        for row in unique_records:
            label = row["path"]
            if not row["update_match"]:
                errors.append(f"filename/payload update mismatch: {label}")
            if row["method"] != "fixed_reward" or row["schema_version"] != 17:
                errors.append(f"unexpected method/schema: {label}")
            for key in (
                "optimizer_present",
                "critic_optimizer_present",
                "normalizer_actor_present",
                "normalizer_critic_present",
                "torch_rng_present",
                "sampler_present",
                "lineage_coherent",
            ):
                if not row[key]:
                    errors.append(f"{key}=false: {label}")
            if row["duplicate_update"]:
                errors.append(f"different checkpoint payloads claim the same update: {label}")
            for key, expected in expected_platform.items():
                if row.get(key) != expected:
                    errors.append(f"{key} mismatches frozen manifest: {label}")
        lineage_ids = sorted({row["checkpoint_lineage_id"] for row in unique_records})
        frozen_lineage_present = manifest["checkpoint_lineage_id"] in lineage_ids
        if not frozen_lineage_present:
            errors.append("the checkpoint frozen by diag_00 is absent from inventory lineage")
        updates = sorted(
            {
                int(row["update_payload"])
                for row in unique_records
                if type(row.get("update_payload")) is int
            }
        )
        warnings: list[str] = []
        if updates and max(updates) < 500:
            warnings.append(
                f"physical checkpoint archive ends at update {max(updates)}; "
                "later behavior may exist only in logs and cannot be resumed"
            )
        rows = [{key: _csv_value(row.get(key)) for key in CSV_FIELDS} for row in records]
        write_csv_exclusive(csv_path, rows, fieldnames=CSV_FIELDS)
        result = diagnostic_result(
            "02",
            PASS if not errors else INVALID_PROTOCOL,
            summary=(
                f"inventoried {len(unique_records)} unique checkpoint payloads "
                f"at updates {updates}"
            ),
            evidence={
                "physical_file_count": len(records),
                "unique_payload_count": len(unique_records),
                "hardlink_alias_count": sum(row.get("hardlink_alias_of") is not None for row in records),
                "available_updates": updates,
                "lineage_ids": lineage_ids,
                "frozen_lineage_present": frozen_lineage_present,
                "inventory_csv": str(csv_path),
                "checkpoint_globs": configured_globs or ["runs/*/checkpoints/*.pt"],
            },
            errors=errors,
            warnings=warnings,
        )
    except DependencyUnavailable as exc:
        write_csv_exclusive(csv_path, [], fieldnames=CSV_FIELDS)
        result = diagnostic_result(
            "02", SKIPPED_DEPENDENCY, summary=str(exc), errors=[str(exc)]
        )
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        write_csv_exclusive(csv_path, [], fieldnames=CSV_FIELDS)
        result = diagnostic_result(
            "02", INVALID_PROTOCOL, summary="checkpoint inventory protocol is invalid", errors=[str(exc)]
        )
    write_json_exclusive(status_path, result)
    print(f"[diag_02] {result['status']} {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
