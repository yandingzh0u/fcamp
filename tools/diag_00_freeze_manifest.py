#!/usr/bin/env python3
"""Freeze the complete stage-0 experiment identity without launching Isaac."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.checkpoint_io import (
    assign_lineages,
    discover_checkpoints,
    summarize_checkpoint,
)
from diagnostics.common.manifest import (
    INVALID_PROTOCOL,
    PASS,
    action_names_from_source,
    canonical_sha256,
    complete_manifest_errors,
    create_source_snapshot,
    git_provenance,
    load_config_tree,
    load_spec,
    recompute_resolved_config_sha256,
    read_json,
    resolve_path,
    resolve_task_motion,
    sha256_file,
    spec_value,
    write_json_exclusive,
)


def _freeze_legacy_domain_inputs(
    spec: dict[str, Any],
    *,
    repo_root: Path,
    motion_sha256: str,
    robot_asset_sha256: str,
    action_schema_sha256: str,
) -> tuple[dict[str, Any], list[str]]:
    """Hash every optional legacy domain entity and verify task identity."""

    raw_manifest = spec_value(spec, "inputs.legacy_domain_manifest", default=None)
    raw_checkpoints = spec_value(spec, "inputs.domain_checkpoints", default={})
    raw_snapshots = spec_value(spec, "inputs.domain_source_snapshots", default={})
    if raw_manifest is None and not raw_checkpoints and not raw_snapshots:
        return {}, []
    errors: list[str] = []
    if not isinstance(raw_checkpoints, dict) or not isinstance(raw_snapshots, dict):
        return {}, ["legacy domain checkpoint/source mappings must be objects"]
    if set(raw_checkpoints) != set(raw_snapshots) or not raw_checkpoints:
        return {}, ["legacy domain checkpoint/source names are missing or inconsistent"]
    manifest_path = resolve_path(raw_manifest, base=repo_root)
    if manifest_path is None or not manifest_path.is_file():
        return {}, [f"legacy domain manifest is unavailable: {manifest_path}"]
    declared = read_json(manifest_path)
    if not isinstance(declared, dict):
        return {}, ["legacy domain manifest must be an object"]
    shared = declared.get("shared_task_identity", {})
    expected_shared = {
        "motion_sha256": motion_sha256,
        "robot_asset_sha256": robot_asset_sha256,
        "action_schema_sha256": action_schema_sha256,
    }
    for field, expected in expected_shared.items():
        if not isinstance(shared, dict) or shared.get(field) != expected:
            errors.append(f"legacy domains use a different {field}")
    domains: dict[str, Any] = {}
    declared_domains = declared.get("domains", {})
    for name in sorted(raw_checkpoints):
        checkpoint = resolve_path(raw_checkpoints[name], base=repo_root)
        snapshot = resolve_path(raw_snapshots[name], base=repo_root)
        if checkpoint is None or not checkpoint.is_file():
            errors.append(f"legacy domain {name} checkpoint is unavailable: {checkpoint}")
            continue
        if snapshot is None or not snapshot.is_file():
            errors.append(f"legacy domain {name} source snapshot is unavailable: {snapshot}")
            continue
        declared_domain = (
            declared_domains.get(name, {}) if isinstance(declared_domains, dict) else {}
        )
        checkpoint_hash = sha256_file(checkpoint)
        snapshot_hash = sha256_file(snapshot)
        if not isinstance(declared_domain, dict):
            errors.append(f"legacy domain {name} manifest entry is not an object")
        else:
            if declared_domain.get("checkpoint_sha256") != checkpoint_hash:
                errors.append(f"legacy domain {name} checkpoint hash mismatch")
            if declared_domain.get("source_snapshot_sha256") != snapshot_hash:
                errors.append(f"legacy domain {name} source snapshot hash mismatch")
        domains[name] = {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "source_snapshot_path": str(snapshot),
            "source_snapshot_sha256": snapshot_hash,
            "git_commit": declared_domain.get("git_commit") if isinstance(declared_domain, dict) else None,
            "git_dirty": declared_domain.get("git_dirty") if isinstance(declared_domain, dict) else None,
            "checkpoint_update": declared_domain.get("checkpoint_update") if isinstance(declared_domain, dict) else None,
        }
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "domains": domains,
    }, errors


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--suite-id", default=None)
    return parser.parse_args()


def _select_checkpoint(repo_root: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        value = explicit.expanduser()
        if not value.is_absolute():
            value = repo_root / value
        return value.resolve()
    candidates = discover_checkpoints(repo_root)
    named = []
    for path in candidates:
        from diagnostics.common.checkpoint_io import update_from_filename

        update = update_from_filename(path)
        if update is not None:
            named.append((update, path))
    if named:
        return max(named, key=lambda item: (item[0], str(item[1])))[1]
    if candidates:
        return candidates[-1]
    raise FileNotFoundError("no checkpoint entity was found under runs/*/checkpoints")


def main() -> int:
    args = _arguments()
    repo_root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    spec_base = Path(spec.get("_spec_path", repo_root)).parent if spec else repo_root
    output_dir_value = args.output_dir or spec_value(
        spec, "output_dir", default="output/largebox_discovery_v1"
    )
    output_dir = resolve_path(output_dir_value, base=repo_root)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    snapshot_path = output_dir / "source_snapshot.tar.gz"
    if manifest_path.exists() or snapshot_path.exists():
        raise FileExistsError(
            "diag_00 is immutable and refuses to replace an existing manifest "
            f"or source snapshot in {output_dir}"
        )

    spec_checkpoint = spec_value(
        spec,
        "inputs.primary_checkpoint",
        "checkpoint_path",
        default=None,
    )
    checkpoint_argument = args.checkpoint or (
        Path(str(spec_checkpoint)) if spec_checkpoint is not None else None
    )
    checkpoint_path = _select_checkpoint(repo_root, checkpoint_argument)
    checkpoint_record = assign_lineages([summarize_checkpoint(checkpoint_path)])[0]
    spec_config = spec_value(
        spec,
        "inputs.primary_config",
        "inputs.config",
        "config_path",
        default=None,
    )
    config_path = args.config
    if config_path is None and checkpoint_record.get("resolved_config_path"):
        config_path = Path(str(checkpoint_record["resolved_config_path"]))
    if config_path is None and spec_config:
        config_path = Path(str(spec_config))
    if config_path is None:
        config_path = repo_root / "configs" / "fixed_reward_largebox.yaml"
    if not config_path.is_absolute():
        config_path = (spec_base / config_path).resolve()
    config = load_config_tree(config_path)
    task_name = str(config.get("environment", {}).get("task", ""))
    motion_path = Path(
        config.get("dataset_path") or resolve_task_motion(repo_root, task_name)
    ).expanduser().resolve()
    robot_path = Path(
        config.get("robot_asset_path")
        or repo_root / "assets" / "robots" / "holosoma_g1" / "g1_29dof.urdf"
    ).expanduser().resolve()
    action_names = action_names_from_source(repo_root)
    action_schema_sha256 = canonical_sha256(action_names)
    resolved_config_sha256 = recompute_resolved_config_sha256(config)
    snapshot = create_source_snapshot(repo_root, snapshot_path)
    provenance = git_provenance(repo_root)

    suite_id = str(
        args.suite_id
        or spec_value(spec, "suite_id", default="largebox_discovery_v1")
    )
    manifest: dict[str, Any] = {
        "diagnostic_id": "00",
        "status": PASS,
        "suite_id": suite_id,
        "suite_version": str(spec_value(spec, "suite_version", default="1.0.0")),
        **provenance,
        "source_snapshot": str(snapshot_path),
        "source_snapshot_sha256": snapshot["sha256"],
        "source_snapshot_file_count": snapshot["file_count"],
        "source_members_sha256": snapshot["members_sha256"],
        "resolved_config_path": str(config_path),
        "resolved_config_sha256": resolved_config_sha256,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
        "checkpoint_update": checkpoint_record["update_payload"],
        "checkpoint_lineage_id": checkpoint_record["checkpoint_lineage_id"],
        "checkpoint_schema_version": checkpoint_record["schema_version"],
        "robot_asset_path": str(robot_path),
        "robot_asset_sha256": sha256_file(robot_path),
        "motion_path": str(motion_path),
        "motion_sha256": sha256_file(motion_path),
        "action_schema_sha256": action_schema_sha256,
        "action_dim": len(action_names),
        "task_name": task_name,
        "method": config.get("method"),
        "collector_mode": "stage0_identity_audit",
        "collector_seed": config.get("training", {}).get("seed"),
        "training_source_snapshot_sha256": config.get("source_snapshot_sha256"),
        "training_git_commit": config.get("git_commit"),
    }
    mismatches: list[str] = []
    legacy_inputs, legacy_errors = _freeze_legacy_domain_inputs(
        spec,
        repo_root=repo_root,
        motion_sha256=manifest["motion_sha256"],
        robot_asset_sha256=manifest["robot_asset_sha256"],
        action_schema_sha256=manifest["action_schema_sha256"],
    )
    manifest["legacy_domain_inputs"] = legacy_inputs
    mismatches.extend(legacy_errors)
    declared_config_hash = config.get("resolved_config_sha256")
    if declared_config_hash and declared_config_hash != resolved_config_sha256:
        mismatches.append(
            "resolved config content does not match its declared SHA-256"
        )
    expected_hashes = {
        "motion_sha256": checkpoint_record.get("dataset_sha256"),
        "robot_asset_sha256": checkpoint_record.get("robot_asset_sha256"),
        "action_schema_sha256": checkpoint_record.get("action_schema_sha256"),
    }
    for field, expected in expected_hashes.items():
        if expected and expected != manifest[field]:
            mismatches.append(
                f"{field} differs from checkpoint platform_identity: "
                f"actual={manifest[field]}, checkpoint={expected}"
            )
    if suite_id.startswith("largebox") and (
        task_name != "largebox_plane" or "largebox" not in motion_path.name.lower()
    ):
        mismatches.append(
            "largebox suite identity does not match task/motion; spin-kick and "
            "largebox evidence must never be mixed"
        )
    if suite_id.startswith("spinkick") and "largebox" in motion_path.name.lower():
        mismatches.append("spin-kick suite points at a largebox motion")
    mismatches.extend(complete_manifest_errors(manifest))
    manifest["errors"] = mismatches
    manifest["warnings"] = (
        ["working tree is dirty; the complete executable source snapshot is authoritative"]
        if manifest["git_dirty"]
        else []
    )
    manifest["status"] = INVALID_PROTOCOL if mismatches else PASS
    write_json_exclusive(manifest_path, manifest)
    print(f"[diag_00] {manifest['status']} {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
