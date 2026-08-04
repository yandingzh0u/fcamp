#!/usr/bin/env python3
"""Audit task, motion, robot DoFs, and the exact action ordering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.checkpoint_io import load_checkpoint
from diagnostics.common.manifest import (
    DependencyUnavailable,
    INVALID_PROTOCOL,
    PASS,
    SKIPPED_DEPENDENCY,
    action_names_from_source,
    canonical_sha256,
    diagnostic_result,
    load_spec,
    read_json,
    resolve_path,
    sha256_file,
    spec_value,
    write_json_exclusive,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def _actor_action_dim(payload: dict) -> int | None:
    policy = payload.get("policy", {})
    for key in ("actor.std", "actor.log_std"):
        tensor = policy.get(key)
        if tensor is not None and hasattr(tensor, "numel"):
            return int(tensor.numel())
    return None


def main() -> int:
    args = _arguments()
    repo_root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = resolve_path(
        args.output_dir or spec_value(spec, "output_dir", default="output/largebox_discovery_v1"),
        base=repo_root,
    )
    assert output_dir is not None
    manifest_path = (args.manifest or output_dir / "manifest.json").expanduser().resolve()
    target = output_dir / "task_identity.json"
    try:
        manifest = read_json(manifest_path)
        action_names = action_names_from_source(repo_root)
        urdf_path = Path(manifest["robot_asset_path"]).resolve()
        motion_path = Path(manifest["motion_path"]).resolve()
        checkpoint_path = Path(manifest["checkpoint_path"]).resolve()
        urdf_root = ET.parse(urdf_path).getroot()
        actuated = [
            str(joint.attrib["name"])
            for joint in urdf_root.findall("joint")
            if joint.attrib.get("type") != "fixed"
        ]
        try:
            import numpy as np
        except ImportError as exc:
            raise DependencyUnavailable("NumPy is required to inspect the motion npz") from exc
        with np.load(motion_path, allow_pickle=False) as motion:
            required = {
                "fps",
                "joint_names",
                "body_names",
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
            }
            missing_motion_keys = sorted(required - set(motion.files))
            motion_joint_names = [str(value) for value in motion["joint_names"]]
            motion_frames = int(motion["joint_pos"].shape[0])
            motion_shapes = {key: list(motion[key].shape) for key in required if key in motion.files}
            fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
            numeric_keys = [
                key
                for key in (
                    "joint_pos", "joint_vel", "body_pos_w", "body_quat_w"
                )
                if key in motion.files
            ]
            motion_finite = all(bool(np.isfinite(motion[key]).all()) for key in numeric_keys)
            shape_contract = (
                motion.get("joint_pos", np.empty((0, 0))).shape == (motion_frames, len(motion_joint_names) + 7)
                and motion.get("joint_vel", np.empty((0, 0))).shape == (motion_frames, len(motion_joint_names) + 6)
                and motion.get("body_pos_w", np.empty((0, 0, 0))).shape == (
                    motion_frames, len(motion.get("body_names", [])), 3
                )
                and motion.get("body_quat_w", np.empty((0, 0, 0))).shape == (
                    motion_frames, len(motion.get("body_names", [])), 4
                )
            )
        payload = load_checkpoint(checkpoint_path)
        actor_dim = _actor_action_dim(payload)
        task_name = str(manifest.get("task_name", ""))
        suite_id = str(manifest.get("suite_id", ""))
        checks = {
            "manifest_passed": manifest.get("status") == PASS,
            "task_registered_as_largebox": task_name == "largebox_plane",
            "suite_is_largebox_only": suite_id == "largebox_discovery_v1",
            "motion_name_is_largebox": "largebox" in motion_path.name.lower(),
            "motion_hash_matches_manifest": sha256_file(motion_path) == manifest.get("motion_sha256"),
            "robot_hash_matches_manifest": sha256_file(urdf_path) == manifest.get("robot_asset_sha256"),
            "action_hash_matches_manifest": canonical_sha256(action_names) == manifest.get("action_schema_sha256"),
            "action_names_unique": len(action_names) == len(set(action_names)),
            "motion_joint_names_unique": len(motion_joint_names) == len(set(motion_joint_names)),
            "action_order_matches_motion": action_names == motion_joint_names,
            "action_set_matches_urdf_actuated_joints": set(action_names) == set(actuated),
            "expected_29_dof": len(action_names) == len(actuated) == len(motion_joint_names) == 29,
            "checkpoint_actor_action_dim_matches": actor_dim == len(action_names),
            "motion_contract_keys_present": not missing_motion_keys,
            "motion_has_at_least_three_frames": motion_frames >= 3,
            "motion_fps_positive_finite": bool(np.isfinite(fps) and fps > 0.0),
            "motion_arrays_finite": motion_finite,
            "motion_shapes_match_named_contract": shape_contract,
        }
        errors = [name for name, passed in checks.items() if not passed]
        status = PASS if not errors else INVALID_PROTOCOL
        result = diagnostic_result(
            "01",
            status,
            summary=(
                "largebox task, motion, robot, and 29-DoF action schema agree"
                if status == PASS
                else "task identity contract is inconsistent"
            ),
            evidence={
                "suite_id": suite_id,
                "task_name": task_name,
                "motion_path": str(motion_path),
                "motion_name": motion_path.name,
                "motion_frames": motion_frames,
                "motion_fps": fps,
                "motion_shapes": motion_shapes,
                "motion_joint_names": motion_joint_names,
                "robot_asset_path": str(urdf_path),
                "urdf_actuated_joint_names": actuated,
                "action_names": action_names,
                "checkpoint_actor_action_dim": actor_dim,
                "checks": checks,
            },
            errors=errors,
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "01", SKIPPED_DEPENDENCY, summary=str(exc), errors=[str(exc)]
        )
    except (FileNotFoundError, KeyError, ValueError, ET.ParseError, json.JSONDecodeError) as exc:
        result = diagnostic_result(
            "01", INVALID_PROTOCOL, summary="task identity audit could not be completed", errors=[str(exc)]
        )
    write_json_exclusive(target, result)
    print(f"[diag_01] {result['status']} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
