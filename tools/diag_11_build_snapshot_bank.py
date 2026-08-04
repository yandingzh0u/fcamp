#!/usr/bin/env python3
"""Build clean/controlled canonical simulator snapshots with shared randomness."""

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
    resolve_path,
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
    write_json_exclusive(path, result)
    print(f"[diag_11] {result['status']} {path}")
    return 0


def main() -> int:
    # Importing torch/Isaac from the system interpreter must yield a scientific
    # dependency status, not a process crash.  The suite is normally launched
    # with ``conda run -n env_isaaclab python ...``.
    try:
        from isaaclab.app import AppLauncher
    except (ImportError, ModuleNotFoundError) as exc:
        args = _base_parser().parse_args()
        output_dir = args.output_dir.expanduser().resolve()
        result = diagnostic_result(
            "11",
            SKIPPED_DEPENDENCY,
            summary="Isaac Lab is unavailable; snapshot bank was not fabricated",
            errors=[str(exc)],
        )
        return _write_status(output_dir / "snapshot_bank.status.json", result)

    parser = _base_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    status_path = output_dir / "snapshot_bank.status.json"
    bank_path = output_dir / "snapshot_bank.pt"

    app_launcher = None
    trainer = None
    try:
        from diagnostics.common.canonical_collection import (
            CanonicalCollectionProtocol,
            assert_startup_randomization_matches,
            build_snapshot_bank_from_environment,
            close_collection_trainer,
            make_collection_trainer,
            restore_snapshot_bank,
            snapshot_bank_startup_fingerprint,
            validate_frozen_collection_semantics,
        )
        from diagnostics.common.checkpoint_io import state_digest

        spec = load_spec(args.spec)
        protocol = CanonicalCollectionProtocol.from_spec(spec)
        frozen = validate_frozen_collection_semantics(spec, protocol)
        manifest_path = output_dir / "manifest.json"
        identity_path = output_dir / "task_identity.json"
        if not manifest_path.is_file() or not identity_path.is_file():
            raise DependencyUnavailable("diag_00/diag_01 artifacts are missing")
        manifest = read_json(manifest_path)
        identity = read_json(identity_path)
        if manifest.get("status") != PASS or identity.get("status") != PASS:
            raise DependencyUnavailable("diag_00 and diag_01 must PASS first")
        if manifest.get("task_name") != "largebox_plane":
            raise ProtocolError("snapshot bank is restricted to largebox_plane")
        if bank_path.exists():
            raise ProtocolError(
                "snapshot_bank.pt already exists; use suite --resume or a new output directory"
            )

        # AppLauncher must precede CoreTrainer/env imports.  Device is frozen by
        # the production fixed-reward config.
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
            runtime_dir=output_dir / "runtime/diag_11",
        )
        bank = build_snapshot_bank_from_environment(trainer.env, protocol)
        startup_digest = assert_startup_randomization_matches(trainer.env, bank)

        # Exact same-bank replay is a protocol check, not a behavior result.
        from engine.env_state import snapshot_env_state

        restore_snapshot_bank(trainer.env, bank, mode="controlled_environment")
        first = state_digest(snapshot_env_state(trainer.env))
        restore_snapshot_bank(trainer.env, bank, mode="controlled_environment")
        second = state_digest(snapshot_env_state(trainer.env))
        if first != second:
            raise ProtocolError("restoring the same snapshot bank twice is not exact")

        bank.save(bank_path)
        phases = [bank.get(snapshot_id).phase for snapshot_id in bank.snapshot_ids]
        reset = [bank.get(snapshot_id).reset_randomization for snapshot_id in bank.snapshot_ids]
        joint_delta = __import__("torch").stack(
            [value["joint_pos_delta"] for value in reset]
        )
        result = diagnostic_result(
            "11",
            PASS,
            summary="canonical clean and NoiseBank-controlled snapshot states were captured",
            evidence={
                "snapshot_bank": str(bank_path),
                "snapshot_bank_sha256": sha256_file(bank_path),
                "snapshot_count": len(bank),
                "snapshot_seed": protocol.snapshot_seed,
                "phase_min": min(phases),
                "phase_max": max(phases),
                "unique_phase_count": len(set(phases)),
                "duplicate_phases": len(phases) - len(set(phases)),
                "reset_joint_delta_abs_max": float(joint_delta.abs().max().item()),
                "startup_randomization_sha256": startup_digest,
                "snapshot_startup_fingerprint": snapshot_bank_startup_fingerprint(bank),
                "exact_restore_digest": first,
                "frozen_semantics": frozen,
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "11", SKIPPED_DEPENDENCY, summary=str(exc), errors=[str(exc)]
        )
    except (FileNotFoundError, KeyError, ValueError, ProtocolError, json.JSONDecodeError) as exc:
        result = diagnostic_result(
            "11",
            INVALID_PROTOCOL,
            summary="snapshot-bank protocol is invalid",
            errors=[str(exc)],
        )
    except (RuntimeError, OSError) as exc:
        # Runtime absence (driver, GPU, extension) is a dependency condition;
        # a tensor/schema assertion above is raised as ProtocolError instead.
        result = diagnostic_result(
            "11",
            SKIPPED_DEPENDENCY,
            summary="Isaac/PhysX snapshot capture could not run",
            errors=[str(exc)],
        )
    finally:
        if trainer is not None:
            try:
                close_collection_trainer(trainer)
            except Exception:
                pass
    code = _write_status(status_path, result)
    return finish_isaac_entrypoint(
        code, isaac_launched=app_launcher is not None, allow_hard_exit=_ALLOW_HARD_EXIT
    )


if __name__ == "__main__":
    raise SystemExit(main())
