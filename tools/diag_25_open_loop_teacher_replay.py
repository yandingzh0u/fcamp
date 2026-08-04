#!/usr/bin/env python3
"""Replay recorded teacher actions open-loop from exact canonical snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.manifest import (
    INVALID_PROTOCOL,
    PASS,
    DependencyUnavailable,
    ProtocolError,
    diagnostic_result,
    load_spec,
    write_json_exclusive,
)
from diagnostics.common.isaac_exit import finish_isaac_entrypoint
from diagnostics.common.policy_class_probe import (
    CANONICAL_INDEX_NAME,
    PolicyClassProtocol,
    canonical_index,
    output_dir_from_args,
    require_primary_teacher_quality,
    run_open_loop_teacher_replay_real,
    select_policy_class_rows,
    validate_external_intervention_result,
)

_ALLOW_HARD_EXIT = __name__ == "__main__"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--canonical-index", type=Path, default=None)
    parser.add_argument("--replay-result", type=Path, default=None)
    try:
        from isaaclab.app import AppLauncher
    except (ImportError, ModuleNotFoundError):
        AppLauncher = None
    if AppLauncher is not None:
        AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _execute_real(args, repo_root, output_dir, index_path, rows, spec, protocol):
    try:
        from isaaclab.app import AppLauncher
        from engine.config import load_config
    except (ImportError, ModuleNotFoundError) as exc:
        raise DependencyUnavailable("Isaac Lab is unavailable for diag_25") from exc
    cfg = load_config(
        repo_root / "configs" / "fixed_reward_largebox.yaml",
        [f"environment.num_envs={len(rows)}"],
    )
    args.headless = True
    args.device = cfg.environment.device
    args._diagnostic_isaac_launched = True
    launcher = AppLauncher(args)
    try:
        try:
            return run_open_loop_teacher_replay_real(
                launcher.app,
                repo_root=repo_root,
                output_dir=output_dir,
                index_path=index_path,
                rows=rows,
                spec=spec,
                protocol=protocol,
            )
        except (RuntimeError, OSError) as exc:
            raise DependencyUnavailable(
                "Isaac/PhysX open-loop replay could not run: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    finally:
        # Evidence is published by main before the dedicated Isaac process
        # exits.  app.close() is intentionally avoided (known deadlock).
        pass


def _validate(
    value: dict[str, object],
    checkpoint_sha256: str,
    protocol: PolicyClassProtocol,
) -> dict[str, object]:
    if value.get("checkpoint_sha256") != checkpoint_sha256:
        raise ProtocolError("open-loop replay used a different checkpoint")
    if value.get("same_snapshot_replay_verified") is not True:
        raise ProtocolError("open-loop replay lacks exact same-snapshot verification")
    if value.get("uses_recorded_applied_actions") is not True:
        raise ProtocolError("open-loop branch did not replay recorded applied actions")
    metrics = value.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ProtocolError("open-loop replay has no metrics")
    exact = metrics.get("exact_identity")
    perturbations = metrics.get("perturbed_comparison")
    if not isinstance(exact, dict) or not isinstance(perturbations, dict):
        raise ProtocolError("open-loop result lacks exact identity or perturbed comparison")
    if float(exact.get("action_max_abs_error", float("inf"))) > protocol.exact_replay_action_atol:
        raise ProtocolError("exact replay action identity exceeds the frozen tolerance")
    observed = sorted(float(value) for value in perturbations)
    if observed != sorted(protocol.open_loop_perturbation_fractions):
        raise ProtocolError(
            "open-loop result does not contain every frozen joint-range perturbation fraction"
        )
    return value


def main() -> int:
    args = _arguments()
    repo_root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(repo_root, spec, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "open_loop.json"
    try:
        protocol = PolicyClassProtocol.from_spec(spec)
        index_path = (args.canonical_index or output_dir / CANONICAL_INDEX_NAME).expanduser().resolve()
        rows = select_policy_class_rows(
            canonical_index(index_path), checkpoint_update=protocol.primary_update
        )
        teacher_quality = require_primary_teacher_quality(
            output_dir, protocol, checkpoint_path=rows.iloc[0]["checkpoint_path"]
        )
        checkpoint_sha256 = str(rows.iloc[0]["checkpoint_sha256"])
        if args.replay_result is not None:
            value = validate_external_intervention_result(
                args.replay_result.expanduser().resolve(),
                operation="open_loop_teacher_replay",
                checkpoint_sha256=checkpoint_sha256,
            )
        else:
            value = dict(
                _execute_real(
                    args, repo_root, output_dir, index_path, rows, spec, protocol
                )
            )
        value = _validate(value, checkpoint_sha256, protocol)
        result = diagnostic_result(
            "25", PASS,
            summary="recorded teacher actions were replayed open-loop from exact snapshots",
            evidence={
                "checkpoint_sha256": checkpoint_sha256,
                "collector_mode": "clean_mean",
                "horizons": [1, 5, 10, 25, protocol.open_loop_horizon],
                "replay_horizon_control_steps": protocol.open_loop_horizon,
                "exact_identity_action_atol": protocol.exact_replay_action_atol,
                "joint_range_perturbation_fractions": list(protocol.open_loop_perturbation_fractions),
                "perturbation_direction": "shared_Rademacher_from_NoiseBank",
                "real_physx_result": value,
                "primary_teacher_quality": teacher_quality,
                "interpretation_guard": (
                    "exact replay is numerical identity only; feedback dependence is measured only "
                    "by open-loop versus feedback from the same frozen perturbed snapshot"
                ),
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "25", "SKIPPED_DEPENDENCY",
            summary="open-loop replay awaits canonical assets and the real Isaac adapter",
            errors=[str(exc)],
            warnings=["missing simulator assets are not scientific failure evidence"],
        )
    except (FileNotFoundError, ProtocolError, ValueError, KeyError, json.JSONDecodeError) as exc:
        result = diagnostic_result(
            "25", INVALID_PROTOCOL,
            summary="open-loop replay protocol failed closed",
            errors=[str(exc)],
        )
    write_json_exclusive(target, result)
    print(f"[diag_25] {result['status']} {target}")
    return finish_isaac_entrypoint(
        0,
        isaac_launched=bool(getattr(args, "_diagnostic_isaac_launched", False)),
        allow_hard_exit=_ALLOW_HARD_EXIT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
