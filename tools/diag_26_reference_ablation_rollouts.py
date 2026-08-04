#!/usr/bin/env python3
"""Run real-PhysX reference freeze, shuffle, and delay interventions."""

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
    load_observation_partition,
    output_dir_from_args,
    require_primary_teacher_quality,
    run_reference_ablation_real,
    select_policy_class_rows,
    validate_external_intervention_result,
)

_ALLOW_HARD_EXIT = __name__ == "__main__"


REQUIRED_INTERVENTIONS = ("baseline", "freeze", "shuffle", "delay")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--canonical-index", type=Path, default=None)
    parser.add_argument("--ablation-result", type=Path, default=None)
    try:
        from isaaclab.app import AppLauncher
    except (ImportError, ModuleNotFoundError):
        AppLauncher = None
    if AppLauncher is not None:
        AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _execute_real(args, repo_root, output_dir, index_path, rows, spec, protocol, partition):
    try:
        from isaaclab.app import AppLauncher
        from engine.config import load_config
    except (ImportError, ModuleNotFoundError) as exc:
        raise DependencyUnavailable("Isaac Lab is unavailable for diag_26") from exc
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
            return run_reference_ablation_real(
                launcher.app,
                repo_root=repo_root,
                output_dir=output_dir,
                index_path=index_path,
                rows=rows,
                spec=spec,
                protocol=protocol,
                partition=partition,
            )
        except (RuntimeError, OSError) as exc:
            raise DependencyUnavailable("Isaac/PhysX reference ablations could not run") from exc
    finally:
        # main durably writes the status before the owning Isaac CLI exits.
        # Avoid app.close(), which is known to hang after collection.
        pass


def _validate(
    value: dict[str, object],
    checkpoint_sha256: str,
    protocol: PolicyClassProtocol,
) -> dict[str, object]:
    if value.get("checkpoint_sha256") != checkpoint_sha256:
        raise ProtocolError("reference ablation used a different checkpoint")
    if value.get("same_snapshot_replay_verified") is not True:
        raise ProtocolError("reference ablation lacks exact same-snapshot verification")
    if value.get("only_reference_terms_intervened") is not True:
        raise ProtocolError("reference ablation changed non-reference inputs or simulator state")
    metrics = value.get("metrics")
    if not isinstance(metrics, dict):
        raise ProtocolError("reference ablation has no metrics mapping")
    missing = [name for name in REQUIRED_INTERVENTIONS if name not in metrics]
    if missing:
        raise ProtocolError(f"reference ablation is missing interventions: {missing}")
    delay = metrics.get("delay")
    if not isinstance(delay, dict) or sorted(int(value) for value in delay) != sorted(
        protocol.ablation_delays
    ):
        raise ProtocolError("reference ablation does not contain every frozen delay")
    return value


def main() -> int:
    args = _arguments()
    repo_root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(repo_root, spec, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "reference_ablation.json"
    try:
        protocol = PolicyClassProtocol.from_spec(spec)
        partition = load_observation_partition(output_dir / "observation_spec.json")
        delays = protocol.ablation_delays
        index_path = (args.canonical_index or output_dir / CANONICAL_INDEX_NAME).expanduser().resolve()
        rows = select_policy_class_rows(
            canonical_index(index_path), checkpoint_update=protocol.primary_update
        )
        teacher_quality = require_primary_teacher_quality(
            output_dir, protocol, checkpoint_path=rows.iloc[0]["checkpoint_path"]
        )
        checkpoint_sha256 = str(rows.iloc[0]["checkpoint_sha256"])
        if args.ablation_result is not None:
            value = validate_external_intervention_result(
                args.ablation_result.expanduser().resolve(),
                operation="reference_ablation",
                checkpoint_sha256=checkpoint_sha256,
            )
        else:
            value = dict(
                _execute_real(
                    args,
                    repo_root,
                    output_dir,
                    index_path,
                    rows,
                    spec,
                    protocol,
                    partition,
                )
            )
        value = _validate(value, checkpoint_sha256, protocol)
        result = diagnostic_result(
            "26", PASS,
            summary="reference interventions completed in real PhysX from matched snapshots",
            evidence={
                "checkpoint_sha256": checkpoint_sha256,
                "collector_mode": "clean_mean",
                "delays_control_steps": list(delays),
                "real_physx_result": value,
                "primary_teacher_quality": teacher_quality,
                "role_guard": "causal reference-dependence diagnosis only; never a candidate policy method",
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "26", "SKIPPED_DEPENDENCY",
            summary="reference ablations await canonical assets and the real Isaac adapter",
            errors=[str(exc)],
            warnings=["missing simulator assets are not scientific failure evidence"],
        )
    except (FileNotFoundError, ProtocolError, ValueError, KeyError, json.JSONDecodeError) as exc:
        result = diagnostic_result(
            "26", INVALID_PROTOCOL,
            summary="reference-ablation protocol failed closed",
            errors=[str(exc)],
        )
    write_json_exclusive(target, result)
    print(f"[diag_26] {result['status']} {target}")
    return finish_isaac_entrypoint(
        0,
        isaac_launched=bool(getattr(args, "_diagnostic_isaac_launched", False)),
        allow_hard_exit=_ALLOW_HARD_EXIT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
