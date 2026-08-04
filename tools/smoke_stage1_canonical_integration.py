#!/usr/bin/env python3
"""Minimal real-Isaac smoke for teacher and official A_amp collection."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--suite-output-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    return parser


def _tree_evidence(tree: dict[str, Any], summary: dict[str, float]) -> dict[str, Any]:
    return {
        "trajectory_shape": list(tree["trajectory"]["done"].shape),
        "actor_observation_shape": list(tree["observation"]["actor_full"].shape),
        "action_mean_shape": list(tree["action"]["mean"].shape),
        "agent_physx_shape": list(tree["imitation"]["agent_physx_raw_frame"].shape),
        "agent_fk_shape": list(tree["imitation"]["agent_fk_aligned_raw_frame"].shape),
        "reference_expert_shape": list(
            tree["imitation"]["reference_expert_raw_frame"].shape
        ),
        "phase_shape": list(tree["imitation"]["phase_normalized_pre_step"].shape),
        "action_abs_max": float(tree["action"]["applied"].abs().max().item()),
        "summary": summary,
    }


def main() -> int:
    try:
        from isaaclab.app import AppLauncher
    except (ImportError, ModuleNotFoundError) as exc:
        args = _parser().parse_args()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        result = diagnostic_result(
            "stage1_smoke",
            SKIPPED_DEPENDENCY,
            summary="Isaac Lab is unavailable",
            errors=[str(exc)],
        )
        write_json_exclusive(args.output_dir / "stage1_canonical_smoke.json", result)
        return 0

    parser = _parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    suite_output = args.suite_output_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    app_launcher = None
    trainer = None
    started = time.monotonic()

    def progress(message: str) -> None:
        print(
            f"[stage1_smoke_progress] t={time.monotonic()-started:.3f}s {message}",
            flush=True,
        )

    try:
        from diagnostics.common.canonical_collection import (
            CanonicalCollectionProtocol,
            build_snapshot_bank_from_environment,
            close_collection_trainer,
            collect_canonical_branch,
            make_collection_trainer,
            read_dense_checkpoint_records,
            validate_canonical_rollout_tree,
        )
        from diagnostics.common.legacy_policy import AAMPPolicyAdapter
        from diagnostics.common.noise_bank import CollectorMode

        if args.num_envs not in (2, 4):
            raise ProtocolError("smoke --num-envs must be 2 or 4")
        if args.horizon < 4 or args.horizon > 16:
            raise ProtocolError("smoke --horizon must lie in [4,16]")
        spec = load_spec(args.spec)
        frozen = CanonicalCollectionProtocol.from_spec(spec)
        protocol = replace(
            frozen,
            num_envs=int(args.num_envs),
            num_snapshots=int(args.num_envs),
            horizon=int(args.horizon),
        )
        manifest_path = suite_output / "manifest.json"
        inventory_path = suite_output / "checkpoints/dense_checkpoint_inventory.csv"
        if not manifest_path.is_file() or not inventory_path.is_file():
            raise DependencyUnavailable(
                "smoke requires diag_00 manifest and diag_10 dense inventory"
            )
        manifest = read_json(manifest_path)
        if manifest.get("status") != PASS:
            raise DependencyUnavailable("diag_00 manifest did not PASS")
        records = read_dense_checkpoint_records(inventory_path)
        teacher = next((record for record in records if record.update == 500), None)
        if teacher is None:
            raise DependencyUnavailable("canonical teacher update 500 is unavailable")

        from engine.config import load_config

        cfg = load_config(
            root / "configs/fixed_reward_largebox.yaml",
            [f"environment.num_envs={protocol.num_envs}"],
        )
        args.headless = True
        args.device = cfg.environment.device
        progress("launching Isaac application")
        app_launcher = AppLauncher(args)
        progress("building collection trainer")
        trainer = make_collection_trainer(
            app_launcher.app,
            repo_root=root,
            protocol=protocol,
            runtime_dir=output / "runtime",
        )
        progress("capturing snapshot bank")
        bank = build_snapshot_bank_from_environment(trainer.env, protocol)
        progress("loading exact official A_amp actor")
        a_amp_record, a_amp = AAMPPolicyAdapter.load(
            repo_root=root, spec=spec, device=trainer.env.device
        )
        progress("official A_amp actor loaded")
        branches = (
            ("teacher_fixed_reward", teacher, None),
            ("A_amp_official", a_amp_record, a_amp),
        )
        evidence: dict[str, Any] = {}
        for name, checkpoint, adapter in branches:
            progress(f"collecting branch={name}")
            tree, _, summary = collect_canonical_branch(
                trainer,
                bank,
                checkpoint,
                protocol=protocol,
                manifest=manifest,
                spec=spec,
                repo_root=root,
                mode=CollectorMode.CONTROLLED_ENVIRONMENT,
                common_sigma=0.0,
                policy_adapter=adapter,
            )
            validate_canonical_rollout_tree(tree)
            evidence[name] = _tree_evidence(tree, summary)
            progress(f"validated branch={name}")
        result = diagnostic_result(
            "stage1_smoke",
            PASS,
            summary="teacher and A_amp passed the shared real-Isaac collector smoke",
            evidence={
                "num_envs": protocol.num_envs,
                "horizon": protocol.horizon,
                "collector_mode": CollectorMode.CONTROLLED_ENVIRONMENT.value,
                "branches": evidence,
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "stage1_smoke", SKIPPED_DEPENDENCY, summary=str(exc), errors=[str(exc)]
        )
    except (
        FileNotFoundError,
        KeyError,
        ValueError,
        RuntimeError,
        ProtocolError,
        json.JSONDecodeError,
    ) as exc:
        result = diagnostic_result(
            "stage1_smoke",
            INVALID_PROTOCOL,
            summary="stage-1 canonical integration smoke failed",
            errors=[str(exc)],
        )
    finally:
        if trainer is not None:
            try:
                close_collection_trainer(trainer)
            except Exception:
                pass
    output.mkdir(parents=True, exist_ok=True)
    write_json_exclusive(output / "stage1_canonical_smoke.json", result)
    print(f"[stage1_smoke] {result['status']}")
    # Do not call app.close(): it is proven to hang after successful capture.
    return finish_isaac_entrypoint(
        0, isaac_launched=app_launcher is not None, allow_hard_exit=_ALLOW_HARD_EXIT
    )


if __name__ == "__main__":
    raise SystemExit(main())
