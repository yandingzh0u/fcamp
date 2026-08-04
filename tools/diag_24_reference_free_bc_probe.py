#!/usr/bin/env python3
"""Train three-seed MLP/GRU reference-free BC probes and request real rollouts."""

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
    read_json,
    write_csv_exclusive,
    write_json_exclusive,
)
from diagnostics.common.isaac_exit import finish_isaac_entrypoint
from diagnostics.common.policy_class_probe import (
    CANONICAL_INDEX_NAME,
    PolicyClassProtocol,
    canonical_index,
    deterministic_group_splits,
    load_canonical_arrays,
    load_observation_partition,
    make_probe_dataset,
    output_dir_from_args,
    require_primary_teacher_quality,
    run_reference_free_bc_real,
    save_model_bundle,
    select_policy_class_rows,
    train_action_phase_probe,
    validate_external_intervention_result,
    validate_reference_free_b_index,
)

_ALLOW_HARD_EXIT = __name__ == "__main__"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--canonical-index", type=Path, default=None)
    parser.add_argument("--probe-device", default="auto")
    parser.add_argument("--closed-loop-result", type=Path, default=None)
    parser.add_argument("--no-closed-loop", action="store_true")
    try:
        from isaaclab.app import AppLauncher
    except (ImportError, ModuleNotFoundError):
        AppLauncher = None
    if AppLauncher is not None:
        AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def _device(raw: str) -> str:
    if raw != "auto":
        return raw
    try:
        import torch
    except ImportError as exc:
        raise DependencyUnavailable("PyTorch is unavailable") from exc
    return "cuda" if torch.cuda.is_available() else "cpu"


def _execute_real(
    args, repo_root, output_dir, index_path, rows, spec, protocol, partition, model_paths
):
    try:
        from isaaclab.app import AppLauncher
        from engine.config import load_config
    except (ImportError, ModuleNotFoundError) as exc:
        raise DependencyUnavailable("Isaac Lab is unavailable for diag_24 closed-loop rollouts") from exc
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
            return run_reference_free_bc_real(
                launcher.app,
                repo_root=repo_root,
                output_dir=output_dir,
                index_path=index_path,
                rows=rows,
                spec=spec,
                protocol=protocol,
                partition=partition,
                model_paths=model_paths,
            )
        except (RuntimeError, OSError) as exc:
            raise DependencyUnavailable("Isaac/PhysX BC rollout could not run") from exc
    finally:
        # Never close Kit here: close() deadlocks on the target runner.  The
        # outer CLI writes/fyncs all evidence, then exits the owning process.
        pass


def main() -> int:
    args = _arguments()
    repo_root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(repo_root, spec, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "bc_probe.json"
    table_path = output_dir / "tables" / "reference_free_bc.csv"
    model_dir = output_dir / "models" / "diag24"
    interface_path = model_dir / "closed_loop_interface.json"
    table_rows: list[dict[str, object]] = []
    try:
        predictability = read_json(output_dir / "predictability.json")
        if predictability.get("status") != PASS:
            raise DependencyUnavailable(
                "diag_23 full-observation sanity has not passed; BC evidence would be uninterpretable"
            )
        protocol = PolicyClassProtocol.from_spec(spec)
        partition = load_observation_partition(output_dir / "observation_spec.json")
        index_path = (args.canonical_index or output_dir / CANONICAL_INDEX_NAME).expanduser().resolve()
        rows = select_policy_class_rows(
            canonical_index(index_path), checkpoint_update=protocol.primary_update
        )
        teacher_quality = require_primary_teacher_quality(
            output_dir, protocol, checkpoint_path=rows.iloc[0]["checkpoint_path"]
        )
        arrays = load_canonical_arrays(index_path, rows)
        splits = deterministic_group_splits(
            arrays.trajectory_ids,
            arrays.snapshot_ids,
            seed=protocol.split_seed,
            fractions=protocol.split_fractions,
        )
        if arrays.actor_no_reference.shape[1] != int(partition["actor_partition"]["proprio_width"]):
            raise ProtocolError("canonical no-reference width disagrees with observation partition")
        device = _device(args.probe_device)
        seeds = protocol.seeds
        model_records: dict[str, object] = {}
        model_paths: list[Path] = []
        for history in protocol.bc_histories:
            kind = "mlp" if history == 1 else "gru"
            label = "MLP-H1" if history == 1 else f"GRU-H{history}"
            dataset = make_probe_dataset(
                arrays.actor_no_reference, arrays, splits, history=history
            )
            per_seed: list[dict[str, object]] = []
            for seed in seeds:
                metrics, bundle = train_action_phase_probe(
                    dataset,
                    kind=kind,
                    seed=seed,
                    epochs=protocol.epochs,
                    batch_size=protocol.batch_size,
                    hidden_dim=protocol.gru_hidden_dim,
                    max_train_samples=protocol.max_probe_samples,
                    learning_rate=protocol.learning_rate,
                    patience=protocol.patience,
                    predict_phase=False,
                    device=device,
                )
                bundle.update(
                    {
                        "input_view": "actor_no_reference",
                        "checkpoint_sha256": arrays.metadata["checkpoint_sha256"],
                        "checkpoint_update": arrays.metadata["checkpoint_update"],
                        "observation_source_sha256": partition["actor"]["source_sha256"],
                        "closed_loop_contract": (
                            "call ReferenceFreeBCInference.reset at episode reset, then act(actor_no_reference)"
                        ),
                    }
                )
                model_path = model_dir / f"{label.lower().replace('-', '_')}_seed_{seed}.pt"
                save_model_bundle(model_path, bundle)
                model_paths.append(model_path)
                action = metrics["action"]
                record = {
                    "model": label,
                    "history_steps": history,
                    "seed": seed,
                    "action_nrmse": action["nrmse"],
                    "action_r2": action["r2"],
                    "model_path": str(model_path),
                    "closed_loop_status": "PENDING",
                    "motion_completion": None,
                    "failure_rate": None,
                }
                table_rows.append(record)
                per_seed.append({"pointwise": metrics, "model_path": str(model_path)})
            model_records[label] = {
                "history_steps": history,
                "architecture": kind,
                "seeds": per_seed,
            }
        interface = {
            "diagnostic_id": "24",
            "status": PASS,
            "interface_version": 1,
            "python_factory": "diagnostics.common.policy_class_probe.load_bc_inference",
            "input": "actor_no_reference [num_envs, no_reference_dim]",
            "reset_contract": "call reset() at every episode/snapshot restore",
            "output": "teacher-coordinate action mean [num_envs, action_dim]",
            "model_paths": [str(path) for path in model_paths],
            "checkpoint_sha256": arrays.metadata["checkpoint_sha256"],
            "preregistered_B_domain": {
                "model": protocol.b_model,
                "history_steps": protocol.b_history,
                "seed": protocol.b_seed,
                "selection": protocol.b_selection,
                "required_output": protocol.b_output_name,
            },
        }
        write_json_exclusive(interface_path, interface)

        closed_loop: dict[str, object]
        closed_loop_status = PASS
        if args.closed_loop_result is not None:
            closed_loop = validate_external_intervention_result(
                args.closed_loop_result.expanduser().resolve(),
                operation="bc_closed_loop",
                checkpoint_sha256=str(arrays.metadata["checkpoint_sha256"]),
            )
        elif args.no_closed_loop:
            raise DependencyUnavailable("real Isaac BC rollout was explicitly disabled")
        else:
            closed_loop = dict(
                _execute_real(
                    args,
                    repo_root,
                    output_dir,
                    index_path,
                    rows,
                    spec,
                    protocol,
                    partition,
                    model_paths,
                )
            )
        # The real helper returns keyed per-model/seed results; never infer
        # closure from pointwise predictions.
        if closed_loop.get("same_snapshot_replay_verified") is not True:
            raise ProtocolError("BC closed-loop result lacks exact snapshot replay verification")
        closed_metrics = closed_loop.get("metrics")
        if not isinstance(closed_metrics, dict):
            raise ProtocolError("BC closed-loop result has no metrics mapping")
        keyed = closed_metrics.get("models", {})
        if not isinstance(keyed, dict):
            raise ProtocolError("BC closed-loop metrics.models must be a mapping")
        for row in table_rows:
            key = f"{row['model']}:seed={row['seed']}"
            item = keyed.get(key)
            if not isinstance(item, dict):
                raise ProtocolError(f"BC closed-loop result is missing model {key}")
            row["closed_loop_status"] = PASS
            row["motion_completion"] = item.get("motion_completion")
            row["failure_rate"] = item.get("failure_rate")
        expected_b_model = model_dir / f"gru_h{protocol.b_history}_seed_{protocol.b_seed}.pt"
        b_index_path = output_dir / protocol.b_output_name
        b_domain = closed_loop.get("b_domain")
        if not isinstance(b_domain, dict):
            raise ProtocolError("BC closed-loop result lacks the preregistered B-domain product")
        if Path(str(b_domain.get("index_path", ""))).expanduser().resolve() != b_index_path:
            raise ProtocolError("BC closed-loop result points to a different B-domain index")
        validated_b = validate_reference_free_b_index(
            b_index_path,
            protocol=protocol,
            model_path=expected_b_model,
            expected_snapshot_ids=tuple(str(value) for value in rows["snapshot_id"].tolist()),
        )
        if b_domain.get("index_sha256") != validated_b["index_sha256"]:
            raise ProtocolError("BC closed-loop B-domain index hash changed after collection")
        result = diagnostic_result(
            "24", PASS,
            summary="reference-free BC point predictions and real closed-loop rollouts completed",
            evidence={
                "checkpoint_sha256": arrays.metadata["checkpoint_sha256"],
                "checkpoint_update": arrays.metadata["checkpoint_update"],
                "models": model_records,
                "closed_loop": closed_loop,
                "reference_free_B_domain": validated_b,
                "closed_loop_interface": str(interface_path),
                "seeds": list(seeds),
                "device": device,
                "primary_teacher_quality": teacher_quality,
                "frozen_probe_budget": {
                    "mlp_hidden_dims": list(protocol.mlp_hidden_dims),
                    "gru_hidden_dim": protocol.gru_hidden_dim,
                    "gru_layers": protocol.gru_layers,
                    "epochs": protocol.epochs,
                    "batch_size": protocol.batch_size,
                    "learning_rate": protocol.learning_rate,
                    "patience": protocol.patience,
                    "maximum_probe_samples": protocol.max_probe_samples,
                },
            },
        )
    except DependencyUnavailable as exc:
        # Pointwise model artifacts may be complete even when Isaac is not
        # installed.  This remains SKIPPED, never a scientific BC failure.
        result = diagnostic_result(
            "24", "SKIPPED_DEPENDENCY",
            summary="BC point prediction completed where possible; real closed-loop evidence is unavailable",
            evidence={
                "pointwise_rows": table_rows,
                "closed_loop_interface": str(interface_path) if interface_path.is_file() else None,
            },
            errors=[str(exc)],
            warnings=["missing Isaac/rollout assets are not evidence that the policy class fails"],
        )
    except (FileNotFoundError, ProtocolError, ValueError, KeyError, json.JSONDecodeError) as exc:
        result = diagnostic_result(
            "24", INVALID_PROTOCOL,
            summary="reference-free BC protocol failed closed",
            evidence={"pointwise_rows": table_rows},
            errors=[str(exc)],
        )
    if table_rows:
        write_csv_exclusive(table_path, table_rows)
    write_json_exclusive(target, result)
    print(f"[diag_24] {result['status']} {target}")
    return finish_isaac_entrypoint(
        0,
        isaac_launched=bool(getattr(args, "_diagnostic_isaac_launched", False)),
        allow_hard_exit=_ALLOW_HARD_EXIT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
