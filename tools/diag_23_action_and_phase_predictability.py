#!/usr/bin/env python3
"""Predict teacher action mean and phase from audited observation views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.manifest import (
    FAIL,
    INVALID_PROTOCOL,
    PASS,
    DependencyUnavailable,
    ProtocolError,
    diagnostic_result,
    load_spec,
    write_json_exclusive,
)
from diagnostics.common.policy_class_probe import (
    CANONICAL_INDEX_NAME,
    PolicyClassProtocol,
    canonical_index,
    deterministic_group_splits,
    load_canonical_arrays,
    load_observation_partition,
    make_probe_dataset,
    named_term_indices,
    output_dir_from_args,
    require_primary_teacher_quality,
    select_policy_class_rows,
    train_action_phase_probe,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--canonical-index", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _device(raw: str) -> str:
    if raw != "auto":
        return raw
    try:
        import torch
    except ImportError as exc:
        raise DependencyUnavailable("PyTorch is unavailable") from exc
    return "cuda" if torch.cuda.is_available() else "cpu"


def _views(arrays, partition: dict[str, object]) -> dict[str, np.ndarray]:
    terms = partition["actor"]["terms"]
    proprio_names = [str(term["name"]) for term in terms if term["role"] == "proprio"]
    base_names = [
        name for name in proprio_names
        if name not in {"foot_contact", "termination_contact", "last_action"}
    ]
    contacts = [name for name in ("foot_contact", "termination_contact") if name in proprio_names]
    base_indices = named_term_indices(partition, names=base_names)
    contact_indices = named_term_indices(partition, names=contacts)
    last_action_indices = named_term_indices(partition, names=["last_action"])
    reference_indices = named_term_indices(partition, role="reference")
    proprio_indices = named_term_indices(partition, role="proprio")
    extracted_no_reference = arrays.actor_full[:, proprio_indices]
    maximum_view_error = float(np.max(np.abs(extracted_no_reference - arrays.actor_no_reference)))
    if maximum_view_error > 1.0e-6:
        raise ProtocolError(
            "canonical actor_no_reference disagrees with the AST-derived proprio projection; "
            f"max_abs_error={maximum_view_error}"
        )
    return {
        "full": arrays.actor_full,
        "no_reference": arrays.actor_no_reference,
        "proprio_base": arrays.actor_full[:, base_indices],
        "proprio_plus_contacts": arrays.actor_full[:, np.concatenate((base_indices, contact_indices))],
        "proprio_plus_last_action": arrays.actor_full[:, np.concatenate((base_indices, last_action_indices))],
        "reference_only": arrays.actor_full[:, reference_indices],
    }


def main() -> int:
    args = _arguments()
    repo_root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(repo_root, spec, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "predictability.json"
    try:
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
        views = _views(arrays, partition)
        matrix = [
            ("full_obs_mlp_h1", "full", "mlp", 1),
            ("proprio_base_mlp_h1", "proprio_base", "mlp", 1),
            ("no_reference_mlp_h1", "no_reference", "mlp", 1),
            ("no_reference_gru_h4", "no_reference", "gru", 4),
            ("no_reference_gru_h8", "no_reference", "gru", 8),
            ("no_reference_gru_h16", "no_reference", "gru", 16),
            ("no_reference_gru_h32", "no_reference", "gru", 32),
            ("proprio_contacts_gru_h32", "proprio_plus_contacts", "gru", 32),
            ("proprio_last_action_gru_h32", "proprio_plus_last_action", "gru", 32),
            ("reference_only_mlp_h1", "reference_only", "mlp", 1),
        ]
        device = _device(args.device)
        models: dict[str, object] = {}
        for name, view_name, kind, history in matrix:
            dataset = make_probe_dataset(
                views[view_name], arrays, splits, history=history
            )
            seed_results: list[dict[str, object]] = []
            for seed in protocol.seeds:
                trajectory_metrics, _ = train_action_phase_probe(
                    dataset,
                    kind=kind,
                    seed=seed,
                    epochs=protocol.epochs,
                    batch_size=protocol.batch_size,
                    hidden_dim=protocol.gru_hidden_dim,
                    max_train_samples=protocol.max_probe_samples,
                    learning_rate=protocol.learning_rate,
                    patience=protocol.patience,
                    device=device,
                )
                phase_metrics, _ = train_action_phase_probe(
                    dataset,
                    kind=kind,
                    seed=seed,
                    epochs=protocol.epochs,
                    batch_size=protocol.batch_size,
                    hidden_dim=protocol.gru_hidden_dim,
                    max_train_samples=protocol.max_probe_samples,
                    learning_rate=protocol.learning_rate,
                    patience=protocol.patience,
                    heldout_phase_interval=protocol.phase_holdout,
                    device=device,
                )
                seed_results.append(
                    {
                        "seed": seed,
                        "heldout_trajectory": trajectory_metrics,
                        "heldout_contiguous_phase": phase_metrics,
                    }
                )
            models[name] = {
                "input_view": view_name,
                "model": kind,
                "history_steps": history,
                "seeds": seed_results,
            }
        full_nrmse = [
            float(item["heldout_trajectory"]["action"]["nrmse"])
            for item in models["full_obs_mlp_h1"]["seeds"]
        ]
        sanity_pass = all(value <= protocol.full_obs_sanity_nrmse for value in full_nrmse)
        result = diagnostic_result(
            "23", PASS if sanity_pass else FAIL,
            summary=(
                "action/phase predictability matrix completed"
                if sanity_pass
                else "full-observation clone sanity control did not reproduce teacher actions"
            ),
            evidence={
                "checkpoint_sha256": arrays.metadata["checkpoint_sha256"],
                "checkpoint_update": arrays.metadata["checkpoint_update"],
                "collector_mode": "clean_mean",
                "teacher_target": "action.mean (never sampled/applied action)",
                "phase_target": "sin/cos circular phase",
                "split": {
                    "ordinary": "held-out complete snapshot/trajectory groups",
                    "phase": "contiguous normalized phase interval excluded from optimization",
                    "heldout_phase_interval": list(protocol.phase_holdout),
                },
                "full_observation_sanity": {
                    "nrmse_by_seed": full_nrmse,
                    "maximum": protocol.full_obs_sanity_nrmse,
                    "required_passing_seeds": len(protocol.seeds),
                    "pass": sanity_pass,
                },
                "models": models,
                "device": device,
                "seeds": list(protocol.seeds),
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
            warnings=(
                ["do not interpret reference-free models until the full-observation sanity control passes"]
                if not sanity_pass else []
            ),
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "23", "SKIPPED_DEPENDENCY",
            summary="predictability probes await sufficient canonical split data",
            errors=[str(exc)],
        )
    except (FileNotFoundError, ProtocolError, ValueError, KeyError, json.JSONDecodeError) as exc:
        result = diagnostic_result(
            "23", INVALID_PROTOCOL,
            summary="action/phase predictability protocol failed closed",
            errors=[str(exc)],
        )
    write_json_exclusive(target, result)
    print(f"[diag_23] {result['status']} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
