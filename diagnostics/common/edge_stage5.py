"""Entrypoint implementations for discovery-suite diagnostics 50--59."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .canonical_collection import load_rollout_index
from .edge_causality import (
    EDGE_MANIFEST_SCHEMA,
    LocalEdgeProtocol,
    aggregate_teacher_outcomes,
    baseline_outperformance,
    build_backend_request,
    build_positive_buffer_manifest,
    candidate_chains,
    compute_edge_overlap,
    diag35_evaluation_critic_manifest,
    evaluate_edge_seed_set,
    evaluate_single_seed,
    held_out_reward_icc,
    invoke_backend,
    load_edge_manifest,
    load_real_edge_backend,
    outcome_edge_evidence,
    read_csv_rows,
    reject_deprecated_legacy,
    select_preregistered_edges,
    transition_crossings,
    validate_evaluation_critics,
    validate_positive_buffer_isolation,
    write_yaml_exclusive,
)
from .manifest import (
    FAIL,
    INVALID_PROTOCOL,
    PASS,
    SKIPPED_DEPENDENCY,
    DependencyUnavailable,
    ProtocolError,
    canonical_sha256,
    diagnostic_result,
    load_spec,
    read_json,
    sha256_file,
    write_csv_exclusive,
    write_json_exclusive,
)
from .policy_class_probe import output_dir_from_args
from .quality_panel import outcome_metrics_from_spec
from .reward_stage4 import (
    load_stage3_critics,
    read_parquet_rows,
    score_canonical_trajectories_streaming,
)
from .reward_validity import reward_seed_agreement


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_stage5_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--backend",
        type=Path,
        default=None,
        help="Explicit real PhysX/AMP backend; no fallback is permitted.",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _context(args: argparse.Namespace) -> tuple[Path, dict[str, Any], Path, LocalEdgeProtocol]:
    root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output = output_dir_from_args(root, spec, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol = LocalEdgeProtocol.from_spec(spec)
    return root, spec, output, protocol


def _device(args: argparse.Namespace) -> str:
    if args.device != "auto":
        return str(args.device)
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _quality_panel(output: Path) -> list[dict[str, Any]]:
    status = read_json(output / "quality_panel.status.json")
    if status.get("status") != PASS:
        raise DependencyUnavailable("diag_40 quality panel is not PASS")
    path = output / "quality_panel.parquet"
    expected = str(status.get("evidence", {}).get("quality_panel_sha256", ""))
    if not path.is_file() or sha256_file(path) != expected:
        raise ProtocolError("quality panel is missing or changed after diag_40")
    return read_parquet_rows(path)


def _canonical_rows_with_split(output: Path) -> tuple[Path, list[dict[str, Any]]]:
    status = read_json(output / "canonical_rollout_index.status.json")
    if status.get("status") != PASS:
        raise DependencyUnavailable("diag_12 canonical rollout collection is not PASS")
    index = output / "canonical_rollout_index.parquet"
    expected = str(
        status.get("evidence", {}).get("canonical_rollout_index_sha256", "")
    )
    if not index.is_file() or sha256_file(index) != expected:
        raise ProtocolError("canonical rollout index is missing or changed")
    rows = load_rollout_index(index)
    split_status = read_json(output / "split_audit.json")
    if split_status.get("status") != PASS:
        raise DependencyUnavailable("diag_14 frozen split audit is not PASS")
    sample_to_split = (
        split_status.get("evidence", {})
        .get("inner_snapshot_split", {})
        .get("sample_to_split")
    )
    if not isinstance(sample_to_split, Mapping):
        raise ProtocolError("diag_14 lacks sample_to_split")
    enriched = []
    for row in rows:
        item = dict(row)
        sample = str(item.get("sample_id", ""))
        if sample in sample_to_split:
            item["split"] = str(sample_to_split[sample])
        enriched.append(item)
    return index, enriched


def _teacher_window_splits(
    index_path: Path,
    panel: Sequence[Mapping[str, Any]],
    update: int,
) -> dict[str, np.ndarray]:
    from .reward_stage4 import canonical_trajectory_window_bank

    result: dict[str, np.ndarray] = {}
    for split in ("train", "validation", "test"):
        rows = [
            dict(row)
            for row in panel
            if str(row.get("policy_domain")) == "teacher_fixed_reward"
            and str(row.get("collector_mode")) == "controlled_environment"
            and float(row.get("common_sigma", np.nan)) == 0.0
            and int(row.get("checkpoint_update", -1)) == int(update)
            and str(row.get("split")) == split
        ]
        if not rows:
            raise DependencyUnavailable(
                f"teacher update {update} lacks controlled {split} trajectories"
            )
        windows, _owners, _unscored = canonical_trajectory_window_bank(index_path, rows)
        if windows.shape[0] < 2:
            raise DependencyUnavailable(
                f"teacher update {update} has fewer than two valid {split} AMP windows"
            )
        result[split] = windows
    return result


def _edge_id(source: int, target: int) -> str:
    return f"T_u{int(source):04d}_to_T_u{int(target):04d}"


def _write_error_result(
    diagnostic_id: str,
    target: Path,
    exc: BaseException,
    *,
    skipped_summary: str,
    invalid_summary: str,
) -> int:
    if isinstance(exc, DependencyUnavailable):
        result = diagnostic_result(
            diagnostic_id,
            SKIPPED_DEPENDENCY,
            summary=skipped_summary,
            errors=[str(exc)],
        )
    else:
        result = diagnostic_result(
            diagnostic_id,
            INVALID_PROTOCOL,
            summary=invalid_summary,
            errors=[str(exc)],
        )
    write_json_exclusive(target, result)
    print(f"[diag_{diagnostic_id}] {result['status']} {target}")
    return 0


def run_diag50() -> int:
    args = parse_stage5_args("Preregister causal teacher edges from outcomes and overlap.")
    root, spec, output, protocol = _context(args)
    target = output / "edge_manifest.status.json"
    manifest_path = output / "edge_manifest.yaml"
    try:
        teacher_status = read_json(output / "teacher_transition.json")
        crossings = transition_crossings(teacher_status, protocol)
        panel = _quality_panel(output)
        metrics = outcome_metrics_from_spec(spec)
        outcomes = aggregate_teacher_outcomes(panel, metrics=metrics)
        available = tuple(
            update
            for update in protocol.candidate_updates
            if update in outcomes
        )
        if len(available) < 2 or 500 not in available:
            raise DependencyUnavailable(
                "fewer than two candidate teacher checkpoints (including u500) have "
                "controlled PhysX outcomes"
            )
        index_path, _ = _canonical_rows_with_split(output)
        reward_icc = held_out_reward_icc(output)
        # Candidate construction is fixed before any online update.  Only pairs
        # that bracket a recorded completion crossing, plus direct-final
        # controls, receive the expensive empirical-overlap audit.
        pair_updates = [
            (source, destination)
            for left, source in enumerate(available)
            for destination in available[left + 1 :]
            if destination == 500
            or any(source < crossing <= destination for crossing in crossings.values())
        ]
        if not pair_updates:
            raise DependencyUnavailable("no candidate pair brackets an observed transition")
        split_cache: dict[int, dict[str, np.ndarray]] = {}
        candidate_records: list[dict[str, Any]] = []
        for source_update, target_update in pair_updates:
            outcome = outcome_edge_evidence(
                outcomes[source_update], outcomes[target_update], metrics=metrics
            )
            record: dict[str, Any] = {
                "edge_id": _edge_id(source_update, target_update),
                "source_update": source_update,
                "target_update": target_update,
                "source_checkpoint_sha256": outcomes[source_update][
                    "checkpoint_sha256"
                ],
                "target_checkpoint_sha256": outcomes[target_update][
                    "checkpoint_sha256"
                ],
                "source_domain": f"T_u{source_update}",
                "target_domain": f"T_u{target_update}",
                "positive_role": "target_agent_physx_controlled",
                "outcome": outcome,
            }
            if outcome["strict_pareto_target_over_source"]:
                if source_update not in split_cache:
                    split_cache[source_update] = _teacher_window_splits(
                        index_path, panel, source_update
                    )
                if target_update not in split_cache:
                    split_cache[target_update] = _teacher_window_splits(
                        index_path, panel, target_update
                    )
                record["overlap"] = compute_edge_overlap(
                    split_cache[source_update],
                    split_cache[target_update],
                    protocol=protocol,
                    reward_icc=reward_icc,
                )
            else:
                record["overlap"] = {
                    "eligible": False,
                    "reason": "target does not strictly Pareto-dominate source outcomes",
                }
            candidate_records.append(record)
        selected, selection_audit = select_preregistered_edges(
            candidate_records, crossings=crossings, final_update=500
        )
        chains = candidate_chains(selected, length=protocol.minimum_chain_edges)
        transition_complete = all(
            bool(item.get("selected"))
            for item in selection_audit
            if "transition_level" in item
        )
        direct_complete = any(
            item.get("control") == "direct_final" and item.get("selected")
            for item in selection_audit
        )
        scientific_pass = bool(selected and transition_complete and direct_complete)
        manifest = {
            "artifact_schema": EDGE_MANIFEST_SCHEMA,
            "status": PASS if scientific_pass else FAIL,
            "selection_time": "before_all_stage5_online_updates",
            "teacher_transition_source": str(output / "teacher_transition.json"),
            "teacher_transition_source_sha256": sha256_file(
                output / "teacher_transition.json"
            ),
            "quality_panel_source": str(output / "quality_panel.parquet"),
            "quality_panel_source_sha256": sha256_file(output / "quality_panel.parquet"),
            "candidate_updates": list(available),
            "recorded_completion_crossings": {
                str(level): update for level, update in crossings.items()
            },
            "selection_rule": {
                "per_transition": [
                    "maximum_eligible_overlap",
                    "median_eligible_overlap",
                ],
                "overlap_order": (
                    "weakest normalized gate, posterior overlap, bidirectional "
                    "coverage, bidirectional ESS, inverse source AUC"
                ),
                "tie_break": "smallest update gap then edge id",
                "target_must_strictly_pareto_dominate_source": True,
            },
            "held_out_reward_icc": reward_icc,
            "edges": selected,
            "selection_audit": selection_audit,
            "candidate_edge_audit": candidate_records,
            "preregistered_three_edge_chains": chains,
            "online_result_used_for_selection": False,
            "legacy_branch_used": False,
        }
        reject_deprecated_legacy(manifest)
        write_yaml_exclusive(manifest_path, manifest)
        result = diagnostic_result(
            "50",
            PASS if scientific_pass else FAIL,
            summary=(
                f"preregistered {len(selected)} condition-matched local edges"
                if scientific_pass
                else "recorded outcomes/overlap did not supply every required edge role"
            ),
            evidence={
                "edge_manifest": str(manifest_path),
                "edge_manifest_sha256": sha256_file(manifest_path),
                "selected_edge_count": len(selected),
                "candidate_edge_count": len(candidate_records),
                "candidate_three_edge_chain_count": len(chains),
                "all_transition_roles_selected": bool(transition_complete),
                "direct_final_control_selected": bool(direct_complete),
                "selection_before_online_updates": True,
                "decision_variable_E": "UNKNOWN",
            },
        )
        write_json_exclusive(target, result)
        print(f"[diag_50] {result['status']} {target}")
        return 0
    except (
        DependencyUnavailable,
        FileNotFoundError,
        ProtocolError,
        ValueError,
        KeyError,
        FloatingPointError,
        RuntimeError,
    ) as exc:
        return _write_error_result(
            "50",
            target,
            exc,
            skipped_summary="edge preregistration awaits recorded teacher/overlap evidence",
            invalid_summary="edge preregistration protocol failed closed",
        )


def run_diag51() -> int:
    args = parse_stage5_args("Audit whether preregistered targets dominate held-out AMP reward.")
    _root, spec, output, protocol = _context(args)
    target = output / "landscape.json"
    try:
        manifest = load_edge_manifest(output / "edge_manifest.yaml")
        panel = [
            row
            for row in _quality_panel(output)
            if str(row.get("policy_domain")) == "teacher_fixed_reward"
            and str(row.get("collector_mode")) == "controlled_environment"
            and float(row.get("common_sigma", np.nan)) == 0.0
        ]
        critics = load_stage3_critics(output, spec, device=_device(args))
        index_path = output / "canonical_rollout_index.parquet"
        scores, unscored = score_canonical_trajectories_streaming(
            index_path, panel, critics, batch_trajectories=8
        )
        valid = set(range(len(panel))) - set(unscored)
        family_scores = scores["T_u500"]
        records = []
        for edge in manifest["edges"]:
            source_indices = [
                index
                for index, row in enumerate(panel)
                if index in valid
                and int(row["checkpoint_update"]) == int(edge["source_update"])
            ]
            target_indices = [
                index
                for index, row in enumerate(panel)
                if index in valid
                and int(row["checkpoint_update"]) == int(edge["target_update"])
            ]
            if not source_indices or not target_indices:
                raise DependencyUnavailable(
                    f"edge {edge['edge_id']} lacks scoreable controlled trajectories"
                )
            source_means = family_scores[:, source_indices].mean(axis=1)
            target_means = family_scores[:, target_indices].mean(axis=1)
            deltas = target_means - source_means
            combined = [
                np.concatenate(
                    (family_scores[seed_index, source_indices], family_scores[seed_index, target_indices])
                )
                for seed_index in range(family_scores.shape[0])
            ]
            agreement = reward_seed_agreement(combined)
            seed_direction = [bool(value > 0.0) for value in deltas]
            passes = bool(
                all(seed_direction)
                and float(agreement["icc_consistency"]) >= protocol.reward_icc_min
            )
            records.append(
                {
                    "edge_id": edge["edge_id"],
                    "source_update": edge["source_update"],
                    "target_update": edge["target_update"],
                    "critic_role": "frozen_diag35_evaluation_only",
                    "critic_training_edge": "A_amp_to_T_u500",
                    "critic_seeds": list(critics.seeds),
                    "source_reward_mean_by_seed": source_means.tolist(),
                    "target_reward_mean_by_seed": target_means.tolist(),
                    "target_minus_source_by_seed": deltas.tolist(),
                    "held_out_seed_direction_consistent": all(seed_direction),
                    "reward_seed_agreement": agreement,
                    "target_reward_dominates": passes,
                    "training_critic_score_used": False,
                }
            )
        scientific_pass = bool(records and all(row["target_reward_dominates"] for row in records))
        result = diagnostic_result(
            "51",
            PASS if scientific_pass else FAIL,
            summary=(
                "every preregistered target dominates frozen held-out AMP reward"
                if scientific_pass
                else "at least one preregistered target does not dominate held-out AMP reward"
            ),
            evidence={
                "edges": records,
                "unscored_short_trajectory_count": len(unscored),
                "evaluation_critic_index": str(critics.index_path),
                "evaluation_critic_index_sha256": critics.index_sha256,
                "source_classifier_used_as_reward": False,
                "decision_variable_E": "UNKNOWN",
            },
        )
        write_json_exclusive(target, result)
        print(f"[diag_51] {result['status']} {target}")
        return 0
    except (
        DependencyUnavailable,
        FileNotFoundError,
        ProtocolError,
        ValueError,
        KeyError,
        FloatingPointError,
        RuntimeError,
    ) as exc:
        return _write_error_result(
            "51",
            target,
            exc,
            skipped_summary="local reward landscape awaits canonical held-out critic evidence",
            invalid_summary="local reward landscape protocol failed closed",
        )


def _run_online_matrix(
    *,
    diagnostic_id: str,
    args: argparse.Namespace,
    output: Path,
    protocol: LocalEdgeProtocol,
    variants: Sequence[tuple[str, str, str]],
    directory: Path,
    summary_rule: str,
    edges: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    manifest = load_edge_manifest(output / "edge_manifest.yaml")
    index_path, index_rows = _canonical_rows_with_split(output)
    evaluation = diag35_evaluation_critic_manifest(output, protocol)
    backend, backend_path, backend_hash = load_real_edge_backend(
        args.repo_root, args.backend
    )
    all_records = []
    selected_edges = list(manifest["edges"] if edges is None else edges)
    if not selected_edges:
        raise DependencyUnavailable("online matrix has no preregistered edges")
    for edge in selected_edges:
        reject_deprecated_legacy(edge)
        positive = build_positive_buffer_manifest(index_rows, edge, split="train")
        for variant, actor_observation, critic_observation in variants:
            seed_audits = []
            seed_artifacts = []
            for seed in protocol.ppo_seeds:
                artifact_path = (
                    directory
                    / str(edge["edge_id"])
                    / str(variant)
                    / f"seed_{seed}.json"
                ).resolve()
                request = build_backend_request(
                    diagnostic_id=diagnostic_id,
                    edge=edge,
                    variant=variant,
                    seed=seed,
                    actor_observation=actor_observation,
                    critic_observation=critic_observation,
                    protocol=protocol,
                    positive_buffer=positive,
                    evaluation_critics=evaluation,
                    output_path=artifact_path,
                )
                raw, audit = invoke_backend(backend, request, protocol=protocol)
                # The backend returns evidence; Stage 5 owns the immutable JSON.
                write_json_exclusive(artifact_path, raw)
                seed_audits.append(audit)
                seed_artifacts.append(
                    {
                        "seed": seed,
                        "path": str(artifact_path),
                        "sha256": sha256_file(artifact_path),
                    }
                )
            aggregate = evaluate_edge_seed_set(seed_audits, protocol=protocol)
            all_records.append(
                {
                    "edge_id": edge["edge_id"],
                    "source_update": edge["source_update"],
                    "target_update": edge["target_update"],
                    "variant": variant,
                    "actor_observation": actor_observation,
                    "critic_observation": critic_observation,
                    "positive_buffer": positive,
                    "run_artifacts": seed_artifacts,
                    "seed_gate": aggregate,
                }
            )
    if not all_records:
        raise DependencyUnavailable("no preregistered edge was available for online execution")
    return {
        "backend": str(backend_path),
        "backend_sha256": backend_hash,
        "canonical_index": str(index_path),
        "canonical_index_sha256": sha256_file(index_path),
        "evaluation_critics": evaluation,
        "records": all_records,
        "summary_rule": summary_rule,
        "real_physx_ppo_only": True,
    }


def run_diag52() -> int:
    args = parse_stage5_args("Run real same-policy-class local AMP edge tests.")
    _root, _spec, output, protocol = _context(args)
    target = output / "edge_runs" / "same_policy_class" / "summary.json"
    try:
        variants = [
            (name, "reference_conditioned_full", "reference_conditioned_privileged")
            for name in protocol.same_policy_variants
        ]
        evidence = _run_online_matrix(
            diagnostic_id="52",
            args=args,
            output=output,
            protocol=protocol,
            variants=variants,
            directory=target.parent,
            summary_rule=(
                "scientific PASS iff at least one frozen/online discriminator edge "
                "passes >=2/3 PPO seeds; controls never create a PASS"
            ),
        )
        main = [
            row
            for row in evidence["records"]
            if row["variant"] in {"frozen_discriminator", "online_discriminator"}
        ]
        scientific_pass = any(row["seed_gate"]["passed"] for row in main)
        result = diagnostic_result(
            "52",
            PASS if scientific_pass else FAIL,
            summary=(
                "same-policy-class local AMP moved at least one preregistered edge"
                if scientific_pass
                else "same-policy-class local AMP did not pass a preregistered edge"
            ),
            evidence={**evidence, "scientific_pass": scientific_pass, "decision_variable_E": "UNKNOWN"},
        )
        write_json_exclusive(target, result)
        print(f"[diag_52] {result['status']} {target}")
        return 0
    except (
        DependencyUnavailable,
        FileNotFoundError,
        ProtocolError,
        ValueError,
        KeyError,
        FloatingPointError,
        RuntimeError,
        ImportError,
    ) as exc:
        return _write_error_result(
            "52",
            target,
            exc,
            skipped_summary="real same-policy-class PhysX/AMP PPO backend is unavailable",
            invalid_summary="same-policy-class edge protocol failed closed",
        )


def run_diag53() -> int:
    args = parse_stage5_args("Run the real no-reference local AMP edge gate.")
    _root, _spec, output, protocol = _context(args)
    target = output / "edge_runs" / "reference_free" / "summary.json"
    try:
        variants = [
            (name, "actor_no_reference", name)
            for name in protocol.reference_free_variants
        ]
        evidence = _run_online_matrix(
            diagnostic_id="53",
            args=args,
            output=output,
            protocol=protocol,
            variants=variants,
            directory=target.parent,
            summary_rule=(
                "primary no-reference-critic edge passes >=2/3 seeds; privileged "
                "critic is reported only as a training upper bound"
            ),
        )
        primary = [
            row
            for row in evidence["records"]
            if row["variant"] == "no_reference_critic"
        ]
        scientific_pass = any(row["seed_gate"]["passed"] for row in primary)
        result = diagnostic_result(
            "53",
            PASS if scientific_pass else FAIL,
            summary=(
                "a genuine no-reference edge passed the frozen causal criteria"
                if scientific_pass
                else "no genuine no-reference edge passed the frozen causal criteria"
            ),
            evidence={**evidence, "scientific_pass": scientific_pass, "decision_variable_E": "UNKNOWN"},
        )
        write_json_exclusive(target, result)
        print(f"[diag_53] {result['status']} {target}")
        return 0
    except (
        DependencyUnavailable,
        FileNotFoundError,
        ProtocolError,
        ValueError,
        KeyError,
        FloatingPointError,
        RuntimeError,
        ImportError,
    ) as exc:
        return _write_error_result(
            "53",
            target,
            exc,
            skipped_summary="real no-reference PhysX/AMP PPO backend or BC initialization is unavailable",
            invalid_summary="no-reference edge protocol failed closed",
        )


def _reverse_edges(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for edge in manifest["edges"]:
        reverse = dict(edge)
        reverse["edge_id"] = f"reverse__{edge['edge_id']}"
        reverse["source_update"], reverse["target_update"] = (
            int(edge["target_update"]),
            int(edge["source_update"]),
        )
        reverse["source_checkpoint_sha256"], reverse["target_checkpoint_sha256"] = (
            edge["target_checkpoint_sha256"],
            edge["source_checkpoint_sha256"],
        )
        reverse["source_domain"], reverse["target_domain"] = (
            edge["target_domain"],
            edge["source_domain"],
        )
        result.append(reverse)
    return result


def run_diag54() -> int:
    args = parse_stage5_args("Run frozen/online and forward/reverse edge factorial controls.")
    _root, _spec, output, protocol = _context(args)
    target = output / "edge_runs" / "variants" / "summary.json"
    try:
        # Both directions are executed through the same rigorously validated
        # per-run backend contract.  A source/target swap is fixed directly
        # from the preregistered manifest, not selected from results.
        manifest = load_edge_manifest(output / "edge_manifest.yaml")
        original_edges = manifest["edges"]
        forward_variants = [
            ("forward_frozen", "reference_conditioned_full", "reference_conditioned_privileged"),
            ("forward_online", "reference_conditioned_full", "reference_conditioned_privileged"),
        ]
        forward = _run_online_matrix(
            diagnostic_id="54",
            args=args,
            output=output,
            protocol=protocol,
            variants=forward_variants,
            directory=target.parent / "forward",
            summary_rule="factorial audit; no single variant is selected post hoc",
            edges=original_edges,
        )
        reverse = _run_online_matrix(
            diagnostic_id="54",
            args=args,
            output=output,
            protocol=protocol,
            variants=[
                ("reverse_frozen", "reference_conditioned_full", "reference_conditioned_privileged"),
                ("reverse_online", "reference_conditioned_full", "reference_conditioned_privileged"),
            ],
            directory=target.parent / "reverse",
            summary_rule="factorial audit; reverse direction is a preregistered negative control",
            edges=_reverse_edges(manifest),
        )
        result = diagnostic_result(
            "54",
            PASS,
            summary="forward/reverse and frozen/online edge factorial completed in real PhysX",
            evidence={
                "forward": forward,
                "reverse": reverse,
                "backend": forward["backend"],
                "backend_sha256": forward["backend_sha256"],
                "original_edge_count": len(original_edges),
                "decision_variable_E": "UNKNOWN",
            },
        )
        write_json_exclusive(target, result)
        print(f"[diag_54] {result['status']} {target}")
        return 0
    except (
        DependencyUnavailable,
        FileNotFoundError,
        ProtocolError,
        ValueError,
        KeyError,
        FloatingPointError,
        RuntimeError,
        ImportError,
    ) as exc:
        return _write_error_result(
            "54",
            target,
            exc,
            skipped_summary="complete real forward/reverse edge factorial is unavailable",
            invalid_summary="edge factorial protocol failed closed",
        )


def run_diag55() -> int:
    args = parse_stage5_args("Run a real sequential chain of three no-reference edges.")
    _root, _spec, output, protocol = _context(args)
    target = output / "edge_runs" / "three_edge_chain" / "summary.json"
    try:
        manifest = load_edge_manifest(output / "edge_manifest.yaml")
        chains = candidate_chains(manifest["edges"], length=protocol.minimum_chain_edges)
        if not chains:
            result = diagnostic_result(
                "55",
                FAIL,
                summary="the preregistered edge graph contains no consecutive three-edge chain",
                evidence={
                    "candidate_chains": [],
                    "posthoc_edge_addition": False,
                    "decision_variable_E": "-",
                },
            )
            write_json_exclusive(target, result)
            print(f"[diag_55] {result['status']} {target}")
            return 0
        backend, backend_path, backend_hash = load_real_edge_backend(
            args.repo_root, args.backend
        )
        method = getattr(backend, "run_chain_experiment", None)
        if not callable(method):
            raise DependencyUnavailable(
                "real backend lacks run_chain_experiment; independent edge runs are "
                "not evidence of sequential transfer"
            )
        evaluation = diag35_evaluation_critic_manifest(output, protocol)
        _index_path, index_rows = _canonical_rows_with_split(output)
        by_id = {str(edge["edge_id"]): edge for edge in manifest["edges"]}
        chain_records = []
        for chain in chains:
            edges = [by_id[edge_id] for edge_id in chain]
            positive = [
                build_positive_buffer_manifest(index_rows, edge, split="train")
                for edge in edges
            ]
            request = {
                "request_schema": "largebox_real_physx_three_edge_chain_request_v1",
                "edges": edges,
                "edge_ids": chain,
                "positive_buffers": positive,
                "evaluation_critics": evaluation,
                "ppo_seeds": list(protocol.ppo_seeds),
                "actor_observation": "actor_no_reference",
                "critic_observation": "no_reference_critic",
                "num_envs": protocol.num_envs,
                "ppo_updates_per_edge": protocol.updates,
                "canonical_evaluation_snapshots": protocol.evaluation_snapshots,
                "physics_engine": "PhysX",
                "output_dir": str(target.parent / canonical_sha256(chain)[:12]),
            }
            raw = method(request)
            if not isinstance(raw, Mapping):
                raise ProtocolError("chain backend returned a non-mapping")
            reject_deprecated_legacy(raw)
            if (
                raw.get("artifact_schema")
                != "largebox_real_physx_three_edge_chain_result_v1"
                or raw.get("execution_kind") != "real_physx_ppo"
                or raw.get("physics_engine") != "PhysX"
                or raw.get("request_sha256") != canonical_sha256(request)
                or int(raw.get("num_envs", -1)) != protocol.num_envs
                or int(raw.get("ppo_updates_per_edge", -1)) != protocol.updates
            ):
                raise ProtocolError("sequential chain result lacks exact real-PhysX provenance")
            if raw.get("edge_ids") != chain:
                raise ProtocolError("sequential chain result changed the preregistered edge order")
            by_seed = raw.get("by_seed")
            if not isinstance(by_seed, Mapping) or set(int(x) for x in by_seed) != set(
                protocol.ppo_seeds
            ):
                raise ProtocolError("chain result lacks exactly the three PPO seed records")
            seed_records = []
            for raw_seed, seed_record in by_seed.items():
                if not isinstance(seed_record, Mapping):
                    raise ProtocolError("chain seed record is not a mapping")
                seed = int(raw_seed)
                edge_results = seed_record.get("edges")
                if not isinstance(edge_results, list) or [
                    item.get("edge_id") for item in edge_results
                ] != chain:
                    raise ProtocolError("chain seed changed the preregistered edge order")
                training_critics = seed_record.get("training_critic_artifacts")
                if not isinstance(training_critics, list):
                    raise ProtocolError("chain seed lacks edge-training critic artifacts")
                validate_evaluation_critics(
                    evaluation,
                    training_critic_artifacts=training_critics,
                    expected_seeds=protocol.held_out_critic_seeds,
                )
                negative_by_edge = seed_record.get("policy_negative_sample_ids_by_edge")
                if not isinstance(negative_by_edge, Mapping):
                    raise ProtocolError("chain seed lacks policy-negative identities by edge")
                edge_audits = []
                failure_increases = []
                progress_drops = []
                for edge_result, positive_manifest in zip(edge_results, positive):
                    if int(edge_result.get("seed", -1)) != seed:
                        raise ProtocolError("chain edge result seed changed")
                    points = edge_result.get("evaluations")
                    if not isinstance(points, list) or tuple(
                        int(point.get("update", -1)) for point in points
                    ) != protocol.evaluation_updates:
                        raise ProtocolError("chain edge lacks every frozen canonical evaluation")
                    for point in points:
                        rewards = point.get("held_out_rewards")
                        if not isinstance(rewards, Mapping) or set(
                            int(value) for value in rewards
                        ) != set(protocol.held_out_critic_seeds):
                            raise ProtocolError("chain edge lacks five held-out critic rewards")
                    negatives = negative_by_edge.get(edge_result["edge_id"])
                    if not isinstance(negatives, list) or not negatives:
                        raise ProtocolError("chain edge lacks policy-negative sample identities")
                    validate_positive_buffer_isolation(
                        positive_manifest, negative_sample_ids=negatives
                    )
                    audit = evaluate_single_seed(edge_result, protocol=protocol)
                    edge_audits.append(audit)
                    failure_increases.append(
                        float(points[-1]["failure_rate"])
                        - float(points[0]["failure_rate"])
                    )
                    progress_drops.append(
                        float(points[0]["task_or_progress"])
                        - float(points[-1]["task_or_progress"])
                    )
                physical = seed_record.get("physical_artifacts")
                if not isinstance(physical, list) or not physical:
                    raise ProtocolError("chain seed lacks persisted physical artifacts")
                for artifact in physical:
                    path = Path(str(artifact.get("path", ""))).expanduser().resolve()
                    if not path.is_file() or sha256_file(path) != str(
                        artifact.get("sha256", "")
                    ):
                        raise ProtocolError(
                            f"chain physical artifact is missing or changed: {path}"
                        )
                seed_records.append(
                    {
                        "seed": seed,
                        "passed": all(audit["passed"] for audit in edge_audits),
                        "edge_audits": edge_audits,
                        "target_distance_reduction": float(
                            np.mean(
                                [audit["target_distance_reduction"] for audit in edge_audits]
                            )
                        ),
                        "failure_rate_increase": float(max(failure_increases)),
                        "task_or_progress_drop": float(max(progress_drops)),
                    }
                )
            passing = sum(bool(item["passed"]) for item in seed_records)
            aggregate_metrics = {
                name: float(np.mean([item[name] for item in seed_records]))
                for name in (
                    "target_distance_reduction",
                    "failure_rate_increase",
                    "task_or_progress_drop",
                )
            }
            chain_records.append(
                {
                    "edge_ids": chain,
                    "passing_seed_count": passing,
                    "passed": passing >= protocol.minimum_passing_seeds,
                    "by_seed": seed_records,
                    "result": {
                        "aggregate_metrics": aggregate_metrics,
                        "raw_result_sha256": canonical_sha256(raw),
                    },
                }
            )
        scientific_pass = any(item["passed"] for item in chain_records)
        result = diagnostic_result(
            "55",
            PASS if scientific_pass else FAIL,
            summary=(
                "a no-reference student crossed three consecutive preregistered edges"
                if scientific_pass
                else "no no-reference student crossed three consecutive preregistered edges"
            ),
            evidence={
                "chains": chain_records,
                "backend": str(backend_path),
                "backend_sha256": backend_hash,
                "posthoc_edge_addition": False,
                "decision_variable_E": "UNKNOWN",
            },
        )
        write_json_exclusive(target, result)
        print(f"[diag_55] {result['status']} {target}")
        return 0
    except (
        DependencyUnavailable,
        FileNotFoundError,
        ProtocolError,
        ValueError,
        KeyError,
        FloatingPointError,
        RuntimeError,
        ImportError,
    ) as exc:
        return _write_error_result(
            "55",
            target,
            exc,
            skipped_summary="real sequential three-edge PhysX experiment is unavailable",
            invalid_summary="three-edge chain protocol failed closed",
        )


def _direct_edge(manifest: Mapping[str, Any]) -> dict[str, Any]:
    matches = [
        dict(edge)
        for edge in manifest["edges"]
        if "direct_final_control" in edge.get("selection_roles", ())
    ]
    if len(matches) != 1:
        raise ProtocolError("edge manifest must contain exactly one direct-final control edge")
    return matches[0]


def _validate_control_result(
    raw: Any,
    *,
    request: Mapping[str, Any],
    protocol: LocalEdgeProtocol,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ProtocolError("control backend returned a non-mapping")
    reject_deprecated_legacy(raw)
    if (
        raw.get("artifact_schema") != "largebox_real_physx_edge_control_result_v1"
        or raw.get("execution_kind") != "real_physx_ppo"
        or raw.get("physics_engine") != "PhysX"
        or raw.get("request_sha256") != canonical_sha256(request)
    ):
        raise ProtocolError("control result lacks exact real-PhysX provenance")
    seed_metrics = raw.get("seed_metrics")
    if not isinstance(seed_metrics, Mapping) or set(
        int(seed) for seed in seed_metrics
    ) != set(protocol.ppo_seeds):
        raise ProtocolError("control result lacks exactly the three PPO seed metrics")
    normalized: dict[int, dict[str, float]] = {}
    required = (
        "target_distance_reduction",
        "failure_rate_increase",
        "task_or_progress_drop",
    )
    for raw_seed, raw_metrics in seed_metrics.items():
        if not isinstance(raw_metrics, Mapping):
            raise ProtocolError("control seed metrics must be mappings")
        seed = int(raw_seed)
        normalized[seed] = {}
        for name in required:
            value = float(raw_metrics.get(name, np.nan))
            if not np.isfinite(value):
                raise ProtocolError(f"control metric {name} is non-finite")
            normalized[seed][name] = value
        artifacts = raw_metrics.get("physical_artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ProtocolError("control seed lacks persisted physical artifacts")
        for artifact in artifacts:
            path = Path(str(artifact.get("path", ""))).expanduser().resolve()
            if not path.is_file() or sha256_file(path) != str(artifact.get("sha256", "")):
                raise ProtocolError(f"control physical artifact is missing or changed: {path}")
    result: dict[str, Any] = {"valid": True, "by_seed": normalized}
    for name in (
        "target_distance_reduction",
        "failure_rate_increase",
        "task_or_progress_drop",
    ):
        result[name] = float(np.mean([normalized[seed][name] for seed in protocol.ppo_seeds]))
    result["raw"] = dict(raw)
    return result


def run_diag56() -> int:
    args = parse_stage5_args("Run direct-final AMP and fixed checkpoint schedule controls.")
    _root, _spec, output, protocol = _context(args)
    target = output / "edge_runs" / "controls" / "direct_and_schedule.json"
    try:
        manifest = load_edge_manifest(output / "edge_manifest.yaml")
        edge = _direct_edge(manifest)
        backend, backend_path, backend_hash = load_real_edge_backend(
            args.repo_root, args.backend
        )
        method = getattr(backend, "run_control_experiment", None)
        if not callable(method):
            raise DependencyUnavailable("real backend lacks run_control_experiment")
        records = {}
        for name in ("direct_final_amp", "fixed_checkpoint_schedule"):
            request = {
                "request_schema": "largebox_real_physx_edge_control_request_v1",
                "control": name,
                "edge": edge,
                "ppo_seeds": list(protocol.ppo_seeds),
                "num_envs": protocol.num_envs,
                "ppo_updates_per_edge": protocol.updates,
                "physics_engine": "PhysX",
                "output_dir": str(target.parent / name),
            }
            records[name] = _validate_control_result(
                method(request), request=request, protocol=protocol
            )
        result = diagnostic_result(
            "56",
            PASS,
            summary="direct-final AMP and fixed-schedule controls completed in real PhysX",
            evidence={
                "controls": records,
                "backend": str(backend_path),
                "backend_sha256": backend_hash,
                "decision_variable_E": "UNKNOWN",
            },
        )
        write_json_exclusive(target, result)
        print(f"[diag_56] {result['status']} {target}")
        return 0
    except (
        DependencyUnavailable,
        FileNotFoundError,
        ProtocolError,
        ValueError,
        KeyError,
        FloatingPointError,
        RuntimeError,
        ImportError,
    ) as exc:
        return _write_error_result(
            "56",
            target,
            exc,
            skipped_summary="real direct-final/fixed-schedule controls are unavailable",
            invalid_summary="direct-final/fixed-schedule control protocol failed closed",
        )


def run_diag57() -> int:
    args = parse_stage5_args("Run the real BC/action-distillation edge control.")
    _root, _spec, output, protocol = _context(args)
    target = output / "edge_runs" / "controls" / "bc_distillation.json"
    try:
        manifest = load_edge_manifest(output / "edge_manifest.yaml")
        edge = _direct_edge(manifest)
        backend, backend_path, backend_hash = load_real_edge_backend(
            args.repo_root, args.backend
        )
        method = getattr(backend, "run_control_experiment", None)
        if not callable(method):
            raise DependencyUnavailable("real backend lacks run_control_experiment")
        request = {
            "request_schema": "largebox_real_physx_edge_control_request_v1",
            "control": "bc_action_distillation",
            "edge": edge,
            "ppo_seeds": list(protocol.ppo_seeds),
            "num_envs": protocol.num_envs,
            "ppo_updates_per_edge": protocol.updates,
            "actor_observation": "actor_no_reference",
            "physics_engine": "PhysX",
            "output_dir": str(target.parent / "bc_action_distillation"),
        }
        record = _validate_control_result(
            method(request), request=request, protocol=protocol
        )
        result = diagnostic_result(
            "57",
            PASS,
            summary="BC/action-distillation control completed in real PhysX",
            evidence={
                "controls": {"bc_action_distillation": record},
                "backend": str(backend_path),
                "backend_sha256": backend_hash,
                "decision_variable_E": "UNKNOWN",
            },
        )
        write_json_exclusive(target, result)
        print(f"[diag_57] {result['status']} {target}")
        return 0
    except (
        DependencyUnavailable,
        FileNotFoundError,
        ProtocolError,
        ValueError,
        KeyError,
        FloatingPointError,
        RuntimeError,
        ImportError,
    ) as exc:
        return _write_error_result(
            "57",
            target,
            exc,
            skipped_summary="real BC/action-distillation control is unavailable",
            invalid_summary="BC/action-distillation control protocol failed closed",
        )


def _load_optional_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return diagnostic_result(
            "missing",
            SKIPPED_DEPENDENCY,
            summary=f"status artifact is absent: {path}",
        )
    payload = read_json(path)
    if payload.get("status") not in {PASS, FAIL, INVALID_PROTOCOL, SKIPPED_DEPENDENCY}:
        raise ProtocolError(f"invalid status artifact: {path}")
    return dict(payload)


def _edge_transfer_rows(payloads: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for diagnostic, payload in payloads.items():
        records = payload.get("evidence", {}).get("records", ())
        if not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, Mapping):
                continue
            gate = record.get("seed_gate", {})
            rows.append(
                {
                    "diagnostic": diagnostic,
                    "edge_id": record.get("edge_id"),
                    "source_update": record.get("source_update"),
                    "target_update": record.get("target_update"),
                    "variant": record.get("variant"),
                    "actor_observation": record.get("actor_observation"),
                    "passed_seed_count": gate.get("passed_seed_count"),
                    "total_seed_count": gate.get("total_seed_count"),
                    "edge_passed": gate.get("passed"),
                }
            )
    return rows


def _best_chain_metrics(chain_status: Mapping[str, Any]) -> dict[str, float] | None:
    chains = chain_status.get("evidence", {}).get("chains")
    if not isinstance(chains, list):
        return None
    passed = [item for item in chains if isinstance(item, Mapping) and item.get("passed") is True]
    if not passed:
        return None
    metrics = passed[0].get("result", {}).get("aggregate_metrics")
    if not isinstance(metrics, Mapping):
        raise ProtocolError("passing chain lacks aggregate_metrics")
    result = {}
    for name in (
        "target_distance_reduction",
        "failure_rate_increase",
        "task_or_progress_drop",
    ):
        value = float(metrics.get(name, np.nan))
        if not np.isfinite(value):
            raise ProtocolError(f"passing chain aggregate {name} is non-finite")
        result[name] = value
    return result


def run_diag58() -> int:
    args = parse_stage5_args("Summarize local edge causal evidence without imputing missing runs.")
    _root, _spec, output, _protocol = _context(args)
    target = output / "edge_gate.json"
    table = output / "tables" / "edge_transfer.csv"
    try:
        paths = {
            "52": output / "edge_runs" / "same_policy_class" / "summary.json",
            "53": output / "edge_runs" / "reference_free" / "summary.json",
            "54": output / "edge_runs" / "variants" / "summary.json",
            "55": output / "edge_runs" / "three_edge_chain" / "summary.json",
            "56": output / "edge_runs" / "controls" / "direct_and_schedule.json",
            "57": output / "edge_runs" / "controls" / "bc_distillation.json",
        }
        payloads = {name: _load_optional_status(path) for name, path in paths.items()}
        if any(payload["status"] == INVALID_PROTOCOL for payload in payloads.values()):
            raise ProtocolError("at least one Stage-5 online/control result is protocol-invalid")
        rows = _edge_transfer_rows(payloads)
        write_csv_exclusive(
            table,
            rows,
            fieldnames=(
                "diagnostic",
                "edge_id",
                "source_update",
                "target_update",
                "variant",
                "actor_observation",
                "passed_seed_count",
                "total_seed_count",
                "edge_passed",
            ),
        )
        missing = [
            name
            for name, payload in payloads.items()
            if payload["status"] == SKIPPED_DEPENDENCY
        ]
        controls: dict[str, Mapping[str, Any]] = {}
        for name in ("56", "57"):
            raw = payloads[name].get("evidence", {}).get("controls", {})
            if isinstance(raw, Mapping):
                controls.update(
                    {str(key): value for key, value in raw.items() if isinstance(value, Mapping)}
                )
        chain_metrics = _best_chain_metrics(payloads["55"])
        if missing:
            status = SKIPPED_DEPENDENCY
            variable = "UNKNOWN"
            comparison = {
                "all_three_outperformed": False,
                "reason": "one or more required real experiments are missing",
            }
            failure_cause = "unresolved_missing_real_experiments"
        elif chain_metrics is None:
            status = FAIL
            variable = "-"
            comparison = {
                "all_three_outperformed": False,
                "reason": "no preregistered no-reference three-edge chain passed",
            }
            same_pass = payloads["52"]["status"] == PASS
            ref_pass = payloads["53"]["status"] == PASS
            failure_cause = (
                "reference_free_policy_class_or_observability"
                if same_pass and not ref_pass
                else "local_amp_reward_or_optimizer"
            )
        else:
            comparison = baseline_outperformance(chain_metrics, controls)
            variable = "+" if comparison["all_three_outperformed"] else "-"
            status = PASS if variable == "+" else FAIL
            failure_cause = (
                "none_gate_passed"
                if variable == "+"
                else "controls_match_or_outperform_edge_chain"
            )
        result = diagnostic_result(
            "58",
            status,
            summary=(
                "local edge evidence is incomplete; causal learnability remains UNKNOWN"
                if status == SKIPPED_DEPENDENCY
                else (
                    "local edge causal gate passed all chain and baseline requirements"
                    if status == PASS
                    else "local edge causal gate was tested and failed"
                )
            ),
            evidence={
                "decision_variable_E": variable,
                "component_statuses": {
                    name: payload["status"] for name, payload in payloads.items()
                },
                "missing_components": missing,
                "edge_transfer_table": str(table),
                "edge_transfer_table_sha256": sha256_file(table),
                "edge_transfer_row_count": len(rows),
                "chain_metrics": chain_metrics,
                "baseline_outperformance": comparison,
                "failure_cause": failure_cause,
                "missing_evidence_is_negative_result": False,
            },
        )
        result["decision_variable_E"] = variable
        write_json_exclusive(target, result)
        print(f"[diag_58] {result['status']} {target}")
        return 0
    except (
        FileNotFoundError,
        ProtocolError,
        ValueError,
        KeyError,
        FloatingPointError,
        RuntimeError,
    ) as exc:
        if not table.exists():
            write_csv_exclusive(
                table,
                [],
                fieldnames=(
                    "diagnostic",
                    "edge_id",
                    "source_update",
                    "target_update",
                    "variant",
                    "actor_observation",
                    "passed_seed_count",
                    "total_seed_count",
                    "edge_passed",
                ),
            )
        return _write_error_result(
            "58",
            target,
            exc,
            skipped_summary="local edge summary awaits real experiment records",
            invalid_summary="local edge summary protocol failed closed",
        )


def run_diag59() -> int:
    args = parse_stage5_args("Enforce the fail-closed full-curriculum abort guard.")
    _root, spec, output, protocol = _context(args)
    target = output / "abort_guard.json"
    try:
        edge_gate = read_json(output / "edge_gate.json")
        if edge_gate.get("status") == INVALID_PROTOCOL:
            raise ProtocolError("diag_58 edge gate is protocol-invalid")
        evidence = edge_gate.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ProtocolError("diag_58 edge gate lacks evidence")
        variable = str(evidence.get("decision_variable_E", "UNKNOWN"))
        if variable not in {"+", "-", "UNKNOWN"}:
            raise ProtocolError("diag_58 decision variable E is invalid")
        comparison = evidence.get("baseline_outperformance", {})
        chain_metrics = evidence.get("chain_metrics")
        authorized = bool(
            variable == "+"
            and isinstance(chain_metrics, Mapping)
            and isinstance(comparison, Mapping)
            and comparison.get("all_three_outperformed") is True
        )
        if variable == "+" and not authorized:
            raise ProtocolError("E+ lacks a passing chain and all three baseline controls")
        full_dir = output / "edge_runs" / "full_curriculum"
        preexisting_full_training = full_dir.exists() and any(full_dir.iterdir())
        if preexisting_full_training and not authorized:
            raise ProtocolError(
                "full curriculum artifacts exist despite a closed abort guard"
            )
        result = diagnostic_result(
            "59",
            PASS,
            summary=(
                "full curriculum entry is authorized by the exact preregistered gate"
                if authorized
                else "full curriculum was successfully blocked; evidence is insufficient or negative"
            ),
            evidence={
                "guard_pass": authorized,
                "curriculum_entry_authorized": authorized,
                "decision_variable_E": variable,
                "minimum_consecutive_no_reference_edges": protocol.minimum_chain_edges,
                "required_controls": list(protocol.controls),
                "all_three_controls_outperformed": bool(
                    isinstance(comparison, Mapping)
                    and comparison.get("all_three_outperformed") is True
                ),
                "full_curriculum_started_by_guard": False,
                "unauthorized_full_curriculum_artifacts_found": False,
                "missing_evidence_treated_as_negative_result": False,
                "missing_evidence_action": "deny authorization and preserve UNKNOWN",
                "source_edge_gate": str(output / "edge_gate.json"),
                "source_edge_gate_sha256": sha256_file(output / "edge_gate.json"),
            },
        )
        result["guard_pass"] = authorized
        result["curriculum_entry_authorized"] = authorized
        result["decision_variable_E"] = variable
        write_json_exclusive(target, result)
        print(f"[diag_59] {result['status']} {target}")
        return 0
    except (
        FileNotFoundError,
        ProtocolError,
        ValueError,
        KeyError,
        FloatingPointError,
        RuntimeError,
    ) as exc:
        return _write_error_result(
            "59",
            target,
            exc,
            skipped_summary="abort guard awaits an edge-gate record",
            invalid_summary="abort guard protocol failed closed",
        )


ENTRYPOINTS = {
    "50": run_diag50,
    "51": run_diag51,
    "52": run_diag52,
    "53": run_diag53,
    "54": run_diag54,
    "55": run_diag55,
    "56": run_diag56,
    "57": run_diag57,
    "58": run_diag58,
    "59": run_diag59,
}


def run_stage5_entrypoint(diagnostic_id: str) -> int:
    try:
        function = ENTRYPOINTS[str(diagnostic_id)]
    except KeyError as exc:  # pragma: no cover
        raise ValueError(f"unknown Stage-5 diagnostic {diagnostic_id}") from exc
    return function()


__all__ = ["run_stage5_entrypoint"]
