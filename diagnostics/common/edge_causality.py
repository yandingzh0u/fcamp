"""Fail-closed contracts for Stage-5 local-edge causal diagnostics.

This module is deliberately *not* an AMP implementation.  It preregisters
edges from recorded PhysX outcomes, keeps target-positive buffers isolated,
audits evaluation critics, and validates artifacts emitted by an optional real
PhysX/PPO backend.  If that backend is absent, callers must report
``SKIPPED_DEPENDENCY``; replay, numpy optimization, or the repository's
fixed-reward PPO are never accepted as substitutes.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .domain_triangle import effective_overlap_audit
from .manifest import (
    DependencyUnavailable,
    ProtocolError,
    canonical_sha256,
    read_json,
    sha256_file,
)
from .quality_panel import OutcomeMetric, pareto_preference
from .reward_validity import reward_seed_agreement


ALLOWED_PRIMARY_DOMAINS = ("K", "T_u200", "T_u500", "A_amp", "B")
FORBIDDEN_LEGACY_TOKENS = ("A_mix", "FCAMP", "causal-GRU", "causal_gru", "H4")
EDGE_MANIFEST_SCHEMA = "largebox_local_edge_manifest_v1"
POSITIVE_BUFFER_SCHEMA = "largebox_edge_positive_buffer_v1"
ONLINE_RESULT_SCHEMA = "largebox_real_physx_local_edge_result_v1"
BACKEND_SCHEMA = "largebox_real_physx_amp_edge_backend_v1"
EVALUATION_CRITIC_ROLE = "frozen_diag35_evaluation_only"
TRAINING_CRITIC_ROLE = "edge_training_only"
TEACHER_DOMAIN = "teacher_fixed_reward"


def reject_deprecated_legacy(value: Any, *, path: str = "root") -> None:
    """Reject the explicitly quarantined A_mix/FCAMP/H4 research branch."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            reject_deprecated_legacy(key, path=f"{path}.<key>")
            reject_deprecated_legacy(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple, set)):
        for index, item in enumerate(value):
            reject_deprecated_legacy(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str):
        lowered = value.lower()
        for token in FORBIDDEN_LEGACY_TOKENS:
            if token.lower() in lowered:
                raise ProtocolError(
                    f"deprecated legacy token {token!r} is forbidden in Stage 5 ({path})"
                )


@dataclass(frozen=True, slots=True)
class LocalEdgeProtocol:
    ppo_seeds: tuple[int, ...]
    held_out_critic_seeds: tuple[int, ...]
    num_envs: int
    updates: int
    evaluation_every: int
    rollout_steps: int
    evaluation_snapshots: int
    candidate_updates: tuple[int, ...]
    transition_levels: tuple[float, ...]
    same_policy_variants: tuple[str, ...]
    reference_free_variants: tuple[str, ...]
    factorial_variants: tuple[str, ...]
    controls: tuple[str, ...]
    minimum_passing_seeds: int
    target_distance_reduction_min: float
    failure_increase_max: float
    progress_drop_max: float
    minimum_chain_edges: int
    reward_icc_min: float
    overlap_threshold: Mapping[str, Any]

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "LocalEdgeProtocol":
        raw = spec.get("analysis_protocols", {}).get("local_edge_causality")
        edge_threshold = spec.get("thresholds", {}).get("single_edge_pass")
        chain_threshold = spec.get("thresholds", {}).get("curriculum_entry")
        overlap = spec.get("thresholds", {}).get("empirical_effective_overlap_edge")
        if not all(
            isinstance(item, Mapping)
            for item in (raw, edge_threshold, chain_threshold, overlap)
        ):
            raise ProtocolError("Stage-5 protocol/threshold mappings are not frozen")
        reject_deprecated_legacy(raw, path="analysis_protocols.local_edge_causality")
        values = cls(
            ppo_seeds=tuple(int(x) for x in raw.get("ppo_seeds", ())),
            held_out_critic_seeds=tuple(
                int(x) for x in raw.get("held_out_critic_seeds", ())
            ),
            num_envs=int(raw.get("num_envs", 0)),
            updates=int(raw.get("ppo_updates_per_edge", 0)),
            evaluation_every=int(raw.get("evaluation_every_updates", 0)),
            rollout_steps=int(raw.get("rollout_steps_per_update", 0)),
            evaluation_snapshots=int(raw.get("canonical_evaluation_snapshots", 0)),
            candidate_updates=tuple(
                int(x) for x in raw.get("edge_selection", {}).get("candidate_updates", ())
            ),
            transition_levels=tuple(
                float(x)
                for x in raw.get("edge_selection", {}).get(
                    "transition_completion_levels", ()
                )
            ),
            same_policy_variants=tuple(
                str(x) for x in raw.get("same_policy_class_variants", ())
            ),
            reference_free_variants=tuple(
                str(x) for x in raw.get("reference_free_critic_variants", ())
            ),
            factorial_variants=tuple(
                str(x) for x in raw.get("edge_factorial_variants", ())
            ),
            controls=tuple(str(x) for x in raw.get("controls", ())),
            minimum_passing_seeds=int(
                edge_threshold.get("minimum_passing_ppo_seeds", 0)
            ),
            target_distance_reduction_min=float(
                edge_threshold.get("target_distance_reduction_min", np.nan)
            ),
            failure_increase_max=float(
                edge_threshold.get("failure_rate_increase_max", np.nan)
            ),
            progress_drop_max=float(
                edge_threshold.get("task_or_progress_drop_max", np.nan)
            ),
            minimum_chain_edges=int(
                chain_threshold.get(
                    "minimum_consecutive_preregistered_no_reference_edges", 0
                )
            ),
            reward_icc_min=float(overlap.get("reward_icc_min", np.nan)),
            overlap_threshold=dict(overlap),
        )
        expected = {
            "ppo_seeds": (20260803, 20260804, 20260805),
            "critic_seeds": (20260803, 20260804, 20260805, 20260806, 20260807),
            "same": (
                "frozen_discriminator",
                "online_discriminator",
                "constant_reward_ppo",
                "no_update_replay",
            ),
            "reference_free": ("privileged_critic", "no_reference_critic"),
            "factorial": (
                "forward_frozen",
                "forward_online",
                "reverse_frozen",
                "reverse_online",
            ),
            "controls": (
                "direct_final_amp",
                "fixed_checkpoint_schedule",
                "bc_action_distillation",
            ),
        }
        if values.ppo_seeds != expected["ppo_seeds"]:
            raise ProtocolError("Stage-5 PPO seeds changed from the frozen three seeds")
        if values.held_out_critic_seeds != expected["critic_seeds"]:
            raise ProtocolError("Stage-5 held-out critic seeds changed")
        if (
            values.num_envs != 256
            or values.updates != 50
            or values.evaluation_every != 5
            or values.rollout_steps != 24
            or values.evaluation_snapshots != 64
        ):
            raise ProtocolError("Stage-5 online/evaluation budget changed")
        if values.transition_levels != (0.10, 0.50, 0.90):
            raise ProtocolError("teacher transition levels changed")
        if tuple(sorted(set(values.candidate_updates))) != values.candidate_updates:
            raise ProtocolError("candidate updates must be sorted and unique")
        if values.same_policy_variants != expected["same"]:
            raise ProtocolError("same-policy-class variants changed")
        if values.reference_free_variants != expected["reference_free"]:
            raise ProtocolError("reference-free critic variants changed")
        if values.factorial_variants != expected["factorial"]:
            raise ProtocolError("edge factorial changed")
        if values.controls != expected["controls"]:
            raise ProtocolError("Stage-5 controls changed")
        if values.minimum_passing_seeds != 2 or values.minimum_chain_edges != 3:
            raise ProtocolError("Stage-5 pass counts changed")
        scalars = (
            values.target_distance_reduction_min,
            values.failure_increase_max,
            values.progress_drop_max,
            values.reward_icc_min,
        )
        if not all(np.isfinite(x) and 0.0 <= x <= 1.0 for x in scalars):
            raise ProtocolError("Stage-5 thresholds are not finite probabilities")
        selection = raw.get("edge_selection", {})
        if (
            selection.get("target_must_strictly_pareto_dominate_source") is not True
            or selection.get("selection_per_transition")
            != ["maximum_eligible_overlap", "median_eligible_overlap"]
            or selection.get("selection_before_online_updates") is not True
            or raw.get("positive_rollout_mode") != "controlled_environment"
            or raw.get("positive_window_role") != "agent_physx_raw_frame"
            or raw.get("full_curriculum_forbidden_until_abort_guard_pass") is not True
        ):
            raise ProtocolError("Stage-5 causal guards differ from the frozen spec")
        return values

    @property
    def evaluation_updates(self) -> tuple[int, ...]:
        return tuple(range(0, self.updates + 1, self.evaluation_every))


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DependencyUnavailable(f"required table is missing: {source}")
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    if not rows:
        raise DependencyUnavailable(f"required table is empty: {source}")
    return rows


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{label} is not numeric: {value!r}") from exc
    if not np.isfinite(result):
        raise ProtocolError(f"{label} is not finite")
    return result


def aggregate_teacher_outcomes(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[OutcomeMetric],
) -> dict[int, dict[str, Any]]:
    """Aggregate only condition-matched controlled teacher PhysX outcomes."""

    filtered = [
        row
        for row in rows
        if str(row.get("policy_domain")) == TEACHER_DOMAIN
        and str(row.get("collector_mode")) == "controlled_environment"
        and _finite_float(row.get("common_sigma"), "common_sigma") == 0.0
    ]
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in filtered:
        grouped.setdefault(int(row["checkpoint_update"]), []).append(row)
    result: dict[int, dict[str, Any]] = {}
    metric_names = tuple(metric.name for metric in metrics)
    for update, group in grouped.items():
        hashes = {str(row.get("checkpoint_sha256", "")) for row in group}
        if len(hashes) != 1 or len(next(iter(hashes), "")) != 64:
            raise ProtocolError(f"teacher update {update} has ambiguous checkpoint identity")
        snapshots = [str(row.get("snapshot_id", "")) for row in group]
        if len(set(snapshots)) != len(snapshots):
            raise ProtocolError(f"teacher update {update} repeats a controlled snapshot")
        means = {
            name: float(np.mean([_finite_float(row[name], name) for row in group]))
            for name in metric_names
        }
        result[update] = {
            "update": update,
            "checkpoint_sha256": next(iter(hashes)),
            "sample_count": len(group),
            "snapshot_ids": sorted(snapshots),
            "outcome_means": means,
        }
    return result


def outcome_edge_evidence(
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    *,
    metrics: Sequence[OutcomeMetric],
) -> dict[str, Any]:
    preference = pareto_preference(
        target["outcome_means"], source["outcome_means"], metrics=metrics
    )
    common_snapshots = sorted(
        set(source.get("snapshot_ids", ())) & set(target.get("snapshot_ids", ()))
    )
    if not common_snapshots:
        raise ProtocolError("candidate teacher edge has no condition-matched snapshots")
    deltas = {
        metric.name: float(
            target["outcome_means"][metric.name]
            - source["outcome_means"][metric.name]
        )
        for metric in metrics
    }
    return {
        "strict_pareto_target_over_source": preference == 1,
        "pareto_preference": int(preference),
        "source_outcome_means": dict(source["outcome_means"]),
        "target_outcome_means": dict(target["outcome_means"]),
        "target_minus_source": deltas,
        "condition_matched_snapshot_count": len(common_snapshots),
        "condition_matched_snapshot_ids_sha256": canonical_sha256(common_snapshots),
    }


def transition_crossings(
    teacher_status: Mapping[str, Any], protocol: LocalEdgeProtocol
) -> dict[float, int]:
    if teacher_status.get("status") != "PASS":
        raise DependencyUnavailable("diag_05 teacher transition is not PASS")
    evidence = teacher_status.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ProtocolError("teacher transition lacks evidence")
    summaries = evidence.get("curve_summaries")
    if not isinstance(summaries, list):
        raise ProtocolError("teacher transition lacks curve summaries")
    primary = [
        row
        for row in summaries
        if isinstance(row, Mapping) and row.get("is_frozen_checkpoint_run") is True
    ]
    if len(primary) != 1:
        raise ProtocolError("teacher transition must identify exactly one frozen curve")
    raw = primary[0].get("completion_crossing_updates")
    if not isinstance(raw, Mapping):
        raise ProtocolError("frozen teacher curve lacks completion crossings")
    crossings: dict[float, int] = {}
    for level in protocol.transition_levels:
        candidates = (str(level), f"{level:.1f}", f"{level:.2f}")
        value = next((raw[key] for key in candidates if key in raw), None)
        if value is None:
            raise DependencyUnavailable(
                f"teacher never recorded the {level:.0%} completion crossing"
            )
        crossings[level] = int(value)
    if not (crossings[0.10] <= crossings[0.50] <= crossings[0.90]):
        raise ProtocolError("teacher completion crossings are not ordered")
    return crossings


def summarize_overlap_audits(
    audits: Sequence[Mapping[str, Any]],
    *,
    threshold: Mapping[str, Any],
    reward_icc: float,
) -> dict[str, Any]:
    if not audits:
        raise ProtocolError("an edge requires at least one overlap audit")
    tau = str(float(threshold["posterior_overlap_tau"]))

    def mean(getter) -> float:
        values = np.asarray([float(getter(item)) for item in audits], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ProtocolError("edge overlap audit contains non-finite metrics")
        return float(values.mean())

    record = {
        "source_auc": mean(lambda x: x["classifier"]["auc"]),
        "posterior_overlap": mean(
            lambda x: x["posterior_overlap"][tau]["balanced"]
        ),
        "forward_knn_coverage": mean(
            lambda x: x["knn"]["source_to_target_coverage"]
        ),
        "reverse_knn_coverage": mean(
            lambda x: x["knn"]["target_to_source_coverage"]
        ),
        "forward_unclipped_ess": mean(
            lambda x: x["ratio_ess"]["forward"]["unclipped"]["ess_fraction"]
        ),
        "reverse_unclipped_ess": mean(
            lambda x: x["ratio_ess"]["reverse"]["unclipped"]["ess_fraction"]
        ),
        "held_out_reward_icc": float(reward_icc),
        "by_seed": [dict(item) for item in audits],
    }
    gates = {
        "source_auc": record["source_auc"] <= float(threshold["source_auc_max"]),
        "posterior_overlap": record["posterior_overlap"]
        >= float(threshold["posterior_overlap_fraction_min"]),
        "forward_knn_coverage": record["forward_knn_coverage"]
        >= float(threshold["forward_knn_coverage_min"]),
        "reverse_knn_coverage": record["reverse_knn_coverage"]
        >= float(threshold["reverse_knn_coverage_min"]),
        "forward_unclipped_ess": record["forward_unclipped_ess"]
        >= float(threshold["forward_ess_over_n_min"]),
        "reverse_unclipped_ess": record["reverse_unclipped_ess"]
        >= float(threshold["reverse_ess_over_n_min"]),
        "reward_icc": record["held_out_reward_icc"]
        >= float(threshold["reward_icc_min"]),
    }
    record["gates"] = gates
    record["eligible"] = bool(all(gates.values()))
    # Frozen, deterministic overlap ordering: the weakest normalized gate is
    # primary; posterior overlap, bidirectional coverage/ESS, and inverse AUC
    # resolve ties.  No online result participates in this tuple.
    normalized = (
        (float(threshold["source_auc_max"]) - record["source_auc"] + 1.0e-12)
        / max(float(threshold["source_auc_max"]), 1.0e-12),
        record["posterior_overlap"]
        / max(float(threshold["posterior_overlap_fraction_min"]), 1.0e-12),
        record["forward_knn_coverage"]
        / max(float(threshold["forward_knn_coverage_min"]), 1.0e-12),
        record["reverse_knn_coverage"]
        / max(float(threshold["reverse_knn_coverage_min"]), 1.0e-12),
        record["forward_unclipped_ess"]
        / max(float(threshold["forward_ess_over_n_min"]), 1.0e-12),
        record["reverse_unclipped_ess"]
        / max(float(threshold["reverse_ess_over_n_min"]), 1.0e-12),
    )
    record["overlap_order_key"] = [
        float(min(normalized)),
        record["posterior_overlap"],
        min(record["forward_knn_coverage"], record["reverse_knn_coverage"]),
        min(record["forward_unclipped_ess"], record["reverse_unclipped_ess"]),
        -record["source_auc"],
    ]
    return record


def compute_edge_overlap(
    source_splits: Mapping[str, np.ndarray],
    target_splits: Mapping[str, np.ndarray],
    *,
    protocol: LocalEdgeProtocol,
    reward_icc: float,
) -> dict[str, Any]:
    for label, splits in (("source", source_splits), ("target", target_splits)):
        if set(splits) != {"train", "validation", "test"}:
            raise ProtocolError(f"{label} edge windows lack a frozen split")
    audits = [
        effective_overlap_audit(
            source_splits["validation"],
            target_splits["validation"],
            source_splits["test"],
            target_splits["test"],
            source_train=source_splits["train"],
            target_train=target_splits["train"],
            seed=seed,
            taus=tuple(
                float(x)
                for x in protocol.overlap_threshold[
                    "posterior_overlap_report_taus"
                ]
            ),
            ratio_clips=tuple(
                float(x) for x in protocol.overlap_threshold["ratio_clips"]
            ),
        )
        for seed in protocol.held_out_critic_seeds
    ]
    return summarize_overlap_audits(
        audits, threshold=protocol.overlap_threshold, reward_icc=reward_icc
    )


def select_preregistered_edges(
    candidate_edges: Sequence[Mapping[str, Any]],
    *,
    crossings: Mapping[float, int],
    final_update: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select maximum/median overlap per transition and a direct-final control."""

    eligible = [
        dict(edge)
        for edge in candidate_edges
        if edge.get("outcome", {}).get("strict_pareto_target_over_source") is True
        and edge.get("overlap", {}).get("eligible") is True
    ]
    for edge in eligible:
        reject_deprecated_legacy(edge)
        edge.setdefault("selection_roles", [])
    selected: dict[str, dict[str, Any]] = {}
    audit: list[dict[str, Any]] = []

    def rank_key(edge: Mapping[str, Any]) -> tuple[Any, ...]:
        overlap_key = tuple(float(x) for x in edge["overlap"]["overlap_order_key"])
        gap = int(edge["target_update"]) - int(edge["source_update"])
        return (*overlap_key, -gap, str(edge["edge_id"]))

    for level in sorted(crossings):
        crossing = int(crossings[level])
        pool = [
            edge
            for edge in eligible
            if int(edge["source_update"]) < crossing <= int(edge["target_update"])
        ]
        ordered = sorted(pool, key=rank_key)
        record: dict[str, Any] = {
            "transition_level": float(level),
            "crossing_update": crossing,
            "eligible_edge_ids": [str(edge["edge_id"]) for edge in ordered],
        }
        if ordered:
            choices = {
                "maximum_eligible_overlap": ordered[-1],
                "median_eligible_overlap": ordered[(len(ordered) - 1) // 2],
            }
            record["selected"] = {
                role: str(edge["edge_id"]) for role, edge in choices.items()
            }
            for role, edge in choices.items():
                edge_id = str(edge["edge_id"])
                chosen = selected.setdefault(edge_id, dict(edge))
                chosen.setdefault("selection_roles", []).append(
                    f"completion_{level:.2f}:{role}"
                )
        else:
            record["selected"] = {}
        audit.append(record)

    direct_pool = [
        edge
        for edge in eligible
        if int(edge["target_update"]) == int(final_update)
        and int(edge["source_update"]) < int(final_update)
    ]
    if direct_pool:
        direct = min(
            direct_pool,
            key=lambda edge: (int(edge["source_update"]), str(edge["edge_id"])),
        )
        edge_id = str(direct["edge_id"])
        chosen = selected.setdefault(edge_id, dict(direct))
        chosen.setdefault("selection_roles", []).append("direct_final_control")
        audit.append(
            {
                "control": "direct_final",
                "selected": edge_id,
                "rule": "earliest confirmed lower-quality checkpoint to final update",
            }
        )
    else:
        audit.append({"control": "direct_final", "selected": None})
    result = sorted(
        selected.values(),
        key=lambda edge: (
            int(edge["source_update"]),
            int(edge["target_update"]),
            str(edge["edge_id"]),
        ),
    )
    for edge in result:
        edge["selection_roles"] = sorted(set(edge.get("selection_roles", ())))
    return result, audit


def candidate_chains(edges: Sequence[Mapping[str, Any]], *, length: int = 3) -> list[list[str]]:
    by_source: dict[int, list[Mapping[str, Any]]] = {}
    for edge in edges:
        by_source.setdefault(int(edge["source_update"]), []).append(edge)
    chains: list[list[str]] = []

    def visit(edge: Mapping[str, Any], path: list[str]) -> None:
        next_path = [*path, str(edge["edge_id"])]
        if len(next_path) == int(length):
            chains.append(next_path)
            return
        for next_edge in sorted(
            by_source.get(int(edge["target_update"]), ()),
            key=lambda item: str(item["edge_id"]),
        ):
            visit(next_edge, next_path)

    for edge in sorted(edges, key=lambda item: str(item["edge_id"])):
        visit(edge, [])
    return sorted(chains)


def write_yaml_exclusive(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise DependencyUnavailable("PyYAML is required for edge_manifest.yaml") from exc
    with target.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False, allow_unicode=True)
    return target


def load_edge_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DependencyUnavailable(f"edge manifest is missing: {source}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise DependencyUnavailable("PyYAML is required to read edge_manifest.yaml") from exc
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping) or payload.get("artifact_schema") != EDGE_MANIFEST_SCHEMA:
        raise ProtocolError("edge manifest schema is invalid")
    result = dict(payload)
    reject_deprecated_legacy(result)
    edges = result.get("edges")
    if not isinstance(edges, list) or not edges:
        raise DependencyUnavailable("edge manifest has no preregistered edges")
    identities = [str(edge.get("edge_id", "")) for edge in edges]
    if not all(identities) or len(set(identities)) != len(identities):
        raise ProtocolError("edge manifest identities are empty or duplicated")
    return result


def build_positive_buffer_manifest(
    index_rows: Sequence[Mapping[str, Any]],
    edge: Mapping[str, Any],
    *,
    split: str = "train",
) -> dict[str, Any]:
    """Bind an edge positive buffer to target-j controlled agent rollouts only."""

    target_update = int(edge["target_update"])
    target_hash = str(edge["target_checkpoint_sha256"])
    selected = [
        row
        for row in index_rows
        if str(row.get("policy_domain")) == TEACHER_DOMAIN
        and int(row.get("checkpoint_update", -1)) == target_update
        and str(row.get("checkpoint_sha256", "")) == target_hash
        and str(row.get("collector_mode")) == "controlled_environment"
        and float(row.get("common_sigma", np.nan)) == 0.0
        and bool(row.get("eligible_for_primary_overlap"))
        and str(row.get("split", split)) == split
    ]
    if not selected:
        raise DependencyUnavailable(
            f"edge {edge['edge_id']} has no target-j controlled {split} positive rows"
        )
    sample_ids = sorted(str(row.get("sample_id", "")) for row in selected)
    if not all(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise ProtocolError("edge positive buffer sample identities are invalid")
    snapshots = sorted(str(row.get("snapshot_id", "")) for row in selected)
    if not all(snapshots):
        raise ProtocolError("edge positive buffer lacks snapshot identities")
    manifest = {
        "artifact_schema": POSITIVE_BUFFER_SCHEMA,
        "edge_id": str(edge["edge_id"]),
        "positive_domain": f"T_u{target_update}",
        "policy_domain": TEACHER_DOMAIN,
        "target_update": target_update,
        "target_checkpoint_sha256": target_hash,
        "collector_mode": "controlled_environment",
        "common_sigma": 0.0,
        "frame_role": "agent_physx_raw_frame",
        "split": split,
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        "sample_ids_sha256": canonical_sha256(sample_ids),
        "snapshot_ids_sha256": canonical_sha256(snapshots),
        "forbidden_positive_sources": ["raw_human_reference", "K", "current_policy"],
    }
    validate_positive_buffer_isolation(manifest)
    return manifest


def validate_positive_buffer_isolation(
    positive: Mapping[str, Any],
    *,
    negative_sample_ids: Sequence[str] = (),
) -> None:
    if positive.get("artifact_schema") != POSITIVE_BUFFER_SCHEMA:
        raise ProtocolError("edge positive buffer schema is invalid")
    reject_deprecated_legacy(positive)
    if (
        positive.get("policy_domain") != TEACHER_DOMAIN
        or positive.get("collector_mode") != "controlled_environment"
        or float(positive.get("common_sigma", np.nan)) != 0.0
        or positive.get("frame_role") != "agent_physx_raw_frame"
        or not str(positive.get("positive_domain", "")).startswith("T_u")
    ):
        raise ProtocolError("positive buffer is not an isolated target-j PhysX buffer")
    samples = tuple(str(x) for x in positive.get("sample_ids", ()))
    if len(samples) != int(positive.get("sample_count", -1)) or len(set(samples)) != len(samples):
        raise ProtocolError("positive buffer sample count/identity is inconsistent")
    if positive.get("sample_ids_sha256") != canonical_sha256(sorted(samples)):
        raise ProtocolError("positive buffer sample hash is inconsistent")
    overlap = set(samples) & {str(x) for x in negative_sample_ids}
    if overlap:
        raise ProtocolError(
            f"positive and policy-negative buffers overlap: {sorted(overlap)[:3]}"
        )


def diag35_evaluation_critic_manifest(
    output_dir: str | Path,
    protocol: LocalEdgeProtocol,
) -> dict[str, Any]:
    root = Path(output_dir).expanduser().resolve()
    status = read_json(root / "ratio_reliability.json")
    if status.get("status") != "PASS":
        raise DependencyUnavailable("diag_35 reward reliability is not PASS")
    evidence = status.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ProtocolError("diag_35 status lacks evidence")
    index_path = Path(str(evidence.get("offline_amp_artifact_index", ""))).expanduser().resolve()
    if (
        not index_path.is_file()
        or sha256_file(index_path)
        != str(evidence.get("offline_amp_artifact_index_sha256", ""))
    ):
        raise ProtocolError("diag_35 critic index is missing or changed")
    index = read_json(index_path)
    records = [
        dict(item)
        for item in index.get("models", ())
        if isinstance(item, Mapping)
        and item.get("source_negative") == "A_amp"
        and item.get("destination_positive") == "T_u500"
    ]
    records.sort(key=lambda item: int(item.get("seed", -1)))
    if tuple(int(item.get("seed", -1)) for item in records) != protocol.held_out_critic_seeds:
        raise DependencyUnavailable("diag_35 lacks five A_amp->T_u500 evaluation critics")
    artifacts = []
    for record in records:
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        digest = str(record.get("sha256", ""))
        if not path.is_file() or sha256_file(path) != digest:
            raise ProtocolError(f"diag_35 evaluation critic changed: {path}")
        artifacts.append(
            {
                "seed": int(record["seed"]),
                "path": str(path),
                "sha256": digest,
                "source_negative": "A_amp",
                "destination_positive": "T_u500",
                "role": EVALUATION_CRITIC_ROLE,
            }
        )
    result = {
        "role": EVALUATION_CRITIC_ROLE,
        "source_index": str(index_path),
        "source_index_sha256": sha256_file(index_path),
        "artifacts": artifacts,
    }
    validate_evaluation_critics(
        result, training_critic_artifacts=(), expected_seeds=protocol.held_out_critic_seeds
    )
    return result


def validate_evaluation_critics(
    evaluation: Mapping[str, Any],
    *,
    training_critic_artifacts: Sequence[Mapping[str, Any]],
    expected_seeds: Sequence[int],
) -> None:
    if evaluation.get("role") != EVALUATION_CRITIC_ROLE:
        raise ProtocolError("critic used for evaluation is not held out from edge training")
    records = evaluation.get("artifacts")
    if not isinstance(records, list):
        raise ProtocolError("evaluation critic manifest lacks artifacts")
    seeds = tuple(int(item.get("seed", -1)) for item in records)
    if seeds != tuple(int(x) for x in expected_seeds):
        raise ProtocolError("evaluation critic seeds differ from the frozen five seeds")
    eval_ids = {
        (str(item.get("path", "")), str(item.get("sha256", ""))) for item in records
    }
    if len(eval_ids) != len(records) or any(len(digest) != 64 for _, digest in eval_ids):
        raise ProtocolError("evaluation critic identities are malformed or duplicated")
    train_ids = {
        (str(item.get("path", "")), str(item.get("sha256", "")))
        for item in training_critic_artifacts
    }
    train_hashes = {digest for _, digest in train_ids}
    eval_hashes = {digest for _, digest in eval_ids}
    if train_ids & eval_ids or train_hashes & eval_hashes:
        raise ProtocolError("edge training critic leaked into held-out evaluation")
    if any(item.get("role") != EVALUATION_CRITIC_ROLE for item in records):
        raise ProtocolError("evaluation artifact has a non-held-out role")


def held_out_reward_icc(output_dir: str | Path) -> float:
    status = read_json(Path(output_dir).expanduser().resolve() / "ratio_reliability.json")
    if status.get("status") != "PASS":
        raise DependencyUnavailable("diag_35 reward reliability is not PASS")
    pairs = status.get("evidence", {}).get("pairs")
    if not isinstance(pairs, list):
        raise ProtocolError("diag_35 status lacks directed-pair evidence")
    matches = [
        item
        for item in pairs
        if isinstance(item, Mapping)
        and item.get("source_negative") == "A_amp"
        and item.get("destination_positive") == "T_u500"
    ]
    if len(matches) != 1:
        raise DependencyUnavailable("diag_35 lacks A_amp->T_u500 reliability evidence")
    return _finite_float(
        matches[0].get("reward_ordering_stability", {}).get("icc_consistency"),
        "diag35 held-out reward ICC",
    )


def _artifact_identity(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record.get("path", "")), str(record.get("sha256", ""))


def load_real_edge_backend(
    repo_root: str | Path,
    explicit: str | Path | None = None,
) -> tuple[ModuleType, Path, str]:
    """Load an explicitly real backend or fail; there is no fallback backend."""

    root = Path(repo_root).expanduser().resolve()
    path = (
        Path(explicit).expanduser().resolve()
        if explicit is not None
        else (root / "diagnostics" / "runtime" / "local_edge_amp_backend.py").resolve()
    )
    if not path.is_file():
        raise DependencyUnavailable(
            "real local-edge AMP backend is absent; this repository exposes only "
            "fixed_reward PPO, which is not a valid substitute"
        )
    spec = importlib.util.spec_from_file_location("largebox_local_edge_backend", path)
    if spec is None or spec.loader is None:
        raise DependencyUnavailable(f"cannot import local-edge backend: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if getattr(module, "LOCAL_EDGE_BACKEND_SCHEMA", None) != BACKEND_SCHEMA:
        raise ProtocolError("local-edge backend schema does not certify real PhysX AMP PPO")
    if getattr(module, "EXECUTION_KIND", None) != "real_physx_ppo":
        raise ProtocolError("local-edge backend execution kind is not real_physx_ppo")
    if not callable(getattr(module, "run_experiment", None)):
        raise ProtocolError("local-edge backend lacks run_experiment(request)")
    return module, path, sha256_file(path)


def build_backend_request(
    *,
    diagnostic_id: str,
    edge: Mapping[str, Any],
    variant: str,
    seed: int,
    actor_observation: str,
    critic_observation: str,
    protocol: LocalEdgeProtocol,
    positive_buffer: Mapping[str, Any],
    evaluation_critics: Mapping[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    request = {
        "request_schema": "largebox_real_physx_local_edge_request_v1",
        "diagnostic_id": str(diagnostic_id),
        "edge": dict(edge),
        "variant": str(variant),
        "seed": int(seed),
        "execution": {
            "physics_engine": "PhysX",
            "num_envs": protocol.num_envs,
            "ppo_updates": protocol.updates,
            "rollout_steps_per_update": protocol.rollout_steps,
            "evaluation_updates": list(protocol.evaluation_updates),
            "canonical_evaluation_snapshots": protocol.evaluation_snapshots,
        },
        "actor_observation": actor_observation,
        "critic_observation": critic_observation,
        "positive_buffer": dict(positive_buffer),
        "evaluation_critics": dict(evaluation_critics),
        "output_path": str(output_path),
        "fixed_reward_weight": 0.0,
        "amp_contract": "exact_repository_BCE_gradient_penalty_softplus_contract",
        "training_critic_role": TRAINING_CRITIC_ROLE,
    }
    reject_deprecated_legacy(request)
    return request


def validate_real_online_result(
    result: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    protocol: LocalEdgeProtocol,
) -> dict[str, Any]:
    """Validate real backend evidence before any scientific pass computation."""

    if result.get("artifact_schema") != ONLINE_RESULT_SCHEMA:
        raise ProtocolError("online edge result schema is invalid")
    reject_deprecated_legacy(result)
    if result.get("execution_kind") != "real_physx_ppo" or result.get("physics_engine") != "PhysX":
        raise ProtocolError("online edge result is not a real PhysX PPO experiment")
    if int(result.get("seed", -1)) != int(request["seed"]):
        raise ProtocolError("online result seed differs from the request")
    if result.get("edge_id") != request["edge"]["edge_id"] or result.get("variant") != request["variant"]:
        raise ProtocolError("online result edge/variant differs from the request")
    if int(result.get("num_envs", -1)) != protocol.num_envs or int(
        result.get("ppo_updates", -1)
    ) != protocol.updates:
        raise ProtocolError("online edge result used the wrong training budget")
    if result.get("request_sha256") != canonical_sha256(request):
        raise ProtocolError("online result is not bound to the exact backend request")
    if result.get("positive_buffer_sha256") != canonical_sha256(request["positive_buffer"]):
        raise ProtocolError("online result used a different positive buffer")
    training_critics = result.get("training_critic_artifacts")
    if not isinstance(training_critics, list):
        raise ProtocolError("online result lacks edge-training critic artifacts")
    if any(item.get("role") != TRAINING_CRITIC_ROLE for item in training_critics):
        raise ProtocolError("online result training critic role is invalid")
    validate_evaluation_critics(
        request["evaluation_critics"],
        training_critic_artifacts=training_critics,
        expected_seeds=protocol.held_out_critic_seeds,
    )
    negative_ids = tuple(str(x) for x in result.get("policy_negative_sample_ids", ()))
    if not negative_ids:
        raise ProtocolError("online edge result lacks policy-negative sample identities")
    validate_positive_buffer_isolation(
        request["positive_buffer"], negative_sample_ids=negative_ids
    )
    evaluations = result.get("evaluations")
    if not isinstance(evaluations, list):
        raise ProtocolError("online edge result lacks canonical evaluations")
    actual_updates = tuple(int(item.get("update", -1)) for item in evaluations)
    if actual_updates != protocol.evaluation_updates:
        raise ProtocolError("online result did not evaluate every frozen five updates")
    required_numeric = (
        "distance_to_source",
        "distance_to_target",
        "failure_rate",
        "task_or_progress",
        "training_critic_score",
    )
    for point in evaluations:
        for key in required_numeric:
            _finite_float(point.get(key), f"evaluation.{key}")
        rewards = point.get("held_out_rewards")
        if not isinstance(rewards, Mapping) or tuple(
            sorted(int(seed) for seed in rewards)
        ) != tuple(sorted(protocol.held_out_critic_seeds)):
            raise ProtocolError("canonical evaluation lacks all five held-out rewards")
        for seed, value in rewards.items():
            _finite_float(value, f"held_out_rewards.{seed}")
    if result.get("actor_observation_contract") != request["actor_observation"]:
        raise ProtocolError("online actor observation differs from the requested contract")
    if result.get("canonical_snapshot_count") != protocol.evaluation_snapshots:
        raise ProtocolError("online canonical evaluation snapshot count changed")
    artifacts = result.get("physical_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ProtocolError("online result lacks persisted physical artifacts")
    for artifact in artifacts:
        path = Path(str(artifact.get("path", ""))).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != str(artifact.get("sha256", "")):
            raise ProtocolError(f"online physical artifact is missing or changed: {path}")
    return evaluate_single_seed(result, protocol=protocol)


def evaluate_single_seed(
    result: Mapping[str, Any], *, protocol: LocalEdgeProtocol
) -> dict[str, Any]:
    points = result.get("evaluations")
    if not isinstance(points, list) or len(points) < 2:
        raise ProtocolError("edge run has no initial/final evaluations")
    initial, final = points[0], points[-1]
    initial_target = _finite_float(initial["distance_to_target"], "initial target distance")
    final_target = _finite_float(final["distance_to_target"], "final target distance")
    if initial_target <= 0.0:
        reduction = 1.0 if final_target < initial_target else 0.0
    else:
        reduction = (initial_target - final_target) / initial_target
    held_out_delta = {
        str(seed): _finite_float(final["held_out_rewards"][str(seed)], "final reward")
        - _finite_float(initial["held_out_rewards"][str(seed)], "initial reward")
        for seed in protocol.held_out_critic_seeds
    }
    gates = {
        "target_distance_reduction": reduction
        >= protocol.target_distance_reduction_min,
        "final_closer_to_target_than_source": _finite_float(
            final["distance_to_target"], "final target distance"
        )
        < _finite_float(final["distance_to_source"], "final source distance"),
        "failure_rate": _finite_float(final["failure_rate"], "final failure")
        - _finite_float(initial["failure_rate"], "initial failure")
        <= protocol.failure_increase_max,
        "task_or_progress": _finite_float(
            initial["task_or_progress"], "initial progress"
        )
        - _finite_float(final["task_or_progress"], "final progress")
        <= protocol.progress_drop_max,
        "held_out_critic_direction_consistent": all(
            delta > 0.0 for delta in held_out_delta.values()
        ),
    }
    # Training score is reported, but deliberately absent from all gates.
    return {
        "seed": int(result["seed"]),
        "passed": bool(all(gates.values())),
        "gates": gates,
        "target_distance_reduction": float(reduction),
        "held_out_reward_delta": held_out_delta,
        "training_critic_score_delta": _finite_float(
            final["training_critic_score"], "final training critic score"
        )
        - _finite_float(
            initial["training_critic_score"], "initial training critic score"
        ),
        "training_critic_score_used_for_pass": False,
    }


def evaluate_edge_seed_set(
    seed_results: Sequence[Mapping[str, Any]],
    *,
    protocol: LocalEdgeProtocol,
) -> dict[str, Any]:
    if tuple(sorted(int(item["seed"]) for item in seed_results)) != tuple(
        sorted(protocol.ppo_seeds)
    ):
        raise ProtocolError("edge result does not contain exactly the three PPO seeds")
    passes = sum(bool(item.get("passed")) for item in seed_results)
    return {
        "passed_seed_count": int(passes),
        "total_seed_count": len(seed_results),
        "passed": passes >= protocol.minimum_passing_seeds,
        "by_seed": [dict(item) for item in seed_results],
    }


def invoke_backend(
    backend: ModuleType,
    request: Mapping[str, Any],
    *,
    protocol: LocalEdgeProtocol,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = backend.run_experiment(dict(request))
    if not isinstance(raw, Mapping):
        raise ProtocolError("real edge backend returned a non-mapping result")
    result = dict(raw)
    audit = validate_real_online_result(result, request=request, protocol=protocol)
    return result, audit


def baseline_outperformance(
    chain: Mapping[str, Any], controls: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Compare preregistered metrics componentwise; no weighted scalar is made."""

    required = ("direct_final_amp", "fixed_checkpoint_schedule", "bc_action_distillation")
    metrics = (
        ("target_distance_reduction", True),
        ("failure_rate_increase", False),
        ("task_or_progress_drop", False),
    )
    details: dict[str, Any] = {}
    for name in required:
        control = controls.get(name)
        if not isinstance(control, Mapping) or control.get("valid") is not True:
            details[name] = {"outperformed": False, "reason": "valid control unavailable"}
            continue
        component = {}
        for metric, higher in metrics:
            first = _finite_float(chain.get(metric), f"chain.{metric}")
            second = _finite_float(control.get(metric), f"control.{name}.{metric}")
            component[metric] = first > second if higher else first < second
        details[name] = {
            "outperformed": bool(all(component.values())),
            "componentwise": component,
        }
    return {
        "all_three_outperformed": all(
            details[name]["outperformed"] for name in required
        ),
        "by_control": details,
        "combination": "none; strict componentwise comparison",
    }


def artifact_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


__all__ = [
    "ALLOWED_PRIMARY_DOMAINS",
    "BACKEND_SCHEMA",
    "EDGE_MANIFEST_SCHEMA",
    "EVALUATION_CRITIC_ROLE",
    "LocalEdgeProtocol",
    "ONLINE_RESULT_SCHEMA",
    "POSITIVE_BUFFER_SCHEMA",
    "TRAINING_CRITIC_ROLE",
    "aggregate_teacher_outcomes",
    "artifact_sha256",
    "baseline_outperformance",
    "build_backend_request",
    "build_positive_buffer_manifest",
    "candidate_chains",
    "compute_edge_overlap",
    "diag35_evaluation_critic_manifest",
    "evaluate_edge_seed_set",
    "evaluate_single_seed",
    "held_out_reward_icc",
    "invoke_backend",
    "load_edge_manifest",
    "load_real_edge_backend",
    "outcome_edge_evidence",
    "read_csv_rows",
    "reject_deprecated_legacy",
    "select_preregistered_edges",
    "summarize_overlap_audits",
    "transition_crossings",
    "validate_evaluation_critics",
    "validate_positive_buffer_isolation",
    "validate_real_online_result",
    "write_yaml_exclusive",
]
