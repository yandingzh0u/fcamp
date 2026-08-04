from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .canonical_collection import load_rollout_index, load_rollout_trajectory
from .manifest import DependencyUnavailable, ProtocolError


@dataclass(frozen=True, slots=True)
class OutcomeMetric:
    name: str
    higher_is_better: bool
    tolerance: float


DEFAULT_PRIMARY_OUTCOMES: tuple[OutcomeMetric, ...] = (
    OutcomeMetric("motion_complete", True, 0.0),
    OutcomeMetric("failure", False, 0.0),
    OutcomeMetric("reference_progress", True, 0.01),
    OutcomeMetric("survival", True, 1.0),
    OutcomeMetric("joint_limit_incidence", False, 0.01),
    OutcomeMetric("undesired_contacts", False, 0.01),
)


def pareto_preference(
    candidate: Mapping[str, float],
    baseline: Mapping[str, float],
    *,
    metrics: Sequence[OutcomeMetric] = DEFAULT_PRIMARY_OUTCOMES,
) -> int:
    """Return +1/-1/0 for strict tolerant Pareto ordering.

    No scalar score is constructed or exposed.
    """

    candidate_better = False
    baseline_better = False
    for metric in metrics:
        if metric.name not in candidate or metric.name not in baseline:
            raise KeyError(f"quality panel is missing {metric.name!r}")
        first = float(candidate[metric.name])
        second = float(baseline[metric.name])
        if not np.isfinite(first) or not np.isfinite(second):
            raise FloatingPointError(f"non-finite outcome {metric.name!r}")
        signed = first - second if metric.higher_is_better else second - first
        if signed > metric.tolerance:
            candidate_better = True
        elif signed < -metric.tolerance:
            baseline_better = True
    if candidate_better and not baseline_better:
        return 1
    if baseline_better and not candidate_better:
        return -1
    return 0


def strict_pareto_pairs(
    rows: Sequence[Mapping[str, float]],
    *,
    metrics: Sequence[OutcomeMetric] = DEFAULT_PRIMARY_OUTCOMES,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            preference = pareto_preference(rows[left], rows[right], metrics=metrics)
            if preference > 0:
                pairs.append((left, right))
            elif preference < 0:
                pairs.append((right, left))
    return pairs


def outcome_metrics_from_spec(spec: Mapping[str, Any]) -> tuple[OutcomeMetric, ...]:
    raw = (
        spec.get("analysis_protocols", {})
        .get("reward_validity", {})
        .get("primary_pareto_outcomes")
    )
    if not isinstance(raw, list) or not raw:
        raise ProtocolError("reward_validity.primary_pareto_outcomes is not frozen")
    metrics: list[OutcomeMetric] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ProtocolError("each primary Pareto outcome must be a mapping")
        metric = OutcomeMetric(
            name=str(item.get("name", "")),
            higher_is_better=bool(item.get("higher_is_better")),
            tolerance=float(item.get("tolerance")),
        )
        if not metric.name or not np.isfinite(metric.tolerance) or metric.tolerance < 0.0:
            raise ProtocolError("primary Pareto outcome has an invalid name/tolerance")
        metrics.append(metric)
    if len({metric.name for metric in metrics}) != len(metrics):
        raise ProtocolError("primary Pareto outcome names are duplicated")
    expected = [(x.name, x.higher_is_better, x.tolerance) for x in DEFAULT_PRIMARY_OUTCOMES]
    actual = [(x.name, x.higher_is_better, x.tolerance) for x in metrics]
    if actual != expected:
        raise ProtocolError(
            f"reward-validity Pareto protocol differs from the frozen contract: {actual}"
        )
    return tuple(metrics)


def _tensor(tree: Mapping[str, Any], section: str, field: str) -> torch.Tensor:
    value = tree.get(section, {}).get(field)
    if not torch.is_tensor(value):
        raise ProtocolError(f"canonical trajectory lacks tensor {section}.{field}")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ProtocolError(f"canonical trajectory {section}.{field} contains NaN or Inf")
    return value


def _optional_tensor(
    tree: Mapping[str, Any], section: str, field: str
) -> torch.Tensor | None:
    value = tree.get(section, {}).get(field)
    if value is None:
        return None
    if not torch.is_tensor(value):
        raise ProtocolError(f"canonical optional field {section}.{field} is not a tensor")
    if value.is_floating_point() and not bool(torch.isfinite(value).all()):
        raise ProtocolError(f"canonical optional field {section}.{field} contains NaN or Inf")
    return value


def trajectory_quality_row(
    index_row: Mapping[str, Any],
    tree: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate actual recorded rollout outcomes without combining metrics.

    Every mandatory field is derived from the canonical PhysX shard.  Metrics
    which the collector cannot establish (reference contact labels and actuator
    effort limits) are explicitly unavailable rather than imputed.
    """

    done = _tensor(tree, "trajectory", "done").bool().reshape(-1)
    failure = _tensor(tree, "trajectory", "failure").bool().reshape(-1)
    complete = _tensor(tree, "trajectory", "motion_complete").bool().reshape(-1)
    progress = _tensor(tree, "outcome", "reference_progress").float().reshape(-1)
    anchor = _tensor(tree, "outcome", "anchor_error").float().reshape(-1)
    body = _tensor(tree, "outcome", "body_error").float().reshape(-1)
    joint = _tensor(tree, "outcome", "joint_error").float().reshape(-1)
    actions = _tensor(tree, "action", "applied").float()
    reward_terms = tree.get("outcome", {}).get("reward_terms")
    if not isinstance(reward_terms, Mapping):
        raise ProtocolError("canonical trajectory lacks outcome.reward_terms")
    joint_limit = reward_terms.get("joint_limit")
    undesired_count = reward_terms.get("undesired_contacts")
    if not torch.is_tensor(joint_limit) or not torch.is_tensor(undesired_count):
        raise ProtocolError(
            "canonical reward terms must expose raw joint_limit and undesired_contacts"
        )
    joint_limit = joint_limit.float().reshape(-1)
    undesired_count = undesired_count.float().reshape(-1)
    lengths = {
        int(value.shape[0])
        for value in (
            done,
            failure,
            complete,
            progress,
            anchor,
            body,
            joint,
            actions,
            joint_limit,
            undesired_count,
        )
    }
    if len(lengths) != 1 or next(iter(lengths), 0) < 1:
        raise ProtocolError("canonical quality fields are not aligned/nonempty")
    if not bool(
        torch.isfinite(joint_limit).all()
        and torch.isfinite(undesired_count).all()
        and torch.isfinite(actions).all()
    ):
        raise ProtocolError("canonical physical quality fields contain NaN or Inf")
    terminal = int(torch.where(done)[0][0].item()) if bool(done.any()) else len(done) - 1
    used = terminal + 1
    action_prefix = actions[:used]
    if used >= 3:
        jerk = action_prefix[2:] - 2.0 * action_prefix[1:-1] + action_prefix[:-2]
        action_jerk = float(torch.linalg.vector_norm(jerk, dim=-1).mean().item())
    else:
        action_jerk = 0.0
    torque = _optional_tensor(tree, "state", "applied_torque")
    velocity = _optional_tensor(tree, "state", "joint_vel")
    power_available = torque is not None and velocity is not None
    mechanical_power: float | None = None
    if power_available:
        if torque.shape != velocity.shape or torque.shape[0] < used:
            raise ProtocolError("applied torque and joint velocity are not aligned")
        mechanical_power = float(
            torch.abs(torque[:used].float() * velocity[:used].float()).mean().item()
        )
    row: dict[str, Any] = {
        "sample_id": str(index_row["sample_id"]),
        "trajectory_id": str(index_row["trajectory_id"]),
        "snapshot_id": str(index_row["snapshot_id"]),
        "checkpoint_id": str(index_row["checkpoint_id"]),
        "checkpoint_sha256": str(index_row["checkpoint_sha256"]),
        "checkpoint_update": int(index_row["checkpoint_update"]),
        "checkpoint_lineage_id": str(index_row["checkpoint_lineage_id"]),
        "policy_domain": str(index_row["policy_domain"]),
        "collector_mode": str(index_row["collector_mode"]),
        "common_sigma": float(index_row["common_sigma"]),
        "motion_complete": float(bool(complete[terminal].item())),
        "failure": float(bool(failure[:used].any().item())),
        "reference_progress": float(progress[terminal].item()),
        "survival": float(used),
        "joint_limit_incidence": float((joint_limit[:used] > 0.0).float().mean().item()),
        "undesired_contacts": float(undesired_count[:used].mean().item()),
        "anchor_error": float(anchor[:used].mean().item()),
        "body_error": float(body[:used].mean().item()),
        "joint_error": float(joint[:used].mean().item()),
        "action_jerk": action_jerk,
        "contact_mode_agreement": None,
        "contact_mode_agreement_available": False,
        "torque_saturation": None,
        "torque_saturation_available": False,
        "mechanical_power": mechanical_power,
        "mechanical_power_available": bool(power_available),
        "num_recorded_steps": int(len(done)),
        "terminal_step": int(terminal),
    }
    for metric in DEFAULT_PRIMARY_OUTCOMES:
        if not np.isfinite(float(row[metric.name])):
            raise ProtocolError(f"mandatory quality outcome {metric.name} is non-finite")
    return row


def build_quality_panel(
    index_path: str | Path,
    *,
    collector_mode: str = "controlled_environment",
    common_sigma: float = 0.0,
) -> list[dict[str, Any]]:
    """Read the unified collector once and return primary-condition rows."""

    source = Path(index_path).expanduser().resolve()
    rows = [
        row
        for row in load_rollout_index(source)
        if str(row["collector_mode"]) == collector_mode
        and float(row["common_sigma"]) == float(common_sigma)
        and bool(row["eligible_for_primary_overlap"])
    ]
    if not rows:
        raise DependencyUnavailable(
            f"no canonical trajectories match {collector_mode}/sigma={common_sigma}"
        )
    cache: dict[str, dict[str, Any]] = {}
    panel = [
        trajectory_quality_row(
            row,
            load_rollout_trajectory(source, row, cache=cache),
        )
        for row in rows
    ]
    identities = [str(row["sample_id"]) for row in panel]
    if len(set(identities)) != len(identities):
        raise ProtocolError("quality-panel sample identities are duplicated")
    return panel


def grouped_strict_pareto_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    group_field: str,
    metrics: Sequence[OutcomeMetric] = DEFAULT_PRIMARY_OUTCOMES,
) -> list[tuple[int, int]]:
    """Return comparable pairs only within a frozen condition group."""

    groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if group_field not in row:
            raise KeyError(group_field)
        groups.setdefault(str(row[group_field]), []).append(index)
    pairs: list[tuple[int, int]] = []
    for indices in groups.values():
        local = [rows[index] for index in indices]
        for winner, loser in strict_pareto_pairs(local, metrics=metrics):
            pairs.append((indices[winner], indices[loser]))
    return pairs


__all__ = [
    "DEFAULT_PRIMARY_OUTCOMES",
    "OutcomeMetric",
    "build_quality_panel",
    "grouped_strict_pareto_pairs",
    "outcome_metrics_from_spec",
    "pareto_preference",
    "strict_pareto_pairs",
    "trajectory_quality_row",
]
