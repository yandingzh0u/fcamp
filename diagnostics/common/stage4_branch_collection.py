"""Extraction helpers for real same-snapshot Stage-4 PhysX branches."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from components.imitation.motion_features import canonicalize_imitation_window

from .manifest import ProtocolError
from .quality_panel import trajectory_quality_row
from .reward_stage4 import AMP_WINDOW_DIM, AMP_WINDOW_STEPS


@dataclass(frozen=True, slots=True)
class BranchDescriptor:
    branch_id: str
    category: str
    checkpoint_id: str
    checkpoint_sha256: str
    checkpoint_update: int
    checkpoint_lineage_id: str
    policy_domain: str
    local_action_candidate: bool
    details: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.branch_id or not self.category:
            raise ProtocolError("branch descriptor identity/category is empty")
        if "A_mix" in {self.branch_id, self.category, self.policy_domain}:
            raise ProtocolError("A_mix/FCAMP is quarantined from Stage-4 branches")

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch_id": self.branch_id,
            "branch_category": self.category,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_update": self.checkpoint_update,
            "checkpoint_lineage_id": self.checkpoint_lineage_id,
            "policy_domain": self.policy_domain,
            "local_action_candidate": self.local_action_candidate,
            "details": dict(self.details),
        }


def _slice_env(value: Any, *, env_index: int, steps: int) -> Any:
    if torch.is_tensor(value):
        if value.ndim < 2:
            return value
        return value[:steps, env_index].detach().cpu().clone()
    if isinstance(value, Mapping):
        return {
            str(key): _slice_env(nested, env_index=env_index, steps=steps)
            for key, nested in value.items()
        }
    return value


def _first_done(done: torch.Tensor, env_index: int, horizon: int) -> int:
    indices = torch.where(done[:horizon, env_index].bool())[0]
    return int(indices[0].item() + 1) if indices.numel() else int(horizon)


def extract_real_branch(
    tree: Mapping[str, Any],
    descriptor: BranchDescriptor,
    *,
    snapshot_ids: Sequence[str],
    demo_history: torch.Tensor,
    horizons: Sequence[int],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[dict[str, Any]],
]:
    """Extract endpoint AMP windows and measured outcomes from one real rollout.

    ``tree`` must record at least ``max(horizons)+1`` pre-step states.  The
    extra state is the physical endpoint after the final evaluated action.
    Initial AMP history is the exact expert history supplied by the simulator;
    every subsequent frame is PhysX readback.
    """

    done = tree.get("trajectory", {}).get("done")
    raw = tree.get("imitation", {}).get("agent_physx_raw_frame")
    body = tree.get("state", {}).get("body_pos")
    root = tree.get("state", {}).get("root_pos")
    if not all(torch.is_tensor(value) for value in (done, raw, body, root)):
        raise ProtocolError("real branch rollout lacks done/imitation/body/root tensors")
    horizons = tuple(int(value) for value in horizons)
    maximum = max(horizons)
    env_count = len(snapshot_ids)
    if (
        done.ndim != 2
        or raw.ndim != 3
        or int(done.shape[0]) < maximum + 1
        or int(raw.shape[0]) < maximum + 1
        or int(done.shape[1]) != env_count
        or int(raw.shape[1]) != env_count
    ):
        raise ProtocolError("branch rollout lacks the physical endpoint after max horizon")
    if raw.shape[-1] * AMP_WINDOW_STEPS != AMP_WINDOW_DIM:
        raise ProtocolError("branch rollout AMP frame dimension changed")
    if tuple(demo_history.shape) != (env_count, AMP_WINDOW_STEPS, raw.shape[-1]):
        raise ProtocolError("demo-seeded branch history has an invalid shape")
    if not bool(
        torch.isfinite(raw).all()
        and torch.isfinite(body).all()
        and torch.isfinite(root).all()
        and torch.isfinite(demo_history).all()
    ):
        raise ProtocolError("real branch tensors contain NaN or Inf")
    track_count = int(body.shape[2])
    endpoints = torch.empty(len(horizons), env_count, AMP_WINDOW_DIM, dtype=torch.float32)
    body_local = torch.empty(maximum, env_count, track_count, 3, dtype=torch.float32)
    root_local = torch.empty(maximum, env_count, 3, dtype=torch.float32)
    active = torch.zeros(maximum, env_count, dtype=torch.bool)
    endpoint_phase = torch.empty(len(horizons), env_count, dtype=torch.float32)
    endpoint_contact = torch.empty(len(horizons), env_count, dtype=torch.long)
    rows: list[dict[str, Any]] = []
    raw_cpu = raw.detach().cpu().float()
    body_cpu = body.detach().cpu().float()
    root_cpu = root.detach().cpu().float()
    done_cpu = done.detach().cpu().bool()
    history_cpu = demo_history.detach().cpu().float()
    for env_index, snapshot_id in enumerate(snapshot_ids):
        terminal_max = _first_done(done_cpu, env_index, maximum)
        initial_root = root_cpu[0, env_index]
        for step in range(maximum):
            endpoint_step = min(step + 1, terminal_max)
            endpoint_root = root_cpu[endpoint_step, env_index]
            body_local[step, env_index] = body_cpu[endpoint_step, env_index] - endpoint_root
            root_local[step, env_index] = endpoint_root - initial_root
            active[step, env_index] = step < terminal_max
        for horizon_index, horizon in enumerate(horizons):
            used = _first_done(done_cpu, env_index, horizon)
            future = raw_cpu[1 : used + 1, env_index]
            sequence = torch.cat((history_cpu[env_index], future), dim=0)
            window = sequence[-AMP_WINDOW_STEPS:]
            if tuple(window.shape) != (AMP_WINDOW_STEPS, raw.shape[-1]):
                raise ProtocolError("demo/PhysX endpoint history is incomplete")
            endpoints[horizon_index, env_index] = canonicalize_imitation_window(
                window.unsqueeze(0)
            ).reshape(-1)
            phase_values = tree["trajectory"]["phase_continuous"]
            contact_values = tree["trajectory"]["contact_mode"]
            endpoint_phase[horizon_index, env_index] = float(
                phase_values[used, env_index].detach().cpu().item()
            )
            endpoint_contact[horizon_index, env_index] = int(
                contact_values[used, env_index].detach().cpu().item()
            )
            trajectory = {"metadata": dict(tree.get("metadata", {}))}
            for section in ("trajectory", "observation", "state", "action", "reference", "outcome", "imitation"):
                trajectory[section] = _slice_env(
                    tree[section], env_index=env_index, steps=used
                )
            index_row = {
                "sample_id": f"{descriptor.branch_id}:{snapshot_id}:h{horizon}",
                "trajectory_id": f"{descriptor.branch_id}:{snapshot_id}",
                "snapshot_id": str(snapshot_id),
                "checkpoint_id": descriptor.checkpoint_id,
                "checkpoint_sha256": descriptor.checkpoint_sha256,
                "checkpoint_update": descriptor.checkpoint_update,
                "checkpoint_lineage_id": descriptor.checkpoint_lineage_id,
                "policy_domain": descriptor.policy_domain,
                "collector_mode": "controlled_environment",
                "common_sigma": 0.0,
            }
            row = trajectory_quality_row(index_row, trajectory)
            row.update(
                {
                    "branch_id": descriptor.branch_id,
                    "branch_category": descriptor.category,
                    "local_action_candidate": descriptor.local_action_candidate,
                    "branch_details_json": json.dumps(
                        dict(descriptor.details), sort_keys=True, separators=(",", ":")
                    ),
                    "horizon": int(horizon),
                    "endpoint_horizon_index": int(horizon_index),
                    "endpoint_env_index": int(env_index),
                    "same_snapshot": True,
                    "real_physx_rollout": True,
                }
            )
            rows.append(row)
    return (
        endpoints,
        body_local,
        root_local,
        active,
        endpoint_phase,
        endpoint_contact,
        rows,
    )


def replay_audit(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    atol: float = 1.0e-6,
) -> dict[str, Any]:
    """Audit the active prefix that determines actions, outcomes and reward.

    Isaac environments automatically reset an environment immediately after a
    terminal transition.  Samples after that transition belong to a new
    episode and can legitimately use a different reset draw; treating them as
    part of the original same-snapshot branch makes an otherwise exact replay
    fail spuriously.  The terminal transition itself remains in scope, so a
    different first termination or any pre-terminal numerical drift still
    fails closed.
    """

    paths = (
        ("trajectory", "done"),
        ("state", "root_pos"),
        ("state", "body_pos"),
        ("action", "applied"),
        ("imitation", "agent_physx_raw_frame"),
    )
    left_done = first.get("trajectory", {}).get("done")
    right_done = second.get("trajectory", {}).get("done")
    if (
        not torch.is_tensor(left_done)
        or not torch.is_tensor(right_done)
        or left_done.shape != right_done.shape
        or left_done.ndim != 2
    ):
        raise ProtocolError("replay field trajectory.done is missing/misaligned")
    left_done = left_done.bool()
    right_done = right_done.bool()
    # Compare through the first terminal transition of either replay.  A
    # different terminal decision at that transition is always a hard error.
    left_count = torch.cumsum(left_done.to(torch.int64), dim=0)
    right_count = torch.cumsum(right_done.to(torch.int64), dim=0)
    active_before = (
        (left_count - left_done.to(torch.int64) == 0)
        & (right_count - right_done.to(torch.int64) == 0)
    )
    done_difference = left_done != right_done
    done_mismatch = done_difference & active_before
    mismatch_locations = torch.nonzero(done_mismatch, as_tuple=False)
    first_done_mismatch = (
        {
            "step": int(mismatch_locations[0, 0].item()),
            "env_index": int(mismatch_locations[0, 1].item()),
            "first_done": bool(left_done[tuple(mismatch_locations[0])].item()),
            "second_done": bool(right_done[tuple(mismatch_locations[0])].item()),
        }
        if mismatch_locations.numel()
        else None
    )

    maximum = 0.0
    fields: dict[str, Any] = {}
    for section, field in paths:
        left = first.get(section, {}).get(field)
        right = second.get(section, {}).get(field)
        if not torch.is_tensor(left) or not torch.is_tensor(right) or left.shape != right.shape:
            raise ProtocolError(f"replay field {section}.{field} is missing/misaligned")
        if left.shape[:2] != active_before.shape:
            raise ProtocolError(f"replay field {section}.{field} lacks [time,env] axes")
        if left.dtype is torch.bool:
            fields[f"{section}.{field}"] = {
                "exact": bool(torch.equal(left[active_before], right[active_before])),
                "mismatch_count": int(torch.sum(left[active_before] != right[active_before]).item()),
            }
            continue
        difference = torch.abs(left[active_before] - right[active_before])
        if not bool(torch.isfinite(difference).all()):
            field_maximum = float("inf")
        else:
            field_maximum = float(torch.max(difference).item()) if difference.numel() else 0.0
        maximum = max(maximum, field_maximum)
        per_step: list[float] = []
        for step in range(int(left.shape[0])):
            selected = active_before[step]
            step_difference = torch.abs(left[step, selected] - right[step, selected])
            per_step.append(
                float(torch.max(step_difference).item()) if step_difference.numel() else 0.0
            )
        first_exceed = next(
            (index for index, value in enumerate(per_step) if not np.isfinite(value) or value > float(atol)),
            None,
        )
        fields[f"{section}.{field}"] = {
            "maximum_abs_error": field_maximum,
            "step_0_max_abs_error": per_step[0] if per_step else 0.0,
            "first_step_exceeding_atol": first_exceed,
            "maximum_error_step": int(np.argmax(per_step)) if per_step else None,
            "per_step_max_abs_error": per_step,
        }
    return {
        "atol": float(atol),
        "active_sample_count": int(active_before.sum().item()),
        "first_done_mismatch": first_done_mismatch,
        "maximum_abs_error": float("inf") if first_done_mismatch is not None else maximum,
        "fields": fields,
        "pass": first_done_mismatch is None and bool(maximum <= float(atol)),
    }


def replay_max_abs_error(first: Mapping[str, Any], second: Mapping[str, Any]) -> float:
    """Backward-compatible scalar view of :func:`replay_audit`."""

    return float(replay_audit(first, second)["maximum_abs_error"])


__all__ = [
    "BranchDescriptor",
    "extract_real_branch",
    "replay_audit",
    "replay_max_abs_error",
]
