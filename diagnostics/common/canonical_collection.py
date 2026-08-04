"""Canonical, condition-matched rollout collection for discovery stage 1.

The pure helpers in this module are importable without Isaac Lab.  The small
set of functions which touch the simulator import repository/Isaac modules
lazily, after an entrypoint has launched :class:`isaaclab.app.AppLauncher`.

The central protocol invariant is that all checkpoints are evaluated in one
``CoreTrainer``/PhysX instance.  Checkpoint changes load *only* the policy
state dict; they never recreate the environment, restore checkpoint RNG, or
run the checkpointer's resume hook.  Consequently startup randomization is a
controlled property of the shared environment rather than a checkpoint
fingerprint.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import csv
import io
import math
import os
from pathlib import Path
from types import MethodType
from typing import Any, Iterator, Mapping, MutableMapping, Protocol, Sequence

import torch

from .checkpoint_io import state_digest
from .feature_views import build_actor_feature_views
from .manifest import (
    DependencyUnavailable,
    ProtocolError,
    sha256_file,
)
from .noise_bank import (
    CollectorMode,
    NoiseBank,
    is_overlap_eligible,
    parse_collector_mode,
)
from .observation_spec import ObservationCardinality, derive_observation_spec
from .rollout_collector import (
    TensorTreeAccumulator,
    construct_action_record,
    policy_mean_and_std,
)
from .snapshot_bank import SnapshotBank
from .imitation_6901 import (
    Commit6901ImitationAdapter,
    IMITATION_FRAME_DIM,
    imitation_contract_metadata,
)


CANONICAL_INDEX_FIELDS = (
    "sample_id",
    "trajectory_id",
    "snapshot_id",
    "env_id",
    "episode_id",
    "checkpoint_id",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_update",
    "checkpoint_lineage_id",
    "policy_domain",
    "collector_mode",
    "common_sigma",
    "eligible_for_primary_overlap",
    "shard_path",
    "shard_env_index",
    "num_steps",
)

CANONICAL_REQUIRED_ROLLOUT_SECTIONS = frozenset(
    {
        "metadata",
        "trajectory",
        "observation",
        "state",
        "action",
        "reference",
        "outcome",
        "imitation",
    }
)

_SNAPSHOT_STATE_PREFIX = "engine"
_CLEAN_STATE = "clean"
_CONTROLLED_STATE = "controlled"


@dataclass(frozen=True, slots=True)
class CanonicalCollectionProtocol:
    num_envs: int
    num_snapshots: int
    horizon: int
    phase_strategy: str
    snapshot_seed: int
    collector_seed: int
    common_sigmas: tuple[float, ...]
    canonical_checkpoint_updates: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.num_envs < 2:
            raise ProtocolError("collection.num_envs must be at least two")
        if self.num_snapshots != self.num_envs:
            raise ProtocolError(
                "stage-1 canonical collection requires one snapshot per persistent "
                "environment; num_snapshots must equal num_envs"
            )
        if self.horizon < 1:
            raise ProtocolError("collection.horizon_control_steps must be positive")
        if self.phase_strategy != "evenly_spaced":
            raise ProtocolError(
                "collection.phase_strategy must be the frozen value 'evenly_spaced'"
            )
        if len(self.common_sigmas) < 3:
            raise ProtocolError(
                "common_action_noise requires at least three frozen scales"
            )
        if any(not math.isfinite(value) or value <= 0.0 for value in self.common_sigmas):
            raise ProtocolError("common action-noise scales must be finite and positive")
        if len(set(self.common_sigmas)) != len(self.common_sigmas):
            raise ProtocolError("common action-noise scales must be unique")
        if not self.canonical_checkpoint_updates:
            raise ProtocolError("canonical_checkpoint_updates must not be empty")
        if any(value < 0 for value in self.canonical_checkpoint_updates):
            raise ProtocolError("canonical checkpoint updates must be non-negative")
        if tuple(sorted(set(self.canonical_checkpoint_updates))) != self.canonical_checkpoint_updates:
            raise ProtocolError(
                "canonical_checkpoint_updates must be sorted and duplicate-free"
            )

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "CanonicalCollectionProtocol":
        raw = spec.get("collection")
        if not isinstance(raw, Mapping):
            raise ProtocolError("suite spec has no frozen collection mapping")
        required = {
            "num_envs",
            "num_snapshots",
            "horizon_control_steps",
            "phase_strategy",
            "snapshot_seed",
            "collector_seed",
            "common_action_noise_scales",
            "canonical_checkpoint_updates",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ProtocolError(f"collection spec is missing fields: {missing}")
        scales = raw["common_action_noise_scales"]
        updates = raw["canonical_checkpoint_updates"]
        if not isinstance(scales, Sequence) or isinstance(scales, (str, bytes)):
            raise ProtocolError("common_action_noise_scales must be a sequence")
        if not isinstance(updates, Sequence) or isinstance(updates, (str, bytes)):
            raise ProtocolError("canonical_checkpoint_updates must be a sequence")
        return cls(
            num_envs=int(raw["num_envs"]),
            num_snapshots=int(raw["num_snapshots"]),
            horizon=int(raw["horizon_control_steps"]),
            phase_strategy=str(raw["phase_strategy"]),
            snapshot_seed=int(raw["snapshot_seed"]),
            collector_seed=int(raw["collector_seed"]),
            common_sigmas=tuple(float(value) for value in scales),
            canonical_checkpoint_updates=tuple(int(value) for value in updates),
        )

    @property
    def branch_variants(self) -> tuple[tuple[CollectorMode, float], ...]:
        return (
            (CollectorMode.CLEAN_MEAN, 0.0),
            (CollectorMode.CONTROLLED_ENVIRONMENT, 0.0),
            *tuple(
                (CollectorMode.COMMON_ACTION_NOISE, sigma)
                for sigma in self.common_sigmas
            ),
            (CollectorMode.NATIVE_STOCHASTIC, 0.0),
        )


def validate_frozen_collection_semantics(
    spec: Mapping[str, Any],
    protocol: CanonicalCollectionProtocol,
) -> dict[str, Any]:
    """Reject an incomplete or silently reinterpreted collection protocol."""

    collection = spec.get("collection")
    if not isinstance(collection, Mapping):
        raise ProtocolError("suite spec has no collection mapping")
    details = collection.get("protocol_details")
    validation = collection.get("validation_protocols")
    if not isinstance(details, Mapping) or not isinstance(validation, Mapping):
        raise ProtocolError(
            "collection.protocol_details and collection.validation_protocols must be frozen"
        )
    expected_details = {
        "phase_grid": (
            "round(linspace(configured_motion_start_phase, "
            "motion_num_frames_minus_2)) independent of rollout horizon"
        ),
        "phase_grid_reason": (
            "horizon is a maximum rollout length and must not collapse a 325-frame "
            "motion to phase zero; trajectories crop at first done"
        ),
        "duplicate_phases_allowed_with_unique_snapshot_ids": True,
        "after_done": "do_not_reset_and_crop_each_trajectory_at_first_done",
        "clean_initial_state": "noise_free_reference_reset_snapshot",
        "nonclean_initial_state": "shared_controlled_noisy_snapshot_from_NoiseBank",
        "controlled_environment_randomness": "shared_observation_and_push_streams_from_NoiseBank",
        "native_stochastic_coupling": "shared_standard_epsilon_per_snapshot_with_checkpoint_learned_std",
    }
    mismatches = {
        key: {"expected": value, "actual": details.get(key)}
        for key, value in expected_details.items()
        if details.get(key) != value
    }
    timing = details.get("tensor_timing")
    if not isinstance(timing, Mapping) or dict(timing) != {
        "observation_state_reference_action": "pre_step",
        "reward_done_outcome": "post_step",
    }:
        mismatches["tensor_timing"] = {
            "expected": {
                "observation_state_reference_action": "pre_step",
                "reward_done_outcome": "post_step",
            },
            "actual": timing,
        }
    controlled_sigma = validation.get("controlled_noise_sigma")
    if controlled_sigma is None or float(controlled_sigma) not in protocol.common_sigmas:
        mismatches["validation_protocols.controlled_noise_sigma"] = {
            "expected": f"one of {protocol.common_sigmas}",
            "actual": controlled_sigma,
        }
    summary_update = validation.get("summary_checkpoint_update")
    if summary_update is None or int(summary_update) not in protocol.canonical_checkpoint_updates:
        mismatches["validation_protocols.summary_checkpoint_update"] = {
            "expected": f"one of {protocol.canonical_checkpoint_updates}",
            "actual": summary_update,
        }
    if mismatches:
        raise ProtocolError(f"frozen collection semantics mismatch: {mismatches}")
    return {
        "protocol_details": dict(details),
        "validation_protocols": dict(validation),
    }


@dataclass(frozen=True, slots=True)
class DenseCheckpointRecord:
    checkpoint_id: str
    path: Path
    sha256: str
    update: int
    lineage_id: str
    policy_domain: str = "teacher_fixed_reward"


class CanonicalPolicyAdapter(Protocol):
    """Strict external-policy seam for one shared canonical environment.

    Stateful/chunk policies must preserve their scheduler state inside this
    adapter.  The collector calls ``reset_branch`` exactly once after restoring
    the snapshot bank and then ``action_record`` once per control step.
    """

    policy_domain: str

    def reset_branch(
        self,
        *,
        env: Any,
        mode: CollectorMode,
        snapshot_ids: Sequence[str],
    ) -> None: ...

    def action_record(
        self,
        *,
        env: Any,
        observation: torch.Tensor,
        imitation: Mapping[str, torch.Tensor],
        common_epsilon: torch.Tensor,
        mode: CollectorMode,
        common_sigma: float,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        step: int,
    ) -> Any: ...

    def provenance(self) -> Mapping[str, Any]: ...


class _TrainerPolicyAdapter:
    policy_domain = "teacher_fixed_reward"

    def __init__(self, trainer: Any, manifest: Mapping[str, Any]) -> None:
        self.trainer = trainer
        self.manifest = manifest

    def reset_branch(
        self,
        *,
        env: Any,
        mode: CollectorMode,
        snapshot_ids: Sequence[str],
    ) -> None:
        del env, mode, snapshot_ids

    def action_record(
        self,
        *,
        env: Any,
        observation: torch.Tensor,
        imitation: Mapping[str, torch.Tensor],
        common_epsilon: torch.Tensor,
        mode: CollectorMode,
        common_sigma: float,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        step: int,
    ) -> Any:
        del env, imitation, step
        mean, native_std = policy_mean_and_std(self.trainer.algo, observation)
        return construct_action_record(
            mean,
            native_std,
            common_epsilon,
            mode=mode,
            common_sigma=common_sigma,
            action_low=action_low,
            action_high=action_high,
        )

    def provenance(self) -> Mapping[str, Any]:
        return {
            "policy_method": "fixed_reward_ppo",
            "policy_source_commit": str(self.manifest["git_commit"]),
            "policy_source_snapshot_sha256": str(
                self.manifest["source_snapshot_sha256"]
            ),
            "policy_resolved_config_sha256": str(
                self.manifest["resolved_config_sha256"]
            ),
        }


def read_dense_checkpoint_records(path: str | Path) -> list[DenseCheckpointRecord]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DependencyUnavailable(f"dense checkpoint inventory is missing: {source}")
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    records: list[DenseCheckpointRecord] = []
    for row in rows:
        if str(row.get("present", "")).lower() not in {"1", "true", "yes"}:
            continue
        raw_path = str(row.get("checkpoint_path", ""))
        raw_hash = str(row.get("checkpoint_sha256", ""))
        raw_lineage = str(row.get("checkpoint_lineage_id", ""))
        if not raw_path or len(raw_hash) != 64 or not raw_lineage:
            raise ProtocolError("dense checkpoint inventory contains an incomplete selected row")
        checkpoint = Path(raw_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise DependencyUnavailable(f"checkpoint disappeared after inventory: {checkpoint}")
        if sha256_file(checkpoint) != raw_hash:
            raise ProtocolError(f"checkpoint hash changed after inventory: {checkpoint}")
        update = int(row["required_update"])
        records.append(
            DenseCheckpointRecord(
                checkpoint_id=f"u{update:04d}-{raw_hash[:12]}",
                path=checkpoint,
                sha256=raw_hash,
                update=update,
                lineage_id=raw_lineage,
            )
        )
    if not records:
        raise DependencyUnavailable("dense checkpoint inventory has no present checkpoints")
    records.sort(key=lambda item: item.update)
    if len({record.update for record in records}) != len(records):
        raise ProtocolError("dense inventory selects multiple checkpoints for one update")
    return records


def _prefix_engine_state(
    snapshot: Mapping[str, torch.Tensor],
    *,
    state_name: str,
    num_envs: int,
) -> dict[str, torch.Tensor]:
    """Encode per-env and global engine tensors into SnapshotBank rows."""

    encoded: dict[str, torch.Tensor] = {}
    for name, value in snapshot.items():
        if not torch.is_tensor(value):
            raise ProtocolError(f"engine snapshot {name!r} is not a tensor")
        value = value.detach().cpu()
        if value.ndim >= 1 and value.shape[0] == num_envs:
            encoded[f"{_SNAPSHOT_STATE_PREFIX}/{state_name}/env/{name}"] = value
        else:
            repeated = value.unsqueeze(0).expand(num_envs, *value.shape).clone()
            encoded[f"{_SNAPSHOT_STATE_PREFIX}/{state_name}/global/{name}"] = repeated
    return encoded


def _engine_state_from_bank(
    bank: SnapshotBank,
    *,
    state_name: str,
) -> dict[str, torch.Tensor]:
    snapshots = [bank.get(snapshot_id) for snapshot_id in bank.snapshot_ids]
    prefix = f"{_SNAPSHOT_STATE_PREFIX}/{state_name}/"
    keys = {key for key in snapshots[0].state if key.startswith(prefix)}
    if not keys:
        raise ProtocolError(f"snapshot bank has no {state_name!r} engine state")
    for snapshot in snapshots[1:]:
        current = {key for key in snapshot.state if key.startswith(prefix)}
        if current != keys:
            raise ProtocolError("snapshot engine fields differ across snapshot rows")
    restored: dict[str, torch.Tensor] = {}
    for key in sorted(keys):
        suffix = key[len(prefix) :]
        if suffix.startswith("env/"):
            name = suffix[len("env/") :]
            restored[name] = torch.stack([snapshot.state[key] for snapshot in snapshots])
        elif suffix.startswith("global/"):
            name = suffix[len("global/") :]
            values = [snapshot.state[key] for snapshot in snapshots]
            if any(not torch.equal(values[0], value) for value in values[1:]):
                raise ProtocolError(f"global engine snapshot {name!r} differs across rows")
            restored[name] = values[0].clone()
        else:
            raise ProtocolError(f"malformed snapshot-bank engine key: {key}")
    return restored


def _stack_snapshot_mapping(
    bank: SnapshotBank,
    attribute: str,
) -> dict[str, torch.Tensor]:
    snapshots = [bank.get(snapshot_id) for snapshot_id in bank.snapshot_ids]
    first = getattr(snapshots[0], attribute)
    keys = set(first)
    if any(set(getattr(snapshot, attribute)) != keys for snapshot in snapshots[1:]):
        raise ProtocolError(f"snapshot {attribute} fields differ across rows")
    return {
        key: torch.stack([getattr(snapshot, attribute)[key] for snapshot in snapshots])
        for key in sorted(keys)
    }


def snapshot_bank_startup_fingerprint(bank: SnapshotBank) -> str:
    return state_digest(_stack_snapshot_mapping(bank, "physics_randomization"))


def current_startup_randomization(env: Any) -> dict[str, torch.Tensor]:
    """Read the startup-randomized values which must survive policy switches."""

    values: dict[str, torch.Tensor] = {
        "default_joint_pos": env.robot.data.default_joint_pos.detach().cpu().clone(),
    }
    view = getattr(env.robot, "root_physx_view", None)
    if view is not None:
        for name, getter_name in (
            ("body_coms", "get_coms"),
            ("material_properties", "get_material_properties"),
        ):
            getter = getattr(view, getter_name, None)
            if callable(getter):
                value = getter()
                if torch.is_tensor(value):
                    values[name] = value.detach().cpu().clone()
    count = int(env.num_envs)
    for name, value in values.items():
        if value.ndim < 1 or value.shape[0] != count:
            raise ProtocolError(
                f"startup randomization tensor {name!r} lacks environment axis {count}"
            )
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ProtocolError(f"startup randomization tensor {name!r} is non-finite")
    return values


def assert_startup_randomization_matches(
    env: Any,
    bank: SnapshotBank,
    *,
    atol: float = 0.0,
) -> str:
    expected = _stack_snapshot_mapping(bank, "physics_randomization")
    current = current_startup_randomization(env)
    if set(expected) != set(current):
        raise ProtocolError(
            "startup randomization fields differ between snapshot and collection environments"
        )
    for name in expected:
        left = expected[name].to(dtype=current[name].dtype)
        matches = left.shape == current[name].shape and bool(
            torch.allclose(left, current[name], rtol=0.0, atol=float(atol))
        )
        if not matches:
            maximum = (
                float(torch.max(torch.abs(left - current[name])).item())
                if left.shape == current[name].shape and left.numel()
                else float("inf")
            )
            raise ProtocolError(
                f"startup randomization changed for {name}; max_abs_error={maximum}"
            )
    return state_digest(current)


def restore_startup_randomization(env: Any, bank: SnapshotBank) -> None:
    """Restore the frozen per-environment physics properties after ``sim.reset``."""

    expected = _stack_snapshot_mapping(bank, "physics_randomization")
    supported = {"default_joint_pos", "body_coms", "material_properties"}
    unknown = set(expected) - supported
    if unknown:
        raise ProtocolError(
            f"snapshot contains unsupported startup-randomization fields: {sorted(unknown)}"
        )
    if "default_joint_pos" not in expected:
        raise ProtocolError("snapshot lacks frozen default_joint_pos")
    default_joint_pos = expected["default_joint_pos"].to(
        device=env.robot.data.default_joint_pos.device,
        dtype=env.robot.data.default_joint_pos.dtype,
    )
    if default_joint_pos.shape != env.robot.data.default_joint_pos.shape:
        raise ProtocolError("snapshot default_joint_pos shape changed")
    env.robot.data.default_joint_pos.copy_(default_joint_pos)

    view = getattr(env.robot, "root_physx_view", None)
    env_ids_cpu = torch.arange(int(env.num_envs), dtype=torch.long, device="cpu")
    for name, setter_name in (
        ("body_coms", "set_coms"),
        ("material_properties", "set_material_properties"),
    ):
        if name not in expected:
            continue
        if view is None or not callable(getattr(view, setter_name, None)):
            raise ProtocolError(f"environment cannot restore startup field {name}")
        getattr(view, setter_name)(expected[name].detach().cpu().clone(), env_ids_cpu)


def evenly_spaced_phases(env: Any, protocol: CanonicalCollectionProtocol) -> torch.Tensor:
    """Return the frozen full-motion phase grid.

    The rollout horizon is deliberately absent from this calculation.  It is
    merely the maximum number of post-snapshot control steps.  Passing it to
    ``_adaptive_phase_range`` collapses a 325-frame motion and a 325-step
    horizon to phase zero, silently destroying the condition-matched audit.
    The final motion frame is excluded because interpolation and the first
    transition need a valid successor frame.
    """

    motion = getattr(env, "motion", None)
    num_frames = int(getattr(motion, "num_frames", 0))
    minimum = int(getattr(env, "motion_start_phase", 0))
    maximum = num_frames - 2
    if maximum < minimum:
        raise ProtocolError(
            "motion interval has no canonical phase with a valid successor: "
            f"start={minimum}, num_frames={num_frames}"
        )
    phases = torch.linspace(
        float(minimum),
        float(maximum),
        steps=protocol.num_snapshots,
        device=env.device,
    ).round().to(torch.long)
    # Duplicate phases are acceptable when the motion has fewer phases than
    # snapshots; snapshot identity also contains controlled reset noise.
    return phases


def canonical_snapshot_ids(phases: torch.Tensor) -> tuple[str, ...]:
    return tuple(
        f"snapshot-{index:04d}-phase-{int(phase):05d}"
        for index, phase in enumerate(phases.detach().cpu().tolist())
    )


def _noise_rows(
    bank: NoiseBank,
    snapshot_ids: Sequence[str],
    *,
    stream: str,
    width: int,
    device: torch.device | str,
) -> torch.Tensor:
    return bank.controlled_uniform(
        snapshot_ids,
        horizon=1,
        width=width,
        stream=stream,
        low=0.0,
        high=1.0,
        device=device,
    )[:, 0]


def _controlled_noisy_reset_from_clean(
    env: Any,
    *,
    phases: torch.Tensor,
    snapshot_ids: Sequence[str],
    noise_bank: NoiseBank,
) -> None:
    """Write a reset-noised state using only the shared :class:`NoiseBank`."""

    from isaaclab.utils.math import quat_from_euler_xyz, quat_mul
    from envs.spec import RESET_JOINT_POSITION_RANGE, RESET_ROOT_POSE_RANGE, VELOCITY_RANGE

    count = int(env.num_envs)
    env_ids = torch.arange(count, device=env.device, dtype=torch.long)
    reference = env.motion.get_frame(phases)
    root_pos = reference["root_pos_w"].clone()
    root_quat = reference["root_quat_w"].clone()
    root_lin_vel = reference["root_lin_vel_w"].clone()
    root_ang_vel = reference["root_ang_vel_w"].clone()
    joint_pos = reference["joint_pos"].clone()
    joint_vel = reference["joint_vel"].clone()

    pose_unit = _noise_rows(
        noise_bank,
        snapshot_ids,
        stream="canonical_reset_pose",
        width=6,
        device=env.device,
    )
    pose_ranges = torch.tensor(RESET_ROOT_POSE_RANGE, device=env.device)
    pose_noise = pose_ranges[:, 0] + (pose_ranges[:, 1] - pose_ranges[:, 0]) * pose_unit
    root_pos += pose_noise[:, :3]
    root_quat[:] = quat_mul(
        quat_from_euler_xyz(pose_noise[:, 3], pose_noise[:, 4], pose_noise[:, 5]),
        root_quat,
    )

    velocity_unit = _noise_rows(
        noise_bank,
        snapshot_ids,
        stream="canonical_reset_velocity",
        width=6,
        device=env.device,
    )
    velocity_ranges = torch.tensor(VELOCITY_RANGE, device=env.device)
    velocity_noise = velocity_ranges[:, 0] + (
        velocity_ranges[:, 1] - velocity_ranges[:, 0]
    ) * velocity_unit
    root_lin_vel += velocity_noise[:, :3]
    root_ang_vel += velocity_noise[:, 3:]

    joint_unit = _noise_rows(
        noise_bank,
        snapshot_ids,
        stream="canonical_reset_joint_position",
        width=int(env.action_dim),
        device=env.device,
    )
    joint_low, joint_high = RESET_JOINT_POSITION_RANGE
    joint_pos += float(joint_low) + (float(joint_high) - float(joint_low)) * joint_unit
    soft_limits = env.robot.data.soft_joint_pos_limits.index_select(0, env_ids)
    joint_pos[:] = torch.clamp(
        joint_pos,
        soft_limits[:, env.action_joint_ids, 0],
        soft_limits[:, env.action_joint_ids, 1],
    )
    env._write_robot_state(
        root_pos=root_pos,
        root_quat=root_quat,
        root_lin_vel=root_lin_vel,
        root_ang_vel=root_ang_vel,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        env_ids=env_ids,
    )

    min_push, max_push = env.push_interval_step_range
    schedule_unit = _noise_rows(
        noise_bank,
        snapshot_ids,
        stream="canonical_reset_push_schedule",
        width=1,
        device=env.device,
    )[:, 0]
    interval = min_push + torch.floor(
        schedule_unit * float(max_push - min_push + 1)
    ).to(torch.long).clamp(max=max_push - min_push)
    env.next_push_step.copy_(env.episode_steps + interval)
    env.first_push_step.fill_(-1)
    env.scene.update(env.physics_dt)


def build_snapshot_bank_from_environment(
    env: Any,
    protocol: CanonicalCollectionProtocol,
) -> SnapshotBank:
    """Capture clean and controlled initial states in a single bank."""

    from engine.env_state import snapshot_env_state

    phases = evenly_spaced_phases(env, protocol)
    snapshot_ids = canonical_snapshot_ids(phases)
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    noise_bank = NoiseBank(seed=protocol.snapshot_seed)

    original_reset_noise = bool(env.reset_noise)
    original_observation_noise = bool(env.observation_noise)
    original_pushes = bool(env.interval_pushes)
    try:
        env.reset_noise = False
        env.observation_noise = False
        env.interval_pushes = False
        env.reset_envs(env_ids, phase_indices=phases)
        clean = snapshot_env_state(env)

        _controlled_noisy_reset_from_clean(
            env,
            phases=phases,
            snapshot_ids=snapshot_ids,
            noise_bank=noise_bank,
        )
        controlled = snapshot_env_state(env)
    finally:
        env.reset_noise = original_reset_noise
        env.observation_noise = original_observation_noise
        env.interval_pushes = original_pushes

    clean_joint = clean["joint_pos"].index_select(1, env.action_joint_ids)
    controlled_joint = controlled["joint_pos"].index_select(1, env.action_joint_ids)
    clean_velocity = clean["root_velocity_w"]
    controlled_velocity = controlled["root_velocity_w"]
    reset_randomization = {
        "joint_pos_delta": controlled_joint - clean_joint,
        "root_pose_delta": controlled["root_pose_w"] - clean["root_pose_w"],
        "root_velocity_delta": controlled_velocity - clean_velocity,
        "noise_bank_seed": torch.full(
            (env.num_envs,), protocol.snapshot_seed, dtype=torch.long
        ),
    }
    startup = current_startup_randomization(env)
    state = {
        **_prefix_engine_state(clean, state_name=_CLEAN_STATE, num_envs=env.num_envs),
        **_prefix_engine_state(
            controlled,
            state_name=_CONTROLLED_STATE,
            num_envs=env.num_envs,
        ),
    }
    return SnapshotBank.from_batched_tensors(
        snapshot_ids=snapshot_ids,
        phase=phases.to(torch.float32).cpu(),
        state=state,
        reset_randomization=reset_randomization,
        physics_randomization=startup,
        bank_seed=protocol.snapshot_seed,
    )


def restore_snapshot_bank(
    env: Any,
    bank: SnapshotBank,
    *,
    mode: CollectorMode | str,
) -> None:
    from engine.env_state import restore_env_state

    parsed = parse_collector_mode(mode)
    state_name = _CLEAN_STATE if parsed is CollectorMode.CLEAN_MEAN else _CONTROLLED_STATE
    state = _engine_state_from_bank(bank, state_name=state_name)
    state = {name: value.to(env.device) for name, value in state.items()}
    # Writing generalized coordinates alone does not clear PhysX solver
    # warm-start/contact-manifold state.  The next simulated step can therefore
    # depend on whichever branch ran previously even when the public state and
    # action are bit-identical.  Validation already uses the same full reset
    # before restoring a saved state; canonical counterfactual branches must do
    # so as well or they are not condition matched.
    env.sim.reset()
    restore_startup_randomization(env, bank)
    restore_env_state(env, state)
    # PhysX property setters can round a float32 COM by one ULP on readback.
    # This uses the same absolute tolerance as the frozen replay identity gate;
    # it does not relax action, termination, or trajectory comparisons.
    assert_startup_randomization_matches(env, bank, atol=1.0e-6)


def policy_only_payload(path: str | Path) -> dict[str, Any]:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise DependencyUnavailable(f"checkpoint is missing: {checkpoint}")
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        payload = torch.load(checkpoint, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("policy"), Mapping):
        raise ProtocolError(f"checkpoint has no policy state mapping: {checkpoint}")
    return payload


def switch_policy_state(
    trainer: Any,
    checkpoint: DenseCheckpointRecord,
    *,
    expected_platform: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Switch only model/normalizer tensors; never restore env/optimizer/RNG."""

    if sha256_file(checkpoint.path) != checkpoint.sha256:
        raise ProtocolError(f"checkpoint changed before policy switch: {checkpoint.path}")
    payload = policy_only_payload(checkpoint.path)
    if int(payload.get("update_idx", -1)) != checkpoint.update:
        raise ProtocolError(
            f"checkpoint update mismatch for {checkpoint.path}: "
            f"expected={checkpoint.update}, payload={payload.get('update_idx')}"
        )
    platform = payload.get("platform_identity")
    if expected_platform is not None and dict(platform or {}) != dict(expected_platform):
        raise ProtocolError("checkpoint platform identity differs from frozen manifest")
    preflight = getattr(trainer.algo, "validate_checkpoint_payload", None)
    if callable(preflight):
        preflight(payload)
    before_env = state_digest(current_startup_randomization(trainer.env))
    trainer.algo.policy.load_state_dict(payload["policy"], strict=True)
    after_env = state_digest(current_startup_randomization(trainer.env))
    if before_env != after_env:
        raise ProtocolError("policy-only checkpoint switch mutated environment startup state")
    trainer.algo.actor.eval()
    trainer.algo.critic.eval()
    return payload


def make_collection_trainer(
    simulation_app: Any,
    *,
    repo_root: str | Path,
    protocol: CanonicalCollectionProtocol,
    runtime_dir: str | Path,
) -> Any:
    """Construct the sole real environment used by a stage-1 entrypoint."""

    from engine.config import load_config
    from engine.trainer import CoreTrainer

    root = Path(repo_root).expanduser().resolve()
    config_path = root / "configs" / "fixed_reward_largebox.yaml"
    if not config_path.is_file():
        raise DependencyUnavailable(f"fixed-reward config is missing: {config_path}")
    cfg = load_config(
        config_path,
        [
            f"environment.num_envs={protocol.num_envs}",
            f"training.seed={protocol.snapshot_seed}",
            "training.resume=''",
            "training.validation_every=0",
            "training.target_validation_steps=0",
        ],
    )
    torch.manual_seed(protocol.snapshot_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(protocol.snapshot_seed)
    return CoreTrainer(
        simulation_app,
        cfg,
        Path(runtime_dir).expanduser().resolve() / "checkpoints",
    )


class _SharedEnvironmentRandomness:
    """Inject NoiseBank observation/push draws keyed by snapshot and step."""

    def __init__(
        self,
        env: Any,
        bank: NoiseBank,
        snapshot_ids: Sequence[str],
        *,
        horizon: int,
    ) -> None:
        self.env = env
        self.bank = bank
        self.snapshot_ids = tuple(snapshot_ids)
        self.horizon = int(horizon)
        self.step = 0
        self.observation_call = 0
        self._original_add_noise = env._add_uniform_noise
        self._original_push = env._apply_interval_pushes

    def begin_observation(self, step: int) -> None:
        self.step = int(step)
        self.observation_call = 0

    def _add_uniform_noise(
        controller: "_SharedEnvironmentRandomness",
        env_self: Any,
        value: torch.Tensor,
        n_min: float,
        n_max: float,
    ) -> torch.Tensor:
        if not env_self.observation_noise:
            return value
        width = int(value[0].numel())
        stream = (
            f"observation:{controller.step}:call:{controller.observation_call}:"
            f"shape:{tuple(value.shape[1:])}"
        )
        controller.observation_call += 1
        noise = controller.bank.controlled_uniform(
            controller.snapshot_ids,
            horizon=1,
            width=width,
            stream=stream,
            low=float(n_min),
            high=float(n_max),
            dtype=value.dtype,
            device=value.device,
        )[:, 0].reshape_as(value)
        return value + noise

    def _apply_interval_pushes(
        controller: "_SharedEnvironmentRandomness",
        env_self: Any,
        eligible_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from envs.spec import VELOCITY_RANGE

        env_self._last_interval_push_mask.zero_()
        if not env_self.interval_pushes:
            return env_self._last_interval_push_mask.clone()
        if eligible_mask is None:
            eligible_mask = torch.ones(
                env_self.num_envs, dtype=torch.bool, device=env_self.device
            )
        eligible_mask = eligible_mask.to(env_self.device, dtype=torch.bool)
        due = torch.where(env_self.episode_steps >= env_self.next_push_step)[0]
        if due.numel():
            due = due[eligible_mask.index_select(0, due)]
        if not due.numel():
            return env_self._last_interval_push_mask.clone()
        env_self._last_interval_push_mask[due] = True
        first = due[env_self.first_push_step[due] < 0]
        if first.numel():
            env_self.first_push_step[first] = env_self.episode_steps[first]

        unit = controller.bank.controlled_uniform(
            controller.snapshot_ids,
            horizon=1,
            width=6,
            stream=f"interval_push_velocity:{controller.step}",
            low=0.0,
            high=1.0,
            device=env_self.device,
        )[:, 0]
        ranges = torch.tensor(VELOCITY_RANGE, device=env_self.device)
        delta = ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * unit
        velocity = env_self.get_mimic_root_velocity_w().index_select(0, due)
        velocity = velocity + delta.index_select(0, due)
        env_self.write_mimic_root_velocity_to_sim(velocity, due)

        min_interval, max_interval = env_self.push_interval_step_range
        schedule = controller.bank.controlled_uniform(
            controller.snapshot_ids,
            horizon=1,
            width=1,
            stream=f"interval_push_schedule:{controller.step}",
            low=0.0,
            high=1.0,
            device=env_self.device,
        )[:, 0, 0]
        interval = min_interval + torch.floor(
            schedule * float(max_interval - min_interval + 1)
        ).to(torch.long).clamp(max=max_interval - min_interval)
        env_self.next_push_step[due] = env_self.episode_steps[due] + interval[due]
        return env_self._last_interval_push_mask.clone()

    def __enter__(self) -> "_SharedEnvironmentRandomness":
        self.env._add_uniform_noise = MethodType(
            lambda env_self, value, n_min, n_max: self._add_uniform_noise(
                env_self, value, n_min, n_max
            ),
            self.env,
        )
        self.env._apply_interval_pushes = MethodType(
            lambda env_self, eligible_mask=None: self._apply_interval_pushes(
                env_self, eligible_mask
            ),
            self.env,
        )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.env._add_uniform_noise = self._original_add_noise
        self.env._apply_interval_pushes = self._original_push


@contextmanager
def collector_environment(
    env: Any,
    bank: NoiseBank,
    snapshot_ids: Sequence[str],
    *,
    mode: CollectorMode | str,
    horizon: int,
) -> Iterator[_SharedEnvironmentRandomness]:
    parsed = parse_collector_mode(mode)
    original = (
        bool(env.observation_noise),
        bool(env.reset_noise),
        bool(env.interval_pushes),
    )
    clean = parsed is CollectorMode.CLEAN_MEAN
    env.observation_noise = not clean
    env.reset_noise = not clean
    env.interval_pushes = not clean
    controller = _SharedEnvironmentRandomness(
        env,
        bank,
        snapshot_ids,
        horizon=horizon,
    )
    try:
        with controller:
            yield controller
    finally:
        env.observation_noise, env.reset_noise, env.interval_pushes = original


def _actor_spec_for_env(env: Any, repo_root: Path):
    cardinality = ObservationCardinality(
        action_joint_count=int(env.action_dim),
        termination_body_count=len(env.termination_body_indices),
        termination_contact_body_count=int(env.termination_contact_body_ids.numel()),
        foot_body_count=int(env.foot_contact_body_ids.numel()),
        track_body_count=len(env.track_body_names),
    )
    return derive_observation_spec(
        repo_root / "envs" / "observation.py",
        stream="actor",
        cardinality=cardinality,
    )


def _contact_mode(env: Any) -> torch.Tensor:
    masks = env.get_contact_mode_masks()
    result = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    result[masks["left_only"]] = 1
    result[masks["right_only"]] = 2
    result[masks["double_support"]] = 3
    return result


def _state_record(env: Any) -> dict[str, torch.Tensor | None]:
    robot = env.robot
    joint_pos, joint_vel = env.get_action_joint_state()
    root_pose = robot.data.root_link_pose_w
    root_velocity = env.get_mimic_root_velocity_w()
    forces = env.contact_sensor.data.net_forces_w_history
    undesired = (
        torch.linalg.vector_norm(
            forces[:, :, env.undesired_contact_body_ids], dim=-1
        ).amax(dim=1)
        > 1.0
    )
    torque = getattr(robot.data, "applied_torque", None)
    if torch.is_tensor(torque):
        torque = torque.index_select(1, env.action_joint_ids)
    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "root_pos": root_pose[:, :3],
        "root_quat": root_pose[:, 3:7],
        "root_lin_vel": root_velocity[:, :3],
        "root_ang_vel": root_velocity[:, 3:],
        "body_pos": robot.data.body_pos_w[:, env.track_body_ids],
        "body_quat": robot.data.body_quat_w[:, env.track_body_ids],
        "foot_contacts": env.get_foot_contact_mask(),
        "undesired_contacts": undesired,
        "applied_torque": torque,
        "joint_limits": robot.data.soft_joint_pos_limits[:, env.action_joint_ids],
    }


def _reference_record(env: Any) -> dict[str, torch.Tensor]:
    reference = env.get_reference_state()
    return {
        "joint_pos": reference["joint_pos"],
        "joint_vel": reference["joint_vel"],
        "root_pos": reference["root_pos_w"],
        "root_quat": reference["root_quat_w"],
        "body_pos": reference["body_pos_w"],
        "body_quat": reference["body_quat_w"],
    }


def _outcome_record(
    env: Any,
    reward: torch.Tensor,
    info: Mapping[str, Any],
) -> dict[str, Any]:
    context = env.get_tracking_context()
    reference = context["reference"]
    reward_terms = dict(info["reward_terms"])
    reward_terms["total"] = reward
    done_terms = dict(info["done_terms"])
    progress_denominator = max(1.0, float(env.motion_end_phase - env.motion_start_phase))
    progress = (env.phase_steps - float(env.motion_start_phase)) / progress_denominator
    anchor_error = torch.linalg.vector_norm(
        reference["anchor_pos_w"] - context["robot_anchor_pos_w"], dim=-1
    )
    body_error = torch.linalg.vector_norm(
        context["body_pos_relative_w"] - context["robot_body_pos_w"], dim=-1
    ).mean(dim=-1)
    joint_error = torch.linalg.vector_norm(
        reference["joint_pos"] - context["robot_joint_pos"], dim=-1
    ) / math.sqrt(float(env.action_dim))
    return {
        "reward_terms": reward_terms,
        "done_terms": done_terms,
        "reference_progress": progress,
        "anchor_error": anchor_error,
        "body_error": body_error,
        "joint_error": joint_error,
        "action_rate": reward_terms["action_rate"],
        "joint_limit_cost": reward_terms["joint_limit"],
        "undesired_contact_cost": reward_terms["undesired_contacts"],
    }


def _metadata(
    manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
    checkpoint: DenseCheckpointRecord,
    mode: CollectorMode,
    collector_seed: int,
    policy_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "suite_version": str(spec.get("suite_version", "")),
        "git_commit": str(manifest["git_commit"]),
        "git_dirty": bool(manifest["git_dirty"]),
        "source_snapshot_sha256": str(manifest["source_snapshot_sha256"]),
        "resolved_config_sha256": str(manifest["resolved_config_sha256"]),
        "checkpoint_sha256": checkpoint.sha256,
        "checkpoint_update": checkpoint.update,
        "checkpoint_lineage_id": checkpoint.lineage_id,
        "policy_domain": checkpoint.policy_domain,
        "robot_asset_sha256": str(manifest["robot_asset_sha256"]),
        "motion_sha256": str(manifest["motion_sha256"]),
        "action_schema_sha256": str(manifest["action_schema_sha256"]),
        "task_name": str(manifest["task_name"]),
        "collector_mode": mode.value,
        "collector_seed": int(collector_seed),
        "policy_method": str(policy_provenance["policy_method"]),
        "policy_source_commit": str(policy_provenance["policy_source_commit"]),
        "policy_source_snapshot_sha256": str(
            policy_provenance["policy_source_snapshot_sha256"]
        ),
        "policy_resolved_config_sha256": str(
            policy_provenance["policy_resolved_config_sha256"]
        ),
        **imitation_contract_metadata(),
    }


def validate_canonical_rollout_tree(tree: Mapping[str, Any]) -> None:
    """Validate the eight-section canonical discovery shard contract."""

    missing = CANONICAL_REQUIRED_ROLLOUT_SECTIONS - set(tree)
    unknown = set(tree) - CANONICAL_REQUIRED_ROLLOUT_SECTIONS
    if missing or unknown:
        raise ProtocolError(
            f"canonical rollout sections mismatch: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    metadata = tree.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ProtocolError("canonical rollout metadata must be a mapping")
    expected_imitation_metadata = imitation_contract_metadata()
    mismatched_metadata = {
        key: {"expected": expected, "actual": metadata.get(key)}
        for key, expected in expected_imitation_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatched_metadata:
        raise ProtocolError(
            f"canonical imitation metadata mismatch: {mismatched_metadata}"
        )
    mode = parse_collector_mode(str(metadata.get("collector_mode")))
    declared_eligibility = metadata.get(
        "eligible_for_primary_overlap", is_overlap_eligible(mode)
    )
    if bool(declared_eligibility) != bool(is_overlap_eligible(mode)):
        raise ProtocolError("collector overlap eligibility contradicts collector mode")

    done = tree["trajectory"].get("done")
    if not torch.is_tensor(done) or done.ndim != 2:
        raise ProtocolError("canonical trajectory.done must have shape [T,N]")
    time_steps, num_envs = done.shape
    imitation = tree["imitation"]
    if not isinstance(imitation, Mapping):
        raise ProtocolError("canonical imitation section must be a mapping")
    required = {
        "agent_physx_raw_frame",
        "agent_fk_aligned_raw_frame",
        "reference_expert_raw_frame",
        "phase_normalized_pre_step",
    }
    if set(imitation) != required:
        raise ProtocolError(
            "canonical imitation fields mismatch: "
            f"expected={sorted(required)}, actual={sorted(imitation)}"
        )
    for name in (
        "agent_physx_raw_frame",
        "agent_fk_aligned_raw_frame",
        "reference_expert_raw_frame",
    ):
        value = imitation[name]
        expected_shape = (time_steps, num_envs, IMITATION_FRAME_DIM)
        if not torch.is_tensor(value) or tuple(value.shape) != expected_shape:
            raise ProtocolError(
                f"imitation.{name} must have shape {expected_shape}, "
                f"got {getattr(value, 'shape', None)}"
            )
        if not bool(torch.isfinite(value).all()):
            raise ProtocolError(f"imitation.{name} contains NaN or Inf")
    phase = imitation["phase_normalized_pre_step"]
    if not torch.is_tensor(phase) or tuple(phase.shape) != (time_steps, num_envs):
        raise ProtocolError(
            "imitation.phase_normalized_pre_step must have shape [T,N]"
        )
    if not bool(torch.isfinite(phase).all()) or bool(
        (phase < 0.0).any() or (phase > 1.0).any()
    ):
        raise ProtocolError("pre-step normalized phase must be finite in [0,1]")


def save_canonical_rollout_shard(
    tree: Mapping[str, Any], path: str | Path
) -> Path:
    validate_canonical_rollout_tree(tree)
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(dict(tree), temporary)
    temporary.replace(destination)
    return destination


def summarize_rollout_tree(tree: Mapping[str, Any]) -> dict[str, float]:
    trajectory = tree["trajectory"]
    outcome = tree["outcome"]
    done = trajectory["done"].bool()
    time_steps, env_count = done.shape
    first_done = torch.full((env_count,), time_steps, dtype=torch.long)
    for env_index in range(env_count):
        indices = torch.where(done[:, env_index])[0]
        if indices.numel():
            first_done[env_index] = indices[0] + 1
    last_index = torch.clamp(first_done - 1, min=0, max=time_steps - 1)
    columns = torch.arange(env_count)
    completed = trajectory["motion_complete"][last_index, columns].float()
    failed = trajectory["failure"][last_index, columns].float()
    progress = outcome["reference_progress"][last_index, columns].float()
    rewards = outcome["reward_terms"]["total"].float()
    valid = torch.arange(time_steps)[:, None] < first_done[None, :]
    returns = (rewards * valid.to(rewards)).sum(dim=0)
    return {
        "steps_mean": float(first_done.float().mean().item()),
        "motion_complete_frac": float(completed.mean().item()),
        "failure_frac": float(failed.mean().item()),
        "reference_progress_mean": float(progress.mean().item()),
        "return_mean": float(returns.mean().item()),
    }


def collect_canonical_branch(
    trainer: Any,
    bank: SnapshotBank,
    checkpoint: DenseCheckpointRecord,
    *,
    protocol: CanonicalCollectionProtocol,
    manifest: Mapping[str, Any],
    spec: Mapping[str, Any],
    repo_root: str | Path,
    mode: CollectorMode | str,
    common_sigma: float,
    policy_adapter: CanonicalPolicyAdapter | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, float]]:
    parsed = parse_collector_mode(mode)
    if parsed is CollectorMode.COMMON_ACTION_NOISE:
        if common_sigma not in protocol.common_sigmas:
            raise ProtocolError("common-action branch uses an unfrozen sigma")
    elif common_sigma != 0.0:
        raise ProtocolError("only common_action_noise branches may set common_sigma")
    env = trainer.env
    if int(env.num_envs) != len(bank):
        raise ProtocolError("snapshot bank size differs from the persistent environment")
    expected_platform = {
        "dataset_sha256": str(manifest["motion_sha256"]),
        "robot_asset_sha256": str(manifest["robot_asset_sha256"]),
        "action_schema_sha256": str(manifest["action_schema_sha256"]),
    }
    if policy_adapter is None:
        switch_policy_state(trainer, checkpoint, expected_platform=expected_platform)
        active_policy: CanonicalPolicyAdapter = _TrainerPolicyAdapter(trainer, manifest)
    else:
        active_policy = policy_adapter
    if str(active_policy.policy_domain) != checkpoint.policy_domain:
        raise ProtocolError(
            "policy adapter/checkpoint domain mismatch: "
            f"adapter={active_policy.policy_domain}, checkpoint={checkpoint.policy_domain}"
        )
    assert_startup_randomization_matches(env, bank, atol=1.0e-6)
    restore_snapshot_bank(env, bank, mode=parsed)
    active_policy.reset_branch(
        env=env,
        mode=parsed,
        snapshot_ids=bank.snapshot_ids,
    )

    noise_bank = NoiseBank(seed=protocol.collector_seed)
    snapshot_ids = bank.snapshot_ids
    epsilon = noise_bank.common_action_epsilon(
        snapshot_ids,
        horizon=protocol.horizon,
        action_dim=int(env.action_dim),
        device=env.device,
    )
    actor_spec = _actor_spec_for_env(env, Path(repo_root).expanduser().resolve())
    imitation_adapter = Commit6901ImitationAdapter(env, repo_root=repo_root)
    accumulator = TensorTreeAccumulator()
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    action_low = trainer.algo.action_low
    action_high = trainer.algo.action_high

    with collector_environment(
        env,
        noise_bank,
        snapshot_ids,
        mode=parsed,
        horizon=protocol.horizon,
    ) as randomness:
        randomness.begin_observation(0)
        observation = env.get_observation()
        for step in range(protocol.horizon):
            actor_views = build_actor_feature_views(observation, actor_spec)
            critic = env.get_critic_observation()
            state = _state_record(env)
            reference = _reference_record(env)
            phase = env.phase_steps.clone()
            imitation = imitation_adapter.record(phase)
            episode_id = env.episode_ids.clone()
            contact = _contact_mode(env)
            action = active_policy.action_record(
                env=env,
                observation=observation,
                imitation=imitation,
                common_epsilon=epsilon[:, step],
                mode=parsed,
                common_sigma=common_sigma,
                action_low=action_low,
                action_high=action_high,
                step=step,
            )
            randomness.begin_observation(step + 1)
            next_observation, reward, done, info = env.step(action.sampled)
            if not torch.equal(env.last_action, action.applied):
                maximum = float(torch.max(torch.abs(env.last_action - action.applied)).item())
                raise ProtocolError(
                    f"collector applied-action path differs from environment; max={maximum}"
                )
            done_terms = info["done_terms"]
            failure = (
                done_terms["anchor_pos_bad"]
                | done_terms["anchor_ori_bad"]
                | done_terms["ee_body_bad"]
            )
            accumulator.append(
                {
                    "trajectory": {
                        "env_id": env_ids,
                        "episode_id": episode_id,
                        "step": torch.full_like(env_ids, step),
                        "phase": torch.floor(phase).to(torch.long),
                        "phase_continuous": phase,
                        "reset_stream": torch.zeros_like(env_ids),
                        "contact_mode": contact,
                        "done": done.bool(),
                        "failure": failure.bool(),
                        "timeout": done_terms["time_out"].bool(),
                        "motion_complete": done_terms["motion_complete"].bool(),
                    },
                    "observation": {
                        "actor_full": actor_views.actor_full,
                        "actor_reference_terms": actor_views.actor_reference_terms,
                        "actor_proprio_terms": actor_views.actor_proprio_terms,
                        "actor_no_reference": actor_views.actor_no_reference,
                        "critic_full": critic,
                    },
                    "state": state,
                    "action": {
                        "mean": action.mean,
                        "std": action.std,
                        "common_epsilon": action.common_epsilon,
                        "sampled": action.sampled,
                        "applied": action.applied,
                    },
                    "reference": reference,
                    "outcome": _outcome_record(env, reward, info),
                    "imitation": imitation,
                }
            )
            observation = next_observation

    stacked = accumulator.finalize()
    suffix = (
        f"-sigma-{str(common_sigma).replace('.', 'p')}"
        if parsed is CollectorMode.COMMON_ACTION_NOISE
        else ""
    )
    trajectory_ids = [
        f"{checkpoint.checkpoint_id}-{parsed.value}{suffix}-{snapshot_id}"
        for snapshot_id in snapshot_ids
    ]
    tree = {
        "metadata": _metadata(
            manifest,
            spec,
            checkpoint,
            parsed,
            protocol.collector_seed,
            active_policy.provenance(),
        ),
        "trajectory": {
            "trajectory_id": trajectory_ids,
            "snapshot_id": list(snapshot_ids),
            **stacked["trajectory"],
        },
        "observation": stacked["observation"],
        "state": stacked["state"],
        "action": stacked["action"],
        "reference": stacked["reference"],
        "outcome": stacked["outcome"],
        "imitation": stacked["imitation"],
    }
    validate_canonical_rollout_tree(tree)
    rows = index_rows_from_tree(
        tree,
        checkpoint=checkpoint,
        mode=parsed,
        common_sigma=common_sigma,
        horizon=protocol.horizon,
    )
    return tree, rows, summarize_rollout_tree(tree)


def index_rows_from_tree(
    tree: Mapping[str, Any],
    *,
    checkpoint: DenseCheckpointRecord,
    mode: CollectorMode | str,
    common_sigma: float,
    horizon: int,
) -> list[dict[str, Any]]:
    parsed = parse_collector_mode(mode)
    done = tree["trajectory"]["done"]
    snapshot_ids = tuple(str(value) for value in tree["trajectory"]["snapshot_id"])
    trajectory_ids = tuple(str(value) for value in tree["trajectory"]["trajectory_id"])
    if done.ndim != 2 or done.shape[1] != len(snapshot_ids):
        raise ProtocolError("rollout tree done tensor is not aligned with snapshot ids")
    if len(trajectory_ids) != len(snapshot_ids):
        raise ProtocolError("rollout tree trajectory/snapshot IDs are not aligned")
    rows: list[dict[str, Any]] = []
    for env_index, (snapshot_id, trajectory_id) in enumerate(
        zip(snapshot_ids, trajectory_ids)
    ):
        indices = torch.where(done[:, env_index])[0]
        num_steps = int(indices[0].item() + 1) if indices.numel() else int(horizon)
        episode = int(tree["trajectory"]["episode_id"][0, env_index].item())
        sample_id = trajectory_id
        rows.append(
            {
                "sample_id": sample_id,
                "trajectory_id": trajectory_id,
                "snapshot_id": snapshot_id,
                "env_id": env_index,
                "episode_id": episode,
                "checkpoint_id": checkpoint.checkpoint_id,
                "checkpoint_path": str(checkpoint.path),
                "checkpoint_sha256": checkpoint.sha256,
                "checkpoint_update": checkpoint.update,
                "checkpoint_lineage_id": checkpoint.lineage_id,
                "policy_domain": checkpoint.policy_domain,
                "collector_mode": parsed.value,
                "common_sigma": float(common_sigma),
                "eligible_for_primary_overlap": bool(is_overlap_eligible(parsed)),
                "shard_path": "",
                "shard_env_index": env_index,
                "num_steps": num_steps,
            }
        )
    return rows


def write_rollout_index(rows: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    if not rows:
        raise ProtocolError("cannot write an empty canonical rollout index")
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(target)
    try:
        import pandas as pd
    except ImportError as exc:
        raise DependencyUnavailable("pandas/pyarrow are required for parquet output") from exc
    frame = pd.DataFrame([{key: row.get(key) for key in CANONICAL_INDEX_FIELDS} for row in rows])
    if tuple(frame.columns) != CANONICAL_INDEX_FIELDS:
        raise ProtocolError("canonical index columns differ from frozen contract")
    buffer = io.BytesIO()
    try:
        frame.to_parquet(buffer, index=False)
    except (ImportError, ModuleNotFoundError) as exc:
        raise DependencyUnavailable("pyarrow or fastparquet is required") from exc
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(buffer.getvalue())
    return target


def load_rollout_index(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DependencyUnavailable(f"canonical rollout index is missing: {source}")
    try:
        import pandas as pd
    except ImportError as exc:
        raise DependencyUnavailable("pandas/pyarrow are required for parquet input") from exc
    frame = pd.read_parquet(source)
    if tuple(frame.columns) != CANONICAL_INDEX_FIELDS:
        raise ProtocolError(
            f"canonical rollout index columns changed: {tuple(frame.columns)}"
        )
    rows = frame.to_dict(orient="records")
    if not rows:
        raise ProtocolError("canonical rollout index is empty")
    return rows


def load_rollout_shard(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DependencyUnavailable(f"rollout shard is missing: {source}")
    try:
        tree = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:
        tree = torch.load(source, map_location="cpu")
    if not isinstance(tree, dict):
        raise ProtocolError(f"rollout shard is not a mapping: {source}")
    return tree


def resolve_shard_path(index_path: str | Path, row: Mapping[str, Any]) -> Path:
    raw = Path(str(row["shard_path"]))
    if raw.is_absolute():
        raise ProtocolError("rollout index shard paths must be output-relative")
    base = Path(index_path).expanduser().resolve().parent
    path = (base / raw).resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:
        raise ProtocolError("rollout shard path escapes suite output") from exc
    return path


def _slice_trajectory_tree(value: Any, env_index: int, num_steps: int) -> Any:
    if torch.is_tensor(value):
        if value.ndim < 2:
            return value
        return value[:num_steps, env_index].clone()
    if isinstance(value, Mapping):
        return {
            str(key): _slice_trajectory_tree(nested, env_index, num_steps)
            for key, nested in value.items()
        }
    if value is None:
        return None
    return value


def load_rollout_trajectory(
    index_path: str | Path,
    row: Mapping[str, Any],
    *,
    cache: MutableMapping[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    shard_path = resolve_shard_path(index_path, row)
    key = str(shard_path)
    tree = cache.get(key) if cache is not None else None
    if tree is None:
        tree = load_rollout_shard(shard_path)
        if cache is not None:
            cache[key] = tree
    env_index = int(row["shard_env_index"])
    num_steps = int(row["num_steps"])
    result = {"metadata": dict(tree["metadata"])}
    for section in (
        "trajectory",
        "observation",
        "state",
        "action",
        "reference",
        "outcome",
        "imitation",
    ):
        result[section] = _slice_trajectory_tree(tree[section], env_index, num_steps)
    result["trajectory"]["trajectory_id"] = str(row["trajectory_id"])
    result["trajectory"]["snapshot_id"] = str(row["snapshot_id"])
    return result


def save_branch_shard(
    tree: Mapping[str, Any],
    *,
    output_dir: str | Path,
    checkpoint: DenseCheckpointRecord,
    mode: CollectorMode | str,
    common_sigma: float,
) -> tuple[Path, str]:
    parsed = parse_collector_mode(mode)
    relative = branch_shard_relative(checkpoint, parsed, common_sigma)
    destination = Path(output_dir).expanduser().resolve() / relative
    if destination.exists():
        raise FileExistsError(destination)
    save_canonical_rollout_shard(tree, destination)
    return destination, relative.as_posix()


def branch_shard_relative(
    checkpoint: DenseCheckpointRecord,
    mode: CollectorMode | str,
    common_sigma: float,
) -> Path:
    parsed = parse_collector_mode(mode)
    sigma = (
        f"_sigma_{str(common_sigma).replace('.', 'p')}"
        if parsed is CollectorMode.COMMON_ACTION_NOISE
        else ""
    )
    return Path("rollouts") / (
        f"checkpoint_{checkpoint.update:04d}_{checkpoint.sha256[:12]}_"
        f"{parsed.value}{sigma}.pt"
    )


def close_collection_trainer(trainer: Any) -> None:
    logger = getattr(trainer, "metrics_logger", None)
    close = getattr(logger, "close", None)
    if callable(close):
        close()


__all__ = [
    "CANONICAL_INDEX_FIELDS",
    "CANONICAL_REQUIRED_ROLLOUT_SECTIONS",
    "CanonicalCollectionProtocol",
    "CanonicalPolicyAdapter",
    "DenseCheckpointRecord",
    "assert_startup_randomization_matches",
    "build_snapshot_bank_from_environment",
    "branch_shard_relative",
    "canonical_snapshot_ids",
    "close_collection_trainer",
    "collect_canonical_branch",
    "collector_environment",
    "current_startup_randomization",
    "evenly_spaced_phases",
    "load_rollout_index",
    "load_rollout_shard",
    "load_rollout_trajectory",
    "make_collection_trainer",
    "policy_only_payload",
    "read_dense_checkpoint_records",
    "resolve_shard_path",
    "restore_snapshot_bank",
    "save_branch_shard",
    "save_canonical_rollout_shard",
    "snapshot_bank_startup_fingerprint",
    "summarize_rollout_tree",
    "switch_policy_state",
    "index_rows_from_tree",
    "validate_frozen_collection_semantics",
    "validate_canonical_rollout_tree",
    "write_rollout_index",
]
