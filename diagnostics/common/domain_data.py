"""Serializable, leakage-audited data contract for Stage-3 domain probes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .manifest import DependencyUnavailable, ProtocolError, canonical_sha256, read_json, sha256_file
from .noise_bank import require_overlap_eligible


SPLITS = ("train", "validation", "test")
REQUIRED_DOMAIN_IDENTITIES = {
    "K": "K",
    "T_u200": "T_early",
    "T_u500": "T",
    "A_amp": "A_amp",
    "B": "B",
}
REQUIRED_DOMAIN_FAMILIES = tuple(REQUIRED_DOMAIN_IDENTITIES.values())
GAP_LAYER_NAMES = (
    "motion_npz", "reset_readback", "teacher_one_step", "teacher_closed_loop"
)
AMP_SOURCE_COMMIT = "6901e302499711e2207687e1342348a4078330f8"
AMP_FRAME_DIM = 239
AMP_WINDOW_STEPS = 10
AMP_FRAME_BLOCKS = (
    ("root_pos", 0, 3),
    ("root_rot6d", 3, 9),
    ("joint_rot6d", 9, 189),
    ("key_body_pos", 189, 204),
    ("root_lin_vel", 204, 207),
    ("root_ang_vel", 207, 210),
    ("dof_vel", 210, 239),
)


def _unicode(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    strings = [str(value) for value in np.asarray(values).reshape(-1).tolist()]
    width = max(1, *(len(value) for value in strings))
    return np.asarray(strings, dtype=f"<U{width}")


def amp_window_feature_groups(window_steps: int = AMP_WINDOW_STEPS) -> dict[str, tuple[int, int]]:
    groups: dict[str, tuple[int, int]] = {}
    for step in range(int(window_steps)):
        offset = step * AMP_FRAME_DIM
        for name, start, stop in AMP_FRAME_BLOCKS:
            groups[f"t{step:02d}.{name}"] = (offset + start, offset + stop)
    return groups


def chronological_amp_windows(
    frames: np.ndarray,
    *,
    done: np.ndarray,
    window_steps: int = AMP_WINDOW_STEPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Build exact chronological AMP windows without reset/wrap padding."""

    frames = np.asarray(frames, dtype=np.float32)
    done = np.asarray(done, dtype=bool).reshape(-1)
    if frames.ndim != 2 or frames.shape[1] != AMP_FRAME_DIM or done.shape != (frames.shape[0],):
        raise ProtocolError("raw AMP frames/done must align as [T,239]/[T]")
    windows: list[np.ndarray] = []
    endpoints: list[int] = []
    for endpoint in range(int(window_steps) - 1, frames.shape[0]):
        start = endpoint + 1 - int(window_steps)
        # require_complete_alive_window: a transition marked done is not an
        # alive endpoint and no reset padding/wrapping is ever introduced.
        if bool(done[start : endpoint + 1].any()):
            continue
        window = frames[start : endpoint + 1].copy()
        window[:, :2] -= window[-1:, :2]
        windows.append(window.reshape(-1))
        endpoints.append(endpoint)
    if not windows:
        raise DependencyUnavailable("trajectory contains no complete alive ten-frame AMP window")
    return np.stack(windows), np.asarray(endpoints, dtype=np.int64)


def _to_numpy(value: Any) -> np.ndarray:
    try:
        import torch
    except ImportError:  # pragma: no cover - canonical shards require torch
        torch = None
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _nested(tree: Mapping[str, Any], dotted: str) -> Any:
    value: Any = tree
    for key in dotted.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise DependencyUnavailable(f"rollout lacks required exact field {dotted!r}")
        value = value[key]
    return value


def build_amp_domain_bundle_from_rollouts(
    entry: Mapping[str, Any],
    *,
    catalog_dir: Path,
    split_audit_path: Path,
    split_source_index: Path,
) -> DomainBundle:
    """Derive a domain packet only from unified collector shards.

    ``agent_fk_aligned_raw`` is intentionally rejected: it exists solely for
    diag_32 decomposition and may never make the source gap look smaller.
    """

    from .canonical_collection import (
        load_rollout_index, load_rollout_trajectory, resolve_shard_path,
    )
    from .imitation_6901 import imitation_contract_metadata

    role_to_field = {
        "agent_physx": "imitation.agent_physx_raw_frame",
        "reference_expert": "imitation.reference_expert_raw_frame",
    }
    role = str(entry.get("frame_role", ""))
    if role == "agent_fk_aligned" or "agent_fk_aligned_raw" in str(entry.get("frame_field", "")):
        raise ProtocolError("agent_fk_aligned_raw_frame is diag_32-only and forbidden in diag_30/36")
    frame_field = str(entry.get("frame_field") or role_to_field.get(role, ""))
    if frame_field not in set(role_to_field.values()):
        raise ProtocolError("domain catalog must select agent_physx or reference_expert raw frames")
    family = str(entry.get("family", ""))
    if family == "K" and role != "reference_expert":
        raise ProtocolError("K must use reference_expert_raw_frame")
    if family != "K" and role != "agent_physx":
        raise ProtocolError("T/A/B execution domains must use agent_physx_raw_frame")
    index_value = str(entry.get("rollout_index", ""))
    if not index_value:
        raise ProtocolError("rollout-derived domain lacks rollout_index")
    index_raw = Path(index_value).expanduser()
    index_path = (index_raw if index_raw.is_absolute() else catalog_dir / index_raw).resolve()
    rows = load_rollout_index(index_path)
    filters = entry.get("filters")
    if not isinstance(filters, Mapping) or not filters:
        raise ProtocolError("rollout-derived domain requires explicit frozen filters")
    selected = [
        row for row in rows
        if all(str(row.get(key)) == str(expected) for key, expected in filters.items())
    ]
    if not selected:
        raise DependencyUnavailable(f"no collector rows match domain filters for {entry.get('name')}")
    checkpoint_hashes = {str(row.get("checkpoint_sha256", "")) for row in selected}
    checkpoint_ids = {str(row.get("checkpoint_id", "")) for row in selected}
    lineages = {str(row.get("checkpoint_lineage_id", "")) for row in selected}
    if len(checkpoint_hashes) != 1 or len(checkpoint_ids) != 1 or len(lineages) != 1:
        raise ProtocolError(
            "one domain must contain exactly one checkpoint/model identity and lineage"
        )
    checkpoint_hash = next(iter(checkpoint_hashes))
    if len(checkpoint_hash) != 64:
        raise ProtocolError("domain checkpoint/model SHA256 is missing")
    modes = {str(row["collector_mode"]) for row in selected}
    require_overlap_eligible(modes)
    if len(modes) != 1:
        raise ProtocolError("one domain may not mix collector modes")
    if modes == {"common_action_noise"} and len({float(row["common_sigma"]) for row in selected}) != 1:
        raise ProtocolError("one domain may not mix common-action-noise scales")

    split_payload = read_json(split_audit_path)
    if split_payload.get("status") != "PASS":
        raise DependencyUnavailable("diag_14 split audit is not PASS")
    sample_to_split = split_payload.get("evidence", {}).get("inner_snapshot_split", {}).get("sample_to_split")
    if not isinstance(sample_to_split, Mapping):
        raise ProtocolError("diag_14 lacks sample_to_split mapping")
    split_rows = load_rollout_index(split_source_index)
    snapshot_to_split: dict[str, str] = {}
    for row in split_rows:
        sample = str(row["sample_id"])
        if sample not in sample_to_split:
            continue
        snapshot = str(row["snapshot_id"])
        split = str(sample_to_split[sample])
        previous = snapshot_to_split.setdefault(snapshot, split)
        if previous != split:
            raise ProtocolError("diag_14 maps one snapshot to multiple splits")
    if not snapshot_to_split:
        raise ProtocolError("cannot derive the frozen snapshot split")

    feature_parts: list[np.ndarray] = []
    scalar_parts: dict[str, list[np.ndarray]] = {
        key: [] for key in (
            "split", "sample_ids", "trajectory_ids", "snapshot_ids",
            "checkpoint_lineage_ids", "phase", "contact_mode", "failure",
            "collector_mode",
        )
    }
    cache: dict[str, dict[str, Any]] = {}
    contract_reference: Mapping[str, Any] | None = None
    skipped_short_trajectories = 0
    for row in selected:
        tree = load_rollout_trajectory(index_path, row, cache=cache)
        metadata = tree.get("metadata")
        if not isinstance(metadata, Mapping):
            raise ProtocolError("rollout metadata must be a mapping")
        expected_contract = imitation_contract_metadata()
        contract = {key: metadata.get(key) for key in expected_contract}
        mismatches = {
            key: {"expected": expected, "actual": contract.get(key)}
            for key, expected in expected_contract.items()
            if contract.get(key) != expected
        }
        if mismatches:
            raise ProtocolError(f"raw imitation-frame contract mismatch: {mismatches}")
        schema_hash = str(contract.get("imitation_frame_schema_sha256", ""))
        if len(schema_hash) != 64 or any(character not in "0123456789abcdef" for character in schema_hash):
            raise ProtocolError("imitation_frame_schema_sha256 is not a lowercase SHA256")
        if contract_reference is None:
            contract_reference = dict(contract)
        elif dict(contract) != dict(contract_reference):
            raise ProtocolError("imitation contracts differ across rollout shards")
        frames = _to_numpy(_nested(tree, frame_field))
        done = _to_numpy(_nested(tree, "trajectory.done")).astype(bool)
        phases = _to_numpy(_nested(tree, "imitation.phase_normalized_pre_step")).astype(np.float64)
        contacts = _to_numpy(_nested(tree, "trajectory.contact_mode"))
        failures = _to_numpy(_nested(tree, "trajectory.failure")).astype(bool)
        if phases.shape != (frames.shape[0],) or contacts.shape != phases.shape or failures.shape != phases.shape:
            raise ProtocolError("imitation endpoint metadata is not aligned with raw frames")
        if np.any(phases < 0.0) or np.any(phases > 1.0) or np.any(np.diff(phases) < -1.0e-12):
            raise ProtocolError("pre-step normalized phase wraps or leaves [0,1]")
        try:
            windows, endpoints = chronological_amp_windows(frames, done=done)
        except DependencyUnavailable:
            skipped_short_trajectories += 1
            continue
        feature_parts.append(windows)
        count = endpoints.size
        snapshot = str(row["snapshot_id"])
        if snapshot not in snapshot_to_split:
            raise ProtocolError(f"snapshot {snapshot!r} has no frozen split")
        scalar_parts["split"].append(np.full(count, snapshot_to_split[snapshot]))
        scalar_parts["sample_ids"].append(
            np.asarray([f"{entry['name']}:{row['sample_id']}:end={int(end)}" for end in endpoints])
        )
        scalar_parts["trajectory_ids"].append(np.full(count, str(row["trajectory_id"])))
        scalar_parts["snapshot_ids"].append(np.full(count, snapshot))
        scalar_parts["checkpoint_lineage_ids"].append(np.full(count, str(row["checkpoint_lineage_id"])))
        scalar_parts["phase"].append(phases[endpoints])
        scalar_parts["contact_mode"].append(contacts[endpoints])
        failure_label = str(entry.get("failure_label", ""))
        if failure_label == "trajectory_eventual":
            scalar_parts["failure"].append(np.full(count, bool(failures.any())))
        elif failure_label == "endpoint":
            scalar_parts["failure"].append(failures[endpoints])
        else:
            raise ProtocolError("domain catalog must freeze failure_label as endpoint or trajectory_eventual")
        scalar_parts["collector_mode"].append(np.full(count, str(row["collector_mode"])))
    if not feature_parts:
        raise DependencyUnavailable(
            f"domain {entry.get('name')} has no complete alive ten-frame windows"
        )
    features = np.concatenate(feature_parts, axis=0)
    values = {name: np.concatenate(parts) for name, parts in scalar_parts.items()}
    split_counts = {
        split: int(np.sum(_unicode(values["split"]) == split)) for split in SPLITS
    }
    if any(count < 2 for count in split_counts.values()):
        raise DependencyUnavailable(
            f"domain {entry.get('name')} lacks windows in a frozen split: {split_counts}"
        )
    shard_paths = sorted({resolve_shard_path(index_path, row) for row in selected})
    return DomainBundle(
        name=str(entry["name"]), family=family, features=features,
        split=values["split"], sample_ids=values["sample_ids"],
        trajectory_ids=values["trajectory_ids"], snapshot_ids=values["snapshot_ids"],
        checkpoint_lineage_ids=values["checkpoint_lineage_ids"],
        phase=values["phase"], contact_mode=values["contact_mode"],
        failure=values["failure"], collector_mode=values["collector_mode"],
        feature_groups=amp_window_feature_groups(), window_steps=AMP_WINDOW_STEPS,
        coordinate_frame="mimickit_g1_amp_window_v1",
        metadata={
            "amp_source_commit": AMP_SOURCE_COMMIT,
            "amp_representation": "mimickit_g1_chronological_window",
            "amp_frame_dim": AMP_FRAME_DIM,
            "amp_window_steps": AMP_WINDOW_STEPS,
            "amp_root_xy_anchor": "newest_frame",
            "frame_role": role,
            "frame_field": frame_field,
            "rollout_index": str(index_path),
            "rollout_index_sha256": sha256_file(index_path),
            "split_audit_sha256": sha256_file(split_audit_path),
            "source_shards": [
                {"path": str(path), "sha256": sha256_file(path)} for path in shard_paths
            ],
            "derived_from_unified_collector": True,
            "filters": dict(filters),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_id": next(iter(checkpoint_ids)),
            "checkpoint_lineage_id": next(iter(lineages)),
            "failure_label": str(entry["failure_label"]),
            "skipped_short_trajectories": skipped_short_trajectories,
            "split_counts": split_counts,
            "imitation_contract": dict(contract_reference or {}),
        },
    )


@dataclass(frozen=True, slots=True)
class DomainBundle:
    name: str
    family: str
    features: np.ndarray
    split: np.ndarray
    sample_ids: np.ndarray
    trajectory_ids: np.ndarray
    snapshot_ids: np.ndarray
    checkpoint_lineage_ids: np.ndarray
    phase: np.ndarray
    contact_mode: np.ndarray
    failure: np.ndarray
    collector_mode: np.ndarray
    feature_groups: Mapping[str, tuple[int, int]]
    window_steps: int
    coordinate_frame: str
    metadata: Mapping[str, Any]

    def validate(self, *, primary: bool = True) -> None:
        features = np.asarray(self.features, dtype=np.float64)
        if self.name not in REQUIRED_DOMAIN_IDENTITIES:
            raise ProtocolError(f"invalid domain identity {self.name!r}/{self.family!r}")
        expected_family = REQUIRED_DOMAIN_IDENTITIES[self.name]
        if self.family != expected_family:
            raise ProtocolError(
                f"domain {self.name!r} must use family {expected_family!r}, got {self.family!r}"
            )
        if features.ndim != 2 or features.shape[0] < 6 or features.shape[1] < 1:
            raise ProtocolError(f"domain {self.name} features must have shape [N,D]")
        if not np.isfinite(features).all():
            raise ProtocolError(f"domain {self.name} contains NaN or Inf")
        count = features.shape[0]
        aligned = {
            "split": np.asarray(self.split),
            "sample_ids": np.asarray(self.sample_ids),
            "trajectory_ids": np.asarray(self.trajectory_ids),
            "snapshot_ids": np.asarray(self.snapshot_ids),
            "checkpoint_lineage_ids": np.asarray(self.checkpoint_lineage_ids),
            "phase": np.asarray(self.phase),
            "contact_mode": np.asarray(self.contact_mode),
            "failure": np.asarray(self.failure),
            "collector_mode": np.asarray(self.collector_mode),
        }
        for field, values in aligned.items():
            if values.shape != (count,):
                raise ProtocolError(f"domain {self.name}.{field} must have shape {(count,)}")
        if len(set(_unicode(self.sample_ids).tolist())) != count:
            raise ProtocolError(f"domain {self.name} sample_ids are not unique")
        unknown_splits = set(_unicode(self.split).tolist()) - set(SPLITS)
        if unknown_splits:
            raise ProtocolError(f"domain {self.name} has invalid splits {sorted(unknown_splits)}")
        for split in SPLITS:
            if int(np.sum(_unicode(self.split) == split)) < 2:
                raise ProtocolError(f"domain {self.name}.{split} needs at least two samples")
        split_values = _unicode(self.split)
        for identity_name, identity_values in (
            ("trajectory_id", self.trajectory_ids),
            ("snapshot_id", self.snapshot_ids),
        ):
            assignments: dict[str, set[str]] = {}
            for identity, split in zip(_unicode(identity_values), split_values):
                assignments.setdefault(str(identity), set()).add(str(split))
            leaks = {identity: sorted(values) for identity, values in assignments.items() if len(values) > 1}
            if leaks:
                raise ProtocolError(
                    f"domain {self.name} leaks {identity_name} across splits: {leaks}"
                )
        if not np.isfinite(np.asarray(self.phase, dtype=np.float64)).all():
            raise ProtocolError(f"domain {self.name} has non-finite phase")
        if int(self.window_steps) <= 0 or not self.coordinate_frame:
            raise ProtocolError("window_steps and coordinate_frame must be explicit")
        covered: set[int] = set()
        for group, bounds in self.feature_groups.items():
            if not group or len(bounds) != 2:
                raise ProtocolError("feature group names/bounds are invalid")
            start, stop = (int(bounds[0]), int(bounds[1]))
            if not 0 <= start < stop <= features.shape[1]:
                raise ProtocolError(f"invalid group {group!r}: {(start, stop)}")
            indices = set(range(start, stop))
            if indices & covered:
                raise ProtocolError("feature groups overlap")
            covered |= indices
        if covered != set(range(features.shape[1])):
            raise ProtocolError("feature groups do not partition the feature vector")
        if primary:
            require_overlap_eligible(_unicode(self.collector_mode).tolist())

    @property
    def feature_schema_sha256(self) -> str:
        return canonical_sha256(
            {
                "width": int(self.features.shape[1]),
                "groups": {name: list(bounds) for name, bounds in self.feature_groups.items()},
                "window_steps": int(self.window_steps),
                "coordinate_frame": self.coordinate_frame,
            }
        )

    def split_features(self, split: str) -> np.ndarray:
        if split not in SPLITS:
            raise ValueError(split)
        return np.asarray(self.features, dtype=np.float64)[_unicode(self.split) == split]

    def select(self, mask: np.ndarray) -> "DomainBundle":
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (self.features.shape[0],):
            raise ValueError("domain selection mask is not aligned")
        return DomainBundle(
            name=self.name,
            family=self.family,
            features=self.features[mask],
            split=self.split[mask],
            sample_ids=self.sample_ids[mask],
            trajectory_ids=self.trajectory_ids[mask],
            snapshot_ids=self.snapshot_ids[mask],
            checkpoint_lineage_ids=self.checkpoint_lineage_ids[mask],
            phase=self.phase[mask],
            contact_mode=self.contact_mode[mask],
            failure=self.failure[mask],
            collector_mode=self.collector_mode[mask],
            feature_groups=self.feature_groups,
            window_steps=self.window_steps,
            coordinate_frame=self.coordinate_frame,
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class GapLayerBundle:
    layers: Mapping[str, np.ndarray]
    sample_ids: np.ndarray
    phase: np.ndarray
    scale: np.ndarray
    feature_groups: Mapping[str, tuple[int, int]]
    metadata: Mapping[str, Any]

    def validate(self) -> None:
        if tuple(self.layers) != GAP_LAYER_NAMES:
            raise ProtocolError(f"gap layers must be ordered exactly as {GAP_LAYER_NAMES}")
        shapes = {np.asarray(value).shape for value in self.layers.values()}
        if len(shapes) != 1:
            raise ProtocolError("gap layers are not sample/feature aligned")
        shape = next(iter(shapes))
        if len(shape) != 2 or shape[0] < 2 or shape[1] < 1:
            raise ProtocolError("gap layers must have shape [N,D]")
        if any(not np.isfinite(np.asarray(value, dtype=np.float64)).all() for value in self.layers.values()):
            raise ProtocolError("gap layer contains NaN or Inf")
        if np.asarray(self.sample_ids).shape != (shape[0],) or np.asarray(self.phase).shape != (shape[0],):
            raise ProtocolError("gap identifiers/phases are not aligned")
        if len(set(_unicode(self.sample_ids).tolist())) != shape[0]:
            raise ProtocolError("gap sample ids are not unique")
        if not np.isfinite(np.asarray(self.phase, dtype=np.float64)).all():
            raise ProtocolError("gap phases are non-finite")
        scale = np.asarray(self.scale, dtype=np.float64)
        if scale.shape != (shape[1],) or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ProtocolError("gap scale must be finite positive [D]")
        covered: set[int] = set()
        for name, bounds in self.feature_groups.items():
            start, stop = int(bounds[0]), int(bounds[1])
            if not name or not 0 <= start < stop <= shape[1]:
                raise ProtocolError(f"invalid gap feature group {name!r}")
            indices = set(range(start, stop))
            if covered & indices:
                raise ProtocolError("gap feature groups overlap")
            covered |= indices
        if covered != set(range(shape[1])):
            raise ProtocolError("gap feature groups do not cover the vector")
        if self.metadata.get("alignment") != "same_phase_conditioned":
            raise ProtocolError("gap bundle lacks same-phase alignment evidence")
        if shape[1] != AMP_FRAME_DIM:
            raise ProtocolError("four-layer gap must use the exact raw 239-D AMP frame")
        from .imitation_6901 import imitation_contract_metadata

        if self.metadata.get("imitation_contract") != imitation_contract_metadata():
            raise ProtocolError("four-layer gap imitation contract/hash is not exact")
        if self.metadata.get("real_physx_replay_verified") is not True:
            raise ProtocolError("four-layer gap lacks real PhysX replay verification")


def load_gap_layer_bundle(path: str | Path) -> GapLayerBundle:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DependencyUnavailable(f"four-layer reference/readback bundle is missing: {source}")
    try:
        with np.load(source, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))
            bundle = GapLayerBundle(
                layers={name: np.asarray(data[name], dtype=np.float64) for name in GAP_LAYER_NAMES},
                sample_ids=np.asarray(data["sample_ids"]),
                phase=np.asarray(data["phase"], dtype=np.float64),
                scale=np.asarray(data["scale"], dtype=np.float64),
                feature_groups={
                    str(name): (int(bounds[0]), int(bounds[1]))
                    for name, bounds in metadata["feature_groups"].items()
                },
                metadata={key: value for key, value in metadata.items() if key != "feature_groups"},
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read four-layer gap bundle {source}: {exc}") from exc
    bundle.validate()
    return bundle


def save_gap_layer_bundle(path: str | Path, bundle: GapLayerBundle) -> Path:
    bundle.validate()
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        **dict(bundle.metadata),
        "feature_groups": {name: list(bounds) for name, bounds in bundle.feature_groups.items()},
    }
    with target.open("xb") as handle:
        np.savez_compressed(
            handle,
            **{name: np.asarray(bundle.layers[name], dtype=np.float32) for name in GAP_LAYER_NAMES},
            sample_ids=_unicode(bundle.sample_ids),
            phase=np.asarray(bundle.phase, dtype=np.float64),
            scale=np.asarray(bundle.scale, dtype=np.float64),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
        )
    return target


def save_domain_bundle(path: str | Path, bundle: DomainBundle) -> Path:
    bundle.validate(primary=True)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        **dict(bundle.metadata),
        "name": bundle.name,
        "family": bundle.family,
        "feature_groups": {name: list(bounds) for name, bounds in bundle.feature_groups.items()},
        "window_steps": int(bundle.window_steps),
        "coordinate_frame": bundle.coordinate_frame,
        "feature_schema_sha256": bundle.feature_schema_sha256,
    }
    with target.open("xb") as handle:
        np.savez_compressed(
            handle,
            features=np.asarray(bundle.features, dtype=np.float32),
            split=_unicode(bundle.split),
            sample_ids=_unicode(bundle.sample_ids),
            trajectory_ids=_unicode(bundle.trajectory_ids),
            snapshot_ids=_unicode(bundle.snapshot_ids),
            checkpoint_lineage_ids=_unicode(bundle.checkpoint_lineage_ids),
            phase=np.asarray(bundle.phase, dtype=np.float64),
            contact_mode=_unicode(bundle.contact_mode),
            failure=np.asarray(bundle.failure, dtype=bool),
            collector_mode=_unicode(bundle.collector_mode),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
        )
    return target


def load_domain_bundle(path: str | Path, *, primary: bool = True) -> DomainBundle:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DependencyUnavailable(f"domain bundle is missing: {source}")
    try:
        with np.load(source, allow_pickle=False) as data:
            required = {
                "features", "split", "sample_ids", "phase", "contact_mode",
                "failure", "collector_mode", "metadata_json", "trajectory_ids",
                "snapshot_ids", "checkpoint_lineage_ids",
            }
            missing = required - set(data.files)
            if missing:
                raise ProtocolError(f"domain bundle {source} lacks {sorted(missing)}")
            metadata = json.loads(str(data["metadata_json"].item()))
            bundle = DomainBundle(
                name=str(metadata["name"]),
                family=str(metadata["family"]),
                features=np.asarray(data["features"], dtype=np.float64),
                split=np.asarray(data["split"]),
                sample_ids=np.asarray(data["sample_ids"]),
                trajectory_ids=np.asarray(data["trajectory_ids"]),
                snapshot_ids=np.asarray(data["snapshot_ids"]),
                checkpoint_lineage_ids=np.asarray(data["checkpoint_lineage_ids"]),
                phase=np.asarray(data["phase"], dtype=np.float64),
                contact_mode=np.asarray(data["contact_mode"]),
                failure=np.asarray(data["failure"], dtype=bool),
                collector_mode=np.asarray(data["collector_mode"]),
                feature_groups={
                    str(name): (int(bounds[0]), int(bounds[1]))
                    for name, bounds in metadata["feature_groups"].items()
                },
                window_steps=int(metadata["window_steps"]),
                coordinate_frame=str(metadata["coordinate_frame"]),
                metadata={key: value for key, value in metadata.items() if key not in {
                    "name", "family", "feature_groups", "window_steps", "coordinate_frame"
                }},
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        if isinstance(exc, ProtocolError):
            raise
        raise ProtocolError(f"cannot read domain bundle {source}: {exc}") from exc
    bundle.validate(primary=primary)
    expected = bundle.metadata.get("feature_schema_sha256")
    if expected is not None and str(expected) != bundle.feature_schema_sha256:
        raise ProtocolError(f"domain bundle feature schema hash mismatch: {source}")
    return bundle


def domain_split(bundle: DomainBundle):
    from .domain_triangle import DomainSplit

    bundle.validate(primary=True)
    return DomainSplit(**{split: bundle.split_features(split) for split in SPLITS})


def amp_domain_feature_contract(bundle: DomainBundle) -> dict[str, Any]:
    """Return the exact cross-stage contract for a standard AMP critic input."""

    from .imitation_6901 import imitation_contract_metadata

    bundle.validate(primary=True)
    expected_imitation = imitation_contract_metadata()
    expected = {
        "amp_source_commit": AMP_SOURCE_COMMIT,
        "amp_representation": "mimickit_g1_chronological_window",
        "amp_frame_dim": AMP_FRAME_DIM,
        "amp_window_steps": AMP_WINDOW_STEPS,
        "amp_root_xy_anchor": "newest_frame",
    }
    mismatches = {
        key: {"expected": value, "actual": bundle.metadata.get(key)}
        for key, value in expected.items()
        if bundle.metadata.get(key) != value
    }
    if (
        mismatches
        or bundle.features.shape[1] != AMP_FRAME_DIM * AMP_WINDOW_STEPS
        or bundle.window_steps != AMP_WINDOW_STEPS
        or bundle.coordinate_frame != "mimickit_g1_amp_window_v1"
        or dict(bundle.feature_groups) != amp_window_feature_groups()
        or bundle.metadata.get("imitation_contract") != expected_imitation
    ):
        raise ProtocolError(
            f"domain {bundle.name} violates the exact standard-AMP feature contract: {mismatches}"
        )
    return {
        "amp_source_commit": AMP_SOURCE_COMMIT,
        "amp_representation": expected["amp_representation"],
        "input_dim": AMP_FRAME_DIM * AMP_WINDOW_STEPS,
        "frame_dim": AMP_FRAME_DIM,
        "window_steps": AMP_WINDOW_STEPS,
        "root_xy_anchor": expected["amp_root_xy_anchor"],
        "coordinate_frame": bundle.coordinate_frame,
        "feature_schema_sha256": bundle.feature_schema_sha256,
        "imitation_contract": expected_imitation,
    }


def load_domain_index(path: str | Path) -> tuple[dict[str, Any], dict[str, DomainBundle]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DependencyUnavailable(f"domain triangle index is missing: {source}")
    payload = read_json(source)
    if not isinstance(payload, Mapping) or payload.get("status") != "PASS":
        raise DependencyUnavailable(f"domain triangle index is not PASS: {source}")
    records = payload.get("domains")
    if not isinstance(records, list) or not records:
        raise ProtocolError("domain triangle index has no domains")
    bundles: dict[str, DomainBundle] = {}
    for record in records:
        if not isinstance(record, Mapping) or "name" not in record or "bundle_path" not in record:
            raise ProtocolError("domain triangle index record is incomplete")
        path_value = Path(str(record["bundle_path"])).expanduser()
        bundle_path = path_value if path_value.is_absolute() else source.parent / path_value
        bundle = load_domain_bundle(bundle_path, primary=True)
        if bundle.name != str(record["name"]):
            raise ProtocolError("domain index name disagrees with bundle")
        expected_hash = record.get("bundle_sha256")
        if expected_hash and sha256_file(bundle_path) != str(expected_hash):
            raise ProtocolError(f"domain bundle hash mismatch: {bundle_path}")
        if bundle.name in bundles:
            raise ProtocolError(f"duplicate domain {bundle.name!r}")
        bundles[bundle.name] = bundle
    validate_domain_triangle(bundles)
    return dict(payload), bundles


def validate_domain_triangle(bundles: Mapping[str, DomainBundle]) -> None:
    names = set(bundles)
    required = set(REQUIRED_DOMAIN_IDENTITIES)
    missing = required - names
    forbidden = names - required
    if missing:
        raise DependencyUnavailable(f"domain triangle awaits frozen domains: {sorted(missing)}")
    if forbidden:
        raise ProtocolError(f"domain triangle contains non-frozen/retired domains: {sorted(forbidden)}")
    schema_hashes = {bundle.feature_schema_sha256 for bundle in bundles.values()}
    if len(schema_hashes) != 1:
        raise ProtocolError("domain bundles do not share one feature/window coordinate schema")
    global_snapshot_splits: dict[str, str] = {}
    global_trajectory_splits: dict[str, str] = {}
    from .imitation_6901 import imitation_contract_metadata

    exact_contract = imitation_contract_metadata()
    exact_groups = amp_window_feature_groups()
    for bundle in bundles.values():
        bundle.validate(primary=True)
        if (
            bundle.features.shape[1] != AMP_FRAME_DIM * AMP_WINDOW_STEPS
            or bundle.window_steps != AMP_WINDOW_STEPS
            or bundle.coordinate_frame != "mimickit_g1_amp_window_v1"
            or dict(bundle.feature_groups) != exact_groups
            or bundle.metadata.get("amp_source_commit") != AMP_SOURCE_COMMIT
            or bundle.metadata.get("imitation_contract") != exact_contract
        ):
            raise ProtocolError(
                f"domain {bundle.name} is not the exact chronological 10x239 AMP representation"
            )
        if bundle.metadata.get("derived_from_unified_collector") is not True:
            raise ProtocolError(f"domain {bundle.name} was not derived from the unified collector")
        for field in ("rollout_index_sha256", "split_audit_sha256"):
            digest = str(bundle.metadata.get(field, ""))
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ProtocolError(f"domain {bundle.name} lacks valid {field}")
        shards = bundle.metadata.get("source_shards")
        if not isinstance(shards, list) or not shards:
            raise ProtocolError(f"domain {bundle.name} lacks source-shard provenance")
        if any(
            not isinstance(record, Mapping)
            or len(str(record.get("sha256", ""))) != 64
            for record in shards
        ):
            raise ProtocolError(f"domain {bundle.name} source-shard provenance is malformed")
        for field, values, assignments in (
            ("snapshot_id", bundle.snapshot_ids, global_snapshot_splits),
            ("trajectory_id", bundle.trajectory_ids, global_trajectory_splits),
        ):
            for identity, split in zip(_unicode(values), _unicode(bundle.split)):
                previous = assignments.setdefault(str(identity), str(split))
                if previous != str(split):
                    raise ProtocolError(
                        f"cross-domain {field} {identity!r} leaks across {previous}/{split}"
                    )


def build_domain_index(
    catalog_path: str | Path,
    *,
    derived_output_dir: str | Path | None = None,
    split_audit_path: str | Path | None = None,
    split_source_index: str | Path | None = None,
) -> dict[str, Any]:
    source = Path(catalog_path).expanduser().resolve()
    if not source.is_file():
        raise DependencyUnavailable(f"domain catalog is missing: {source}")
    catalog = read_json(source)
    entries = catalog.get("domains") if isinstance(catalog, Mapping) else None
    if not isinstance(entries, list) or not entries:
        raise ProtocolError("domain catalog must contain a non-empty domains list")
    bundles: dict[str, DomainBundle] = {}
    records: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ProtocolError("domain catalog entry must be a mapping")
        if "bundle_path" in entry:
            raw = Path(str(entry["bundle_path"])).expanduser()
            bundle_path = (raw if raw.is_absolute() else source.parent / raw).resolve()
            bundle = load_domain_bundle(bundle_path, primary=True)
        elif "rollout_index" in entry:
            if derived_output_dir is None or split_audit_path is None or split_source_index is None:
                raise DependencyUnavailable(
                    "rollout-derived domain needs output, split audit, and split-source index"
                )
            name = str(entry.get("name", ""))
            if not name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in name):
                raise ProtocolError(f"unsafe domain name {name!r}")
            bundle = build_amp_domain_bundle_from_rollouts(
                entry,
                catalog_dir=source.parent,
                split_audit_path=Path(split_audit_path).expanduser().resolve(),
                split_source_index=Path(split_source_index).expanduser().resolve(),
            )
            bundle_path = Path(derived_output_dir).expanduser().resolve() / f"{name}.npz"
            save_domain_bundle(bundle_path, bundle)
        else:
            raise ProtocolError("domain catalog entry needs bundle_path or rollout_index")
        if entry.get("name") not in (None, bundle.name):
            raise ProtocolError("domain catalog name disagrees with bundle")
        if bundle.name in bundles:
            raise ProtocolError(f"duplicate domain {bundle.name!r}")
        bundles[bundle.name] = bundle
        records.append(
            {
                "name": bundle.name,
                "family": bundle.family,
                "bundle_path": str(bundle_path),
                "bundle_sha256": sha256_file(bundle_path),
                "sample_count": int(bundle.features.shape[0]),
                "feature_width": int(bundle.features.shape[1]),
                "feature_schema_sha256": bundle.feature_schema_sha256,
                "window_steps": int(bundle.window_steps),
                "coordinate_frame": bundle.coordinate_frame,
                "split_counts": {
                    split: int(np.sum(_unicode(bundle.split) == split)) for split in SPLITS
                },
                "collector_modes": sorted(set(_unicode(bundle.collector_mode).tolist())),
            }
        )
    validate_domain_triangle(bundles)
    return {
        "schema_version": "1.0.0",
        "status": "PASS",
        "catalog_path": str(source),
        "catalog_sha256": sha256_file(source),
        "domains": records,
        "domain_families": sorted({bundle.family for bundle in bundles.values()}),
        "feature_schema_sha256": next(iter({bundle.feature_schema_sha256 for bundle in bundles.values()})),
        "scope": "K/T/A/B domain windows in a shared target-coordinate representation",
        "native_stochastic_excluded": True,
    }


__all__ = [
    "AMP_FRAME_DIM", "AMP_SOURCE_COMMIT", "AMP_WINDOW_STEPS", "DomainBundle",
    "GAP_LAYER_NAMES", "GapLayerBundle", "REQUIRED_DOMAIN_FAMILIES",
    "REQUIRED_DOMAIN_IDENTITIES", "SPLITS",
    "amp_domain_feature_contract", "amp_window_feature_groups",
    "build_amp_domain_bundle_from_rollouts",
    "build_domain_index", "chronological_amp_windows", "domain_split", "load_domain_bundle",
    "load_domain_index", "load_gap_layer_bundle", "save_domain_bundle",
    "save_gap_layer_bundle", "validate_domain_triangle",
]
