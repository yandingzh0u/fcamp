"""Shared contracts and pure analysis for discovery-suite Stage 4.

The module intentionally separates three kinds of evidence:

* the uncombined physical outcome panel;
* genuine commit-6901 offline AMP rewards produced by Stage 3; and
* exact same-snapshot PhysX branch records.

No source-classifier probability and no manually combined outcome value is
accepted as a reward.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from scipy.stats import spearmanr

from components.imitation.motion_features import canonicalize_imitation_window

from .imitation_6901 import IMITATION_FRAME_DIM, imitation_contract_metadata
from .canonical_collection import load_rollout_index, load_rollout_trajectory
from .domain_data import chronological_amp_windows
from .manifest import (
    DependencyUnavailable,
    ProtocolError,
    canonical_sha256,
    read_json,
    sha256_file,
)
from .offline_amp import (
    OFFLINE_AMP_ARTIFACT_SCHEMA,
    OfflineAMPFit,
    SOURCE_COMMIT,
    directed_offline_amp_protocol,
    load_offline_amp_fit,
)
from .quality_panel import OutcomeMetric, pareto_preference
from .reward_validity import reward_seed_agreement, strict_pairwise_accuracy


STAGE3_INDEX_SCHEMA = "largebox_diag35_directed_offline_amp_index_v1"
BRANCH_BANK_SCHEMA = "largebox_same_snapshot_branch_bank_v1"
BRANCH_ROW_SCHEMA = "largebox_same_snapshot_branch_rows_v1"
CEM_BANK_SCHEMA = "largebox_cem_exploitability_v1"
INTERVENTION_BANK_SCHEMA = "largebox_reward_intervention_bank_v1"
BLIND_VIDEO_SOURCE_SCHEMA = "largebox_blind_video_sources_v1"
PRIMARY_REWARD_FAMILIES = ("K", "T_u500")
AMP_WINDOW_STEPS = 10
AMP_WINDOW_DIM = AMP_WINDOW_STEPS * IMITATION_FRAME_DIM


@dataclass(frozen=True, slots=True)
class RewardValidityProtocol:
    primary_metrics: tuple[OutcomeMetric, ...]
    secondary_metrics: tuple[str, ...]
    branch_horizons: tuple[int, ...]
    checkpoint_updates: tuple[int, ...]
    interpolation_alphas: tuple[float, ...]
    perturbation_scales: tuple[float, ...]
    minimum_distinct_branch_sources: int
    cem: Mapping[str, Any]
    interventions: Mapping[str, Any]
    blind_video: Mapping[str, Any]

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "RewardValidityProtocol":
        raw = spec.get("analysis_protocols", {}).get("reward_validity")
        if not isinstance(raw, Mapping):
            raise ProtocolError("analysis_protocols.reward_validity is not frozen")
        raw_metrics = raw.get("primary_pareto_outcomes")
        if not isinstance(raw_metrics, list) or not raw_metrics:
            raise ProtocolError("primary_pareto_outcomes is absent")
        metrics = tuple(
            OutcomeMetric(
                str(item["name"]),
                bool(item["higher_is_better"]),
                float(item["tolerance"]),
            )
            for item in raw_metrics
        )
        secondary = tuple(str(value) for value in raw.get("secondary_unweighted_panel", ()))
        horizons = tuple(int(value) for value in raw.get("branch_horizons", ()))
        updates = tuple(int(value) for value in raw.get("checkpoint_branch_updates", ()))
        alphas = tuple(float(value) for value in raw.get("action_mean_interpolation_alphas", ()))
        perturbations = tuple(float(value) for value in raw.get("action_perturbation_scales", ()))
        minimum = int(raw.get("minimum_distinct_branch_sources", 0))
        if horizons != (1, 5, 10, 25, 50):
            raise ProtocolError("Stage-4 branch horizons changed from [1,5,10,25,50]")
        if tuple(sorted(set(updates))) != updates or len(updates) < 2:
            raise ProtocolError("checkpoint branch updates must be sorted and distinct")
        if any(not 0.0 < value < 1.0 for value in alphas):
            raise ProtocolError("action interpolation alphas must lie strictly in (0,1)")
        if any(not np.isfinite(value) or value <= 0.0 for value in perturbations):
            raise ProtocolError("action perturbation scales must be finite and positive")
        if minimum < 3:
            raise ProtocolError("same-state branching requires at least three source categories")
        cem = raw.get("cem")
        interventions = raw.get("interventions")
        blind_video = raw.get("blind_video")
        if not all(isinstance(value, Mapping) for value in (cem, interventions, blind_video)):
            raise ProtocolError("CEM/intervention/blind-video protocols must be mappings")
        if int(cem["horizon"]) != 10 or int(cem["population"]) != 256:
            raise ProtocolError("CEM horizon/population differs from the frozen protocol")
        if int(cem["elite_count"]) >= int(cem["population"]):
            raise ProtocolError("CEM elite count must be smaller than the population")
        return cls(
            primary_metrics=metrics,
            secondary_metrics=secondary,
            branch_horizons=horizons,
            checkpoint_updates=updates,
            interpolation_alphas=alphas,
            perturbation_scales=perturbations,
            minimum_distinct_branch_sources=minimum,
            cem=dict(cem),
            interventions=dict(interventions),
            blind_video=dict(blind_video),
        )


@dataclass(slots=True)
class Stage3Critics:
    by_positive: dict[str, list[OfflineAMPFit]]
    records_by_positive: dict[str, list[dict[str, Any]]]
    protocol_sha256: str
    feature_contract_sha256: str
    feature_schema_sha256: str
    index_path: Path
    index_sha256: str
    all_records: tuple[Mapping[str, Any], ...]

    @property
    def seeds(self) -> tuple[int, ...]:
        first = self.by_positive[PRIMARY_REWARD_FAMILIES[0]]
        return tuple(fit.seed for fit in first)


def load_stage3_critics(
    output_dir: str | Path,
    spec: Mapping[str, Any],
    *,
    device: str = "cpu",
) -> Stage3Critics:
    """Load only persisted Stage-3 standard-AMP artifacts, never a probe model."""

    root = Path(output_dir).expanduser().resolve()
    status_path = root / "ratio_reliability.json"
    if not status_path.is_file():
        raise DependencyUnavailable("diag_35 directed reward-reliability status is missing")
    status = read_json(status_path)
    if status.get("status") != "PASS":
        raise DependencyUnavailable("diag_35 did not produce valid directed offline AMP critics")
    evidence = status.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ProtocolError("diag_36 status has no evidence mapping")
    index_path = Path(str(evidence.get("offline_amp_artifact_index", ""))).expanduser().resolve()
    expected_index_hash = str(evidence.get("offline_amp_artifact_index_sha256", ""))
    if not index_path.is_file() or sha256_file(index_path) != expected_index_hash:
        raise ProtocolError("diag_35 offline AMP artifact index is missing or changed")
    index = read_json(index_path)
    if not isinstance(index, Mapping):
        raise ProtocolError("offline AMP artifact index is not a mapping")
    if (
        index.get("status") != "PASS"
        or index.get("artifact_schema") != STAGE3_INDEX_SCHEMA
        or index.get("source_commit") != SOURCE_COMMIT
    ):
        raise ProtocolError("offline reward index is not the exact standard AMP family")
    frozen_protocol = spec.get("analysis_protocols", {}).get("offline_amp_critic")
    if not isinstance(frozen_protocol, Mapping):
        raise ProtocolError("offline_amp_critic protocol is absent from the suite spec")
    protocol_sha256 = canonical_sha256(dict(frozen_protocol))
    if (
        index.get("base_protocol_sha256") != protocol_sha256
        or index.get("base_protocol") != dict(frozen_protocol)
    ):
        raise ProtocolError("offline AMP artifact base protocol differs from the frozen spec")
    feature_contract = index.get("feature_contract")
    if not isinstance(feature_contract, Mapping):
        raise ProtocolError("offline AMP artifact index lacks the feature contract")
    feature_contract_sha256 = canonical_sha256(dict(feature_contract))
    if index.get("feature_contract_sha256") != feature_contract_sha256:
        raise ProtocolError("offline AMP feature-contract hash is inconsistent")
    exact = {
        "amp_source_commit": SOURCE_COMMIT,
        "amp_representation": "mimickit_g1_chronological_window",
        "input_dim": AMP_WINDOW_DIM,
        "frame_dim": IMITATION_FRAME_DIM,
        "window_steps": AMP_WINDOW_STEPS,
        "root_xy_anchor": "newest_frame",
        "imitation_contract": imitation_contract_metadata(),
    }
    mismatches = {
        key: {"expected": value, "actual": feature_contract.get(key)}
        for key, value in exact.items()
        if feature_contract.get(key) != value
    }
    if mismatches:
        raise ProtocolError(f"offline AMP feature contract is not exact: {mismatches}")
    feature_schema_sha256 = str(feature_contract.get("feature_schema_sha256", ""))
    if len(feature_schema_sha256) != 64:
        raise ProtocolError("offline AMP feature schema SHA256 is malformed")
    raw_models = index.get("models")
    if not isinstance(raw_models, list):
        raise ProtocolError("offline AMP artifact index has no model records")
    expected_seeds = tuple(int(value) for value in frozen_protocol.get("seeds", ()))
    if len(expected_seeds) != 5:
        raise ProtocolError("reward validity requires five frozen AMP seeds")
    actual_keys = {
        (
            str(record.get("source_negative", "")),
            str(record.get("destination_positive", "")),
            int(record.get("seed", -1)),
        )
        for record in raw_models
        if isinstance(record, Mapping)
    }
    if len(actual_keys) != len(raw_models):
        raise ProtocolError("diag_35 critic index contains duplicate edge/seed identities")
    # A_mix/FCAMP is explicitly quarantined by the user.  It may be described
    # in legacy audit metadata, but Stage 4 never loads it and never requires
    # its edge artifacts.  The primary reward contract needs only these two
    # directed edges; other non-A_mix edges remain optional probes.
    required_keys = {
        ("A_amp", positive, seed)
        for positive in PRIMARY_REWARD_FAMILIES
        for seed in expected_seeds
    }
    missing_required = required_keys - actual_keys
    if missing_required:
        raise DependencyUnavailable(
            f"diag35 lacks primary non-A_mix reward critics: {sorted(missing_required)}"
        )
    # Hash every catalogued artifact.  Merely trusting paths for non-primary
    # probe edges would allow a later source-classifier file to be substituted.
    for raw_record in raw_models:
        assert isinstance(raw_record, Mapping)
        source_name = str(raw_record.get("source_negative", ""))
        destination_name = str(raw_record.get("destination_positive", ""))
        if "A_mix" in (source_name, destination_name):
            # Do not even hash/load quarantined models: their existence is not
            # formal Stage-4 evidence and must not become a dependency.
            continue
        path = Path(str(raw_record.get("path", ""))).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != str(raw_record.get("sha256", "")):
            raise ProtocolError(f"directed AMP critic file/hash changed: {path}")
        edge_protocol = directed_offline_amp_protocol(
            frozen_protocol,
            negative_domain=source_name,
            positive_domain=destination_name,
        )
        edge_hash = canonical_sha256(edge_protocol)
        if (
            raw_record.get("base_protocol_sha256") != protocol_sha256
            or raw_record.get("protocol_sha256") != edge_hash
            or raw_record.get("feature_contract_sha256") != feature_contract_sha256
        ):
            raise ProtocolError("directed AMP critic record contract/hash differs")
    by_positive: dict[str, list[OfflineAMPFit]] = {}
    by_records: dict[str, list[dict[str, Any]]] = {}
    for positive in PRIMARY_REWARD_FAMILIES:
        records = [
            dict(record)
            for record in raw_models
            if isinstance(record, Mapping)
            and str(record.get("destination_positive")) == positive
            and str(record.get("source_negative")) == "A_amp"
        ]
        records.sort(key=lambda item: int(item["seed"]))
        if tuple(int(item["seed"]) for item in records) != expected_seeds:
            raise DependencyUnavailable(
                f"offline AMP {positive}-positive artifacts do not cover all frozen seeds"
            )
        fits: list[OfflineAMPFit] = []
        for record in records:
            path = Path(str(record["path"])).expanduser().resolve()
            if not path.is_file() or sha256_file(path) != str(record.get("sha256")):
                raise ProtocolError(f"offline AMP model file/hash changed: {path}")
            fit, metadata = load_offline_amp_fit(
                path,
                device=device,
                expected_protocol_sha256=str(record["protocol_sha256"]),
                expected_feature_contract_sha256=feature_contract_sha256,
            )
            if metadata.get("artifact_schema") != OFFLINE_AMP_ARTIFACT_SCHEMA:
                raise ProtocolError("reward model is not a persisted offline AMP critic")
            provenance = metadata.get("training_provenance")
            if not isinstance(provenance, Mapping) or (
                provenance.get("source_negative") != "A_amp"
                or provenance.get("destination_positive") != positive
            ):
                raise ProtocolError("offline AMP critic training provenance is directed incorrectly")
            fits.append(fit)
        by_positive[positive] = fits
        by_records[positive] = records
    return Stage3Critics(
        by_positive=by_positive,
        records_by_positive=by_records,
        protocol_sha256=protocol_sha256,
        feature_contract_sha256=feature_contract_sha256,
        feature_schema_sha256=feature_schema_sha256,
        index_path=index_path,
        index_sha256=expected_index_hash,
        all_records=tuple(
            dict(record)
            for record in raw_models
            if "A_mix" not in {
                str(record.get("source_negative", "")),
                str(record.get("destination_positive", "")),
            }
        ),
    )


def flatten_amp_windows(raw: torch.Tensor) -> torch.Tensor:
    if raw.ndim != 3 or tuple(raw.shape[1:]) != (
        AMP_WINDOW_STEPS,
        IMITATION_FRAME_DIM,
    ):
        raise ProtocolError(
            f"raw AMP window must be [B,{AMP_WINDOW_STEPS},{IMITATION_FRAME_DIM}]"
        )
    if raw.dtype != torch.float32 or not bool(torch.isfinite(raw).all()):
        raise ProtocolError("raw AMP window must be finite float32")
    return canonicalize_imitation_window(raw).reshape(raw.shape[0], AMP_WINDOW_DIM)


@torch.no_grad()
def fit_reward_tensor(fit: OfflineAMPFit, flat: torch.Tensor) -> torch.Tensor:
    if flat.ndim != 2 or flat.shape[1] != AMP_WINDOW_DIM:
        raise ProtocolError(f"AMP score input must be [N,{AMP_WINDOW_DIM}]")
    values = flat.to(device=fit.mean.device, dtype=torch.float32)
    normalized = torch.clamp(
        (values - fit.mean) / torch.sqrt(torch.clamp(fit.variance, min=1.0e-8)),
        -fit.clip,
        fit.clip,
    )
    fit.model.eval()
    logits = fit.model(normalized)
    reward = fit.reward_scale * torch.nn.functional.softplus(logits)
    maximum = -fit.reward_scale * np.log(fit.reward_epsilon)
    reward = torch.clamp(reward, max=float(maximum))
    if not bool(torch.isfinite(reward).all()):
        raise ProtocolError("offline AMP reward contains NaN or Inf")
    return reward


def score_flat_bank(
    critics: Stage3Critics,
    features: np.ndarray,
) -> dict[str, np.ndarray]:
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != AMP_WINDOW_DIM or not np.isfinite(values).all():
        raise ProtocolError("trajectory reward bank must be finite [N,2390]")
    result: dict[str, np.ndarray] = {}
    for family, fits in critics.by_positive.items():
        result[family] = np.stack([fit.rewards(values) for fit in fits], axis=0)
    return result


def canonical_trajectory_window_bank(
    index_path: str | Path,
    panel_rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Build exact alive 10-frame windows and a trajectory-owner vector.

    A trajectory ending before ten real frames has no valid Stage-3 score and
    is reported as unscored.  It is never padded with reference data here.
    """

    source = Path(index_path).expanduser().resolve()
    index = {str(row["sample_id"]): row for row in load_rollout_index(source)}
    cache: dict[str, dict[str, Any]] = {}
    parts: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    unscored: list[int] = []
    for panel_index, panel in enumerate(panel_rows):
        sample_id = str(panel["sample_id"])
        row = index.get(sample_id)
        if row is None:
            raise ProtocolError(f"quality row {sample_id!r} is absent from canonical index")
        tree = load_rollout_trajectory(source, row, cache=cache)
        frame = tree.get("imitation", {}).get("agent_physx_raw_frame")
        done = tree.get("trajectory", {}).get("done")
        if not torch.is_tensor(frame) or not torch.is_tensor(done):
            raise ProtocolError("canonical trajectory lacks exact agent AMP frames/done")
        try:
            windows, _ = chronological_amp_windows(
                frame.detach().cpu().numpy(), done=done.detach().cpu().numpy()
            )
        except DependencyUnavailable:
            unscored.append(panel_index)
            continue
        parts.append(windows)
        owners.append(np.full(windows.shape[0], panel_index, dtype=np.int64))
    if not parts:
        raise DependencyUnavailable("quality panel contains no valid alive 10-frame AMP windows")
    features = np.concatenate(parts, axis=0)
    owner = np.concatenate(owners)
    if not np.isfinite(features).all() or features.shape[1] != AMP_WINDOW_DIM:
        raise ProtocolError("canonical trajectory window bank is invalid")
    return features, owner, unscored


def aggregate_window_scores(
    scores: np.ndarray,
    owners: np.ndarray,
    *,
    row_count: int,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64).reshape(-1)
    owner = np.asarray(owners, dtype=np.int64).reshape(-1)
    if values.shape != owner.shape or not np.isfinite(values).all():
        raise ProtocolError("window rewards and trajectory owners are not aligned")
    if owner.size == 0 or np.any(owner < 0) or np.any(owner >= int(row_count)):
        raise ProtocolError("trajectory owner index is invalid")
    counts = np.bincount(owner, minlength=int(row_count))
    sums = np.bincount(owner, weights=values, minlength=int(row_count))
    result = np.full(int(row_count), np.nan, dtype=np.float64)
    valid = counts > 0
    result[valid] = sums[valid] / counts[valid]
    return result


def score_canonical_trajectories_streaming(
    index_path: str | Path,
    panel_rows: Sequence[Mapping[str, Any]],
    critics: Stage3Critics,
    *,
    batch_trajectories: int = 8,
) -> tuple[dict[str, np.ndarray], list[int]]:
    """Score long canonical rollouts without materializing the full 2.9 GB bank."""

    count = len(panel_rows)
    if count < 1 or batch_trajectories < 1:
        raise ProtocolError("streaming trajectory score budget is invalid")
    output = {
        family: np.full((len(fits), count), np.nan, dtype=np.float64)
        for family, fits in critics.by_positive.items()
    }
    unscored: list[int] = []
    for start in range(0, count, int(batch_trajectories)):
        stop = min(count, start + int(batch_trajectories))
        batch = panel_rows[start:stop]
        try:
            features, owners, local_unscored = canonical_trajectory_window_bank(
                index_path, batch
            )
        except DependencyUnavailable as exc:
            if "no valid alive" not in str(exc):
                raise
            unscored.extend(range(start, stop))
            continue
        unscored.extend(start + index for index in local_unscored)
        for family, fits in critics.by_positive.items():
            for fit_index, fit in enumerate(fits):
                window_scores = fit.rewards(features, batch_size=2048)
                values = aggregate_window_scores(
                    window_scores, owners, row_count=len(batch)
                )
                output[family][fit_index, start:stop] = values
    expected_missing = set(unscored)
    for family, values in output.items():
        actual_missing = set(np.where(~np.isfinite(values).all(axis=0))[0].tolist())
        if actual_missing != expected_missing:
            raise ProtocolError(
                f"streaming {family} score missingness is not explained by short trajectories"
            )
    return output, sorted(expected_missing)


def load_directed_critic_edge(
    critics: Stage3Critics,
    *,
    source_negative: str,
    destination_positive: str,
    device: str = "cpu",
) -> list[OfflineAMPFit]:
    """Load one of diag35's preregistered directed edges on demand."""

    records = [
        dict(record)
        for record in critics.all_records
        if str(record.get("source_negative")) == source_negative
        and str(record.get("destination_positive")) == destination_positive
    ]
    records.sort(key=lambda item: int(item["seed"]))
    if tuple(int(record["seed"]) for record in records) != critics.seeds:
        raise DependencyUnavailable(
            f"diag35 edge {source_negative}->{destination_positive} lacks frozen seeds"
        )
    fits: list[OfflineAMPFit] = []
    for record in records:
        fit, metadata = load_offline_amp_fit(
            Path(str(record["path"])).expanduser().resolve(),
            device=device,
            expected_protocol_sha256=str(record["protocol_sha256"]),
            expected_feature_contract_sha256=critics.feature_contract_sha256,
        )
        provenance = metadata.get("training_provenance")
        if not isinstance(provenance, Mapping) or (
            provenance.get("source_negative") != source_negative
            or provenance.get("destination_positive") != destination_positive
        ):
            raise ProtocolError("directed critic payload provenance differs from diag35 index")
        if int(fit.seed) != int(record["seed"]) or fit.source_commit != SOURCE_COMMIT:
            raise ProtocolError("directed critic seed/source commit mismatch")
        fits.append(fit)
    return fits


def _parquet_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    if not rows:
        raise ProtocolError("cannot serialize an empty Stage-4 table")
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - suite dependency
        raise DependencyUnavailable("pandas/pyarrow is required for Stage-4 parquet") from exc
    buffer = io.BytesIO()
    try:
        pd.DataFrame([dict(row) for row in rows]).to_parquet(buffer, index=False)
    except (ImportError, ModuleNotFoundError) as exc:
        raise DependencyUnavailable("pyarrow or fastparquet is required") from exc
    return buffer.getvalue()


def write_parquet_exclusive(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    target = Path(path).expanduser().resolve()
    payload = _parquet_bytes(rows)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    _fsync_parent_directory(target)
    return target


def read_parquet_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DependencyUnavailable(f"Stage-4 parquet is missing: {source}")
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise DependencyUnavailable("pandas/pyarrow is required") from exc
    rows = pd.read_parquet(source).to_dict(orient="records")
    if not rows:
        raise ProtocolError(f"Stage-4 parquet is empty: {source}")
    return [dict(row) for row in rows]


def write_branch_bank(path: str | Path, payload: Mapping[str, Any]) -> Path:
    validate_branch_bank(payload)
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_parent_directory(target)
    return target


def _fsync_parent_directory(path: Path) -> None:
    """Make collector artifacts durable before a safe Isaac hard exit."""

    try:
        descriptor = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def load_branch_bank(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise DependencyUnavailable(f"same-snapshot branch bank is missing: {source}")
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ProtocolError("same-snapshot branch bank is not a mapping")
    result = dict(payload)
    validate_branch_bank(result)
    return result


def validate_branch_bank(payload: Mapping[str, Any]) -> None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("schema") != BRANCH_BANK_SCHEMA:
        raise ProtocolError("same-snapshot branch bank schema is invalid")
    required_true = (
        "real_physx_rollouts",
        "same_snapshot_replay_verified",
        "shared_environment_randomness",
        "demo_seeded_commit6901_history",
    )
    if any(metadata.get(key) is not True for key in required_true):
        raise ProtocolError("same-snapshot branch bank lacks real replay guarantees")
    if metadata.get("source_classifier_used_as_reward") is not False:
        raise ProtocolError("source-classifier scores are forbidden in Stage 4")
    if metadata.get("critic_source_commit") != SOURCE_COMMIT:
        raise ProtocolError("branch bank reward is not from the frozen AMP critic")
    branch_ids = tuple(str(value) for value in metadata.get("branch_ids", ()))
    snapshot_ids = tuple(str(value) for value in metadata.get("snapshot_ids", ()))
    horizons = tuple(int(value) for value in metadata.get("horizons", ()))
    track_names = tuple(str(value) for value in metadata.get("track_body_names", ()))
    if not branch_ids or len(set(branch_ids)) != len(branch_ids):
        raise ProtocolError("branch IDs are empty or duplicated")
    if not snapshot_ids or len(set(snapshot_ids)) != len(snapshot_ids):
        raise ProtocolError("branch snapshot IDs are empty or duplicated")
    if horizons != (1, 5, 10, 25, 50):
        raise ProtocolError("branch bank horizons differ from the frozen protocol")
    endpoint = payload.get("endpoint_windows")
    body = payload.get("body_pos_local")
    root = payload.get("root_pos_local")
    active = payload.get("active")
    phase = payload.get("endpoint_phase")
    contact = payload.get("endpoint_contact_mode")
    if not all(torch.is_tensor(value) for value in (endpoint, body, root, active)):
        raise ProtocolError("branch bank is missing trajectory tensors")
    b, h, n, d = endpoint.shape if endpoint.ndim == 4 else (0, 0, 0, 0)
    if (b, h, n, d) != (len(branch_ids), len(horizons), len(snapshot_ids), AMP_WINDOW_DIM):
        raise ProtocolError("branch endpoint-window shape is invalid")
    if body.ndim != 5 or tuple(body.shape[:3]) != (b, max(horizons), n):
        raise ProtocolError("branch body trajectory shape is invalid")
    if body.shape[3] != len(track_names) or body.shape[4] != 3:
        raise ProtocolError("branch body names/positions are not aligned")
    if tuple(root.shape) != (b, max(horizons), n, 3):
        raise ProtocolError("branch root trajectory shape is invalid")
    if tuple(active.shape) != (b, max(horizons), n) or active.dtype is not torch.bool:
        raise ProtocolError("branch active mask shape/type is invalid")
    if not torch.is_tensor(phase) or tuple(phase.shape) != (b, len(horizons), n):
        raise ProtocolError("branch endpoint phase shape is invalid")
    if not torch.is_tensor(contact) or tuple(contact.shape) != (b, len(horizons), n):
        raise ProtocolError("branch endpoint contact-mode shape is invalid")
    if contact.dtype not in (torch.int32, torch.int64) or bool(
        (contact < 0).any() or (contact > 3).any()
    ):
        raise ProtocolError("branch endpoint contact modes must be integer 0..3")
    for name, value in (
        ("endpoint_windows", endpoint),
        ("body_pos_local", body),
        ("root_pos_local", root),
        ("endpoint_phase", phase),
    ):
        if not bool(torch.isfinite(value).all()):
            raise ProtocolError(f"branch bank {name} contains NaN or Inf")


def validate_branch_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    protocol: RewardValidityProtocol,
    seeds: Sequence[int] | None = None,
) -> None:
    """Validate the tabular half of a replay-verified branch bank.

    The table contains only measured outcomes and persisted-critic scores.  It
    must never contain a hand-combined quality target.
    """

    if not rows:
        raise ProtocolError("same-snapshot branch rows are empty")
    forbidden = {"quality", "quality_score", "weighted_quality", "Q", "label_score"}
    for row in rows:
        present = forbidden.intersection(row)
        if present:
            raise ProtocolError(f"branch rows contain a forbidden combined target: {sorted(present)}")
        for name in ("branch_id", "branch_category", "snapshot_id", "horizon"):
            if name not in row or str(row[name]) == "":
                raise ProtocolError(f"branch row lacks {name}")
        if int(row["horizon"]) not in protocol.branch_horizons:
            raise ProtocolError("branch row uses an unregistered horizon")
        if "A_mix" in {str(row["branch_id"]), str(row["branch_category"])}:
            raise ProtocolError("A_mix/FCAMP entered a formal Stage-4 branch row")
        for metric in protocol.primary_metrics:
            if metric.name not in row or not np.isfinite(float(row[metric.name])):
                raise ProtocolError(f"branch row lacks finite outcome {metric.name}")
        for family in PRIMARY_REWARD_FAMILIES:
            mean_name = f"reward_{family}_mean"
            if mean_name not in row or not np.isfinite(float(row[mean_name])):
                raise ProtocolError(f"branch row lacks finite {mean_name}")
            if seeds is not None:
                values = []
                for seed in seeds:
                    name = f"reward_{family}_seed_{int(seed)}"
                    if name not in row or not np.isfinite(float(row[name])):
                        raise ProtocolError(f"branch row lacks finite {name}")
                    values.append(float(row[name]))
                if not np.isclose(float(row[mean_name]), np.mean(values), atol=1.0e-6):
                    raise ProtocolError(f"branch row {mean_name} is not the seed mean")


def branch_pair_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    metrics: Sequence[OutcomeMetric],
    reward_families: Sequence[str] = PRIMARY_REWARD_FAMILIES,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["snapshot_id"]), int(row["horizon"])), []).append(row)
    records: list[dict[str, Any]] = []
    for (snapshot_id, horizon), group in groups.items():
        for left in range(len(group)):
            for right in range(left + 1, len(group)):
                preference = pareto_preference(group[left], group[right], metrics=metrics)
                if preference == 0:
                    continue
                winner, loser = (
                    (group[left], group[right]) if preference > 0 else (group[right], group[left])
                )
                for family in reward_families:
                    winner_score = float(winner[f"reward_{family}_mean"])
                    loser_score = float(loser[f"reward_{family}_mean"])
                    if not np.isfinite(winner_score) or not np.isfinite(loser_score):
                        raise ProtocolError("branch reward score contains NaN or Inf")
                    records.append(
                        {
                            "snapshot_id": snapshot_id,
                            "horizon": horizon,
                            "winner_branch_id": str(winner["branch_id"]),
                            "loser_branch_id": str(loser["branch_id"]),
                            "winner_category": str(winner["branch_category"]),
                            "loser_category": str(loser["branch_category"]),
                            "reward_family": family,
                            "winner_reward": winner_score,
                            "loser_reward": loser_score,
                            "reward_correct": bool(winner_score > loser_score),
                            "reward_tie": bool(winner_score == loser_score),
                        }
                    )
    return records


def reward_accuracy_by_family(
    rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
) -> dict[str, Any]:
    identity = {
        (str(row["snapshot_id"]), int(row["horizon"]), str(row["branch_id"])): index
        for index, row in enumerate(rows)
    }
    result: dict[str, Any] = {}
    for family in PRIMARY_REWARD_FAMILIES:
        family_pairs = [record for record in pairs if record["reward_family"] == family]
        unique_pairs = [
            (
                identity[(str(record["snapshot_id"]), int(record["horizon"]), str(record["winner_branch_id"]))],
                identity[(str(record["snapshot_id"]), int(record["horizon"]), str(record["loser_branch_id"]))],
            )
            for record in family_pairs
        ]
        if not unique_pairs:
            result[family] = {"pair_count": 0, "status": "NO_STRICT_PAIRS"}
            continue
        by_seed = []
        seed_banks = []
        for seed in seeds:
            values = np.asarray([float(row[f"reward_{family}_seed_{int(seed)}"]) for row in rows])
            seed_banks.append(values)
            by_seed.append({"seed": int(seed), **strict_pairwise_accuracy(values, unique_pairs)})
        ensemble = np.mean(np.stack(seed_banks), axis=0)
        result[family] = {
            "pair_count": len(unique_pairs),
            "by_seed": by_seed,
            "ensemble": strict_pairwise_accuracy(ensemble, unique_pairs),
            "seed_agreement": reward_seed_agreement(seed_banks),
        }
    return result


def per_outcome_spearman(
    rows: Sequence[Mapping[str, Any]],
    reward_values: np.ndarray,
    outcome_names: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    reward = np.asarray(reward_values, dtype=np.float64).reshape(-1)
    if reward.shape != (len(rows),) or not np.isfinite(reward).all():
        raise ProtocolError("reward/outcome rows are not aligned and finite")
    result: dict[str, dict[str, float | int]] = {}
    for name in outcome_names:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ProtocolError(f"outcome {name} contains NaN or Inf")
        statistic = float(spearmanr(reward, values).statistic)
        result[name] = {
            "count": len(rows),
            "spearman": statistic if np.isfinite(statistic) else 0.0,
            "constant_outcome": bool(np.ptp(values) == 0.0),
        }
    return result


def deterministic_blind_order(pair_id: str, seed: int) -> bool:
    """Return True when the winner is placed left, keyed only by opaque ID."""

    digest = canonical_sha256({"pair_id": str(pair_id), "ordering_seed": int(seed)})
    return bool(int(digest[:2], 16) & 1)


def classify_cem_hacking(
    records: Sequence[Mapping[str, Any]],
    *,
    alpha: float,
    bootstrap_replicates: int = 1000,
    seed: int = 20260803,
) -> dict[str, Any]:
    """Apply the frozen paired-bootstrap reward-hacking rule.

    Records are paired *snapshot/seed* comparisons, rather than three seed
    means.  That distinction is important: an exact sign test over only three
    means cannot attain the preregistered 0.05 significance level.
    """

    if not records:
        raise DependencyUnavailable("CEM exploitability has no paired seed records")
    gains = np.asarray([float(record["search_reward_gain"]) for record in records])
    if not np.isfinite(gains).all():
        raise ProtocolError("CEM reward gains contain NaN or Inf")
    observed = float(np.mean(gains))
    if int(bootstrap_replicates) < 100:
        raise ProtocolError("paired bootstrap needs at least 100 replicates")
    rng = np.random.default_rng(int(seed))
    centered = gains - observed
    draw = rng.integers(0, len(centered), size=(int(bootstrap_replicates), len(centered)))
    null_means = centered[draw].mean(axis=1)
    p_value = float((1 + np.sum(null_means >= observed)) / (1 + len(null_means)))
    significant_gain = observed > 0.0 and p_value <= float(alpha)
    regressions = [
        bool(record.get("strict_pareto_regression")) or bool(record.get("clear_failure"))
        for record in records
    ]
    hacking = bool(significant_gain and any(regressions))
    return {
        "paired_seed_count": len(records),
        "paired_snapshot_seed_record_count": len(records),
        "mean_search_reward_gain": observed,
        "one_sided_paired_bootstrap_p": p_value,
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(seed),
        "alpha": float(alpha),
        "significant_reward_gain": significant_gain,
        "physical_regression_or_failure_count": int(sum(regressions)),
        "significant_reward_hacking": hacking,
    }


__all__ = [
    "AMP_WINDOW_DIM",
    "AMP_WINDOW_STEPS",
    "BRANCH_BANK_SCHEMA",
    "BRANCH_ROW_SCHEMA",
    "CEM_BANK_SCHEMA",
    "INTERVENTION_BANK_SCHEMA",
    "BLIND_VIDEO_SOURCE_SCHEMA",
    "PRIMARY_REWARD_FAMILIES",
    "RewardValidityProtocol",
    "Stage3Critics",
    "branch_pair_rows",
    "canonical_trajectory_window_bank",
    "classify_cem_hacking",
    "deterministic_blind_order",
    "fit_reward_tensor",
    "flatten_amp_windows",
    "load_branch_bank",
    "load_directed_critic_edge",
    "load_stage3_critics",
    "aggregate_window_scores",
    "per_outcome_spearman",
    "read_parquet_rows",
    "reward_accuracy_by_family",
    "score_flat_bank",
    "score_canonical_trajectories_streaming",
    "validate_branch_bank",
    "validate_branch_rows",
    "write_branch_bank",
    "write_parquet_exclusive",
]
