"""Shared, fail-closed machinery for policy-class diagnostics 20--27.

The discovery suite deliberately separates collection from analysis.  This
module is the only stage-2 reader of the canonical rollout index and shards;
the entry-point scripts never launch a second, subtly different collector.
It also reconstructs the frozen checkpoint actor without importing Isaac Lab.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from diagnostics.common.manifest import (
    DependencyUnavailable,
    ProtocolError,
    read_json,
    sha256_file,
)
from diagnostics.common.observation_spec import (
    ObservationCardinality,
    ObservationSpec,
    derive_observation_specs,
)


CANONICAL_INDEX_NAME = "canonical_rollout_index.parquet"
PRIMARY_POLICY_CLASS_MODE = "clean_mean"
HISTORY_LENGTHS = (1, 2, 4, 8, 16, 32)
BC_HISTORIES = (1, 4, 8, 16, 32)


@dataclass(frozen=True)
class PolicyClassProtocol:
    primary_update: int
    auxiliary_updates: tuple[int, ...]
    required_primary_completion: float
    seeds: tuple[int, ...]
    reference_offsets: tuple[int, ...]
    alias_histories: tuple[int, ...]
    bc_histories: tuple[int, ...]
    knn_k: int
    max_probe_samples: int
    max_knn_samples: int
    mlp_hidden_dims: tuple[int, ...]
    gru_hidden_dim: int
    gru_layers: int
    epochs: int
    batch_size: int
    learning_rate: float
    patience: int
    phase_holdout: tuple[float, float]
    full_obs_sanity_nrmse: float
    phase_recovery_improvement: float
    ablation_delays: tuple[int, ...]
    exact_replay_action_atol: float
    open_loop_perturbation_fractions: tuple[float, ...]
    open_loop_horizon: int
    b_model: str
    b_history: int
    b_seed: int
    b_selection: str
    b_collector_modes: tuple[str, ...]
    b_common_sigmas: tuple[float, ...]
    b_output_name: str
    split_seed: int
    split_fractions: tuple[float, float, float]

    @classmethod
    def from_spec(cls, spec: Mapping[str, Any]) -> "PolicyClassProtocol":
        raw = spec.get("analysis_protocols", {}).get("policy_class")
        collection = spec.get("collection")
        if not isinstance(collection, Mapping):
            raise ProtocolError("spec lacks frozen collection protocol")
        if not isinstance(raw, Mapping):
            raise ProtocolError("spec lacks frozen analysis_protocols.policy_class")
        model = raw.get("probe_models")
        if not isinstance(model, Mapping):
            raise ProtocolError("policy_class.probe_models is missing")
        open_loop = raw.get("open_loop_replay")
        if not isinstance(open_loop, Mapping):
            raise ProtocolError("policy_class.open_loop_replay is missing")
        b_domain = raw.get("reference_free_B_domain")
        if not isinstance(b_domain, Mapping):
            raise ProtocolError("policy_class.reference_free_B_domain is missing")
        required = {
            "primary_teacher_checkpoint_update",
            "auxiliary_teacher_checkpoint_updates",
            "required_primary_teacher_motion_completion_min",
            "seeds",
            "reference_phase_offsets",
            "history_steps_for_aliasing",
            "history_steps_for_bc",
            "knn_k",
            "maximum_probe_samples",
            "maximum_knn_samples",
            "action_difference_thresholds",
            "contiguous_phase_holdout",
            "full_observation_sanity_action_nrmse_max",
            "phase_recoverability_relative_to_constant_circular_baseline_min",
            "reference_ablation_delays",
        }
        missing = sorted(required - set(raw))
        if missing:
            raise ProtocolError(f"policy_class protocol is missing fields: {missing}")
        model_required = {
            "mlp_hidden_dims", "gru_hidden_dim", "gru_layers", "epochs_max",
            "batch_size", "optimizer", "learning_rate",
            "early_stopping_patience", "checkpoint_selection",
        }
        model_missing = sorted(model_required - set(model))
        if model_missing:
            raise ProtocolError(f"policy_class probe_models is missing fields: {model_missing}")
        if raw["action_difference_thresholds"] != (
            "training_split_adjacent_action_distance_quantiles_0.50_0.90_0.95_with_0.95_primary"
        ):
            raise ProtocolError("unknown action-difference threshold protocol")
        if str(model["optimizer"]) != "AdamW" or str(model["checkpoint_selection"]) != (
            "best_validation_loss_test_untouched"
        ):
            raise ProtocolError("unsupported frozen probe optimization protocol")
        values = cls(
            primary_update=int(raw["primary_teacher_checkpoint_update"]),
            auxiliary_updates=tuple(int(value) for value in raw["auxiliary_teacher_checkpoint_updates"]),
            required_primary_completion=float(raw["required_primary_teacher_motion_completion_min"]),
            seeds=tuple(int(value) for value in raw["seeds"]),
            reference_offsets=tuple(int(value) for value in raw["reference_phase_offsets"]),
            alias_histories=tuple(int(value) for value in raw["history_steps_for_aliasing"]),
            bc_histories=tuple(int(value) for value in raw["history_steps_for_bc"]),
            knn_k=int(raw["knn_k"]),
            max_probe_samples=int(raw["maximum_probe_samples"]),
            max_knn_samples=int(raw["maximum_knn_samples"]),
            mlp_hidden_dims=tuple(int(value) for value in model["mlp_hidden_dims"]),
            gru_hidden_dim=int(model["gru_hidden_dim"]),
            gru_layers=int(model["gru_layers"]),
            epochs=int(model["epochs_max"]),
            batch_size=int(model["batch_size"]),
            learning_rate=float(model["learning_rate"]),
            patience=int(model["early_stopping_patience"]),
            phase_holdout=tuple(float(value) for value in raw["contiguous_phase_holdout"]),
            full_obs_sanity_nrmse=float(raw["full_observation_sanity_action_nrmse_max"]),
            phase_recovery_improvement=float(
                raw["phase_recoverability_relative_to_constant_circular_baseline_min"]
            ),
            ablation_delays=tuple(int(value) for value in raw["reference_ablation_delays"]),
            exact_replay_action_atol=float(open_loop["exact_identity_action_atol"]),
            open_loop_perturbation_fractions=tuple(
                float(value)
                for value in open_loop[
                    "initial_joint_position_perturbation_fraction_of_joint_range"
                ]
            ),
            open_loop_horizon=int(open_loop["horizon_control_steps"]),
            b_model=str(b_domain["model"]),
            b_history=int(b_domain["history_steps"]),
            b_seed=int(b_domain["seed"]),
            b_selection=str(b_domain["selection"]),
            b_collector_modes=tuple(str(value) for value in b_domain["collector_modes"]),
            b_common_sigmas=tuple(
                float(value) for value in b_domain["common_action_noise_scales"]
            ),
            b_output_name=str(b_domain["required_output"]),
            split_seed=int(collection["trajectory_split_seed"]),
            split_fractions=tuple(float(value) for value in collection["trajectory_split_fractions"]),
        )
        if len(values.seeds) != 3 or len(set(values.seeds)) != 3:
            raise ProtocolError("policy-class probes require exactly three distinct seeds")
        if values.alias_histories != HISTORY_LENGTHS or values.bc_histories != BC_HISTORIES:
            raise ProtocolError("history protocols differ from the frozen discovery design")
        if values.mlp_hidden_dims != (256, 256) or values.gru_layers != 1:
            raise ProtocolError("current audited probe implementation supports only the frozen model shapes")
        if not (0.0 <= values.phase_holdout[0] < values.phase_holdout[1] <= 1.0):
            raise ProtocolError("contiguous phase holdout is invalid")
        if any(value <= 0 for value in values.ablation_delays):
            raise ProtocolError("reference ablation delays must be positive")
        if open_loop.get("perturbation_direction") != "shared_Rademacher_from_NoiseBank":
            raise ProtocolError("unsupported open-loop perturbation direction")
        if open_loop.get("comparison") != (
            "recorded_nominal_action_open_loop_vs_teacher_feedback_from_same_perturbed_snapshot"
        ):
            raise ProtocolError("unsupported open-loop comparison")
        if values.exact_replay_action_atol < 0.0 or any(
            value <= 0.0 for value in values.open_loop_perturbation_fractions
        ):
            raise ProtocolError("open-loop replay tolerances are invalid")
        if values.open_loop_horizon != 50:
            raise ProtocolError("open-loop replay horizon differs from the frozen 50-step protocol")
        if values.open_loop_horizon > int(collection["horizon_control_steps"]):
            raise ProtocolError("open-loop replay horizon exceeds canonical collection horizon")
        if (
            values.b_model,
            values.b_history,
            values.b_seed,
            values.b_selection,
            values.b_collector_modes,
            values.b_output_name,
        ) != (
            "GRU",
            32,
            20260803,
            "preregistered_maximum_context_not_posthoc_best",
            ("clean_mean", "controlled_environment", "common_action_noise"),
            "canonical_B_rollout_index.parquet",
        ):
            raise ProtocolError("reference-free B-domain policy/collector selection changed")
        collection_sigmas = tuple(float(value) for value in collection["common_action_noise_scales"])
        if values.b_common_sigmas != collection_sigmas:
            raise ProtocolError("B-domain common-noise scales differ from canonical collection")
        if values.b_history not in values.bc_histories or values.b_seed not in values.seeds:
            raise ProtocolError("B-domain model is outside the frozen BC probe grid")
        if len(values.split_fractions) != 3 or abs(sum(values.split_fractions) - 1.0) > 1.0e-9:
            raise ProtocolError("trajectory split fractions must contain train/validation/test and sum to one")
        return values


def import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised outside the research env
        raise DependencyUnavailable(
            "PyTorch is required; run the diagnostic in conda env_isaaclab"
        ) from exc
    return torch


def import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise DependencyUnavailable(
            "pandas and a parquet engine are required for canonical rollout indices"
        ) from exc
    return pd


def output_dir_from_args(
    repo_root: Path,
    spec: Mapping[str, Any],
    explicit: Path | None,
) -> Path:
    raw = explicit or Path(str(spec.get("output_dir", "output/largebox_discovery_v1")))
    value = Path(raw).expanduser()
    return (value if value.is_absolute() else repo_root / value).resolve()


def _literal_symbols(path: Path) -> dict[str, Any]:
    """Safely resolve literal module constants, including simple aliases."""

    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes: dict[str, ast.AST] = {}
    for statement in module.body:
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if isinstance(target, ast.Name):
                    nodes[target.id] = statement.value
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            nodes[statement.target.id] = statement.value
    resolved: dict[str, Any] = {}

    def resolve(name: str, stack: tuple[str, ...] = ()) -> Any:
        if name in resolved:
            return resolved[name]
        if name in stack or name not in nodes:
            raise ProtocolError(f"cannot resolve literal constant {name!r} in {path}")
        node = nodes[name]
        if isinstance(node, ast.Name):
            value = resolve(node.id, (*stack, name))
        else:
            try:
                value = ast.literal_eval(node)
            except (ValueError, TypeError) as exc:
                raise ProtocolError(
                    f"constant {name!r} in {path} is not an audited literal"
                ) from exc
        resolved[name] = value
        return value

    for name in nodes:
        try:
            resolve(name)
        except ProtocolError:
            # Most production constants are computed objects.  A caller that
            # needs one of them will fail explicitly below.
            continue
    return resolved


def derive_repository_observation_specs(repo_root: Path) -> tuple[dict[str, ObservationSpec], dict[str, Any]]:
    """AST-derive actor/critic slices and all cardinalities from source."""

    observation_source = repo_root / "envs" / "observation.py"
    spec_source = repo_root / "envs" / "spec.py"
    robot_source = repo_root / "envs" / "robots" / "g1.py"
    if not all(path.is_file() for path in (observation_source, spec_source, robot_source)):
        raise FileNotFoundError("production observation/spec/robot source is incomplete")
    constants = _literal_symbols(spec_source)
    robot_constants = _literal_symbols(robot_source)
    try:
        action_names = robot_constants["G1_29DOF_ACTION_NAMES"]
        track_bodies = constants["MIMIC_BODY_NAMES"]
        termination_bodies = constants["MIMIC_TERMINATION_BODY_NAMES"]
        foot_bodies = constants["MIMIC_FOOT_BODY_NAMES"]
    except KeyError as exc:
        raise ProtocolError(f"required observation cardinality constant is missing: {exc}") from exc
    cardinality = ObservationCardinality(
        action_joint_count=len(action_names),
        termination_body_count=len(termination_bodies),
        termination_contact_body_count=len(termination_bodies),
        foot_body_count=len(foot_bodies),
        track_body_count=len(track_bodies),
    )
    specs = derive_observation_specs(observation_source, cardinality)
    declared_dims = {
        "actor": constants.get("OBS_DIM"),
        "critic": constants.get("CRITIC_OBS_DIM"),
    }
    mismatches = {
        stream: {"derived": specs[stream].total_dim, "declared": declared}
        for stream, declared in declared_dims.items()
        if not isinstance(declared, int) or specs[stream].total_dim != declared
    }
    if mismatches:
        raise ProtocolError(f"AST-derived observation dimensions disagree with source: {mismatches}")
    provenance = {
        "cardinality": {
            "action_joint_count": cardinality.action_joint_count,
            "termination_body_count": cardinality.termination_body_count,
            "termination_contact_body_count": cardinality.termination_contact_body_count,
            "foot_body_count": cardinality.foot_body_count,
            "track_body_count": cardinality.track_body_count,
        },
        "cardinality_sources": {
            "action_joint_count": f"{robot_source}:G1_29DOF_ACTION_NAMES",
            "termination_body_count": f"{spec_source}:MIMIC_TERMINATION_BODY_NAMES",
            "termination_contact_body_count": "same runtime list as MIMIC_TERMINATION_BODY_NAMES in envs/g1_mimic.py",
            "foot_body_count": f"{spec_source}:MIMIC_FOOT_BODY_NAMES",
            "track_body_count": f"{spec_source}:MIMIC_BODY_NAMES",
        },
        "observation_source": str(observation_source.resolve()),
        "observation_source_sha256": sha256_file(observation_source),
    }
    return specs, provenance


def load_observation_partition(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get("status") != "PASS":
        raise DependencyUnavailable(f"observation partition is not PASS: {path}")
    for stream in ("actor", "critic"):
        value = payload.get(stream)
        if not isinstance(value, dict) or not isinstance(value.get("terms"), list):
            raise ProtocolError(f"observation partition has no {stream} term layout")
    return payload


def named_term_indices(partition: Mapping[str, Any], *, role: str | None = None, names: Iterable[str] | None = None) -> np.ndarray:
    terms = partition["actor"]["terms"]
    selected_names = None if names is None else set(str(value) for value in names)
    indices: list[int] = []
    for term in terms:
        if role is not None and term.get("role") != role:
            continue
        if selected_names is not None and term.get("name") not in selected_names:
            continue
        indices.extend(range(int(term["start"]), int(term["stop"])))
    if selected_names is not None:
        observed = {term["name"] for term in terms if term["name"] in selected_names}
        if observed != selected_names:
            raise ProtocolError(f"unknown named actor terms: {sorted(selected_names - observed)}")
    return np.asarray(indices, dtype=np.int64)


def _read_rollout_index(index_path: Path):
    """Read the stage-1 index, preferring its canonical helper when present."""

    if not index_path.is_file():
        raise DependencyUnavailable(f"canonical rollout index is missing: {index_path}")
    try:
        module = importlib.import_module("diagnostics.common.canonical_collection")
        loader = getattr(module, "load_rollout_index", None)
        if callable(loader):
            result = loader(index_path)
            # The stage-2 contract is a DataFrame-like table.  Converting here
            # makes downstream behavior independent of the helper's exact type.
            pd = import_pandas()
            return result.copy() if isinstance(result, pd.DataFrame) else pd.DataFrame(result)
    except (ImportError, AttributeError):
        pass
    pd = import_pandas()
    try:
        return pd.read_parquet(index_path)
    except Exception as exc:
        raise ProtocolError(f"cannot read canonical rollout index {index_path}: {exc}") from exc


REQUIRED_INDEX_COLUMNS = frozenset(
    {
        "sample_id",
        "trajectory_id",
        "snapshot_id",
        "checkpoint_update",
        "checkpoint_sha256",
        "checkpoint_lineage_id",
        "collector_mode",
        "shard_path",
        "shard_env_index",
        "num_steps",
    }
)


def canonical_index(index_path: Path):
    frame = _read_rollout_index(index_path)
    missing = REQUIRED_INDEX_COLUMNS - set(frame.columns)
    if missing:
        raise ProtocolError(f"canonical rollout index is missing columns: {sorted(missing)}")
    if frame.empty:
        raise DependencyUnavailable("canonical rollout index contains no trajectories")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ProtocolError("canonical rollout sample_id values are not unique")
    if (frame["num_steps"].astype(int) <= 0).any():
        raise ProtocolError("canonical rollout index contains empty trajectories")
    return frame


def select_policy_class_rows(
    frame,
    *,
    mode: str = PRIMARY_POLICY_CLASS_MODE,
    checkpoint_sha256: str | None = None,
    checkpoint_update: int | None = None,
):
    selected = frame[frame["collector_mode"].astype(str) == str(mode)]
    if checkpoint_sha256:
        selected = selected[selected["checkpoint_sha256"].astype(str) == checkpoint_sha256]
    elif checkpoint_update is not None:
        selected = selected[selected["checkpoint_update"].astype(int) == int(checkpoint_update)]
    else:
        latest = int(selected["checkpoint_update"].astype(int).max()) if not selected.empty else None
        if latest is not None:
            selected = selected[selected["checkpoint_update"].astype(int) == latest]
    if selected.empty:
        qualifier = checkpoint_sha256 or checkpoint_update or "latest"
        raise DependencyUnavailable(
            f"no canonical {mode} trajectories exist for checkpoint {qualifier}"
        )
    lineages = set(selected["checkpoint_lineage_id"].astype(str))
    hashes = set(selected["checkpoint_sha256"].astype(str))
    if len(lineages) != 1 or len(hashes) != 1:
        raise ProtocolError("policy-class probe must use exactly one checkpoint and lineage")
    return selected.sort_values(["trajectory_id", "shard_env_index"], kind="stable").reset_index(drop=True)


def require_primary_teacher_quality(
    output_dir: Path,
    protocol: PolicyClassProtocol,
    *,
    checkpoint_path: str | Path,
) -> dict[str, Any]:
    """Require the preregistered clean capability result for the primary teacher."""

    transition = output_dir / "teacher_transition.json"
    table = output_dir / "tables" / "teacher_learning_curve.csv"
    if not transition.is_file() or not table.is_file():
        raise DependencyUnavailable("primary-teacher clean capability evidence is missing")
    status = read_json(transition)
    if not isinstance(status, Mapping) or status.get("status") != "PASS":
        raise DependencyUnavailable("teacher learning-curve audit has not passed")
    pd = import_pandas()
    frame = pd.read_csv(table)
    required = {"update", "protocol", "motion_complete_frac", "source_paths", "run_id"}
    if not required <= set(frame.columns):
        raise ProtocolError(
            f"teacher learning curve lacks quality fields: {sorted(required - set(frame.columns))}"
        )
    selected = frame[
        (frame["update"].astype(int) == protocol.primary_update)
        & (frame["protocol"].astype(str) == "clean_mean")
    ].copy()
    checkpoint_run = Path(checkpoint_path).expanduser().resolve().parent.parent.name
    selected = selected[selected["run_id"].astype(str) == checkpoint_run]
    selected["motion_complete_frac"] = pd.to_numeric(
        selected["motion_complete_frac"], errors="coerce"
    )
    selected = selected[selected["motion_complete_frac"].notna()]
    if selected.empty:
        raise DependencyUnavailable(
            "no condition-matched clean validation exists for primary teacher "
            f"run={checkpoint_run} update={protocol.primary_update}"
        )
    completions = selected["motion_complete_frac"].astype(float)
    if float(completions.max() - completions.min()) > 1.0e-9:
        raise ProtocolError(
            "primary teacher has conflicting clean completion records; refusing best-result selection"
        )
    row = selected.iloc[0]
    completion = float(row["motion_complete_frac"])
    if completion < protocol.required_primary_completion:
        raise DependencyUnavailable(
            "primary teacher fails the frozen clean capability gate: "
            f"completion={completion}, required={protocol.required_primary_completion}"
        )
    return {
        "checkpoint_update": protocol.primary_update,
        "motion_completion": completion,
        "required_motion_completion": protocol.required_primary_completion,
        "run_id": str(row["run_id"]),
        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
        "source_paths": str(row["source_paths"]),
        "protocol": "clean_mean",
        "role": "capability gate only; never stochastic occupancy evidence",
    }


def _resolve_shard_path(index_path: Path, raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    candidates = [index_path.parent / path, index_path.parent.parent / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _tree_slice_env(value: Any, *, steps: int, env_index: int, num_envs_hint: int | None) -> Any:
    torch = import_torch()
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.detach().cpu()
        if value.shape[0] < steps:
            raise ProtocolError(
                f"shard tensor leading length {value.shape[0]} is shorter than indexed {steps}"
            )
        sliced = value[:steps]
        if sliced.ndim >= 2 and num_envs_hint is not None and sliced.shape[1] == num_envs_hint:
            if not (0 <= env_index < num_envs_hint):
                raise ProtocolError(f"shard_env_index {env_index} is outside [0,{num_envs_hint})")
            sliced = sliced[:, env_index]
        return sliced.detach().cpu()
    if isinstance(value, Mapping):
        return {
            str(key): _tree_slice_env(
                nested, steps=steps, env_index=env_index, num_envs_hint=num_envs_hint
            )
            for key, nested in value.items()
        }
    return value


def load_indexed_trajectory(index_path: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    """Load one [T,...] branch through the stage-1 helper or generic shard reader."""

    try:
        module = importlib.import_module("diagnostics.common.canonical_collection")
        loader = getattr(module, "load_rollout_trajectory", None)
        if callable(loader):
            result = loader(index_path, row)
            if not isinstance(result, Mapping):
                raise ProtocolError("stage-1 load_rollout_trajectory returned a non-mapping")
            return dict(result)
    except ImportError:
        pass
    torch = import_torch()
    shard_path = _resolve_shard_path(index_path, row["shard_path"])
    if not shard_path.is_file():
        raise DependencyUnavailable(f"canonical rollout shard is missing: {shard_path}")
    try:
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        shard = torch.load(shard_path, map_location="cpu")
    if not isinstance(shard, Mapping):
        raise ProtocolError(f"rollout shard is not a mapping: {shard_path}")
    required = {"trajectory", "observation", "action"}
    if not required <= set(shard):
        raise ProtocolError(f"rollout shard lacks stage-2 sections: {sorted(required - set(shard))}")
    # Infer N from actor_full, the canonical mandatory tensor.
    actor = shard["observation"].get("actor_full")
    if not torch.is_tensor(actor) or actor.ndim < 2:
        raise ProtocolError("canonical actor_full must have shape [T,N,D] or [T,D]")
    num_envs = int(actor.shape[1]) if actor.ndim >= 3 else None
    return _tree_slice_env(
        shard,
        steps=int(row["num_steps"]),
        env_index=int(row["shard_env_index"]),
        num_envs_hint=num_envs,
    )


@dataclass(frozen=True)
class CanonicalArrays:
    actor_full: np.ndarray
    actor_no_reference: np.ndarray
    actor_reference_terms: np.ndarray
    actor_proprio_terms: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray
    phases: np.ndarray
    contact_modes: np.ndarray
    trajectory_ids: np.ndarray
    snapshot_ids: np.ndarray
    steps: np.ndarray
    sample_ids: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        count = self.actor_full.shape[0]
        arrays = (
            self.actor_no_reference,
            self.actor_reference_terms,
            self.actor_proprio_terms,
            self.action_mean,
            self.action_std,
        )
        if self.actor_full.ndim != 2 or any(value.ndim != 2 or value.shape[0] != count for value in arrays):
            raise ProtocolError("canonical feature/action arrays must be aligned [N,D]")
        for value in (self.phases, self.contact_modes, self.trajectory_ids, self.snapshot_ids, self.steps, self.sample_ids):
            if value.shape != (count,):
                raise ProtocolError("canonical scalar/id arrays are not aligned")
        if not all(np.isfinite(value).all() for value in arrays[:5]):
            raise ProtocolError("canonical policy-class arrays contain NaN or Inf")
        if not np.isfinite(self.phases).all():
            raise ProtocolError("canonical phases contain NaN or Inf")


def _numpy_2d(value: Any, name: str) -> np.ndarray:
    torch = import_torch()
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.ndim != 2:
        raise ProtocolError(f"canonical {name} must have shape [T,D], got {result.shape}")
    return result.astype(np.float32, copy=False)


def _numpy_1d(value: Any, name: str, length: int, *, default: Any = None) -> np.ndarray:
    torch = import_torch()
    if value is None:
        if default is None:
            raise ProtocolError(f"canonical {name} is missing")
        return np.full(length, default)
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.ndim == 0:
        result = np.full(length, result.item())
    if result.shape != (length,):
        raise ProtocolError(f"canonical {name} must have shape [T], got {result.shape}")
    return result


def load_canonical_arrays(index_path: Path, rows) -> CanonicalArrays:
    fields: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "actor_full", "actor_no_reference", "actor_reference_terms",
            "actor_proprio_terms", "action_mean", "action_std", "phases",
            "contact_modes", "trajectory_ids", "snapshot_ids", "steps", "sample_ids",
        )
    }
    checkpoint_hashes: set[str] = set()
    checkpoint_updates: set[int] = set()
    checkpoint_paths: set[str] = set()
    for _, series in rows.iterrows():
        row = series.to_dict()
        tree = load_indexed_trajectory(index_path, row)
        observation = tree.get("observation")
        action = tree.get("action")
        trajectory = tree.get("trajectory")
        if not all(isinstance(value, Mapping) for value in (observation, action, trajectory)):
            raise ProtocolError("canonical shard sections are not mappings")
        actor_full = _numpy_2d(observation.get("actor_full"), "observation.actor_full")
        length = actor_full.shape[0]
        fields["actor_full"].append(actor_full)
        for output_name, source_name in (
            ("actor_no_reference", "actor_no_reference"),
            ("actor_reference_terms", "actor_reference_terms"),
            ("actor_proprio_terms", "actor_proprio_terms"),
        ):
            fields[output_name].append(_numpy_2d(observation.get(source_name), f"observation.{source_name}"))
        fields["action_mean"].append(_numpy_2d(action.get("mean"), "action.mean"))
        fields["action_std"].append(_numpy_2d(action.get("std"), "action.std"))
        phase_value = trajectory.get("phase_continuous", trajectory.get("phase"))
        fields["phases"].append(_numpy_1d(phase_value, "trajectory.phase_continuous", length).astype(np.float64))
        fields["contact_modes"].append(
            _numpy_1d(trajectory.get("contact_mode"), "trajectory.contact_mode", length, default=-1)
        )
        fields["trajectory_ids"].append(np.full(length, str(row["trajectory_id"]), dtype=object))
        fields["snapshot_ids"].append(np.full(length, str(row["snapshot_id"]), dtype=object))
        steps = trajectory.get("step")
        if steps is None:
            steps = np.arange(length, dtype=np.int64)
        fields["steps"].append(_numpy_1d(steps, "trajectory.step", length).astype(np.int64))
        fields["sample_ids"].append(np.full(length, str(row["sample_id"]), dtype=object))
        checkpoint_hashes.add(str(row["checkpoint_sha256"]))
        checkpoint_updates.add(int(row["checkpoint_update"]))
        if row.get("checkpoint_path"):
            checkpoint_paths.add(str(row["checkpoint_path"]))
    if not fields["actor_full"]:
        raise DependencyUnavailable("selected canonical rollout rows contain no samples")
    values = {name: np.concatenate(parts, axis=0) for name, parts in fields.items()}
    return CanonicalArrays(
        actor_full=values["actor_full"],
        actor_no_reference=values["actor_no_reference"],
        actor_reference_terms=values["actor_reference_terms"],
        actor_proprio_terms=values["actor_proprio_terms"],
        action_mean=values["action_mean"],
        action_std=values["action_std"],
        phases=values["phases"],
        contact_modes=values["contact_modes"],
        trajectory_ids=values["trajectory_ids"],
        snapshot_ids=values["snapshot_ids"],
        steps=values["steps"],
        sample_ids=values["sample_ids"],
        metadata={
            "checkpoint_sha256": next(iter(checkpoint_hashes)) if len(checkpoint_hashes) == 1 else None,
            "checkpoint_update": next(iter(checkpoint_updates)) if len(checkpoint_updates) == 1 else None,
            "checkpoint_paths": sorted(checkpoint_paths),
            "trajectory_count": int(len(set(values["trajectory_ids"].tolist()))),
            "sample_count": int(values["actor_full"].shape[0]),
        },
    )


def deterministic_group_splits(
    trajectory_ids: Sequence[Any],
    snapshot_ids: Sequence[Any],
    *,
    seed: int = 20260803,
    fractions: tuple[float, float, float] = (0.70, 0.15, 0.15),
) -> np.ndarray:
    """Split complete snapshot/trajectory groups; never individual windows."""

    trajectory_ids = np.asarray(trajectory_ids).astype(str)
    snapshot_ids = np.asarray(snapshot_ids).astype(str)
    if trajectory_ids.shape != snapshot_ids.shape:
        raise ProtocolError("trajectory and snapshot ids are not aligned")
    if len(fractions) != 3 or any(value < 0.0 for value in fractions) or abs(sum(fractions) - 1.0) > 1.0e-9:
        raise ProtocolError("split fractions must be three nonnegative values summing to one")
    group_by_trajectory: dict[str, str] = {}
    for trajectory, snapshot in zip(trajectory_ids, snapshot_ids):
        previous = group_by_trajectory.setdefault(trajectory, snapshot)
        if previous != snapshot:
            raise ProtocolError("one trajectory_id maps to multiple snapshots")
    snapshots = sorted(set(group_by_trajectory.values()))
    if len(snapshots) < 3:
        raise DependencyUnavailable(
            "at least three independent snapshots are required for train/validation/test"
        )
    scored = sorted(
        snapshots,
        key=lambda value: hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest(),
    )
    split_for: dict[str, str] = {}
    for index, snapshot in enumerate(scored):
        fraction = (index + 0.5) / len(scored)
        train_stop = float(fractions[0])
        validation_stop = train_stop + float(fractions[1])
        split_for[snapshot] = (
            "train" if fraction < train_stop
            else ("validation" if fraction < validation_stop else "test")
        )
    # Small group counts can leave a bin empty; reserve one deterministic group
    # for each evaluation split without fragmenting a snapshot.
    if "validation" not in split_for.values():
        split_for[scored[-2]] = "validation"
    if "test" not in split_for.values():
        split_for[scored[-1]] = "test"
    if "train" not in split_for.values():
        split_for[scored[0]] = "train"
    result = np.asarray([split_for[value] for value in snapshot_ids], dtype=object)
    for split in ("train", "validation", "test"):
        if not np.any(result == split):
            raise ProtocolError(f"group split produced an empty {split} set")
    return result


def phase_period(phases: np.ndarray) -> float:
    values = np.asarray(phases, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all() or values.size < 2:
        raise ProtocolError("cannot infer phase period from invalid phases")
    lower, upper = float(values.min()), float(values.max())
    if lower >= -1.0e-6 and upper <= 1.0 + 1.0e-6:
        return 1.0
    unique = np.unique(values)
    deltas = np.diff(unique)
    step = float(np.median(deltas[deltas > 1.0e-9])) if np.any(deltas > 1.0e-9) else 1.0
    return max(upper - lower + step, step)


def normalized_phase(phases: np.ndarray, period: float) -> np.ndarray:
    values = np.asarray(phases, dtype=np.float64)
    origin = float(np.min(values))
    return np.mod(values - origin, float(period)) / float(period)


def build_history_end_indices(trajectory_ids: Sequence[Any], history: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(trajectory_ids)
    if history < 1:
        raise ProtocolError("history must be positive")
    end_indices: list[int] = []
    starts: list[int] = []
    for end in range(history - 1, ids.size):
        start = end + 1 - history
        if np.all(ids[start : end + 1] == ids[end]):
            starts.append(start)
            end_indices.append(end)
    if not end_indices:
        raise DependencyUnavailable(f"no trajectory has a valid history of {history} steps")
    return np.asarray(starts, dtype=np.int64), np.asarray(end_indices, dtype=np.int64)


def history_windows(features: np.ndarray, trajectory_ids: Sequence[Any], history: int) -> tuple[np.ndarray, np.ndarray]:
    starts, ends = build_history_end_indices(trajectory_ids, history)
    windows = np.stack([features[start : end + 1] for start, end in zip(starts, ends)])
    return windows.astype(np.float32, copy=False), ends


def deterministic_subsample(indices: np.ndarray, maximum: int, *, seed: int) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size <= maximum:
        return indices
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(indices, size=int(maximum), replace=False))


def _checkpoint_candidates(repo_root: Path, output_dir: Path, row: Mapping[str, Any], explicit: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit is not None:
        value = explicit.expanduser()
        candidates.append((value if value.is_absolute() else repo_root / value).resolve())
    if row.get("checkpoint_path"):
        candidates.append(Path(str(row["checkpoint_path"])).expanduser().resolve())
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if isinstance(manifest, Mapping) and manifest.get("checkpoint_path"):
            candidates.append(Path(str(manifest["checkpoint_path"])).expanduser().resolve())
    for csv_path in (
        output_dir / "checkpoint_inventory.csv",
        output_dir / "checkpoints" / "dense_checkpoint_inventory.csv",
    ):
        if not csv_path.is_file():
            continue
        pd = import_pandas()
        frame = pd.read_csv(csv_path)
        for _, item in frame.iterrows():
            if str(item.get("checkpoint_sha256", "")) == str(row.get("checkpoint_sha256", "")):
                # diag_02's full inventory calls this column ``path`` while
                # diag_10's frozen dense inventory calls it
                # ``checkpoint_path``.  Both are declared suite artifacts.
                # Eagerly indexing only ``path`` made resolution fail before
                # it could use the already-valid canonical-row path.
                value = item.get("checkpoint_path")
                if not isinstance(value, str) or not value:
                    value = item.get("path")
                if isinstance(value, str) and value:
                    candidates.append(Path(value).expanduser().resolve())
    update = int(row["checkpoint_update"])
    # Filename filtering avoids hashing every checkpoint in a large archive.
    candidates.extend(repo_root.glob(f"runs/*/checkpoints/update_{update:04d}.pt"))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def resolve_checkpoint(
    repo_root: Path,
    output_dir: Path,
    row: Mapping[str, Any],
    *,
    explicit: Path | None = None,
) -> Path:
    expected = str(row["checkpoint_sha256"])
    mismatched: list[str] = []
    for path in _checkpoint_candidates(repo_root, output_dir, row, explicit):
        if not path.is_file():
            continue
        actual = sha256_file(path)
        if actual == expected:
            return path
        mismatched.append(f"{path}={actual}")
    detail = "; ".join(mismatched[:3])
    raise DependencyUnavailable(
        f"checkpoint entity with SHA-256 {expected} is unavailable"
        + (f"; mismatched candidates: {detail}" if detail else "")
    )


class FrozenCheckpointPolicy:
    """Exact HOLOSOMA actor/normalizer reconstruction, without an Isaac env."""

    def __init__(self, checkpoint: Path, *, device: str = "cpu") -> None:
        torch = import_torch()
        from models.holosoma_ppo import EmpiricalNormalization, PPOActor

        try:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(checkpoint, map_location="cpu")
        if not isinstance(payload, Mapping) or not isinstance(payload.get("policy"), Mapping):
            raise ProtocolError("checkpoint has no policy state")
        state = payload["policy"]
        config = payload.get("config", {})
        parameters = config.get("parameters", {}) if isinstance(config, Mapping) else {}
        first_weight = state.get("actor.actor_module.0.weight")
        std = state.get("actor.std")
        if not torch.is_tensor(first_weight) or not torch.is_tensor(std):
            raise ProtocolError("checkpoint actor layout is incompatible with fixed_reward")
        obs_dim = int(first_weight.shape[1])
        action_dim = int(std.numel())
        linear_weights = [
            (key, value)
            for key, value in state.items()
            if str(key).startswith("actor.actor_module.") and str(key).endswith(".weight")
        ]
        linear_weights.sort(key=lambda item: int(str(item[0]).split(".")[2]))
        hidden = tuple(int(value.shape[0]) for _, value in linear_weights[:-1])
        activation = str(parameters.get("activation", "ELU"))
        actor = PPOActor(
            observation_dim=obs_dim,
            hidden_dims=hidden,
            activation=activation,
            num_actions=action_dim,
            init_noise_std=1.0,
        )
        normalizer = EmpiricalNormalization(obs_dim, "cpu")
        actor_state = {
            str(key)[len("actor.") :]: value
            for key, value in state.items()
            if str(key).startswith("actor.")
        }
        normalizer_state = {
            str(key)[len("actor_obs_normalizer.") :]: value
            for key, value in state.items()
            if str(key).startswith("actor_obs_normalizer.")
        }
        actor.load_state_dict(actor_state, strict=True)
        normalizer.load_state_dict(normalizer_state, strict=True)
        self.device = torch.device(device)
        self.actor = actor.to(self.device).eval()
        self.normalizer = normalizer.to(self.device).eval()
        self.observation_dim = obs_dim
        self.action_dim = action_dim
        self.checkpoint_path = checkpoint.resolve()
        self.checkpoint_sha256 = sha256_file(checkpoint)
        self.update = payload.get("update_idx")

    def mean(self, observation: np.ndarray, *, batch_size: int = 65536) -> np.ndarray:
        torch = import_torch()
        values = np.asarray(observation, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.observation_dim:
            raise ProtocolError(
                f"policy expected observations [N,{self.observation_dim}], got {values.shape}"
            )
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, values.shape[0], int(batch_size)):
                batch = torch.as_tensor(values[start : start + batch_size], device=self.device)
                output = self.actor.act_inference(self.normalizer(batch, update=False))
                if not bool(torch.isfinite(output).all()):
                    raise ProtocolError("checkpoint policy produced NaN or Inf")
                outputs.append(output.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0)

    @property
    def std(self) -> np.ndarray:
        return self.actor.std.detach().cpu().numpy().copy()


def action_prediction_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    from diagnostics.common.policy_class import action_prediction_metrics as metrics

    return metrics(prediction, target)


def circular_phase_metrics(prediction_angle: np.ndarray, target_phase: np.ndarray) -> dict[str, float]:
    predicted = np.mod(np.asarray(prediction_angle, dtype=np.float64), 2.0 * np.pi)
    target = np.mod(np.asarray(target_phase, dtype=np.float64), 1.0) * (2.0 * np.pi)
    delta = np.abs(np.angle(np.exp(1j * (predicted - target))))
    baseline_angle = np.angle(np.mean(np.exp(1j * target)))
    baseline = np.abs(np.angle(np.exp(1j * (baseline_angle - target))))
    return {
        "circular_mae_radians": float(delta.mean()),
        "circular_rmse_radians": float(np.sqrt(np.mean(np.square(delta)))),
        "circular_mae_fraction_cycle": float(delta.mean() / (2.0 * np.pi)),
        "constant_baseline_mae_radians": float(baseline.mean()),
        "relative_to_constant_baseline": float(delta.mean() / max(float(baseline.mean()), 1.0e-8)),
    }


@dataclass(frozen=True)
class ProbeDataset:
    x: np.ndarray
    action: np.ndarray
    phase_normalized: np.ndarray
    splits: np.ndarray
    end_indices: np.ndarray
    history: int


def make_probe_dataset(
    features: np.ndarray,
    arrays: CanonicalArrays,
    splits: np.ndarray,
    *,
    history: int,
) -> ProbeDataset:
    windows, ends = history_windows(features, arrays.trajectory_ids, history)
    x = windows[:, -1] if history == 1 else windows
    period = phase_period(arrays.phases)
    phase = normalized_phase(arrays.phases, period)[ends]
    return ProbeDataset(
        x=x,
        action=arrays.action_mean[ends],
        phase_normalized=phase,
        splits=splits[ends],
        end_indices=ends,
        history=int(history),
    )


def _fit_standardization(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axes = tuple(range(train.ndim - 1))
    mean = np.mean(train, axis=axes, keepdims=True)
    std = np.std(train, axis=axes, keepdims=True)
    return mean.astype(np.float32), np.maximum(std, 1.0e-6).astype(np.float32)


def _model_for(kind: str, input_dim: int, output_dim: int, *, hidden_dim: int = 256):
    from diagnostics.common.policy_class import GRUBehaviorClone, MLPBehaviorClone

    if kind == "mlp":
        return MLPBehaviorClone(input_dim, output_dim, hidden=(hidden_dim, hidden_dim))
    if kind == "gru":
        return GRUBehaviorClone(input_dim, output_dim, hidden_dim=hidden_dim)
    raise ProtocolError(f"unknown probe model kind: {kind}")


def train_action_phase_probe(
    dataset: ProbeDataset,
    *,
    kind: str,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden_dim: int = 256,
    max_train_samples: int = 100000,
    learning_rate: float = 3.0e-4,
    patience: int = 8,
    heldout_phase_interval: tuple[float, float] | None = None,
    predict_phase: bool = True,
    device: str = "cpu",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train a joint action/phase probe and return metrics plus serializable bundle."""

    torch = import_torch()
    from diagnostics.common.policy_class import BCTrainingConfig, train_behavior_clone

    train_mask = dataset.splits == "train"
    validation_mask = dataset.splits == "validation"
    test_mask = dataset.splits == "test"
    interval_mask = np.zeros(dataset.phase_normalized.shape, dtype=bool)
    if heldout_phase_interval is not None:
        start, stop = map(float, heldout_phase_interval)
        if not (0.0 <= start < stop <= 1.0):
            raise ProtocolError("held-out phase interval must be inside [0,1]")
        interval_mask = (dataset.phase_normalized >= start) & (dataset.phase_normalized < stop)
        train_mask &= ~interval_mask
        validation_mask &= ~interval_mask
        # This is the *separate* phase-generalization protocol: all samples in
        # the contiguous interval are hidden from optimization, while the
        # ordinary call above already provides the held-out-trajectory result.
        test_mask = interval_mask
    for name, mask in (("train", train_mask), ("validation", validation_mask), ("test", test_mask)):
        if int(mask.sum()) < 2:
            raise DependencyUnavailable(f"probe has fewer than two {name} samples")
    train_indices = deterministic_subsample(
        np.flatnonzero(train_mask), max_train_samples, seed=seed
    )
    validation_indices = deterministic_subsample(
        np.flatnonzero(validation_mask), max(2000, max_train_samples // 5), seed=seed + 1
    )
    test_indices = deterministic_subsample(
        np.flatnonzero(test_mask), max(5000, max_train_samples // 5), seed=seed + 2
    )
    x_mean, x_std = _fit_standardization(dataset.x[train_indices])
    action_mean = dataset.action[train_indices].mean(axis=0, keepdims=True).astype(np.float32)
    action_std = np.maximum(dataset.action[train_indices].std(axis=0, keepdims=True), 1.0e-6).astype(np.float32)

    def transform_x(indices: np.ndarray) -> np.ndarray:
        return ((dataset.x[indices] - x_mean) / x_std).astype(np.float32)

    def target(indices: np.ndarray) -> np.ndarray:
        normalized_action = (dataset.action[indices] - action_mean) / action_std
        if not predict_phase:
            return normalized_action.astype(np.float32)
        phase_angle = 2.0 * np.pi * dataset.phase_normalized[indices]
        phase_pair = np.stack((np.sin(phase_angle), np.cos(phase_angle)), axis=-1)
        return np.concatenate((normalized_action, phase_pair), axis=-1).astype(np.float32)

    output_dim = dataset.action.shape[1] + (2 if predict_phase else 0)
    model = _model_for(kind, dataset.x.shape[-1], output_dim, hidden_dim=hidden_dim)
    model, training = train_behavior_clone(
        model,
        transform_x(train_indices),
        target(train_indices),
        transform_x(validation_indices),
        target(validation_indices),
        seed=seed,
        config=BCTrainingConfig(
            epochs=int(epochs),
            batch_size=min(int(batch_size), int(train_indices.size)),
            learning_rate=float(learning_rate),
            patience=int(patience),
        ),
        device=device,
    )
    model.eval()
    with torch.no_grad():
        prediction = model(torch.as_tensor(transform_x(test_indices), device=device)).cpu().numpy()
    action_prediction = prediction[:, : dataset.action.shape[1]] * action_std + action_mean
    metrics = {
        "action": action_prediction_metrics(action_prediction, dataset.action[test_indices]),
        "phase": (
            circular_phase_metrics(
                np.arctan2(prediction[:, -2], prediction[:, -1]),
                dataset.phase_normalized[test_indices],
            )
            if predict_phase else None
        ),
        "training": training,
        "history": dataset.history,
        "model_kind": kind,
        "seed": int(seed),
        "train_count": int(train_indices.size),
        "validation_count": int(validation_indices.size),
        "test_count": int(test_indices.size),
        "heldout_phase_interval": list(heldout_phase_interval) if heldout_phase_interval else None,
    }
    bundle = {
        "bundle_version": 1,
        "architecture": kind,
        "history": int(dataset.history),
        "input_dim": int(dataset.x.shape[-1]),
        "action_dim": int(dataset.action.shape[1]),
        "hidden_dim": int(hidden_dim),
        "input_mean": torch.as_tensor(x_mean),
        "input_std": torch.as_tensor(x_std),
        "action_mean": torch.as_tensor(action_mean),
        "action_std": torch.as_tensor(action_std),
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "seed": int(seed),
        "predicts_phase": bool(predict_phase),
    }
    return metrics, bundle


def save_model_bundle(path: Path, bundle: Mapping[str, Any]) -> None:
    torch = import_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite model bundle: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(bundle), temporary)
    temporary.replace(path)


class ReferenceFreeBCInference:
    """Closed-loop adapter for saved diag-24 action/phase bundles."""

    def __init__(self, bundle: Mapping[str, Any], *, device: str = "cpu") -> None:
        torch = import_torch()
        self.torch = torch
        self.device = torch.device(device)
        self.history = int(bundle["history"])
        self.action_dim = int(bundle["action_dim"])
        self.input_dim = int(bundle["input_dim"])
        self.predicts_phase = bool(bundle.get("predicts_phase", False))
        self.model = _model_for(
            str(bundle["architecture"]),
            self.input_dim,
            self.action_dim + (2 if self.predicts_phase else 0),
            hidden_dim=int(bundle["hidden_dim"]),
        ).to(self.device)
        self.model.load_state_dict(bundle["state_dict"], strict=True)
        self.model.eval()
        self.input_mean = torch.as_tensor(bundle["input_mean"], device=self.device)
        self.input_std = torch.as_tensor(bundle["input_std"], device=self.device)
        self.action_mean = torch.as_tensor(bundle["action_mean"], device=self.device)
        self.action_std = torch.as_tensor(bundle["action_std"], device=self.device)
        self._history: list[Any] = []

    def reset(self) -> None:
        self._history.clear()

    def act(self, actor_no_reference) -> Any:
        torch = self.torch
        observation = torch.as_tensor(actor_no_reference, dtype=torch.float32, device=self.device)
        if observation.ndim != 2 or observation.shape[1] != self.input_dim:
            raise ProtocolError(
                f"BC adapter expected [B,{self.input_dim}], got {tuple(observation.shape)}"
            )
        self._history.append(observation)
        self._history = self._history[-self.history :]
        padded = [self._history[0]] * (self.history - len(self._history)) + self._history
        window = torch.stack(padded, dim=1)
        model_input = window[:, -1] if self.history == 1 else window
        # Stored transform dimensions are [1,D] for MLP and [1,1,D] for GRU;
        # normal broadcasting handles both layouts.
        normalized = (model_input - self.input_mean) / self.input_std
        with torch.no_grad():
            prediction = self.model(normalized)[..., : self.action_dim]
        return prediction * self.action_std + self.action_mean


def load_bc_inference(path: Path, *, device: str = "cpu") -> ReferenceFreeBCInference:
    torch = import_torch()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, Mapping) or payload.get("bundle_version") != 1:
        raise ProtocolError(f"invalid BC model bundle: {path}")
    return ReferenceFreeBCInference(payload, device=device)


def real_isaac_helper(operation: str):
    """Return a stage-1 real-simulator operation, never an offline surrogate.

    Stage 1 owns simulator construction and exact snapshot restoration.  We
    intentionally use capability discovery so stage 2 remains importable on a
    CPU-only analysis host while accepting either a module-level helper or the
    agreed ``RealIsaacCanonicalSession`` adapter.
    """

    try:
        module = importlib.import_module("diagnostics.common.canonical_collection")
    except ImportError as exc:
        raise DependencyUnavailable("stage-1 canonical Isaac helper is unavailable") from exc
    direct_names = {
        "open_loop_teacher_replay": ("run_open_loop_teacher_replay", "open_loop_teacher_replay"),
        "reference_ablation": ("run_reference_ablation_rollouts", "reference_ablation_rollouts"),
        "bc_closed_loop": ("run_reference_free_bc_rollout", "reference_free_bc_rollout"),
    }
    for name in direct_names.get(operation, ()):
        function = getattr(module, name, None)
        if callable(function):
            return function
    session = getattr(module, "RealIsaacCanonicalSession", None)
    if session is not None and hasattr(session, operation):
        return (session, operation)
    raise DependencyUnavailable(
        f"stage-1 canonical helper does not expose real Isaac operation {operation!r}"
    )


def invoke_real_isaac(operation: str, **kwargs: Any) -> Mapping[str, Any]:
    helper = real_isaac_helper(operation)
    if isinstance(helper, tuple):
        session_type, method_name = helper
        # A session constructor may accept only a subset of the standard
        # context.  Passing by signature keeps the contract explicit.
        signature = inspect.signature(session_type)
        constructor = {key: value for key, value in kwargs.items() if key in signature.parameters}
        session = session_type(**constructor)
        method = getattr(session, method_name)
        signature = inspect.signature(method)
        call = {key: value for key, value in kwargs.items() if key in signature.parameters}
        result = method(**call)
    else:
        signature = inspect.signature(helper)
        call = {key: value for key, value in kwargs.items() if key in signature.parameters}
        result = helper(**call)
    if not isinstance(result, Mapping):
        raise ProtocolError(f"real Isaac operation {operation} returned a non-mapping")
    return dict(result)


def validate_external_intervention_result(
    path: Path,
    *,
    operation: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ProtocolError("external intervention result must be a JSON object")
    if value.get("operation") != operation:
        raise ProtocolError(
            f"external intervention operation mismatch: expected {operation!r}"
        )
    if value.get("checkpoint_sha256") != checkpoint_sha256:
        raise ProtocolError("external intervention used a different checkpoint")
    if value.get("same_snapshot_replay_verified") is not True:
        raise ProtocolError("external intervention lacks exact same-snapshot replay verification")
    metrics = value.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ProtocolError("external intervention has no metrics mapping")
    return value


def _ordered_real_rows(rows: Any, snapshot_ids: Sequence[str]) -> list[dict[str, Any]]:
    records = rows.to_dict(orient="records") if hasattr(rows, "to_dict") else list(rows)
    by_snapshot: dict[str, dict[str, Any]] = {}
    for raw in records:
        row = dict(raw)
        snapshot = str(row["snapshot_id"])
        if snapshot in by_snapshot:
            raise ProtocolError(f"multiple primary trajectories exist for snapshot {snapshot}")
        by_snapshot[snapshot] = row
    missing = [snapshot for snapshot in snapshot_ids if snapshot not in by_snapshot]
    extra = sorted(set(by_snapshot) - set(snapshot_ids))
    if missing or extra:
        raise ProtocolError(
            f"canonical rows and snapshot bank differ; missing={missing}, extra={extra}"
        )
    return [by_snapshot[snapshot] for snapshot in snapshot_ids]


def _stack_real_field(
    index_path: Path,
    rows: Sequence[Mapping[str, Any]],
    section: str,
    field: str,
    *,
    steps: int,
):
    torch = import_torch()
    values = []
    cache: dict[str, dict[str, Any]] = {}
    from diagnostics.common.canonical_collection import load_rollout_trajectory

    for row in rows:
        trajectory = load_rollout_trajectory(index_path, row, cache=cache)
        value = trajectory.get(section, {}).get(field)
        if not torch.is_tensor(value) or value.shape[0] < steps:
            raise DependencyUnavailable(
                f"canonical {section}.{field} lacks {steps} valid steps for {row['snapshot_id']}"
            )
        values.append(value[:steps])
    result = torch.stack(values, dim=1)
    if result.is_floating_point() and not bool(torch.isfinite(result).all()):
        raise ProtocolError(f"canonical {section}.{field} contains NaN or Inf")
    return result


def _real_checkpoint_record(row: Mapping[str, Any]):
    from diagnostics.common.canonical_collection import DenseCheckpointRecord

    path = Path(str(row["checkpoint_path"])).expanduser().resolve()
    if not path.is_file():
        raise DependencyUnavailable(f"primary checkpoint entity is missing: {path}")
    return DenseCheckpointRecord(
        checkpoint_id=str(row["checkpoint_id"]),
        path=path,
        sha256=str(row["checkpoint_sha256"]),
        update=int(row["checkpoint_update"]),
        lineage_id=str(row["checkpoint_lineage_id"]),
    )


def _real_context(
    simulation_app: Any,
    *,
    repo_root: Path,
    output_dir: Path,
    index_path: Path,
    rows: Any,
    spec: Mapping[str, Any],
    runtime_name: str,
):
    from diagnostics.common.canonical_collection import (
        CanonicalCollectionProtocol,
        make_collection_trainer,
        switch_policy_state,
    )
    from diagnostics.common.snapshot_bank import SnapshotBank

    collection = CanonicalCollectionProtocol.from_spec(spec)
    bank_path = output_dir / "snapshot_bank.pt"
    if not bank_path.is_file():
        raise DependencyUnavailable(f"canonical snapshot bank is missing: {bank_path}")
    bank = SnapshotBank.load(bank_path)
    ordered = _ordered_real_rows(rows, bank.snapshot_ids)
    checkpoint = _real_checkpoint_record(ordered[0])
    trainer = make_collection_trainer(
        simulation_app,
        repo_root=repo_root,
        protocol=collection,
        runtime_dir=output_dir / "runtime" / runtime_name,
    )
    switch_policy_state(trainer, checkpoint)
    return trainer, bank, ordered, collection, checkpoint


def _branch_outcomes(
    trainer: Any,
    *,
    horizon: int,
    action_function: Any,
    observation_transform: Any | None = None,
) -> tuple[dict[str, Any], list[Any]]:
    """Run one real branch and summarize first-episode outcomes only."""

    torch = import_torch()
    env = trainer.env
    active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    failed = torch.zeros_like(active)
    completed = torch.zeros_like(active)
    returns = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
    lengths = torch.full(
        (env.num_envs,), int(horizon), dtype=torch.long, device=env.device
    )
    actions: list[Any] = []
    observation = env.get_observation()
    checkpoints = tuple(value for value in (1, 5, 10, 25, 50, horizon) if value <= horizon)
    by_horizon: dict[str, Any] = {}
    for step in range(horizon):
        policy_observation = (
            observation_transform(step, observation) if observation_transform is not None else observation
        )
        action = action_function(step, policy_observation)
        if action.shape != (env.num_envs, env.action_dim):
            raise ProtocolError(
                f"real branch action has wrong shape {tuple(action.shape)}"
            )
        actions.append(action.detach().cpu().clone())
        next_observation, reward, done, info = env.step(action)
        returns[active] += reward[active]
        done_terms = info["done_terms"]
        step_failure = (
            done_terms["anchor_pos_bad"]
            | done_terms["anchor_ori_bad"]
            | done_terms["ee_body_bad"]
        )
        first_done = active & done.bool()
        failed[first_done] = step_failure[first_done]
        completed[first_done] = done_terms["motion_complete"][first_done]
        lengths[first_done] = step + 1
        active[first_done] = False
        observation = next_observation
        if step + 1 in checkpoints:
            by_horizon[str(step + 1)] = {
                "failure_rate": float(failed.float().mean().item()),
                "motion_completion": float(completed.float().mean().item()),
                "still_active_fraction": float(active.float().mean().item()),
                "return_mean": float(returns.mean().item()),
            }
    return {
        "horizon": int(horizon),
        "motion_completion": float(completed.float().mean().item()),
        "failure_rate": float(failed.float().mean().item()),
        "still_active_fraction": float(active.float().mean().item()),
        "steps_mean": float(lengths.float().mean().item()),
        "return_mean": float(returns.mean().item()),
        "by_horizon": by_horizon,
    }, actions


def _apply_joint_rademacher_perturbation(
    env: Any,
    bank: Any,
    *,
    fraction: float,
    seed: int,
) -> None:
    torch = import_torch()
    from diagnostics.common.noise_bank import NoiseBank

    joint_pos, joint_vel = env.get_action_joint_state()
    limits = env.robot.data.soft_joint_pos_limits[:, env.action_joint_ids]
    joint_range = limits[..., 1] - limits[..., 0]
    noise = NoiseBank(seed=seed).controlled_uniform(
        bank.snapshot_ids,
        horizon=1,
        width=int(env.action_dim),
        stream="diag25_joint_position_rademacher",
        low=0.0,
        high=1.0,
        device=env.device,
    )[:, 0]
    direction = torch.where(noise >= 0.5, torch.ones_like(noise), -torch.ones_like(noise))
    perturbed = torch.clamp(
        joint_pos + float(fraction) * joint_range * direction,
        limits[..., 0],
        limits[..., 1],
    )
    root_pose = env.robot.data.root_link_pose_w
    root_velocity = env.get_mimic_root_velocity_w()
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env._write_robot_state(
        root_pos=root_pose[:, :3] - env.scene.env_origins,
        root_quat=root_pose[:, 3:7],
        root_lin_vel=root_velocity[:, :3],
        root_ang_vel=root_velocity[:, 3:],
        joint_pos=perturbed,
        joint_vel=joint_vel,
        env_ids=env_ids,
    )
    env.scene.update(env.physics_dt)


def run_open_loop_teacher_replay_real(
    simulation_app: Any,
    *,
    repo_root: Path,
    output_dir: Path,
    index_path: Path,
    rows: Any,
    spec: Mapping[str, Any],
    protocol: PolicyClassProtocol,
) -> dict[str, Any]:
    """Real PhysX diag-25 operation using the stage-1 persistent-env protocol."""

    torch = import_torch()
    from diagnostics.common.canonical_collection import (
        close_collection_trainer,
        collector_environment,
        policy_mean_and_std,
        restore_snapshot_bank,
    )
    from diagnostics.common.noise_bank import NoiseBank
    from engine.env_state import restore_env_state, snapshot_env_state

    trainer = None
    try:
        trainer, bank, ordered, collection, checkpoint = _real_context(
            simulation_app,
            repo_root=repo_root,
            output_dir=output_dir,
            index_path=index_path,
            rows=rows,
            spec=spec,
            runtime_name="diag_25",
        )
        horizon = protocol.open_loop_horizon
        if collection.horizon < horizon:
            raise DependencyUnavailable(
                f"canonical horizon {collection.horizon} is shorter than frozen open-loop horizon {horizon}"
            )
        canonical_mean = _stack_real_field(
            index_path, ordered, "action", "mean", steps=horizon
        ).to(trainer.env.device)
        canonical_applied = _stack_real_field(
            index_path, ordered, "action", "applied", steps=horizon
        ).to(trainer.env.device)
        env = trainer.env
        noise = NoiseBank(seed=collection.collector_seed)

        # Exact baseline: action identity throughout a deterministic replay.
        restore_snapshot_bank(env, bank, mode="clean_mean")
        max_action_error = 0.0
        with collector_environment(
            env, noise, bank.snapshot_ids, mode="clean_mean", horizon=horizon
        ):
            observation = env.get_observation()
            for step in range(horizon):
                mean, _ = policy_mean_and_std(trainer.algo, observation)
                max_action_error = max(
                    max_action_error,
                    float(torch.max(torch.abs(mean - canonical_mean[step])).item()),
                )
                observation, _, _, _ = env.step(canonical_applied[step])
        exact = {
            "action_max_abs_error": max_action_error,
            "action_atol": protocol.exact_replay_action_atol,
            "action_identity": max_action_error <= protocol.exact_replay_action_atol,
            "role": "deterministic numerical identity sanity only",
        }
        if not exact["action_identity"]:
            raise ProtocolError(
                f"exact replay action identity failed: {max_action_error}"
            )

        perturbations: dict[str, Any] = {}
        for fraction in protocol.open_loop_perturbation_fractions:
            restore_snapshot_bank(env, bank, mode="clean_mean")
            _apply_joint_rademacher_perturbation(
                env, bank, fraction=fraction, seed=protocol.seeds[0]
            )
            perturbed_snapshot = snapshot_env_state(env)
            with collector_environment(
                env, noise, bank.snapshot_ids, mode="clean_mean", horizon=horizon
            ):
                open_loop, _ = _branch_outcomes(
                    trainer,
                    horizon=horizon,
                    action_function=lambda step, observation: canonical_applied[step],
                )
            open_final = snapshot_env_state(env)
            restore_env_state(env, perturbed_snapshot)
            with collector_environment(
                env, noise, bank.snapshot_ids, mode="clean_mean", horizon=horizon
            ):
                feedback, feedback_actions = _branch_outcomes(
                    trainer,
                    horizon=horizon,
                    action_function=lambda step, observation: trainer.algo.deterministic_action(observation),
                )
            feedback_final = snapshot_env_state(env)
            feedback_action_delta = torch.stack(
                [
                    torch.linalg.vector_norm(
                        feedback_actions[step].to(env.device) - canonical_applied[step],
                        dim=-1,
                    ) / math.sqrt(float(env.action_dim))
                    for step in range(horizon)
                ]
            )
            final_joint_delta = (
                open_final["joint_pos"].index_select(1, env.action_joint_ids)
                - feedback_final["joint_pos"].index_select(1, env.action_joint_ids)
            )
            final_root_delta = open_final["root_pose_w"][:, :3] - feedback_final["root_pose_w"][:, :3]
            perturbations[str(fraction)] = {
                "fraction_of_joint_range": fraction,
                "open_loop_recorded_nominal_actions": open_loop,
                "teacher_feedback": feedback,
                "feedback_minus_open_loop": {
                    key: float(feedback[key]) - float(open_loop[key])
                    for key in ("motion_completion", "failure_rate", "steps_mean", "return_mean")
                },
                "feedback_action_deviation_from_nominal": {
                    "mean": float(feedback_action_delta.mean().item()),
                    "p95": float(torch.quantile(feedback_action_delta.flatten(), 0.95).item()),
                    "max": float(feedback_action_delta.max().item()),
                },
                "final_feedback_vs_open_loop_state": {
                    "joint_position_rmse": float(torch.sqrt(final_joint_delta.square().mean()).item()),
                    "root_position_rmse": float(torch.sqrt(final_root_delta.square().mean()).item()),
                },
            }
        return {
            "operation": "open_loop_teacher_replay",
            "checkpoint_sha256": checkpoint.sha256,
            "same_snapshot_replay_verified": True,
            "uses_recorded_applied_actions": True,
            "metrics": {
                "exact_identity": exact,
                "perturbed_comparison": perturbations,
            },
        }
    finally:
        if trainer is not None:
            close_collection_trainer(trainer)


def run_reference_ablation_real(
    simulation_app: Any,
    *,
    repo_root: Path,
    output_dir: Path,
    index_path: Path,
    rows: Any,
    spec: Mapping[str, Any],
    protocol: PolicyClassProtocol,
    partition: Mapping[str, Any],
) -> dict[str, Any]:
    """Real PhysX diag-26 action-input intervention branches."""

    torch = import_torch()
    from diagnostics.common.canonical_collection import (
        close_collection_trainer,
        collector_environment,
        restore_snapshot_bank,
    )
    from diagnostics.common.noise_bank import NoiseBank

    trainer = None
    try:
        trainer, bank, ordered, collection, checkpoint = _real_context(
            simulation_app,
            repo_root=repo_root,
            output_dir=output_dir,
            index_path=index_path,
            rows=rows,
            spec=spec,
            runtime_name="diag_26",
        )
        env = trainer.env
        horizon = collection.horizon
        noise = NoiseBank(seed=collection.collector_seed)
        reference_indices = torch.as_tensor(
            named_term_indices(partition, role="reference"),
            dtype=torch.long,
            device=env.device,
        )
        nonreference = torch.as_tensor(
            named_term_indices(partition, role="proprio"),
            dtype=torch.long,
            device=env.device,
        )
        permutation_unit = noise.controlled_uniform(
            bank.snapshot_ids,
            horizon=1,
            width=1,
            stream="diag26_reference_shuffle_permutation",
            low=0.0,
            high=1.0,
            device=env.device,
        )[:, 0, 0]
        permutation = torch.argsort(permutation_unit)

        def run(kind: str, delay: int = 0) -> dict[str, Any]:
            restore_snapshot_bank(env, bank, mode="clean_mean")
            history: list[Any] = []
            frozen: Any | None = None
            nonreference_error = 0.0

            def transform(step: int, observation: Any):
                nonlocal frozen, nonreference_error
                current_reference = observation.index_select(1, reference_indices)
                history.append(current_reference.detach().clone())
                if frozen is None:
                    frozen = current_reference.detach().clone()
                if kind == "baseline":
                    replacement = current_reference
                elif kind == "freeze":
                    replacement = frozen
                elif kind == "shuffle":
                    replacement = current_reference.index_select(0, permutation)
                elif kind == "delay":
                    replacement = history[max(0, len(history) - 1 - int(delay))]
                else:  # pragma: no cover - closed enumeration below
                    raise ProtocolError(f"unknown reference intervention {kind}")
                modified = observation.clone()
                modified[:, reference_indices] = replacement
                nonreference_error = max(
                    nonreference_error,
                    float(
                        torch.max(
                            torch.abs(
                                modified.index_select(1, nonreference)
                                - observation.index_select(1, nonreference)
                            )
                        ).item()
                    ),
                )
                return modified

            with collector_environment(
                env, noise, bank.snapshot_ids, mode="clean_mean", horizon=horizon
            ):
                metrics, _ = _branch_outcomes(
                    trainer,
                    horizon=horizon,
                    action_function=lambda step, observation: trainer.algo.deterministic_action(observation),
                    observation_transform=transform,
                )
            metrics["nonreference_max_abs_change"] = nonreference_error
            if nonreference_error != 0.0:
                raise ProtocolError("reference intervention changed a proprio slice")
            return metrics

        metrics = {
            "baseline": run("baseline"),
            "freeze": run("freeze"),
            "shuffle": run("shuffle"),
            "delay": {str(delay): run("delay", delay) for delay in protocol.ablation_delays},
        }
        return {
            "operation": "reference_ablation",
            "checkpoint_sha256": checkpoint.sha256,
            "same_snapshot_replay_verified": True,
            "only_reference_terms_intervened": True,
            "metrics": metrics,
        }
    finally:
        if trainer is not None:
            close_collection_trainer(trainer)


class _ReferenceFreeBCPolicyAdapter:
    """Stage-1 canonical-collector adapter for the preregistered B policy."""

    policy_domain = "reference_free_B"

    def __init__(
        self,
        model_path: Path,
        *,
        no_reference_indices: Sequence[int],
        device: str,
        manifest: Mapping[str, Any],
        protocol: PolicyClassProtocol,
    ) -> None:
        torch = import_torch()
        try:
            bundle = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:
            bundle = torch.load(model_path, map_location="cpu")
        if not isinstance(bundle, Mapping):
            raise ProtocolError("preregistered B model is not a BC bundle")
        expected = {
            "bundle_version": 1,
            "architecture": protocol.b_model.lower(),
            "history": protocol.b_history,
            "seed": protocol.b_seed,
            "predicts_phase": False,
            "input_view": "actor_no_reference",
        }
        mismatches = {
            key: {"expected": value, "actual": bundle.get(key)}
            for key, value in expected.items()
            if bundle.get(key) != value
        }
        if mismatches:
            raise ProtocolError(f"preregistered B model bundle mismatch: {mismatches}")
        self.model_path = model_path.expanduser().resolve()
        self.model_sha256 = sha256_file(self.model_path)
        self.policy = load_bc_inference(self.model_path, device=device)
        self.indices = torch.as_tensor(
            tuple(int(value) for value in no_reference_indices),
            dtype=torch.long,
            device=device,
        )
        if int(self.indices.numel()) != self.policy.input_dim:
            raise ProtocolError(
                "preregistered B model input width differs from the AST no-reference view"
            )
        self.manifest = dict(manifest)

    def reset_branch(self, *, env: Any, mode: Any, snapshot_ids: Sequence[str]) -> None:
        del env, mode
        if not snapshot_ids:
            raise ProtocolError("B-domain branch has no snapshot identities")
        self.policy.reset()

    def action_record(
        self,
        *,
        env: Any,
        observation: Any,
        imitation: Mapping[str, Any],
        common_epsilon: Any,
        mode: Any,
        common_sigma: float,
        action_low: Any,
        action_high: Any,
        step: int,
    ) -> Any:
        del step
        from diagnostics.common.rollout_collector import construct_action_record

        required_imitation = {
            "agent_physx_raw_frame",
            "agent_fk_aligned_raw_frame",
            "reference_expert_raw_frame",
            "phase_normalized_pre_step",
        }
        if set(imitation) != required_imitation:
            raise ProtocolError("B policy did not receive the exact Stage-1 imitation contract")
        no_reference = observation.index_select(1, self.indices.to(observation.device))
        mean = self.policy.act(no_reference)
        if tuple(mean.shape) != (int(env.num_envs), int(env.action_dim)):
            raise ProtocolError(f"B policy returned wrong action shape {tuple(mean.shape)}")
        native_std = mean.new_zeros(mean.shape)
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
            "policy_method": "reference_free_bc_gru_h32",
            "policy_source_commit": str(self.manifest["git_commit"]),
            "policy_source_snapshot_sha256": str(
                self.manifest["source_snapshot_sha256"]
            ),
            "policy_resolved_config_sha256": str(
                self.manifest["resolved_config_sha256"]
            ),
        }


def _preregistered_b_model_path(
    model_paths: Sequence[Path], protocol: PolicyClassProtocol
) -> Path:
    expected_stem = f"gru_h{protocol.b_history}_seed_{protocol.b_seed}"
    selected = [Path(path).expanduser().resolve() for path in model_paths if Path(path).stem == expected_stem]
    if len(selected) != 1:
        raise ProtocolError(
            f"expected exactly one preregistered B model {expected_stem!r}, found {len(selected)}"
        )
    return selected[0]


def validate_reference_free_b_index(
    index_path: Path,
    *,
    protocol: PolicyClassProtocol,
    model_path: Path,
    expected_snapshot_ids: Sequence[str],
) -> dict[str, Any]:
    """Validate the complete preregistered B collector product, including shards."""

    from diagnostics.common.canonical_collection import (
        load_rollout_index,
        load_rollout_shard,
        resolve_shard_path,
        validate_canonical_rollout_tree,
    )

    records = load_rollout_index(index_path)
    expected_model = model_path.expanduser().resolve()
    model_sha = sha256_file(expected_model)
    expected_checkpoint_id = f"GRU-H{protocol.b_history}-seed{protocol.b_seed}"
    expected_branches = {
        ("clean_mean", 0.0),
        ("controlled_environment", 0.0),
        *(("common_action_noise", float(sigma)) for sigma in protocol.b_common_sigmas),
    }
    snapshots = tuple(str(value) for value in expected_snapshot_ids)
    if not snapshots or len(set(snapshots)) != len(snapshots):
        raise ProtocolError("expected B snapshot identities are empty or duplicated")
    observed_pairs: set[tuple[str, str, float]] = set()
    shard_paths: set[Path] = set()
    for row in records:
        branch = (str(row["collector_mode"]), float(row["common_sigma"]))
        if branch not in expected_branches:
            raise ProtocolError(f"B index contains an unfrozen collector branch: {branch}")
        if str(row.get("policy_domain")) != "reference_free_B":
            raise ProtocolError("B index policy_domain is not reference_free_B")
        expected_fields = {
            "checkpoint_id": expected_checkpoint_id,
            "checkpoint_sha256": model_sha,
            "checkpoint_update": protocol.primary_update,
            "checkpoint_lineage_id": "diag24-bc-v1",
        }
        mismatches = {
            key: {"expected": value, "actual": row.get(key)}
            for key, value in expected_fields.items()
            if str(row.get(key)) != str(value)
        }
        if mismatches:
            raise ProtocolError(f"B index model identity mismatch: {mismatches}")
        if Path(str(row["checkpoint_path"])).expanduser().resolve() != expected_model:
            raise ProtocolError("B index points to a different model bundle")
        if bool(row["eligible_for_primary_overlap"]) is not True:
            raise ProtocolError("an overlap-eligible B branch was marked ineligible")
        snapshot = str(row["snapshot_id"])
        key = (snapshot, branch[0], branch[1])
        if key in observed_pairs:
            raise ProtocolError(f"B index duplicates condition {key}")
        observed_pairs.add(key)
        shard_paths.add(resolve_shard_path(index_path, row))
    expected_pairs = {
        (snapshot, mode, sigma)
        for snapshot in snapshots
        for mode, sigma in expected_branches
    }
    if observed_pairs != expected_pairs:
        missing = sorted(expected_pairs - observed_pairs)
        extra = sorted(observed_pairs - expected_pairs)
        raise ProtocolError(
            f"B index does not cover the snapshot/branch product; missing={missing[:5]}, extra={extra[:5]}"
        )
    for shard_path in sorted(shard_paths):
        tree = load_rollout_shard(shard_path)
        validate_canonical_rollout_tree(tree)
        metadata = tree["metadata"]
        if (
            metadata.get("policy_domain") != "reference_free_B"
            or metadata.get("checkpoint_sha256") != model_sha
            or metadata.get("checkpoint_lineage_id") != "diag24-bc-v1"
        ):
            raise ProtocolError(f"B shard metadata identity mismatch: {shard_path}")
    return {
        "index_path": str(index_path.expanduser().resolve()),
        "index_sha256": sha256_file(index_path),
        "model_path": str(expected_model),
        "model_sha256": model_sha,
        "policy_domain": "reference_free_B",
        "checkpoint_id": expected_checkpoint_id,
        "checkpoint_lineage_id": "diag24-bc-v1",
        "snapshot_count": len(snapshots),
        "trajectory_count": len(records),
        "branch_count": len(expected_branches),
        "branches": [
            {"collector_mode": mode, "common_sigma": sigma}
            for mode, sigma in sorted(expected_branches)
        ],
        "exact_stage1_imitation_contract": True,
    }


def _collect_reference_free_b_domain(
    trainer: Any,
    bank: Any,
    *,
    repo_root: Path,
    output_dir: Path,
    spec: Mapping[str, Any],
    protocol: PolicyClassProtocol,
    partition: Mapping[str, Any],
    model_path: Path,
    collection: Any,
) -> dict[str, Any]:
    from diagnostics.common.canonical_collection import (
        DenseCheckpointRecord,
        branch_shard_relative,
        collect_canonical_branch,
        save_branch_shard,
        write_rollout_index,
    )
    from diagnostics.common.noise_bank import CollectorMode

    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise DependencyUnavailable(f"frozen suite manifest is missing: {manifest_path}")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "PASS":
        raise DependencyUnavailable("diag_00 manifest must PASS before B-domain collection")
    adapter = _ReferenceFreeBCPolicyAdapter(
        model_path,
        no_reference_indices=named_term_indices(partition, role="proprio"),
        device=str(trainer.env.device),
        manifest=manifest,
        protocol=protocol,
    )
    record = DenseCheckpointRecord(
        checkpoint_id=f"GRU-H{protocol.b_history}-seed{protocol.b_seed}",
        path=model_path.expanduser().resolve(),
        sha256=adapter.model_sha256,
        update=protocol.primary_update,
        lineage_id="diag24-bc-v1",
        policy_domain="reference_free_B",
    )
    variants = (
        (CollectorMode.CLEAN_MEAN, 0.0),
        (CollectorMode.CONTROLLED_ENVIRONMENT, 0.0),
        *tuple(
            (CollectorMode.COMMON_ACTION_NOISE, float(sigma))
            for sigma in protocol.b_common_sigmas
        ),
    )
    if tuple(mode.value for mode, _ in variants[:2]) != protocol.b_collector_modes[:2]:
        raise ProtocolError("B collector mode order differs from the frozen protocol")
    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for mode, sigma in variants:
        tree, rows, summary = collect_canonical_branch(
            trainer,
            bank,
            record,
            protocol=collection,
            manifest=manifest,
            spec=spec,
            repo_root=repo_root,
            mode=mode,
            common_sigma=sigma,
            policy_adapter=adapter,
        )
        expected_relative = branch_shard_relative(record, mode, sigma)
        _, relative = save_branch_shard(
            tree,
            output_dir=output_dir,
            checkpoint=record,
            mode=mode,
            common_sigma=sigma,
        )
        if relative != expected_relative.as_posix():
            raise ProtocolError("B branch shard naming changed during save")
        for row in rows:
            row["shard_path"] = relative
        all_rows.extend(rows)
        summaries[f"{mode.value}:sigma={sigma:g}"] = summary
    b_index = output_dir / protocol.b_output_name
    write_rollout_index(all_rows, b_index)
    validated = validate_reference_free_b_index(
        b_index,
        protocol=protocol,
        model_path=model_path,
        expected_snapshot_ids=bank.snapshot_ids,
    )
    return {**validated, "summaries": summaries}


def run_reference_free_bc_real(
    simulation_app: Any,
    *,
    repo_root: Path,
    output_dir: Path,
    index_path: Path,
    rows: Any,
    spec: Mapping[str, Any],
    protocol: PolicyClassProtocol,
    partition: Mapping[str, Any],
    model_paths: Sequence[Path],
) -> dict[str, Any]:
    """Run saved diag-24 policies from the canonical clean snapshot bank."""

    torch = import_torch()
    from diagnostics.common.canonical_collection import (
        close_collection_trainer,
        collector_environment,
        restore_snapshot_bank,
    )
    from diagnostics.common.noise_bank import NoiseBank

    trainer = None
    try:
        trainer, bank, ordered, collection, checkpoint = _real_context(
            simulation_app,
            repo_root=repo_root,
            output_dir=output_dir,
            index_path=index_path,
            rows=rows,
            spec=spec,
            runtime_name="diag_24",
        )
        env = trainer.env
        horizon = collection.horizon
        noise = NoiseBank(seed=collection.collector_seed)
        no_reference_indices = torch.as_tensor(
            named_term_indices(partition, role="proprio"),
            dtype=torch.long,
            device=env.device,
        )
        b_model_path = _preregistered_b_model_path(model_paths, protocol)
        restore_snapshot_bank(env, bank, mode="clean_mean")
        with collector_environment(
            env, noise, bank.snapshot_ids, mode="clean_mean", horizon=horizon
        ):
            teacher, _ = _branch_outcomes(
                trainer,
                horizon=horizon,
                action_function=lambda step, observation: trainer.algo.deterministic_action(observation),
            )
        models: dict[str, Any] = {}
        for model_path in model_paths:
            policy = load_bc_inference(Path(model_path), device=str(env.device))
            policy.reset()
            restore_snapshot_bank(env, bank, mode="clean_mean")
            with collector_environment(
                env, noise, bank.snapshot_ids, mode="clean_mean", horizon=horizon
            ):
                metrics, _ = _branch_outcomes(
                    trainer,
                    horizon=horizon,
                    action_function=lambda step, observation, policy=policy: policy.act(
                        observation.index_select(1, no_reference_indices)
                    ),
                )
            stem = Path(model_path).stem
            # Saved names are mlp_h1_seed_... / gru_hN_seed_....
            parts = stem.split("_seed_")
            prefix, seed = parts if len(parts) == 2 else (stem, "unknown")
            model_label = prefix.upper().replace("_", "-")
            models[f"{model_label}:seed={seed}"] = metrics
        b_domain = _collect_reference_free_b_domain(
            trainer,
            bank,
            repo_root=repo_root,
            output_dir=output_dir,
            spec=spec,
            protocol=protocol,
            partition=partition,
            model_path=b_model_path,
            collection=collection,
        )
        return {
            "operation": "bc_closed_loop",
            "checkpoint_sha256": checkpoint.sha256,
            "same_snapshot_replay_verified": True,
            "metrics": {"teacher_baseline": teacher, "models": models},
            "b_domain": b_domain,
        }
    finally:
        if trainer is not None:
            close_collection_trainer(trainer)


__all__ = [
    "BC_HISTORIES",
    "CANONICAL_INDEX_NAME",
    "CanonicalArrays",
    "FrozenCheckpointPolicy",
    "HISTORY_LENGTHS",
    "PRIMARY_POLICY_CLASS_MODE",
    "PolicyClassProtocol",
    "ProbeDataset",
    "ReferenceFreeBCInference",
    "action_prediction_metrics",
    "build_history_end_indices",
    "canonical_index",
    "circular_phase_metrics",
    "derive_repository_observation_specs",
    "deterministic_group_splits",
    "deterministic_subsample",
    "history_windows",
    "invoke_real_isaac",
    "load_bc_inference",
    "load_canonical_arrays",
    "load_observation_partition",
    "make_probe_dataset",
    "named_term_indices",
    "normalized_phase",
    "output_dir_from_args",
    "phase_period",
    "resolve_checkpoint",
    "require_primary_teacher_quality",
    "run_open_loop_teacher_replay_real",
    "run_reference_ablation_real",
    "run_reference_free_bc_real",
    "validate_reference_free_b_index",
    "save_model_bundle",
    "select_policy_class_rows",
    "train_action_phase_probe",
    "validate_external_intervention_result",
]
