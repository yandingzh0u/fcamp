"""Read-only checkpoint inventory and lineage helpers.

Checkpoint files are pickle containers.  The diagnostic therefore uses
PyTorch's restricted ``weights_only`` loader, never imports repository model
classes, and records both a file hash and a structural summary.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from diagnostics.common.manifest import (
    DependencyUnavailable,
    ProtocolError,
    canonical_sha256,
    read_json,
    sha256_file,
)


_UPDATE_RE = re.compile(r"(?:update|checkpoint)[_-]?(\d+)", re.IGNORECASE)


def import_torch():
    try:
        import torch
    except ImportError as exc:
        raise DependencyUnavailable(
            "PyTorch is required to inspect checkpoint payloads; run in env_isaaclab"
        ) from exc
    return torch


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    torch = import_torch()
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.6
        payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict):
        raise ProtocolError(f"checkpoint payload must be a mapping: {source}")
    return payload


def update_from_filename(path: str | Path) -> int | None:
    match = _UPDATE_RE.search(Path(path).stem)
    return int(match.group(1)) if match else None


def _tensor_bytes(value: Any) -> bytes:
    torch = import_torch()
    if not torch.is_tensor(value):
        raise TypeError(type(value))
    tensor = value.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    try:
        body = tensor.numpy().tobytes(order="C")
    except (TypeError, RuntimeError):
        body = bytes(tensor.view(torch.uint8).tolist())
    return header + b"\0" + body


def state_digest(value: Any) -> str:
    """Stable structural/content digest for nested checkpoint state."""

    torch = import_torch()
    digest = hashlib.sha256()

    def visit(node: Any) -> None:
        if torch.is_tensor(node):
            digest.update(b"tensor\0")
            digest.update(_tensor_bytes(node))
        elif isinstance(node, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(node, key=lambda item: str(item)):
                digest.update(str(key).encode("utf-8", "surrogateescape"))
                digest.update(b"\0")
                visit(node[key])
        elif isinstance(node, (list, tuple)):
            digest.update(type(node).__name__.encode("ascii") + b"\0")
            for item in node:
                visit(item)
        elif isinstance(node, (str, int, float, bool)) or node is None:
            digest.update(repr(node).encode("utf-8"))
            digest.update(b"\0")
        else:
            digest.update(type(node).__qualname__.encode("utf-8"))
            digest.update(b"\0")
            digest.update(repr(node).encode("utf-8", "backslashreplace"))

    visit(value)
    return digest.hexdigest()


def semantic_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the resume semantics, excluding logging/budget controls."""

    normalized = copy.deepcopy(dict(config))
    training = normalized.get("training")
    if isinstance(training, dict):
        keep = {"seed"}
        normalized["training"] = {
            key: training[key] for key in sorted(keep & set(training))
        }
    for key in (
        "git_commit",
        "git_dirty",
        "source_snapshot",
        "source_snapshot_sha256",
        "resolved_config_sha256",
        "dataset_path",
        "robot_asset_path",
    ):
        normalized.pop(key, None)
    return normalized


def semantic_config_sha256(config: Mapping[str, Any]) -> str:
    return canonical_sha256(semantic_config(config))


def _run_dir_for_checkpoint(path: Path) -> Path:
    for parent in path.parents:
        if parent.name == "checkpoints":
            return parent.parent
    return path.parent


def _resolved_config_for(path: Path) -> tuple[Path | None, dict[str, Any] | None]:
    run_dir = _run_dir_for_checkpoint(path)
    candidate = run_dir / "resolved_config.json"
    if not candidate.is_file():
        return None, None
    value = read_json(candidate)
    return candidate, value if isinstance(value, dict) else None


def summarize_checkpoint(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    payload = load_checkpoint(source)
    update_payload = payload.get("update_idx")
    update_name = update_from_filename(source)
    config = payload.get("config") if isinstance(payload.get("config"), Mapping) else {}
    policy = payload.get("policy") if isinstance(payload.get("policy"), Mapping) else {}
    algo = payload.get("algo_state") if isinstance(payload.get("algo_state"), Mapping) else {}
    optimizer = payload.get("optimizer") if isinstance(payload.get("optimizer"), Mapping) else None
    sampler = payload.get("adaptive_sampler_state")
    platform = payload.get("platform_identity")
    platform = dict(platform) if isinstance(platform, Mapping) else {}
    resolved_path, resolved = _resolved_config_for(source)
    actor_normalizer = any(str(key).startswith("actor_obs_normalizer.") for key in policy)
    critic_normalizer = any(str(key).startswith("critic_obs_normalizer.") for key in policy)
    optimizer_states = (
        len(optimizer.get("state", {}))
        if isinstance(optimizer, Mapping) and isinstance(optimizer.get("state"), Mapping)
        else 0
    )
    record: dict[str, Any] = {
        "path": str(source),
        "run_dir": str(_run_dir_for_checkpoint(source)),
        "size_bytes": int(stat.st_size),
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "checkpoint_sha256": sha256_file(source),
        "update_filename": update_name,
        "update_payload": update_payload if type(update_payload) is int else None,
        "update_match": update_name is None or update_name == update_payload,
        "method": config.get("method"),
        "seed": (
            config.get("training", {}).get("seed")
            if isinstance(config.get("training"), Mapping)
            else None
        ),
        "schema_version": algo.get("fixed_reward_schema_version"),
        "semantic_config_sha256": semantic_config_sha256(config),
        "policy_state_sha256": state_digest(policy),
        "optimizer_present": optimizer is not None,
        "optimizer_state_count": optimizer_states,
        "optimizer_state_sha256": state_digest(optimizer) if optimizer is not None else None,
        "critic_optimizer_present": isinstance(algo.get("critic_optimizer"), Mapping),
        "normalizer_actor_present": actor_normalizer,
        "normalizer_critic_present": critic_normalizer,
        "normalizer_actor_count": _scalar(policy.get("actor_obs_normalizer.count")),
        "normalizer_critic_count": _scalar(policy.get("critic_obs_normalizer.count")),
        "torch_rng_present": "torch_rng_state" in payload,
        "torch_rng_sha256": state_digest(payload.get("torch_rng_state")) if "torch_rng_state" in payload else None,
        "cuda_rng_present": "cuda_rng_state" in payload,
        "cuda_rng_sha256": state_digest(payload.get("cuda_rng_state")) if "cuda_rng_state" in payload else None,
        "sampler_present": sampler is not None,
        "sampler_version": sampler.get("version") if isinstance(sampler, Mapping) else None,
        "sampler_state_sha256": state_digest(sampler) if sampler is not None else None,
        "env_transitions_total": payload.get("env_transitions_total"),
        "actor_optimizer_steps_total": algo.get("actor_optimizer_steps_total"),
        "critic_optimizer_steps_total": algo.get("critic_optimizer_steps_total"),
        "dataset_sha256": platform.get("dataset_sha256"),
        "robot_asset_sha256": platform.get("robot_asset_sha256"),
        "action_schema_sha256": platform.get("action_schema_sha256"),
        "resolved_config_path": str(resolved_path) if resolved_path else None,
        "source_snapshot_sha256": resolved.get("source_snapshot_sha256") if resolved else None,
        "resume_path": (
            resolved.get("training", {}).get("resume")
            if resolved and isinstance(resolved.get("training"), Mapping)
            else None
        ),
    }
    return record


def _scalar(value: Any) -> int | float | None:
    if value is None:
        return None
    torch = import_torch()
    if torch.is_tensor(value) and value.numel() == 1:
        return value.detach().cpu().item()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    return None


def _lineage_anchor(records: list[dict[str, Any]]) -> dict[str, Any]:
    first = min(
        records,
        key=lambda item: (
            item.get("update_payload") if item.get("update_payload") is not None else 10**18,
            item["path"],
        ),
    )
    resume_path = first.get("resume_path")
    parent_hash = None
    if resume_path:
        parent = Path(str(resume_path)).expanduser()
        if parent.is_file():
            parent_hash = sha256_file(parent)
        else:
            parent_hash = f"unresolved:{parent}"
    return {
        "run_dir": first["run_dir"],
        "semantic_config_sha256": first["semantic_config_sha256"],
        "source_snapshot_sha256": first.get("source_snapshot_sha256"),
        "parent_checkpoint_sha256": parent_hash,
        # Fresh runs lack an explicit UUID in the checkpoint schema.  Including
        # the run directory avoids incorrectly merging independent fresh runs.
        "fresh_run_identity": None if parent_hash else first["run_dir"],
    }


def assign_lineages(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    inode_groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    hash_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in records:
        inode_groups[(int(item["device"]), int(item["inode"]))].append(item)
        hash_groups[str(item["checkpoint_sha256"])].append(item)
    for group in inode_groups.values():
        canonical = min(
            group,
            key=lambda item: (
                item.get("update_filename") is None,
                item["path"],
            ),
        )
        for item in group:
            item["hardlink_alias_of"] = (
                None if item is canonical else canonical["path"]
            )
    for group in hash_groups.values():
        canonical = min(
            group,
            key=lambda item: (
                item.get("hardlink_alias_of") is not None,
                item.get("update_filename") is None,
                item["path"],
            ),
        )
        for item in group:
            item["content_alias_of"] = None if item is canonical else canonical["path"]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["run_dir"])].append(record)
    group_meta: dict[str, dict[str, Any]] = {}
    checkpoint_to_group: dict[str, str] = {}
    for run_dir, run_records in grouped.items():
        semantic_values = {item["semantic_config_sha256"] for item in run_records}
        platform_values = {
            (
                item.get("dataset_sha256"),
                item.get("robot_asset_sha256"),
                item.get("action_schema_sha256"),
            )
            for item in run_records
        }
        anchor = _lineage_anchor(run_records)
        group_meta[run_dir] = {
            "semantic": next(iter(semantic_values)) if len(semantic_values) == 1 else None,
            "platform": next(iter(platform_values)) if len(platform_values) == 1 else None,
            "coherent": len(semantic_values) == 1 and len(platform_values) == 1,
            "anchor": anchor,
            "parent_hash": anchor.get("parent_checkpoint_sha256"),
        }
        for item in run_records:
            checkpoint_to_group.setdefault(str(item["checkpoint_sha256"]), run_dir)

    resolving: set[str] = set()

    def resolve_lineage(run_dir: str) -> str:
        meta = group_meta[run_dir]
        existing = meta.get("lineage_id")
        if isinstance(existing, str):
            return existing
        if run_dir in resolving:
            meta["coherent"] = False
            lineage = canonical_sha256({"cycle_at": run_dir})
            meta["lineage_id"] = lineage
            return lineage
        resolving.add(run_dir)
        parent_hash = meta.get("parent_hash")
        parent_run = checkpoint_to_group.get(str(parent_hash)) if parent_hash else None
        if parent_run and parent_run != run_dir:
            parent = group_meta[parent_run]
            compatible = (
                meta["coherent"]
                and parent["coherent"]
                and meta["semantic"] == parent["semantic"]
                and meta["platform"] == parent["platform"]
            )
            if compatible:
                lineage = resolve_lineage(parent_run)
                meta["parent_resolved"] = True
            else:
                meta["coherent"] = False
                lineage = canonical_sha256(meta["anchor"])
                meta["parent_resolved"] = False
        elif parent_hash:
            # The parent entity may live outside the scanned glob.  Its content
            # hash remains a stable ancestry anchor, but this cannot prove
            # semantic compatibility until the parent is inventoried.
            lineage = canonical_sha256({"external_parent_checkpoint_sha256": parent_hash})
            meta["parent_resolved"] = False
        else:
            lineage = canonical_sha256(meta["anchor"])
            meta["parent_resolved"] = None
        meta["lineage_id"] = lineage
        resolving.discard(run_dir)
        return lineage

    for run_dir in sorted(grouped):
        resolve_lineage(run_dir)

    for run_dir, run_records in grouped.items():
        meta = group_meta[run_dir]
        coherent = bool(meta["coherent"])
        lineage_id = str(meta["lineage_id"])
        branch_id = canonical_sha256(
            {
                "lineage_id": lineage_id,
                "run_dir": run_dir,
                "parent_hash": meta.get("parent_hash"),
            }
        )
        seen_updates: dict[int, str] = {}
        for item in run_records:
            update = item.get("update_payload")
            previous_hash = seen_updates.get(update) if isinstance(update, int) else None
            duplicate_update = (
                previous_hash is not None
                and previous_hash != item["checkpoint_sha256"]
                and item.get("content_alias_of") is None
            )
            if isinstance(update, int) and previous_hash is None:
                seen_updates[update] = item["checkpoint_sha256"]
            item["checkpoint_lineage_id"] = lineage_id
            item["checkpoint_branch_id"] = branch_id
            item["lineage_coherent"] = coherent
            item["lineage_parent_resolved"] = meta.get("parent_resolved")
            item["duplicate_update"] = duplicate_update
            item["lineage_evidence"] = (
                "run provenance + semantic config + content-addressed resume ancestry"
            )
    return records


def inventory_checkpoints(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    resolved = sorted({Path(path).expanduser().resolve() for path in paths})
    return assign_lineages([summarize_checkpoint(path) for path in resolved])


def discover_checkpoints(repo_root: str | Path, globs: Iterable[str] = ()) -> list[Path]:
    root = Path(repo_root).expanduser().resolve()
    patterns = list(globs) or ["runs/*/checkpoints/*.pt"]
    found: set[Path] = set()
    for pattern in patterns:
        candidate = Path(pattern).expanduser()
        if candidate.is_absolute():
            # pathlib cannot glob an absolute pattern from another Path.  Split
            # at the first glob metacharacter and fall back to glob.glob.
            import glob

            for value in glob.glob(str(candidate), recursive=True):
                path = Path(value)
                if path.is_file():
                    found.add(path.resolve())
        else:
            for path in root.glob(str(candidate)):
                if path.is_file():
                    found.add(path.resolve())
    return sorted(found)


def compare_checkpoint_payloads(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    metric_atol: float = 0.0,
    metric_rtol: float = 0.0,
) -> dict[str, Any]:
    from diagnostics.common.statistics import compare_numeric_mappings

    sections = (
        "policy",
        "optimizer",
        "algo_state",
        "adaptive_sampler_state",
        "torch_rng_state",
        "cuda_rng_state",
    )
    digests = {
        section: {
            "first": state_digest(first.get(section)),
            "second": state_digest(second.get(section)),
        }
        for section in sections
        if section in first or section in second
    }
    exact_sections = {
        key: value["first"] == value["second"] for key, value in digests.items()
    }
    first_metrics = first.get("metrics", {}) if isinstance(first.get("metrics"), Mapping) else {}
    second_metrics = second.get("metrics", {}) if isinstance(second.get("metrics"), Mapping) else {}
    telemetry_prefixes = ("perf/", "timing/")

    def deterministic_metrics(values: Mapping[str, Any]) -> dict[str, Any]:
        return {
            str(key): value
            for key, value in values.items()
            if not str(key).startswith(telemetry_prefixes)
            and "train_wall" not in str(key)
        }

    def telemetry_metrics(values: Mapping[str, Any]) -> dict[str, Any]:
        return {
            str(key): value
            for key, value in values.items()
            if str(key).startswith(telemetry_prefixes)
            or "train_wall" in str(key)
        }

    metrics = compare_numeric_mappings(
        deterministic_metrics(first_metrics),
        deterministic_metrics(second_metrics),
        atol=metric_atol,
        rtol=metric_rtol,
    )
    telemetry = compare_numeric_mappings(
        telemetry_metrics(first_metrics),
        telemetry_metrics(second_metrics),
        atol=metric_atol,
        rtol=metric_rtol,
    )
    counters = {
        key: {
            "first": first.get(key),
            "second": second.get(key),
            "exact": first.get(key) == second.get(key),
        }
        for key in ("update_idx", "env_transitions_total")
    }
    wall_clock = {
        "first": first.get("train_wall_seconds_total"),
        "second": second.get("train_wall_seconds_total"),
        "excluded_from_reproducibility": True,
    }
    reproducible = (
        all(exact_sections.values())
        and metrics["all_close"]
        and all(value["exact"] for value in counters.values())
    )
    return {
        "section_digests": digests,
        "exact_sections": exact_sections,
        "all_state_exact": all(exact_sections.values()),
        "deterministic_metrics": metrics,
        "nondeterministic_telemetry": telemetry,
        "counters": counters,
        "train_wall_seconds_total": wall_clock,
        "telemetry_exclusion_rule": "perf/*, timing/*, and train_wall* are measured wall-clock telemetry and cannot affect PASS/FAIL",
        "reproducible": reproducible,
    }
