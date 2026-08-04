"""Spec loading, contract checks, and atomic suite reporting."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from diagnostics import SUITE_SCHEMA_VERSION
from diagnostics.common.status import (
    DiagnosticResult,
    DiagnosticStatus,
    aggregate_status,
    status_counts,
)


DECISION_VARIABLES = ("R", "T", "E", "L", "F", "X")
FINAL_ARTIFACTS = (
    "manifest",
    "status",
    "checkpoint_inventory",
    "canonical_rollout_index",
    "policy_class_gate",
    "domain_triangle_gate",
    "reward_validity_gate",
    "edge_gate",
    "family_decision_matrix",
    "discovery_report",
    "diagnostic_results",
    "source_and_logs",
)


class SpecError(ValueError):
    """Raised when the frozen suite specification is incomplete or ambiguous."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_structured_file(path: str | Path) -> dict[str, Any]:
    """Load JSON or YAML while keeping the base runner dependency-free.

    The checked-in ``.yaml`` spec is deliberately JSON-compatible YAML.  A
    user-provided non-JSON YAML file is accepted only when PyYAML is installed.
    """

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as json_exc:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise SpecError(
                f"{source} is not JSON-compatible YAML and PyYAML is unavailable"
            ) from json_exc
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise SpecError(f"{source} must contain a top-level mapping")
    return payload


def _flatten_diagnostics(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    stages = spec.get("stages")
    if not isinstance(stages, list) or not stages:
        raise SpecError("spec.stages must be a non-empty list")
    for stage in stages:
        if not isinstance(stage, Mapping):
            raise SpecError("each stage must be a mapping")
        stage_id = str(stage.get("id", ""))
        diagnostics = stage.get("diagnostics")
        if not isinstance(diagnostics, list) or not diagnostics:
            raise SpecError(f"stage {stage_id!r} has no diagnostics")
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, Mapping):
                raise SpecError(f"stage {stage_id!r} contains a non-mapping diagnostic")
            item = dict(diagnostic)
            item["stage"] = stage_id
            flattened.append(item)
    return flattened


def validate_spec(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate identities, the DAG, statuses, thresholds, and frozen decisions."""

    for key in (
        "schema_version",
        "suite_id",
        "suite_version",
        "fail_closed",
        "status_values",
        "frozen_research_boundaries",
        "thresholds",
        "decision_variables",
        "decision_matrix",
        "stages",
    ):
        if key not in spec:
            raise SpecError(f"missing required spec field: {key}")
    expected_statuses = [status.value for status in DiagnosticStatus]
    if spec["status_values"] != expected_statuses:
        raise SpecError(
            f"status_values must be frozen exactly as {expected_statuses}, got {spec['status_values']!r}"
        )
    if spec["fail_closed"] is not True:
        raise SpecError("fail_closed must be true")
    variables = spec["decision_variables"]
    if not isinstance(variables, Mapping) or tuple(variables) != DECISION_VARIABLES:
        raise SpecError(f"decision_variables must be ordered exactly as {DECISION_VARIABLES}")

    diagnostics = _flatten_diagnostics(spec)
    ids = [str(item.get("id", "")) for item in diagnostics]
    if any(not identifier for identifier in ids) or len(ids) != len(set(ids)):
        raise SpecError("diagnostic ids must be non-empty and unique")
    id_set = set(ids)
    for item in diagnostics:
        for key in ("name", "entrypoint", "dependencies", "expected_outputs", "status_artifact"):
            if key not in item:
                raise SpecError(f"diagnostic {item.get('id')} is missing {key}")
        dependencies = item["dependencies"]
        if not isinstance(dependencies, list):
            raise SpecError(f"diagnostic {item['id']} dependencies must be a list")
        unknown = sorted(set(str(value) for value in dependencies) - id_set)
        if unknown:
            raise SpecError(f"diagnostic {item['id']} has unknown dependencies: {unknown}")
        if str(item["id"]) in {str(value) for value in dependencies}:
            raise SpecError(f"diagnostic {item['id']} depends on itself")

    # Kahn's algorithm detects cycles independently of file ordering.
    remaining = {identifier: set() for identifier in ids}
    for item in diagnostics:
        remaining[str(item["id"])] = {str(value) for value in item["dependencies"]}
    emitted: set[str] = set()
    while remaining:
        ready = [identifier for identifier, deps in remaining.items() if deps <= emitted]
        if not ready:
            raise SpecError(f"diagnostic dependency graph contains a cycle: {sorted(remaining)}")
        for identifier in ready:
            emitted.add(identifier)
            del remaining[identifier]

    matrix = spec["decision_matrix"]
    if not isinstance(matrix, list) or [row.get("priority") for row in matrix] != list(
        range(1, len(matrix) + 1)
    ):
        raise SpecError("decision_matrix priorities must be contiguous and frozen in list order")
    if len(matrix) != 6:
        raise SpecError("the frozen family decision matrix must contain exactly six rows")
    return diagnostics


def topological_diagnostics(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return a stable topological order, preserving spec order where possible."""

    diagnostics = validate_spec(spec)
    by_id = {str(item["id"]): item for item in diagnostics}
    order_index = {str(item["id"]): index for index, item in enumerate(diagnostics)}
    indegree = {identifier: 0 for identifier in by_id}
    children = {identifier: [] for identifier in by_id}
    for identifier, item in by_id.items():
        for dependency in item["dependencies"]:
            dep_id = str(dependency)
            indegree[identifier] += 1
            children[dep_id].append(identifier)
    ready = sorted((key for key, value in indegree.items() if value == 0), key=order_index.get)
    ordered: list[dict[str, Any]] = []
    while ready:
        identifier = ready.pop(0)
        ordered.append(by_id[identifier])
        for child in children[identifier]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort(key=order_index.get)
    if len(ordered) != len(diagnostics):  # Defensive; validate_spec already checks this.
        raise SpecError("diagnostic dependency graph is not acyclic")
    return ordered


def atomic_write_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = True,
) -> None:
    """Write JSON atomically so interrupted runs never leave partial status."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing artifact: {target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def default_artifacts() -> dict[str, str | None]:
    return {
        "manifest": "manifest.json",
        "status": "status.json",
        "checkpoint_inventory": "checkpoint_inventory.csv",
        "canonical_rollout_index": "canonical_rollout_index.parquet",
        "policy_class_gate": "policy_class_gate.json",
        "domain_triangle_gate": "domain_triangle_gate.json",
        "reward_validity_gate": "reward_validity_gate.json",
        "edge_gate": "edge_gate.json",
        "family_decision_matrix": "family_decision_matrix.json",
        "discovery_report": "discovery_report.md",
        "diagnostic_results": "diagnostic_results.txt",
        "source_and_logs": "source_and_logs.tar.gz",
    }


def build_suite_result(
    *,
    spec: Mapping[str, Any],
    spec_sha256: str,
    spec_path: str,
    output_dir: str,
    dry_run: bool,
    results: Sequence[DiagnosticResult],
    started_at: str,
    finished_at: str | None = None,
) -> dict[str, Any]:
    """Build a schema-complete, fail-closed status payload."""

    result_dicts = [result.to_dict() for result in results]
    decision_variables = {
        variable: {
            "value": "UNKNOWN",
            "evidence_diagnostics": [],
            "reason": "not yet derived by diag_63_family_decision_matrix.py",
        }
        for variable in DECISION_VARIABLES
    }
    return {
        "schema_version": SUITE_SCHEMA_VERSION,
        "suite_id": str(spec["suite_id"]),
        "suite_version": str(spec["suite_version"]),
        "spec_sha256": spec_sha256,
        "generated_at": utc_now_iso(),
        "execution": {
            "mode": "dry_run" if dry_run else "execute",
            "dry_run": dry_run,
            "fail_closed": True,
            "spec_path": spec_path,
            "output_dir": output_dir,
            "started_at": started_at,
            "finished_at": finished_at,
        },
        "overall_status": aggregate_status(results).value,
        "status_counts": status_counts(results),
        "diagnostics": result_dicts,
        "decision_variables": decision_variables,
        "decision": {
            "status": "DEFERRED",
            "matched_rule_id": None,
            "selected_family": None,
            "reason": "R/T/E/L/F/X are incomplete; no family may be selected fail-open.",
        },
        "artifacts": default_artifacts(),
    }


def read_status_artifact(path: str | Path) -> tuple[DiagnosticStatus, str | None, dict[str, Any]]:
    """Read an entrypoint result without guessing a status from its exit code."""

    artifact = Path(path)
    try:
        payload = json.loads(artifact.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SpecError(f"cannot read status artifact {artifact}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SpecError(f"status artifact {artifact} must contain a mapping")
    if "status" not in payload:
        raise SpecError(f"status artifact {artifact} has no top-level status")
    status = DiagnosticStatus.parse(payload["status"])
    reason_value = payload.get("reason")
    reason = None if reason_value is None else str(reason_value)
    return status, reason, payload


__all__ = [
    "DECISION_VARIABLES",
    "FINAL_ARTIFACTS",
    "SpecError",
    "atomic_write_json",
    "build_suite_result",
    "default_artifacts",
    "load_structured_file",
    "read_status_artifact",
    "sha256_file",
    "topological_diagnostics",
    "utc_now_iso",
    "validate_spec",
]
