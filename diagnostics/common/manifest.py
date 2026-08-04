"""Provenance and fail-closed result helpers for the discovery suite.

This module deliberately has no Isaac Lab dependency.  Stage-0 diagnostics
must be able to audit an experiment even on a machine that cannot launch the
simulator.  All writers use exclusive creation: a diagnostic can never turn a
previous result into a different result in place.
"""

from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
from typing import Any, Mapping, Sequence


PASS = "PASS"
FAIL = "FAIL"
INVALID_PROTOCOL = "INVALID_PROTOCOL"
SKIPPED_DEPENDENCY = "SKIPPED_DEPENDENCY"
DIAGNOSTIC_STATUSES = frozenset(
    {PASS, FAIL, INVALID_PROTOCOL, SKIPPED_DEPENDENCY}
)

MANIFEST_REQUIRED_HASH_FIELDS = (
    "source_snapshot_sha256",
    "resolved_config_sha256",
    "checkpoint_sha256",
    "robot_asset_sha256",
    "motion_sha256",
    "action_schema_sha256",
)


class ProtocolError(ValueError):
    """The requested diagnostic would violate its scientific protocol."""


class DependencyUnavailable(RuntimeError):
    """A valid protocol cannot run because a runtime artifact is unavailable."""


def validate_status(status: str) -> str:
    value = str(status)
    if value not in DIAGNOSTIC_STATUSES:
        raise ValueError(
            f"status must be one of {sorted(DIAGNOSTIC_STATUSES)}, got {value!r}"
        )
    return value


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def read_json(path: str | Path) -> Any:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_spec(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - dependency gate
        raise DependencyUnavailable("PyYAML is required to read the suite spec") from exc
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ProtocolError(f"suite spec must be a mapping: {source}")
    value["_spec_path"] = str(source)
    return value


def spec_value(
    spec: Mapping[str, Any],
    *paths: str,
    default: Any = None,
) -> Any:
    """Return the first present dotted path from a permissive suite spec."""

    for dotted in paths:
        node: Any = spec
        present = True
        for key in dotted.split("."):
            if not isinstance(node, Mapping) or key not in node:
                present = False
                break
            node = node[key]
        if present and node not in (None, ""):
            return node
    return default


def resolve_path(
    value: str | Path | None,
    *,
    base: str | Path,
) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base) / path
    return path.resolve()


def _exclusive_open(path: Path, *, newline: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Python's x mode maps to O_CREAT|O_EXCL and therefore cannot silently
    # replace a result, including under concurrent suite launches.
    return path.open("x", encoding="utf-8", newline=newline)


def write_json_exclusive(path: str | Path, value: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    status = value.get("status")
    if status is not None:
        validate_status(str(status))
    with _exclusive_open(target) as handle:
        json.dump(
            value,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
        # Isaac/Kit can hang during orderly application shutdown on this
        # platform.  Diagnostic entrypoints therefore may use os._exit after
        # publishing their final status; make that status durable first.
        handle.flush()
        os.fsync(handle.fileno())
    # Persist the directory entry as well, so a hard process exit cannot lose
    # a just-created status file after the file itself was fsynced.
    try:
        directory_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        # Some filesystems do not permit directory fsync.  The file fsync
        # above remains the mandatory durability barrier.
        pass
    return target


def write_csv_exclusive(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> Path:
    target = Path(path).expanduser().resolve()
    if fieldnames is None:
        ordered: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    ordered.append(str(key))
        fieldnames = ordered
    with _exclusive_open(target, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        directory_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass
    return target


def diagnostic_result(
    diagnostic_id: str,
    status: str,
    *,
    summary: str,
    evidence: Mapping[str, Any] | None = None,
    errors: Sequence[str] = (),
    warnings: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "diagnostic_id": str(diagnostic_id),
        "status": validate_status(status),
        "summary": str(summary),
        "evidence": dict(evidence or {}),
        "errors": [str(item) for item in errors],
        "warnings": [str(item) for item in warnings],
    }


def _git(repo_root: Path, *args: str, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout if binary else completed.stdout.decode("utf-8", "strict")


def git_provenance(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).expanduser().resolve()
    try:
        commit = str(_git(root, "rev-parse", "HEAD")).strip()
        status_raw = _git(root, "status", "--porcelain=v1", "-z", binary=True)
        assert isinstance(status_raw, bytes)
        diff = _git(root, "diff", "--binary", "HEAD", binary=True)
        assert isinstance(diff, bytes)
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        raise DependencyUnavailable(f"git provenance is unavailable: {exc}") from exc
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise ProtocolError(f"git commit is not a full SHA-1: {commit!r}")
    entries = [part.decode("utf-8", "surrogateescape") for part in status_raw.split(b"\0") if part]
    return {
        "git_commit": commit,
        "git_dirty": bool(entries),
        "git_status_entries": entries,
        "git_status_sha256": sha256_bytes(status_raw),
        "git_diff_sha256": sha256_bytes(diff),
    }


_SOURCE_DIRS = (
    "components",
    "configs",
    "diagnostics",
    "engine",
    "envs",
    "method",
    "models",
    "tests",
    "tools",
)
_ROOT_SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".json"}


def source_files(repo_root: str | Path) -> list[Path]:
    root = Path(repo_root).expanduser().resolve()
    files: set[Path] = set()
    for dirname in _SOURCE_DIRS:
        directory = root / dirname
        if not directory.is_dir():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if "__pycache__" in relative.parts or path.suffix == ".pyc":
                continue
            files.add(path.resolve())
    for path in root.iterdir():
        if path.is_file() and path.suffix in _ROOT_SOURCE_SUFFIXES:
            files.add(path.resolve())
    return sorted(files, key=lambda item: item.relative_to(root).as_posix())


def create_source_snapshot(
    repo_root: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    """Create a deterministic gzip tar containing tracked and dirty source.

    File modes and bytes are preserved, while uid/gid/mtime are normalized so
    identical source produces an identical archive hash.
    """

    root = Path(repo_root).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    files = source_files(root)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as zipped:
                with tarfile.open(fileobj=zipped, mode="w", format=tarfile.GNU_FORMAT) as archive:
                    for path in files:
                        relative = path.relative_to(root).as_posix()
                        info = archive.gettarinfo(str(path), arcname=relative)
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        info.mtime = 0
                        with path.open("rb") as handle:
                            archive.addfile(info, handle)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "file_count": len(files),
        "members_sha256": canonical_sha256(
            [
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path),
                }
                for path in files
            ]
        ),
    }


def parse_python_literal_assignment(path: str | Path, name: str) -> Any:
    source = Path(path).expanduser().resolve()
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(target, ast.Name) and target.id == name for target in targets):
            value = node.value
            if value is None:
                break
            return ast.literal_eval(value)
    raise KeyError(f"literal assignment {name!r} not found in {source}")


def action_names_from_source(repo_root: str | Path) -> list[str]:
    root = Path(repo_root).expanduser().resolve()
    path = root / "envs" / "robots" / "g1.py"
    values = parse_python_literal_assignment(path, "G1_29DOF_ASSET_JOINT_NAMES")
    if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
        raise ProtocolError(f"G1 action schema is not a literal list[str]: {path}")
    return list(values)


def resolve_task_motion(repo_root: str | Path, task_name: str) -> Path:
    # envs.tasks has no Isaac dependency and is the single source of task
    # identity.  Import lazily so this common module remains portable.
    root = Path(repo_root).expanduser().resolve()
    import sys

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from envs.tasks import resolve_task

    return resolve_task(str(task_name)).motion_file.resolve()


def load_config_tree(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() == ".json":
        value = read_json(source)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency gate
            raise DependencyUnavailable("PyYAML is required to read config") from exc
        with source.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ProtocolError(f"config must be a mapping: {source}")
    return value


def recompute_resolved_config_sha256(config: Mapping[str, Any]) -> str:
    # train.py hashes the resolved tree immediately before adding this field.
    identity = dict(config)
    identity.pop("resolved_config_sha256", None)
    encoded = json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    return sha256_bytes(encoded)


def complete_manifest_errors(manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    commit = manifest.get("git_commit")
    if not isinstance(commit, str) or len(commit) != 40 or commit == "unknown":
        errors.append("git_commit must be a full, known commit hash")
    for field in MANIFEST_REQUIRED_HASH_FIELDS:
        value = manifest.get(field)
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"{field} must be a concrete SHA-256")
        elif value in {"missing", "unknown", ""}:
            errors.append(f"{field} may not be {value!r}")
    update = manifest.get("checkpoint_update")
    if type(update) is not int or update < 0:
        errors.append("checkpoint_update must be a non-negative integer")
    lineage = manifest.get("checkpoint_lineage_id")
    if not isinstance(lineage, str) or not lineage:
        errors.append("checkpoint_lineage_id is required")
    task_name = manifest.get("task_name")
    if not isinstance(task_name, str) or not task_name:
        errors.append("task_name is required")
    return errors
