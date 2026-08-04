#!/usr/bin/env python3
"""Run the frozen AMP Research Discovery Suite in dependency order.

This runner is intentionally conservative: a missing executable, unreadable
status artifact, undeclared output, or protocol exception can never be treated
as scientific success.  Individual entrypoints own their scientific PASS/FAIL
decision and communicate it through a top-level ``status`` field.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.reporting import (  # noqa: E402
    SpecError,
    atomic_write_json,
    build_suite_result,
    load_structured_file,
    read_status_artifact,
    sha256_file,
    topological_diagnostics,
    utc_now_iso,
)
from diagnostics.common.status import DiagnosticResult, DiagnosticStatus  # noqa: E402


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fail-closed AMP Research Discovery Suite."
    )
    parser.add_argument("--spec", required=True, help="Frozen discovery-suite YAML/JSON spec.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override spec.output_dir. Existing non-empty directories require --resume.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the complete execution plan without creating files.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse valid existing diagnostic artifacts and continue missing nodes.",
    )
    return parser.parse_args(argv)


def _resolve_repo_root(spec: Mapping[str, Any]) -> Path:
    configured = Path(str(spec.get("repo_root", REPO_ROOT)))
    if not configured.is_absolute():
        configured = (Path.cwd() / configured).resolve()
    if not configured.is_dir():
        raise SpecError(f"repo_root is not a directory: {configured}")
    return configured


def _resolve_output_dir(
    spec: Mapping[str, Any], repo_root: Path, override: str | None
) -> Path:
    configured = Path(override if override is not None else str(spec.get("output_dir", "")))
    if not str(configured):
        raise SpecError("spec.output_dir is empty and --output-dir was not provided")
    if not configured.is_absolute():
        configured = repo_root / configured
    return configured.resolve()


def _safe_relative_path(root: Path, relative: str, *, label: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        raise SpecError(f"{label} must be relative to the suite output/repository: {relative}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise SpecError(f"{label} escapes its allowed root: {relative}") from exc
    return resolved


def _command_for(
    diagnostic: Mapping[str, Any],
    *,
    repo_root: Path,
    spec_path: Path,
    output_dir: Path,
) -> tuple[Path, tuple[str, ...]]:
    entrypoint = _safe_relative_path(
        repo_root, str(diagnostic["entrypoint"]), label="diagnostic entrypoint"
    )
    command = (
        sys.executable,
        str(entrypoint),
        "--spec",
        str(spec_path),
        "--output-dir",
        str(output_dir),
    )
    return entrypoint, command


def _dependency_blockers(
    diagnostic: Mapping[str, Any], completed: Mapping[str, DiagnosticResult]
) -> tuple[str, ...]:
    policy = str(diagnostic.get("dependency_policy", "all_pass"))
    if policy not in {"all_pass", "completed_valid", "recorded_valid"}:
        raise SpecError(
            f"diagnostic {diagnostic['id']} has invalid dependency_policy {policy!r}"
        )
    if policy == "all_pass":
        accepted = {DiagnosticStatus.PASS}
    elif policy == "completed_valid":
        accepted = {DiagnosticStatus.PASS, DiagnosticStatus.FAIL}
    else:
        # Summary/report nodes must be allowed to record UNKNOWN when real
        # evidence is unavailable.  INVALID_PROTOCOL remains a hard blocker.
        accepted = {
            DiagnosticStatus.PASS,
            DiagnosticStatus.FAIL,
            DiagnosticStatus.SKIPPED_DEPENDENCY,
        }
    blockers: list[str] = []
    for raw_dependency in diagnostic["dependencies"]:
        dependency = str(raw_dependency)
        result = completed.get(dependency)
        if result is None or result.status not in accepted:
            blockers.append(dependency)
    return tuple(blockers)


def _base_result_args(
    diagnostic: Mapping[str, Any], command: Sequence[str]
) -> dict[str, Any]:
    return {
        "diagnostic_id": str(diagnostic["id"]),
        "name": str(diagnostic["name"]),
        "stage": str(diagnostic["stage"]),
        "dependencies": tuple(str(value) for value in diagnostic["dependencies"]),
        "command": tuple(str(value) for value in command),
        "required": bool(diagnostic.get("required", True)),
        "blocking": bool(diagnostic.get("blocking", False)),
        "expected_outputs": tuple(str(value) for value in diagnostic["expected_outputs"]),
        "status_artifact": str(diagnostic["status_artifact"]),
    }


def _observed_outputs(output_dir: Path, expected: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        relative
        for relative in expected
        if _safe_relative_path(output_dir, relative, label="expected output").exists()
    )


def _reuse_existing(
    diagnostic: Mapping[str, Any],
    command: Sequence[str],
    output_dir: Path,
) -> DiagnosticResult | None:
    status_path = _safe_relative_path(
        output_dir, str(diagnostic["status_artifact"]), label="status artifact"
    )
    if not status_path.exists():
        return None
    base = _base_result_args(diagnostic, command)
    expected = base["expected_outputs"]
    observed = _observed_outputs(output_dir, expected)
    try:
        status, reason, _payload = read_status_artifact(status_path)
    except (SpecError, ValueError) as exc:
        return DiagnosticResult(
            **base,
            status=DiagnosticStatus.INVALID_PROTOCOL,
            reason=f"invalid existing status artifact: {exc}",
            observed_outputs=observed,
            metadata={"resumed": True},
        )
    missing = sorted(set(expected) - set(observed))
    if status is DiagnosticStatus.PASS and missing:
        status = DiagnosticStatus.INVALID_PROTOCOL
        reason = f"existing PASS artifact is missing declared outputs: {missing}"
    return DiagnosticResult(
        **base,
        status=status,
        reason=reason or "reused existing valid status artifact",
        observed_outputs=observed,
        metadata={"resumed": True},
    )


def _run_one(
    diagnostic: Mapping[str, Any],
    *,
    command: tuple[str, ...],
    repo_root: Path,
    output_dir: Path,
) -> DiagnosticResult:
    base = _base_result_args(diagnostic, command)
    diagnostic_id = str(diagnostic["id"])
    log_dir = output_dir / "logs" / "suite_runner"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"diag_{diagnostic_id}.stdout.log"
    stderr_path = log_dir / f"diag_{diagnostic_id}.stderr.log"
    started_at = utc_now_iso()
    started_clock = time.monotonic()
    exit_code: int | None = None
    protocol_error: str | None = None
    try:
        # The suite directory itself is protected by --resume.  Append-only
        # runner logs let an interrupted process be resumed without deleting
        # provenance from the first attempt.
        with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open(
            "a", encoding="utf-8"
        ) as stderr_handle:
            completed = subprocess.run(
                command,
                cwd=repo_root,
                stdout=stdout_handle,
                stderr=stderr_handle,
                check=False,
            )
            exit_code = completed.returncode
    except OSError as exc:
        protocol_error = f"entrypoint process could not be started: {exc}"
    finished_at = utc_now_iso()
    duration = time.monotonic() - started_clock
    observed = _observed_outputs(output_dir, base["expected_outputs"])

    if protocol_error is not None:
        status = DiagnosticStatus.INVALID_PROTOCOL
        reason = protocol_error
    elif exit_code != 0:
        status = DiagnosticStatus.INVALID_PROTOCOL
        reason = f"entrypoint crashed or violated its process contract (exit_code={exit_code})"
    else:
        status_path = _safe_relative_path(
            output_dir, str(diagnostic["status_artifact"]), label="status artifact"
        )
        if not status_path.is_file():
            status = DiagnosticStatus.INVALID_PROTOCOL
            reason = f"entrypoint returned zero but did not write {diagnostic['status_artifact']}"
        else:
            try:
                status, reason, _payload = read_status_artifact(status_path)
            except (SpecError, ValueError) as exc:
                status = DiagnosticStatus.INVALID_PROTOCOL
                reason = str(exc)
    missing = sorted(set(base["expected_outputs"]) - set(observed))
    if status is DiagnosticStatus.PASS and missing:
        status = DiagnosticStatus.INVALID_PROTOCOL
        reason = f"entrypoint reported PASS but declared outputs are missing: {missing}"
    return DiagnosticResult(
        **base,
        status=status,
        reason=reason,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration,
        exit_code=exit_code,
        observed_outputs=observed,
        stdout_log=str(stdout_path.relative_to(output_dir)),
        stderr_log=str(stderr_path.relative_to(output_dir)),
    )


def _skipped_result(
    diagnostic: Mapping[str, Any],
    command: Sequence[str],
    reason: str,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> DiagnosticResult:
    return DiagnosticResult(
        **_base_result_args(diagnostic, command),
        status=DiagnosticStatus.SKIPPED_DEPENDENCY,
        reason=reason,
        metadata={} if metadata is None else metadata,
    )


def _write_progress(
    *,
    spec: Mapping[str, Any],
    spec_path: Path,
    spec_sha256: str,
    output_dir: Path,
    results: Sequence[DiagnosticResult],
    started_at: str,
    finished_at: str | None = None,
) -> None:
    payload = build_suite_result(
        spec=spec,
        spec_sha256=spec_sha256,
        spec_path=str(spec_path),
        output_dir=str(output_dir),
        dry_run=False,
        results=results,
        started_at=started_at,
        finished_at=finished_at,
    )
    _hydrate_frozen_decision(payload, output_dir)
    atomic_write_json(output_dir / "status.json", payload)


def _hydrate_frozen_decision(payload: dict[str, Any], output_dir: Path) -> None:
    """Copy only diag_63's validated frozen decision into the suite status."""

    artifact = output_dir / "family_decision_matrix.json"
    if not artifact.is_file():
        return
    try:
        value = load_structured_file(artifact)
        if value.get("diagnostic_id") != "63" or value.get("status") != "PASS":
            return
        evidence = value.get("evidence")
        if not isinstance(evidence, Mapping):
            return
        variables = evidence.get("decision_variables")
        decision = evidence.get("decision")
        if not isinstance(variables, Mapping) or set(variables) != {
            "R", "T", "E", "L", "F", "X"
        }:
            return
        if not isinstance(decision, Mapping):
            return
        payload["decision_variables"] = dict(variables)
        payload["decision"] = dict(decision)
    except (OSError, SpecError, ValueError):
        # diag_63/runner status already records the protocol failure.  Never
        # turn an unreadable decision artifact into a selected family.
        return


def run(argv: Sequence[str] | None = None) -> tuple[dict[str, Any], int]:
    args = _parse_args(argv)
    spec_path = Path(args.spec).expanduser().resolve()
    spec = load_structured_file(spec_path)
    diagnostics = topological_diagnostics(spec)
    repo_root = _resolve_repo_root(spec)
    output_dir = _resolve_output_dir(spec, repo_root, args.output_dir)
    spec_digest = sha256_file(spec_path)
    started_at = utc_now_iso()

    if args.dry_run:
        results: list[DiagnosticResult] = []
        for diagnostic in diagnostics:
            entrypoint, command = _command_for(
                diagnostic,
                repo_root=repo_root,
                spec_path=spec_path,
                output_dir=output_dir,
            )
            results.append(
                _skipped_result(
                    diagnostic,
                    command,
                    "dry_run_not_executed",
                    metadata={"entrypoint_exists": entrypoint.is_file()},
                )
            )
        payload = build_suite_result(
            spec=spec,
            spec_sha256=spec_digest,
            spec_path=str(spec_path),
            output_dir=str(output_dir),
            dry_run=True,
            results=results,
            started_at=started_at,
            finished_at=utc_now_iso(),
        )
        return payload, 0

    if output_dir.exists() and any(output_dir.iterdir()) and not args.resume:
        raise SpecError(
            f"output directory is not empty: {output_dir}; use --resume or a fresh --output-dir"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    completed: dict[str, DiagnosticResult] = {}
    blocking_failure: str | None = None

    for diagnostic in diagnostics:
        identifier = str(diagnostic["id"])
        entrypoint, command = _command_for(
            diagnostic,
            repo_root=repo_root,
            spec_path=spec_path,
            output_dir=output_dir,
        )
        if args.resume:
            existing = _reuse_existing(diagnostic, command, output_dir)
            if existing is not None:
                result = existing
                completed[identifier] = result
                if result.blocking and result.status is DiagnosticStatus.INVALID_PROTOCOL:
                    blocking_failure = identifier
                _write_progress(
                    spec=spec,
                    spec_path=spec_path,
                    spec_sha256=spec_digest,
                    output_dir=output_dir,
                    results=list(completed.values()),
                    started_at=started_at,
                )
                continue
        if blocking_failure is not None:
            result = _skipped_result(
                diagnostic,
                command,
                f"fail_closed_after_blocking_diagnostic:{blocking_failure}",
            )
        else:
            blockers = _dependency_blockers(diagnostic, completed)
            if blockers:
                result = _skipped_result(
                    diagnostic,
                    command,
                    "dependency_not_accepted:" + ",".join(blockers),
                    metadata={"blocked_by": list(blockers)},
                )
            elif not entrypoint.is_file():
                result = _skipped_result(
                    diagnostic,
                    command,
                    f"entrypoint_not_implemented:{diagnostic['entrypoint']}",
                )
            else:
                result = _run_one(
                    diagnostic,
                    command=command,
                    repo_root=repo_root,
                    output_dir=output_dir,
                )
        completed[identifier] = result
        if result.blocking and result.status is DiagnosticStatus.INVALID_PROTOCOL:
            blocking_failure = identifier
        _write_progress(
            spec=spec,
            spec_path=spec_path,
            spec_sha256=spec_digest,
            output_dir=output_dir,
            results=list(completed.values()),
            started_at=started_at,
        )

    results = list(completed.values())
    finished_at = utc_now_iso()
    payload = build_suite_result(
        spec=spec,
        spec_sha256=spec_digest,
        spec_path=str(spec_path),
        output_dir=str(output_dir),
        dry_run=False,
        results=results,
        started_at=started_at,
        finished_at=finished_at,
    )
    _hydrate_frozen_decision(payload, output_dir)
    atomic_write_json(output_dir / "status.json", payload)
    statuses = {result.status for result in results}
    if DiagnosticStatus.INVALID_PROTOCOL in statuses:
        return payload, 3
    if DiagnosticStatus.FAIL in statuses:
        return payload, 2
    # A partially implemented suite is explicitly represented as SKIPPED, but
    # the runner itself completed its orchestration contract successfully.
    return payload, 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        payload, exit_code = run(argv)
    except (OSError, SpecError, ValueError) as exc:
        print(f"INVALID_PROTOCOL: {exc}", file=sys.stderr)
        return 3
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
