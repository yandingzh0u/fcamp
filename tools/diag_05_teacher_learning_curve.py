#!/usr/bin/env python3
"""Reconstruct teacher outcome curves without stitching checkpoint lineages."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.manifest import (
    INVALID_PROTOCOL,
    PASS,
    diagnostic_result,
    load_spec,
    read_json,
    resolve_path,
    spec_value,
    write_csv_exclusive,
    write_json_exclusive,
)
from diagnostics.common.statistics import first_threshold_crossing, monotonic_violations


CURVE_FIELDS = (
    "curve_id",
    "run_id",
    "source_snapshot_sha256",
    "update",
    "protocol",
    "fixed_seed",
    "steps_mean",
    "steps_min",
    "steps_max",
    "return_mean",
    "failure_frac",
    "motion_complete_frac",
    "reference_progress_mean",
    "checkpoint_available",
    "checkpoint_sha256",
    "source_kinds",
    "source_paths",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--include-git-deleted-logs", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _run_identity(run_dir: Path) -> tuple[str, str | None]:
    resolved = run_dir / "resolved_config.json"
    if resolved.is_file():
        config = read_json(resolved)
        return run_dir.name, config.get("source_snapshot_sha256")
    return run_dir.name, None


def _row_from_validation_metrics(
    update: int,
    metrics: dict[str, Any],
    *,
    run_id: str,
    source_snapshot: str | None,
    source_kind: str,
    source_path: str,
) -> dict[str, Any]:
    def metric(name: str) -> float | None:
        return _float(metrics.get(f"validation/{name}", metrics.get(name)))

    curve_id = f"{run_id}:{source_snapshot or 'snapshot_unknown'}"
    return {
        "curve_id": curve_id,
        "run_id": run_id,
        "source_snapshot_sha256": source_snapshot,
        "update": int(update),
        "protocol": "clean_mean",
        "fixed_seed": metric("fixed_seed"),
        "steps_mean": metric("steps_mean"),
        "steps_min": metric("steps_min"),
        "steps_max": metric("steps_max"),
        "return_mean": metric("return_mean"),
        "failure_frac": metric("failure_frac"),
        "motion_complete_frac": metric("motion_complete_frac"),
        "reference_progress_mean": metric("reference_progress_mean"),
        "checkpoint_available": False,
        "checkpoint_sha256": None,
        "source_kinds": [source_kind],
        "source_paths": [source_path],
    }


def _metrics_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    run_id, snapshot = _run_identity(path.parents[1])
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed JSONL {path}:{line_number}: {exc}") from exc
            metrics = entry.get("metrics", {})
            if not isinstance(metrics, dict) or "validation/failure_frac" not in metrics:
                continue
            rows.append(
                _row_from_validation_metrics(
                    int(entry["update"]), metrics,
                    run_id=run_id, source_snapshot=snapshot,
                    source_kind="metrics_jsonl", source_path=str(path),
                )
            )
    return rows


_TOKEN_RE = re.compile(r"([A-Za-z0-9_]+)=(-?[0-9]+(?:\.[0-9]+)?)")


def _log_rows(text: str, *, source_path: str, fallback_run_id: str) -> list[dict[str, Any]]:
    run_match = re.search(r"^\[RUN\] dir=(.+)$", text, re.MULTILINE)
    run_id = Path(run_match.group(1).strip()).name if run_match else fallback_run_id
    snapshot_match = re.search(r"^\[RUN\] source_snapshot_sha256=([0-9a-f]{64})$", text, re.MULTILINE)
    snapshot = snapshot_match.group(1) if snapshot_match else None
    current_update: int | None = None
    by_update: dict[int, dict[str, Any]] = {}
    for line in text.splitlines():
        match = re.search(r"\[VALIDATION_START\] update=(\d+)", line)
        if match:
            current_update = int(match.group(1))
            by_update.setdefault(current_update, {})
            continue
        if current_update is None:
            continue
        if line.startswith("[VAL]"):
            tokens = {key: float(value) for key, value in _TOKEN_RE.findall(line)}
            by_update[current_update].update(
                {
                    "steps_mean": tokens.get("steps_mean"),
                    "steps_min": tokens.get("steps_min"),
                    "steps_max": tokens.get("steps_max"),
                    "return_mean": tokens.get("return"),
                }
            )
        elif line.startswith("[VAL_OUTCOME]"):
            tokens = {key: float(value) for key, value in _TOKEN_RE.findall(line)}
            by_update[current_update].update(
                {
                    "failure_frac": tokens.get("failure"),
                    "motion_complete_frac": tokens.get("motion_complete"),
                    "reference_progress_mean": tokens.get("reference_progress"),
                }
            )
    rows: list[dict[str, Any]] = []
    for update, metrics in by_update.items():
        row = _row_from_validation_metrics(
            update, metrics,
            run_id=run_id, source_snapshot=snapshot,
            source_kind="train_log", source_path=source_path,
        )
        # _row_from_validation_metrics accepts both prefixed and bare names.
        rows.append(row)
    return rows


def _deleted_git_logs(repo_root: Path) -> Iterable[tuple[str, str]]:
    completed = subprocess.run(
        ["git", "ls-files", "--deleted", "runs/*/logs/train.log"],
        cwd=repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=True,
    )
    for relative in completed.stdout.splitlines():
        blob = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=repo_root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True,
        ).stdout
        yield relative, blob


def _merge_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    merged: dict[tuple[str, int], dict[str, Any]] = {}
    errors: list[str] = []
    numeric = (
        "fixed_seed", "steps_mean", "steps_min", "steps_max", "return_mean",
        "failure_frac", "motion_complete_frac", "reference_progress_mean",
    )
    # Console validation lines intentionally round step/return metrics to two
    # decimals and outcome fractions to five decimals.  JSONL remains the
    # authoritative value; differences within the known print precision are
    # not conflicting evidence.
    print_tolerance = {
        "steps_mean": 5.1e-3,
        "steps_min": 5.1e-3,
        "steps_max": 5.1e-3,
        "return_mean": 5.1e-3,
        "failure_frac": 5.1e-6,
        "motion_complete_frac": 5.1e-6,
        "reference_progress_mean": 5.1e-6,
        "fixed_seed": 0.0,
    }
    for row in rows:
        key = (str(row["curve_id"]), int(row["update"]))
        if key not in merged:
            merged[key] = dict(row)
            continue
        target = merged[key]
        has_console_rounding = (
            "train_log" in target["source_kinds"]
            or "train_log" in row["source_kinds"]
        )
        for name in numeric:
            old = target.get(name)
            new = row.get(name)
            if old is None:
                target[name] = new
            elif new is not None and abs(float(old) - float(new)) > (
                print_tolerance[name] if has_console_rounding else 1.0e-12
            ):
                errors.append(
                    f"conflicting {name} for curve={key[0]} update={key[1]}: {old} vs {new}"
                )
        target["source_kinds"] = sorted(set(target["source_kinds"]) | set(row["source_kinds"]))
        target["source_paths"] = sorted(set(target["source_paths"]) | set(row["source_paths"]))
    return sorted(merged.values(), key=lambda row: (row["curve_id"], row["update"])), errors


def _checkpoint_index(output_dir: Path) -> dict[tuple[str, int], str]:
    csv_path = output_dir / "checkpoint_inventory.csv"
    if not csv_path.is_file():
        return {}
    result: dict[tuple[str, int], str] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("content_alias_of") or not row.get("update_payload"):
                continue
            run_id = Path(row["run_dir"]).name
            result[(run_id, int(row["update_payload"]))] = row["checkpoint_sha256"]
    return result


def main() -> int:
    args = _arguments()
    repo_root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = resolve_path(
        args.output_dir or spec_value(spec, "output_dir", default="output/largebox_discovery_v1"),
        base=repo_root,
    )
    assert output_dir is not None
    target = output_dir / "teacher_transition.json"
    table = output_dir / "tables" / "teacher_learning_curve.csv"
    try:
        manifest = read_json(output_dir / "manifest.json")
        rows: list[dict[str, Any]] = []
        for path in sorted(repo_root.glob("runs/*/logs/metrics.jsonl")):
            rows.extend(_metrics_jsonl_rows(path))
        for path in sorted(repo_root.glob("runs/*/logs/train.log")):
            rows.extend(_log_rows(path.read_text(encoding="utf-8"), source_path=str(path), fallback_run_id=path.parents[1].name))
        archived_logs: list[str] = []
        if args.include_git_deleted_logs:
            for relative, text in _deleted_git_logs(repo_root):
                archived_logs.append(relative)
                rows.extend(_log_rows(text, source_path=f"git:HEAD:{relative}", fallback_run_id=Path(relative).parents[1].name))
        merged, conflicts = _merge_rows(rows)
        checkpoint_index = _checkpoint_index(output_dir)
        for row in merged:
            checkpoint_hash = checkpoint_index.get((row["run_id"], int(row["update"])))
            row["checkpoint_available"] = checkpoint_hash is not None
            row["checkpoint_sha256"] = checkpoint_hash
            row["source_kinds"] = ";".join(row["source_kinds"])
            row["source_paths"] = ";".join(row["source_paths"])
        primary_run = Path(manifest["checkpoint_path"]).parent.parent.name
        curve_summaries: list[dict[str, Any]] = []
        for curve_id in sorted({row["curve_id"] for row in merged}):
            curve = [row for row in merged if row["curve_id"] == curve_id]
            complete = [row for row in curve if row.get("motion_complete_frac") is not None]
            summary = {
                "curve_id": curve_id,
                "run_id": curve[0]["run_id"],
                "source_snapshot_sha256": curve[0]["source_snapshot_sha256"],
                "updates": [row["update"] for row in curve],
                "physical_checkpoint_updates": [row["update"] for row in curve if row["checkpoint_available"]],
                "completion_crossing_updates": {
                    str(threshold): first_threshold_crossing(complete, "motion_complete_frac", threshold)
                    for threshold in (0.10, 0.50, 0.90)
                },
                "failure_below_updates": {
                    str(threshold): first_threshold_crossing(complete, "failure_frac", threshold, direction="below")
                    for threshold in (0.90, 0.50, 0.10)
                },
                "progress_crossing_updates": {
                    str(threshold): first_threshold_crossing(complete, "reference_progress_mean", threshold)
                    for threshold in (0.10, 0.50, 0.90)
                },
                "completion_monotonic_violations": monotonic_violations(complete, "motion_complete_frac", tolerance=1.0e-6),
                "progress_monotonic_violations": monotonic_violations(complete, "reference_progress_mean", tolerance=1.0e-6),
                "is_frozen_checkpoint_run": curve[0]["run_id"] == primary_run,
            }
            curve_summaries.append(summary)
        errors = list(conflicts)
        if not merged:
            errors.append("no clean validation points were found")
        primary = [item for item in curve_summaries if item["is_frozen_checkpoint_run"]]
        if not primary:
            errors.append("no learning curve belongs to the frozen checkpoint run")
        write_csv_exclusive(table, merged, fieldnames=CURVE_FIELDS)
        warnings = [
            "curves with different run/source_snapshot identities are intentionally not stitched",
            "log-only validation updates prove recorded behavior but cannot be resumed without a checkpoint entity",
            "the reconstructed data are clean deterministic validation, not stochastic occupancy",
        ]
        result = diagnostic_result(
            "05", PASS if not errors else INVALID_PROTOCOL,
            summary=(
                f"reconstructed {len(curve_summaries)} separate teacher curves with {len(merged)} validation points"
                if not errors else "teacher curve reconstruction is ambiguous"
            ),
            evidence={
                "frozen_run_id": primary_run,
                "curve_summaries": curve_summaries,
                "archived_deleted_logs_used": archived_logs,
                "table": str(table),
                "lineage_guard": "rows from distinct run/source snapshots remain distinct curves",
            },
            errors=errors,
            warnings=warnings,
        )
    except (FileNotFoundError, KeyError, ValueError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        write_csv_exclusive(table, [], fieldnames=CURVE_FIELDS)
        result = diagnostic_result(
            "05", INVALID_PROTOCOL, summary="teacher curve inputs are invalid", errors=[str(exc)]
        )
    write_json_exclusive(target, result)
    print(f"[diag_05] {result['status']} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
