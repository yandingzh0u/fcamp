#!/usr/bin/env python3
"""Compare clean, training-like, and controlled-noise validation protocols."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.manifest import (
    INVALID_PROTOCOL,
    PASS,
    SKIPPED_DEPENDENCY,
    diagnostic_result,
    load_spec,
    read_json,
    resolve_path,
    spec_value,
    write_json_exclusive,
)


REQUIRED_MODES = ("clean_mean", "training_like", "controlled_noise")
OUTCOME_FIELDS = (
    "steps_mean",
    "motion_complete_frac",
    "failure_frac",
    "reference_progress_mean",
    "return_mean",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--protocol-result",
        action="append",
        default=[],
        metavar="MODE=PATH",
        help="JSON/CSV result with checkpoint_sha256 and validation metrics",
    )
    return parser.parse_args()


def _normalize_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    for field in OUTCOME_FIELDS:
        for key in (field, f"validation/{field}"):
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                result[field] = float(value)
                break
    return result


def _load_result(
    path: Path,
    mode: str,
    manifest: dict[str, Any],
    *,
    frozen_run_artifact: bool = False,
) -> dict[str, Any]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"empty protocol CSV: {path}")
        row = rows[-1]
        metrics = {
            key: float(value)
            for key, value in row.items()
            if value not in (None, "")
            and key.startswith("validation/")
        }
        metadata: dict[str, Any] = {}
    else:
        value = read_json(path)
        if not isinstance(value, dict):
            raise ValueError(f"protocol result must be a JSON object: {path}")
        metrics = value.get("metrics", value.get("evidence", {}).get("metrics", value))
        if not isinstance(metrics, dict):
            raise ValueError(f"protocol result has no metrics mapping: {path}")
        metadata = value
    return {
        "mode": mode,
        "path": str(path),
        "checkpoint_sha256": metadata.get("checkpoint_sha256") or (
            manifest["checkpoint_sha256"] if frozen_run_artifact else None
        ),
        "checkpoint_update": metadata.get("checkpoint_update") or (
            manifest["checkpoint_update"] if frozen_run_artifact else None
        ),
        "metrics": _normalize_metrics(metrics),
    }


def _discover_clean(manifest: dict[str, Any]) -> Path | None:
    checkpoint = Path(manifest["checkpoint_path"])
    run_dir = checkpoint.parent.parent
    candidate = run_dir / "logs" / "validation_summary.csv"
    return candidate if candidate.is_file() else None


def _discover_collected_protocol(output_dir: Path, mode: str) -> Path | None:
    """Find the condition-matched Stage-1 summary used by DAG execution.

    ``diag_04`` runs after ``diag_12`` in the frozen DAG.  Command-line paths
    remain useful for standalone audits, but the suite runner must not need an
    out-of-band argument to consume its declared dependency.
    """

    directory = output_dir / "validation_protocols"
    for suffix in (".json", ".csv"):
        candidate = directory / f"{mode}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _frozen_summary_checkpoint(
    spec: dict[str, Any], output_dir: Path
) -> tuple[int, str, str]:
    """Resolve the validation checkpoint frozen by the collection protocol.

    The stage-0 manifest intentionally identifies the *primary lineage audit*
    entity (u200).  Validation protocol summaries are independently frozen at
    u500.  Conflating those two roles turns three correctly condition-matched
    u500 recollections into a false protocol error.
    """

    update = int(spec["collection"]["validation_protocols"]["summary_checkpoint_update"])
    inventory_path = output_dir / "checkpoints" / "dense_checkpoint_inventory.csv"
    with inventory_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if int(row.get("required_update", -1)) == update
            and str(row.get("present", "")).lower() == "true"
        ]
    if len(rows) != 1:
        raise ValueError(
            f"validation summary requires exactly one present u{update} dense checkpoint"
        )
    digest = str(rows[0].get("checkpoint_sha256", ""))
    path = str(rows[0].get("checkpoint_path", ""))
    if len(digest) != 64 or not path:
        raise ValueError("validation summary checkpoint identity is incomplete")
    return update, digest, path


def main() -> int:
    args = _arguments()
    repo_root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = resolve_path(
        args.output_dir or spec_value(spec, "output_dir", default="output/largebox_discovery_v1"),
        base=repo_root,
    )
    assert output_dir is not None
    target = output_dir / "protocol_gap.json"
    try:
        manifest = read_json(output_dir / "manifest.json")
        frozen_update, frozen_digest, frozen_path = _frozen_summary_checkpoint(
            spec, output_dir
        )
        mappings: dict[str, Path] = {}
        for item in args.protocol_result:
            if "=" not in item:
                raise ValueError(f"--protocol-result expects MODE=PATH, got {item!r}")
            mode, raw_path = item.split("=", 1)
            if mode not in REQUIRED_MODES or mode in mappings:
                raise ValueError(f"unknown or duplicate protocol mode {mode!r}")
            mappings[mode] = Path(raw_path).expanduser().resolve()
        for mode in REQUIRED_MODES:
            if mode not in mappings:
                collected = _discover_collected_protocol(output_dir, mode)
                if collected is not None:
                    mappings[mode] = collected
        discovered_clean = None
        if "clean_mean" not in mappings:
            clean = _discover_clean(manifest)
            if clean is not None:
                mappings["clean_mean"] = clean
                discovered_clean = clean.resolve()
        results = {
            mode: _load_result(
                path,
                mode,
                manifest,
                frozen_run_artifact=(
                    mode == "clean_mean"
                    and discovered_clean is not None
                    and path.resolve() == discovered_clean
                ),
            )
            for mode, path in mappings.items()
        }
        errors: list[str] = []
        for mode, item in results.items():
            if item["checkpoint_sha256"] != frozen_digest:
                errors.append(f"{mode} used a different checkpoint")
            if item["checkpoint_update"] != frozen_update:
                errors.append(f"{mode} used a different checkpoint update")
            missing_fields = sorted(set(OUTCOME_FIELDS) - set(item["metrics"]))
            if missing_fields:
                errors.append(f"{mode} is missing outcomes: {missing_fields}")
        missing_modes = [mode for mode in REQUIRED_MODES if mode not in results]
        deltas: dict[str, Any] = {}
        if "clean_mean" in results:
            clean_metrics = results["clean_mean"]["metrics"]
            for mode, item in results.items():
                if mode == "clean_mean":
                    continue
                deltas[mode] = {
                    field: item["metrics"][field] - clean_metrics[field]
                    for field in OUTCOME_FIELDS
                    if field in item["metrics"] and field in clean_metrics
                }
        if errors:
            status = INVALID_PROTOCOL
            summary = "validation protocols are not condition-matched"
        elif missing_modes:
            status = SKIPPED_DEPENDENCY
            summary = "clean validation exists, but training-like and controlled-noise recollections are still required"
        else:
            status = PASS
            summary = "all three validation protocols were compared on the frozen checkpoint"
        result = diagnostic_result(
            "04", status, summary=summary,
            evidence={
                "frozen_checkpoint_sha256": frozen_digest,
                "frozen_checkpoint_update": frozen_update,
                "frozen_checkpoint_path": frozen_path,
                "checkpoint_identity_role": (
                    "collection.validation_protocols.summary_checkpoint_update; "
                    "distinct from the stage-0 primary lineage-audit checkpoint"
                ),
                "required_modes": list(REQUIRED_MODES),
                "observed_modes": sorted(results),
                "missing_modes": missing_modes,
                "protocols": results,
                "deltas_from_clean_mean": deltas,
                "interpretation_guard": "clean deterministic success is capability evidence, never stochastic occupancy evidence",
            },
            errors=errors,
            warnings=(
                ["missing rollout modes are a dependency gap, not a scientific failure"]
                if missing_modes else []
            ),
        )
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        result = diagnostic_result(
            "04", INVALID_PROTOCOL, summary="validation gap protocol is invalid", errors=[str(exc)]
        )
    write_json_exclusive(target, result)
    print(f"[diag_04] {result['status']} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
