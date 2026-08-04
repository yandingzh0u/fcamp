#!/usr/bin/env python3
"""Validate and index aligned K/T/A/B domain-window bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.domain_data import build_domain_index
from diagnostics.common.manifest import (
    INVALID_PROTOCOL, PASS, SKIPPED_DEPENDENCY, DependencyUnavailable,
    ProtocolError, diagnostic_result, load_spec, write_json_exclusive,
)
from diagnostics.common.noise_bank import NoiseProtocolError
from diagnostics.common.policy_class_probe import output_dir_from_args


def _catalog_from_frozen_spec(
    spec: dict,
    *,
    repo_root: Path,
    output_dir: Path,
    target: Path,
) -> Path:
    protocol = spec.get("analysis_protocols", {}).get("domain_triangle_domains")
    if not isinstance(protocol, dict):
        raise DependencyUnavailable(
            "domain_catalog.json is absent and analysis_protocols.domain_triangle_domains is not frozen"
        )
    entries = protocol.get("catalog")
    if (
        protocol.get("protocol_version") != "largebox_domain_triangle_v1"
        or protocol.get("primary_collector_mode") != "controlled_environment"
        or protocol.get("failure_label") != "trajectory_eventual"
        or not isinstance(entries, list)
        or not entries
    ):
        raise ProtocolError("domain_triangle_domains protocol/header is not the frozen primary contract")
    expected = {
        "K": ("K", "reference_expert"),
        "T_u200": ("T_early", "agent_physx"),
        "T_u500": ("T", "agent_physx"),
        "A_amp": ("A_amp", "agent_physx"),
        "B": ("B", "agent_physx"),
    }
    actual = {
        str(entry.get("name")): (str(entry.get("family")), str(entry.get("frame_role")))
        for entry in entries if isinstance(entry, dict)
    }
    if actual != expected:
        raise ProtocolError(f"domain triangle catalog identities changed: {actual}")
    resolved = []
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            raise ProtocolError("domain_triangle_domains entries must be mappings")
        entry = dict(raw_entry)
        for field in ("rollout_index", "bundle_path"):
            if field not in entry:
                continue
            rendered = str(entry[field]).format(
                repo_root=str(repo_root), output_dir=str(output_dir)
            )
            path = Path(rendered).expanduser()
            # Frozen catalog filenames name formal suite-output assets.  A
            # caller can still use {repo_root} to name a repository asset.
            entry[field] = str((path if path.is_absolute() else output_dir / path).resolve())
        resolved.append(entry)
    write_json_exclusive(
        target,
        {
            "schema_version": "1.0.0",
            "generated_from_frozen_spec": True,
            "suite_id": spec.get("suite_id"),
            "protocol": {
                key: value for key, value in protocol.items() if key != "catalog"
            },
            "domains": resolved,
        },
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--domain-catalog", type=Path, default=None)
    args = parser.parse_args()
    repo_root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(repo_root, spec, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "domain_triangle_index.json"
    status_path = output_dir / "domain_triangle_index.status.json"
    try:
        protocols = spec.get("analysis_protocols")
        if not isinstance(protocols, dict) or not isinstance(protocols.get("temporal_windows"), dict):
            raise ProtocolError("analysis_protocols.temporal_windows is not frozen")
        window = protocols["temporal_windows"]
        required_window = {
            "steps": 10,
            "stride": 1,
            "require_complete_alive_window": True,
            "allow_reset_crossing": False,
            "allow_wrap_crossing": False,
        }
        if any(window.get(key) != value for key, value in required_window.items()):
            raise ProtocolError("temporal window protocol differs from the frozen Stage-3 contract")
        if args.domain_catalog is not None:
            catalog = args.domain_catalog.expanduser().resolve()
        else:
            catalog = output_dir / "domain_catalog.json"
            if not catalog.is_file():
                catalog = _catalog_from_frozen_spec(
                    spec, repo_root=repo_root, output_dir=output_dir, target=catalog
                )
        index = build_domain_index(
            catalog,
            derived_output_dir=output_dir / "domain_bundles",
            split_audit_path=output_dir / "split_audit.json",
            split_source_index=output_dir / "canonical_rollout_index.parquet",
        )
        wrong = [
            record["name"] for record in index["domains"]
            if int(record.get("feature_width", 0)) <= 0
            or int(record.get("window_steps", -1)) != int(window["steps"])
        ]
        if wrong:
            raise ProtocolError(f"empty domain feature views: {wrong}")
        # Bundle schema hashes include the ten-step window contract; requiring
        # one hash closes silent K/T/A coordinate mismatches.
        index.update({
            "diagnostic_id": "30",
            "temporal_window_protocol": window,
            "feature_role": "shared physical motion window; reference actor terms are excluded",
        })
        write_json_exclusive(index_path, index)
        result = diagnostic_result(
            "30", PASS,
            summary="K/T/A/B windows share one validated physical coordinate and split contract",
            evidence={
                "domain_index": str(index_path),
                "domains": [record["name"] for record in index["domains"]],
                "feature_schema_sha256": index["feature_schema_sha256"],
                "native_stochastic_excluded": True,
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "30", SKIPPED_DEPENDENCY,
            summary="domain triangle awaits real K/T_u200/T_u500/A_amp/B bundles",
            errors=[str(exc)],
        )
    except (FileNotFoundError, ProtocolError, NoiseProtocolError, ValueError, KeyError, json.JSONDecodeError) as exc:
        result = diagnostic_result(
            "30", INVALID_PROTOCOL,
            summary="domain triangle alignment failed closed",
            errors=[str(exc)],
        )
    write_json_exclusive(status_path, result)
    print(f"[diag_30] {result['status']} {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
