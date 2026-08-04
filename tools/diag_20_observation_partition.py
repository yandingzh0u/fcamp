#!/usr/bin/env python3
"""AST-derive leakage-safe named actor/critic observation partitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.manifest import (
    INVALID_PROTOCOL,
    PASS,
    DependencyUnavailable,
    ProtocolError,
    diagnostic_result,
    load_spec,
    write_json_exclusive,
)
from diagnostics.common.policy_class_probe import (
    derive_repository_observation_specs,
    output_dir_from_args,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    repo_root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(repo_root, spec, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    partition_path = output_dir / "observation_spec.json"
    status_path = output_dir / "observation_spec.status.json"
    try:
        specs, provenance = derive_repository_observation_specs(repo_root)
        actor_reference = [term.name for term in specs["actor"].reference_terms]
        actor_proprio = [term.name for term in specs["actor"].proprio_terms]
        partition = {
            "diagnostic_id": "20",
            "status": PASS,
            "schema_version": "1.0.0",
            **provenance,
            "actor": specs["actor"].to_dict(),
            "critic": specs["critic"].to_dict(),
            "actor_partition": {
                "reference_terms": actor_reference,
                "proprio_terms": actor_proprio,
                "reference_width": int(sum(term.width for term in specs["actor"].reference_terms)),
                "proprio_width": int(sum(term.width for term in specs["actor"].proprio_terms)),
                "complete": True,
                "disjoint": True,
            },
            "derivation_guard": (
                "term order is parsed from production torch.cat AST; cardinalities are parsed "
                "from production source constants; no diagnostic offset is hand-written"
            ),
        }
        write_json_exclusive(partition_path, partition)
        result = diagnostic_result(
            "20",
            PASS,
            summary="actor and critic named slices were derived from production source",
            evidence={
                "observation_spec": str(partition_path),
                "actor_dim": specs["actor"].total_dim,
                "critic_dim": specs["critic"].total_dim,
                "actor_reference_terms": actor_reference,
                "actor_proprio_terms": actor_proprio,
                "source_sha256": specs["actor"].source_sha256,
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "20", "SKIPPED_DEPENDENCY",
            summary="observation source dependency is unavailable",
            errors=[str(exc)],
        )
    except (FileNotFoundError, ProtocolError, ValueError, SyntaxError, json.JSONDecodeError) as exc:
        result = diagnostic_result(
            "20", INVALID_PROTOCOL,
            summary="observation partition could not be derived fail-closed",
            errors=[str(exc)],
        )
    write_json_exclusive(status_path, result)
    print(f"[diag_20] {result['status']} {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
