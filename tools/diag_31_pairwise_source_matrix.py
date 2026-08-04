#!/usr/bin/env python3
"""Train five-seed calibrated source probes for every Stage-3 domain pair."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.domain_data import domain_split, load_domain_index
from diagnostics.common.domain_triangle import pairwise_source_matrix
from diagnostics.common.manifest import (
    INVALID_PROTOCOL, PASS, SKIPPED_DEPENDENCY, DependencyUnavailable,
    ProtocolError, diagnostic_result, load_spec, write_csv_exclusive,
    write_json_exclusive,
)
from diagnostics.common.policy_class_probe import output_dir_from_args


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--domain-index", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(root, spec, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "pairwise_source_matrix.status.json"
    try:
        protocol = spec.get("analysis_protocols", {}).get("source_classifier", {})
        seeds = tuple(int(value) for value in protocol.get("seeds", ()))
        if (
            len(seeds) != 5
            or protocol.get("model") != "standardized_balanced_logistic_regression"
            or protocol.get("calibration") != "sigmoid_3fold_train_only"
            or int(protocol.get("max_iterations", 0)) != 1000
        ):
            raise ProtocolError("five-seed source-classifier protocol is not frozen")
        index_path = (args.domain_index or output_dir / "domain_triangle_index.json").expanduser().resolve()
        _, bundles = load_domain_index(index_path)
        matrix, details = pairwise_source_matrix(
            {name: domain_split(bundle) for name, bundle in bundles.items()},
            seeds=seeds,
            calibrate=True,
        )
        names = sorted(matrix)
        table_path = output_dir / "tables" / "source_auc_matrix.csv"
        write_csv_exclusive(
            table_path,
            [{"domain": name, **{other: matrix[name][other] for other in names}} for name in names],
            fieldnames=("domain", *names),
        )
        detail_path = output_dir / "pairwise_source_matrix.json"
        serialized = []
        for (first, second), records in details.items():
            serialized.append({
                "first": first,
                "second": second,
                "seeds": [
                    {key: value for key, value in record.items() if key not in {"probabilities", "labels"}}
                    for record in records
                ],
                "mean_auc": matrix[first][second],
                "same_held_out_bank_across_seeds": True,
            })
        write_json_exclusive(detail_path, {
            "diagnostic_id": "31", "status": PASS, "seeds": list(seeds),
            "pairs": serialized,
            "semantics": "finite-sample source separability; not motion quality or reachability",
            "native_stochastic_excluded": True,
        })
        result = diagnostic_result(
            "31", PASS, summary="all domain pairs received five calibrated held-out source probes",
            evidence={"auc_matrix": str(table_path), "details": str(detail_path), "seeds": list(seeds)},
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result("31", SKIPPED_DEPENDENCY, summary="source matrix awaits domain bundles", errors=[str(exc)])
    except (FileNotFoundError, ProtocolError, ValueError, KeyError) as exc:
        result = diagnostic_result("31", INVALID_PROTOCOL, summary="pairwise source protocol failed closed", errors=[str(exc)])
    write_json_exclusive(status_path, result)
    print(f"[diag_31] {result['status']} {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
