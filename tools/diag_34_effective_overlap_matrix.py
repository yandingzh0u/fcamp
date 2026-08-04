#!/usr/bin/env python3
"""Compute explicitly bidirectional empirical effective-overlap matrices."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.domain_data import load_domain_index
from diagnostics.common.domain_triangle import effective_overlap_audit
from diagnostics.common.manifest import (
    INVALID_PROTOCOL, PASS, SKIPPED_DEPENDENCY, DependencyUnavailable,
    ProtocolError, diagnostic_result, load_spec, write_csv_exclusive,
    write_json_exclusive,
)
from diagnostics.common.policy_class_probe import output_dir_from_args


def _mean(records, getter):
    return float(np.mean([float(getter(record)) for record in records]))


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
    status_path = output_dir / "effective_overlap_matrix.status.json"
    try:
        protocols = spec.get("analysis_protocols", {})
        overlap_protocol = protocols.get("effective_overlap", {})
        seeds = tuple(int(value) for value in protocols.get("source_classifier", {}).get("seeds", ()))
        threshold = spec.get("thresholds", {}).get("empirical_effective_overlap_edge", {})
        taus = tuple(float(value) for value in threshold.get("posterior_overlap_report_taus", ()))
        clips = tuple(float(value) for value in threshold.get("ratio_clips", ()))
        frozen_overlap = (
            int(overlap_protocol.get("knn_neighbors", 0)) == 5
            and float(overlap_protocol.get("within_domain_radius_quantile", -1)) == 0.95
            and tuple(float(value) for value in overlap_protocol.get("posterior_overlap_taus", ())) == (0.05, 0.1, 0.2)
            and float(overlap_protocol.get("density_ratio_probability_epsilon", -1)) == 1.0e-6
            and tuple(float(value) for value in overlap_protocol.get("ratio_sensitivity_clips", ())) == (10.0, 20.0, 100.0)
            and int(overlap_protocol.get("bootstrap_replicates", 0)) == 500
            and float(overlap_protocol.get("bootstrap_confidence", -1)) == 0.95
            and overlap_protocol.get("gate_ess_variant") == "unclipped"
        )
        if len(seeds) != 5 or taus != (0.05, 0.1, 0.2) or clips != (10.0, 20.0, 100.0) or not frozen_overlap:
            raise ProtocolError("effective-overlap seeds/taus/clips differ from the frozen protocol")
        index_path = (args.domain_index or output_dir / "domain_triangle_index.json").expanduser().resolve()
        _, bundles = load_domain_index(index_path)
        names = sorted(bundles)
        forward = {name: {other: (1.0 if name == other else float("nan")) for other in names} for name in names}
        reverse = {name: {other: (1.0 if name == other else float("nan")) for other in names} for name in names}
        pair_records = []
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                first, second = names[left], names[right]
                source, target = bundles[first], bundles[second]
                audits = [
                    effective_overlap_audit(
                        source.split_features("validation"), target.split_features("validation"),
                        source.split_features("test"), target.split_features("test"),
                        source_train=source.split_features("train"),
                        target_train=target.split_features("train"),
                        seed=seed, taus=taus, ratio_clips=clips,
                    )
                    for seed in seeds
                ]
                forward_value = _mean(audits, lambda record: record["knn"]["source_to_target_coverage"])
                reverse_value = _mean(audits, lambda record: record["knn"]["target_to_source_coverage"])
                forward[first][second] = forward_value
                forward[second][first] = reverse_value
                reverse[first][second] = reverse_value
                reverse[second][first] = forward_value
                pair_records.append({
                    "source": first, "target": second,
                    "mean_auc": _mean(audits, lambda record: record["classifier"]["auc"]),
                    "posterior_overlap": {
                        str(tau): _mean(audits, lambda record, tau=tau: record["posterior_overlap"][str(tau)]["balanced"])
                        for tau in taus
                    },
                    "forward_knn_coverage": forward_value,
                    "reverse_knn_coverage": reverse_value,
                    "forward_label_mixing": _mean(audits, lambda record: record["knn"]["source_label_mixing"]),
                    "reverse_label_mixing": _mean(audits, lambda record: record["knn"]["target_label_mixing"]),
                    "ratio_ess": {
                        direction: {
                            label: {
                                "ess_fraction_mean": _mean(audits, lambda record, direction=direction, label=label: record["ratio_ess"][direction][label]["ess_fraction"]),
                                "bootstrap_lower_mean": _mean(audits, lambda record, direction=direction, label=label: record["ratio_ess"][direction][label]["bootstrap"]["lower"]),
                                "bootstrap_upper_mean": _mean(audits, lambda record, direction=direction, label=label: record["ratio_ess"][direction][label]["bootstrap"]["upper"]),
                            }
                            for label in ("unclipped", *(str(value) for value in clips))
                        }
                        for direction in ("forward", "reverse")
                    },
                    "by_seed": audits,
                })
        tables = output_dir / "tables"
        write_csv_exclusive(
            tables / "effective_overlap_forward.csv",
            [{"domain": name, **forward[name]} for name in names], fieldnames=("domain", *names),
        )
        write_csv_exclusive(
            tables / "effective_overlap_reverse.csv",
            [{"domain": name, **reverse[name]} for name in names], fieldnames=("domain", *names),
        )
        detail_path = output_dir / "effective_overlap_details.json"
        write_json_exclusive(detail_path, {
            "diagnostic_id": "34", "status": PASS, "seeds": list(seeds),
            "pairs": pair_records,
            "interpretation": "empirical effective overlap; never mathematical support or policy reachability",
            "native_stochastic_excluded": True,
        })
        result = diagnostic_result(
            "34", PASS, summary="bidirectional empirical-overlap diagnostics were computed without native policy noise",
            evidence={"forward_matrix": str(tables / "effective_overlap_forward.csv"), "reverse_matrix": str(tables / "effective_overlap_reverse.csv"), "details": str(detail_path)},
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result("34", SKIPPED_DEPENDENCY, summary="effective-overlap matrix awaits domain bundles", errors=[str(exc)])
    except (FileNotFoundError, ProtocolError, ValueError, KeyError) as exc:
        result = diagnostic_result("34", INVALID_PROTOCOL, summary="effective-overlap protocol failed closed", errors=[str(exc)])
    write_json_exclusive(status_path, result)
    print(f"[diag_34] {result['status']} {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
