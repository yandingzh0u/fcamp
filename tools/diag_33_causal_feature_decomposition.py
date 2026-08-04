#!/usr/bin/env python3
"""Run group-swap source interventions for every aligned domain pair."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.domain_data import domain_split, load_domain_index
from diagnostics.common.domain_triangle import causal_group_swap_audit
from diagnostics.common.manifest import (
    INVALID_PROTOCOL, PASS, SKIPPED_DEPENDENCY, DependencyUnavailable,
    ProtocolError, diagnostic_result, load_spec, write_json_exclusive,
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
    target = output_dir / "causal_sources.json"
    try:
        seeds = tuple(int(value) for value in spec.get("analysis_protocols", {}).get("source_classifier", {}).get("seeds", ()))
        if len(seeds) != 5:
            raise ProtocolError("causal source decomposition requires the frozen five seeds")
        index_path = (args.domain_index or output_dir / "domain_triangle_index.json").expanduser().resolve()
        _, bundles = load_domain_index(index_path)
        names = sorted(bundles)
        pairs = []
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                first, second = names[left], names[right]
                seed_records = [
                    causal_group_swap_audit(
                        domain_split(bundles[first]), domain_split(bundles[second]),
                        groups=bundles[first].feature_groups, seed=seed,
                    )
                    for seed in seeds
                ]
                aggregates = {}
                for group in bundles[first].feature_groups:
                    records = [record["groups"][group] for record in seed_records]
                    aggregates[group] = {
                        key: float(np.mean([float(record[key]) for record in records]))
                        for key in (
                            "group_only_frozen_auc", "erase_group_frozen_auc",
                            "baseline_auc_minus_erasure_auc",
                            "swap_auc", "swap_toward_donor_probability_mean",
                        )
                    }
                metric_names = (
                    "group_only_frozen_auc", "erase_group_frozen_auc",
                    "baseline_auc_minus_erasure_auc", "swap_auc",
                    "swap_toward_donor_probability_mean",
                )
                by_physical_block = {}
                by_history_step = {}
                for group, record in aggregates.items():
                    if "." not in group:
                        continue
                    step, block = group.split(".", 1)
                    by_physical_block.setdefault(block, []).append(record)
                    by_history_step.setdefault(step, []).append(record)

                def summarize(grouped):
                    return {
                        name: {
                            metric: float(np.mean([row[metric] for row in rows]))
                            for metric in metric_names
                        }
                        for name, rows in grouped.items()
                    }
                pairs.append({
                    "first": first, "second": second, "by_seed": seed_records,
                    "mean_by_group": aggregates,
                    "mean_by_physical_block": summarize(by_physical_block),
                    "mean_by_history_step": summarize(by_history_step),
                })
        result = diagnostic_result(
            "33", PASS,
            summary="held-out group swaps localized source-separating information",
            evidence={
                "seeds": list(seeds), "pairs": pairs,
                "scope": "diagnosis only; no feature is proposed for masking or manual reweighting",
                "native_stochastic_excluded": True,
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result("33", SKIPPED_DEPENDENCY, summary="causal source audit awaits domain bundles", errors=[str(exc)])
    except (FileNotFoundError, ProtocolError, ValueError, KeyError) as exc:
        result = diagnostic_result("33", INVALID_PROTOCOL, summary="causal feature protocol failed closed", errors=[str(exc)])
    write_json_exclusive(target, result)
    print(f"[diag_33] {result['status']} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
