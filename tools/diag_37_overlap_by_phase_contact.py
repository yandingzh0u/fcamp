#!/usr/bin/env python3
"""Stratify empirical overlap by frozen phase, contact, and failure cells."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.domain_data import SPLITS, load_domain_index
from diagnostics.common.domain_triangle import effective_overlap_audit
from diagnostics.common.manifest import (
    INVALID_PROTOCOL, PASS, SKIPPED_DEPENDENCY, DependencyUnavailable,
    ProtocolError, diagnostic_result, load_spec, write_json_exclusive,
)
from diagnostics.common.policy_class_probe import output_dir_from_args


def _cell_mask(bundle, *, phase_bin: int, bins: int, contact: str, failure: bool) -> np.ndarray:
    phases = np.asarray(bundle.phase, dtype=np.float64)
    phase_indices = np.minimum((phases * bins).astype(np.int64), bins - 1)
    return (
        (phase_indices == int(phase_bin))
        & (np.asarray(bundle.contact_mode).astype(str) == str(contact))
        & (np.asarray(bundle.failure, dtype=bool) == bool(failure))
    )


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
    target = output_dir / "conditional_overlap_matrices.json"
    try:
        protocols = spec.get("analysis_protocols", {})
        condition = protocols.get("conditional_overlap", {})
        overlap_protocol = protocols.get("effective_overlap", {})
        seeds = tuple(int(value) for value in protocols.get("source_classifier", {}).get("seeds", ()))
        bins = int(condition.get("phase_bins", 0))
        phase_range = condition.get("phase_range")
        minima = {
            "train": int(condition.get("minimum_train_samples_per_domain", 0)),
            "validation": int(condition.get("minimum_validation_samples_per_domain", 0)),
            "test": int(condition.get("minimum_test_samples_per_domain", 0)),
        }
        if (
            bins != 16 or phase_range != [0.0, 1.0] or len(seeds) != 5
            or condition.get("sparse_stratum_status") != SKIPPED_DEPENDENCY
            or condition.get("contact_strata") != "observed_foot_contact_bitmask"
            or any(value <= 0 for value in minima.values())
        ):
            raise ProtocolError("conditional overlap protocol differs from the frozen contract")
        taus = tuple(float(value) for value in overlap_protocol.get("posterior_overlap_taus", ()))
        clips = tuple(float(value) for value in overlap_protocol.get("ratio_sensitivity_clips", ()))
        k = int(overlap_protocol.get("knn_neighbors", 0))
        if taus != (0.05, 0.1, 0.2) or clips != (10.0, 20.0, 100.0) or k != 5:
            raise ProtocolError("effective-overlap analysis protocol is incomplete")
        index_path = (args.domain_index or output_dir / "domain_triangle_index.json").expanduser().resolve()
        _, bundles = load_domain_index(index_path)
        for bundle in bundles.values():
            phases = np.asarray(bundle.phase, dtype=np.float64)
            if np.any(phases < 0.0) or np.any(phases > 1.0):
                raise ProtocolError(f"domain {bundle.name} phase is not normalized to [0,1]")
        names = sorted(bundles)
        cells = []
        eligible_count = 0
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                first, second = names[left], names[right]
                contacts = sorted(
                    set(np.asarray(bundles[first].contact_mode).astype(str).tolist())
                    | set(np.asarray(bundles[second].contact_mode).astype(str).tolist())
                )
                for phase_bin in range(bins):
                    for contact in contacts:
                        for failure in (False, True):
                            masks = {
                                first: _cell_mask(bundles[first], phase_bin=phase_bin, bins=bins, contact=contact, failure=failure),
                                second: _cell_mask(bundles[second], phase_bin=phase_bin, bins=bins, contact=contact, failure=failure),
                            }
                            counts = {
                                name: {
                                    split: int(np.sum(masks[name] & (np.asarray(bundles[name].split).astype(str) == split)))
                                    for split in SPLITS
                                }
                                for name in (first, second)
                            }
                            sparse = any(counts[name][split] < minima[split] for name in (first, second) for split in SPLITS)
                            cell = {
                                "first": first, "second": second, "phase_bin": phase_bin,
                                "phase_interval": [phase_bin / bins, (phase_bin + 1) / bins],
                                "contact_mode": contact, "failure": failure,
                                "sample_counts": counts,
                            }
                            if sparse:
                                cell.update({
                                    "status": SKIPPED_DEPENDENCY,
                                    "reason": "cell does not meet preregistered per-domain split minima",
                                })
                            else:
                                audits = []
                                for seed in seeds:
                                    views = {}
                                    for name in (first, second):
                                        selected = bundles[name].select(masks[name])
                                        views[name] = {split: selected.split_features(split) for split in SPLITS}
                                    audits.append(effective_overlap_audit(
                                        views[first]["validation"], views[second]["validation"],
                                        views[first]["test"], views[second]["test"],
                                        source_train=views[first]["train"],
                                        target_train=views[second]["train"],
                                        seed=seed, taus=taus, ratio_clips=clips, k=k,
                                    ))
                                cell.update({"status": PASS, "by_seed": audits})
                                eligible_count += 1
                            cells.append(cell)
        status = PASS if eligible_count else SKIPPED_DEPENDENCY
        result = diagnostic_result(
            "37", status,
            summary=(
                "conditional empirical-overlap cells were evaluated"
                if eligible_count else "all conditional cells were sparse under preregistered minima"
            ),
            evidence={
                "phase_bins": bins, "sample_minima": minima,
                "eligible_cell_count": eligible_count, "total_cell_count": len(cells),
                "cells": cells, "native_stochastic_excluded": True,
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result("37", SKIPPED_DEPENDENCY, summary="conditional overlap awaits domain bundles", errors=[str(exc)])
    except (FileNotFoundError, ProtocolError, ValueError, KeyError) as exc:
        result = diagnostic_result("37", INVALID_PROTOCOL, summary="conditional overlap protocol failed closed", errors=[str(exc)])
    write_json_exclusive(target, result)
    print(f"[diag_37] {result['status']} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
