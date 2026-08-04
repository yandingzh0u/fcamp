#!/usr/bin/env python3
"""Decompose motion-file, reset-readback, one-step, and closed-loop gaps."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.domain_data import GAP_LAYER_NAMES, load_gap_layer_bundle
from diagnostics.common.domain_triangle import normalized_gap
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
    parser.add_argument("--gap-bundle", type=Path, default=None)
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(root, spec, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "gap_decomposition.json"
    try:
        source = (args.gap_bundle or output_dir / "reference_readback_execution_layers.npz").expanduser().resolve()
        bundle = load_gap_layer_bundle(source)
        transitions = (
            ("motion_to_reset_readback", "motion_npz", "reset_readback"),
            ("reset_readback_to_teacher_one_step", "reset_readback", "teacher_one_step"),
            ("teacher_one_step_to_closed_loop", "teacher_one_step", "teacher_closed_loop"),
            ("motion_to_teacher_closed_loop", "motion_npz", "teacher_closed_loop"),
        )
        gaps = {}
        for label, first, second in transitions:
            record = normalized_gap(bundle.layers[first], bundle.layers[second], scale=bundle.scale)
            record["by_feature_group"] = {
                name: normalized_gap(
                    bundle.layers[first][:, start:stop],
                    bundle.layers[second][:, start:stop],
                    scale=bundle.scale[start:stop],
                )
                for name, (start, stop) in bundle.feature_groups.items()
            }
            gaps[label] = record
        result = diagnostic_result(
            "32", PASS,
            summary="same-phase domain gap was separated into four causal execution layers",
            evidence={
                "layers": list(GAP_LAYER_NAMES), "sample_count": int(bundle.phase.size),
                "gap_bundle": str(source), "gaps": gaps,
                "alignment": bundle.metadata["alignment"],
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "32", SKIPPED_DEPENDENCY,
            summary="four-layer PhysX readback experiment has not been collected",
            errors=[str(exc)],
        )
    except (FileNotFoundError, ProtocolError, ValueError, KeyError) as exc:
        result = diagnostic_result(
            "32", INVALID_PROTOCOL,
            summary="four-layer gap decomposition failed closed",
            errors=[str(exc)],
        )
    write_json_exclusive(target, result)
    print(f"[diag_32] {result['status']} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
