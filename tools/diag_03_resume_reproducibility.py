#!/usr/bin/env python3
"""Compare two independent one-update resumes from one frozen RNG state."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.checkpoint_io import compare_checkpoint_payloads, load_checkpoint
from diagnostics.common.manifest import (
    FAIL,
    INVALID_PROTOCOL,
    PASS,
    SKIPPED_DEPENDENCY,
    diagnostic_result,
    load_spec,
    read_json,
    resolve_path,
    sha256_file,
    spec_value,
    write_json_exclusive,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--result-a", type=Path, default=None)
    parser.add_argument("--result-b", type=Path, default=None)
    parser.add_argument("--metric-atol", type=float, default=0.0)
    parser.add_argument("--metric-rtol", type=float, default=0.0)
    return parser.parse_args()


def _resume_command(repo_root: Path, config: Path, checkpoint: Path, update: int, name: str) -> str:
    parts = [
        "conda", "run", "-n", "env_isaaclab", "python", str(repo_root / "train.py"),
        "--config", str(config), "--run_name", name,
        "--set", f"training.resume={checkpoint}",
        "--set", f"training.max_updates={update + 1}",
        "--set", "training.save_every=1",
        "--set", "training.validation_every=0",
        "--set", "training.reset_optimizer_on_resume=false",
        "--set", "training.reset_sampler_on_resume=false",
    ]
    return " ".join(shlex.quote(value) for value in parts)


def main() -> int:
    args = _arguments()
    repo_root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = resolve_path(
        args.output_dir or spec_value(spec, "output_dir", default="output/largebox_discovery_v1"),
        base=repo_root,
    )
    assert output_dir is not None
    target = output_dir / "resume_diff.json"
    try:
        manifest = read_json(output_dir / "manifest.json")
        inventory = read_json(output_dir / "checkpoint_inventory.status.json")
        if manifest.get("status") != PASS or inventory.get("status") != PASS:
            raise ValueError("manifest and checkpoint inventory must PASS before resume audit")
        checkpoint = (args.checkpoint or Path(manifest["checkpoint_path"])).expanduser().resolve()
        base = load_checkpoint(checkpoint)
        base_update = base.get("update_idx")
        if type(base_update) is not int:
            raise ValueError("base checkpoint has no integer update_idx")
        state_contract = {
            "checkpoint_sha256_matches_manifest": sha256_file(checkpoint) == manifest["checkpoint_sha256"],
            "optimizer_present": "optimizer" in base,
            "critic_optimizer_present": "critic_optimizer" in base.get("algo_state", {}),
            "sampler_state_present": "adaptive_sampler_state" in base,
            "torch_rng_state_present": "torch_rng_state" in base,
            "cuda_rng_state_present": "cuda_rng_state" in base,
            "reset_optimizer_on_resume_false": not bool(base.get("config", {}).get("training", {}).get("reset_optimizer_on_resume")),
            "reset_sampler_on_resume_false": not bool(base.get("config", {}).get("training", {}).get("reset_sampler_on_resume")),
        }
        if not all(state_contract.values()):
            result = diagnostic_result(
                "03",
                INVALID_PROTOCOL,
                summary="base checkpoint cannot reproduce the complete training state",
                evidence={"base_checkpoint": str(checkpoint), "state_contract": state_contract},
                errors=[key for key, passed in state_contract.items() if not passed],
            )
            write_json_exclusive(target, result)
            print(f"[diag_03] {result['status']} {target}")
            return 0
        spec_probe_a = spec_value(spec, "inputs.resume_probe_a", default=None)
        spec_probe_b = spec_value(spec, "inputs.resume_probe_b", default=None)
        result_a_path = args.result_a or (
            resolve_path(spec_probe_a, base=repo_root) if spec_probe_a else None
        )
        result_b_path = args.result_b or (
            resolve_path(spec_probe_b, base=repo_root) if spec_probe_b else None
        )
        if result_a_path is None:
            matches = sorted(repo_root.glob("runs/discovery_resume_probe_a*/checkpoints/update_*.pt"))
            result_a_path = matches[-1] if matches else None
        if result_b_path is None:
            matches = sorted(repo_root.glob("runs/discovery_resume_probe_b*/checkpoints/update_*.pt"))
            result_b_path = matches[-1] if matches else None
        if result_a_path is None or result_b_path is None:
            config = Path(manifest["resolved_config_path"])
            result = diagnostic_result(
                "03",
                SKIPPED_DEPENDENCY,
                summary="two independent one-update Isaac resume artifacts have not been collected",
                evidence={
                    "base_checkpoint": str(checkpoint),
                    "base_checkpoint_sha256": manifest["checkpoint_sha256"],
                    "base_update": base_update,
                    "state_contract": state_contract,
                    "required_result_update": base_update + 1,
                    "collection_commands": [
                        _resume_command(repo_root, config, checkpoint, base_update, "discovery_resume_repro_a"),
                        _resume_command(repo_root, config, checkpoint, base_update, "discovery_resume_repro_b"),
                    ],
                    "comparison_contract": "policy, optimizers, algo, sampler, CPU/CUDA RNG exact; metrics exact by default",
                },
                warnings=["preflight passed; SKIPPED is not evidence of reproducibility"],
            )
        else:
            path_a = result_a_path.expanduser().resolve()
            path_b = result_b_path.expanduser().resolve()
            if path_a == path_b or sha256_file(path_a) == manifest["checkpoint_sha256"]:
                raise ValueError("resume results must be two new, independent checkpoint artifacts")
            result_a = load_checkpoint(path_a)
            result_b = load_checkpoint(path_b)
            expected_update = base_update + 1
            protocol_checks = {
                "different_result_paths": path_a != path_b,
                "result_a_is_next_update": result_a.get("update_idx") == expected_update,
                "result_b_is_next_update": result_b.get("update_idx") == expected_update,
                "same_platform_identity": result_a.get("platform_identity") == result_b.get("platform_identity") == base.get("platform_identity"),
                "same_saved_config": result_a.get("config") == result_b.get("config"),
            }
            if not all(protocol_checks.values()):
                result = diagnostic_result(
                    "03", INVALID_PROTOCOL,
                    summary="resume artifacts do not implement the same one-update protocol",
                    evidence={"protocol_checks": protocol_checks},
                    errors=[key for key, passed in protocol_checks.items() if not passed],
                )
            else:
                comparison = compare_checkpoint_payloads(
                    result_a, result_b,
                    metric_atol=args.metric_atol,
                    metric_rtol=args.metric_rtol,
                )
                uninterrupted_audit_path = resolve_path(
                    spec_value(
                        spec,
                        "inputs.restart_vs_uninterrupted_audit",
                        default=None,
                    ),
                    base=repo_root,
                )
                uninterrupted_audit = (
                    read_json(uninterrupted_audit_path)
                    if uninterrupted_audit_path is not None
                    and uninterrupted_audit_path.is_file()
                    else None
                )
                warnings = [
                    "this PASS/FAIL tests twin restarts from one checkpoint; it does not imply equality to an uninterrupted training process"
                ]
                if isinstance(uninterrupted_audit, dict) and uninterrupted_audit.get("status") == FAIL:
                    warnings.append(
                        "the separately recorded restart-vs-uninterrupted audit failed; restart branches must not be substituted into the uninterrupted canonical path"
                    )
                result = diagnostic_result(
                    "03",
                    PASS if comparison["reproducible"] else FAIL,
                    summary=(
                        "one-update resume is reproducible"
                        if comparison["reproducible"]
                        else "same-checkpoint, same-RNG resumes diverged"
                    ),
                    evidence={
                        "base_checkpoint": str(checkpoint),
                        "base_checkpoint_sha256": manifest["checkpoint_sha256"],
                        "result_a": str(path_a),
                        "result_b": str(path_b),
                        "protocol_checks": protocol_checks,
                        "comparison": comparison,
                        "scope_guard": "twin_restart_reproducibility_only",
                        "restart_vs_uninterrupted_audit": uninterrupted_audit,
                    },
                    warnings=warnings,
                )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        result = diagnostic_result(
            "03", INVALID_PROTOCOL, summary="resume reproducibility protocol is invalid", errors=[str(exc)]
        )
    write_json_exclusive(target, result)
    print(f"[diag_03] {result['status']} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
