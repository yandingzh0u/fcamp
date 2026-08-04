#!/usr/bin/env python3
"""Measure persisted AMP reward sensitivity to registered physical interventions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diagnostics.common.manifest import (
    INVALID_PROTOCOL,
    PASS,
    SKIPPED_DEPENDENCY,
    DependencyUnavailable,
    ProtocolError,
    diagnostic_result,
    load_spec,
    read_json,
    sha256_file,
    write_json_exclusive,
)
from diagnostics.common.policy_class_probe import output_dir_from_args
from diagnostics.common.reward_stage4 import (
    AMP_WINDOW_DIM,
    INTERVENTION_BANK_SCHEMA,
    PRIMARY_REWARD_FAMILIES,
    RewardValidityProtocol,
    load_stage3_critics,
)


def _load_bank(path: Path) -> dict[str, object]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("schema") != INTERVENTION_BANK_SCHEMA:
        raise ProtocolError("reward-intervention bank schema is invalid")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--intervention-bank", type=Path, default=None)
    parser.set_defaults(execute_real=True)
    parser.add_argument("--execute-real", dest="execute_real", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--no-execute-real", dest="execute_real", action="store_false")
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    spec = load_spec(args.spec)
    output_dir = output_dir_from_args(root, spec, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "intervention.json"
    try:
        protocol = RewardValidityProtocol.from_spec(spec)
        branching_path = output_dir / "branching.json"
        if not branching_path.is_file():
            raise DependencyUnavailable("diag_42 branching status is absent")
        branching = read_json(branching_path)
        if branching.get("status") != PASS:
            raise DependencyUnavailable("diag_42 has no real branch bank for interventions")
        branch_path = Path(str(branching["evidence"]["branch_bank"])).expanduser().resolve()
        branch_hash = str(branching["evidence"].get("branch_bank_sha256", ""))
        if sha256_file(branch_path) != branch_hash:
            raise ProtocolError("branch bank changed after diag_42")
        bank_path = (
            args.intervention_bank or output_dir / "reward_intervention_bank.pt"
        ).expanduser().resolve()
        if args.execute_real and not bank_path.is_file():
            from diagnostics.common.stage4_sim import run_reward_interventions_real

            completed = run_reward_interventions_real(
                repo_root=root,
                output_dir=output_dir,
                spec=spec,
                branch_bank_path=branch_path,
                result_path=bank_path,
            )
            if completed is not True:
                raise ProtocolError("real intervention executor did not confirm completion")
        if not bank_path.is_file():
            raise DependencyUnavailable(
                "registered intervention bank over real PhysX windows is absent"
            )
        payload = _load_bank(bank_path)
        if payload.get("source_branch_bank_sha256") != branch_hash:
            raise ProtocolError("intervention bank is not derived from the validated branch bank")
        if payload.get("intervention_protocol") != dict(protocol.interventions):
            raise ProtocolError("intervention bank differs from the frozen protocol")
        if payload.get("real_base_physx_windows") is not True:
            raise ProtocolError("intervention bases are not real PhysX windows")
        if payload.get("source_classifier_used_as_reward") is not False:
            raise ProtocolError("source classifier is forbidden as an intervention reward")
        if payload.get("A_mix") not in (None, "legacy_quarantined"):
            raise ProtocolError("A_mix/FCAMP entered the intervention bank")
        records = payload.get("records")
        base = payload.get("base_windows")
        changed = payload.get("intervened_windows")
        if not isinstance(records, list) or not records:
            raise ProtocolError("intervention bank contains no records")
        if not torch.is_tensor(base) or not torch.is_tensor(changed):
            raise ProtocolError("intervention bank lacks exact before/after windows")
        if tuple(base.shape) != (len(records), AMP_WINDOW_DIM) or changed.shape != base.shape:
            raise ProtocolError("intervention window shape is invalid")
        if base.dtype != torch.float32 or changed.dtype != torch.float32:
            raise ProtocolError("intervention windows must be float32")
        if not bool(torch.isfinite(base).all() and torch.isfinite(changed).all()):
            raise ProtocolError("intervention windows contain NaN or Inf")
        allowed_types = {"static_offset", "phase_swap", "contact_swap"}
        seen_ids: set[str] = set()
        for record in records:
            if not isinstance(record, dict):
                raise ProtocolError("intervention record is not a mapping")
            identity = str(record.get("intervention_id", ""))
            kind = str(record.get("intervention_type", ""))
            if not identity or identity in seen_ids or kind not in allowed_types:
                raise ProtocolError("intervention identity/type is invalid")
            seen_ids.add(identity)
            if kind == "static_offset":
                magnitude = float(record.get("magnitude", np.nan))
                if magnitude not in tuple(float(v) for v in protocol.interventions["standardized_static_offset_magnitudes"]):
                    raise ProtocolError("static offset magnitude is not registered")
                if not str(record.get("physical_view", "")):
                    raise ProtocolError("static offset has no named physical view")
            elif kind == "phase_swap":
                if int(record.get("phase_offset", 0)) not in tuple(int(v) for v in protocol.interventions["phase_offsets"]):
                    raise ProtocolError("phase offset is not registered")
            else:
                if str(record.get("contact_swap", "")) not in tuple(protocol.interventions["contact_swaps"]):
                    raise ProtocolError("contact swap is not registered")
        critics = load_stage3_critics(output_dir, spec, device="cpu")
        before_np = base.numpy()
        after_np = changed.numpy()
        enriched: list[dict[str, object]] = [dict(record) for record in records]
        summaries: dict[str, dict[str, dict[str, float]]] = {}
        for family in PRIMARY_REWARD_FAMILIES:
            seed_deltas = []
            for fit in critics.by_positive[family]:
                delta = fit.rewards(after_np) - fit.rewards(before_np)
                seed_deltas.append(delta)
                for index, value in enumerate(delta):
                    enriched[index][f"reward_delta_{family}_seed_{fit.seed}"] = float(value)
            mean_delta = np.mean(np.stack(seed_deltas), axis=0)
            for index, value in enumerate(mean_delta):
                enriched[index][f"reward_delta_{family}_mean"] = float(value)
            summaries[family] = {}
            for kind in sorted(allowed_types):
                values = np.asarray(
                    [mean_delta[index] for index, record in enumerate(records) if record["intervention_type"] == kind],
                    dtype=np.float64,
                )
                if values.size:
                    summaries[family][kind] = {
                        "count": float(values.size),
                        "mean_delta": float(values.mean()),
                        "mean_absolute_delta": float(np.abs(values).mean()),
                        "positive_fraction": float(np.mean(values > 0.0)),
                    }
        result = diagnostic_result(
            "45",
            PASS,
            summary="registered interventions were rescored by exact persisted AMP critics",
            evidence={
                "intervention_bank": str(bank_path),
                "intervention_bank_sha256": sha256_file(bank_path),
                "record_count": len(enriched),
                "supported_intervention_types": sorted({str(row["intervention_type"]) for row in records}),
                "unsupported_interventions": payload.get("unsupported_interventions", []),
                "reward_sensitivity": summaries,
                "records": enriched,
                "base_windows_real_physx": True,
                "metric_combination": "none",
                "A_mix": "legacy_quarantined",
            },
        )
    except DependencyUnavailable as exc:
        result = diagnostic_result(
            "45",
            SKIPPED_DEPENDENCY,
            summary="intervention sensitivity awaits registered real-window assets",
            evidence={"executor_seam": "diagnostics.common.stage4_sim.run_reward_interventions_real"},
            errors=[str(exc)],
        )
    except (FileNotFoundError, ProtocolError, ValueError, KeyError, FloatingPointError, RuntimeError) as exc:
        result = diagnostic_result(
            "45", INVALID_PROTOCOL, summary="intervention protocol failed closed", errors=[str(exc)]
        )
    write_json_exclusive(target, result)
    print(f"[diag_45] {result['status']} {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
