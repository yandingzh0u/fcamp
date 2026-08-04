"""Fail-closed adapter points for real Isaac/PhysX Stage-4 experiments.

The pure diagnostic package deliberately does not import Isaac Lab at module
import time.  A site-specific simulator launcher can implement these callables
without changing any Stage-4 analyzer.  Until it is installed, the callables
raise ``DependencyUnavailable``; callers record SKIPPED_DEPENDENCY and never
manufacture counterfactual data from stored observations.

Every executor must write the frozen artifact named by its arguments using an
exclusive create, finish/fsync it, and return exactly ``True``.  The consuming
diagnostic then validates the schema and all scientific invariants again.
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import os
import subprocess
import sys
from typing import Any, Mapping

from .manifest import DependencyUnavailable, ProtocolError


def _missing(name: str, contract: str) -> None:
    raise DependencyUnavailable(
        f"{name} has no installed real Isaac/PhysX backend; required contract: {contract}"
    )


def run_same_state_branching_real(
    *,
    repo_root: Path,
    output_dir: Path,
    spec: Mapping[str, Any],
    bank_path: Path,
    row_path: Path,
) -> bool:
    """Create replay-verified branches from exact simulator snapshots.

    Required outputs are ``largebox_same_snapshot_branch_bank_v1`` and a
    branch-quality parquet containing independently measured outcomes and the
    exact persisted Stage-3 critic scores.  A_mix/FCAMP is forbidden.
    """

    if importlib.util.find_spec("isaaclab") is None:
        _missing(
            "run_same_state_branching_real",
            "Isaac Lab import plus same snapshot/NoiseBank PhysX execution",
        )
    spec_path = spec.get("_spec_path")
    if not spec_path:
        raise DependencyUnavailable("real branch executor requires a persisted frozen spec path")
    # Branch scoring is defined only for the persisted five-seed standard AMP
    # critics produced by diag_35.  Check this before launching a multi-minute
    # Isaac process; absence is an upstream dependency gap, not a malformed
    # simulator protocol.
    critic_status = output_dir / "ratio_reliability.json"
    if not critic_status.is_file():
        raise DependencyUnavailable(
            "diag_42 requires diag_35 persisted AMP critics; ratio_reliability.json is absent"
        )
    from .manifest import read_json

    if read_json(critic_status).get("status") != "PASS":
        raise DependencyUnavailable("diag_42 requires a PASS diag_35 critic bank")
    entrypoint = repo_root / "tools" / "collect_stage4_branches_real.py"
    if not entrypoint.is_file():
        raise DependencyUnavailable(f"real branch collector is missing: {entrypoint}")
    command = [
        sys.executable,
        str(entrypoint),
        "--spec",
        str(spec_path),
        "--output-dir",
        str(output_dir),
        "--repo-root",
        str(repo_root),
        "--bank-path",
        str(bank_path),
        "--row-path",
        str(row_path),
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-40:])
        raise ProtocolError(
            f"real Stage-4 branch collector exited {completed.returncode}:\n{tail}"
        )
    if not bank_path.is_file() or not row_path.is_file():
        raise ProtocolError("real branch collector exited successfully without both artifacts")
    return True


def run_cem_exploitability_real(
    *,
    repo_root: Path,
    output_dir: Path,
    spec: Mapping[str, Any],
    branch_bank_path: Path,
    result_path: Path,
) -> bool:
    """Run short-horizon CEM in PhysX against a search critic only."""

    if importlib.util.find_spec("isaaclab") is None:
        _missing(
            "run_cem_exploitability_real",
            "Isaac Lab import, H=10/population=256 and an independent audit critic",
        )
    spec_path = spec.get("_spec_path")
    if not spec_path:
        raise DependencyUnavailable("real CEM executor requires a persisted frozen spec path")
    entrypoint = repo_root / "tools" / "collect_stage4_cem_real.py"
    if not entrypoint.is_file():
        raise DependencyUnavailable(f"real CEM collector is missing: {entrypoint}")
    command = [
        sys.executable,
        str(entrypoint),
        "--spec",
        str(spec_path),
        "--output-dir",
        str(output_dir),
        "--repo-root",
        str(repo_root),
        "--branch-bank",
        str(branch_bank_path),
        "--result-path",
        str(result_path),
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-40:])
        raise ProtocolError(f"real Stage-4 CEM collector exited {completed.returncode}:\n{tail}")
    if not result_path.is_file():
        raise ProtocolError("real CEM collector exited successfully without its result bank")
    return True


def run_reward_interventions_real(
    *,
    repo_root: Path,
    output_dir: Path,
    spec: Mapping[str, Any],
    branch_bank_path: Path,
    result_path: Path,
) -> bool:
    """Create registered static/phase/contact reward interventions."""

    del repo_root, output_dir
    import numpy as np
    import torch

    from .reward_stage4 import (
        AMP_WINDOW_DIM,
        INTERVENTION_BANK_SCHEMA,
        RewardValidityProtocol,
        load_branch_bank,
    )

    protocol = RewardValidityProtocol.from_spec(spec)
    bank = load_branch_bank(branch_bank_path)
    metadata = bank["metadata"]
    branch_ids = tuple(str(value) for value in metadata["branch_ids"])
    try:
        base_index = branch_ids.index("checkpoint_u0500")
    except ValueError as exc:
        raise DependencyUnavailable("interventions require the real u500 checkpoint branch") from exc
    windows = bank["endpoint_windows"][base_index].float()
    phases = bank["endpoint_phase"][base_index].float()
    contacts = bank["endpoint_contact_mode"][base_index].long()
    if windows.ndim != 3 or windows.shape[-1] != AMP_WINDOW_DIM:
        raise ProtocolError("u500 intervention base-window bank is invalid")
    # Complete, non-overlapping physical views of one 239-D AMP frame.
    frame_views = {
        "root_position": (0, 3),
        "root_orientation_6d": (3, 9),
        "joint_rotations_6d": (9, 189),
        "key_body_positions": (189, 204),
        "root_linear_velocity": (204, 207),
        "root_angular_velocity": (207, 210),
        "dof_velocity": (210, 239),
    }
    base_rows: list[torch.Tensor] = []
    changed_rows: list[torch.Tensor] = []
    records: list[dict[str, Any]] = []
    unsupported: list[dict[str, Any]] = []
    flat_scale = windows.reshape(-1, AMP_WINDOW_DIM).std(dim=0, unbiased=False)
    horizon_values = tuple(int(value) for value in metadata["horizons"])
    snapshot_ids = tuple(str(value) for value in metadata["snapshot_ids"])

    def append(
        base: torch.Tensor,
        changed: torch.Tensor,
        *,
        horizon_index: int,
        snapshot_index: int,
        kind: str,
        details: Mapping[str, Any],
    ) -> None:
        index = len(records)
        base_rows.append(base.clone())
        changed_rows.append(changed.clone())
        records.append(
            {
                "intervention_id": f"iv-{index:06d}",
                "intervention_type": kind,
                "base_branch_id": "checkpoint_u0500",
                "horizon": horizon_values[horizon_index],
                "snapshot_id": snapshot_ids[snapshot_index],
                **dict(details),
            }
        )

    for view, (start, stop) in frame_views.items():
        seed = int(canonical_sha256({"view": view, "seed": 20260803})[:16], 16)
        rng = np.random.default_rng(seed)
        direction = torch.as_tensor(
            rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=stop - start)
        )
        indices = torch.cat(
            [torch.arange(step * 239 + start, step * 239 + stop) for step in range(10)]
        )
        repeated_direction = direction.repeat(10)
        for magnitude in protocol.interventions["standardized_static_offset_magnitudes"]:
            for horizon_index in range(windows.shape[0]):
                for snapshot_index in range(windows.shape[1]):
                    base = windows[horizon_index, snapshot_index]
                    changed = base.clone()
                    changed[indices] += float(magnitude) * flat_scale[indices] * repeated_direction
                    append(
                        base,
                        changed,
                        horizon_index=horizon_index,
                        snapshot_index=snapshot_index,
                        kind="static_offset",
                        details={
                            "physical_view": view,
                            "magnitude": float(magnitude),
                            "direction_seed": seed,
                            "standardization": "u500_real_endpoint_population_std",
                        },
                    )

    for horizon_index in range(windows.shape[0]):
        phase_values = phases[horizon_index]
        contact_values = contacts[horizon_index]
        for snapshot_index in range(windows.shape[1]):
            base = windows[horizon_index, snapshot_index]
            current_phase = float(phase_values[snapshot_index])
            for offset in protocol.interventions["phase_offsets"]:
                target = current_phase + float(offset)
                distance = torch.abs(phase_values - target)
                source_index = int(torch.argmin(distance).item())
                append(
                    base,
                    windows[horizon_index, source_index],
                    horizon_index=horizon_index,
                    snapshot_index=snapshot_index,
                    kind="phase_swap",
                    details={
                        "phase_offset": int(offset),
                        "source_snapshot_id": snapshot_ids[source_index],
                        "actual_phase_delta": float(phase_values[source_index] - phase_values[snapshot_index]),
                    },
                )
            current_contact = int(contact_values[snapshot_index])
            for swap in protocol.interventions["contact_swaps"]:
                desired = None
                if swap == "left_right" and current_contact in (1, 2):
                    desired = 3 - current_contact
                    eligible = torch.where(contact_values == desired)[0]
                elif swap == "temporal_neighbor":
                    eligible = torch.where(contact_values != current_contact)[0]
                else:
                    eligible = torch.empty(0, dtype=torch.long)
                if eligible.numel() == 0:
                    unsupported.append(
                        {
                            "contact_swap": str(swap),
                            "horizon": horizon_values[horizon_index],
                            "snapshot_id": snapshot_ids[snapshot_index],
                            "reason": "no condition-matched real endpoint with requested contact mode",
                        }
                    )
                    continue
                distances = torch.abs(phase_values.index_select(0, eligible) - current_phase)
                source_index = int(eligible[int(torch.argmin(distances).item())].item())
                append(
                    base,
                    windows[horizon_index, source_index],
                    horizon_index=horizon_index,
                    snapshot_index=snapshot_index,
                    kind="contact_swap",
                    details={
                        "contact_swap": str(swap),
                        "base_contact_mode": current_contact,
                        "source_contact_mode": int(contact_values[source_index]),
                        "source_snapshot_id": snapshot_ids[source_index],
                    },
                )
    if not records:
        raise DependencyUnavailable("registered intervention generator produced no supported record")
    payload = {
        "schema": INTERVENTION_BANK_SCHEMA,
        "source_branch_bank_sha256": sha256_file(branch_bank_path),
        "intervention_protocol": dict(protocol.interventions),
        "real_base_physx_windows": True,
        "source_classifier_used_as_reward": False,
        "A_mix": "legacy_quarantined",
        "physical_view_partition": frame_views,
        "records": records,
        "unsupported_interventions": unsupported,
        "base_windows": torch.stack(base_rows),
        "intervened_windows": torch.stack(changed_rows),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(str(result_path.parent), os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return True


def render_blind_video_sources_real(
    *,
    repo_root: Path,
    output_dir: Path,
    spec: Mapping[str, Any],
    branch_bank_path: Path,
    source_manifest_path: Path,
) -> bool:
    """Render actual PhysX branch videos before anonymous randomization."""

    del repo_root, output_dir, spec, branch_bank_path, source_manifest_path
    _missing(
        "render_blind_video_sources_real",
        "32 real branch pairs rendered with common camera/timing and no labels",
    )


__all__ = [
    "render_blind_video_sources_real",
    "run_cem_exploitability_real",
    "run_reward_interventions_real",
    "run_same_state_branching_real",
]
