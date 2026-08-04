from __future__ import annotations

import importlib.util
from pathlib import Path

from diagnostics.common.canonical_collection import CanonicalCollectionProtocol


_PATH = Path(__file__).resolve().parents[2] / "tools/diag_10_dense_teacher_checkpoints.py"
_SPEC = importlib.util.spec_from_file_location("diag10_selection", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _record(update: int, run: str, digest: str, lineage: str = "main") -> dict:
    return {
        "content_alias_of": None,
        "method": "fixed_reward",
        "dataset_sha256": "motion",
        "robot_asset_sha256": "robot",
        "action_schema_sha256": "action",
        "update_payload": update,
        "checkpoint_sha256": digest,
        "checkpoint_lineage_id": lineage,
        "checkpoint_branch_id": f"branch-{run}",
        "lineage_coherent": True,
        "run_dir": f"/repo/runs/{run}",
        "path": f"/repo/runs/{run}/checkpoints/update_{update:04d}.pt",
    }


def test_audit_only_resume_branch_can_never_be_selected_as_canonical() -> None:
    spec = {
        "collection": {
            "canonical_run_preferences": ["main_a", "main_b"],
            "audit_only_run_patterns": ["audit_u274_u276"],
        }
    }
    protocol = CanonicalCollectionProtocol(
        num_envs=2,
        num_snapshots=2,
        horizon=2,
        phase_strategy="evenly_spaced",
        snapshot_seed=1,
        collector_seed=1,
        common_sigmas=(0.1, 0.2, 0.3),
        canonical_checkpoint_updates=(276,),
    )
    records = [
        _record(276, "audit_u274_u276", "a" * 64, lineage="restart"),
        _record(276, "main_b", "b" * 64, lineage="main"),
    ]
    rows, errors = _MODULE._select_records(
        records,
        protocol,
        {
            "motion_sha256": "motion",
            "robot_asset_sha256": "robot",
            "action_schema_sha256": "action",
        },
        spec,
    )
    assert not errors
    assert rows[0]["present"] is True
    assert rows[0]["checkpoint_sha256"] == "b" * 64
    assert "audit_u274_u276" not in rows[0]["source_run_dir"]
