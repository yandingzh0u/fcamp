from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from diagnostics.common.manifest import DependencyUnavailable, ProtocolError, load_spec
from diagnostics.common.quality_panel import trajectory_quality_row
from diagnostics.common.reward_stage4 import (
    AMP_WINDOW_DIM,
    BRANCH_BANK_SCHEMA,
    RewardValidityProtocol,
    classify_cem_hacking,
    deterministic_blind_order,
    validate_branch_bank,
    validate_branch_rows,
)
from diagnostics.common.stage4_sim import run_same_state_branching_real


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC = REPO_ROOT / "diagnostics" / "specs" / "largebox_discovery_v1.yaml"


def _protocol() -> RewardValidityProtocol:
    return RewardValidityProtocol.from_spec(load_spec(SPEC))


def test_quality_row_uses_real_recorded_fields_without_scalar() -> None:
    steps = 4
    index = {
        "sample_id": "sample",
        "trajectory_id": "trajectory",
        "snapshot_id": "snapshot",
        "checkpoint_id": "u200",
        "checkpoint_sha256": "a" * 64,
        "checkpoint_update": 200,
        "checkpoint_lineage_id": "lineage",
        "policy_domain": "teacher_fixed_reward",
        "collector_mode": "controlled_environment",
        "common_sigma": 0.0,
    }
    tree = {
        "trajectory": {
            "done": torch.tensor([False, False, False, True]),
            "failure": torch.tensor([False, False, False, True]),
            "motion_complete": torch.tensor([False, False, False, False]),
        },
        "outcome": {
            "reference_progress": torch.tensor([0.1, 0.2, 0.3, 0.4]),
            "anchor_error": torch.ones(steps),
            "body_error": 2 * torch.ones(steps),
            "joint_error": 3 * torch.ones(steps),
            "reward_terms": {
                "joint_limit": torch.tensor([0.0, 1.0, 0.0, 1.0]),
                "undesired_contacts": torch.tensor([0.0, 0.0, 2.0, 2.0]),
            },
        },
        "action": {"applied": torch.arange(steps * 2, dtype=torch.float32).reshape(steps, 2)},
        "state": {},
    }
    row = trajectory_quality_row(index, tree)
    assert row["failure"] == 1.0
    assert row["joint_limit_incidence"] == 0.5
    assert row["undesired_contacts"] == 1.0
    assert row["contact_mode_agreement_available"] is False
    assert not {"quality", "quality_score", "weighted_quality", "Q"}.intersection(row)


def test_branch_bank_requires_real_replay_guarantees() -> None:
    horizons = (1, 5, 10, 25, 50)
    payload = {
        "metadata": {
            "schema": BRANCH_BANK_SCHEMA,
            "real_physx_rollouts": True,
            "same_snapshot_replay_verified": True,
            "shared_environment_randomness": True,
            "demo_seeded_commit6901_history": True,
            "source_classifier_used_as_reward": False,
            "critic_source_commit": "6901e302499711e2207687e1342348a4078330f8",
            "branch_ids": ["u200", "u500", "perturb_005"],
            "snapshot_ids": ["s0", "s1"],
            "horizons": list(horizons),
            "track_body_names": ["pelvis"],
        },
        "endpoint_windows": torch.zeros(3, 5, 2, AMP_WINDOW_DIM),
        "body_pos_local": torch.zeros(3, 50, 2, 1, 3),
        "root_pos_local": torch.zeros(3, 50, 2, 3),
        "active": torch.ones(3, 50, 2, dtype=torch.bool),
        "endpoint_phase": torch.zeros(3, 5, 2),
        "endpoint_contact_mode": torch.zeros(3, 5, 2, dtype=torch.long),
    }
    validate_branch_bank(payload)
    payload["metadata"]["real_physx_rollouts"] = False
    with pytest.raises(ProtocolError, match="real replay guarantees"):
        validate_branch_bank(payload)


def test_branch_rows_reject_combined_quality_and_amix() -> None:
    protocol = _protocol()
    base = {
        "branch_id": "u200",
        "branch_category": "checkpoint",
        "snapshot_id": "s0",
        "horizon": 10,
        **{metric.name: 0.0 for metric in protocol.primary_metrics},
        "reward_K_mean": 1.0,
        "reward_T_u500_mean": 1.0,
    }
    validate_branch_rows([base], protocol=protocol)
    with pytest.raises(ProtocolError, match="forbidden combined"):
        validate_branch_rows([{**base, "quality_score": 1.0}], protocol=protocol)
    with pytest.raises(ProtocolError, match="A_mix"):
        validate_branch_rows([{**base, "branch_id": "A_mix"}], protocol=protocol)


def test_cem_hacking_uses_paired_snapshot_records() -> None:
    records = [
        {"search_reward_gain": 1.0, "strict_pareto_regression": True, "clear_failure": False}
        for _ in range(64)
    ]
    result = classify_cem_hacking(
        records, alpha=0.05, bootstrap_replicates=1000, seed=20260803
    )
    assert result["significant_reward_gain"] is True
    assert result["significant_reward_hacking"] is True
    assert result["paired_snapshot_seed_record_count"] == 64


def test_blind_order_is_deterministic_and_executor_fails_closed(tmp_path: Path) -> None:
    assert deterministic_blind_order("p0", 20260803) == deterministic_blind_order(
        "p0", 20260803
    )
    with pytest.raises(DependencyUnavailable, match="persisted frozen spec path"):
        run_same_state_branching_real(
            repo_root=tmp_path,
            output_dir=tmp_path,
            spec={},
            bank_path=tmp_path / "bank.pt",
            row_path=tmp_path / "rows.parquet",
        )
