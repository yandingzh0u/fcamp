from __future__ import annotations

import pytest

from diagnostics.common.edge_causality import (
    POSITIVE_BUFFER_SCHEMA,
    build_positive_buffer_manifest,
    validate_positive_buffer_isolation,
)
from diagnostics.common.manifest import ProtocolError


def _edge() -> dict:
    return {
        "edge_id": "T_u0200_to_T_u0220",
        "source_update": 200,
        "target_update": 220,
        "source_checkpoint_sha256": "a" * 64,
        "target_checkpoint_sha256": "b" * 64,
    }


def _row(sample: str, *, update: int = 220, frame_domain: str = "teacher_fixed_reward") -> dict:
    return {
        "sample_id": sample,
        "snapshot_id": f"snapshot-{sample}",
        "policy_domain": frame_domain,
        "checkpoint_update": update,
        "checkpoint_sha256": "b" * 64,
        "collector_mode": "controlled_environment",
        "common_sigma": 0.0,
        "eligible_for_primary_overlap": True,
        "split": "train",
    }


def test_positive_buffer_contains_only_target_agent_physx_rows() -> None:
    rows = [
        _row("target-0"),
        _row("target-1"),
        _row("source", update=200),
        _row("reference", frame_domain="reference_expert"),
    ]
    manifest = build_positive_buffer_manifest(rows, _edge())
    assert manifest["artifact_schema"] == POSITIVE_BUFFER_SCHEMA
    assert manifest["frame_role"] == "agent_physx_raw_frame"
    assert manifest["positive_domain"] == "T_u220"
    assert manifest["sample_ids"] == ["target-0", "target-1"]


def test_policy_negative_cannot_leak_into_positive_buffer() -> None:
    manifest = build_positive_buffer_manifest([_row("target-0")], _edge())
    with pytest.raises(ProtocolError, match="overlap"):
        validate_positive_buffer_isolation(
            manifest, negative_sample_ids=["different", "target-0"]
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("frame_role", "reference_expert_raw_frame"),
        ("collector_mode", "native_stochastic"),
        ("policy_domain", "current_policy"),
        ("positive_domain", "K"),
    ],
)
def test_source_or_domain_tampering_fails_closed(field: str, value: object) -> None:
    manifest = build_positive_buffer_manifest([_row("target-0")], _edge())
    manifest[field] = value
    with pytest.raises(ProtocolError):
        validate_positive_buffer_isolation(manifest)


def test_target_checkpoint_hash_is_part_of_row_selection() -> None:
    row = _row("target-0")
    row["checkpoint_sha256"] = "c" * 64
    with pytest.raises(RuntimeError, match="no target-j"):
        build_positive_buffer_manifest([row], _edge())
