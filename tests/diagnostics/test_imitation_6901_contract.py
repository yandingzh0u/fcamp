from __future__ import annotations

import json
from pathlib import Path

import torch

from components.imitation.motion_features import canonicalize_imitation_window
from diagnostics.common.imitation_6901 import (
    IMITATION_FRAME_DIM,
    IMITATION_FRAME_SCHEMA_SHA256,
    IMITATION_KEY_BODY_NAMES,
    imitation_contract_metadata,
)
from diagnostics.common.legacy_policy import AAMPPolicyAdapter, _ExactAMPH1Actor
from diagnostics.common.noise_bank import CollectorMode
from envs.imitation_data import build_g1_imitation_frame


ROOT = Path(__file__).resolve().parents[2]


def _frame(root_xy: tuple[float, float]) -> torch.Tensor:
    root_pos = torch.tensor([[root_xy[0], root_xy[1], 0.8]])
    return build_g1_imitation_frame(
        root_pos=root_pos,
        root_quat_wxyz=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        joint_pos=torch.zeros(1, 29),
        key_body_pos=root_pos[:, None, :].expand(1, 5, 3),
        root_lin_vel=torch.zeros(1, 3),
        root_ang_vel=torch.zeros(1, 3),
        joint_vel=torch.zeros(1, 29),
    )


def test_commit_6901_frame_is_239d_and_fixed_head_is_identity() -> None:
    frame = _frame((2.0, -3.0))
    assert frame.shape == (1, IMITATION_FRAME_DIM)
    assert IMITATION_FRAME_DIM == 239
    # root xyz + root rot6d + 15 actuated joint rotations precede the fixed
    # head entry in the 30-joint rotation block.
    head_start = 3 + 6 + 15 * 6
    torch.testing.assert_close(
        frame[0, head_start : head_start + 6],
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
    )


def test_raw_frames_keep_root_xy_until_window_boundary() -> None:
    raw = torch.stack((_frame((2.0, 4.0))[0], _frame((5.0, 9.0))[0]))
    torch.testing.assert_close(raw[:, :2], torch.tensor([[2.0, 4.0], [5.0, 9.0]]))
    canonical = canonicalize_imitation_window(raw)
    torch.testing.assert_close(
        canonical[:, :2], torch.tensor([[-3.0, -5.0], [0.0, 0.0]])
    )
    # The raw causal history remains untouched.
    torch.testing.assert_close(raw[:, :2], torch.tensor([[2.0, 4.0], [5.0, 9.0]]))


def test_imitation_metadata_is_frozen_and_schema_declares_eighth_section() -> None:
    metadata = imitation_contract_metadata()
    assert metadata["imitation_frame_dim"] == 239
    assert metadata["imitation_frame_schema_sha256"] == IMITATION_FRAME_SCHEMA_SHA256
    assert len(IMITATION_FRAME_SCHEMA_SHA256) == 64
    assert metadata["imitation_key_body_names"] == list(IMITATION_KEY_BODY_NAMES)
    assert metadata["imitation_agent_negative_field"] == "agent_physx_raw_frame"
    assert metadata["imitation_reference_positive_field"] == "reference_expert_raw_frame"
    schema = json.loads(
        (ROOT / "diagnostics/schemas/rollout_schema.json").read_text(encoding="utf-8")
    )
    assert "imitation" in schema["required"]
    assert schema["properties"]["imitation"]["required"] == [
        "agent_physx_raw_frame",
        "agent_fk_aligned_raw_frame",
        "reference_expert_raw_frame",
        "phase_normalized_pre_step",
    ]


def test_official_amp_adapter_uses_physx_frame_and_fixed_point05_std() -> None:
    actor = _ExactAMPH1Actor(237, 29)
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.zero_()
        actor.mean_head.bias.fill_(0.2)
    adapter = AAMPPolicyAdapter(
        actor=actor,
        normalizer_mean=torch.zeros(1, 237),
        normalizer_std=torch.ones(1, 237),
        normalizer_clip=10.0,
        source_snapshot_sha256="a" * 64,
        resolved_config_sha256="b" * 64,
    )
    frame = _frame((2.0, 4.0)).repeat(3, 1)
    record = adapter.action_record(
        env=None,
        observation=torch.full((3, 171), float("nan")),
        imitation={"agent_physx_raw_frame": frame},
        common_epsilon=torch.ones(3, 29),
        mode=CollectorMode.NATIVE_STOCHASTIC,
        common_sigma=0.0,
        action_low=torch.full((29,), -100.0),
        action_high=torch.full((29,), 100.0),
        step=0,
    )
    torch.testing.assert_close(record.mean, torch.full((3, 29), 0.2))
    torch.testing.assert_close(record.std, torch.full((3, 29), 0.05))
    torch.testing.assert_close(record.sampled, torch.full((3, 29), 0.25))
