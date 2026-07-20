from __future__ import annotations

import pytest
import torch

from engine.validation_metrics import ChunkBoundaryDiagnostics


def _state(batch: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(batch, 2),
        torch.zeros(batch, 2),
        torch.zeros(batch, 3),
    )


def _update(
    diagnostics: ChunkBoundaryDiagnostics,
    *,
    active: torch.Tensor,
    offset: int | torch.Tensor,
    action_value: torch.Tensor,
    joint_pos_value: torch.Tensor,
    joint_vel_value: torch.Tensor,
    root_ang_value: torch.Tensor,
) -> None:
    diagnostics.update(
        active_mask=active,
        chunk_offset=offset,
        action=action_value[:, None].expand(-1, 2).clone(),
        joint_pos=joint_pos_value[:, None].expand(-1, 2).clone(),
        joint_vel=joint_vel_value[:, None].expand(-1, 2).clone(),
        root_ang_vel=root_ang_value[:, None].expand(-1, 3).clone(),
        reference_joint_pos=torch.zeros(2, 2),
        reference_joint_vel=torch.zeros(2, 2),
        reference_root_ang_vel=torch.zeros(2, 3),
    )


def test_chunk_boundary_metrics_separate_reset_boundary_internal_and_done() -> None:
    initial_action, initial_joint_vel, initial_root_ang_vel = _state()
    diagnostics = ChunkBoundaryDiagnostics(
        horizon=4,
        initial_action=initial_action,
        initial_joint_vel=initial_joint_vel,
        initial_root_ang_vel=initial_root_ang_vel,
    )

    # Both environments execute the first post-reset action.  Environment one
    # then terminates; keeping it out of active_mask prevents zero-action tail
    # samples from contaminating every statistic.
    _update(
        diagnostics,
        active=torch.tensor([True, True]),
        offset=0,
        action_value=torch.tensor([1.0, 2.0]),
        joint_pos_value=torch.tensor([1.0, 2.0]),
        joint_vel_value=torch.tensor([1.0, 2.0]),
        root_ang_value=torch.tensor([1.0, 2.0]),
    )
    action_values = (2.0, 4.0, 7.0, 11.0)
    joint_vel_values = (3.0, 6.0, 10.0, 15.0)
    root_ang_values = (2.0, 4.0, 7.0, 11.0)
    for offset, (action, joint_vel, root_ang) in enumerate(
        zip(action_values, joint_vel_values, root_ang_values, strict=True),
        start=1,
    ):
        _update(
            diagnostics,
            active=torch.tensor([True, False]),
            offset=offset % 4,
            action_value=torch.tensor([action, -999.0]),
            joint_pos_value=torch.tensor([action, -999.0]),
            joint_vel_value=torch.tensor([joint_vel, -999.0]),
            root_ang_value=torch.tensor([root_ang, -999.0]),
        )

    metrics = diagnostics.metrics()

    assert metrics["validation/chunk_action_delta_reset_first_count"] == 2.0
    assert metrics["validation/chunk_action_delta_reset_first_mean"] == pytest.approx(1.5)
    assert metrics["validation/chunk_action_d2_reset_first_count"] == 0.0
    assert metrics["validation/chunk_action_d2_reset_first_mean"] == -1.0
    assert metrics["validation/chunk_joint_pos_error_reset_first_mean"] == pytest.approx(1.5)
    assert metrics["validation/chunk_joint_vel_error_reset_first_mean"] == pytest.approx(1.5)
    assert metrics["validation/chunk_root_ang_vel_error_reset_first_mean"] == pytest.approx(1.5)

    assert metrics["validation/chunk_action_delta_internal_count"] == 3.0
    assert metrics["validation/chunk_action_delta_internal_mean"] == pytest.approx(2.0)
    assert metrics["validation/chunk_action_delta_boundary_count"] == 1.0
    assert metrics["validation/chunk_action_delta_boundary_mean"] == pytest.approx(4.0)
    assert metrics["validation/chunk_action_delta_boundary_internal_ratio"] == pytest.approx(2.0)

    assert metrics["validation/chunk_action_d2_internal_mean"] == pytest.approx(2.0 / 3.0)
    assert metrics["validation/chunk_action_d2_boundary_mean"] == pytest.approx(1.0)
    assert metrics["validation/chunk_action_d2_boundary_internal_ratio"] == pytest.approx(1.5)
    assert metrics["validation/chunk_joint_vel_jump_internal_mean"] == pytest.approx(3.0)
    assert metrics["validation/chunk_joint_vel_jump_boundary_mean"] == pytest.approx(5.0)
    assert metrics["validation/chunk_root_ang_vel_jump_internal_mean"] == pytest.approx(2.0)
    assert metrics["validation/chunk_root_ang_vel_jump_boundary_mean"] == pytest.approx(4.0)

    # Offset zero contains both first-step samples plus the later real chunk
    # boundary; the inactive environment contributes no -999 tail sample.
    assert metrics["validation/chunk_offset0_joint_pos_error_count"] == 3.0
    assert metrics["validation/chunk_offset1_joint_vel_error_count"] == 1.0
    assert metrics["validation/chunk_offset2_root_ang_vel_error_count"] == 1.0
    assert metrics["validation/chunk_offset3_joint_pos_error_count"] == 1.0


def test_per_environment_reset_never_becomes_a_chunk_boundary() -> None:
    initial_action, initial_joint_vel, initial_root_ang_vel = _state()
    diagnostics = ChunkBoundaryDiagnostics(
        horizon=4,
        initial_action=initial_action,
        initial_joint_vel=initial_joint_vel,
        initial_root_ang_vel=initial_root_ang_vel,
    )
    _update(
        diagnostics,
        active=torch.tensor([True, True]),
        offset=0,
        action_value=torch.tensor([1.0, 1.0]),
        joint_pos_value=torch.tensor([1.0, 1.0]),
        joint_vel_value=torch.tensor([1.0, 1.0]),
        root_ang_value=torch.tensor([1.0, 1.0]),
    )

    reset_action = torch.tensor([[1.0, 1.0], [10.0, 10.0]])
    reset_joint_vel = torch.tensor([[1.0, 1.0], [20.0, 20.0]])
    reset_root_ang_vel = torch.tensor([[1.0, 1.0, 1.0], [30.0, 30.0, 30.0]])
    diagnostics.reset(
        torch.tensor([False, True]),
        initial_action=reset_action,
        initial_joint_vel=reset_joint_vel,
        initial_root_ang_vel=reset_root_ang_vel,
    )
    _update(
        diagnostics,
        active=torch.tensor([True, True]),
        offset=torch.tensor([1, 0]),
        action_value=torch.tensor([2.0, 11.0]),
        joint_pos_value=torch.tensor([2.0, 11.0]),
        joint_vel_value=torch.tensor([2.0, 21.0]),
        root_ang_value=torch.tensor([2.0, 31.0]),
    )

    metrics = diagnostics.metrics()
    assert metrics["validation/chunk_action_delta_reset_first_count"] == 3.0
    assert metrics["validation/chunk_action_delta_internal_count"] == 1.0
    assert metrics["validation/chunk_action_delta_boundary_count"] == 0.0
    assert metrics["validation/chunk_action_delta_boundary_mean"] == -1.0
    assert metrics["validation/chunk_action_delta_boundary_internal_ratio"] == -1.0


def test_empty_chunk_metrics_have_stable_safe_schema() -> None:
    initial_action, initial_joint_vel, initial_root_ang_vel = _state(batch=1)
    diagnostics = ChunkBoundaryDiagnostics(
        horizon=4,
        initial_action=initial_action,
        initial_joint_vel=initial_joint_vel,
        initial_root_ang_vel=initial_root_ang_vel,
    )

    metrics = diagnostics.metrics(prefix="validation_directional")
    required = (
        "validation_directional/chunk_action_delta_boundary_count",
        "validation_directional/chunk_action_delta_boundary_mean",
        "validation_directional/chunk_action_delta_boundary_p95",
        "validation_directional/chunk_action_delta_boundary_p99",
        "validation_directional/chunk_action_d2_boundary_internal_ratio",
        "validation_directional/chunk_joint_vel_jump_internal_mean",
        "validation_directional/chunk_root_ang_vel_jump_reset_first_mean",
        "validation_directional/chunk_joint_vel_error_reset_first_mean",
        "validation_directional/chunk_offset0_joint_pos_error_mean",
        "validation_directional/chunk_offset1_joint_vel_error_mean",
        "validation_directional/chunk_offset2_root_ang_vel_error_mean",
        "validation_directional/chunk_offset3_joint_pos_error_count",
    )
    assert set(required).issubset(metrics)
    assert metrics[required[0]] == 0.0
    for key in required[1:]:
        if key.endswith("_count"):
            assert metrics[key] == 0.0
        else:
            assert metrics[key] == -1.0
