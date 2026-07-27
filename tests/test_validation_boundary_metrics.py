from __future__ import annotations

import pytest
import torch

from engine.validation_metrics import (
    C2ServoDiagnostics,
    ChunkBoundaryDiagnostics,
)
from envs.action_servo import advance_position_servo


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

    # Environment one terminates on this current transition.  Validation passes
    # active-before AND not-terminal, so neither that terminal endpoint nor its
    # later inactive tail can contaminate a statistic.
    _update(
        diagnostics,
        active=torch.tensor([True, False]),
        offset=0,
        action_value=torch.tensor([1.0, 2.0]),
        joint_pos_value=torch.tensor([1.0, 2.0]),
        joint_vel_value=torch.tensor([1.0, 2.0]),
        root_ang_value=torch.tensor([1.0, 2.0]),
    )
    # Finish the first H=4 chunk, then execute one complete measured chunk.
    action_values = (2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0)
    joint_vel_values = (3.0, 6.0, 10.0, 15.0, 21.0, 28.0, 36.0)
    root_ang_values = (2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 29.0)
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

    assert metrics["validation/chunk_action_delta_reset_first_count"] == 1.0
    assert metrics["validation/chunk_action_delta_reset_first_mean"] == pytest.approx(1.0)
    assert metrics["validation/chunk_action_d2_reset_first_count"] == 0.0
    assert metrics["validation/chunk_action_d2_reset_first_mean"] == -1.0
    assert metrics["validation/chunk_joint_pos_error_reset_first_mean"] == pytest.approx(1.0)
    assert metrics["validation/chunk_joint_vel_error_reset_first_mean"] == pytest.approx(1.0)
    assert metrics["validation/chunk_root_ang_vel_error_reset_first_mean"] == pytest.approx(1.0)

    assert metrics["validation/chunk_action_delta_internal_count"] == 3.0
    assert metrics["validation/chunk_action_delta_internal_mean"] == pytest.approx(6.0)
    assert metrics["validation/chunk_action_delta_boundary_count"] == 1.0
    assert metrics["validation/chunk_action_delta_boundary_mean"] == pytest.approx(4.0)
    assert metrics["validation/chunk_action_delta_boundary_internal_ratio"] == pytest.approx(
        2.0 / 3.0
    )

    assert metrics["validation/chunk_action_d2_internal_mean"] == pytest.approx(1.0)
    assert metrics["validation/chunk_action_d2_boundary_mean"] == pytest.approx(1.0)
    assert metrics["validation/chunk_action_d2_boundary_internal_ratio"] == pytest.approx(1.0)
    assert metrics["validation/chunk_action_d3_internal_mean"] == pytest.approx(0.0)
    assert metrics["validation/chunk_action_d3_boundary_mean"] == pytest.approx(0.0)
    assert metrics["validation/chunk_joint_vel_jump_internal_mean"] == pytest.approx(7.0)
    assert metrics["validation/chunk_joint_vel_jump_boundary_mean"] == pytest.approx(5.0)
    assert metrics["validation/chunk_root_ang_vel_jump_internal_mean"] == pytest.approx(6.0)
    assert metrics["validation/chunk_root_ang_vel_jump_boundary_mean"] == pytest.approx(4.0)

    # Offset zero contains the surviving first step and later real boundary.
    assert metrics["validation/chunk_offset0_joint_pos_error_count"] == 2.0
    assert metrics["validation/chunk_offset1_joint_vel_error_count"] == 2.0
    assert metrics["validation/chunk_offset2_root_ang_vel_error_count"] == 2.0
    assert metrics["validation/chunk_offset3_joint_pos_error_count"] == 2.0


def test_per_environment_reset_never_becomes_a_chunk_boundary() -> None:
    initial_action, initial_joint_vel, initial_root_ang_vel = _state()
    diagnostics = ChunkBoundaryDiagnostics(
        horizon=4,
        initial_action=initial_action,
        initial_joint_vel=initial_joint_vel,
        initial_root_ang_vel=initial_root_ang_vel,
    )
    for offset in range(4):
        value = float(offset + 1)
        _update(
            diagnostics,
            active=torch.tensor([True, True]),
            offset=offset,
            action_value=torch.tensor([value, value]),
            joint_pos_value=torch.tensor([value, value]),
            joint_vel_value=torch.tensor([value, value]),
            root_ang_value=torch.tensor([value, value]),
        )

    reset_action = torch.tensor([[4.0, 4.0], [10.0, 10.0]])
    reset_joint_vel = torch.tensor([[4.0, 4.0], [20.0, 20.0]])
    reset_root_ang_vel = torch.tensor([[4.0, 4.0, 4.0], [30.0, 30.0, 30.0]])
    diagnostics.reset(
        torch.tensor([False, True]),
        initial_action=reset_action,
        initial_joint_vel=reset_joint_vel,
        initial_root_ang_vel=reset_root_ang_vel,
    )
    _update(
        diagnostics,
        active=torch.tensor([True, True]),
        offset=0,
        action_value=torch.tensor([5.0, 11.0]),
        joint_pos_value=torch.tensor([5.0, 11.0]),
        joint_vel_value=torch.tensor([5.0, 21.0]),
        root_ang_value=torch.tensor([5.0, 31.0]),
    )

    metrics = diagnostics.metrics()
    assert metrics["validation/chunk_action_delta_reset_first_count"] == 3.0
    assert metrics["validation/chunk_action_delta_internal_count"] == 0.0
    assert metrics["validation/chunk_action_delta_boundary_count"] == 1.0
    assert metrics["validation/chunk_action_delta_boundary_mean"] == pytest.approx(1.0)
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
        "validation_directional/chunk_action_d3_boundary_internal_ratio",
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


def _servo_update(
    diagnostics: C2ServoDiagnostics,
    *,
    active: torch.Tensor,
    offset: int | torch.Tensor,
    phase: torch.Tensor,
    target_action: torch.Tensor,
    previous_target_action: torch.Tensor,
    previous_action: torch.Tensor,
    previous_rate: torch.Tensor,
    previous_acceleration: torch.Tensor,
    actual_action: torch.Tensor,
    actual_rate: torch.Tensor,
    actual_acceleration: torch.Tensor,
    reference_action: torch.Tensor,
    projection_mask: torch.Tensor | None = None,
) -> None:
    if projection_mask is None:
        projection_mask = torch.zeros_like(actual_action, dtype=torch.bool)
    diagnostics.update(
        active_mask=active,
        chunk_offset=offset,
        transition_end_phase_steps=phase,
        target_action=target_action,
        previous_target_action=previous_target_action,
        previous_action=previous_action,
        previous_command_rate=previous_rate,
        previous_command_acceleration=previous_acceleration,
        actual_action=actual_action,
        actual_command_rate=actual_rate,
        actual_command_acceleration=actual_acceleration,
        action_projection_mask=projection_mask,
        reference_action=reference_action,
    )


def _advance(
    target_action: torch.Tensor,
    action: torch.Tensor,
    rate: torch.Tensor,
    acceleration: torch.Tensor,
    *,
    dt: float = 0.02,
    omega: float = 67.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return advance_position_servo(
        target_action,
        action,
        rate,
        acceleration,
        dt=dt,
        omega=omega,
    )


def _warm_up_servo(
    diagnostics: C2ServoDiagnostics,
    zeros: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Execute the complete first H chunk, which must not enter ratios."""

    action = rate = acceleration = zeros
    previous_target = zeros
    for offset in range(diagnostics.horizon):
        target = zeros
        next_state = _advance(target, action, rate, acceleration)
        _servo_update(
            diagnostics,
            active=torch.ones(zeros.shape[0], dtype=torch.bool),
            offset=offset,
            phase=torch.full((zeros.shape[0],), float(offset)),
            target_action=target,
            previous_target_action=previous_target,
            previous_action=action,
            previous_rate=rate,
            previous_acceleration=acceleration,
            actual_action=next_state[0],
            actual_rate=next_state[1],
            actual_acceleration=next_state[2],
            reference_action=torch.full_like(zeros, float(offset)),
        )
        previous_target = target
        action, rate, acceleration = next_state
    return action, rate, acceleration, previous_target


def test_c2_servo_diagnostics_bucket_exact_causal_values_by_offset() -> None:
    batch = 4
    dt = 0.02
    omega = 67.0
    zeros = torch.zeros(batch, 2)
    diagnostics = C2ServoDiagnostics(
        horizon=4,
        control_dt=dt,
        omega=omega,
        initial_action=zeros,
        initial_reference_action=zeros,
    )

    action, rate, acceleration, previous_target = _warm_up_servo(
        diagnostics, zeros
    )

    target_values = torch.arange(1.0, 5.0)
    target_action = target_values[:, None].expand(-1, 2).clone()
    next_action, next_rate, next_acceleration = _advance(
        target_action, action, rate, acceleration, dt=dt, omega=omega
    )
    _servo_update(
        diagnostics,
        active=torch.ones(batch, dtype=torch.bool),
        offset=torch.arange(4),
        phase=torch.full((batch,), 100.0),
        target_action=target_action,
        previous_target_action=previous_target,
        previous_action=action,
        previous_rate=rate,
        previous_acceleration=acceleration,
        actual_action=next_action,
        actual_rate=next_rate,
        actual_acceleration=next_acceleration,
        reference_action=torch.full((batch, 2), 6.0),
    )

    metrics = diagnostics.metrics()
    unit_action, _, _ = _advance(
        torch.ones(1, 1),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        torch.zeros(1, 1),
        dt=dt,
        omega=omega,
    )
    action_gain = float(unit_action.item())
    for offset, target in enumerate(target_values.tolist()):
        stem = f"validation/servo_offset{offset}"
        assert metrics[f"{stem}_target_action_abs_count"] == 1.0
        assert metrics[f"{stem}_target_action_abs_mean"] == pytest.approx(target)
        assert metrics[f"{stem}_target_increment_abs_mean"] == pytest.approx(target)
        assert metrics[f"{stem}_target_increment_d2_abs_mean"] == pytest.approx(
            target
        )
        assert metrics[f"{stem}_previous_command_rate_abs_mean"] == 0.0
        assert metrics[f"{stem}_position_error_abs_mean"] == pytest.approx(target)
        assert metrics[f"{stem}_initial_jerk_abs_mean"] == pytest.approx(
            omega**3 * target
        )
        assert metrics[f"{stem}_predicted_action_d2_abs_mean"] == pytest.approx(
            action_gain * target, rel=1.0e-5, abs=1.0e-8
        )
        assert metrics[f"{stem}_actual_action_d2_abs_mean"] == pytest.approx(
            action_gain * target, rel=1.0e-5, abs=1.0e-8
        )
        assert metrics[f"{stem}_reference_action_delta_abs_mean"] == pytest.approx(
            3.0
        )
        assert metrics[f"{stem}_reference_action_d2_abs_mean"] == pytest.approx(
            2.0
        )
        assert metrics[f"{stem}_prediction_residual_abs_mean"] == pytest.approx(
            0.0, abs=1.0e-7
        )
        assert metrics[
            f"{stem}_rate_prediction_residual_abs_mean"
        ] == pytest.approx(0.0, abs=1.0e-6)
        assert metrics[
            f"{stem}_acceleration_prediction_residual_abs_mean"
        ] == pytest.approx(0.0, abs=1.0e-5)
        assert metrics[f"{stem}_projection_joint_fraction_mean"] == 0.0

    assert metrics[
        "validation/servo_position_error_abs_boundary_internal_ratio"
    ] == pytest.approx(1.0 / 3.0)
    assert metrics[
        "validation/servo_actual_action_d2_abs_boundary_internal_ratio"
    ] == pytest.approx(1.0 / 3.0)
    assert metrics[
        "validation/servo_reference_action_d2_abs_boundary_internal_ratio"
    ] == pytest.approx(1.0)
    assert metrics[
        "validation/servo_target_increment_d2_abs_boundary_internal_ratio"
    ] == pytest.approx(1.0 / 3.0)
    assert metrics[
        "validation/servo_reference_action_delta_abs_boundary_internal_ratio"
    ] == pytest.approx(1.0)


def test_c2_servo_diagnostics_separates_target_increment_from_tracking_error() -> None:
    zeros = torch.zeros(1, 1)
    diagnostics = C2ServoDiagnostics(
        horizon=4,
        control_dt=0.02,
        omega=67.0,
        initial_action=zeros,
        initial_reference_action=zeros,
    )
    with pytest.raises(
        ValueError,
        match=r"previous_target_action must have shape \(1, 1\)",
    ):
        _servo_update(
            diagnostics,
            active=torch.tensor([True]),
            offset=0,
            phase=torch.zeros(1),
            target_action=zeros,
            previous_target_action=torch.zeros(1, 2),
            previous_action=zeros,
            previous_rate=zeros,
            previous_acceleration=zeros,
            actual_action=zeros,
            actual_rate=zeros,
            actual_acceleration=zeros,
            reference_action=zeros,
        )

    action, rate, acceleration, previous_target = _warm_up_servo(
        diagnostics, zeros
    )
    for offset, (target_value, reference_value) in enumerate(
        ((1.0, 0.0), (1.0, 1.0), (1.25, 2.0))
    ):
        target = torch.full_like(zeros, target_value)
        next_state = _advance(target, action, rate, acceleration)
        action_before_step = action
        _servo_update(
            diagnostics,
            active=torch.tensor([True]),
            offset=offset,
            phase=torch.tensor([float(offset)]),
            target_action=target,
            previous_target_action=previous_target,
            previous_action=action,
            previous_rate=rate,
            previous_acceleration=acceleration,
            actual_action=next_state[0],
            actual_rate=next_state[1],
            actual_acceleration=next_state[2],
            reference_action=torch.full_like(zeros, reference_value),
        )
        previous_target = target
        action, rate, acceleration = next_state

    metrics = diagnostics.metrics()
    assert metrics[
        "validation/servo_offset2_target_increment_abs_mean"
    ] == pytest.approx(0.25)
    assert metrics[
        "validation/servo_offset2_target_increment_d2_abs_mean"
    ] == pytest.approx(0.25)
    assert metrics[
        "validation/servo_offset2_position_error_abs_mean"
    ] == pytest.approx(float((torch.tensor(1.25) - action_before_step).item()))


def test_c2_servo_phase_window_uses_transition_end_phase() -> None:
    batch = 5
    zeros = torch.zeros(batch, 1)
    diagnostics = C2ServoDiagnostics(
        horizon=4,
        control_dt=0.02,
        omega=67.0,
        initial_action=zeros,
        initial_reference_action=zeros,
    )
    action, rate, acceleration, previous_target = _warm_up_servo(
        diagnostics, zeros
    )

    target_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 9.0])
    target_action = target_values[:, None]
    next_state = _advance(target_action, action, rate, acceleration)
    _servo_update(
        diagnostics,
        active=torch.tensor([True, True, True, True, False]),
        offset=0,
        phase=torch.tensor([279.0, 280.0, 310.0, 311.0, 300.0]),
        target_action=target_action,
        previous_target_action=previous_target,
        previous_action=action,
        previous_rate=rate,
        previous_acceleration=acceleration,
        actual_action=next_state[0],
        actual_rate=next_state[1],
        actual_acceleration=next_state[2],
        reference_action=torch.full((batch, 1), 6.0),
    )

    metrics = diagnostics.metrics()
    assert metrics["validation/servo_offset0_target_action_abs_count"] == 4.0
    assert metrics["validation/servo_offset0_target_action_abs_p95"] == pytest.approx(
        float(torch.quantile(torch.tensor([1.0, 2.0, 3.0, 4.0]), 0.95))
    )
    assert metrics["validation/servo_offset0_target_action_abs_p99"] == pytest.approx(
        float(torch.quantile(torch.tensor([1.0, 2.0, 3.0, 4.0]), 0.99))
    )
    phase_stem = "validation/servo_transition_end_phase280_310_offset0"
    assert metrics[f"{phase_stem}_target_action_abs_count"] == 2.0
    assert metrics[f"{phase_stem}_target_action_abs_mean"] == pytest.approx(2.5)


def test_c2_servo_diagnostics_isolate_projection_residual() -> None:
    zeros = torch.zeros(1, 1)
    diagnostics = C2ServoDiagnostics(
        horizon=4,
        control_dt=0.02,
        omega=67.0,
        initial_action=zeros,
        initial_reference_action=zeros,
    )
    action, rate, acceleration, previous_target = _warm_up_servo(
        diagnostics, zeros
    )

    target_action = torch.ones_like(zeros)
    predicted = _advance(target_action, action, rate, acceleration)
    _servo_update(
        diagnostics,
        active=torch.tensor([True]),
        offset=2,
        phase=torch.tensor([2.0]),
        target_action=target_action,
        previous_target_action=previous_target,
        previous_action=action,
        previous_rate=rate,
        previous_acceleration=acceleration,
        actual_action=predicted[0] + 0.01,
        actual_rate=predicted[1] + 0.02,
        actual_acceleration=predicted[2] - 0.03,
        reference_action=torch.full((1, 1), 6.0),
        projection_mask=torch.ones_like(zeros, dtype=torch.bool),
    )

    metrics = diagnostics.metrics()
    assert metrics[
        "validation/servo_offset2_reference_action_d2_abs_mean"
    ] == pytest.approx(2.0)
    assert metrics[
        "validation/servo_offset2_projection_joint_fraction_mean"
    ] == pytest.approx(1.0)
    assert metrics[
        "validation/servo_offset2_prediction_residual_abs_mean"
    ] == pytest.approx(0.01)
    assert metrics[
        "validation/servo_offset2_rate_prediction_residual_abs_mean"
    ] == pytest.approx(0.02, abs=1.0e-6)
    assert metrics[
        "validation/servo_offset2_acceleration_prediction_residual_abs_mean"
    ] == pytest.approx(0.03, abs=5.0e-5)


def test_empty_c2_servo_diagnostics_have_stable_schema() -> None:
    diagnostics = C2ServoDiagnostics(
        horizon=4,
        control_dt=0.02,
        omega=67.0,
        initial_action=torch.zeros(1, 2),
        initial_reference_action=torch.zeros(1, 2),
    )

    metrics = diagnostics.metrics(prefix="validation_directional")
    required = (
        "validation_directional/servo_offset0_target_increment_abs_count",
        "validation_directional/servo_offset0_target_increment_d2_abs_count",
        "validation_directional/servo_reference_action_delta_abs_boundary_internal_ratio",
        "validation_directional/servo_offset0_position_error_abs_count",
        "validation_directional/servo_offset3_actual_action_d3_abs_p99",
        "validation_directional/servo_position_error_abs_boundary_internal_ratio",
        "validation_directional/servo_transition_end_phase280_310_offset0_reference_action_d2_abs_mean",
        "validation_directional/servo_transition_end_phase280_310_projection_joint_fraction_internal_mean",
    )
    assert set(required).issubset(metrics)
    assert metrics[required[0]] == 0.0
    for key in required[1:]:
        if key.endswith("_count"):
            assert metrics[key] == 0.0
        else:
            assert metrics[key] == -1.0
