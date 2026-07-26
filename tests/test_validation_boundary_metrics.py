from __future__ import annotations

import pytest
import torch

from engine.validation_metrics import (
    ChunkBoundaryDiagnostics,
    RateControllerDiagnostics,
)


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


def _rate_update(
    diagnostics: RateControllerDiagnostics,
    *,
    active: torch.Tensor,
    offset: int | torch.Tensor,
    phase: torch.Tensor,
    raw: torch.Tensor,
    previous_rate: torch.Tensor,
    actual_rate: torch.Tensor,
    reference_action: torch.Tensor,
) -> None:
    diagnostics.update(
        active_mask=active,
        chunk_offset=offset,
        transition_end_phase_steps=phase,
        raw_target_rate=raw,
        previous_command_rate=previous_rate,
        actual_command_rate=actual_rate,
        reference_action=reference_action,
    )


def test_rate_diagnostics_bucket_exact_causal_values_by_offset() -> None:
    batch = 4
    dt = 0.1
    decay = 0.5
    rate_limit = torch.tensor([10.0, 10.0])
    diagnostics = RateControllerDiagnostics(
        horizon=4,
        control_dt=dt,
        decay=decay,
        initial_reference_action=torch.zeros(batch, 2),
        command_rate_limit=rate_limit,
    )

    # Seed the first reference delta.  Reset-first is deliberately absent from
    # all offset buckets, including offset zero.
    zeros = torch.zeros(batch, 2)
    _rate_update(
        diagnostics,
        active=torch.ones(batch, dtype=torch.bool),
        offset=0,
        phase=torch.zeros(batch),
        raw=zeros,
        previous_rate=zeros,
        actual_rate=zeros,
        reference_action=torch.ones(batch, 2),
    )

    target_values = torch.arange(1.0, 5.0)
    raw = torch.atanh((target_values / 10.0)[:, None]).expand(-1, 2).clone()
    previous_rate = torch.zeros_like(raw)
    actual_rate = 0.5 * target_values[:, None].expand(-1, 2)
    _rate_update(
        diagnostics,
        active=torch.ones(batch, dtype=torch.bool),
        offset=torch.arange(4),
        phase=torch.full((batch,), 100.0),
        raw=raw,
        previous_rate=previous_rate,
        actual_rate=actual_rate,
        reference_action=torch.full((batch, 2), 3.0),
    )

    metrics = diagnostics.metrics()
    for offset, target in enumerate(target_values.tolist()):
        stem = f"validation/rate_offset{offset}"
        assert metrics[f"{stem}_target_rate_abs_count"] == 1.0
        assert metrics[f"{stem}_target_rate_abs_mean"] == pytest.approx(target)
        assert metrics[f"{stem}_target_rate_support_mean"] == pytest.approx(
            target / 10.0
        )
        assert metrics[f"{stem}_previous_command_rate_abs_mean"] == 0.0
        assert metrics[f"{stem}_rate_error_abs_mean"] == pytest.approx(target)
        assert metrics[f"{stem}_predicted_action_d2_abs_mean"] == pytest.approx(
            0.05 * target
        )
        assert metrics[f"{stem}_actual_action_d2_abs_mean"] == pytest.approx(
            0.05 * target
        )
        assert metrics[f"{stem}_reference_action_d2_abs_mean"] == pytest.approx(
            1.0
        )
        assert metrics[f"{stem}_prediction_residual_abs_mean"] == pytest.approx(
            0.0, abs=1.0e-7
        )
        assert metrics[f"{stem}_projection_joint_fraction_mean"] == 0.0

    assert metrics[
        "validation/rate_rate_error_abs_boundary_internal_ratio"
    ] == pytest.approx(1.0 / 3.0)
    assert metrics[
        "validation/rate_actual_action_d2_abs_boundary_internal_ratio"
    ] == pytest.approx(1.0 / 3.0)
    assert metrics[
        "validation/rate_reference_action_d2_abs_boundary_internal_ratio"
    ] == pytest.approx(1.0)


def test_rate_diagnostics_phase_window_is_inclusive_and_uses_transition_end_phase() -> None:
    batch = 5
    rate_limit = torch.tensor([10.0])
    diagnostics = RateControllerDiagnostics(
        horizon=4,
        control_dt=0.02,
        decay=0.5,
        initial_reference_action=torch.zeros(batch, 1),
        command_rate_limit=rate_limit,
    )
    zeros = torch.zeros(batch, 1)
    _rate_update(
        diagnostics,
        active=torch.ones(batch, dtype=torch.bool),
        offset=0,
        phase=torch.zeros(batch),
        raw=zeros,
        previous_rate=zeros,
        actual_rate=zeros,
        reference_action=torch.ones(batch, 1),
    )

    target_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 9.0])
    raw = torch.atanh((target_values / 10.0)[:, None])
    proposed_rate = 0.5 * target_values[:, None]
    _rate_update(
        diagnostics,
        active=torch.tensor([True, True, True, True, False]),
        offset=0,
        phase=torch.tensor([279.0, 280.0, 310.0, 311.0, 300.0]),
        raw=raw,
        previous_rate=zeros,
        actual_rate=proposed_rate,
        reference_action=torch.full((batch, 1), 3.0),
    )

    metrics = diagnostics.metrics()
    assert metrics["validation/rate_offset0_target_rate_abs_count"] == 4.0
    assert metrics["validation/rate_offset0_target_rate_abs_p95"] == pytest.approx(
        float(torch.quantile(torch.tensor([1.0, 2.0, 3.0, 4.0]), 0.95))
    )
    assert metrics["validation/rate_offset0_target_rate_abs_p99"] == pytest.approx(
        float(torch.quantile(torch.tensor([1.0, 2.0, 3.0, 4.0]), 0.99))
    )
    phase_stem = "validation/rate_transition_end_phase280_310_offset0"
    assert metrics[f"{phase_stem}_target_rate_abs_count"] == 2.0
    assert metrics[f"{phase_stem}_target_rate_abs_mean"] == pytest.approx(2.5)


def test_rate_diagnostics_separate_reference_d2_and_projection_residual() -> None:
    rate_limit = torch.tensor([10.0])
    diagnostics = RateControllerDiagnostics(
        horizon=4,
        control_dt=0.02,
        decay=0.5,
        initial_reference_action=torch.zeros(1, 1),
        command_rate_limit=rate_limit,
    )
    _rate_update(
        diagnostics,
        active=torch.tensor([True]),
        offset=0,
        phase=torch.tensor([0.0]),
        raw=torch.zeros(1, 1),
        previous_rate=torch.zeros(1, 1),
        actual_rate=torch.zeros(1, 1),
        reference_action=torch.ones(1, 1),
    )

    raw = torch.full((1, 1), 100.0)
    previous_rate = torch.full((1, 1), 10.0)
    actual_rate = torch.full((1, 1), 0.5)
    _rate_update(
        diagnostics,
        active=torch.tensor([True]),
        offset=1,
        phase=torch.tensor([1.0]),
        raw=raw,
        previous_rate=previous_rate,
        actual_rate=actual_rate,
        reference_action=torch.full((1, 1), 3.0),
    )
    _rate_update(
        diagnostics,
        active=torch.tensor([True]),
        offset=2,
        phase=torch.tensor([2.0]),
        raw=torch.zeros(1, 1),
        previous_rate=torch.zeros(1, 1),
        actual_rate=torch.zeros(1, 1),
        reference_action=torch.full((1, 1), 6.0),
    )

    metrics = diagnostics.metrics()
    assert metrics[
        "validation/rate_offset1_reference_action_d2_abs_mean"
    ] == pytest.approx(1.0)
    assert metrics[
        "validation/rate_offset2_reference_action_d2_abs_mean"
    ] == pytest.approx(1.0)
    assert metrics[
        "validation/rate_offset1_projection_joint_fraction_mean"
    ] == pytest.approx(1.0)
    assert metrics[
        "validation/rate_offset1_prediction_residual_abs_mean"
    ] == pytest.approx(0.19)


def test_rate_projection_diagnostic_ignores_float32_rate_reconstruction_noise() -> None:
    batch = 256
    dt = 0.02
    decay = 0.5
    rate_limit = torch.tensor([10.0, 10.0])
    diagnostics = RateControllerDiagnostics(
        horizon=4,
        control_dt=dt,
        decay=decay,
        initial_reference_action=torch.zeros(batch, 2),
        command_rate_limit=rate_limit,
    )
    zeros = torch.zeros(batch, 2)
    _rate_update(
        diagnostics,
        active=torch.ones(batch, dtype=torch.bool),
        offset=0,
        phase=torch.zeros(batch),
        raw=zeros,
        previous_rate=zeros,
        actual_rate=zeros,
        reference_action=torch.ones(batch, 2),
    )

    generator = torch.Generator().manual_seed(19)
    previous_action = 2.0 * torch.rand(batch, 2, generator=generator) - 1.0
    previous_rate = 10.0 * torch.rand(batch, 2, generator=generator) - 5.0
    raw = 0.8 * torch.rand(batch, 2, generator=generator) - 0.4
    target_rate = rate_limit * torch.tanh(raw)
    proposed_rate = decay * previous_rate + (1.0 - decay) * target_rate
    applied_action = previous_action + dt * proposed_rate
    reconstructed_rate = (applied_action - previous_action) / dt
    _rate_update(
        diagnostics,
        active=torch.ones(batch, dtype=torch.bool),
        offset=1,
        phase=torch.ones(batch),
        raw=raw,
        previous_rate=previous_rate,
        actual_rate=reconstructed_rate,
        reference_action=torch.full((batch, 2), 3.0),
    )

    metrics = diagnostics.metrics()
    assert metrics[
        "validation/rate_offset1_projection_joint_fraction_mean"
    ] == 0.0
    assert metrics[
        "validation/rate_offset1_prediction_residual_abs_p99"
    ] < 1.0e-5


def test_empty_rate_diagnostics_have_stable_schema() -> None:
    diagnostics = RateControllerDiagnostics(
        horizon=4,
        control_dt=0.02,
        decay=0.5,
        initial_reference_action=torch.zeros(1, 2),
        command_rate_limit=torch.ones(2),
    )

    metrics = diagnostics.metrics(prefix="validation_directional")
    required = (
        "validation_directional/rate_offset0_rate_error_abs_count",
        "validation_directional/rate_offset3_actual_action_d2_abs_p99",
        "validation_directional/rate_rate_error_abs_boundary_internal_ratio",
        "validation_directional/rate_transition_end_phase280_310_offset0_reference_action_d2_abs_mean",
        "validation_directional/rate_transition_end_phase280_310_projection_joint_fraction_internal_mean",
    )
    assert set(required).issubset(metrics)
    assert metrics[required[0]] == 0.0
    for key in required[1:]:
        assert metrics[key] == -1.0
