from __future__ import annotations

import pytest
import torch

from engine.validation_metrics import StepDiagnostics


def _state(
    batch: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros(batch, 2),
        torch.zeros(batch, 2),
        torch.zeros(batch, 3),
        torch.zeros(batch, 3),
    )


def _update(
    diagnostics: StepDiagnostics,
    *,
    active: torch.Tensor,
    action_value: torch.Tensor,
    joint_pos_value: torch.Tensor,
    joint_vel_value: torch.Tensor,
    root_lin_value: torch.Tensor,
    root_ang_value: torch.Tensor,
) -> None:
    batch = int(active.numel())
    diagnostics.update(
        active_mask=active,
        action=action_value[:, None].expand(-1, 2).clone(),
        joint_pos=joint_pos_value[:, None].expand(-1, 2).clone(),
        joint_vel=joint_vel_value[:, None].expand(-1, 2).clone(),
        root_lin_vel=root_lin_value[:, None].expand(-1, 3).clone(),
        root_ang_vel=root_ang_value[:, None].expand(-1, 3).clone(),
        reference_joint_pos=torch.zeros(batch, 2),
        reference_joint_vel=torch.zeros(batch, 2),
        reference_root_lin_vel=torch.zeros(batch, 3),
        reference_root_ang_vel=torch.zeros(batch, 3),
    )


def test_step_metrics_separate_initial_action_and_consecutive_dynamics() -> None:
    initial_action, initial_joint_vel, initial_root_lin, initial_root_ang = (
        _state()
    )
    diagnostics = StepDiagnostics(
        initial_action=initial_action,
        initial_joint_vel=initial_joint_vel,
        initial_root_lin_vel=initial_root_lin,
        initial_root_ang_vel=initial_root_ang,
    )

    # Both environments execute the first post-reset action. Environment one
    # then becomes inactive and must never contribute its sentinel values.
    _update(
        diagnostics,
        active=torch.tensor([True, True]),
        action_value=torch.tensor([1.0, 2.0]),
        joint_pos_value=torch.tensor([1.0, 2.0]),
        joint_vel_value=torch.tensor([1.0, 2.0]),
        root_lin_value=torch.tensor([1.0, 2.0]),
        root_ang_value=torch.tensor([1.0, 2.0]),
    )
    for action, joint_vel, root_lin, root_ang in zip(
        (2.0, 4.0, 7.0),
        (3.0, 6.0, 10.0),
        (2.0, 4.0, 7.0),
        (3.0, 6.0, 10.0),
        strict=True,
    ):
        _update(
            diagnostics,
            active=torch.tensor([True, False]),
            action_value=torch.tensor([action, -999.0]),
            joint_pos_value=torch.tensor([action, -999.0]),
            joint_vel_value=torch.tensor([joint_vel, -999.0]),
            root_lin_value=torch.tensor([root_lin, -999.0]),
            root_ang_value=torch.tensor([root_ang, -999.0]),
        )

    metrics = diagnostics.metrics()
    assert metrics["validation/initial/action_delta/count"] == 2.0
    assert metrics["validation/initial/action_delta/mean"] == pytest.approx(1.5)
    assert metrics["validation/dynamics/action_delta/count"] == 3.0
    assert metrics["validation/dynamics/action_delta/mean"] == pytest.approx(2.0)
    assert metrics["validation/dynamics/action_d2/mean"] == pytest.approx(
        2.0 / 3.0
    )
    assert metrics[
        "validation/dynamics/joint_vel_jump/mean"
    ] == pytest.approx(3.0)
    assert metrics[
        "validation/dynamics/root_lin_vel_jump/mean"
    ] == pytest.approx(2.0)
    assert metrics[
        "validation/dynamics/root_ang_vel_jump/mean"
    ] == pytest.approx(3.0)
    assert metrics["validation/tracking/joint_pos_mae/count"] == 5.0
    assert metrics["validation/tracking/root_lin_vel_mae/count"] == 5.0


def test_per_environment_reset_restarts_the_consecutive_step_contract() -> None:
    initial_action, initial_joint_vel, initial_root_lin, initial_root_ang = (
        _state()
    )
    diagnostics = StepDiagnostics(
        initial_action=initial_action,
        initial_joint_vel=initial_joint_vel,
        initial_root_lin_vel=initial_root_lin,
        initial_root_ang_vel=initial_root_ang,
    )
    _update(
        diagnostics,
        active=torch.tensor([True, True]),
        action_value=torch.tensor([1.0, 1.0]),
        joint_pos_value=torch.tensor([1.0, 1.0]),
        joint_vel_value=torch.tensor([1.0, 1.0]),
        root_lin_value=torch.tensor([1.0, 1.0]),
        root_ang_value=torch.tensor([1.0, 1.0]),
    )

    diagnostics.reset(
        torch.tensor([False, True]),
        initial_action=torch.tensor([[1.0, 1.0], [10.0, 10.0]]),
        initial_joint_vel=torch.tensor([[1.0, 1.0], [20.0, 20.0]]),
        initial_root_lin_vel=torch.tensor(
            [[1.0, 1.0, 1.0], [30.0, 30.0, 30.0]]
        ),
        initial_root_ang_vel=torch.tensor(
            [[1.0, 1.0, 1.0], [40.0, 40.0, 40.0]]
        ),
    )
    _update(
        diagnostics,
        active=torch.tensor([True, True]),
        action_value=torch.tensor([2.0, 11.0]),
        joint_pos_value=torch.tensor([2.0, 11.0]),
        joint_vel_value=torch.tensor([2.0, 21.0]),
        root_lin_value=torch.tensor([2.0, 31.0]),
        root_ang_value=torch.tensor([2.0, 41.0]),
    )

    metrics = diagnostics.metrics()
    assert metrics["validation/initial/action_delta/count"] == 3.0
    assert metrics["validation/dynamics/action_delta/count"] == 1.0
    assert metrics["validation/dynamics/action_delta/mean"] == pytest.approx(1.0)
    assert metrics["validation/dynamics/joint_vel_jump/count"] == 1.0
    assert metrics["validation/tracking/joint_pos_mae/count"] == 4.0


def test_empty_step_metrics_have_stable_h1_schema() -> None:
    initial_action, initial_joint_vel, initial_root_lin, initial_root_ang = (
        _state(batch=1)
    )
    diagnostics = StepDiagnostics(
        initial_action=initial_action,
        initial_joint_vel=initial_joint_vel,
        initial_root_lin_vel=initial_root_lin,
        initial_root_ang_vel=initial_root_ang,
    )

    metrics = diagnostics.metrics(prefix="validation_directional")
    required = (
        "validation_directional/dynamics/action_delta/count",
        "validation_directional/dynamics/action_delta/mean",
        "validation_directional/dynamics/action_d2/p95",
        "validation_directional/dynamics/joint_vel_jump/p99",
        "validation_directional/dynamics/root_lin_vel_jump/mean",
        "validation_directional/dynamics/root_ang_vel_jump/mean",
        "validation_directional/tracking/joint_pos_mae/mean",
        "validation_directional/tracking/joint_vel_mae/p95",
        "validation_directional/tracking/root_lin_vel_mae/p99",
        "validation_directional/tracking/root_ang_vel_mae/count",
        "validation_directional/initial/action_delta/mean",
    )
    assert set(required).issubset(metrics)
    for key in required:
        assert metrics[key] == (0.0 if key.endswith("/count") else -1.0)
    assert not any(
        marker in key
        for key in metrics
        for marker in ("chunk", "boundary", "internal", "offset")
    )
