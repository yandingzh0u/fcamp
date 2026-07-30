from __future__ import annotations

import pytest
import torch

from engine.validation_metrics import terminal_phase_metrics


def test_terminal_phase_metrics_are_cause_specific_and_stable_when_empty() -> None:
    phases = torch.tensor([10.0, 20.0, 30.0])
    metrics = terminal_phase_metrics(
        "validation/terminal/failure",
        phases,
        torch.tensor([True, True, False]),
        motion_end_phase=100.0,
    )
    assert metrics["validation/terminal/failure/count"] == 2.0
    assert metrics["validation/terminal/failure/phase_mean"] == pytest.approx(15.0)
    assert metrics["validation/terminal/failure/phase_p50"] == pytest.approx(15.0)
    assert metrics["validation/terminal/failure/phase_p95"] == pytest.approx(19.5)
    assert metrics["validation/terminal/failure/phase_progress_mean"] == pytest.approx(0.15)

    empty = terminal_phase_metrics(
        "validation/terminal/time_out",
        phases,
        torch.zeros(3, dtype=torch.bool),
        motion_end_phase=100.0,
    )
    assert empty["validation/terminal/time_out/count"] == 0.0
    assert empty["validation/terminal/time_out/phase_p50"] == -1.0
