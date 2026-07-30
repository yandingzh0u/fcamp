from __future__ import annotations

import inspect

import pytest
import torch

from components.rollout.reset_diagnostics import ResetPhaseRecorder


def test_reset_recorder_reports_exact_uniform_population_metrics() -> None:
    recorder = ResetPhaseRecorder(
        8,
        start_phase=0,
        device="cpu",
        log_num_bins=2,
    )
    recorder.begin()
    recorder.record(torch.tensor([0, 0, 1, 3, 4, 7]))
    metrics = recorder.finish()

    assert metrics["train_reset/all/count"] == 6
    assert metrics["train_reset/all/first_frame_bin_count"] == 2
    assert metrics[
        "train_reset/all/first_frame_bin_fraction"
    ] == pytest.approx(2 / 6)
    assert metrics["train_reset/all/other_frame_bin_count"] == 4
    assert metrics["train_reset/all/phase_min"] == 0
    assert metrics["train_reset/all/phase_mean"] == pytest.approx(2.5)
    assert metrics["train_reset/all/phase_p50"] == 1
    assert metrics["train_reset/all/phase_p95"] == 7
    assert metrics["train_reset/all/phase_max"] == 7
    assert metrics["train_reset/all/bin_0_count"] == 4
    assert metrics["train_reset/all/bin_0_fraction"] == pytest.approx(4 / 6)
    assert metrics["train_reset/all/bin_1_count"] == 2
    assert metrics["train_reset/all/bin_1_fraction"] == pytest.approx(2 / 6)
    assert metrics["train_reset/uniform_contract"] == 1
    assert metrics["train_reset/uniform_low"] == 0
    assert metrics["train_reset/uniform_high_exclusive"] == 7
    assert metrics["train_reset/uniform_floor_bin_count"] == 7


def test_reset_recorder_has_only_the_amp_reset_api() -> None:
    record_parameters = inspect.signature(ResetPhaseRecorder.record).parameters
    finish_parameters = inspect.signature(ResetPhaseRecorder.finish).parameters

    assert set(record_parameters) == {"self", "phases"}
    assert set(finish_parameters) == {"self"}


def test_reset_recorder_rejects_nonfinite_phase_data() -> None:
    recorder = ResetPhaseRecorder(8, start_phase=0, device="cpu")
    recorder.begin()

    with pytest.raises(ValueError, match="finite"):
        recorder.record(torch.tensor([0.0, float("nan")]))
