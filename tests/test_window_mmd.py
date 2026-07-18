from __future__ import annotations

import pytest
import torch

from components.evaluation.window_mmd import (
    DemoFeatureNormalizer,
    PhaseMatchedWindowMMD,
    sanitize_reference_phases,
)


def _tracker(*, num_envs: int = 4, windows: tuple[int, ...] = (2, 3)) -> PhaseMatchedWindowMMD:
    demo_dataset = torch.tensor(
        [
            [0.0, 0.0, 0.7, -1.0, 0.5],
            [0.2, 0.1, 0.8, 0.0, 1.0],
            [0.5, 0.4, 0.9, 1.0, 1.5],
            [0.9, 0.8, 1.0, 2.0, 2.0],
        ]
    )
    return PhaseMatchedWindowMMD(
        num_envs=num_envs,
        normalizer=DemoFeatureNormalizer.fit(demo_dataset),
        device="cpu",
        window_sizes=windows,
        bandwidths=(0.5, 1.0, 2.0),
        max_envs=num_envs,
        reference_phase_start=0.0,
        reference_phase_end=4.0,
    )


def test_identical_phase_matched_windows_have_zero_mmd_for_irregular_time_steps() -> None:
    tracker = _tracker()
    # These stand in for exact demo frames queried at non-uniform reference
    # phases. The tracker compares the paired phase path, not an assumed +1 grid.
    phases = (0.0, 0.35, 1.7, 1.85, 3.2, 3.75)
    for phase in phases:
        frame = torch.tensor([phase, 0.3 * phase, 0.7, phase**2, -phase]).repeat(4, 1)
        tracker.update_selected(
            frame,
            frame.clone(),
            torch.ones(4, dtype=torch.bool),
            torch.full((4,), phase),
        )

    metrics = tracker.metrics()
    assert metrics["validation/window_mmd2_w2"] == pytest.approx(0.0, abs=1.0e-8)
    assert metrics["validation/window_mmd2_w3"] == pytest.approx(0.0, abs=1.0e-8)
    assert metrics["validation/window_mmd_windows_w2"] == 20.0
    assert metrics["validation/window_mmd_windows_w3"] == 16.0
    assert metrics["validation/window_mmd_pair_gap_min_w2"] == 2.0
    assert metrics["validation/window_mmd_pair_gap_max_w2"] == 2.0
    assert metrics["validation/window_mmd_pair_gap_min_w3"] == 3.0
    assert metrics["validation/window_mmd_pair_gap_max_w3"] == 3.0
    assert metrics["validation/window_phase_endpoint_count_w3"] == 16.0
    assert metrics["validation/window_phase_endpoint_mean_w3"] == pytest.approx(2.625)
    assert metrics["validation/window_phase_endpoint_max_w3"] == pytest.approx(3.75)
    assert metrics["validation/window_reference_progress_max_w3"] == pytest.approx(0.9375)
    assert metrics["validation/window_reference_progress_span_w3"] == pytest.approx(0.5125)


def test_temporally_wrong_policy_windows_produce_positive_mmd() -> None:
    tracker = _tracker(num_envs=6, windows=(3,))
    for step in range(8):
        demo = torch.tensor(
            [float(step), 0.25 * step, 0.8, float(step * step), -float(step)]
        ).repeat(6, 1)
        policy = demo.clone()
        policy[:, 3:] += torch.tensor([5.0, 3.0])
        tracker.update_selected(
            policy,
            demo,
            torch.ones(6, dtype=torch.bool),
            torch.full((6,), float(step)),
        )

    metrics = tracker.metrics()
    assert metrics["validation/window_mmd2_w3"] > 1.0e-4
    assert metrics["validation/window_mmd_pairs_w3"] > 0


def test_dead_frame_breaks_continuity_and_never_enters_a_window() -> None:
    tracker = _tracker(num_envs=2, windows=(3,))
    frame = torch.zeros(2, 5)
    tracker.update_selected(frame, frame, torch.ones(2, dtype=torch.bool), torch.zeros(2))
    tracker.update_selected(
        frame, frame, torch.tensor([True, False]), torch.ones(2)
    )
    tracker.update_selected(frame, frame, torch.ones(2, dtype=torch.bool), torch.full((2,), 2.0))
    # Only env zero has three consecutive alive post-transition frames.
    assert tracker.metrics()["validation/window_mmd_windows_w3"] == 1.0
    tracker.update_selected(frame, frame, torch.ones(2, dtype=torch.bool), torch.full((2,), 3.0))
    # Env one still has only two consecutive frames; it remains excluded.
    assert tracker.metrics()["validation/window_mmd_windows_w3"] == 2.0


def test_selected_environment_ids_are_deterministic_evenly_spaced_and_bounded() -> None:
    tracker = PhaseMatchedWindowMMD(
        num_envs=10,
        normalizer=DemoFeatureNormalizer.fit(torch.randn(5, 5)),
        device="cpu",
        window_sizes=(2,),
        max_envs=4,
    )
    assert tracker.env_ids.tolist() == [0, 2, 5, 7]
    assert tracker.selected_env_count == 4


def test_terminal_and_wrapped_reference_phases_are_safe_but_invalid() -> None:
    phases = torch.tensor([-0.25, 0.0, 3.75, 4.0, 4.25, float("nan"), float("inf")])
    safe, valid = sanitize_reference_phases(phases, num_frames=5)

    torch.testing.assert_close(
        safe,
        torch.tensor([0.0, 0.0, 3.75, 4.0, 4.0, 0.0, 4.0]),
    )
    assert valid.tolist() == [False, True, True, True, False, False, False]
