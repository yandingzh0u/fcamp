import pytest
import torch

from components.rollout.reset_diagnostics import ResetPhaseRecorder
from components.rollout.training_streams import (
    CURRICULUM_STREAM,
    PHASE0_STREAM,
    Phase0AttemptTracker,
    Phase0CurriculumStreams,
)


class _TwoBinSampler:
    num_bins = 2

    @staticmethod
    def frames_to_bins(frame_ids: torch.Tensor) -> torch.Tensor:
        return frame_ids // 4


def _streams(num_envs: int = 10) -> Phase0CurriculumStreams:
    return Phase0CurriculumStreams.create(
        num_envs,
        phase0_fraction=0.10,
        phase0_start=0,
        device="cpu",
    )


def test_stream_partition_is_deterministic_and_exact_for_10_90() -> None:
    first = _streams()
    second = _streams()

    expected = torch.tensor(
        [
            PHASE0_STREAM,
            CURRICULUM_STREAM,
            CURRICULUM_STREAM,
            CURRICULUM_STREAM,
            CURRICULUM_STREAM,
            CURRICULUM_STREAM,
            CURRICULUM_STREAM,
            CURRICULUM_STREAM,
            CURRICULUM_STREAM,
            CURRICULUM_STREAM,
        ],
        dtype=torch.int8,
    )
    assert torch.equal(first.stream_ids, expected)
    assert torch.equal(second.stream_ids, expected)
    assert torch.equal(first.phase0_ids, torch.tensor([0]))
    assert torch.equal(first.curriculum_ids, torch.arange(1, 10))
    assert first.phase0_fraction == pytest.approx(0.10)


def test_production_8192_partition_assigns_819_phase0_attempts() -> None:
    streams = _streams(num_envs=8192)

    assert streams.phase0_ids.numel() == 819
    assert streams.curriculum_ids.numel() == 7373
    assert float(streams.phase0_ids.numel() / 8192) == pytest.approx(
        0.0999755859375
    )


def test_single_environment_is_always_assigned_to_phase0_stream() -> None:
    streams = _streams(num_envs=1)

    assert streams.stream_ids.tolist() == [PHASE0_STREAM]
    assert streams.phase0_ids.tolist() == [0]
    assert streams.curriculum_ids.numel() == 0


def test_reset_phases_preserve_reset_id_order_and_sample_only_curriculum() -> None:
    streams = _streams()
    reset_ids = torch.tensor([5, 0, 7, 1, 3])
    sample_counts: list[int] = []

    def sample_curriculum(count: int) -> torch.Tensor:
        sample_counts.append(count)
        return torch.tensor([11, 22, 33, 44])

    phases, reset_streams = streams.reset_phases(
        reset_ids,
        sample_curriculum,
    )

    assert sample_counts == [4]
    assert reset_streams.tolist() == [
        CURRICULUM_STREAM,
        PHASE0_STREAM,
        CURRICULUM_STREAM,
        CURRICULUM_STREAM,
        CURRICULUM_STREAM,
    ]
    assert phases.tolist() == [11, 0, 22, 33, 44]


def test_reset_phases_do_not_call_curriculum_sampler_for_phase0_only() -> None:
    streams = _streams()

    def unexpected_sample(_: int) -> torch.Tensor:
        raise AssertionError("curriculum sampler must not be called")

    phases, reset_streams = streams.reset_phases(
        torch.tensor([0]),
        unexpected_sample,
    )

    assert phases.tolist() == [0]
    assert reset_streams.tolist() == [PHASE0_STREAM]


def test_phase0_attempt_tracker_persists_and_classifies_real_terminals() -> None:
    streams = _streams()
    tracker = Phase0AttemptTracker(streams.stream_ids)
    tracker.start(streams.phase0_ids)
    alive = streams.phase0_mask
    empty = torch.zeros(10, dtype=torch.bool)

    tracker.begin_update()
    tracker.observe_step(alive, empty, empty, empty, empty)
    tracker.observe_step(alive, empty, empty, empty, empty)
    first = tracker.metrics()
    assert first["stream/phase0_attempt/inflight_count"] == 1
    assert first["stream/phase0_attempt/inflight_age_max"] == 2

    # An optimizer boundary resets only update-local counters, never trajectory
    # identity or age.
    tracker.begin_update()
    success = torch.zeros(10, dtype=torch.bool)
    success[0] = True
    tracker.observe_step(alive, success, empty, empty, success)
    resolved = tracker.metrics()
    assert resolved["stream/phase0_attempt/update_succeeded"] == 1
    assert resolved["stream/phase0_attempt/cumulative_success_rate"] == 1
    assert resolved["stream/phase0_attempt/inflight_count"] == 0

    tracker.start(streams.phase0_ids)
    failure = torch.zeros(10, dtype=torch.bool)
    failure[0] = True
    tracker.observe_step(alive, failure, failure, empty, empty)
    final = tracker.metrics()
    assert final["stream/phase0_attempt/cumulative_started"] == 2
    assert final["stream/phase0_attempt/cumulative_resolved"] == 2
    assert final["stream/phase0_attempt/cumulative_failed"] == 1
    assert final["stream/phase0_attempt/cumulative_succeeded"] == 1
    assert final["stream/phase0_attempt/conservation_error"] == 0


def test_phase0_attempt_resume_interrupts_inflight_without_faking_outcome() -> None:
    streams = _streams()
    original = Phase0AttemptTracker(streams.stream_ids)
    original.start(streams.phase0_ids)
    alive = streams.phase0_mask
    empty = torch.zeros(10, dtype=torch.bool)
    original.observe_step(alive, empty, empty, empty, empty)

    restored = Phase0AttemptTracker(streams.stream_ids)
    restored.load_state_dict(original.state_dict())
    restored.interrupt_inflight()
    restored.start(streams.phase0_ids)
    metrics = restored.metrics()

    assert metrics["stream/phase0_attempt/cumulative_started"] == 2
    assert metrics["stream/phase0_attempt/cumulative_interrupted"] == 1
    assert metrics["stream/phase0_attempt/cumulative_resolved"] == 0
    assert metrics["stream/phase0_attempt/inflight_count"] == 1
    assert metrics["stream/phase0_attempt/conservation_error"] == 0


def test_reset_recorder_reports_exact_all_stream_and_bin_metrics() -> None:
    recorder = ResetPhaseRecorder(
        8,
        start_phase=0,
        device="cpu",
    )
    recorder.begin()
    recorder.record(
        torch.tensor([0, 0, 1, 3, 4, 7]),
        torch.tensor(
            [
                PHASE0_STREAM,
                PHASE0_STREAM,
                CURRICULUM_STREAM,
                CURRICULUM_STREAM,
                CURRICULUM_STREAM,
                CURRICULUM_STREAM,
            ]
        ),
    )
    metrics = recorder.finish(_TwoBinSampler())

    assert metrics["train_reset/all/count"] == 6
    assert metrics["train_reset/all/start_count"] == 2
    assert metrics["train_reset/all/start_fraction"] == pytest.approx(2 / 6)
    assert metrics["train_reset/all/nonstart_count"] == 4
    assert metrics["train_reset/all/phase_min"] == 0
    assert metrics["train_reset/all/phase_mean"] == pytest.approx(2.5)
    assert metrics["train_reset/all/phase_p50"] == 1
    assert metrics["train_reset/all/phase_p95"] == 7
    assert metrics["train_reset/all/phase_max"] == 7
    assert metrics["train_reset/all/bin_0_count"] == 4
    assert metrics["train_reset/all/bin_0_fraction"] == pytest.approx(4 / 6)
    assert metrics["train_reset/all/bin_1_count"] == 2
    assert metrics["train_reset/all/bin_1_fraction"] == pytest.approx(2 / 6)

    assert metrics["train_reset/phase0/count"] == 2
    assert metrics["train_reset/phase0/start_fraction"] == 1
    assert metrics["train_reset/phase0/phase_mean"] == 0
    assert metrics["train_reset/phase0/bin_0_count"] == 2
    assert metrics["train_reset/phase0/bin_1_count"] == 0

    assert metrics["train_reset/curriculum/count"] == 4
    assert metrics["train_reset/curriculum/start_fraction"] == 0
    assert metrics["train_reset/curriculum/phase_mean"] == pytest.approx(3.75)
    assert metrics["train_reset/curriculum/phase_p50"] == 3
    assert metrics["train_reset/curriculum/phase_p95"] == 7
    assert metrics["train_reset/curriculum/bin_0_count"] == 2
    assert metrics["train_reset/curriculum/bin_1_count"] == 2


def test_reset_recorder_uses_empty_sentinels_for_all_groups() -> None:
    recorder = ResetPhaseRecorder(
        8,
        start_phase=0,
        device="cpu",
    )
    recorder.begin()
    all_empty = recorder.finish(_TwoBinSampler())

    assert all_empty["train_reset/all/count"] == 0
    assert all_empty["train_reset/all/start_fraction"] == 0
    assert all_empty["train_reset/all/bin_0_fraction"] == 0
    for name in ("min", "mean", "p50", "p95", "max"):
        assert all_empty[f"train_reset/all/phase_{name}"] == -1

    recorder.begin()
    recorder.record(
        torch.tensor([0, 0]),
        torch.tensor([PHASE0_STREAM, PHASE0_STREAM]),
    )
    missing_curriculum = recorder.finish(_TwoBinSampler())

    assert missing_curriculum["train_reset/curriculum/count"] == 0
    assert missing_curriculum["train_reset/curriculum/start_fraction"] == 0
    assert missing_curriculum["train_reset/curriculum/bin_0_count"] == 0
    for name in ("min", "mean", "p50", "p95", "max"):
        assert missing_curriculum[
            f"train_reset/curriculum/phase_{name}"
        ] == -1
