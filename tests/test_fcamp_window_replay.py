from __future__ import annotations

import copy

import pytest
import torch

from components.replay.fcamp_window_buffer import (
    FCAMPReplayStateError,
    FCAMPWindowReplay,
)


def _windows(values: list[float], *, history_len: int = 2, frame_dim: int = 3) -> torch.Tensor:
    base = torch.tensor(values, dtype=torch.float32).view(-1, 1, 1)
    time = torch.arange(history_len, dtype=torch.float32).view(1, -1, 1) * 0.01
    field = torch.arange(frame_dim, dtype=torch.float32).view(1, 1, -1) * 0.001
    return base + time + field


def _offer(
    replay: FCAMPWindowReplay,
    values: list[float],
    *,
    stream_id: int,
    end_times: list[int] | None = None,
    dirty: list[bool] | None = None,
) -> None:
    if end_times is None:
        end_times = [int(value) for value in values]
    if dirty is None:
        dirty = [False] * len(values)
    replay.offer(
        _windows(values),
        end_times=torch.tensor(end_times, dtype=torch.long),
        stream_id=stream_id,
        dirty=torch.tensor(dirty, dtype=torch.bool),
    )


def test_fill_stores_complete_raw_float32_windows_and_partition_metadata() -> None:
    replay = FCAMPWindowReplay({0: 3, 1: 2}, 2, 3)
    replay.begin_update(0, replacement_quotas={0: 1, 1: 1})
    source = _windows([10.0, 20.0]).to(dtype=torch.float64)
    replay.offer(
        source,
        end_times=torch.tensor([101, 102], dtype=torch.int32),
        stream_id=0,
        dirty=torch.zeros(2, dtype=torch.bool),
    )
    _offer(replay, [90.0], stream_id=1, end_times=[909])
    replay.commit_update()
    state = replay.state_dict()
    stored = state["partitions"][0]
    assert stored["data"].dtype == torch.float32
    assert stored["data"].shape == (2, 2, 3)
    torch.testing.assert_close(stored["data"], source.float())
    assert stored["end_time"].tolist() == [101, 102]
    assert stored["insert_update"].tolist() == [0, 0]
    assert replay.size(0) == 2
    assert len(replay) == 3


def test_streaming_reservoir_is_batching_invariant_and_not_prefix_biased() -> None:
    one_batch = FCAMPWindowReplay({0: 5}, 2, 3)
    split_batch = FCAMPWindowReplay({0: 5}, 2, 3)
    values = [float(value) for value in range(20)]

    one_batch.begin_update(
        0,
        replacement_quotas={0: 2},
        generator=torch.Generator().manual_seed(1234),
    )
    _offer(one_batch, values, stream_id=0)
    one_batch.commit_update()

    split_batch.begin_update(
        0,
        replacement_quotas={0: 2},
        generator=torch.Generator().manual_seed(1234),
    )
    _offer(split_batch, values[:7], stream_id=0)
    _offer(split_batch, values[7:13], stream_id=0)
    _offer(split_batch, values[13:], stream_id=0)
    split_batch.commit_update()

    first = one_batch.state_dict()["partitions"][0]
    second = split_batch.state_dict()["partitions"][0]
    torch.testing.assert_close(first["data"], second["data"])
    torch.testing.assert_close(first["end_time"], second["end_time"])
    # The reservoir saw the whole update: it is neither the first nor last five.
    selected = set(first["end_time"].tolist())
    assert selected != set(range(5))
    assert selected != set(range(15, 20))
    assert first["offered_count"] == 20
    assert first["inserted_count"] == 5
    assert first["dropped_count"] == 15


def test_full_partition_replaces_exact_quota_with_unique_uniform_victims() -> None:
    replay = FCAMPWindowReplay({0: 4}, 2, 3)
    replay.begin_update(2, replacement_quotas={0: 2})
    _offer(replay, [0.0, 1.0, 2.0, 3.0], stream_id=0)
    replay.commit_update()

    before = replay.state_dict()["partitions"][0]["data"].clone()
    replay.begin_update(
        5,
        replacement_quotas={0: 2},
        generator=torch.Generator().manual_seed(77),
    )
    _offer(replay, [100.0, 101.0, 102.0], stream_id=0)
    _offer(replay, [103.0, 104.0, 105.0, 106.0], stream_id=0)
    replay.commit_update()

    state = replay.state_dict()["partitions"][0]
    changed = (state["data"][:, 0, 0] != before[:, 0, 0]).nonzero().flatten()
    assert changed.numel() == 2
    assert bool((state["data"][changed, 0, 0] >= 100.0).all())
    assert state["insert_update"].tolist().count(5) == 2
    assert state["offered_count"] == 11
    assert state["inserted_count"] == 4
    assert state["replaced_count"] == 2
    assert state["dropped_count"] == 5

    metrics = replay.statistics(current_update=7)
    assert metrics["replay/replacement_count"] == 2.0
    assert metrics["replay/residence_update_age_mean"] == pytest.approx(3.5)
    assert metrics["replay/residence_update_age_max"] == 5.0
    assert metrics["replay/residence_update_age_p50"] == pytest.approx(3.5)
    assert metrics["replay/stream_0_residence_update_age_mean"] == pytest.approx(3.5)


def test_sampling_is_stream_local_and_preserves_endpoint_alignment() -> None:
    replay = FCAMPWindowReplay({0: 3, 1: 4}, 2, 3)
    replay.begin_update(0, replacement_quotas={0: 1, 1: 2})
    _offer(replay, [10.0, 11.0, 12.0], stream_id=0, end_times=[110, 111, 112])
    _offer(
        replay,
        [100.0, 101.0, 102.0, 103.0],
        stream_id=1,
        end_times=[210, 211, 212, 213],
    )
    replay.commit_update()

    windows0, ends0 = replay.sample(
        30,
        stream_id=0,
        generator=torch.Generator().manual_seed(9),
    )
    windows1, ends1 = replay.sample(
        40,
        stream_id=1,
        generator=torch.Generator().manual_seed(10),
    )
    assert windows0.shape == (30, 2, 3)
    assert windows1.shape == (40, 2, 3)
    torch.testing.assert_close(windows0[:, 0, 0] + 100, ends0.float())
    torch.testing.assert_close(windows1[:, 0, 0] + 110, ends1.float())
    assert bool((windows0[:, 0, 0] < 20).all())
    assert bool((windows1[:, 0, 0] >= 100).all())

    unique, unique_ends = replay.sample(
        3,
        stream_id=0,
        replacement=False,
        generator=torch.Generator().manual_seed(11),
    )
    assert torch.unique(unique[:, 0, 0]).numel() == 3
    torch.testing.assert_close(unique[:, 0, 0] + 100, unique_ends.float())


def test_dirty_offer_poison_is_transactional_and_never_becomes_live_data() -> None:
    replay = FCAMPWindowReplay({0: 3}, 2, 3)
    replay.begin_update(0, replacement_quotas={0: 1})
    _offer(replay, [1.0], stream_id=0)
    with pytest.raises(ValueError, match="dirty FCAMP windows"):
        _offer(
            replay,
            [2.0, 3.0],
            stream_id=0,
            dirty=[False, True],
        )
    assert replay.size(0) == 0
    with pytest.raises(RuntimeError, match="poisoned"):
        replay.commit_update()
    metrics = replay.statistics()
    assert metrics["replay/dirty_insert_count"] == 0.0
    assert metrics["replay/dirty_rejected_count"] == 1.0
    assert metrics["replay/total_offered"] == 0.0
    replay.abort_update()

    # The same update number can be retried because the failed transaction did
    # not advance the committed update clock.
    replay.begin_update(0, replacement_quotas={0: 1})
    _offer(replay, [8.0], stream_id=0)
    replay.commit_update()
    assert replay.size(0) == 1
    assert replay.state_dict()["partitions"][0]["data"][0, 0, 0] == 8.0
    assert replay.statistics()["replay/dirty_insert_count"] == 0.0


def test_exact_replacement_quota_refuses_short_current_stream() -> None:
    replay = FCAMPWindowReplay({0: 2}, 2, 3)
    replay.begin_update(0, replacement_quotas={0: 2})
    _offer(replay, [1.0, 2.0], stream_id=0)
    replay.commit_update()
    before = copy.deepcopy(replay.state_dict())

    replay.begin_update(1, replacement_quotas={0: 2})
    _offer(replay, [20.0], stream_id=0)
    with pytest.raises(RuntimeError, match="exact replacement quota 2"):
        replay.commit_update()
    replay.abort_update()
    after = replay.state_dict()
    torch.testing.assert_close(
        before["partitions"][0]["data"], after["partitions"][0]["data"]
    )
    assert after["partitions"][0]["replaced_count"] == 0
    assert after["latest_update"] == 0


def test_state_roundtrip_is_exact_and_schema_is_strict() -> None:
    replay = FCAMPWindowReplay({0: 2, 1: 2}, 2, 3)
    replay.begin_update(0, replacement_quotas={0: 1, 1: 1})
    _offer(replay, [1.0, 2.0], stream_id=0, end_times=[10, 20])
    _offer(replay, [3.0], stream_id=1, end_times=[30])
    replay.commit_update()
    replay.begin_update(1, replacement_quotas={0: 1, 1: 1})
    _offer(replay, [9.0, 10.0], stream_id=0, end_times=[90, 100])
    _offer(replay, [4.0], stream_id=1, end_times=[40])
    replay.commit_update()
    state = replay.state_dict()

    restored = FCAMPWindowReplay({0: 2, 1: 2}, 2, 3)
    assert restored.load_state_dict(state)
    restored_state = restored.state_dict()
    assert restored_state.keys() == state.keys()
    assert restored_state["latest_update"] == 1
    for stream_id in (0, 1):
        for key in ("data", "end_time", "insert_update"):
            torch.testing.assert_close(
                restored_state["partitions"][stream_id][key],
                state["partitions"][stream_id][key],
            )
        assert restored_state["partitions"][stream_id]["offered_count"] == state[
            "partitions"
        ][stream_id]["offered_count"]

    legacy = copy.deepcopy(state)
    legacy.pop("schema_version")
    with pytest.raises(FCAMPReplayStateError, match="top-level"):
        restored.load_state_dict(legacy)

    dirty = copy.deepcopy(state)
    dirty["partitions"][0]["dirty_insert_count"] = 1
    with pytest.raises(FCAMPReplayStateError, match="dirty replay"):
        restored.load_state_dict(dirty)

    bad_counters = copy.deepcopy(state)
    bad_counters["partitions"][0]["offered_count"] += 1
    with pytest.raises(FCAMPReplayStateError, match="do not conserve"):
        restored.load_state_dict(bad_counters)

    wrong_shape = FCAMPWindowReplay({0: 2, 1: 2}, 3, 3)
    with pytest.raises(FCAMPReplayStateError, match="shape schema"):
        wrong_shape.load_state_dict(state)


def test_invalid_offer_poison_and_checkpoint_during_transaction_are_rejected() -> None:
    replay = FCAMPWindowReplay({0: 2}, 2, 3)
    replay.begin_update(0, replacement_quotas={0: 1})
    with pytest.raises(ValueError, match="non-finite"):
        replay.offer(
            torch.full((1, 2, 3), float("nan")),
            end_times=torch.tensor([1]),
            stream_id=0,
            dirty=torch.tensor([False]),
        )
    with pytest.raises(RuntimeError, match="active replay update"):
        replay.state_dict()
    replay.abort_update()

    replay.begin_update(0, replacement_quotas={0: 1})
    with pytest.raises(TypeError, match="dirty"):
        replay.offer(
            torch.zeros(1, 2, 3),
            end_times=torch.tensor([1]),
            stream_id=0,
            dirty=torch.tensor([0]),
        )
    replay.abort_update()


def test_update_and_stream_contracts_are_explicit() -> None:
    replay = FCAMPWindowReplay({0: 2, 1: 3}, 2, 3)
    with pytest.raises(ValueError, match="exactly the configured streams"):
        replay.begin_update(0, replacement_quotas={0: 1})
    with pytest.raises(ValueError, match="must be in"):
        replay.begin_update(0, replacement_quotas={0: 3, 1: 1})

    replay.begin_update(0, replacement_quotas={0: 1, 1: 1})
    _offer(replay, [1.0], stream_id=0)
    replay.commit_update()
    with pytest.raises(ValueError, match="advance monotonically"):
        replay.begin_update(0, replacement_quotas={0: 1, 1: 1})
    with pytest.raises(KeyError, match="unknown replay stream"):
        replay.sample(1, stream_id=99)
