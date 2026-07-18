from __future__ import annotations

import pytest
import torch

from components.replay.frame_trajectory import FrameTrajectoryReplay


def _push_step(
    replay: FrameTrajectoryReplay,
    *,
    frames: list[float],
    env_ids: list[int],
    time: int,
    streams: int | list[int],
) -> None:
    stream: int | torch.Tensor
    if isinstance(streams, int):
        stream = streams
    else:
        stream = torch.tensor(streams, dtype=torch.long)
    replay.push_frames(
        torch.tensor(frames, dtype=torch.float32).unsqueeze(1),
        env_ids=torch.tensor(env_ids, dtype=torch.long),
        episode_ids=torch.zeros(len(env_ids), dtype=torch.long),
        reference_times=torch.full((len(env_ids),), time, dtype=torch.long),
        ages=torch.full((len(env_ids),), time, dtype=torch.long),
        update=1,
        stream=stream,
    )


def test_stream_filtered_sampling_and_statistics() -> None:
    replay = FrameTrajectoryReplay(32, 1, num_envs=2, pin_memory=False)
    for time in range(1, 5):
        _push_step(
            replay,
            frames=[10.0 + time, 100.0 + time],
            env_ids=[0, 1],
            time=time,
            streams=[0, 1],
        )

    stream0, ends0 = replay.sample_windows(
        8,
        3,
        stream_id=0,
        generator=torch.Generator().manual_seed(1),
    )
    stream1, ends1 = replay.sample_windows(
        8,
        3,
        stream_id=1,
        generator=torch.Generator().manual_seed(2),
    )

    assert stream0.shape == (8, 3, 1)
    assert stream1.shape == (8, 3, 1)
    torch.testing.assert_close(stream0[:, -1, 0], ends0.float() + 10.0)
    torch.testing.assert_close(stream1[:, -1, 0], ends1.float() + 100.0)
    assert bool((stream0 < 100.0).all())
    assert bool((stream1 >= 100.0).all())

    stats = replay.statistics()
    assert stats["replay/stream_0_size"] == 4.0
    assert stats["replay/stream_1_size"] == 4.0
    assert stats["replay/stream_0_eligible_endpoints"] == 4.0
    assert stats["replay/stream_1_eligible_endpoints"] == 4.0


def test_stream_change_breaks_predecessor_chain() -> None:
    replay = FrameTrajectoryReplay(16, 1, num_envs=1, pin_memory=False)
    _push_step(replay, frames=[1.0], env_ids=[0], time=1, streams=0)
    _push_step(replay, frames=[2.0], env_ids=[0], time=2, streams=0)
    _push_step(replay, frames=[3.0], env_ids=[0], time=3, streams=1)
    _push_step(replay, frames=[4.0], env_ids=[0], time=4, streams=1)
    _push_step(replay, frames=[5.0], env_ids=[0], time=5, streams=1)

    assert replay._predecessor[2].item() == -1
    torch.testing.assert_close(replay._chain_indices(1, 2), torch.tensor([0, 1]))
    torch.testing.assert_close(replay._chain_indices(4, 3), torch.tensor([2, 3, 4]))

    # Chain validation is defensive even if a serialized predecessor is corrupt.
    replay._predecessor[2] = 1
    assert replay._chain_indices(2, 2) is None
    chains, valid = replay._chain_indices_batch(torch.tensor([2]), 2)
    assert chains.shape == (0, 2)
    assert valid.tolist() == [False]


def test_stream_state_roundtrip_and_legacy_checkpoint_default() -> None:
    replay = FrameTrajectoryReplay(16, 1, num_envs=2, pin_memory=False)
    for time in range(1, 4):
        _push_step(
            replay,
            frames=[float(time), float(10 + time)],
            env_ids=[0, 1],
            time=time,
            streams=[0, 1],
        )
    state = replay.state_dict()
    assert state["stream"].dtype == torch.int8

    restored = FrameTrajectoryReplay(16, 1, num_envs=2, pin_memory=False)
    assert restored.load_state_dict(state)
    torch.testing.assert_close(restored._stream, replay._stream)
    windows, _ = restored.sample_windows(4, 2, stream_id=1)
    assert windows.shape == (4, 2, 1)
    assert bool((windows >= 10.0).all())

    legacy_state = dict(state)
    legacy_state.pop("stream")
    legacy = FrameTrajectoryReplay(16, 1, num_envs=2, pin_memory=False)
    assert legacy.load_state_dict(legacy_state)
    assert bool((legacy._stream[legacy._valid] == 0).all())
    assert bool((legacy._stream[~legacy._valid] == -1).all())

    legacy.clear()
    assert not bool(legacy._valid.any())
    assert bool((legacy._stream == -1).all())


def test_stream_tensor_is_trimmed_with_oversized_push_and_validated() -> None:
    replay = FrameTrajectoryReplay(3, 1, num_envs=5, pin_memory=False)
    replay.push_frames(
        torch.arange(5, dtype=torch.float32).unsqueeze(1),
        env_ids=torch.arange(5),
        episode_ids=torch.zeros(5, dtype=torch.long),
        reference_times=torch.ones(5, dtype=torch.long),
        ages=torch.ones(5, dtype=torch.long),
        update=1,
        stream=torch.tensor([0, 1, 0, 1, 0]),
    )
    torch.testing.assert_close(replay._frames[:, 0], torch.tensor([2.0, 3.0, 4.0]))
    torch.testing.assert_close(replay._stream, torch.tensor([0, 1, 0], dtype=torch.int8))

    with pytest.raises(ValueError, match="shape"):
        _push_step(replay, frames=[1.0], env_ids=[0], time=2, streams=[0, 1])
    with pytest.raises(TypeError, match="integer dtype"):
        replay.push_frames(
            torch.ones(1, 1),
            env_ids=torch.tensor([0]),
            episode_ids=torch.tensor([0]),
            reference_times=torch.tensor([2]),
            ages=torch.tensor([2]),
            update=2,
            stream=torch.tensor([0.0]),
        )
    with pytest.raises(ValueError, match="int8"):
        _push_step(replay, frames=[1.0], env_ids=[0], time=2, streams=128)
