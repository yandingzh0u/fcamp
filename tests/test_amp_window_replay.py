from __future__ import annotations

import torch

from components.replay.amp_window_buffer import AMPWindowReplay


def _windows(count: int) -> torch.Tensor:
    values = torch.arange(count, dtype=torch.float32)
    return values[:, None, None].expand(-1, 2, 1).clone()


def test_replay_uses_mimickit_capacity_permutation_stream() -> None:
    replay = AMPWindowReplay(capacity=8, history_len=2, frame_dim=1)
    replay.push(
        _windows(5),
        end_times=torch.arange(5, dtype=torch.float32) + 0.25,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(17)
    expected_generator = torch.Generator(device="cpu")
    expected_generator.manual_seed(17)

    order0 = torch.randperm(8, generator=expected_generator)
    expected0 = torch.remainder(order0[:4], 5)
    sampled0, endpoints0 = replay.sample(4, generator=generator)
    torch.testing.assert_close(sampled0[:, 0, 0], expected0.float())
    torch.testing.assert_close(
        endpoints0,
        expected0.float() + 0.25,
    )

    order1 = torch.randperm(8, generator=expected_generator)
    expected1 = torch.remainder(
        torch.cat((order0[4:], order1[:1])),
        5,
    )
    sampled1, _ = replay.sample(5, generator=generator)
    torch.testing.assert_close(sampled1[:, 0, 0], expected1.float())


def test_full_replay_samples_every_slot_before_repeating() -> None:
    replay = AMPWindowReplay(capacity=8, history_len=2, frame_dim=1)
    replay.push(
        _windows(8),
        end_times=torch.arange(8, dtype=torch.float32) + 0.5,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(3)

    sampled, endpoints = replay.sample(8, generator=generator)

    assert torch.equal(
        torch.sort(sampled[:, 0, 0]).values,
        torch.arange(8, dtype=torch.float32),
    )
    assert torch.equal(
        torch.sort(endpoints).values,
        torch.arange(8, dtype=torch.float32) + 0.5,
    )


def test_replay_checkpoint_preserves_sampling_stream_and_fractional_metadata() -> None:
    replay = AMPWindowReplay(capacity=8, history_len=2, frame_dim=1)
    replay.push(
        _windows(6),
        end_times=torch.arange(6, dtype=torch.float32) + 0.125,
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(29)
    replay.sample(3, generator=generator)

    restored = AMPWindowReplay(capacity=8, history_len=2, frame_dim=1)
    restored.load_state_dict(replay.state_dict())
    restored_generator = torch.Generator(device="cpu")
    restored_generator.set_state(generator.get_state())

    actual_windows, actual_endpoints = replay.sample(
        6,
        generator=generator,
    )
    restored_windows, restored_endpoints = restored.sample(
        6,
        generator=restored_generator,
    )
    torch.testing.assert_close(restored_windows, actual_windows)
    torch.testing.assert_close(restored_endpoints, actual_endpoints)

