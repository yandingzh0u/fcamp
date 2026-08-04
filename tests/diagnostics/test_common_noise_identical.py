import torch

from diagnostics.common.noise_bank import NoiseBank


def test_common_action_noise_is_identical_across_checkpoint_collections() -> None:
    bank = NoiseBank(seed=20260804)
    snapshot_ids = ["snapshot-a", "snapshot-b", "snapshot-c"]

    checkpoint_50 = bank.common_action_epsilon(snapshot_ids, horizon=16, action_dim=29)
    checkpoint_200 = bank.common_action_epsilon(snapshot_ids, horizon=16, action_dim=29)
    torch.testing.assert_close(checkpoint_50, checkpoint_200, rtol=0.0, atol=0.0)

    reordered = bank.common_action_epsilon(
        list(reversed(snapshot_ids)), horizon=16, action_dim=29
    ).flip(0)
    torch.testing.assert_close(checkpoint_50, reordered, rtol=0.0, atol=0.0)
    assert not torch.equal(checkpoint_50[0], checkpoint_50[1])


def test_environment_streams_are_reproducible_but_independent() -> None:
    bank = NoiseBank(seed=9)
    ids = ["s0", "s1"]
    reset = bank.controlled_uniform(ids, horizon=4, width=3, stream="reset", low=-1, high=1)
    repeat = bank.controlled_uniform(ids, horizon=4, width=3, stream="reset", low=-1, high=1)
    pushes = bank.controlled_uniform(ids, horizon=4, width=3, stream="push", low=-1, high=1)
    torch.testing.assert_close(reset, repeat, rtol=0.0, atol=0.0)
    assert not torch.equal(reset, pushes)
