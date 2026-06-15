from __future__ import annotations

import torch

from engine.grpo_advantage import absorbing_reward_to_go


def test_later_death_never_scores_lower_under_identical_prefix() -> None:
    # Two branches with the SAME non-negative per-chunk score prefix; one dies at chunk 3, the
    # other survives to chunk 6. The longer-surviving branch's trajectory return (RTG[...,0])
    # must be >= the earlier-dying branch's. Death cannot raise the return.
    gamma = 0.9
    scores = torch.tensor([[[0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]]]).repeat(1, 2, 1)
    valid = torch.ones(1, 2, 8, dtype=torch.bool)
    valid[0, 0, 3:] = False  # branch 0 dies after chunk 2 (chunks 3..7 absorbing zero)
    rtg = absorbing_reward_to_go(scores, valid, chunk_gamma=gamma)
    early_return = rtg[0, 0, 0]
    late_return = rtg[0, 1, 0]
    assert float(late_return.item()) >= float(early_return.item())


def test_death_is_monotone_in_survival_length() -> None:
    # Sweep death chunk from 1..8; with constant non-negative score the trajectory return must
    # be non-decreasing in how long the branch survives.
    gamma = 0.95
    n = 8
    scores = torch.full((1, n, n), 0.3)
    valid = torch.ones(1, n, n, dtype=torch.bool)
    for b in range(n):
        valid[0, b, b + 1 :] = False  # branch b survives b+1 chunks
    rtg = absorbing_reward_to_go(scores, valid, chunk_gamma=gamma)
    returns = rtg[0, :, 0]
    diffs = returns[1:] - returns[:-1]
    assert torch.all(diffs >= -1e-6), returns.tolist()


def test_dead_chunks_carry_zero_rtg() -> None:
    gamma = 0.9
    scores = torch.full((1, 1, 5), 0.5)
    valid = torch.ones(1, 1, 5, dtype=torch.bool)
    valid[0, 0, 2:] = False
    rtg = absorbing_reward_to_go(scores, valid, chunk_gamma=gamma)
    # chunks 2,3,4 are absorbing => zero RTG.
    assert torch.allclose(rtg[0, 0, 2:], torch.zeros(3), atol=1e-7)
    # alive chunks accumulate only their own live scores: chunk1 = 0.5, chunk0 = 0.5 + g*0.5.
    assert abs(float(rtg[0, 0, 1].item()) - 0.5) < 1e-6
    assert abs(float(rtg[0, 0, 0].item()) - (0.5 + gamma * 0.5)) < 1e-6


def test_full_survival_return_is_geometric_sum() -> None:
    gamma = 0.9
    score = 0.4
    n = 6
    scores = torch.full((1, 1, n), score)
    valid = torch.ones(1, 1, n, dtype=torch.bool)
    rtg = absorbing_reward_to_go(scores, valid, chunk_gamma=gamma)
    expected = sum((gamma ** t) * score for t in range(n))
    assert abs(float(rtg[0, 0, 0].item()) - expected) < 1e-5


def test_non_negative_scores_give_non_negative_return() -> None:
    torch.manual_seed(0)
    scores = torch.rand(3, 4, 10)  # all in [0,1]
    valid = torch.rand(3, 4, 10) > 0.3
    rtg = absorbing_reward_to_go(scores, valid, chunk_gamma=0.99)
    assert torch.all(rtg >= -1e-7)
