from __future__ import annotations

import torch

from engine.grpo_advantage import group_relative_advantages, trajectory_advantages_per_chunk


def test_same_trajectory_chunks_share_one_advantage() -> None:
    # 1 group, 4 branches, 5 chunks; all branches alive for all chunks.
    trajectory_return = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    trajectory_valid = torch.ones(1, 4, dtype=torch.bool)
    chunk_valid = torch.ones(1, 4, 5, dtype=torch.bool)

    adv = trajectory_advantages_per_chunk(trajectory_return, trajectory_valid, chunk_valid)

    # Every chunk of a given branch must carry the SAME scalar advantage.
    for branch in range(4):
        per_chunk = adv[0, branch]
        assert torch.allclose(per_chunk, per_chunk[0].expand_as(per_chunk))


def test_dead_chunks_are_masked() -> None:
    trajectory_return = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    trajectory_valid = torch.ones(1, 4, dtype=torch.bool)
    # Branch 0 dies after chunk 2 (chunks 3,4 invalid); others alive throughout.
    chunk_valid = torch.ones(1, 4, 5, dtype=torch.bool)
    chunk_valid[0, 0, 3:] = False

    adv = trajectory_advantages_per_chunk(trajectory_return, trajectory_valid, chunk_valid)

    assert torch.all(adv[0, 0, 3:] == 0.0)
    # The alive chunks of branch 0 still carry its (nonzero) trajectory advantage.
    assert torch.all(adv[0, 0, :3] != 0.0)


def test_equal_returns_give_zero_advantage() -> None:
    trajectory_return = torch.full((1, 4), 3.5)
    trajectory_valid = torch.ones(1, 4, dtype=torch.bool)
    chunk_valid = torch.ones(1, 4, 5, dtype=torch.bool)

    adv = trajectory_advantages_per_chunk(trajectory_return, trajectory_valid, chunk_valid)

    assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-5)


def test_branch_permutation_invariance() -> None:
    trajectory_return = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    trajectory_valid = torch.ones(1, 4, dtype=torch.bool)
    chunk_valid = torch.ones(1, 4, 3, dtype=torch.bool)

    adv = trajectory_advantages_per_chunk(trajectory_return, trajectory_valid, chunk_valid)

    perm = torch.tensor([2, 0, 3, 1])
    adv_perm = trajectory_advantages_per_chunk(
        trajectory_return[:, perm],
        trajectory_valid[:, perm],
        chunk_valid[:, perm],
    )
    # Permuting the branch order must permute the advantages identically (no cross-branch leak).
    assert torch.allclose(adv[:, perm], adv_perm, atol=1e-6)


def test_single_surviving_branch_does_not_collapse_other_chunks() -> None:
    # Regression for the old per-chunk renormalization bug: with trajectory-level advantage,
    # a branch's late chunks keep the SAME advantage as its early chunks even if other branches
    # died earlier (their trajectory return is just lower, not their late-chunk advantage zeroed).
    trajectory_return = torch.tensor([[5.0, 1.0, 1.0, 1.0]])
    trajectory_valid = torch.ones(1, 4, dtype=torch.bool)
    chunk_valid = torch.ones(1, 4, 4, dtype=torch.bool)
    # Branches 1,2,3 die after chunk 0; branch 0 survives all 4 chunks.
    chunk_valid[0, 1:, 1:] = False

    adv = trajectory_advantages_per_chunk(trajectory_return, trajectory_valid, chunk_valid)

    # Branch 0 (the survivor) keeps a constant nonzero advantage across all its chunks.
    assert torch.all(adv[0, 0] != 0.0)
    assert torch.allclose(adv[0, 0], adv[0, 0, 0].expand(4))


def test_group_relative_advantage_unbiased_std_and_zero_mean() -> None:
    returns = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    adv = group_relative_advantages(returns, valid)
    # Zero mean across the group.
    assert abs(float(adv.mean().item())) < 1e-6
    # Uses unbiased (n-1) std: mean=2.5, var=(1.5^2+0.5^2+0.5^2+1.5^2)/3=5/3, std=sqrt(5/3).
    import math
    expected = (returns[0] - 2.5) / math.sqrt(5.0 / 3.0 + 1e-8)
    assert torch.allclose(adv[0], expected, atol=1e-4)
