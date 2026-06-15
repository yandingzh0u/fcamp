from __future__ import annotations

import torch


def group_relative_advantages(returns: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Group-relative (GRPO) advantage: per-group standardize returns across the branch axis.

    Args:
      returns:    (num_groups, num_branches) scalar return per branch.
      valid_mask: (num_groups, num_branches) bool, which branches are valid samples.

    Returns:
      advantages: (num_groups, num_branches), zero for invalid branches.

    Each group's mean/std are computed over its VALID branches only (the standard GRPO control
    variate). std uses the unbiased (n-1) denominator. When a group has <=1 valid branch the
    advantage is 0 (no usable group baseline / spread).
    """
    if returns.ndim != 2:
        raise ValueError(f"returns must have shape (groups, branches), got {tuple(returns.shape)}")
    if valid_mask.shape != returns.shape:
        raise ValueError("valid_mask must match returns")
    returns_float = returns.to(torch.float32)
    valid_float = valid_mask.to(dtype=torch.float32)
    counts = valid_float.sum(dim=1, keepdim=True).clamp(min=1.0)
    means = (returns_float * valid_float).sum(dim=1, keepdim=True) / counts
    centered = (returns_float - means) * valid_float
    denom = (counts - 1.0).clamp(min=1.0)
    variances = centered.square().sum(dim=1, keepdim=True) / denom
    stds = torch.sqrt(variances + 1e-8)
    advantages = (returns_float - means) / stds
    return torch.where(valid_mask, advantages.to(dtype=returns.dtype), torch.zeros_like(returns))


def absorbing_reward_to_go(
    chunk_scores: torch.Tensor,
    chunk_valid: torch.Tensor,
    *,
    chunk_gamma: float,
) -> torch.Tensor:
    """Fixed-length absorbing-state discounted reward-to-go.

    Args:
      chunk_scores: (num_groups, num_branches, num_chunks) per-chunk non-negative score.
      chunk_valid:  (num_groups, num_branches, num_chunks) bool, branch alive at chunk.
      chunk_gamma:  per-chunk discount.

    Returns:
      reward_to_go: (num_groups, num_branches, num_chunks). RTG[...,0] is the full trajectory
      return. After death a branch is an absorbing zero-score state (valid is 0 there), so a
      branch that dies earlier can only have a LOWER discounted return, never higher — death
      can never increase the return by truncating anything (all scores are non-negative).
    """
    if chunk_scores.shape != chunk_valid.shape or chunk_scores.ndim != 3:
        raise ValueError("chunk_scores and chunk_valid must be matching (groups, branches, chunks) tensors")
    valid_float = chunk_valid.to(dtype=chunk_scores.dtype)
    first_life = chunk_scores * valid_float
    num_chunks = chunk_scores.shape[-1]
    reward_to_go = torch.zeros_like(first_life)
    running = torch.zeros_like(first_life[:, :, 0])
    for chunk_idx in range(num_chunks - 1, -1, -1):
        running = first_life[:, :, chunk_idx] + chunk_gamma * running
        reward_to_go[:, :, chunk_idx] = running
        running = running * valid_float[:, :, chunk_idx]
    return reward_to_go


def trajectory_advantages_per_chunk(
    trajectory_return: torch.Tensor,
    trajectory_valid: torch.Tensor,
    chunk_valid: torch.Tensor,
) -> torch.Tensor:
    """Trajectory-level GRPO advantage broadcast to every chunk (original GRPO contract).

    One sample == one 120-frame branch trajectory. Its full discounted return is normalized
    ONCE across the group's branches, and the resulting scalar advantage is broadcast to every
    chunk of that trajectory (then masked to chunks the branch was alive for). This matches the
    original "one sample, one normalized advantage broadcast to the whole sample" GRPO contract
    and avoids per-chunk re-normalization (which arbitrarily re-weights time segments and
    collapses late-chunk advantage to 0 once only one branch survives).

    Args:
      trajectory_return: (num_groups, num_branches) full discounted return per branch.
      trajectory_valid:  (num_groups, num_branches) bool, branch is a valid trajectory sample.
      chunk_valid:       (num_groups, num_branches, num_chunks) bool, branch alive at chunk.

    Returns:
      advantages_per_chunk: (num_groups, num_branches, num_chunks), zero on invalid chunks.
    """
    if chunk_valid.ndim != 3:
        raise ValueError(f"chunk_valid must be (groups, branches, chunks), got {tuple(chunk_valid.shape)}")
    trajectory_advantage = group_relative_advantages(trajectory_return, trajectory_valid)
    num_chunks = chunk_valid.shape[-1]
    advantages = trajectory_advantage.unsqueeze(-1).expand(-1, -1, num_chunks)
    return advantages * chunk_valid.to(dtype=advantages.dtype)
