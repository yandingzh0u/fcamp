from __future__ import annotations

import torch


def compute_gae_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    *,
    gamma: float,
    lam: float,
    timeouts: torch.Tensor | None = None,
    normalize_advantage: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape (env, steps), got {tuple(rewards.shape)}")
    if dones.shape != rewards.shape or values.shape != rewards.shape:
        raise ValueError("dones and values must match rewards shape")
    if last_values.shape != (rewards.shape[0],):
        raise ValueError(f"last_values must have shape {(rewards.shape[0],)}, got {tuple(last_values.shape)}")

    rewards_for_gae = rewards
    if timeouts is not None:
        if timeouts.shape != rewards.shape:
            raise ValueError(f"timeouts must have shape {tuple(rewards.shape)}, got {tuple(timeouts.shape)}")
        rewards_for_gae = rewards_for_gae + gamma * values * timeouts.to(dtype=rewards.dtype)

    returns = torch.empty_like(rewards)
    advantage = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
    for step_index in range(rewards.shape[1] - 1, -1, -1):
        next_values = last_values if step_index == rewards.shape[1] - 1 else values[:, step_index + 1]
        next_is_not_terminal = 1.0 - dones[:, step_index].to(dtype=rewards.dtype)
        delta = rewards_for_gae[:, step_index] + gamma * next_is_not_terminal * next_values - values[:, step_index]
        advantage = delta + gamma * lam * next_is_not_terminal * advantage
        returns[:, step_index] = advantage + values[:, step_index]

    advantages = returns - values
    if normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages
