from __future__ import annotations

import pytest
import torch

from components.credit.amp_chunk_credit import (
    compute_amp_chunk_gae,
    normalize_and_clip_amp_advantages,
)


def _ordinary_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    bootstrap_mask: torch.Tensor,
    trace_mask: torch.Tensor,
    *,
    gamma: float,
    gae_lambda: float,
) -> torch.Tensor:
    deltas = rewards + gamma * bootstrap_mask * next_values - values
    result = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[0])
    for step in range(rewards.shape[0] - 1, -1, -1):
        running = (
            deltas[step]
            + gamma * gae_lambda * trace_mask[step] * running
        )
        result[step] = running
    return result


def test_h1_is_exactly_ordinary_gae() -> None:
    rewards = torch.tensor(
        [[1.0, -0.5], [0.25, 2.0], [3.0, 0.75]],
        dtype=torch.float64,
    )
    values = torch.tensor(
        [[0.2, 0.1], [0.4, -0.2], [0.3, 0.5]],
        dtype=torch.float64,
    )
    next_values = torch.tensor(
        [[0.4, -0.2], [0.3, 0.5], [0.8, 1.2]],
        dtype=torch.float64,
    )
    bootstrap = torch.tensor([[1, 1], [1, 1], [1, 0]], dtype=torch.float64)
    trace = torch.tensor([[1, 1], [1, 1], [0, 0]], dtype=torch.float64)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    gamma = 0.93
    gae_lambda = 0.81

    result = compute_amp_chunk_gae(
        rewards,
        values,
        next_values,
        torch.ones_like(rewards, dtype=torch.long),
        bootstrap,
        trace,
        valid,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    expected = _ordinary_gae(
        rewards,
        values,
        next_values,
        bootstrap,
        trace,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    torch.testing.assert_close(result.advantages, expected)
    torch.testing.assert_close(result.value_targets, values + expected)


def test_h4_full_chunk_uses_variable_duration_discounts() -> None:
    gamma = 0.9
    gae_lambda = 0.8
    primitive_rewards = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0], [0.5, 1.0, 1.5, 2.0]],
        dtype=torch.float64,
    )
    discounted_rewards = torch.stack(
        [
            sum(gamma**i * row[i] for i in range(4))
            for row in primitive_rewards
        ]
    ).unsqueeze(1)
    values = torch.tensor([[0.4], [0.7]], dtype=torch.float64)
    next_values = torch.tensor([[0.7], [1.1]], dtype=torch.float64)
    durations = torch.full((2, 1), 4, dtype=torch.long)
    bootstrap = torch.ones((2, 1), dtype=torch.bool)
    trace = torch.tensor([[True], [False]])
    valid = torch.ones((2, 1), dtype=torch.bool)

    result = compute_amp_chunk_gae(
        discounted_rewards,
        values,
        next_values,
        durations,
        bootstrap,
        trace,
        valid,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    delta1 = discounted_rewards[1, 0] + gamma**4 * 1.1 - 0.7
    delta0 = discounted_rewards[0, 0] + gamma**4 * 0.7 - 0.4
    expected1 = delta1
    expected0 = delta0 + (gamma * gae_lambda) ** 4 * expected1
    torch.testing.assert_close(
        result.advantages[:, 0],
        torch.stack((expected0, expected1)),
    )


def test_terminal_prefix_uses_executed_duration_without_bootstrap() -> None:
    gamma = 0.9
    reward0 = 2.0
    reward1 = 5.0
    discounted_reward = torch.tensor(
        [[reward0 + gamma * reward1]],
        dtype=torch.float64,
    )
    result = compute_amp_chunk_gae(
        discounted_reward,
        values=torch.tensor([[1.25]], dtype=torch.float64),
        next_values=torch.tensor([[999.0]], dtype=torch.float64),
        durations=torch.tensor([[2]]),
        bootstrap_mask=torch.tensor([[False]]),
        trace_mask=torch.tensor([[False]]),
        valid_mask=torch.tensor([[True]]),
        gamma=gamma,
        gae_lambda=0.95,
    )

    expected = reward0 + gamma * reward1 - 1.25
    assert result.advantages.item() == pytest.approx(expected)
    assert result.value_targets.item() == pytest.approx(reward0 + gamma * reward1)


def test_timeout_bootstraps_but_cuts_trace() -> None:
    gamma = 0.9
    rewards = torch.tensor([[1.0], [100.0]], dtype=torch.float64)
    result = compute_amp_chunk_gae(
        rewards,
        values=torch.tensor([[2.0], [0.0]], dtype=torch.float64),
        next_values=torch.tensor([[3.0], [0.0]], dtype=torch.float64),
        durations=torch.tensor([[3], [1]]),
        bootstrap_mask=torch.tensor([[True], [False]]),
        trace_mask=torch.tensor([[False], [False]]),
        valid_mask=torch.tensor([[True], [True]]),
        gamma=gamma,
        gae_lambda=0.95,
    )

    assert result.advantages[0, 0].item() == pytest.approx(
        1.0 + gamma**3 * 3.0 - 2.0
    )
    assert result.advantages[1, 0].item() == pytest.approx(100.0)


def test_padded_invalid_decisions_are_zero_and_cannot_carry_trace() -> None:
    result = compute_amp_chunk_gae(
        discounted_rewards=torch.tensor(
            [[1.0, 2.0], [3.0, 9.0e30], [9.0e30, 9.0e30]],
            dtype=torch.float64,
        ),
        values=torch.tensor(
            [[0.0, 0.0], [0.0, 9.0e30], [9.0e30, 9.0e30]],
            dtype=torch.float64,
        ),
        next_values=torch.tensor(
            [[0.0, 0.0], [0.0, 9.0e30], [9.0e30, 9.0e30]],
            dtype=torch.float64,
        ),
        durations=torch.tensor([[1, 1], [1, 0], [0, 0]]),
        bootstrap_mask=torch.ones((3, 2), dtype=torch.bool),
        trace_mask=torch.ones((3, 2), dtype=torch.bool),
        valid_mask=torch.tensor(
            [[True, True], [True, False], [False, False]]
        ),
        gamma=0.5,
        gae_lambda=1.0,
    )

    torch.testing.assert_close(
        result.advantages,
        torch.tensor(
            [[2.5, 2.0], [3.0, 0.0], [0.0, 0.0]],
            dtype=torch.float64,
        ),
    )
    assert bool((result.value_targets[~torch.tensor(
        [[True, True], [True, False], [False, False]]
    )] == 0).all())


def test_advantage_normalization_is_global_valid_and_clipped() -> None:
    advantages = torch.tensor(
        [[-100.0, 1.0], [2.0, 3.0], [100.0, 999.0]],
        dtype=torch.float64,
    )
    valid = torch.tensor(
        [[True, True], [True, True], [True, False]]
    )
    result = normalize_and_clip_amp_advantages(
        advantages,
        valid,
        clip=1.0,
        epsilon=1.0e-12,
    )

    selected = advantages[valid]
    expected = (advantages - selected.mean()) / selected.std(
        unbiased=True
    ).clamp_min(1.0e-12)
    expected = torch.where(valid, expected.clamp(-1.0, 1.0), 0.0)
    torch.testing.assert_close(result, expected)
    assert result[2, 1].item() == 0.0


def test_single_valid_advantage_is_a_safe_actor_noop() -> None:
    result = normalize_and_clip_amp_advantages(
        torch.tensor([[7.0, 999.0]]),
        torch.tensor([[True, False]]),
    )

    torch.testing.assert_close(result, torch.zeros_like(result))
