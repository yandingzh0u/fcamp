from __future__ import annotations

import torch

from components.credit.temporal_credit import (
    compute_dual_channel_gae,
    normalize_actor_mixture,
    with_chunk_shared_actor_credit,
)


def _compute(
    rewards: torch.Tensor,
    *,
    values: torch.Tensor | None = None,
    next_values: torch.Tensor | None = None,
    channel_valid: torch.Tensor | None = None,
    channel_bootstrap: torch.Tensor | None = None,
    channel_trace: torch.Tensor | None = None,
):
    time_steps, num_envs, _ = rewards.shape
    shape = (time_steps, num_envs)
    values = torch.zeros_like(rewards) if values is None else values
    next_values = torch.zeros_like(rewards) if next_values is None else next_values
    return compute_dual_channel_gae(
        rewards,
        values,
        next_values,
        torch.ones(shape),
        torch.ones(shape),
        torch.ones(shape, dtype=torch.bool),
        gamma=1.0,
        gae_lambda=1.0,
        chunk_horizon=2,
        normalization="none",
        actor_weights=(1.0, 1.0),
        channel_valid_mask=channel_valid,
        channel_bootstrap_mask=channel_bootstrap,
        channel_trace_mask=channel_trace,
    )


def test_amp_push_edge_dirty_gap_and_recovery_form_separate_segments() -> None:
    # t=0 is followed by an external push.  For W=4, endpoints t=1..3 are
    # contaminated; t=4 is the first clean window and starts a new AMP segment.
    rewards = torch.tensor(
        [
            [[1.0, 1.0]],
            [[1.0, 1000.0]],
            [[1.0, 1000.0]],
            [[1.0, 1000.0]],
            [[1.0, 5.0]],
            [[1.0, 7.0]],
        ]
    )
    channel_valid = torch.ones_like(rewards, dtype=torch.bool)
    channel_valid[1:4, :, 1] = False
    channel_bootstrap = torch.ones_like(rewards)
    channel_trace = torch.ones_like(rewards)
    channel_bootstrap[0, :, 1] = 0.0
    channel_trace[0, :, 1] = 0.0
    values = torch.zeros_like(rewards)
    values[1:4, :, 1] = 123.0

    credit = _compute(
        rewards,
        values=values,
        channel_valid=channel_valid,
        channel_bootstrap=channel_bootstrap,
        channel_trace=channel_trace,
    )

    # Task credit is unaffected and crosses the entire rollout.
    torch.testing.assert_close(
        credit.advantages[:, 0, 0], torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
    )
    # AMP neither crosses the push edge nor consumes the W-1 dirty endpoints.
    torch.testing.assert_close(
        credit.advantages[:, 0, 1], torch.tensor([1.0, 0.0, 0.0, 0.0, 12.0, 7.0])
    )
    torch.testing.assert_close(credit.value_targets[1:4, 0, 1], torch.zeros(3))
    torch.testing.assert_close(
        credit.actor_advantage_components[1:4, 0, 1], torch.zeros(3)
    )
    assert credit.channel_valid_mask is not None
    torch.testing.assert_close(credit.channel_valid_mask, channel_valid)


def test_amp_push_edge_cuts_channel_bootstrap_without_cutting_task() -> None:
    rewards = torch.zeros(1, 1, 2)
    values = torch.zeros_like(rewards)
    next_values = torch.tensor([[[3.0, 5.0]]])
    channel_bootstrap = torch.ones_like(rewards)
    channel_bootstrap[..., 1] = 0.0
    result = compute_dual_channel_gae(
        rewards,
        values,
        next_values,
        torch.ones(1, 1),
        torch.ones(1, 1),
        torch.ones(1, 1, dtype=torch.bool),
        gamma=0.5,
        gae_lambda=1.0,
        chunk_horizon=1,
        normalization="none",
        channel_bootstrap_mask=channel_bootstrap,
    )
    torch.testing.assert_close(result.advantages[0, 0], torch.tensor([1.5, 0.0]))


def test_weighted_normalizer_excludes_invalid_amp_samples() -> None:
    rewards = torch.tensor(
        [[[1.0, 2.0]], [[3.0, 1000.0]], [[5.0, 8.0]], [[7.0, 10.0]]]
    )
    channel_valid = torch.ones_like(rewards, dtype=torch.bool)
    channel_valid[1, :, 1] = False
    raw = _compute(rewards, channel_valid=channel_valid)
    valid = torch.ones(4, 1, dtype=torch.bool)
    normalized = normalize_actor_mixture(
        raw,
        valid,
        torch.full((4, 1), 0.25),
    )

    # Invalid AMP is absent, not treated as an ordinary zero and then centered
    # into a non-zero negative AMP component.
    assert normalized.actor_advantage_components[1, 0, 1].item() == 0.0
    assert normalized.actor_advantage_components[1, 0, 0].item() != 0.0
    torch.testing.assert_close(
        normalized.actor_advantage,
        normalized.actor_advantage_components.sum(dim=-1),
    )


def test_chunk_shared_masks_invalid_amp_at_the_receiving_offset() -> None:
    rewards = torch.tensor(
        [[[1.0, 10.0]], [[2.0, 20.0]], [[3.0, 30.0]], [[4.0, 40.0]]]
    )
    channel_valid = torch.ones_like(rewards, dtype=torch.bool)
    channel_valid[1, :, 1] = False
    primitive = _compute(rewards, channel_valid=channel_valid)
    valid = torch.ones(4, 1, dtype=torch.bool)
    shared = with_chunk_shared_actor_credit(
        primitive,
        valid,
        chunk_horizon=2,
        normalization="none",
    )

    assert shared.actor_advantage_components[1, 0, 1].item() == 0.0
    assert shared.actor_advantage_components[1, 0, 0].item() != 0.0
    torch.testing.assert_close(
        shared.mixed_advantage,
        shared.actor_advantage_components.sum(dim=-1),
    )


def test_legacy_api_matches_explicit_shared_channel_masks() -> None:
    torch.manual_seed(11)
    rewards = torch.randn(4, 2, 2)
    base_valid = torch.tensor(
        [[True, True], [True, False], [True, True], [False, True]]
    )
    shared_bootstrap = torch.tensor(
        [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )
    shared_trace = torch.tensor(
        [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    )
    values = torch.randn_like(rewards)
    next_values = torch.randn_like(rewards)
    common = dict(
        rewards=rewards,
        values=values,
        next_values=next_values,
        bootstrap_mask=shared_bootstrap,
        trace_mask=shared_trace,
        valid_mask=base_valid,
        gamma=0.97,
        gae_lambda=0.91,
        chunk_horizon=2,
        normalization="global",
        actor_weights=(1.0, 0.25),
    )
    legacy = compute_dual_channel_gae(**common)
    explicit = compute_dual_channel_gae(
        **common,
        channel_valid_mask=base_valid.unsqueeze(-1).expand_as(rewards),
        channel_bootstrap_mask=shared_bootstrap.unsqueeze(-1).expand_as(rewards),
        channel_trace_mask=shared_trace.unsqueeze(-1).expand_as(rewards),
    )
    for name in (
        "td_errors",
        "advantages",
        "value_targets",
        "mixed_advantage",
        "actor_advantage",
        "actor_advantage_components",
        "channel_valid_mask",
    ):
        torch.testing.assert_close(getattr(legacy, name), getattr(explicit, name))
