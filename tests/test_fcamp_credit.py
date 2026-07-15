from __future__ import annotations

import torch

from components.credit.temporal_credit import (
    compute_dual_channel_gae,
    resolve_terminal_masks,
    with_chunk_shared_actor_credit,
)
from models.dual_flow_critic import SharedEncoderDualFlowCritic


def _credit(
    rewards: torch.Tensor,
    *,
    values: torch.Tensor | None = None,
    next_values: torch.Tensor | None = None,
    bootstrap_mask: torch.Tensor | None = None,
    trace_mask: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    gamma: float = 1.0,
    gae_lambda: float = 1.0,
    normalization: str = "none",
):
    time_steps, num_envs, _ = rewards.shape
    mask_shape = (time_steps, num_envs)
    return compute_dual_channel_gae(
        rewards,
        torch.zeros_like(rewards) if values is None else values,
        torch.zeros_like(rewards) if next_values is None else next_values,
        torch.ones(mask_shape) if bootstrap_mask is None else bootstrap_mask,
        torch.ones(mask_shape) if trace_mask is None else trace_mask,
        torch.ones(mask_shape, dtype=torch.bool) if valid_mask is None else valid_mask,
        gamma=gamma,
        gae_lambda=gae_lambda,
        chunk_horizon=2,
        normalization=normalization,
        actor_weights=(2.0, 3.0),
    )


def test_primitive_gae_crosses_chunk_boundary() -> None:
    # H=2: t=1 is the end of chunk 0 and must see rewards from chunk 1.
    rewards = torch.tensor(
        [[[1.0, 10.0]], [[2.0, 20.0]], [[3.0, 30.0]], [[4.0, 40.0]]]
    )
    result = _credit(rewards)
    expected = torch.tensor(
        [[[10.0, 100.0]], [[9.0, 90.0]], [[7.0, 70.0]], [[4.0, 40.0]]]
    )
    torch.testing.assert_close(result.advantages, expected)
    torch.testing.assert_close(result.value_targets, expected)
    torch.testing.assert_close(
        result.actor_advantage, 2.0 * expected[..., 0] + 3.0 * expected[..., 1]
    )


def test_failure_and_timeout_use_different_bootstrap_masks() -> None:
    rewards = torch.tensor([[[1.0, 2.0], [1.0, 2.0]]])
    values = torch.tensor([[[3.0, 4.0], [3.0, 4.0]]])
    next_values = torch.tensor([[[10.0, 20.0], [10.0, 20.0]]])
    # env0: failure (no bootstrap); env1: timeout (bootstrap), both cut trace.
    bootstrap = torch.tensor([[0.0, 1.0]])
    trace = torch.zeros_like(bootstrap)
    result = _credit(
        rewards,
        values=values,
        next_values=next_values,
        bootstrap_mask=bootstrap,
        trace_mask=trace,
        gamma=0.5,
    )
    torch.testing.assert_close(result.advantages[0, 0], torch.tensor([-2.0, -2.0]))
    torch.testing.assert_close(result.advantages[0, 1], torch.tensor([3.0, 8.0]))
    torch.testing.assert_close(result.value_targets[0, 0], rewards[0, 0])
    torch.testing.assert_close(
        result.value_targets[0, 1], torch.tensor([6.0, 12.0])
    )


def test_overlapping_timeout_never_overrides_tracking_failure() -> None:
    done = torch.tensor([True, True, True, False])
    failure, timeout, motion_complete = resolve_terminal_masks(
        done,
        timeout=torch.tensor([True, True, False, True]),
        motion_complete=torch.tensor([False, True, True, True]),
        failure=torch.tensor([True, False, False, True]),
    )
    torch.testing.assert_close(failure, torch.tensor([True, False, False, False]))
    torch.testing.assert_close(timeout, torch.tensor([False, False, False, False]))
    torch.testing.assert_close(motion_complete, torch.tensor([False, True, True, False]))


def test_past_reward_cannot_change_later_prefix_advantage() -> None:
    base_rewards = torch.tensor(
        [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]], [[7.0, 8.0]]]
    )
    changed_rewards = base_rewards.clone()
    changed_rewards[0] += torch.tensor([1000.0, -1000.0])
    base = _credit(base_rewards)
    changed = _credit(changed_rewards)
    torch.testing.assert_close(base.advantages[1:], changed.advantages[1:])
    assert not torch.allclose(base.advantages[0], changed.advantages[0])


def test_invalid_steps_are_zero_and_break_trace() -> None:
    rewards = torch.ones(3, 1, 2)
    valid = torch.tensor([[True], [False], [True]])
    result = _credit(rewards, valid_mask=valid)
    torch.testing.assert_close(result.advantages[0, 0], torch.ones(2))
    torch.testing.assert_close(result.advantages[1, 0], torch.zeros(2))
    torch.testing.assert_close(result.advantages[2, 0], torch.ones(2))
    torch.testing.assert_close(result.value_targets[1, 0], torch.zeros(2))
    torch.testing.assert_close(result.normalized_advantages[1, 0], torch.zeros(2))


def test_per_channel_per_offset_normalization() -> None:
    rewards = torch.tensor(
        [
            [[1.0, 10.0]],
            [[2.0, 20.0]],
            [[3.0, 30.0]],
            [[4.0, 40.0]],
        ]
    )
    # lambda=0 makes advantages equal rewards and isolates normalization.
    result = compute_dual_channel_gae(
        rewards,
        torch.zeros_like(rewards),
        torch.zeros_like(rewards),
        torch.ones(4, 1),
        torch.ones(4, 1),
        torch.ones(4, 1, dtype=torch.bool),
        gamma=1.0,
        gae_lambda=0.0,
        chunk_horizon=2,
        normalization="per_channel_per_offset",
        actor_weights=(1.0, 1.0),
    )
    expected = torch.tensor(
        [[[-1.0, -1.0]], [[-1.0, -1.0]], [[1.0, 1.0]], [[1.0, 1.0]]]
    )
    torch.testing.assert_close(result.normalized_advantages, expected)


def test_chunk_shared_ablation_broadcasts_start_credit_only_to_actor() -> None:
    rewards = torch.tensor(
        [[[1.0, 10.0]], [[2.0, 20.0]], [[3.0, 30.0]], [[4.0, 40.0]]]
    )
    primitive = _credit(rewards)
    valid = torch.ones(4, 1, dtype=torch.bool)
    shared = with_chunk_shared_actor_credit(
        primitive,
        valid,
        chunk_horizon=2,
        normalization="none",
        actor_weights=(2.0, 3.0),
    )

    expected_channels = torch.tensor(
        [[[10.0, 100.0]], [[10.0, 100.0]], [[7.0, 70.0]], [[7.0, 70.0]]]
    )
    expected_components = expected_channels * torch.tensor([2.0, 3.0])
    torch.testing.assert_close(shared.normalized_advantages, expected_components)
    torch.testing.assert_close(
        shared.actor_advantage,
        2.0 * expected_channels[..., 0] + 3.0 * expected_channels[..., 1],
    )
    # Dual Flow critics still receive the same primitive-step targets; this
    # ablation isolates actor credit assignment rather than changing critics.
    torch.testing.assert_close(shared.value_targets, primitive.value_targets)
    torch.testing.assert_close(shared.advantages, primitive.advantages)


def test_dual_flow_critic_shares_encoder_but_not_heads() -> None:
    torch.manual_seed(7)
    critic = SharedEncoderDualFlowCritic(
        context_dim=6,
        encoder_hidden_dims=(12,),
        embedding_dim=8,
        head_hidden_dims=(10,),
        flow_steps=2,
        eval_samples=4,
    )
    context = torch.randn(5, 6)
    targets = torch.randn(5, 2)

    assert critic.task_head is not critic.amp_head
    task_parameter_ids = {id(parameter) for parameter in critic.task_head.parameters()}
    amp_parameter_ids = {id(parameter) for parameter in critic.amp_head.parameters()}
    assert task_parameter_ids.isdisjoint(amp_parameter_ids)

    values = critic.evaluate(context)
    samples = critic.sample(context, num_samples=3, deterministic=True)
    losses = critic.flow_matching_loss(context, targets, fm_samples=2)
    assert values.shape == (5, 2)
    assert samples.shape == (5, 3, 2)
    assert losses.shape == (5, 2)

    # One channel updates the shared encoder and only its own value head.
    critic.zero_grad(set_to_none=True)
    critic.flow_matching_loss_channel(context, targets[:, 0], "task").mean().backward()
    assert all(parameter.grad is not None for parameter in critic.encoder.parameters())
    assert all(parameter.grad is not None for parameter in critic.task_head.parameters())
    assert all(parameter.grad is None for parameter in critic.amp_head.parameters())

    # The joint objective reaches both independent heads.
    critic.zero_grad(set_to_none=True)
    losses = critic.flow_matching_loss(context, targets, fm_samples=2)
    losses.mean().backward()
    assert all(parameter.grad is not None for parameter in critic.amp_head.parameters())
