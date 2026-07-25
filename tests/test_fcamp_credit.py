from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from components.credit.temporal_credit import (
    compute_dual_channel_gae,
    normalize_actor_mixture,
    resolve_terminal_masks,
)
from models.dual_flow_critic import SharedEncoderDualFlowCritic
from components.rollout.training_streams import Phase0CurriculumStreams
from method.fcamp import FCAMP


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
        actor_weights=(2.0, 3.0))


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
        gamma=0.5)
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
        failure=torch.tensor([True, False, False, True]))
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


def test_actor_uses_one_weighted_global_advantage_normalizer() -> None:
    rewards = torch.tensor(
        [
            [[1.0, 0.1], [4.0, 0.3]],
            [[2.0, 0.4], [3.0, 0.2]],
            [[5.0, 0.2], [2.0, 0.6]],
            [[3.0, 0.5], [8.0, 0.1]],
        ]
    )
    valid = torch.ones(4, 2, dtype=torch.bool)
    raw = compute_dual_channel_gae(
        rewards,
        torch.zeros_like(rewards),
        torch.zeros_like(rewards),
        torch.ones(4, 2),
        torch.ones(4, 2),
        valid,
        gamma=1.0,
        gae_lambda=0.0,
        chunk_horizon=2,
        normalization="none",
        actor_weights=(1.0, 0.2))
    # Match a 10/90 actor objective: each stream's valid samples share its mass.
    sample_weights = torch.tensor(
        [[0.025, 0.225], [0.025, 0.225], [0.025, 0.225], [0.025, 0.225]]
    )
    result = normalize_actor_mixture(raw, valid, sample_weights)

    raw_components = raw.actor_advantage_components
    component_mean = (
        sample_weights.unsqueeze(-1) * raw_components
    ).sum(dim=(0, 1))
    centered_components = raw_components - component_mean
    centered_mixed = centered_components.sum(dim=-1)
    inv_std = torch.rsqrt(
        (sample_weights * centered_mixed.square()).sum() + 1.0e-8
    )
    expected_components = centered_components * inv_std
    expected_actor = expected_components.sum(dim=-1)

    torch.testing.assert_close(
        result.actor_advantage_components,
        expected_components)
    torch.testing.assert_close(result.actor_advantage, expected_actor)
    torch.testing.assert_close(
        result.actor_advantage_components.sum(dim=-1),
        result.actor_advantage)
    assert abs(float((sample_weights * result.actor_advantage).sum())) < 1.0e-6
    assert abs(
        float(
            (sample_weights * result.actor_advantage.square()).sum()
        )
        - 1.0
    ) < 1.0e-5
    # AMP retains its physical 0.2 scale; it is not independently inflated to
    # unit variance.
    assert (
        result.actor_advantage_components[..., 1].std(unbiased=False)
        < result.actor_advantage_components[..., 0].std(unbiased=False)
    )


def test_fcamp_assign_credit_has_one_normalizer_across_10_90_streams() -> None:
    algo = object.__new__(FCAMP)
    algo.cfg = SimpleNamespace(
        credit=SimpleNamespace(
            advantage_normalization="global",
            task_weight=1.0,
            amp_weight=1.0,
            mode="causal_frame"),
        discount_gamma=0.0,
        gae_lambda=0.0,
        streams=SimpleNamespace(phase0_fraction=0.10))
    algo.training_streams = Phase0CurriculumStreams.create(
        10,
        phase0_fraction=0.10,
        phase0_start=0,
        device="cpu")
    task = torch.tensor(
        [
            [
                [10.0, 12.0],
                [0.0, 1.0],
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
                [5.0, 6.0],
                [6.0, 7.0],
                [7.0, 8.0],
                [8.0, 9.0],
            ]
        ]
    )
    valid = torch.ones(1, 10, 2, dtype=torch.bool)
    rollout = {
        "task_reward": task,
        "amp_reward_credit": torch.zeros_like(task),
        "values": torch.zeros(1, 10, 2, 2),
        "next_values": torch.zeros(1, 10, 2, 2),
        "bootstrap_mask": valid,
        "trace_mask": valid,
        "valid": valid}

    algo._assign_credit(rollout)
    actor = rollout["advantages"]
    phase0_mean = actor[:, :1].mean()
    curriculum_mean = actor[:, 1:].mean()
    objective_mean = 0.10 * phase0_mean + 0.90 * curriculum_mean
    objective_square_mean = (
        0.10 * actor[:, :1].square().mean()
        + 0.90 * actor[:, 1:].square().mean()
    )

    assert abs(float(phase0_mean)) > 0.1
    assert abs(float(curriculum_mean)) > 0.1
    assert abs(float(objective_mean)) < 1.0e-6
    assert abs(float(objective_square_mean) - 1.0) < 1.0e-5
    torch.testing.assert_close(
        rollout["mixed_advantage"],
        rollout["channel_advantages"].sum(dim=-1))


def test_actor_weights_change_only_mixed_actor_credit() -> None:
    rewards = torch.randn(5, 3, 2)
    values = torch.randn_like(rewards)
    next_values = torch.randn_like(rewards)
    common = dict(
        rewards=rewards,
        values=values,
        next_values=next_values,
        bootstrap_mask=torch.ones(5, 3),
        trace_mask=torch.ones(5, 3),
        valid_mask=torch.ones(5, 3, dtype=torch.bool),
        gamma=0.97,
        gae_lambda=0.91,
        chunk_horizon=2,
        normalization="none")
    task_only = compute_dual_channel_gae(
        **common,
        actor_weights=(1.0, 0.0))
    mixed = compute_dual_channel_gae(
        **common,
        actor_weights=(1.0, 0.25))

    torch.testing.assert_close(task_only.td_errors, mixed.td_errors)
    torch.testing.assert_close(task_only.advantages, mixed.advantages)
    torch.testing.assert_close(task_only.value_targets, mixed.value_targets)
    torch.testing.assert_close(
        mixed.mixed_advantage,
        mixed.advantages[..., 0] + 0.25 * mixed.advantages[..., 1])
    assert not torch.allclose(task_only.mixed_advantage, mixed.mixed_advantage)


@pytest.mark.parametrize(
    "legacy_mode",
    ["per_offset", "per_channel_per_offset", "per_channel_global"])
def test_multi_normalizer_actor_credit_modes_are_rejected(
    legacy_mode: str) -> None:
    rewards = torch.ones(2, 1, 2)
    with pytest.raises(ValueError, match="global.*none"):
        compute_dual_channel_gae(
            rewards,
            torch.zeros_like(rewards),
            torch.zeros_like(rewards),
            torch.ones(2, 1),
            torch.ones(2, 1),
            torch.ones(2, 1, dtype=torch.bool),
            gamma=1.0,
            gae_lambda=0.0,
            chunk_horizon=2,
            normalization=legacy_mode)


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
