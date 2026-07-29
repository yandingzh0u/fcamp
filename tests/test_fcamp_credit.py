from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from components.credit.temporal_credit import (
    compute_amp_gae,
    normalize_amp_advantage,
    resolve_terminal_masks,
)
from components.rollout.training_streams import CURRICULUM_STREAM, PHASE0_STREAM
from method.fcamp import FCAMP
from models.flow_critic import FlowCritic


def _credit(
    rewards: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
    bootstrap: torch.Tensor | None = None,
    trace: torch.Tensor | None = None,
):
    valid = torch.ones_like(rewards, dtype=torch.bool) if valid is None else valid
    bootstrap = valid if bootstrap is None else bootstrap
    trace = valid if trace is None else trace
    values = torch.zeros_like(rewards)
    return compute_amp_gae(
        rewards,
        values,
        values,
        bootstrap,
        trace,
        valid,
        gamma=1.0,
        gae_lambda=1.0,
    )


def test_amp_credit_is_causal_and_crosses_chunk_boundaries() -> None:
    rewards = torch.tensor([[0.0], [0.0], [1.0], [0.0]])
    credit = _credit(rewards)
    torch.testing.assert_close(
        credit.advantages[:, 0],
        torch.tensor([1.0, 1.0, 1.0, 0.0]),
    )

    changed_past = rewards.clone()
    changed_past[0] = 7.0
    changed = _credit(changed_past)
    torch.testing.assert_close(
        changed.advantages[1:],
        credit.advantages[1:],
    )


def test_delayed_w16_amp_reward_credits_every_preceding_real_action() -> None:
    """A W=16 discriminator cannot score the first W-2 action endpoints.

    Those actions are nevertheless real policy decisions.  Once the first
    clean discriminator reward arrives at endpoint W-2, chronological GAE must
    carry its discounted credit through every preceding action, rather than
    treating the warm-up endpoints as an invalid trace segment.
    """

    window = 16
    first_reward_step = window - 2
    horizon = first_reward_step + 3
    action_valid = torch.ones(horizon, 1, dtype=torch.bool)
    endpoint_valid = torch.zeros_like(action_valid)
    endpoint_valid[first_reward_step:] = True
    rewards = torch.zeros(horizon, 1)
    rewards[first_reward_step] = 2.5
    gamma = 0.99
    gae_lambda = 0.95

    # Endpoint validity gates reward production only.  It must not erase the
    # preceding, valid actor actions from the temporal credit trace.
    assert not bool(endpoint_valid[:first_reward_step].any())
    assert bool(endpoint_valid[first_reward_step])
    credit = compute_amp_gae(
        rewards,
        torch.zeros_like(rewards),
        torch.zeros_like(rewards),
        action_valid,
        action_valid,
        action_valid,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    expected = torch.zeros_like(rewards)
    decay = gamma * gae_lambda
    for step in range(first_reward_step + 1):
        expected[step] = rewards[first_reward_step] * decay ** (
            first_reward_step - step
        )
    torch.testing.assert_close(credit.advantages, expected)
    assert bool((credit.advantages[:first_reward_step] > 0.0).all())


def test_delayed_amp_credit_does_not_cross_intervention_or_terminal_cut() -> None:
    """A delayed style reward may cross warm-up actions, but never a cut edge."""

    window = 16
    first_reward_step = window - 2
    cut_step = 6
    horizon = first_reward_step + 2
    action_valid = torch.ones(horizon, 1, dtype=torch.bool)
    rewards = torch.zeros(horizon, 1)
    rewards[first_reward_step] = 1.0
    trace = action_valid.clone()
    bootstrap = action_valid.clone()
    # Both an intervention edge and a terminal transition use this same GAE
    # boundary contract: no later TD error may cross this primitive action.
    trace[cut_step] = False
    bootstrap[cut_step] = False

    credit = compute_amp_gae(
        rewards,
        torch.zeros_like(rewards),
        torch.zeros_like(rewards),
        bootstrap,
        trace,
        action_valid,
        gamma=1.0,
        gae_lambda=1.0,
    )

    expected = torch.zeros_like(rewards)
    expected[cut_step + 1 : first_reward_step + 1] = 1.0
    torch.testing.assert_close(credit.advantages, expected)
    assert bool((credit.advantages[: cut_step + 1] == 0.0).all())


def test_fcamp_gae_consumes_dt_scaled_reward_not_raw_reward() -> None:
    algo = object.__new__(FCAMP)
    stream_ids = torch.tensor(
        [PHASE0_STREAM, CURRICULUM_STREAM],
        dtype=torch.int8,
    )
    algo.cfg = SimpleNamespace(
        discount_gamma=0.0,
        gae_lambda=0.0,
        streams=SimpleNamespace(phase0_fraction=0.5),
    )
    algo.env = SimpleNamespace(dt=0.02)
    algo.training_streams = SimpleNamespace(stream_ids=stream_ids)
    rollout = {
        "valid": torch.ones(1, 2, 2, dtype=torch.bool),
        "amp_valid": torch.ones(1, 2, 2, dtype=torch.bool),
        "amp_reward": torch.ones(1, 2, 2),
        "values": torch.zeros(1, 2, 2),
        "next_values": torch.zeros(1, 2, 2),
        "bootstrap_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        "trace_mask": torch.ones(1, 2, 2, dtype=torch.bool),
        "stream_ids": stream_ids,
    }

    algo._assign_credit(rollout)

    torch.testing.assert_close(
        rollout["amp_reward_credit"],
        torch.full((1, 2, 2), 0.02),
    )
    torch.testing.assert_close(
        rollout["amp_advantages"],
        rollout["amp_reward_credit"],
    )
    torch.testing.assert_close(
        rollout["value_targets"],
        rollout["amp_reward_credit"],
    )


def test_fcamp_assign_credit_keeps_w16_warmup_actions_trainable() -> None:
    algo = object.__new__(FCAMP)
    algo.cfg = SimpleNamespace(
        discount_gamma=0.99,
        gae_lambda=0.95,
        streams=SimpleNamespace(phase0_fraction=0.1),
    )
    algo.env = SimpleNamespace(dt=1.0)
    algo.training_streams = SimpleNamespace(
        stream_ids=torch.tensor([PHASE0_STREAM], dtype=torch.int8)
    )
    chunks = 5
    horizon = 4
    time_steps = chunks * horizon
    first_endpoint_step = 14

    def chunk_layout(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(chunks, horizon, 1).permute(0, 2, 1)

    rewards_time = torch.zeros(time_steps, 1)
    rewards_time[first_endpoint_step] = 2.0
    endpoint_time = torch.zeros(time_steps, 1, dtype=torch.bool)
    endpoint_time[first_endpoint_step:] = True
    action_time = torch.ones(time_steps, 1, dtype=torch.bool)
    rollout = {
        "valid": chunk_layout(action_time),
        "amp_valid": chunk_layout(endpoint_time),
        "amp_reward": chunk_layout(rewards_time),
        "values": chunk_layout(torch.zeros_like(rewards_time)),
        "next_values": chunk_layout(torch.zeros_like(rewards_time)),
        "bootstrap_mask": chunk_layout(action_time),
        "trace_mask": chunk_layout(action_time),
    }

    algo._assign_credit(rollout)

    credit_valid_time = (
        rollout["credit_valid"].permute(0, 2, 1).reshape(time_steps, 1)
    )
    amp_advantage_time = (
        rollout["amp_advantages"].permute(0, 2, 1).reshape(time_steps, 1)
    )
    delayed_amp_reachable_time = (
        rollout["delayed_amp_reachable"]
        .permute(0, 2, 1)
        .reshape(time_steps, 1)
    )
    assert torch.equal(credit_valid_time, action_time)
    assert bool((amp_advantage_time[:first_endpoint_step] > 0.0).all())
    assert bool(
        delayed_amp_reachable_time[: first_endpoint_step + 1].all()
    )
    assert not bool(
        delayed_amp_reachable_time[first_endpoint_step + 1 :].any()
    )
    assert amp_advantage_time[first_endpoint_step].item() == pytest.approx(2.0)


def test_fcamp_assign_credit_rejects_reward_without_endpoint() -> None:
    algo = object.__new__(FCAMP)
    algo.cfg = SimpleNamespace(
        discount_gamma=0.99,
        gae_lambda=0.95,
        streams=SimpleNamespace(phase0_fraction=0.1),
    )
    algo.env = SimpleNamespace(dt=1.0)
    algo.training_streams = SimpleNamespace(
        stream_ids=torch.tensor([PHASE0_STREAM], dtype=torch.int8)
    )
    rollout = {
        "valid": torch.ones(1, 1, 1, dtype=torch.bool),
        "amp_valid": torch.zeros(1, 1, 1, dtype=torch.bool),
        "amp_reward": torch.ones(1, 1, 1),
        "values": torch.zeros(1, 1, 1),
        "next_values": torch.zeros(1, 1, 1),
        "bootstrap_mask": torch.ones(1, 1, 1, dtype=torch.bool),
        "trace_mask": torch.ones(1, 1, 1, dtype=torch.bool),
    }

    with pytest.raises(RuntimeError, match="non-zero AMP reward"):
        algo._assign_credit(rollout)


def test_invalid_actions_break_amp_trace_and_are_excluded() -> None:
    rewards = torch.tensor([[0.0], [0.0], [0.0], [0.0], [1.0]])
    valid = torch.tensor([[True], [False], [False], [True], [True]])
    credit = _credit(rewards, valid=valid)

    torch.testing.assert_close(credit.advantages[:, 0], torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0]))
    assert torch.equal(credit.valid_mask, valid)


def test_intervention_edge_cuts_bootstrap_and_trace() -> None:
    rewards = torch.tensor([[0.0], [1.0]])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    cut = torch.tensor([[False], [True]])
    credit = _credit(rewards, valid=valid, bootstrap=cut, trace=cut)
    torch.testing.assert_close(credit.advantages[:, 0], torch.tensor([0.0, 1.0]))


def test_advantage_normalization_uses_exact_weighted_measure() -> None:
    rewards = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    raw = _credit(rewards)
    weights = torch.tensor([[0.05, 0.45], [0.05, 0.45]])
    normalized = normalize_amp_advantage(raw, raw.valid_mask, weights)

    actor = normalized.actor_advantage
    assert abs(float((actor * weights).sum().item() / weights.sum().item())) < 1.0e-6
    assert abs(float((actor.square() * weights).sum().item() / weights.sum().item()) - 1.0) < 1.0e-5


def test_terminal_resolution_is_failure_first() -> None:
    done = torch.tensor([True, True, True, False])
    failure, timeout, complete = resolve_terminal_masks(
        done,
        timeout=torch.tensor([True, True, False, True]),
        motion_complete=torch.tensor([True, False, True, True]),
        failure=torch.tensor([True, False, False, True]),
    )
    assert torch.equal(failure, torch.tensor([True, False, False, False]))
    assert torch.equal(timeout, torch.tensor([False, True, False, False]))
    assert torch.equal(complete, torch.tensor([False, False, True, False]))


def test_flow_critic_has_one_scalar_amp_head() -> None:
    critic = FlowCritic(
        context_dim=7,
        encoder_hidden_dims=(16, 8),
        head_hidden_dims=(8,),
        activation="elu",
        flow_steps=2,
        eval_samples=3,
    )
    context = torch.randn(5, 7)
    targets = torch.randn(5)

    assert critic.evaluate(context).shape == (5,)
    losses = critic.flow_matching_loss(context, targets, fm_samples=2)
    assert losses.shape == (5,)
    losses.mean().backward()
    assert all(parameter.grad is not None for parameter in critic.parameters())
