from __future__ import annotations

from types import SimpleNamespace

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


def test_dirty_windows_break_amp_trace_and_are_excluded() -> None:
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
