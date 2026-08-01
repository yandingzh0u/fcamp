from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from components.credit.task_credit import (
    compute_task_gae,
    normalize_actor_advantage,
    resolve_terminal_masks,
)
from components.rollout.fixed_reward_contract import (
    FIXED_REWARD_CHECKPOINT_CONTRACT,
)
from method.fixed_reward import REWARD_TERM_WEIGHTS, FixedRewardFlowCPS
from models.value_critic import ValueCritic


ROOT = Path(__file__).resolve().parents[1]


def test_scalar_gae_obeys_failure_timeout_and_completion_semantics() -> None:
    rewards = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [4.0, 4.0, 4.0],
            [8.0, 8.0, 8.0],
        ]
    )
    values = torch.zeros_like(rewards)
    next_values = torch.zeros_like(rewards)
    next_values[1] = torch.tensor([100.0, 100.0, 100.0])
    valid = torch.ones_like(rewards, dtype=torch.bool)
    bootstrap = torch.ones_like(valid)
    trace = torch.ones_like(valid)

    # At t=1: failure, timeout, completion. Failure/completion do not
    # bootstrap; timeout does. Every terminal truncates the GAE trace.
    bootstrap[1, 0] = False
    bootstrap[1, 2] = False
    trace[1] = False
    result = compute_task_gae(
        rewards,
        values,
        next_values,
        bootstrap,
        trace,
        valid,
        gamma=1.0,
        gae_lambda=1.0,
    )

    torch.testing.assert_close(
        result.advantages[:, 0], torch.tensor([3.0, 2.0, 12.0, 8.0])
    )
    torch.testing.assert_close(
        result.advantages[:, 1], torch.tensor([103.0, 102.0, 12.0, 8.0])
    )
    torch.testing.assert_close(
        result.advantages[:, 2], torch.tensor([3.0, 2.0, 12.0, 8.0])
    )


def test_ordinary_single_step_transitions_propagate_chronologically() -> None:
    rewards = torch.arange(1.0, 7.0).reshape(6, 1)
    zeros = torch.zeros_like(rewards)
    valid = torch.ones_like(rewards, dtype=torch.bool)
    result = compute_task_gae(
        rewards,
        zeros,
        zeros,
        valid,
        valid,
        valid,
        gamma=1.0,
        gae_lambda=1.0,
    )
    torch.testing.assert_close(
        result.advantages[:, 0],
        torch.tensor([21.0, 20.0, 18.0, 15.0, 11.0, 6.0]),
    )


def test_weighted_advantage_normalizer_matches_10_90_measure() -> None:
    advantages = torch.tensor([[0.0, 10.0], [2.0, 14.0]])
    valid = torch.ones_like(advantages, dtype=torch.bool)
    weights = torch.tensor([[0.05, 0.45], [0.05, 0.45]])
    credit = SimpleNamespace(
        advantages=advantages,
        value_targets=advantages + 1.0,
    )
    normalized = normalize_actor_advantage(credit, valid, weights)

    weighted_mean = (weights * normalized.actor_advantage).sum()
    weighted_variance = (weights * normalized.actor_advantage.square()).sum()
    assert weighted_mean.item() == pytest.approx(0.0, abs=1.0e-6)
    assert weighted_variance.item() == pytest.approx(1.0, abs=1.0e-6)
    torch.testing.assert_close(normalized.advantages, advantages)
    torch.testing.assert_close(normalized.value_targets, advantages + 1.0)


def test_terminal_resolution_is_failure_first() -> None:
    done = torch.tensor([True, True, True, False])
    timeout = torch.tensor([True, True, False, True])
    complete = torch.tensor([True, False, True, True])
    failure = torch.tensor([True, False, False, True])
    resolved_failure, resolved_timeout, resolved_complete = (
        resolve_terminal_masks(done, timeout, complete, failure)
    )
    torch.testing.assert_close(
        resolved_failure, torch.tensor([True, False, False, False])
    )
    torch.testing.assert_close(
        resolved_timeout, torch.tensor([False, True, False, False])
    )
    torch.testing.assert_close(
        resolved_complete, torch.tensor([False, False, True, False])
    )


def test_state_only_value_critic_is_scalar_and_backpropagates() -> None:
    torch.manual_seed(4)
    critic = ValueCritic(
        observation_dim=7,
        hidden_dims=(12, 8),
        activation="elu",
    )
    observation = torch.randn(5, 7)
    target = torch.randn(5)
    values = critic(observation)
    loss = (values - target).square().mean()

    assert values.shape == (5,)
    assert torch.isfinite(values).all()
    loss.backward()
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and float(parameter.grad.abs().sum()) > 0.0
        for parameter in critic.parameters()
    )
    with pytest.raises(ValueError, match="observation must have shape"):
        critic(torch.randn(5, 8))


def test_reward_decomposition_has_exact_weights_and_one_dt() -> None:
    algo = object.__new__(FixedRewardFlowCPS)
    algo.env = SimpleNamespace(dt=0.02)
    terms = {
        "anchor_pos_reward": torch.tensor([0.9, 0.8]),
        "anchor_ori_reward": torch.tensor([0.7, 0.6]),
        "body_pos_reward": torch.tensor([0.5, 0.4]),
        "body_ori_reward": torch.tensor([0.3, 0.2]),
        "action_rate": torch.tensor([0.1, 0.2]),
        "joint_limit": torch.tensor([0.01, 0.02]),
        "undesired_contacts": torch.tensor([1.0, 2.0]),
        # Removed velocity rewards must never be read.
        "body_lin_vel_reward": torch.tensor([float("nan"), float("nan")]),
        "body_ang_vel_reward": torch.tensor([float("nan"), float("nan")]),
    }
    contributions = algo._reward_contributions(terms)
    assert set(contributions) == set(REWARD_TERM_WEIGHTS)
    for key, weight in REWARD_TERM_WEIGHTS.items():
        torch.testing.assert_close(
            contributions[key], terms[key] * weight * 0.02
        )
    expected = (
        0.5 * terms["anchor_pos_reward"]
        + 0.5 * terms["anchor_ori_reward"]
        + 2.0 * terms["body_pos_reward"]
        + 2.0 * terms["body_ori_reward"]
        - 0.1 * terms["action_rate"]
        - 10.0 * terms["joint_limit"]
        - 0.1 * terms["undesired_contacts"]
    ) * 0.02
    reconstructed = sum(contributions.values())
    torch.testing.assert_close(reconstructed, expected)
    assert float((reconstructed - expected).abs().max()) < 1.0e-6


def test_schema_14_rejects_every_old_algorithm_before_restore() -> None:
    assert FIXED_REWARD_CHECKPOINT_CONTRACT == {
        "fixed_reward_schema_version": 14,
        "control_semantics": "closed_loop_h1_v1",
        "policy_semantics": "primitive_flow_cps_ppo_v1",
        "action_semantics": "absolute_tanh_action_v1",
        "cps_semantics": "joint_covariance_learned_global_eta_v1",
        "critic_semantics": "state_only_scalar_value_v1",
        "gae_semantics": "primitive_gae_v1",
    }
    algo = object.__new__(FixedRewardFlowCPS)
    with pytest.raises(ValueError, match="semantic contract mismatch"):
        algo._validate_checkpoint_contract(
            {
                **FIXED_REWARD_CHECKPOINT_CONTRACT,
                "fixed_reward_schema_version": 4,
            }
        )
    algo._validate_checkpoint_contract(FIXED_REWARD_CHECKPOINT_CONTRACT)
    with pytest.raises(ValueError, match="removed state"):
        algo._validate_checkpoint_contract(
            {**FIXED_REWARD_CHECKPOINT_CONTRACT, "disc_optimizer": {}}
        )


def test_active_algorithm_is_only_single_step_flow_and_scalar_value() -> None:
    active_paths = (
        ROOT / "method" / "fixed_reward.py",
        ROOT / "models" / "flow_cps_policy.py",
        ROOT / "models" / "value_critic.py",
        ROOT / "configs" / "fixed_reward_largebox.yaml",
    )
    forbidden = (
        "GRU",
        "GRUCell",
        "frame_pos_embed",
        "causal_cell",
        "cumulative_residual",
        "_prefix_context",
        "prefix_context_normalizer",
        "chunk_boundary",
        "boundary_internal_ratio",
        "chunk_offset",
        "frame_0_",
        "offset_innovation_rms",
        "TaskFlowCritic",
        "FlowCPSBase",
    )
    for path in active_paths:
        source = path.read_text(encoding="utf-8")
        assert all(marker not in source for marker in forbidden)

    assert not (ROOT / "components" / "rollout" / "flow_cps_base.py").exists()
    assert not (ROOT / "models" / "task_flow_critic.py").exists()
