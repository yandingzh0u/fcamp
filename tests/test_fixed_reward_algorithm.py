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
from components.rollout.flow_cps_base import FlowCPSBase
from method.fixed_reward import (
    FIXED_REWARD_CHECKPOINT_CONTRACT,
    REWARD_TERM_WEIGHTS,
    FixedRewardFlowCPS,
)
from models.task_flow_critic import TaskFlowCritic


ROOT = Path(__file__).resolve().parents[1]


def test_scalar_gae_crosses_chunks_and_obeys_terminal_semantics() -> None:
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

    # At t=1: failure, timeout, completion. The first/last do not bootstrap;
    # timeout bootstraps but all three truncate the trace.
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
        result.advantages[:, 0],
        torch.tensor([3.0, 2.0, 12.0, 8.0]),
    )
    torch.testing.assert_close(
        result.advantages[:, 1],
        torch.tensor([103.0, 102.0, 12.0, 8.0]),
    )
    torch.testing.assert_close(
        result.advantages[:, 2],
        torch.tensor([3.0, 2.0, 12.0, 8.0]),
    )


def test_ordinary_transition_propagates_across_chunk_boundary() -> None:
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


def test_one_weighted_advantage_normalizer_matches_10_90_measure() -> None:
    advantages = torch.tensor(
        [
            [0.0, 10.0],
            [2.0, 14.0],
        ]
    )
    valid = torch.ones_like(advantages, dtype=torch.bool)
    weights = torch.tensor(
        [
            [0.05, 0.45],
            [0.05, 0.45],
        ]
    )
    credit = SimpleNamespace(
        advantages=advantages,
        value_targets=advantages + 1.0,
    )
    normalized = normalize_actor_advantage(
        credit,
        valid,
        weights,
    )

    weighted_mean = (weights * normalized.actor_advantage).sum()
    weighted_variance = (
        weights * normalized.actor_advantage.square()
    ).sum()
    assert weighted_mean.item() == pytest.approx(0.0, abs=1.0e-6)
    assert weighted_variance.item() == pytest.approx(1.0, abs=1.0e-6)
    torch.testing.assert_close(normalized.advantages, advantages)
    torch.testing.assert_close(
        normalized.value_targets,
        advantages + 1.0,
    )


def test_terminal_resolution_is_failure_first() -> None:
    done = torch.tensor([True, True, True, False])
    timeout = torch.tensor([True, True, False, True])
    complete = torch.tensor([True, False, True, True])
    failure = torch.tensor([True, False, False, True])
    resolved_failure, resolved_timeout, resolved_complete = (
        resolve_terminal_masks(done, timeout, complete, failure)
    )
    torch.testing.assert_close(
        resolved_failure,
        torch.tensor([True, False, False, False]),
    )
    torch.testing.assert_close(
        resolved_timeout,
        torch.tensor([False, True, False, False]),
    )
    torch.testing.assert_close(
        resolved_complete,
        torch.tensor([False, False, True, False]),
    )


def test_task_flow_critic_is_scalar_and_backpropagates() -> None:
    torch.manual_seed(4)
    critic = TaskFlowCritic(
        context_dim=7,
        encoder_hidden_dims=(12, 8),
        head_hidden_dims=(6,),
        activation="elu",
        flow_steps=3,
        eval_samples=4,
    )
    context = torch.randn(5, 7)
    target = torch.randn(5)
    values = critic.evaluate(context)
    losses = critic.flow_matching_loss(context, target, fm_samples=2)

    assert values.shape == (5,)
    assert losses.shape == (5,)
    assert torch.isfinite(values).all()
    assert torch.isfinite(losses).all()
    losses.mean().backward()
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in critic.parameters()
    )


def test_prefix_context_keeps_causal_latent_prefix_only() -> None:
    algo = object.__new__(FixedRewardFlowCPS)
    algo.critic_obs_dim = 4
    algo.actor_obs_dim = 3
    algo.num_act = 2
    algo.horizon_h = 4
    algo.chunk_dim = 8
    algo.prefix_context_dim = 2 * 4 + 3 + 2 + 8 + 2 * 4
    batch = 2
    current = torch.randn(batch, 4)
    chunk_start_critic = torch.randn(batch, 4)
    chunk_start_actor = torch.randn(batch, 3)
    previous_action = torch.randn(batch, 2)
    latent = torch.randn(batch, 8)
    context = algo._prefix_context_raw(
        current,
        chunk_start_critic,
        chunk_start_actor,
        previous_action,
        latent,
        2,
    )

    expected_prefix = torch.zeros(batch, 4, 2)
    expected_prefix[:, :2] = latent.reshape(batch, 4, 2)[:, :2]
    expected_mask = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0]]
    ).expand(batch, -1)
    expected_offset = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0]]
    ).expand(batch, -1)
    torch.testing.assert_close(
        context,
        torch.cat(
            (
                current,
                chunk_start_critic,
                chunk_start_actor,
                previous_action,
                expected_prefix.reshape(batch, -1),
                expected_mask,
                expected_offset,
            ),
            dim=-1,
        ),
    )


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
        # Deliberately non-finite removed dataset fields: they are ignored.
        "body_lin_vel_reward": torch.tensor([float("nan"), float("nan")]),
        "body_ang_vel_reward": torch.tensor([float("nan"), float("nan")]),
    }
    contributions = algo._reward_contributions(terms)
    assert set(contributions) == set(REWARD_TERM_WEIGHTS)
    for key, weight in REWARD_TERM_WEIGHTS.items():
        torch.testing.assert_close(
            contributions[key],
            terms[key] * weight * 0.02,
        )
    reconstructed = sum(contributions.values())
    expected = (
        0.5 * terms["anchor_pos_reward"]
        + 0.5 * terms["anchor_ori_reward"]
        + 2.0 * terms["body_pos_reward"]
        + 2.0 * terms["body_ori_reward"]
        - 0.1 * terms["action_rate"]
        - 10.0 * terms["joint_limit"]
        - 0.1 * terms["undesired_contacts"]
    ) * 0.02
    torch.testing.assert_close(reconstructed, expected)
    assert (reconstructed - expected).abs().max().item() < 1.0e-6


def test_checkpoint_schema_one_rejects_legacy_before_base_restore(
    monkeypatch,
) -> None:
    algo = object.__new__(FixedRewardFlowCPS)
    base_restore_calls: list[dict] = []

    def record_base_restore(self, payload, reset_optimizer=False):
        del self, reset_optimizer
        base_restore_calls.append(payload)

    monkeypatch.setattr(
        FlowCPSBase,
        "load_extra_checkpoint_state",
        record_base_restore,
    )
    with pytest.raises(ValueError, match="fixed_reward_schema_version"):
        algo.load_extra_checkpoint_state(
            {"fcamp_schema_version": 15},
            reset_optimizer=False,
        )
    assert base_restore_calls == []

    algo._validate_checkpoint_contract(
        {
            **FIXED_REWARD_CHECKPOINT_CONTRACT,
            "learning_rate": 3.0e-4,
        }
    )
    with pytest.raises(ValueError, match="legacy"):
        algo._validate_checkpoint_contract(
            {
                **FIXED_REWARD_CHECKPOINT_CONTRACT,
                "disc_optimizer": {},
            }
        )


def test_active_algorithm_modules_do_not_import_removed_subsystems() -> None:
    paths = (
        ROOT / "method" / "fixed_reward.py",
        ROOT / "models" / "task_flow_critic.py",
        ROOT / "components" / "credit" / "task_credit.py",
    )
    import_markers = (
        "components.imitation",
        "components.replay",
        "style_discriminator",
        "dual_flow_critic",
        "window_mmd",
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert all(marker not in source for marker in import_markers)
