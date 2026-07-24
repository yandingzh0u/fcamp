from __future__ import annotations

import math

import pytest
import torch

from components.imitation.motion_features import (
    ImitationFeatureSchema,
    build_imitation_frame,
    canonicalize_imitation_window,
    quat_to_rot6d,
)
from components.imitation.temporal_history import TemporalFeatureHistory
from components.normalization.running_stats import RunningNormalizer
from components.imitation.style_reward import discriminator_style_reward
from models.style_discriminator import StyleDiscriminator, compute_style_discriminator_loss


def test_feature_schema_is_233_and_policy_demo_share_builder() -> None:
    schema = ImitationFeatureSchema()
    assert schema.history_len == 16
    assert schema.frame_dim == 233
    assert schema.window_dim == 3728
    batch = 3
    identity = torch.zeros(batch, 4)
    identity[:, 0] = 1.0
    joint_identity = torch.zeros(batch, 29, 4)
    joint_identity[..., 0] = 1.0
    kwargs = dict(
        root_pos=torch.randn(batch, 3),
        root_quat=identity,
        joint_rotation=joint_identity,
        key_body_pos=torch.randn(batch, 5, 3),
        root_lin_vel=torch.randn(batch, 3),
        root_ang_vel=torch.randn(batch, 3),
        dof_vel=torch.randn(batch, 29),
        schema=schema,
    )
    policy = build_imitation_frame(**kwargs)
    demo = build_imitation_frame(**kwargs)
    assert policy.shape == (batch, 233)
    torch.testing.assert_close(policy, demo)


def test_rotation_6d_uses_mimickit_tangent_and_normal() -> None:
    # Identity maps tangent x and normal z to themselves.
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(
        quat_to_rot6d(identity),
        torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]),
    )


def test_imitation_window_canonicalization_uses_final_root_xy_and_preserves_height() -> None:
    window = torch.randn(2, 4, 8)
    window[:, :, :3] = torch.tensor(
        [
            [[10.0, -4.0, 0.72], [11.5, -3.0, 0.75], [13.0, -1.0, 0.78], [14.0, 2.0, 0.81]],
            [[-8.0, 7.0, 1.02], [-7.0, 5.5, 0.98], [-4.0, 4.0, 0.94], [-1.0, 3.0, 0.90]],
        ]
    )
    raw = window.clone()

    canonical = canonicalize_imitation_window(window)
    expected = raw.clone()
    expected[:, :, :2] -= raw[:, -1:, :2]

    torch.testing.assert_close(canonical, expected)
    torch.testing.assert_close(canonical[:, -1, :2], torch.zeros(2, 2))
    torch.testing.assert_close(canonical[:, :, 2], raw[:, :, 2])
    torch.testing.assert_close(canonical[:, :, 3:], raw[:, :, 3:])
    # Canonicalization is an output transform and must not mutate stored frames.
    torch.testing.assert_close(window, raw)

    shifted = raw.clone()
    shifted[:, :, :2] += torch.tensor([[37.0, -19.0], [-11.0, 23.0]])[:, None, :]
    torch.testing.assert_close(canonicalize_imitation_window(shifted), canonical)


def test_history_order_push_and_subset_expert_reset() -> None:
    history = TemporalFeatureHistory(2, history_len=4, feature_dim=1)
    initial = torch.tensor([[[0.0], [1.0], [2.0], [3.0]], [[10.0], [11.0], [12.0], [13.0]]])
    history.reset(initial[:, 0])
    for frame_idx in range(1, 4):
        history.push(initial[:, frame_idx])
    history.push(torch.tensor([[4.0], [14.0]]))
    assert history.flatten().tolist() == [[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 13.0, 14.0]]

    env1 = torch.tensor([1])
    env1_history = torch.tensor([[[19.0], [20.0], [21.0], [22.0], [23.0]]])
    history.reset(env1_history[:, 0], env1)
    for frame_idx in range(1, 5):
        history.push(env1_history[:, frame_idx], env1)
    assert history.flatten(env1).tolist() == [[20.0, 21.0, 22.0, 23.0]]
    # Resetting env1 must not alter env0 or reset at a chunk boundary.
    assert history.flatten(torch.tensor([0])).tolist() == [[1.0, 2.0, 3.0, 4.0]]


def test_history_canonicalized_flatten_does_not_modify_raw_ring_frames() -> None:
    history = TemporalFeatureHistory(1, history_len=3, feature_dim=5)
    initial = torch.tensor(
        [[[10.0, -4.0, 0.72, 1.0, 2.0], [12.0, -1.0, 0.76, 3.0, 4.0], [15.0, 3.0, 0.80, 5.0, 6.0]]]
    )
    history.reset(initial[:, 0])
    history.push(initial[:, 1])
    history.push(initial[:, 2])
    history.push(torch.tensor([[19.0, 8.0, 0.84, 7.0, 8.0]]))
    raw_window = torch.tensor(
        [[[12.0, -1.0, 0.76, 3.0, 4.0], [15.0, 3.0, 0.80, 5.0, 6.0], [19.0, 8.0, 0.84, 7.0, 8.0]]]
    )

    torch.testing.assert_close(history.window(), raw_window)
    canonical = history.flatten(canonicalize_root=True).reshape(1, 3, 5)
    torch.testing.assert_close(canonical, canonicalize_imitation_window(raw_window))
    torch.testing.assert_close(canonical[:, -1, :2], torch.zeros(1, 2))
    torch.testing.assert_close(canonical[:, :, 2], raw_window[:, :, 2])
    # Reading a canonicalized view must leave both window() and default flatten raw.
    torch.testing.assert_close(history.window(), raw_window)
    torch.testing.assert_close(history.flatten().reshape(1, 3, 5), raw_window)


def test_demo_seeded_history_is_fixed_width_from_first_policy_step() -> None:
    history = TemporalFeatureHistory(1, history_len=4, feature_dim=1)
    seed = torch.tensor([[[0.0], [1.0], [2.0], [3.0]]], dtype=torch.float32)
    history.reset_seeded(seed)

    assert history.seeded.tolist() == [True]
    assert history.ready.tolist() == [False]
    history.push(torch.tensor([[4.0]], dtype=torch.float32))

    assert history.ready.tolist() == [True]
    assert history.causal_ready.tolist() == [True]
    assert history.window_ages().tolist() == [[-2, -1, 0, 1]]
    assert history.flatten().tolist() == [[1.0, 2.0, 3.0, 4.0]]


def test_intervention_after_excludes_exactly_next_w_minus_one_windows() -> None:
    history = TemporalFeatureHistory(1, history_len=4, feature_dim=1)
    history.reset_seeded(
        torch.tensor([[[0.0], [1.0], [2.0], [3.0]]], dtype=torch.float32)
    )

    # The intervention is after endpoint 4, so that endpoint remains clean.
    history.push(
        torch.tensor([[4.0]], dtype=torch.float32),
        intervention_after=torch.tensor([True]),
    )
    assert history.causal_ready.tolist() == [True]
    assert history.flatten().tolist() == [[1.0, 2.0, 3.0, 4.0]]

    for value in (5.0, 6.0, 7.0):
        history.push(torch.tensor([[value]], dtype=torch.float32))
        assert history.ready.tolist() == [True]
        assert history.causal_ready.tolist() == [False]
        with pytest.raises(RuntimeError, match="external intervention"):
            history.window()

    history.push(torch.tensor([[8.0]], dtype=torch.float32))
    assert history.causal_ready.tolist() == [True]
    assert history.flatten().tolist() == [[5.0, 6.0, 7.0, 8.0]]


def test_history_rejects_uninitialized_push() -> None:
    history = TemporalFeatureHistory(1, history_len=4, feature_dim=2)
    with pytest.raises(RuntimeError, match="imitation history"):
        history.push(torch.zeros(1, 2))


def test_imitation_reward_matches_definition_and_clamps() -> None:
    logits = torch.tensor([-2.0, 0.0, 2.0, 100.0])
    rewards = discriminator_style_reward(logits, scale=2.0)
    expected = -2.0 * torch.log(torch.clamp(1.0 - torch.sigmoid(logits), min=1.0e-4))
    torch.testing.assert_close(rewards, expected)
    assert rewards[-1].item() == pytest.approx(-2.0 * math.log(1.0e-4), rel=1e-6)


def test_standard_discriminator_loss_has_bilateral_gp_and_backpropagates() -> None:
    torch.manual_seed(3)
    discriminator = StyleDiscriminator(8, hidden_dims=(16, 8))
    output = compute_style_discriminator_loss(
        discriminator,
        expert_observations=torch.randn(5, 8),
        policy_observations=torch.randn(4, 8),
        replay_observations=torch.randn(3, 8),
        gradient_penalty_weight=10.0,
        logit_regularization_weight=0.01,
    )
    assert torch.isfinite(output.loss)
    assert output.metrics["disc/expert_gradient_penalty"] > 0
    assert output.metrics["disc/fake_gradient_penalty"] > 0
    assert "disc/replay_bce" in output.metrics
    output.loss.backward()
    assert all(parameter.grad is not None for parameter in discriminator.parameters())


def test_normalizer_freeze_commit_and_state_dict_roundtrip() -> None:
    normalizer = RunningNormalizer(2, clip=10.0)
    samples = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    normalizer.freeze()
    normalizer.record(samples)
    torch.testing.assert_close(normalizer.normalize(samples), samples)
    with pytest.raises(RuntimeError, match="frozen"):
        normalizer.commit()
    normalizer.unfreeze()
    assert normalizer.commit()
    torch.testing.assert_close(normalizer.mean, torch.tensor([2.0, 3.0]))
    normalized = normalizer.normalize(samples)
    torch.testing.assert_close(normalized.mean(dim=0), torch.zeros(2))

    restored = RunningNormalizer(2)
    restored.load_state_dict(normalizer.state_dict())
    torch.testing.assert_close(restored.normalize(samples), normalized)
