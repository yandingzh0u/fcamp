from __future__ import annotations

import pytest
import torch

from net.mixgrpo import MIN_POLICY_OBS_DIM, FlowMatchingPolicy


def test_flow_policy_prepares_observation_by_padding_and_truncating() -> None:
    policy = FlowMatchingPolicy(obs_dim=10, action_dim=3, horizon=2, hidden_dims=(16,))

    short_obs = torch.ones(5, 7)
    long_obs = torch.arange(5 * (MIN_POLICY_OBS_DIM + 3), dtype=torch.float32).view(5, MIN_POLICY_OBS_DIM + 3)

    padded = policy._prepare_observation(short_obs)
    truncated = policy._prepare_observation(long_obs)

    assert padded.shape == (5, MIN_POLICY_OBS_DIM)
    assert torch.allclose(padded[:, :7], short_obs)
    assert torch.allclose(padded[:, 7:], torch.zeros(5, MIN_POLICY_OBS_DIM - 7))
    assert truncated.shape == (5, MIN_POLICY_OBS_DIM)
    assert torch.equal(truncated, long_obs[:, :MIN_POLICY_OBS_DIM])


def test_flow_policy_velocity_field_shape_and_default_linear_initialization() -> None:
    torch.manual_seed(0)
    policy = FlowMatchingPolicy(
        obs_dim=154,
        action_dim=3,
        horizon=2,
        hidden_dims=(16,),
    )
    obs = torch.randn(4, 154)
    noisy_actions = torch.full((4, 6), 3.0)
    time = torch.linspace(0.0, 1.0, 4)

    velocity = policy.velocity_field(obs, noisy_actions, time)

    assert velocity.shape == (4, 6)
    assert not torch.allclose(velocity, torch.zeros(4, 6))


def test_flow_policy_action_transform_squashes_to_robot_scale() -> None:
    policy = FlowMatchingPolicy(
        obs_dim=154,
        action_dim=3,
        horizon=1,
        hidden_dims=(16,),
        action_squash_scale=5.0,
    )
    raw = torch.tensor([[-100.0, -0.25, 100.0]])

    transformed = policy._action_transform(raw)

    assert torch.all(transformed.abs() <= 5.0)
    assert transformed[0, 0].item() == pytest.approx(-5.0)
    assert transformed[0, 2].item() == pytest.approx(5.0)
    assert transformed[0, 1].item() == pytest.approx(float(5.0 * torch.tanh(torch.tensor(-0.25 / 5.0))))


def test_flow_policy_rejects_bad_inputs() -> None:
    policy = FlowMatchingPolicy(obs_dim=154, action_dim=3, horizon=2, hidden_dims=(16,))

    with pytest.raises(ValueError, match="action_squash_scale"):
        FlowMatchingPolicy(obs_dim=154, action_dim=3, action_squash_scale=0.0)
    with pytest.raises(ValueError, match="steps must be >= 1"):
        policy._validate_inputs(torch.zeros(4, 154), torch.zeros(4, 6), steps=0)
    with pytest.raises(ValueError, match="Expected noise dim"):
        policy._validate_inputs(torch.zeros(4, 154), torch.zeros(4, 5), steps=1)
    with pytest.raises(ValueError, match="Observation batch size"):
        policy._validate_inputs(torch.zeros(3, 154), torch.zeros(4, 6), steps=1)
