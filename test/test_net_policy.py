from __future__ import annotations

import pytest
import torch

from net import FlowMatchingPolicy
from net.common import MIN_POLICY_OBS_DIM, ResidualMLPBlock, sinusoidal_time_embedding


def test_sinusoidal_time_embedding_shape_values_and_validation() -> None:
    time = torch.tensor([0.0, 1.0])
    embedding = sinusoidal_time_embedding(time, 4)

    assert embedding.shape == (2, 4)
    assert torch.allclose(embedding[0], torch.tensor([0.0, 0.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="embedding_dim must be even"):
        sinusoidal_time_embedding(time, 3)


def test_residual_mlp_block_preserves_shape_and_gradients() -> None:
    block = ResidualMLPBlock(hidden_dim=8)
    x = torch.randn(4, 8, requires_grad=True)

    y = block(x)
    y.sum().backward()

    assert y.shape == x.shape
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_flow_policy_prepares_observation_by_padding_and_truncating() -> None:
    policy = FlowMatchingPolicy(obs_dim=10, action_dim=3, horizon=2, hidden_dim=16, time_embed_dim=8, depth=1)

    short_obs = torch.ones(5, 7)
    long_obs = torch.arange(5 * (MIN_POLICY_OBS_DIM + 3), dtype=torch.float32).view(5, MIN_POLICY_OBS_DIM + 3)

    padded = policy._prepare_observation(short_obs)
    truncated = policy._prepare_observation(long_obs)

    assert padded.shape == (5, MIN_POLICY_OBS_DIM)
    assert torch.allclose(padded[:, :7], short_obs)
    assert torch.allclose(padded[:, 7:], torch.zeros(5, MIN_POLICY_OBS_DIM - 7))
    assert truncated.shape == (5, MIN_POLICY_OBS_DIM)
    assert torch.equal(truncated, long_obs[:, :MIN_POLICY_OBS_DIM])


def test_flow_policy_forward_shape_bounds_and_zero_noise_initial_output() -> None:
    torch.manual_seed(0)
    policy = FlowMatchingPolicy(
        obs_dim=154,
        action_dim=3,
        horizon=2,
        hidden_dim=16,
        time_embed_dim=8,
        depth=1,
        action_limit=0.5,
    )
    obs = torch.randn(4, 154)
    noise = torch.zeros(4, 6)

    action = policy(obs, noise, steps=4)

    assert action.shape == (4, 2, 3)
    assert torch.all(action.abs() <= 0.5 + 1e-6)
    assert torch.allclose(action, torch.zeros_like(action))


def test_flow_policy_rejects_bad_inputs() -> None:
    policy = FlowMatchingPolicy(obs_dim=154, action_dim=3, horizon=2, hidden_dim=16, time_embed_dim=8, depth=1)

    with pytest.raises(ValueError, match="steps must be >= 1"):
        policy(torch.zeros(4, 154), torch.zeros(4, 6), steps=0)
    with pytest.raises(ValueError, match="Expected noise dim"):
        policy(torch.zeros(4, 154), torch.zeros(4, 5), steps=1)
    with pytest.raises(ValueError, match="Observation batch size"):
        policy(torch.zeros(3, 154), torch.zeros(4, 6), steps=1)
