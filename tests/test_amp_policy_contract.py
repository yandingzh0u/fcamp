from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from components.rollout.amp_gaussian_base import AMPGaussianBase
from models.amp_actor_critic import (
    AMPActorCritic,
    DiagonalGaussian,
    GaussianActor,
    ValueMLP,
)


@pytest.mark.parametrize("horizon", [1, 4])
def test_actor_emits_direct_absolute_action_chunk_for_any_horizon(
    horizon: int,
) -> None:
    torch.manual_seed(1)
    actor = GaussianActor(
        observation_dim=5,
        action_dim=3,
        horizon=horizon,
        hidden_dims=(16, 8),
    )
    observations = torch.randn(7, 5)

    mean = actor(observations)
    distribution = actor.distribution(observations)
    noise = torch.randn_like(mean)
    actions = distribution.sample(noise)

    assert mean.shape == (7, horizon, 3)
    assert actions.shape == (7, horizon, 3)
    torch.testing.assert_close(distribution.mean, mean)
    torch.testing.assert_close(
        actions,
        mean + 0.05 * noise,
        atol=1.0e-7,
        rtol=1.0e-6,
    )


def test_default_actor_and_critic_have_official_relu_topology() -> None:
    actor = GaussianActor(11, 6, horizon=4)
    critic = ValueMLP(11)

    actor_linears = [
        module
        for module in actor.trunk
        if isinstance(module, nn.Linear)
    ]
    critic_linears = [
        module
        for module in critic.trunk
        if isinstance(module, nn.Linear)
    ]
    assert [
        (layer.in_features, layer.out_features)
        for layer in actor_linears
    ] == [(11, 1024), (1024, 512)]
    assert [
        (layer.in_features, layer.out_features)
        for layer in critic_linears
    ] == [(11, 1024), (1024, 512)]
    assert all(
        isinstance(module, (nn.Linear, nn.ReLU))
        for module in actor.trunk
    )
    assert all(
        isinstance(module, (nn.Linear, nn.ReLU))
        for module in critic.trunk
    )
    assert actor.mean_head.in_features == 512
    assert actor.mean_head.out_features == 24
    assert critic.value_head.in_features == 512
    assert critic.value_head.out_features == 1


def test_actor_mean_head_initialization_and_fixed_std_are_exact() -> None:
    actor = GaussianActor(5, 3, horizon=4, hidden_dims=(16, 8))

    for module in actor.trunk:
        if isinstance(module, nn.Linear):
            torch.testing.assert_close(
                module.bias,
                torch.zeros_like(module.bias),
            )
    assert float(actor.mean_head.weight.max()) <= 0.01
    assert float(actor.mean_head.weight.min()) >= -0.01
    torch.testing.assert_close(
        actor.mean_head.bias,
        torch.zeros_like(actor.mean_head.bias),
    )
    assert "log_std" in dict(actor.named_buffers())
    assert "log_std" not in dict(actor.named_parameters())
    torch.testing.assert_close(
        actor.log_std.exp(),
        torch.full((4, 3), 0.05),
        atol=1.0e-8,
        rtol=1.0e-6,
    )


def test_value_hidden_biases_match_official_zero_initialization() -> None:
    critic = ValueMLP(5, hidden_dims=(16, 8))

    for module in critic.trunk:
        if isinstance(module, nn.Linear):
            torch.testing.assert_close(
                module.bias,
                torch.zeros_like(module.bias),
            )


def test_actor_has_no_trainable_or_state_dependent_std_path() -> None:
    actor = GaussianActor(5, 3, horizon=4, hidden_dims=(16, 8))

    assert "log_std" in dict(actor.named_buffers())
    assert all("std" not in name for name, _ in actor.named_parameters())
    assert not hasattr(actor, "log_std_head")


def test_gaussian_density_entropy_and_kl_match_torch_reference() -> None:
    torch.manual_seed(7)
    old_mean = torch.randn(3, 4, 2)
    old_log_std = torch.randn(4, 2) * 0.2 - 1.0
    new_mean = torch.randn(3, 4, 2)
    new_log_std = torch.randn(4, 2) * 0.2 - 0.8
    actions = torch.randn(3, 4, 2)
    old = DiagonalGaussian(old_mean, old_log_std)
    new = DiagonalGaussian(new_mean, new_log_std)

    old_ref = torch.distributions.Normal(
        old_mean,
        old_log_std.exp(),
    )
    new_ref = torch.distributions.Normal(
        new_mean,
        new_log_std.exp(),
    )
    expected_log_prob = old_ref.log_prob(actions).sum(dim=-1)
    expected_entropy = old_ref.entropy().sum(dim=-1)
    expected_kl = torch.distributions.kl_divergence(
        old_ref,
        new_ref,
    ).sum(dim=-1)

    torch.testing.assert_close(old.log_prob(actions), expected_log_prob)
    torch.testing.assert_close(old.entropy(), expected_entropy)
    torch.testing.assert_close(old.kl_divergence(new), expected_kl)
    torch.testing.assert_close(
        old.chunk_log_prob(actions),
        expected_log_prob.sum(dim=-1),
    )
    torch.testing.assert_close(
        old.chunk_entropy(),
        expected_entropy.sum(dim=-1),
    )
    torch.testing.assert_close(
        old.chunk_kl_divergence(new),
        expected_kl.sum(dim=-1),
    )


def test_horizon_offsets_are_not_integrated_or_anchored() -> None:
    mean = torch.zeros(1, 4, 1)
    distribution = DiagonalGaussian(
        mean,
        torch.full((4, 1), math.log(0.05)),
    )
    noise = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])

    actions = distribution.sample(noise)

    torch.testing.assert_close(
        actions.flatten(),
        torch.tensor([0.05, 0.10, 0.15, 0.20]),
    )
    assert "previous" not in inspect.signature(
        GaussianActor.forward
    ).parameters
    assert "last_action" not in inspect.signature(
        GaussianActor.forward
    ).parameters


def test_actor_critic_is_only_a_feedforward_relu_mlp() -> None:
    model = AMPActorCritic(5, 3, horizon=4)

    allowed_types = {
        AMPActorCritic,
        GaussianActor,
        ValueMLP,
        nn.Sequential,
        nn.Linear,
        nn.ReLU,
    }
    assert all(type(module) in allowed_types for module in model.modules())


def test_value_mlp_returns_one_scalar_per_observation() -> None:
    critic = ValueMLP(5, hidden_dims=(16, 8))

    values = critic(torch.randn(9, 5))

    assert values.shape == (9, 1)
    assert bool(torch.isfinite(values).all())


class _DummyEnv:
    action_dim = 3
    observation_dim = 5
    # This deliberately differs: the new base must not silently restore the
    # old reference-conditioned privileged critic contract.
    critic_observation_dim = 17
    device = torch.device("cpu")


@pytest.mark.parametrize("horizon", [1, 4])
def test_rollout_base_uses_fixed_std_direct_actions_and_sgd(
    horizon: int,
) -> None:
    cfg = SimpleNamespace(
        horizon=horizon,
        actor_hidden_dims=(1024, 512),
        critic=SimpleNamespace(hidden_dims=(1024, 512)),
        policy_lr=1.0e-4,
        value_lr=2.0e-4,
    )
    base = AMPGaussianBase(cfg, _DummyEnv())
    base.build()
    normalized_observations = torch.randn(6, 5)
    noise = torch.randn(6, horizon, 3)

    sample = base.sample_normalized_action_chunk(
        normalized_observations,
        noise=noise,
    )
    stats = base.recompute_policy_statistics(
        normalized_observations,
        sample.actions.detach(),
        sample.mean.detach(),
        sample.log_std.detach(),
    )

    assert base.horizon == horizon
    assert base.critic_obs_dim == base.actor_obs_dim == 5
    assert isinstance(base.actor_optimizer, torch.optim.SGD)
    assert isinstance(base.critic_optimizer, torch.optim.SGD)
    assert base.actor_optimizer.param_groups[0]["momentum"] == pytest.approx(
        0.9
    )
    assert base.critic_optimizer.param_groups[0]["momentum"] == pytest.approx(
        0.9
    )
    assert sample.actions.shape == (6, horizon, 3)
    assert sample.log_prob.shape == (6, horizon)
    assert sample.entropy.shape == (6, horizon)
    torch.testing.assert_close(
        stats.log_prob,
        sample.log_prob,
        atol=2.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        stats.old_to_new_kl,
        torch.zeros_like(stats.old_to_new_kl),
        atol=1.0e-7,
        rtol=0.0,
    )
