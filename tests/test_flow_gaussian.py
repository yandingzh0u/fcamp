from __future__ import annotations

from types import SimpleNamespace

import math
import pytest
import torch
from torch import nn

from components.rollout.flow_gaussian_base import FlowGaussianBase
from models.flow_chunk_policy import FlowMatchingPolicy, flow_ode_mean


def _policy() -> FlowMatchingPolicy:
    return FlowMatchingPolicy(
        obs_dim=5,
        action_dim=3,
        horizon=4,
        hidden_dims=(24, 16),
        activation="elu",
        action_squash_scale=5.0,
    )


def _future_action_gradient(policy: FlowMatchingPolicy, frame: int) -> float:
    torch.manual_seed(0)
    obs = torch.randn(2, policy.obs_dim)
    noise = torch.randn(2, policy.chunk_dim, requires_grad=True)
    sigmas = torch.linspace(1.0, 0.0, 4, dtype=noise.dtype)
    latent = noise * 0.8
    velocity = policy.velocity_field(obs, latent, torch.ones(2))
    next_latent = flow_ode_mean(velocity, latent, sigmas, 0)
    actions = policy._action_transform(
        next_latent[0],
        prev_action=torch.zeros(policy.action_dim),
    ).view(policy.horizon, policy.action_dim)
    actions[frame].sum().backward()
    future = noise.grad[0, (frame + 1) * policy.action_dim :]
    return float(future.abs().max().item()) if future.numel() else 0.0


def test_flow_ode_step_matches_reference_expression() -> None:
    model_output = torch.tensor([[1.0, -2.0]])
    latents = torch.tensor([[0.25, 0.5]])
    sigmas = torch.tensor([1.0, 0.75])
    expected = latents + model_output * (sigmas[1] - sigmas[0])
    assert torch.equal(flow_ode_mean(model_output, latents, sigmas, 0), expected)


def test_causal_velocity_is_order_sensitive() -> None:
    torch.manual_seed(11)
    policy = _policy()
    obs = torch.randn(2, policy.obs_dim)
    time = torch.full((2,), 0.37)
    chunk = torch.randn(2, policy.horizon, policy.action_dim)
    chunk[:, 0] -= 2.0
    chunk[:, 1] += 2.0
    swapped = chunk.clone()
    swapped[:, [0, 1]] = swapped[:, [1, 0]]

    velocity = policy.velocity_field(obs, chunk.flatten(1), time).view_as(chunk)
    swapped_velocity = policy.velocity_field(
        obs, swapped.flatten(1), time
    ).view_as(chunk)
    delta = (velocity - swapped_velocity).abs().mean(dim=(0, 2))
    assert bool((delta[1:] > 1.0e-5).all())


def test_future_token_does_not_change_earlier_velocities() -> None:
    torch.manual_seed(17)
    policy = _policy()
    obs = torch.randn(2, policy.obs_dim)
    time = torch.full((2,), 0.61)
    chunk = torch.randn(2, policy.horizon, policy.action_dim)
    changed = chunk.clone()
    changed[:, 3] += 10.0 * torch.randn_like(changed[:, 3])

    velocity = policy.velocity_field(obs, chunk.flatten(1), time).view_as(chunk)
    changed_velocity = policy.velocity_field(
        obs, changed.flatten(1), time
    ).view_as(chunk)
    assert torch.equal(velocity[:, :3], changed_velocity[:, :3])
    assert not torch.equal(velocity[:, 3], changed_velocity[:, 3])


def test_residual_actions_have_no_future_gradient_leak() -> None:
    policy = _policy()
    for frame in range(policy.horizon - 1):
        assert _future_action_gradient(policy, frame) <= 1.0e-6


def test_zero_residual_holds_previous_action() -> None:
    policy = _policy()
    previous = torch.tensor([[0.3, -0.7, 1.2]])
    actions = policy._action_transform(
        torch.zeros(1, policy.chunk_dim),
        prev_action=previous,
    ).view(policy.horizon, policy.action_dim)
    torch.testing.assert_close(
        actions,
        previous.expand_as(actions),
        atol=1.0e-5,
        rtol=0.0,
    )


def test_residual_actions_use_symmetric_command_domain() -> None:
    policy = _policy()
    previous = torch.tensor([[0.4, -1.7, 2.1]])
    extreme = torch.tensor(
        [[100.0, -100.0, 100.0] * policy.horizon],
        dtype=torch.float32,
    )
    actions = policy._action_transform(
        extreme,
        prev_action=previous,
    ).view(policy.horizon, policy.action_dim)
    assert bool((actions >= -5.0).all())
    assert bool((actions <= 5.0).all())
    torch.testing.assert_close(
        actions[-1],
        torch.tensor([5.0, -5.0, 5.0]),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_sampled_gaussian_density_recomputes_exactly_with_finite_gradients() -> None:
    torch.manual_seed(23)
    policy = _policy()
    policy.action_path_log_std = nn.Parameter(
        torch.full(
            (policy.horizon, policy.action_dim),
            math.log(0.5),
        )
    )

    flow = FlowGaussianBase.__new__(FlowGaussianBase)
    flow.cfg = SimpleNamespace(flow_steps=3)
    flow._policy = policy
    flow.num_act = policy.action_dim
    flow.horizon_h = policy.horizon
    flow.chunk_dim = policy.chunk_dim
    flow.action_path_std_min = 0.02
    flow.action_path_std_max = 1.5

    observations = torch.randn(5, policy.obs_dim)
    (
        sampled_residual,
        sampled_path,
        sampled_log_probs,
        old_mean_path,
        old_log_std,
    ) = flow._sample_gaussian_action_path(observations)
    recomputed, analytic_kl, entropy = (
        flow._recompute_gaussian_action_path_stats(
        observations,
        sampled_path.detach(),
        old_mean_path.detach(),
        old_log_std.detach(),
        )
    )

    torch.testing.assert_close(
        recomputed,
        sampled_log_probs.detach(),
        atol=3.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        analytic_kl,
        torch.zeros_like(analytic_kl),
        atol=1.0e-7,
        rtol=0.0,
    )
    assert entropy.shape == sampled_log_probs.shape
    recomputed.sum().backward()
    gradients = [
        parameter.grad
        for parameter in policy.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert policy.action_path_log_std.grad is not None
    assert float(policy.action_path_log_std.grad.abs().sum().item()) > 0.0
    torch.testing.assert_close(
        torch.cumsum(
            sampled_residual.view_as(sampled_path),
            dim=1,
        ),
        sampled_path,
    )


def test_gaussian_has_one_trainable_path_std_per_horizon_joint() -> None:
    torch.manual_seed(29)
    policy = _policy()
    policy.action_path_log_std = nn.Parameter(
        torch.full(
            (policy.horizon, policy.action_dim),
            math.log(0.5),
        )
    )
    flow = FlowGaussianBase.__new__(FlowGaussianBase)
    flow.cfg = SimpleNamespace(flow_steps=2)
    flow._policy = policy
    flow.num_act = policy.action_dim
    flow.horizon_h = policy.horizon
    flow.chunk_dim = policy.chunk_dim
    flow.action_path_std_min = 0.02
    flow.action_path_std_max = 1.5

    observations = torch.randn(20_000, policy.obs_dim)
    with torch.no_grad():
        (
            sampled_residual,
            sampled_path,
            _,
            mean_path,
            log_std,
        ) = flow._sample_gaussian_action_path(observations)
    path_noise = sampled_path - mean_path
    empirical = path_noise.std(dim=0, unbiased=False)
    torch.testing.assert_close(
        empirical,
        torch.full_like(empirical, 0.5),
        atol=0.015,
        rtol=0.0,
    )
    per_h_rms = path_noise.square().mean(dim=(0, 2)).sqrt()
    assert float(per_h_rms[-1] / per_h_rms[0]) == pytest.approx(
        1.0,
        abs=0.025,
    )
    torch.testing.assert_close(
        torch.cumsum(
            sampled_residual.view_as(sampled_path),
            dim=1,
        ),
        sampled_path,
        atol=2.0e-6,
        rtol=0.0,
    )
    assert tuple(policy.action_path_log_std.shape) == (
        policy.horizon,
        policy.action_dim,
    )
    torch.testing.assert_close(
        log_std,
        policy.action_path_log_std,
    )


def test_sampled_path_maps_directly_to_each_absolute_action_offset() -> None:
    torch.manual_seed(31)
    policy = _policy()
    policy.action_path_log_std = nn.Parameter(
        torch.full(
            (policy.horizon, policy.action_dim),
            math.log(0.8),
        )
    )
    flow = FlowGaussianBase.__new__(FlowGaussianBase)
    flow.cfg = SimpleNamespace(flow_steps=2)
    flow._policy = policy
    flow.num_act = policy.action_dim
    flow.horizon_h = policy.horizon
    flow.chunk_dim = policy.chunk_dim
    flow.action_path_std_min = 0.02
    flow.action_path_std_max = 1.5

    observations = torch.randn(32, policy.obs_dim)
    previous_action = torch.empty(32, policy.action_dim).uniform_(-3.5, 3.5)
    with torch.no_grad():
        sampled_residual, sampled_path, _, _, _ = (
            flow._sample_gaussian_action_path(observations)
        )
        transformed = policy._action_transform(
            sampled_residual,
            prev_action=previous_action,
        ).view(32, policy.horizon, policy.action_dim)
    scale = policy.action_squash_scale
    initial_path = scale * torch.atanh(previous_action / scale)
    direct = scale * torch.tanh(
        (initial_path[:, None, :] + sampled_path) / scale
    )
    torch.testing.assert_close(
        transformed,
        direct,
        atol=2.0e-6,
        rtol=1.0e-6,
    )
