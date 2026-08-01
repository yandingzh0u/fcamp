from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest
import torch

from models.flow_cps_policy import FlowMatchingPolicy, flow_ode_mean
from models.value_critic import ValueCritic


def _policy(
    *,
    steps: int = 3,
    rank: int = 2,
    eta: float = 0.35,
) -> FlowMatchingPolicy:
    return FlowMatchingPolicy(
        obs_dim=5,
        action_dim=3,
        hidden_dims=(24, 16),
        activation="elu",
        action_limit=5.0,
        flow_steps=steps,
        cps_noise_init=eta,
        cps_cov_rank=rank,
    )


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value))


def _eta_raw(eta: float) -> float:
    epsilon = 1.0e-4
    unit = (float(eta) - epsilon) / (1.0 - 2.0 * epsilon)
    return math.log(unit / (1.0 - unit))


def test_flow_ode_step_matches_reference_expression() -> None:
    model_output = torch.tensor([[1.0, -2.0]])
    latents = torch.tensor([[0.25, 0.5]])
    sigmas = torch.tensor([1.0, 0.75])
    expected = latents + model_output * (sigmas[1] - sigmas[0])
    torch.testing.assert_close(
        flow_ode_mean(model_output, latents, sigmas, 0),
        expected,
    )


def test_velocity_field_is_one_joint_action_vector() -> None:
    torch.manual_seed(7)
    policy = _policy()
    observation = torch.randn(4, policy.obs_dim)
    latent = torch.randn(4, policy.action_dim, requires_grad=True)
    time = torch.rand(4)
    velocity = policy.velocity_field(observation, latent, time)
    assert velocity.shape == (4, policy.action_dim)
    velocity.square().mean().backward()
    assert latent.grad is not None
    assert bool(torch.isfinite(latent.grad).all())
    assert float(latent.grad.abs().sum()) > 0.0


def test_absolute_action_transform_is_bounded_and_has_no_anchor_argument() -> None:
    policy = _policy()
    signature = inspect.signature(policy.action_from_latent)
    assert tuple(signature.parameters) == ("latent",)

    zero = torch.zeros(2, policy.action_dim)
    torch.testing.assert_close(
        policy.action_from_latent(zero),
        zero,
        atol=0.0,
        rtol=0.0,
    )
    extreme = torch.tensor([[100.0, -100.0, 0.25]])
    action = policy.action_from_latent(extreme)
    assert action.shape == extreme.shape
    assert bool((action <= policy.action_limit).all())
    assert bool((action >= -policy.action_limit).all())
    assert action[0, 0] > 4.999
    assert action[0, 1] < -4.999


def test_deterministic_action_has_no_time_cache_or_sequence_axis() -> None:
    torch.manual_seed(11)
    policy = _policy(steps=4)
    observation = torch.randn(6, policy.obs_dim)
    action = policy.deterministic_action(observation)
    assert action.shape == (6, policy.action_dim)
    assert bool(torch.isfinite(action).all())
    assert float(action.abs().max()) < policy.action_limit


def test_policy_source_has_no_sequence_or_incremental_decoder_state() -> None:
    source = inspect.getsource(FlowMatchingPolicy).lower()
    for forbidden in (
        "gru",
        "frame_pos",
        "frame_idx",
        "horizon",
        "chunk",
        "prev_action",
        "previous_action",
        "atanh",
        "cumsum",
        "cumulative",
    ):
        assert forbidden not in source
    assert not (
        Path(__file__).parents[1]
        / "components"
        / "rollout"
        / "flow_cps_base.py"
    ).exists()


def test_sampled_path_density_recomputes_exactly_and_all_groups_backpropagate() -> None:
    torch.manual_seed(23)
    policy = _policy(steps=4, rank=2)
    observation = torch.randn(7, policy.obs_dim)
    action, latent_path, sampled_log_probs, diagnostics = policy.sample(
        observation
    )

    assert action.shape == (7, policy.action_dim)
    assert latent_path.shape == (
        7,
        policy.flow_steps + 1,
        policy.action_dim,
    )
    assert sampled_log_probs.shape == (7, policy.flow_steps)
    assert diagnostics["innovation_rms_per_flow_step"].shape == (
        policy.flow_steps,
    )
    assert bool((action.abs() < policy.action_limit).all())
    assert all(bool(torch.isfinite(value).all()) for value in diagnostics.values())

    recomputed = policy.recompute_log_probs(
        observation,
        latent_path.detach(),
    )
    torch.testing.assert_close(
        recomputed,
        sampled_log_probs.detach(),
        atol=3.0e-6,
        rtol=1.0e-6,
    )
    weights = torch.linspace(
        0.7,
        1.3,
        recomputed.numel(),
        dtype=recomputed.dtype,
    ).view_as(recomputed)
    (recomputed * weights).sum().backward()

    gradients = {
        name: parameter.grad
        for name, parameter in policy.named_parameters()
    }
    assert all(gradient is not None for gradient in gradients.values())
    assert all(
        bool(torch.isfinite(gradient).all())
        for gradient in gradients.values()
        if gradient is not None
    )
    assert float(policy.cps_diag_raw.grad.abs().sum()) > 0.0
    assert float(policy.cps_lowrank_raw.grad.abs().sum()) > 0.0
    assert float(policy.cps_eta_raw.grad.abs()) > 0.0
    backbone_gradient = sum(
        float(gradient.abs().sum())
        for name, gradient in gradients.items()
        if name.startswith("velocity_net.") and gradient is not None
    )
    assert backbone_gradient > 0.0


@pytest.mark.parametrize("rank", [0, 2])
def test_covariance_is_positive_definite_with_unit_mean_variance(
    rank: int,
) -> None:
    torch.manual_seed(31)
    policy = _policy(steps=2, rank=rank)
    for step_index in range(policy.flow_steps):
        (
            diagonal,
            low_rank,
            covariance,
            cholesky,
            log_cholesky_diagonal,
            mean_variance,
        ) = policy.covariance_factors(step_index)
        assert diagonal.shape == (policy.action_dim,)
        assert low_rank.shape == (policy.action_dim, rank)
        assert covariance.shape == (policy.action_dim, policy.action_dim)
        torch.testing.assert_close(
            torch.trace(covariance) / policy.action_dim,
            torch.tensor(1.0),
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            mean_variance,
            torch.tensor(1.0),
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            cholesky @ cholesky.transpose(0, 1),
            covariance,
            atol=2.0e-6,
            rtol=1.0e-6,
        )
        assert bool(torch.isfinite(log_cholesky_diagonal).all())
        assert float(torch.linalg.eigvalsh(covariance).min()) > 0.0


def test_covariance_shape_normalization_removes_common_scale() -> None:
    policy = _policy(steps=1, rank=2)
    diagonal = torch.tensor([0.35, 0.8, 1.4])
    low_rank = torch.tensor(
        [[0.20, -0.10], [0.35, 0.05], [-0.15, 0.30]]
    )
    normalized: list[torch.Tensor] = []
    for factor in (0.4, 3.0):
        with torch.no_grad():
            policy.cps_diag_raw[0].copy_(
                _inverse_softplus(factor * diagonal - 1.0e-4)
            )
            policy.cps_lowrank_raw[0].copy_(factor * low_rank)
        normalized.append(policy.covariance_factors(0)[2].detach().clone())
    torch.testing.assert_close(
        normalized[0],
        normalized[1],
        atol=2.0e-6,
        rtol=2.0e-6,
    )


def test_eta_is_the_identifiable_global_scale_and_can_update() -> None:
    policy = _policy(steps=4, eta=0.35)
    assert policy.cps_eta_raw.ndim == 0
    torch.testing.assert_close(
        policy.eta(),
        torch.tensor(0.35),
        atol=1.0e-7,
        rtol=0.0,
    )
    reference = torch.zeros(2, policy.action_dim)
    covariance_before = policy.covariance_factors(0)[2].detach().clone()
    coefficients: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}
    for eta in (0.2, 0.8):
        with torch.no_grad():
            policy.cps_eta_raw.copy_(torch.tensor(_eta_raw(eta)))
        preserved, noise = policy.cps_step_coefficients(0, reference)
        torch.testing.assert_close(
            preserved.square() + noise.square(),
            torch.tensor(1.0),
            atol=1.0e-7,
            rtol=0.0,
        )
        coefficients[eta] = (preserved, noise)
    expected_ratio = math.sin(0.4 * math.pi) / math.sin(0.1 * math.pi)
    torch.testing.assert_close(
        coefficients[0.8][1] / coefficients[0.2][1],
        torch.tensor(expected_ratio),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        policy.covariance_factors(0)[2],
        covariance_before,
        atol=0.0,
        rtol=0.0,
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-3)
    before = policy.cps_eta_raw.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    policy.cps_step_coefficients(0, reference)[1].square().backward()
    assert policy.cps_eta_raw.grad is not None
    assert bool(torch.isfinite(policy.cps_eta_raw.grad))
    assert float(policy.cps_eta_raw.grad.abs()) > 0.0
    optimizer.step()
    assert not torch.equal(policy.cps_eta_raw.detach(), before)


def test_joint_cps_log_prob_matches_manual_gaussian() -> None:
    torch.manual_seed(41)
    policy = _policy(steps=1, rank=2)
    innovation = torch.randn(4, policy.action_dim)
    noise_coefficient = torch.tensor(0.37)
    actual = policy._innovation_log_prob(
        innovation,
        noise_coefficient,
        0,
    )
    _, _, _, cholesky, log_diagonal, _ = policy.covariance_factors(0)
    whitened = torch.linalg.solve_triangular(
        cholesky,
        (innovation / noise_coefficient).transpose(0, 1),
        upper=False,
    ).transpose(0, 1)
    expected = (
        -0.5
        * (
            whitened.square().sum(dim=-1)
            + policy.action_dim * math.log(2.0 * math.pi)
        )
        - log_diagonal.sum()
        - policy.action_dim * torch.log(noise_coefficient)
    )
    torch.testing.assert_close(actual, expected, atol=2.0e-6, rtol=1.0e-6)


def test_value_critic_is_state_only_scalar_and_backpropagates() -> None:
    torch.manual_seed(53)
    critic = ValueCritic(7, (16, 8), "elu")
    observation = torch.randn(6, 7)
    target = torch.randn(6)
    value = critic(observation)
    assert value.shape == (6,)
    loss = (value - target).square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in critic.parameters())
    assert all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in critic.parameters()
        if parameter.grad is not None
    )
