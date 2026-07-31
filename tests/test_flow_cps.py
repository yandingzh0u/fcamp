from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import torch
from torch import nn

from components.rollout.flow_cps_base import FlowCPSBase
from models.flow_cps_policy import FlowMatchingPolicy, flow_ode_mean


def _policy() -> FlowMatchingPolicy:
    return FlowMatchingPolicy(
        obs_dim=5,
        action_dim=3,
        horizon=4,
        hidden_dims=(24, 16),
        activation="elu",
        action_squash_scale=5.0,
    )


def _eta_raw(eta: float) -> float:
    eta_eps = 1.0e-4
    eta_unit = (float(eta) - eta_eps) / (1.0 - 2.0 * eta_eps)
    return math.log(eta_unit / (1.0 - eta_unit))


def _flow(
    *,
    steps: int = 3,
    rank: int = 2,
    eta: float = 0.35,
    zero_backbone: bool = False,
) -> FlowCPSBase:
    policy = _policy()
    if zero_backbone:
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.zero_()
    policy.cps_diag_raw = nn.Parameter(
        torch.full((steps, policy.action_dim), 0.5)
    )
    policy.cps_lowrank_raw = nn.Parameter(
        0.05 * torch.randn(steps, policy.action_dim, rank)
    )
    policy.cps_eta_raw = nn.Parameter(torch.tensor(_eta_raw(eta)))

    flow = FlowCPSBase.__new__(FlowCPSBase)
    flow.cfg = SimpleNamespace(flow_steps=steps)
    flow._policy = policy
    flow.num_act = policy.action_dim
    flow.horizon_h = policy.horizon
    flow.chunk_dim = policy.chunk_dim
    flow.cps_cov_rank = rank
    return flow


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(value))


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


def test_cps_source_has_no_old_path_noise_transform() -> None:
    source = inspect.getsource(FlowCPSBase)
    assert "_chunk_path_from_innovations" not in source
    assert "_innovations_from_chunk_path" not in source
    assert "_path_from_residual" not in source
    assert "_residual_from_path" not in source
    assert "torch.cumsum" not in source


def test_sampled_cps_density_recomputes_exactly_with_finite_gradients() -> None:
    torch.manual_seed(23)
    flow = _flow()
    policy = flow._policy

    observations = torch.randn(5, policy.obs_dim)
    _, latent_path, sampled_log_probs, _ = flow._sample_cps_path(observations)
    recomputed = flow._recompute_cps_path_stats(
        observations,
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

    named_gradients = {
        name: parameter.grad
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    }
    assert all(gradient is not None for gradient in named_gradients.values())
    assert all(
        bool(torch.isfinite(gradient).all())
        for gradient in named_gradients.values()
        if gradient is not None
    )
    assert (
        policy.cps_diag_raw.grad is not None
        and float(policy.cps_diag_raw.grad.abs().sum()) > 0.0
    )
    assert (
        policy.cps_lowrank_raw.grad is not None
        and float(policy.cps_lowrank_raw.grad.abs().sum()) > 0.0
    )
    assert (
        policy.cps_eta_raw.grad is not None
        and float(policy.cps_eta_raw.grad.abs()) > 0.0
    )
    backbone_gradients = [
        gradient
        for name, gradient in named_gradients.items()
        if not name.startswith("cps_") and gradient is not None
    ]
    assert backbone_gradients
    assert sum(
        float(gradient.abs().sum()) for gradient in backbone_gradients
    ) > 0.0


def test_direct_residual_offsets_have_equal_variance_and_independent_rank_noise() -> None:
    torch.manual_seed(101)
    flow = _flow(steps=1, rank=1, zero_backbone=True)
    policy = flow._policy

    # Make the low-rank realization dominate so an erroneous [B, R] draw
    # broadcast over H would create an unmistakable cross-offset correlation.
    desired_diag = torch.full((policy.action_dim,), 0.05)
    desired_lowrank = torch.tensor([[1.0], [0.6], [-0.8]])
    with torch.no_grad():
        policy.cps_diag_raw[0].copy_(
            _inverse_softplus(desired_diag - 1.0e-4)
        )
        policy.cps_lowrank_raw[0].copy_(desired_lowrank)

    batch = 30_000
    observations = torch.zeros(batch, policy.obs_dim)
    with torch.no_grad():
        final_latent, _, _, _ = flow._sample_cps_path(observations)
        _, noise_coeff, _ = flow._cps_step_coeffs(
            0,
            torch.tensor([1.0, 0.0]),
            final_latent,
        )
    residual = final_latent.view(
        batch,
        policy.horizon,
        policy.action_dim,
    )
    centered = residual - residual.mean(dim=0, keepdim=True)
    frame_variance = centered.square().mean(dim=0).mean(dim=-1)
    expected_variance = noise_coeff.square()
    torch.testing.assert_close(
        frame_variance,
        expected_variance.expand_as(frame_variance),
        atol=0.0,
        rtol=0.05,
    )
    assert float(frame_variance.max() / frame_variance.min()) < 1.05

    dominant_joint = centered[:, :, 0]
    covariance = dominant_joint.transpose(0, 1) @ dominant_joint / batch
    std = torch.sqrt(torch.diagonal(covariance))
    correlation = covariance / (std[:, None] * std[None, :])
    off_diagonal = correlation[
        ~torch.eye(policy.horizon, dtype=torch.bool)
    ]
    assert float(off_diagonal.abs().max()) < 0.04


def test_shared_covariance_is_psd_unit_trace_and_scale_invariant() -> None:
    torch.manual_seed(211)
    flow = _flow(steps=1, rank=2)
    policy = flow._policy
    base_diag = torch.tensor([0.35, 0.8, 1.4])
    base_lowrank = torch.tensor(
        [[0.20, -0.10], [0.35, 0.05], [-0.15, 0.30]]
    )

    normalized_covariances = []
    for factor in (0.4, 3.0):
        with torch.no_grad():
            policy.cps_diag_raw[0].copy_(
                _inverse_softplus(
                    factor * base_diag - 1.0e-4
                )
            )
            policy.cps_lowrank_raw[0].copy_(
                factor * base_lowrank
            )
        (
            diag,
            lowrank,
            covariance,
            cholesky,
            log_diag_cholesky,
            normalized_trace,
        ) = flow._cps_covariance_factors(
            0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        assert diag.shape == (policy.action_dim,)
        assert lowrank.shape == (
            policy.action_dim,
            flow.cps_cov_rank,
        )
        assert covariance.shape == (
            policy.action_dim,
            policy.action_dim,
        )
        torch.testing.assert_close(
            normalized_trace,
            torch.tensor(1.0),
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(
            torch.trace(covariance) / policy.action_dim,
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
        assert bool(torch.isfinite(log_diag_cholesky).all())
        assert float(torch.linalg.eigvalsh(covariance).min()) > 0.0
        normalized_covariances.append(covariance.detach().clone())

    torch.testing.assert_close(
        normalized_covariances[0],
        normalized_covariances[1],
        atol=2.0e-6,
        rtol=2.0e-6,
    )


def test_eta_is_global_identifiable_scale_and_can_update() -> None:
    torch.manual_seed(307)
    flow = _flow(steps=4, eta=0.35)
    policy = flow._policy
    assert policy.cps_eta_raw.ndim == 0
    torch.testing.assert_close(
        flow._cps_eta_value(),
        torch.tensor(0.35),
        atol=1.0e-7,
        rtol=0.0,
    )

    sigma_schedule = torch.linspace(1.0, 0.0, 5)
    reference = torch.zeros(2, policy.chunk_dim)
    covariance_before = flow._cps_covariance_factors(
        0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )[2]

    coefficients: dict[float, tuple[torch.Tensor, torch.Tensor]] = {}
    for eta in (0.2, 0.8):
        with torch.no_grad():
            policy.cps_eta_raw.copy_(torch.tensor(_eta_raw(eta)))
        predicted, noise, recovered_eta = flow._cps_step_coeffs(
            0,
            sigma_schedule,
            reference,
        )
        torch.testing.assert_close(
            recovered_eta,
            torch.tensor(eta),
            atol=1.0e-7,
            rtol=0.0,
        )
        torch.testing.assert_close(
            predicted.square() + noise.square(),
            torch.tensor(1.0),
            atol=1.0e-7,
            rtol=0.0,
        )
        coefficients[eta] = (predicted, noise)

    expected_ratio = math.sin(0.4 * math.pi) / math.sin(0.1 * math.pi)
    torch.testing.assert_close(
        coefficients[0.8][1] / coefficients[0.2][1],
        torch.tensor(expected_ratio),
        atol=1.0e-6,
        rtol=1.0e-6,
    )
    covariance_after = flow._cps_covariance_factors(
        0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )[2]
    torch.testing.assert_close(
        covariance_after,
        covariance_before,
        atol=0.0,
        rtol=0.0,
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=1.0e-3)
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert id(policy.cps_eta_raw) in optimizer_ids
    eta_raw_before = policy.cps_eta_raw.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    _, differentiable_noise, _ = flow._cps_step_coeffs(
        0,
        sigma_schedule,
        reference,
    )
    differentiable_noise.square().backward()
    assert policy.cps_eta_raw.grad is not None
    assert bool(torch.isfinite(policy.cps_eta_raw.grad))
    assert float(policy.cps_eta_raw.grad.abs()) > 0.0
    optimizer.step()
    assert not torch.equal(
        policy.cps_eta_raw.detach(),
        eta_raw_before,
    )


def test_direct_residual_log_prob_matches_manual_shared_gaussian() -> None:
    torch.manual_seed(401)
    flow = _flow(steps=1, rank=2, eta=0.35)
    policy = flow._policy
    innovation = torch.randn(
        4,
        policy.horizon,
        policy.action_dim,
    )
    noise_coeff = torch.tensor(0.37)

    actual = flow._cps_innovation_log_prob(
        innovation,
        noise_coeff,
        0,
    )
    _, _, _, cholesky, log_diag_cholesky, _ = (
        flow._cps_covariance_factors(
            0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    )
    flat = (innovation / noise_coeff).reshape(
        -1,
        policy.action_dim,
    )
    whitened = torch.linalg.solve_triangular(
        cholesky,
        flat.transpose(0, 1),
        upper=False,
    ).transpose(0, 1)
    expected = (
        -0.5
        * (
            whitened.square().sum(dim=-1)
            + policy.action_dim * math.log(2.0 * math.pi)
        )
        - log_diag_cholesky.sum()
        - policy.action_dim * torch.log(noise_coeff)
    ).view(innovation.shape[0], policy.horizon)
    torch.testing.assert_close(
        actual,
        expected,
        atol=2.0e-6,
        rtol=1.0e-6,
    )
