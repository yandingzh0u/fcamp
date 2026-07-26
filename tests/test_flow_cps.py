from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import torch

from components.rollout.flow_cps_base import FlowCPSBase
from envs.action_rate import decode_raw_target_rate
from models.flow_cps_policy import FlowMatchingPolicy


class _CPSGeometryEnv:
    """Minimal environment contract needed by the production actor/CPS code."""

    def __init__(
        self,
        *,
        num_envs: int = 2,
        obs_dim: int = 8,
        critic_dim: int = 8,
        action_dim: int = 3,
    ) -> None:
        self.num_envs = num_envs
        self.observation_dim = obs_dim
        self.critic_observation_dim = critic_dim
        self.action_dim = action_dim
        self.device = torch.device("cpu")
        self.max_episode_steps = 20
        self.dt = 0.02
        self.command_rate_decay = 2.0 ** (-self.dt / 0.08)
        self.command_rate_limit = torch.full((action_dim,), 5.0)

    def adaptive_sampling_stats(self) -> dict[str, float]:
        return {}


class _FlowCPSHarness(FlowCPSBase):
    """Concrete adapter exposing only production Flow/CPS geometry."""

    def collect(self, obs: torch.Tensor) -> dict:
        raise NotImplementedError

    def update(self, rollout: dict, collect_time: float) -> dict:
        raise NotImplementedError

    def log(self, update_idx: int, max_updates: int, metrics: dict) -> None:
        raise NotImplementedError

    def log_banner(self) -> None:
        raise NotImplementedError


def _build_algo(env: _CPSGeometryEnv, **overrides) -> _FlowCPSHarness:
    values = {
        "horizon": 4,
        "rollout_env_steps": 8,
        "flow_steps": 2,
        "cps_physical_rms": 0.05,
        "cps_trainable": True,
        "actor_hidden_dims": (16, 16),
        "critic_hidden_dims": (16, 16),
        "activation": "elu",
        "discount_gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "desired_kl": 0.01,
        "policy_epochs": 2,
        "num_mini_batches": 2,
        "micro_batch_size": 64,
        "value_loss_coef": 1.0,
        "policy_lr": 1.0e-3,
        "value_lr": 1.0e-3,
        "weight_decay": 0.0,
        "critic_weight_decay": 0.0,
        "empirical_normalization": False,
        "init_at_random_ep_len": False,
        "max_grad_norm": 1.0,
        "kl_early_stop_factor": 4.0,
        "advantage_normalization": "global",
    }
    values.update(overrides)
    algo = _FlowCPSHarness(
        cfg=SimpleNamespace(**values),
        env=env,
        simulation_app=None,
    )
    algo.build()
    return algo


def test_flow_cps_base_remains_an_abstract_algorithm_component() -> None:
    assert inspect.isabstract(FlowCPSBase)
    assert not inspect.isabstract(_FlowCPSHarness)


def test_flow_cps_uses_one_final_dense_cholesky_and_exact_recompute() -> None:
    torch.manual_seed(0)
    env = _CPSGeometryEnv(num_envs=4)
    algo = _build_algo(env)
    actor_obs = torch.randn(env.num_envs, env.observation_dim)
    raw_z, mean_z, old_log_probs = algo._sample_final_cps(actor_obs)

    assert torch.isfinite(old_log_probs).all()
    assert old_log_probs.shape == (env.num_envs, algo.horizon_h)
    assert raw_z.shape == (env.num_envs, algo.horizon_h * env.action_dim)
    flat_dim = algo.horizon_h * env.action_dim
    expected_params = flat_dim * (flat_dim + 1) // 2
    assert algo._policy.cps_cholesky_raw.shape == (expected_params,)
    assert not hasattr(algo._policy, "cps_diag_raw")
    assert not hasattr(algo._policy, "cps_lowrank_raw")

    recomputed = algo._recompute_final_cps_log_prob(actor_obs, raw_z)
    torch.testing.assert_close(
        recomputed,
        old_log_probs,
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    assert mean_z.shape == raw_z.shape
    statistics = algo._final_cps_statistics()
    assert abs(statistics["policy/cps_physical_rms_target"] - 0.05) < 1.0e-6
    assert abs(statistics["policy/cps_physical_rms_achieved"] - 0.05) < 1.0e-5
    assert statistics["policy/cps_params"] == float(expected_params)


def test_final_cps_conditional_log_probs_sum_to_joint_density() -> None:
    torch.manual_seed(1)
    env = _CPSGeometryEnv(num_envs=5)
    algo = _build_algo(env)
    with torch.no_grad():
        offdiagonal = (~algo._policy._cps_diagonal_mask).nonzero(
            as_tuple=False
        ).squeeze(-1)
        algo._policy.cps_cholesky_raw[offdiagonal[:8]] = torch.linspace(
            -0.25,
            0.25,
            8,
        )
    actor_obs = torch.randn(5, env.observation_dim)
    raw_z, mean_z, conditional = algo._sample_final_cps(actor_obs)
    chol, _ = algo._effective_cps_cholesky(
        device=raw_z.device,
        dtype=raw_z.dtype,
    )
    covariance = chol @ chol.transpose(0, 1)
    offdiag_covariance = covariance - torch.diag_embed(
        torch.diagonal(covariance)
    )
    assert float(offdiag_covariance.abs().max().item()) > 0.0
    joint = torch.distributions.MultivariateNormal(
        loc=mean_z,
        scale_tril=chol,
    ).log_prob(raw_z)
    torch.testing.assert_close(
        conditional.sum(dim=-1),
        joint,
        atol=2.0e-5,
        rtol=2.0e-5,
    )


def test_exact_conditional_kl_is_nonnegative_and_sums_to_joint_kl() -> None:
    torch.manual_seed(17)
    env = _CPSGeometryEnv(num_envs=3)
    algo = _build_algo(env)
    dim = algo.chunk_dim
    old_mean = torch.randn(3, dim) * 0.1
    new_mean = torch.randn(3, dim) * 0.1
    old_chol = torch.eye(dim) + torch.randn(dim, dim).tril() * 0.015
    new_chol = torch.eye(dim) + torch.randn(dim, dim).tril() * 0.015
    old_chol.diagonal().clamp_(min=0.5)
    new_chol.diagonal().clamp_(min=0.5)

    frame_kl = algo._final_cps_expected_conditional_kl(
        old_mean,
        new_mean,
        old_chol,
        new_chol,
    )
    assert frame_kl.shape == (3, algo.horizon_h)
    assert bool((frame_kl >= 0.0).all())

    covariance_whitened = torch.linalg.solve_triangular(
        new_chol.double(),
        old_chol.double(),
        upper=False,
    )
    mean_whitened = torch.linalg.solve_triangular(
        new_chol.double(),
        (old_mean - new_mean).double().transpose(0, 1),
        upper=False,
    ).transpose(0, 1)
    joint_kl = 0.5 * (
        covariance_whitened.square().sum()
        + mean_whitened.square().sum(dim=-1)
        - float(dim)
        + 2.0
        * (
            torch.log(torch.diagonal(new_chol.double())).sum()
            - torch.log(torch.diagonal(old_chol.double())).sum()
        )
    )
    torch.testing.assert_close(
        frame_kl.double().sum(dim=-1),
        joint_kl,
        atol=2.0e-6,
        rtol=2.0e-6,
    )


def test_exact_conditional_kl_is_zero_for_identical_policy() -> None:
    env = _CPSGeometryEnv(num_envs=3)
    algo = _build_algo(env)
    mean = torch.randn(3, algo.chunk_dim)
    chol, _ = algo._effective_cps_cholesky(
        device=env.device,
        dtype=torch.float32,
    )
    frame_kl = algo._final_cps_expected_conditional_kl(
        mean,
        mean,
        chol,
        chol,
    )
    torch.testing.assert_close(frame_kl, torch.zeros_like(frame_kl))


def test_final_cps_uses_one_aggregate_physical_scale() -> None:
    env = _CPSGeometryEnv()
    algo = _build_algo(env, cps_physical_rms=0.037)
    chol, _ = algo._effective_cps_cholesky(
        device=env.device,
        dtype=torch.float32,
    )
    response = algo._policy.cps_physical_response
    achieved = torch.sqrt(
        (response @ chol).square().sum() / float(3 * env.action_dim)
    )
    torch.testing.assert_close(
        achieved,
        torch.tensor(0.037),
        atol=1.0e-6,
        rtol=1.0e-6,
    )

    first = chol.detach().clone()
    algo._flow_mean_raw(torch.randn(7, env.observation_dim))
    second, _ = algo._effective_cps_cholesky(
        device=env.device,
        dtype=torch.float32,
    )
    torch.testing.assert_close(first, second)


def test_flow_mean_uses_fixed_initial_likelihood_scale() -> None:
    torch.manual_seed(4)
    env = _CPSGeometryEnv()
    algo = _build_algo(env)
    actor_obs = torch.randn(2, env.observation_dim)
    scale = algo._policy.flow_mean_raw_scale.detach().clone()
    original = algo._policy.velocity_field

    def unit_velocity(observation, flow_state, time):
        del observation, time
        return torch.ones_like(flow_state)

    algo._policy.velocity_field = unit_velocity
    try:
        mean = algo._flow_mean_raw(actor_obs)
    finally:
        algo._policy.velocity_field = original

    torch.testing.assert_close(
        mean,
        torch.full_like(mean, -float(scale.item())),
    )
    initial_chol, _ = algo._effective_cps_cholesky(
        device=env.device,
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        torch.diagonal(initial_chol),
        torch.full((algo.chunk_dim,), float(scale.item())),
    )


def test_cps_covariance_shape_has_hard_condition_bound() -> None:
    env = _CPSGeometryEnv()
    algo = _build_algo(env)
    with torch.no_grad():
        algo._policy.cps_cholesky_raw.normal_(mean=0.0, std=10.0)
    shape = algo._raw_cps_cholesky(
        device=env.device,
        dtype=torch.float32,
    )
    singular_values = torch.linalg.svdvals(shape)
    radius = algo._policy.CPS_CHOLESKY_SHAPE_RADIUS
    assert float(singular_values.min().item()) >= 1.0 - radius - 1.0e-5
    assert float(singular_values.max().item()) <= 1.0 + radius + 1.0e-5
    covariance_condition = (
        float(singular_values.max().item())
        / float(singular_values.min().item())
    ) ** 2
    assert (
        covariance_condition
        <= ((1.0 + radius) / (1.0 - radius)) ** 2 + 1.0e-4
    )


def test_cps_physical_response_matches_rate_decoder_jacobian() -> None:
    env = _CPSGeometryEnv()
    algo = _build_algo(env)
    horizon = algo.horizon_h
    action_dim = env.action_dim
    dt = env.dt
    rho = env.command_rate_decay
    rate_limit = env.command_rate_limit

    def physical_features(flat_raw_z: torch.Tensor) -> torch.Tensor:
        raw_z = flat_raw_z.view(horizon, action_dim)
        rate = torch.zeros(action_dim)
        previous_delta = torch.zeros(action_dim)
        deltas = []
        d2 = []
        for frame in range(horizon):
            desired_rate = rate_limit * torch.tanh(raw_z[frame])
            rate = rho * rate + (1.0 - rho) * desired_rate
            delta = dt * rate
            deltas.append(delta)
            d2.append(delta - previous_delta)
            previous_delta = delta
        tail = torch.stack(
            [
                dt * (rho ** (frame + 1)) * rate
                for frame in range(horizon)
            ]
        )
        return torch.cat(
            [
                torch.stack(deltas).reshape(-1) / math.sqrt(float(horizon)),
                torch.stack(d2).reshape(-1) / math.sqrt(float(horizon)),
                tail.reshape(-1) / math.sqrt(float(horizon)),
            ]
        )

    jacobian = torch.func.jacrev(physical_features)(
        torch.zeros(horizon * action_dim)
    )
    torch.testing.assert_close(
        algo._policy.cps_physical_response,
        jacobian,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_true_nonlinear_decoder_noise_respects_physical_rms_budget() -> None:
    torch.manual_seed(23)
    env = _CPSGeometryEnv(action_dim=3)
    env.command_rate_limit = torch.tensor([80.0, 100.0, 120.0])
    algo = _build_algo(env, cps_physical_rms=0.05)
    chol, _ = algo._effective_cps_cholesky(
        device=env.device,
        dtype=torch.float32,
    )
    samples = 32768
    raw_z = (
        torch.randn(samples, algo.chunk_dim) @ chol.transpose(0, 1)
    ).view(samples, algo.horizon_h, env.action_dim)
    action = torch.zeros(samples, env.action_dim)
    rate = torch.zeros_like(action)
    previous_delta = torch.zeros_like(action)
    low = torch.full((env.action_dim,), -5.0)
    high = torch.full((env.action_dim,), 5.0)
    deltas = []
    d2 = []
    for frame in range(algo.horizon_h):
        next_action = decode_raw_target_rate(
            raw_z[:, frame],
            action,
            rate,
            env.command_rate_limit,
            low,
            high,
            control_dt=env.dt,
            decay=env.command_rate_decay,
        )
        delta = next_action - action
        rate = delta / env.dt
        action = next_action
        deltas.append(delta)
        d2.append(delta - previous_delta)
        previous_delta = delta

    tail = []
    for _ in range(algo.horizon_h):
        next_action = decode_raw_target_rate(
            torch.zeros_like(action),
            action,
            rate,
            env.command_rate_limit,
            low,
            high,
            control_dt=env.dt,
            decay=env.command_rate_decay,
        )
        delta = next_action - action
        rate = delta / env.dt
        action = next_action
        tail.append(delta)

    group_rms_sq = [
        torch.stack(group, dim=1).square().mean()
        for group in (deltas, d2, tail)
    ]
    nonlinear_rms = torch.sqrt(sum(group_rms_sq) / 3.0)
    assert float(nonlinear_rms.item()) <= 0.0505
    assert float(nonlinear_rms.item()) >= 0.0475


def _causal_raw_target_grad_leak(
    policy: FlowMatchingPolicy,
    frame_k: int,
    *,
    horizon: int = 4,
    action_dim: int = 3,
) -> float:
    """Return max |grad| of frame k with respect to future Flow tokens."""

    torch.manual_seed(0)
    obs = torch.randn(2, policy.obs_dim)
    noise = torch.randn(2, policy.chunk_dim, requires_grad=True)
    sigma_schedule = torch.linspace(1.0, 0.0, 4, dtype=noise.dtype)
    latent = noise * 0.8
    time = torch.full((2,), 1.0, dtype=noise.dtype)
    velocity = policy.velocity_field(obs, latent, time)
    dt = sigma_schedule[1] - sigma_schedule[0]
    new_latent = latent + velocity * dt
    raw_targets = new_latent[0].view(horizon, action_dim)
    raw_targets[frame_k].sum().backward()
    future = noise.grad[0, (frame_k + 1) * action_dim :]
    return (
        float(future.abs().max().item())
        if future.numel() > 0
        else 0.0
    )


def test_causal_raw_target_no_future_gradient_leak() -> None:
    policy = FlowMatchingPolicy(
        obs_dim=5,
        action_dim=3,
        horizon=4,
        hidden_dims=(16, 16),
        activation="elu",
    )
    for frame in range(3):
        leak = _causal_raw_target_grad_leak(policy, frame)
        assert leak <= 1.0e-6


def test_causal_velocity_is_order_sensitive_after_swapped_prefix() -> None:
    torch.manual_seed(11)
    horizon = 4
    action_dim = 3
    policy = FlowMatchingPolicy(
        obs_dim=5,
        action_dim=action_dim,
        horizon=horizon,
        hidden_dims=(24, 16),
        activation="elu",
    )
    obs = torch.randn(2, policy.obs_dim)
    time = torch.full((2,), 0.37)
    chunk = torch.randn(2, horizon, action_dim)
    chunk[:, 0] -= 2.0
    chunk[:, 1] += 2.0
    swapped = chunk.clone()
    swapped[:, [0, 1]] = swapped[:, [1, 0]]

    velocity = policy.velocity_field(
        obs,
        chunk.reshape(2, -1),
        time,
    ).view(2, horizon, action_dim)
    swapped_velocity = policy.velocity_field(
        obs,
        swapped.reshape(2, -1),
        time,
    ).view(2, horizon, action_dim)
    mean_abs_delta = (velocity - swapped_velocity).abs().mean(dim=(0, 2))
    for frame in range(1, horizon):
        assert float(mean_abs_delta[frame].item()) > 1.0e-5


def test_causal_velocity_future_token_does_not_change_earlier_frames() -> None:
    torch.manual_seed(17)
    horizon = 4
    action_dim = 3
    policy = FlowMatchingPolicy(
        obs_dim=5,
        action_dim=action_dim,
        horizon=horizon,
        hidden_dims=(24, 16),
        activation="elu",
    )
    obs = torch.randn(2, policy.obs_dim)
    time = torch.full((2,), 0.61)
    chunk = torch.randn(2, horizon, action_dim)
    future_changed = chunk.clone()
    future_changed[:, 3] += 10.0 * torch.randn_like(future_changed[:, 3])

    velocity = policy.velocity_field(
        obs,
        chunk.reshape(2, -1),
        time,
    ).view(2, horizon, action_dim)
    changed_velocity = policy.velocity_field(
        obs,
        future_changed.reshape(2, -1),
        time,
    ).view(2, horizon, action_dim)
    assert torch.equal(velocity[:, :3], changed_velocity[:, :3])
    assert not torch.equal(velocity[:, 3], changed_velocity[:, 3])


def test_policy_output_is_frame_major_raw_target_rate() -> None:
    policy = FlowMatchingPolicy(
        obs_dim=5,
        action_dim=3,
        horizon=4,
        hidden_dims=(16, 16),
        activation="elu",
    )
    observation = torch.randn(2, policy.obs_dim)
    flow_state = torch.arange(24, dtype=torch.float32).view(2, 12)
    velocity = policy.velocity_field(
        observation,
        flow_state,
        torch.full((2,), 0.5),
    )
    assert velocity.shape == (2, 4 * 3)
    assert velocity.view(2, 4, 3).shape == (2, 4, 3)
    assert not hasattr(policy, "_action_transform")
    assert not hasattr(policy, "action_transform")
