from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from components.rollout.flow_cps_base import FlowCPSBase
from envs.action_servo import advance_position_servo
from models.flow_cps_policy import FlowMatchingPolicy


class _CPSGeometryEnv:
    """Minimal environment contract needed by the production Flow/CPS core."""

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
        self.command_position_servo_omega = 67.0
        action_mid = torch.linspace(-0.4, 0.4, action_dim)
        action_half_range = torch.linspace(0.75, 1.75, action_dim)
        self.action_low = action_mid - action_half_range
        self.action_high = action_mid + action_half_range

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
        "cps_target_increment_rms": 0.05,
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


def test_flow_cps_samples_one_token_with_one_shared_joint_cholesky() -> None:
    torch.manual_seed(0)
    env = _CPSGeometryEnv(num_envs=4)
    algo = _build_algo(env)
    actor_obs = torch.randn(env.num_envs, env.observation_dim)

    raw_z, mean_z, old_log_prob = algo._sample_final_cps(actor_obs)

    assert raw_z.shape == (env.num_envs, env.action_dim)
    assert mean_z.shape == raw_z.shape
    assert old_log_prob.shape == (env.num_envs,)
    assert torch.isfinite(old_log_prob).all()
    expected_params = env.action_dim * (env.action_dim + 1) // 2
    assert algo._policy.cps_cholesky_raw.shape == (expected_params,)
    assert algo.chunk_dim == algo.horizon_h * env.action_dim
    torch.testing.assert_close(
        algo._recompute_final_cps_log_prob(actor_obs, raw_z),
        old_log_prob,
        atol=2.0e-5,
        rtol=2.0e-5,
    )

    statistics = algo._final_cps_statistics()
    assert (
        abs(statistics["policy/cps_target_increment_rms_target"] - 0.05)
        < 1.0e-6
    )
    assert (
        abs(statistics["policy/cps_target_increment_rms_achieved"] - 0.05)
        < 1.0e-6
    )
    assert statistics["policy/cps_params"] == float(expected_params)


def test_shared_cps_log_prob_preserves_arbitrary_leading_dimensions() -> None:
    torch.manual_seed(1)
    env = _CPSGeometryEnv(num_envs=5)
    algo = _build_algo(env)
    with torch.no_grad():
        offdiagonal = (~algo._policy._cps_diagonal_mask).nonzero(
            as_tuple=False
        ).squeeze(-1)
        algo._policy.cps_cholesky_raw[offdiagonal] = torch.linspace(
            -0.2,
            0.2,
            offdiagonal.numel(),
        )

    mean = torch.randn(2, 4, env.action_dim) * 0.1
    chol, _ = algo._effective_cps_cholesky(
        device=env.device,
        dtype=mean.dtype,
    )
    raw = mean + torch.randn_like(mean) @ chol.transpose(0, 1)
    actual = algo._final_cps_conditional_log_prob(raw, mean)
    expected = torch.distributions.MultivariateNormal(
        loc=mean,
        scale_tril=chol,
    ).log_prob(raw)

    assert actual.shape == (2, 4)
    torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)


def test_recomputed_log_prob_preserves_arbitrary_leading_dimensions() -> None:
    torch.manual_seed(2)
    env = _CPSGeometryEnv(obs_dim=6, action_dim=3)
    algo = _build_algo(env)
    observations = torch.randn(2, 4, env.observation_dim)
    means = algo._flow_mean_raw(
        observations.reshape(-1, env.observation_dim)
    ).reshape(2, 4, env.action_dim)
    chol, _ = algo._effective_cps_cholesky(
        device=env.device,
        dtype=means.dtype,
    )
    raw = means + torch.randn_like(means) @ chol.transpose(0, 1)

    actual = algo._recompute_final_cps_log_prob(observations, raw)
    expected = algo._final_cps_conditional_log_prob(raw, means)

    assert actual.shape == (2, 4)
    torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)


def test_exact_conditional_kl_matches_each_shared_gaussian_context() -> None:
    torch.manual_seed(17)
    env = _CPSGeometryEnv(num_envs=3)
    algo = _build_algo(env)
    action_dim = env.action_dim
    old_mean = torch.randn(3, 4, action_dim) * 0.1
    new_mean = torch.randn(3, 4, action_dim) * 0.1
    old_chol = torch.eye(action_dim) + (
        torch.randn(action_dim, action_dim).tril() * 0.015
    )
    new_chol = torch.eye(action_dim) + (
        torch.randn(action_dim, action_dim).tril() * 0.015
    )
    old_chol.diagonal().clamp_(min=0.5)
    new_chol.diagonal().clamp_(min=0.5)

    actual = algo._final_cps_expected_conditional_kl(
        old_mean,
        new_mean,
        old_chol,
        new_chol,
    )
    covariance_whitened = torch.linalg.solve_triangular(
        new_chol.double(),
        old_chol.double(),
        upper=False,
    )
    mean_whitened = torch.linalg.solve_triangular(
        new_chol.double(),
        (old_mean - new_mean).double().reshape(-1, action_dim).transpose(0, 1),
        upper=False,
    ).transpose(0, 1)
    covariance_term = (
        covariance_whitened.square().sum()
        - float(action_dim)
        + 2.0
        * (
            torch.log(torch.diagonal(new_chol.double())).sum()
            - torch.log(torch.diagonal(old_chol.double())).sum()
        )
    )
    expected = 0.5 * (
        covariance_term + mean_whitened.square().sum(dim=-1)
    )
    expected = expected.reshape_as(actual)

    assert actual.shape == (3, 4)
    assert bool((actual >= 0.0).all())
    torch.testing.assert_close(
        actual.double(),
        expected,
        atol=2.0e-6,
        rtol=2.0e-6,
    )


def test_exact_conditional_kl_is_zero_for_identical_policy() -> None:
    env = _CPSGeometryEnv(num_envs=3)
    algo = _build_algo(env)
    mean = torch.randn(3, 4, env.action_dim)
    chol, _ = algo._effective_cps_cholesky(
        device=env.device,
        dtype=torch.float32,
    )
    actual = algo._final_cps_expected_conditional_kl(
        mean,
        mean,
        chol,
        chol,
    )
    torch.testing.assert_close(actual, torch.zeros_like(actual))


def test_cps_scale_is_defined_only_by_one_target_increment() -> None:
    env = _CPSGeometryEnv()
    algo = _build_algo(env, cps_target_increment_rms=0.037)
    chol, _ = algo._effective_cps_cholesky(
        device=env.device,
        dtype=torch.float32,
    )
    response = algo._policy.cps_target_increment_response
    achieved = torch.sqrt(
        (response @ chol).square().sum() / float(env.action_dim)
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
    env = _CPSGeometryEnv()
    algo = _build_algo(env)
    actor_obs = torch.randn(2, env.observation_dim)
    scale = algo._policy.flow_mean_raw_scale.detach().clone()
    original = algo._policy.velocity_field

    def unit_velocity(observation, flow_token, time):
        del observation, time
        return torch.ones_like(flow_token)

    algo._policy.velocity_field = unit_velocity
    try:
        mean = algo._flow_mean_raw(actor_obs)
    finally:
        algo._policy.velocity_field = original

    assert mean.shape == (2, env.action_dim)
    torch.testing.assert_close(
        mean,
        -torch.full_like(mean, float(scale.item())),
    )
    initial_chol, _ = algo._effective_cps_cholesky(
        device=env.device,
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        torch.diagonal(initial_chol),
        torch.full((env.action_dim,), float(scale.item())),
    )
    assert algo.deterministic_actions(actor_obs).shape == (
        actor_obs.shape[0],
        1,
        env.action_dim,
    )


def test_zero_flow_mean_is_zero_target_increment() -> None:
    env = _CPSGeometryEnv()
    algo = _build_algo(env)
    actor_obs = torch.randn(3, env.observation_dim)
    original = algo._policy.velocity_field

    def zero_velocity(observation, flow_token, time):
        del observation, time
        return torch.zeros_like(flow_token)

    algo._policy.velocity_field = zero_velocity
    try:
        mean_z = algo._flow_mean_raw(actor_obs)
    finally:
        algo._policy.velocity_field = original

    torch.testing.assert_close(mean_z, torch.zeros_like(mean_z))


def test_zero_residual_holds_the_previous_target() -> None:
    env = _CPSGeometryEnv()
    algo = _build_algo(env)
    anchor = torch.stack(
        (env.action_low, algo.target_action_mid, env.action_high)
    )
    target = algo._target_residual_to_target_action(
        torch.zeros_like(anchor),
        anchor,
    )
    torch.testing.assert_close(target, anchor, atol=2.0e-6, rtol=0.0)


def test_residual_has_one_persistent_bounded_target_map() -> None:
    env = _CPSGeometryEnv(action_dim=3)
    algo = _build_algo(env)
    residual = torch.tensor(
        [
            [-10.0, -1.0, 0.0],
            [2.0, -2.0, 0.75],
        ]
    )
    anchor = torch.tensor(
        [
            [-0.4, 0.2, 0.7],
            [0.3, -0.1, -0.6],
        ]
    )

    target_action = algo._target_residual_to_target_action(residual, anchor)
    normalized_anchor = (
        anchor - algo.target_action_mid
    ) / algo.target_action_half_range
    expected = (
        algo.target_action_mid
        + algo.target_action_half_range
        * torch.tanh(torch.atanh(normalized_anchor) + residual)
    )
    torch.testing.assert_close(target_action, expected)
    assert bool((target_action >= env.action_low).all())
    assert bool((target_action <= env.action_high).all())


def test_persistent_target_decoder_is_invariant_to_chunk_partitioning() -> None:
    env = _CPSGeometryEnv(action_dim=3)
    algo = _build_algo(env)
    residuals = torch.tensor(
        [
            [0.30, -0.20, 0.10],
            [-0.15, 0.25, -0.05],
            [0.08, 0.12, -0.18],
            [-0.20, -0.10, 0.22],
            [0.14, -0.16, 0.07],
            [-0.09, 0.04, 0.13],
            [0.11, 0.06, -0.12],
            [-0.07, -0.08, 0.09],
        ]
    )

    def execute(partitions: tuple[int, ...]) -> torch.Tensor:
        assert sum(partitions) == residuals.shape[0]
        action = torch.tensor([[0.20, -0.15, 0.35]])
        rate = torch.tensor([[0.10, -0.05, 0.08]])
        acceleration = torch.tensor([[-0.20, 0.12, -0.04]])
        target = torch.tensor([[-0.10, 0.30, -0.25]])
        trajectory = []
        offset = 0
        for chunk_size in partitions:
            for residual in residuals[offset : offset + chunk_size]:
                target = algo._target_residual_to_target_action(
                    residual.unsqueeze(0),
                    target,
                )
                action, rate, acceleration = advance_position_servo(
                    target,
                    action,
                    rate,
                    acceleration,
                    dt=env.dt,
                    omega=env.command_position_servo_omega,
                )
                trajectory.append(
                    torch.cat(
                        (target, action, rate, acceleration),
                        dim=-1,
                    )
                )
            offset += chunk_size
        return torch.cat(trajectory, dim=0)

    unpartitioned = execute((8,))
    for partitions in (
        (4, 4),
        (2, 4, 2),
        (1, 1, 1, 1, 1, 1, 1, 1),
    ):
        torch.testing.assert_close(
            execute(partitions),
            unpartitioned,
            atol=0.0,
            rtol=0.0,
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


def test_target_increment_response_is_exact_midpoint_jacobian() -> None:
    env = _CPSGeometryEnv()
    algo = _build_algo(env)

    def target_increment(raw_z: torch.Tensor) -> torch.Tensor:
        midpoint = algo.target_action_mid
        return algo._target_residual_to_target_action(
            raw_z.unsqueeze(0),
            midpoint.unsqueeze(0),
        ).squeeze(0) - midpoint

    jacobian = torch.func.jacrev(target_increment)(
        torch.zeros(env.action_dim)
    )
    torch.testing.assert_close(
        algo._policy.cps_target_increment_response,
        jacobian,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_linearized_target_increment_noise_matches_rms_budget() -> None:
    torch.manual_seed(23)
    env = _CPSGeometryEnv(action_dim=3)
    algo = _build_algo(env, cps_target_increment_rms=0.05)
    chol, _ = algo._effective_cps_cholesky(
        device=env.device,
        dtype=torch.float32,
    )
    raw_z = torch.randn(65536, env.action_dim) @ chol.transpose(0, 1)
    target_increment = (
        raw_z * algo.target_action_half_range.unsqueeze(0)
    )
    sampled_rms = torch.sqrt(target_increment.square().mean())
    assert float(sampled_rms.item()) == pytest.approx(0.05, abs=5.0e-4)


def test_policy_is_one_position_free_shared_token_network() -> None:
    torch.manual_seed(11)
    policy = FlowMatchingPolicy(
        obs_dim=5,
        action_dim=3,
        hidden_dims=(24, 16),
        activation="elu",
    )
    parameter_names = tuple(name for name, _ in policy.named_parameters())
    assert not hasattr(policy, "frame_pos_embed")
    assert not hasattr(policy, "causal_cell")
    assert not any("pos" in name or "gru" in name for name in parameter_names)

    observation = torch.randn(2, policy.obs_dim)
    flow_token = torch.randn(2, policy.action_dim)
    time = torch.full((2,), 0.37)
    expected = policy.velocity_field(observation, flow_token, time)
    repeated = policy.velocity_field(
        observation[:, None, :].expand(-1, 4, -1).reshape(-1, policy.obs_dim),
        flow_token[:, None, :].expand(-1, 4, -1).reshape(-1, policy.action_dim),
        time[:, None].expand(-1, 4).reshape(-1),
    ).reshape(2, 4, policy.action_dim)

    assert expected.shape == (2, policy.action_dim)
    for offset in range(4):
        torch.testing.assert_close(repeated[:, offset], expected)


def test_policy_rejects_a_chunk_as_one_flow_token() -> None:
    policy = FlowMatchingPolicy(
        obs_dim=5,
        action_dim=3,
        hidden_dims=(16, 16),
        activation="elu",
    )
    observation = torch.randn(2, policy.obs_dim)
    with pytest.raises(ValueError, match="flow_token"):
        policy._validate_inputs(
            observation,
            torch.randn(2, 4 * policy.action_dim),
            2,
        )


def test_policy_has_no_hidden_action_transform() -> None:
    policy = FlowMatchingPolicy(
        obs_dim=5,
        action_dim=3,
        hidden_dims=(16, 16),
        activation="elu",
    )
    assert not hasattr(policy, "_action_transform")
    assert not hasattr(policy, "action_transform")
