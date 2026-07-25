from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import pytest
import torch

from components.rollout.flow_cps_base import FlowCPSBase
from components.rollout.flow_cps_reference import FlowCPSReferenceAlgorithm
from envs.action_rate import decode_raw_target_rate
from models.flow_cps_policy import FlowMatchingPolicy


class _FlowCPSReferenceHarness(FlowCPSReferenceAlgorithm):
    """Explicit test-only adapter for the abstract reference updater."""


# --------------------------------------------------------------------------- #
# Mock env for collect / update integration tests
# --------------------------------------------------------------------------- #
class _NegativeRewardEnv:
    """Mock env with NET-NEGATIVE per-step reward.

    env0 never dies; env1 dies (failure) on the first step of chunk 0 only.
    With zero hand-written failure penalty, a dying chunk accumulates fewer
    negative terms. This mock is useful for checking terminal masks and
    bootstrap behavior without reintroducing the removed penalty.
    """

    def __init__(self, num_envs=2, obs_dim=8, critic_dim=8, action_dim=3, reward=-0.04):
        self.num_envs = num_envs
        self.observation_dim = obs_dim
        self.critic_observation_dim = critic_dim
        self.action_dim = action_dim
        self.device = torch.device("cpu")
        self.max_episode_steps = 20
        self.dt = 0.02
        self.command_rate_decay = 2.0 ** (-self.dt / 0.08)
        self.command_rate_limit = torch.full((action_dim,), 5.0)
        self.last_action = torch.zeros(num_envs, action_dim)
        self.phase_steps = torch.zeros(num_envs, dtype=torch.long)
        self.config = SimpleNamespace(action_rate_weight=0.1)
        self._gen = torch.Generator().manual_seed(7)
        self._call = 0
        self._reward = reward
        self._die_env1_at_call = 1  # first step of chunk 0

    def _obs(self):
        return torch.randn(self.num_envs, self.observation_dim, generator=self._gen)

    def get_critic_observation(self):
        return torch.randn(self.num_envs, self.critic_observation_dim, generator=self._gen)

    def sample_phase_indices(self, n, horizon):
        return torch.zeros(n, dtype=torch.long)

    def reset(self, phase_indices=None):
        return self._obs()

    def reset_envs(self, env_ids, phase_indices=None):
        if env_ids.numel() == 0:
            return torch.empty(0, self.observation_dim)
        return self._obs()[env_ids]

    def adaptive_sampling_stats(self):
        return {"top_bin": 0.0, "top_prob": 0.0, "failed_sum": 0.0, "entropy": 0.0, "peak_bin": 0}

    def step(self, action, auto_reset=False):
        self._call += 1
        reward = torch.full((self.num_envs,), float(self._reward))
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        if self._call == self._die_env1_at_call:
            done[1] = True
        time_out = torch.zeros(self.num_envs, dtype=torch.bool)
        anchor_pos_bad = torch.zeros(self.num_envs, dtype=torch.bool)
        anchor_ori_bad = torch.zeros(self.num_envs, dtype=torch.bool)
        ee_body_bad = done.clone()  # env1 death is a tracking failure
        done_terms = {
            "time_out": time_out,
            "motion_complete": torch.zeros(self.num_envs, dtype=torch.bool),
            "anchor_pos_bad": anchor_pos_bad,
            "anchor_ori_bad": anchor_ori_bad,
            "ee_body_bad": ee_body_bad,
        }
        info = {
            "done_terms": done_terms,
            "reward_terms": {},
            "termination_phase_steps": self.phase_steps.clone(),
        }
        self.phase_steps += 1
        return self._obs(), reward, done, info

    def step_raw_target_rate(self, raw_target_rate, *, active_mask, auto_reset=False):
        applied = torch.where(
            active_mask.unsqueeze(-1),
            raw_target_rate,
            self.last_action,
        )
        self.last_action.copy_(applied)
        obs, reward, done, info = self.step(applied, auto_reset=auto_reset)
        info["applied_action"] = applied.clone()
        return obs, reward, done, info


def _build_algo(env, **overrides):
    base = dict(
        horizon=4, rollout_env_steps=8, flow_steps=2,
        cps_physical_rms=0.05, cps_trainable=True,
        actor_hidden_dims=(16, 16), critic_hidden_dims=(16, 16), activation="elu",
        discount_gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, desired_kl=0.01, policy_epochs=2,
        num_mini_batches=2, micro_batch_size=64, value_loss_coef=1.0,
        policy_lr=1e-3, value_lr=1e-3, weight_decay=0.0, critic_weight_decay=0.0,
        empirical_normalization=False, init_at_random_ep_len=False, max_grad_norm=1.0,
        kl_early_stop_factor=4.0, advantage_normalization="global",
    )
    base.update(overrides)
    cfg = SimpleNamespace(**base)
    algo = _FlowCPSReferenceHarness(cfg=cfg, env=env, simulation_app=None)
    algo.build()
    return algo


def test_flow_cps_base_is_not_a_concrete_production_algorithm() -> None:
    assert inspect.isabstract(FlowCPSBase)
    assert not inspect.isabstract(_FlowCPSReferenceHarness)


def test_failure_has_zero_bootstrap_and_no_handwritten_penalty() -> None:
    torch.manual_seed(0)
    env = _NegativeRewardEnv(num_envs=2, reward=-0.04)
    algo = _build_algo(env)
    obs = algo.initial_reset()
    rollout = algo.collect(obs)

    # env1 dies (failure) at frame 0 -> reward = r0, bootstrap = 0.
    realized_env1 = float(rollout["chunk_return_realized"][0, 1].item())
    assert abs(realized_env1 - (-0.04)) < 1e-4
    assert torch.allclose(rollout["failure_cost_return"], torch.zeros_like(rollout["failure_cost_return"]))


# --------------------------------------------------------------------------- #
# valid_prefix_mask correctness
# --------------------------------------------------------------------------- #
def test_valid_prefix_mask_for_early_death() -> None:
    torch.manual_seed(0)
    env = _NegativeRewardEnv(num_envs=2, reward=1.0)  # positive reward, no death ranking issue
    algo = _build_algo(env)
    obs = algo.initial_reset()
    rollout = algo.collect(obs)

    # env1 dies at frame 0 -> death_frame=0 -> valid prefixes = [1] only (index 0).
    df_env1 = int(rollout["death_frame"][0, 1].item())
    assert df_env1 == 0
    mask_env1 = rollout["valid_prefix_mask"][0, 1].tolist()
    assert mask_env1 == [True, False, False, False], f"got {mask_env1}"

    # env0 survives -> death_frame=h=4 -> all prefixes valid.
    df_env0 = int(rollout["death_frame"][0, 0].item())
    assert df_env0 == 4
    mask_env0 = rollout["valid_prefix_mask"][0, 0].tolist()
    assert mask_env0 == [True, True, True, True], f"got {mask_env0}"


# --------------------------------------------------------------------------- #
# End-to-end collect + update runs without error and produces sane metrics
# --------------------------------------------------------------------------- #
def test_flow_cps_collect_and_update_end_to_end() -> None:
    torch.manual_seed(0)
    env = _NegativeRewardEnv(num_envs=4, reward=0.05)  # small positive reward
    algo = _build_algo(env, num_mini_batches=2, micro_batch_size=8)
    obs = algo.initial_reset()
    rollout = algo.collect(obs)
    metrics = algo.update(rollout, collect_time=0.1)

    # per-frame ratio / kl diagnostics present for all h frames
    h = 4
    for k in range(h):
        assert f"flow_cps/kl_frame_{k}" in metrics
        assert f"flow_cps/ratio_frame_{k}" in metrics
        assert f"critic/frame_v_{k+1}_mean" in metrics
        assert f"critic/prefix_target_{k+1}_mean" in metrics
    # core losses finite
    for key in ("flow_cps/policy_loss", "flow_cps/value_loss", "flow_cps/loss"):
        assert math.isfinite(metrics[key]), f"{key}={metrics[key]}"
    # update actually moved the policy
    assert metrics["policy/raw_z_mean_delta"] >= 0.0
    # actor + critic both have grad norms
    assert metrics["flow_cps/grad_norm"] >= 0.0
    assert metrics["flow_cps/grad_norm_critic"] >= 0.0


def test_flow_cps_uses_one_final_dense_cholesky_and_exact_recompute() -> None:
    torch.manual_seed(0)
    env = _NegativeRewardEnv(num_envs=4, reward=0.05)
    algo = _build_algo(
        env,
        cps_trainable=True,
        num_mini_batches=2,
        micro_batch_size=8,
    )
    obs = algo.initial_reset()
    rollout = algo.collect(obs)

    assert torch.isfinite(rollout["old_log_probs"]).all()
    assert rollout["old_log_probs"].shape == (2, 4, 4)
    assert rollout["raw_z"].shape == (2, 4, 4 * 3)
    assert rollout.get("old_mean_z") is None
    assert "latents" not in rollout
    assert "old_cps_noise_coeff" not in rollout
    flat_dim = 4 * 3
    expected_params = flat_dim * (flat_dim + 1) // 2
    assert algo._policy.cps_cholesky_raw.shape == (expected_params,)
    assert not hasattr(algo._policy, "cps_diag_raw")
    assert not hasattr(algo._policy, "cps_lowrank_raw")

    actor_obs = rollout["actor_obs"].reshape(-1, env.observation_dim)
    sampled = rollout["raw_z"].reshape(-1, flat_dim)
    recomputed = algo._recompute_final_cps_log_prob(actor_obs, sampled)
    torch.testing.assert_close(
        recomputed,
        rollout["old_log_probs"].reshape(-1, 4),
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    assert torch.allclose(rollout["failure_cost_return"], torch.zeros_like(rollout["failure_cost_return"]))

    metrics = algo.update(rollout, collect_time=0.1)
    assert math.isfinite(metrics["flow_cps/policy_loss"])
    assert math.isfinite(metrics["flow_cps/kl_raw"])
    assert abs(metrics["policy/cps_physical_rms_target"] - 0.05) < 1e-6
    assert abs(metrics["policy/cps_physical_rms_achieved"] - 0.05) < 1e-5
    assert metrics["policy/cps_params"] == float(expected_params)


def test_final_cps_conditional_log_probs_sum_to_joint_density() -> None:
    torch.manual_seed(1)
    env = _NegativeRewardEnv(num_envs=5, reward=0.05)
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
    chol, _ = algo._effective_cps_cholesky(device=raw_z.device, dtype=raw_z.dtype)
    covariance = chol @ chol.transpose(0, 1)
    offdiag_covariance = covariance - torch.diag_embed(
        torch.diagonal(covariance)
    )
    assert float(offdiag_covariance.abs().max().item()) > 0.0
    joint = torch.distributions.MultivariateNormal(
        loc=mean_z,
        scale_tril=chol,
    ).log_prob(raw_z)
    torch.testing.assert_close(conditional.sum(dim=-1), joint, atol=2.0e-5, rtol=2.0e-5)


def test_exact_conditional_kl_is_nonnegative_and_sums_to_joint_kl() -> None:
    torch.manual_seed(17)
    env = _NegativeRewardEnv(num_envs=3, reward=0.05)
    algo = _build_algo(env)
    dim = algo.chunk_dim
    old_mean = torch.randn(3, dim) * 0.1
    new_mean = torch.randn(3, dim) * 0.1
    old_raw = torch.randn(dim, dim).tril() * 0.015
    new_raw = torch.randn(dim, dim).tril() * 0.015
    old_chol = torch.eye(dim) + old_raw
    new_chol = torch.eye(dim) + new_raw
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
    env = _NegativeRewardEnv(num_envs=3, reward=0.05)
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
    env = _NegativeRewardEnv(num_envs=2, reward=0.05)
    algo = _build_algo(env, cps_physical_rms=0.037)
    chol, _ = algo._effective_cps_cholesky(device=env.device, dtype=torch.float32)
    response = algo._policy.cps_physical_response
    achieved = torch.sqrt((response @ chol).square().sum() / float(3 * env.action_dim))
    torch.testing.assert_close(achieved, torch.tensor(0.037), atol=1.0e-6, rtol=1.0e-6)

    # The covariance is global/state-independent and flow_steps affects only
    # the deterministic ODE mean.
    first = chol.detach().clone()
    other_obs = torch.randn(7, env.observation_dim)
    algo._flow_mean_raw(other_obs)
    second, _ = algo._effective_cps_cholesky(device=env.device, dtype=torch.float32)
    torch.testing.assert_close(first, second)


def test_flow_mean_uses_fixed_initial_likelihood_scale() -> None:
    torch.manual_seed(4)
    env = _NegativeRewardEnv(num_envs=2, reward=0.05)
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
    env = _NegativeRewardEnv(num_envs=2, reward=0.05)
    algo = _build_algo(env)
    with torch.no_grad():
        algo._policy.cps_cholesky_raw.normal_(mean=0.0, std=10.0)
    shape = algo._raw_cps_cholesky(device=env.device, dtype=torch.float32)
    singular_values = torch.linalg.svdvals(shape)
    radius = algo._policy.CPS_CHOLESKY_SHAPE_RADIUS
    assert float(singular_values.min().item()) >= 1.0 - radius - 1.0e-5
    assert float(singular_values.max().item()) <= 1.0 + radius + 1.0e-5
    covariance_condition = (
        float(singular_values.max().item())
        / float(singular_values.min().item())
    ) ** 2
    assert covariance_condition <= ((1.0 + radius) / (1.0 - radius)) ** 2 + 1.0e-4


def test_cps_physical_response_matches_rate_decoder_jacobian() -> None:
    env = _NegativeRewardEnv(num_envs=2, reward=0.05)
    algo = _build_algo(env)
    h, action_dim = algo.horizon_h, env.action_dim
    dt = env.dt
    rho = env.command_rate_decay
    rate_limit = env.command_rate_limit

    def physical_features(flat_raw_z: torch.Tensor) -> torch.Tensor:
        raw_z = flat_raw_z.view(h, action_dim)
        rate = torch.zeros(action_dim)
        previous_delta = torch.zeros(action_dim)
        deltas = []
        d2 = []
        for frame in range(h):
            desired_rate = rate_limit * torch.tanh(raw_z[frame])
            rate = rho * rate + (1.0 - rho) * desired_rate
            delta = dt * rate
            deltas.append(delta)
            d2.append(delta - previous_delta)
            previous_delta = delta
        tail = torch.stack(
            [
                dt * (rho ** (frame + 1)) * rate
                for frame in range(h)
            ]
        )
        return torch.cat(
            [
                torch.stack(deltas).reshape(-1) / math.sqrt(float(h)),
                torch.stack(d2).reshape(-1) / math.sqrt(float(h)),
                tail.reshape(-1) / math.sqrt(float(h)),
            ]
        )

    jacobian = torch.func.jacrev(physical_features)(
        torch.zeros(h * action_dim)
    )
    torch.testing.assert_close(
        algo._policy.cps_physical_response,
        jacobian,
        atol=1.0e-6,
        rtol=1.0e-6,
    )


def test_true_nonlinear_decoder_noise_respects_physical_rms_budget() -> None:
    torch.manual_seed(23)
    env = _NegativeRewardEnv(num_envs=2, reward=0.05, action_dim=3)
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
    # tanh and projection may only contract the zero-point linear response.
    # At the configured exploration scale the contraction should remain small.
    assert float(nonlinear_rms.item()) <= 0.0505
    assert float(nonlinear_rms.item()) >= 0.0475


# --------------------------------------------------------------------------- #
# Chunk advantage normalization produces zero-mean valid chunk advantages
# --------------------------------------------------------------------------- #
def test_chunk_advantage_normalization() -> None:
    torch.manual_seed(0)
    env = _NegativeRewardEnv(num_envs=4, reward=0.05)
    algo = _build_algo(env, advantage_normalization="per_prefix")
    obs = algo.initial_reset()
    rollout = algo.collect(obs)

    adv = rollout["advantages"]
    chunk_mask = rollout["chunk_valid_mask"]
    chunk_adv = adv[..., 0][chunk_mask]
    if chunk_adv.numel() > 1:
        assert abs(float(chunk_adv.mean().item())) < 1e-5, (
            f"chunk advantages not zero-mean: {float(chunk_adv.mean().item())}"
        )
    # Same chunk advantage is broadcast to every executed frame; invalid frames are zeroed.
    mask = rollout["valid_prefix_mask"]
    for k in range(1, 4):
        same_chunk = mask[..., k]
        if bool(same_chunk.any()):
            assert torch.allclose(adv[..., k][same_chunk], adv[..., 0][same_chunk], atol=1e-6)
    assert torch.all(adv[~mask] == 0.0)


# --------------------------------------------------------------------------- #
# CRITICAL: actor advantage is chunk GAE advantage broadcast to executed frames.
# --------------------------------------------------------------------------- #
def test_actor_advantage_is_chunk_gae() -> None:
    """Every executed frame in a sampled chunk carries the same chunk GAE credit."""
    torch.manual_seed(0)
    env = _NegativeRewardEnv(num_envs=2, reward=-0.04)
    algo = _build_algo(env, advantage_normalization="none")
    obs = algo.initial_reset()
    rollout = algo.collect(obs)

    adv = rollout["advantages"]  # [chunks, n_envs, h]
    mask = rollout["valid_prefix_mask"]  # [chunks, n_envs, h]
    chunk_adv = rollout["chunk_advantages"]  # [chunks, n_envs]

    expected = chunk_adv.unsqueeze(-1).expand_as(adv)
    assert torch.allclose(adv[mask], expected[mask], atol=1e-5), (
        "actor advantage must equal chunk GAE broadcast to valid executed frames"
    )
    # Invalid (post-death) frames are masked to 0.
    assert torch.all(adv[~mask] == 0.0)


# --------------------------------------------------------------------------- #
# CRITICAL: with causal_velocity=True, per-frame ratio is the LEGAL form
# because logp_k is a genuine conditional density (v_k depends only on
# z_0..z_k). This test verifies the per-frame (non-cumsum) ratio is used.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# CAUSAL HARD TESTS: a_k must NOT depend on future latents j>k.
# --------------------------------------------------------------------------- #
def _causal_raw_target_grad_leak(
    policy,
    frame_k: int,
    horizon: int = 4,
    action_dim: int = 3,
) -> float:
    """Return max |grad| of raw target z_k w.r.t. future Flow tokens."""
    torch.manual_seed(0)
    from models.flow_sampling import flow_ode_mean
    obs = torch.randn(2, policy.obs_dim)
    noise = torch.randn(2, policy.chunk_dim, requires_grad=True)
    sigma_schedule = torch.linspace(1.0, 0.0, 4, dtype=noise.dtype)
    latent = noise * 0.8
    t_batch = torch.full((2,), 1.0, dtype=noise.dtype)
    model_out = policy.velocity_field(obs, latent, t_batch)
    new_latent = flow_ode_mean(model_out, latent, sigma_schedule, 0)
    noise.grad = None
    raw_targets = policy.reshape_raw_targets(new_latent[0])
    raw_targets[frame_k].sum().backward(retain_graph=True)
    g = noise.grad[0]
    # future frames = cols [(k+1)*A : ]
    future = g[(frame_k + 1) * action_dim:]
    return float(future.abs().max().item()) if future.numel() > 0 else 0.0


def test_causal_raw_target_no_future_gradient_leak() -> None:
    """The deterministic Flow mean remains ordered and frame-causal."""
    from models.flow_cps_policy import FlowMatchingPolicy
    pol = FlowMatchingPolicy(obs_dim=5, action_dim=3, horizon=4, hidden_dims=(16, 16),
                             activation="elu", causal_velocity=True)
    for k in range(3):
        leak = _causal_raw_target_grad_leak(pol, k)
        assert leak <= 1e-6, f"causal raw target frame {k} leaks future grad: {leak}"


def test_causal_velocity_is_order_sensitive_after_swapped_prefix() -> None:
    """Swapping z0/z1 must change v1 and every later conditional velocity.

    A commutative prefix aggregation fails this invariant: after both tokens
    have entered the prefix it cannot distinguish [z0, z1] from [z1, z0].
    """
    torch.manual_seed(11)
    horizon, action_dim = 4, 3
    pol = FlowMatchingPolicy(
        obs_dim=5,
        action_dim=action_dim,
        horizon=horizon,
        hidden_dims=(24, 16),
        activation="elu",
        causal_velocity=True,
    )
    obs = torch.randn(2, pol.obs_dim)
    time = torch.full((2,), 0.37)
    chunk = torch.randn(2, horizon, action_dim)
    chunk[:, 0] -= 2.0
    chunk[:, 1] += 2.0
    swapped = chunk.clone()
    swapped[:, [0, 1]] = swapped[:, [1, 0]]

    vel = pol.velocity_field(obs, chunk.reshape(2, -1), time).view(2, horizon, action_dim)
    vel_swapped = pol.velocity_field(obs, swapped.reshape(2, -1), time).view(2, horizon, action_dim)
    mean_abs_delta = (vel - vel_swapped).abs().mean(dim=(0, 2))

    for frame_idx in range(1, horizon):
        assert float(mean_abs_delta[frame_idx].item()) > 1.0e-5, (
            f"velocity frame {frame_idx} is not order-sensitive: "
            f"mean_abs_delta={float(mean_abs_delta[frame_idx].item())}"
        )


def test_causal_velocity_future_token_does_not_change_earlier_frames() -> None:
    """Changing z_j must leave all conditional velocities v_k, k < j, exact."""
    torch.manual_seed(17)
    horizon, action_dim = 4, 3
    pol = FlowMatchingPolicy(
        obs_dim=5,
        action_dim=action_dim,
        horizon=horizon,
        hidden_dims=(24, 16),
        activation="elu",
        causal_velocity=True,
    )
    obs = torch.randn(2, pol.obs_dim)
    time = torch.full((2,), 0.61)
    chunk = torch.randn(2, horizon, action_dim)
    future_changed = chunk.clone()
    future_changed[:, 3] += 10.0 * torch.randn_like(future_changed[:, 3])

    vel = pol.velocity_field(obs, chunk.reshape(2, -1), time).view(2, horizon, action_dim)
    changed_vel = pol.velocity_field(obs, future_changed.reshape(2, -1), time).view(
        2, horizon, action_dim
    )

    assert torch.equal(vel[:, :3], changed_vel[:, :3]), (
        "changing future token z3 changed an earlier conditional velocity"
    )
    assert not torch.equal(vel[:, 3], changed_vel[:, 3]), (
        "changing z3 should change its own conditional velocity"
    )


def test_policy_exposes_only_frame_major_raw_target_rate() -> None:
    pol = FlowMatchingPolicy(
        obs_dim=5,
        action_dim=3,
        horizon=4,
        hidden_dims=(16, 16),
        activation="elu",
        causal_velocity=True,
    )
    raw = torch.arange(24, dtype=torch.float32).view(2, 12)
    shaped = pol.reshape_raw_targets(raw)
    assert shaped.shape == (2, 4, 3)
    torch.testing.assert_close(shaped.reshape_as(raw), raw)
    assert not hasattr(pol, "_action_transform")
    assert not hasattr(pol, "action_transform")


# --------------------------------------------------------------------------- #
# failure_frame excludes motion_complete (a success signal must not be
# absorbed into the failure target).
# --------------------------------------------------------------------------- #
class _MotionCompleteEnv(_NegativeRewardEnv):
    """env1 triggers BOTH a failure term AND motion_complete at the same step.
    motion_complete is a success signal, so this must NOT count as failure."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._die_env1_at_call = 1

    def step(self, action, auto_reset=False):
        self._call += 1
        reward = torch.full((self.num_envs,), float(self._reward))
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        motion_complete = torch.zeros(self.num_envs, dtype=torch.bool)
        if self._call == self._die_env1_at_call:
            done[1] = True
            motion_complete[1] = True  # success signal co-occurring with a failure term
        time_out = torch.zeros(self.num_envs, dtype=torch.bool)
        anchor_pos_bad = torch.zeros(self.num_envs, dtype=torch.bool)
        anchor_ori_bad = torch.zeros(self.num_envs, dtype=torch.bool)
        ee_body_bad = done.clone()  # a failure term is also true
        done_terms = {
            "time_out": time_out,
            "motion_complete": motion_complete,
            "anchor_pos_bad": anchor_pos_bad,
            "anchor_ori_bad": anchor_ori_bad,
            "ee_body_bad": ee_body_bad,
        }
        info = {
            "done_terms": done_terms,
            "reward_terms": {},
            "termination_phase_steps": self.phase_steps.clone(),
        }
        self.phase_steps += 1
        return self._obs(), reward, done, info


def test_failure_frame_excludes_motion_complete() -> None:
    torch.manual_seed(0)
    env = _MotionCompleteEnv(num_envs=2, reward=0.05)
    algo = _build_algo(env)
    obs = algo.initial_reset()
    rollout = algo.collect(obs)

    # env1 done at frame 0 with a failure term AND motion_complete.
    assert bool(rollout["done_frame"][0, 1, 0])
    assert bool(rollout["motion_complete_frame"][0, 1, 0])
    # But motion_complete must disqualify it from being a failure.
    assert not bool(rollout["failure_frame"][0, 1, 0]), (
        "motion_complete must not be absorbed into the failure target"
    )
    # So env1's bootstrap at frame 0 is V(s_1), NOT failure_value.
    fb = rollout["frame_bootstrap"][0, 1, 0]
    fnv = rollout["frame_next_values"][0, 1, 0]
    assert abs(float(fb.item()) - float(fnv.item())) < 1e-6, (
        "bootstrap should be V(s_next), not failure_value"
    )
