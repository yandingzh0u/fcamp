from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from algorithms.sfpo import SFPO
from networks.flow_policy import FlowMatchingPolicy


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


def _build_algo(env, **overrides):
    base = dict(
        horizon=4, rollout_env_steps=8, flow_steps=2,
        sde_noise_std=0.8, sde_std_trainable=True,
        action_squash_scale=5.0,
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
    algo = SFPO(cfg=cfg, env=env, simulation_app=None)
    algo.build()
    return algo


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
def test_sfpo_collect_and_update_end_to_end() -> None:
    torch.manual_seed(0)
    env = _NegativeRewardEnv(num_envs=4, reward=0.05)  # small positive reward
    algo = _build_algo(env, num_mini_batches=2, micro_batch_size=8)
    obs = algo.initial_reset()
    rollout = algo.collect(obs)
    metrics = algo.update(rollout, collect_time=0.1)

    # per-frame ratio / kl diagnostics present for all h frames
    h = 4
    for k in range(h):
        assert f"sfpo/kl_frame_{k}" in metrics
        assert f"sfpo/ratio_frame_{k}" in metrics
        assert f"critic/frame_v_{k+1}_mean" in metrics
        assert f"critic/prefix_target_{k+1}_mean" in metrics
    # core losses finite
    for key in ("sfpo/policy_loss", "sfpo/value_loss", "sfpo/loss"):
        assert math.isfinite(metrics[key]), f"{key}={metrics[key]}"
    # update actually moved the policy
    assert metrics["policy/action_delta"] >= 0.0
    # actor + critic both have grad norms
    assert metrics["sfpo/grad_norm"] >= 0.0
    assert metrics["sfpo/grad_norm_critic"] >= 0.0


def test_sfpo_learned_sde_density_collect_and_update() -> None:
    torch.manual_seed(0)
    env = _NegativeRewardEnv(num_envs=4, reward=0.05)
    algo = _build_algo(
        env,
        sde_noise_std=0.4,
        sde_std_trainable=True,
        num_mini_batches=2,
        micro_batch_size=8,
    )
    obs = algo.initial_reset()
    rollout = algo.collect(obs)

    assert torch.isfinite(rollout["old_log_probs"]).all()
    assert rollout["old_sde_std"].gt(0.0).any()
    assert algo._policy.sde_log_std.shape == (2, 4, 3)
    assert algo._policy.sde_log_std.numel() == 24
    assert torch.allclose(rollout["failure_cost_return"], torch.zeros_like(rollout["failure_cost_return"]))

    metrics = algo.update(rollout, collect_time=0.1)
    assert math.isfinite(metrics["sfpo/policy_loss"])
    assert math.isfinite(metrics["sfpo/kl_raw"])
    assert metrics["policy/sde_std_mean"] > 0.0
    assert metrics["policy/sde_std_params"] == 24.0


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
def _causal_action_grad_leak(policy, frame_k: int, horizon: int = 4, action_dim: int = 3) -> float:
    """Return max |grad| of a_k w.r.t. future noise frames j>k."""
    torch.manual_seed(0)
    from networks.flow_sampling import flow_ode_mean
    obs = torch.randn(2, policy.obs_dim)
    noise = torch.randn(2, policy.chunk_dim, requires_grad=True)
    sigma_schedule = torch.linspace(1.0, 0.0, 4, dtype=noise.dtype)
    latent = noise * 0.8
    t_batch = torch.full((2,), 1.0, dtype=noise.dtype)
    model_out = policy.velocity_field(obs, latent, t_batch)
    new_latent = flow_ode_mean(model_out, latent, sigma_schedule, 0)
    noise.grad = None
    actions = policy._action_transform(new_latent[0], prev_action=torch.zeros(action_dim))
    actions = actions.view(horizon, action_dim)
    actions[frame_k].sum().backward(retain_graph=True)
    g = noise.grad[0]
    # future frames = cols [(k+1)*A : ]
    future = g[(frame_k + 1) * action_dim:]
    return float(future.abs().max().item()) if future.numel() > 0 else 0.0


def test_causal_action_no_future_gradient_leak() -> None:
    """∂a_k / ∂z_j = 0 for j > k (direct-delta is causal by construction)."""
    from networks.flow_policy import FlowMatchingPolicy
    pol = FlowMatchingPolicy(obs_dim=5, action_dim=3, horizon=4, hidden_dims=(16, 16),
                             activation="elu", action_squash_scale=5.0, causal_velocity=True)
    pol.set_action_max_delta(0.5)
    for k in range(3):
        leak = _causal_action_grad_leak(pol, k)
        assert leak <= 1e-6, f"causal action frame {k} leaks future grad: {leak}"


def test_causal_action_no_future_gradient_leak_absolute() -> None:
    """v5 absolute transform: a_k = scale*tanh(raw_k/scale) depends only on
    z_k, so per-frame causality (and thus per-frame PPO ratio legality) is
    preserved WITHOUT the hard delta cap. This is the v5 guarantee -- the
    chunk policy keeps a legal per-frame ratio after removing direct-delta."""
    from networks.flow_policy import FlowMatchingPolicy
    pol = FlowMatchingPolicy(obs_dim=5, action_dim=3, horizon=4, hidden_dims=(16, 16),
                             activation="elu", action_squash_scale=5.0, causal_velocity=True)
    pol.set_action_max_delta(None)  # v5 absolute mode
    assert pol.action_max_delta is None
    for k in range(3):
        leak = _causal_action_grad_leak(pol, k)
        assert leak <= 1e-6, f"absolute action frame {k} leaks future grad: {leak}"


def test_causal_action_no_future_gradient_leak_residual() -> None:
    """v6 residual_absolute transform: a_k = scale*tanh((u_prev + sum_{i<=k}
    raw_i)/scale) depends only on z_0..z_k, so per-frame causality (and thus
    per-frame PPO ratio legality) is preserved. This is the v6 guarantee --
    prev-action-anchored full-support residuals keep a legal per-frame ratio
    while granting PPO-like recovery authority (unlike v4's hard delta cap)."""
    from networks.flow_policy import FlowMatchingPolicy
    pol = FlowMatchingPolicy(obs_dim=5, action_dim=3, horizon=4, hidden_dims=(16, 16),
                             activation="elu", action_squash_scale=5.0, causal_velocity=True)
    pol.action_transform = "residual_absolute"
    pol.set_action_max_delta(None)  # residual mode does not use a delta cap
    assert pol.action_transform == "residual_absolute"
    for k in range(3):
        leak = _causal_action_grad_leak(pol, k)
        assert leak <= 1e-6, f"residual action frame {k} leaks future grad: {leak}"


def test_residual_absolute_anchors_on_prev_action() -> None:
    """v6 residual_absolute: zero residual chunk => executed chunk holds the
    previous action exactly (a_k = prev for all k). This is the 'zero output =
    hold current action' property that pure absolute lacks (and which v5
    collapsed without)."""
    from networks.flow_policy import FlowMatchingPolicy
    torch.manual_seed(0)
    pol = FlowMatchingPolicy(obs_dim=5, action_dim=3, horizon=4, hidden_dims=(16, 16),
                             activation="elu", action_squash_scale=5.0, causal_velocity=True)
    pol.action_transform = "residual_absolute"
    prev = torch.tensor([[0.3, -0.7, 1.2]])
    zero_chunk = torch.zeros(1, pol.chunk_dim)
    actions = pol._action_transform(zero_chunk, prev_action=prev).view(pol.horizon, pol.action_dim)
    assert torch.allclose(actions, prev.expand(pol.horizon, -1), atol=1e-5), (
        f"zero residual must hold prev_action, got {actions}"
    )


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
