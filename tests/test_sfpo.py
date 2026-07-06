from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from algorithms.sfpo import SFPO
from networks.flow_policy import FlowMatchingPolicy


# --------------------------------------------------------------------------- #
# Per-frame flow log-prob consistency (rollout == recompute)
# --------------------------------------------------------------------------- #
def test_sfpo_per_frame_log_probs_recomputed_on_policy() -> None:
    torch.manual_seed(0)
    cfg = SimpleNamespace(flow_steps=3, sde_eta=0.7, init_noise_std=0.8, horizon=2)
    algo = SFPO(cfg=cfg, env=None, simulation_app=None)
    algo._policy = FlowMatchingPolicy(
        obs_dim=5,
        action_dim=3,
        horizon=2,
        hidden_dims=(16, 16),
        activation="elu",
        action_squash_scale=5.0,
    )
    algo.chunk_dim = algo._policy.chunk_dim
    algo.num_act = 3
    algo.horizon_h = 2
    obs = torch.randn(7, 5)
    initial_noise = torch.randn(7, algo.chunk_dim)
    sde_noise = torch.randn(7, cfg.flow_steps, algo.chunk_dim)

    _, latent_path, old_log_probs = algo._sde_ode_rollout_actions_per_frame(
        obs,
        initial_noise=initial_noise,
        sde_noise=sde_noise,
    )
    recomputed = algo._compute_transition_log_probs_per_frame(
        obs,
        latent_path,
        torch.arange(cfg.flow_steps),
    )

    assert recomputed.shape == old_log_probs.shape
    assert recomputed.shape == (7, cfg.flow_steps, 2)
    assert torch.allclose(recomputed, old_log_probs, atol=1e-5)


# --------------------------------------------------------------------------- #
# Mock env for collect / update integration tests
# --------------------------------------------------------------------------- #
class _NegativeRewardEnv:
    """Mock env with NET-NEGATIVE per-step reward.

    env0 never dies; env1 dies (failure) on the first step of chunk 0 only.
    With negative rewards, a dying chunk accumulates fewer negative terms, so
    its RAW chunk return ranks ABOVE the surviving chunk -- this is the
    "death chunk ranked backwards" bug. The absorbing failure target must flip
    the ranking so the dying chunk's REALIZED return ranks below survival.
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
        horizon=4, rollout_env_steps=8, flow_steps=2, sde_eta=0.7, init_noise_std=0.8,
        action_squash_scale=5.0, eval_initial_noise="zero",
        actor_hidden_dims=(16, 16), critic_hidden_dims=(16, 16), activation="elu",
        discount_gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, desired_kl=0.01, policy_epochs=2,
        num_mini_batches=2, micro_batch_size=64, value_loss_coef=1.0,
        value_clip_range=0.2, use_clipped_value_loss=True,
        policy_lr=1e-3, value_lr=1e-3, weight_decay=0.0, critic_weight_decay=0.0,
        empirical_normalization=False, init_at_random_ep_len=False, max_grad_norm=1.0,
        causal_velocity=True, causal_arch="prefix_cumsum",
        action_transform="delta", action_max_delta=0.5, failure_penalty=1.0,
        kl_early_stop_factor=4.0, advantage_normalization="global",
    )
    base.update(overrides)
    cfg = SimpleNamespace(**base)
    algo = SFPO(cfg=cfg, env=env, simulation_app=None)
    algo.build()
    return algo


# --------------------------------------------------------------------------- #
# THE core test: terminal failure cost fixes "death ranked backwards"
# --------------------------------------------------------------------------- #
def test_terminal_failure_cost_fixes_death_ranking() -> None:
    """With net-negative per-step reward, a dying chunk would accumulate fewer
    negative terms and rank ABOVE survival (the "death ranked backwards" bug).
    The terminal failure cost (immediate penalty on the failure frame + bootstrap
    0) must flip the ranking so the dying chunk's realized return ranks below
    survival.
    """
    torch.manual_seed(0)
    env = _NegativeRewardEnv(num_envs=2, reward=-0.04)
    algo = _build_algo(env, failure_penalty=1.0)
    obs = algo.initial_reset()
    rollout = algo.collect(obs)

    # env1 dies (failure) at frame 0 of chunk 0; env0 survives.
    assert bool(rollout["done_frame"][0, 1, 0])
    assert bool(rollout["failure_frame"][0, 1, 0])
    assert not bool(rollout["done_frame"][0, 0].any())

    # BUG reproduction (raw return, no failure cost): env1 dies early so it
    # collects fewer negative rewards -> masked raw return ranks ABOVE env0.
    # Use reward_raw * alive (masked, excludes failure penalty) summed over the
    # chunk to reproduce the original "death saves negative reward" bug.
    gamma = 0.99
    alive = rollout["alive_frame"][0].to(dtype=torch.float32)  # [n_envs, h]
    reward_raw = rollout["reward_raw"][0]  # [n_envs, h]
    gamma_pow = torch.tensor([gamma ** i for i in range(4)], dtype=torch.float32)
    masked_raw_env0 = float((reward_raw[0] * alive[0] * gamma_pow).sum().item())
    masked_raw_env1 = float((reward_raw[1] * alive[1] * gamma_pow).sum().item())
    assert masked_raw_env1 > masked_raw_env0, (
        f"expected dying env1 masked raw ({masked_raw_env1}) > surviving env0 ({masked_raw_env0})"
    )

    realized_env0 = float(rollout["chunk_return_realized"][0, 0].item())
    realized_env1 = float(rollout["chunk_return_realized"][0, 1].item())
    # FIX (terminal failure cost): env1 realized return must rank BELOW env0.
    assert realized_env1 < realized_env0, (
        f"terminal failure cost failed: dying env1 realized ({realized_env1}) "
        f"must be < surviving env0 ({realized_env0})"
    )
    # env1: failure frame reward = r0 - failure_penalty, bootstrap = 0 (terminal).
    expected_env1 = -0.04 - 1.0  # r0 - failure_penalty, no bootstrap
    assert abs(realized_env1 - expected_env1) < 1e-4, (
        f"env1 realized {realized_env1} ~= expected {expected_env1}"
    )


def test_zero_failure_penalty_reproduces_ranking_bug() -> None:
    """With failure_penalty=0, the terminal cost is 0 and bootstrap is still 0
    on failure, so a dying chunk on net-negative reward still ranks above
    survival (the original bug). This documents WHY the failure cost is needed."""
    torch.manual_seed(0)
    env = _NegativeRewardEnv(num_envs=2, reward=-0.04)
    algo = _build_algo(env, failure_penalty=0.0)
    obs = algo.initial_reset()
    rollout = algo.collect(obs)

    # env1 dies (failure) at frame 0 -> reward = r0 - 0 = -0.04, bootstrap = 0.
    realized_env1 = float(rollout["chunk_return_realized"][0, 1].item())
    assert abs(realized_env1 - (-0.04)) < 1e-4
    # env0 survives 4 frames of -0.04 -> masked raw return = -0.1576.
    gamma = 0.99
    alive = rollout["alive_frame"][0, 0].to(dtype=torch.float32)
    reward_raw_env0 = rollout["reward_raw"][0, 0]
    gamma_pow = torch.tensor([gamma ** i for i in range(4)], dtype=torch.float32)
    raw_env0 = float((reward_raw_env0 * alive * gamma_pow).sum().item())
    assert abs(raw_env0 - (-0.04 * (1 + 0.99 + 0.99**2 + 0.99**3))) < 1e-4
    # BUG: the dying chunk's realized return (-0.04) ranks ABOVE the surviving
    # chunk's raw return (-0.1576) -- death "saves" negative reward.
    assert realized_env1 > raw_env0


# --------------------------------------------------------------------------- #
# valid_prefix_mask correctness
# --------------------------------------------------------------------------- #
def test_valid_prefix_mask_for_early_death() -> None:
    torch.manual_seed(0)
    env = _NegativeRewardEnv(num_envs=2, reward=1.0)  # positive reward, no death ranking issue
    algo = _build_algo(env, failure_penalty=1.0)
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
    algo = _build_algo(env, failure_penalty=2.0, num_mini_batches=2, micro_batch_size=8)
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


# --------------------------------------------------------------------------- #
# Per-prefix advantage normalization produces zero-mean per prefix
# --------------------------------------------------------------------------- #
def test_per_prefix_advantage_normalization() -> None:
    torch.manual_seed(0)
    env = _NegativeRewardEnv(num_envs=4, reward=0.05)
    algo = _build_algo(env, advantage_normalization="per_prefix")
    obs = algo.initial_reset()
    rollout = algo.collect(obs)

    adv = rollout["advantages"]
    mask = rollout["valid_prefix_mask"]
    h = 4
    for k in range(h):
        col = adv[..., k][mask[..., k]]
        if col.numel() > 1:
            assert abs(float(col.mean().item())) < 1e-5, (
                f"prefix {k+1} advantages not zero-mean: {float(col.mean().item())}"
            )
    # invalid prefixes are zeroed
    assert torch.all(adv[~mask] == 0.0)


# --------------------------------------------------------------------------- #
# CRITICAL: actor advantage is the per-frame GAE advantage
# (frame_v_targets - frame_values), NOT a multi-prefix objective.
# --------------------------------------------------------------------------- #
def test_actor_advantage_is_per_frame_gae() -> None:
    """A_j = gae_advantages_j = frame_v_targets_j - frame_values_j.

    This is the standard PPO/GAE estimator: frame-j unit, lambda-smoothed,
    consistent with the critic V target. Replaces d1b506c's
    A_k = T_{k+1} - V(s_0) which mixed early-reward credit into all later
    prefixes.
    """
    torch.manual_seed(0)
    env = _NegativeRewardEnv(num_envs=2, reward=-0.04)
    algo = _build_algo(env, failure_penalty=1.0, advantage_normalization="none")
    obs = algo.initial_reset()
    rollout = algo.collect(obs)

    frame_v_targets = rollout["frame_v_targets"]  # [chunks, n_envs, h]
    frame_values = rollout["frame_values"]  # [chunks, n_envs, h]
    adv = rollout["advantages"]  # [chunks, n_envs, h]
    mask = rollout["valid_prefix_mask"]  # [chunks, n_envs, h]

    expected = frame_v_targets - frame_values
    # On valid (alive) frames, advantage == per-frame GAE advantage.
    assert torch.allclose(adv[mask], expected[mask], atol=1e-5), (
        "actor advantage must equal per-frame GAE (frame_v_targets - frame_values) on valid frames"
    )
    # Invalid (post-death) frames are masked to 0.
    assert torch.all(adv[~mask] == 0.0)


# --------------------------------------------------------------------------- #
# CRITICAL: with causal_velocity=True, per-frame ratio is the LEGAL form
# because logp_k is a genuine conditional density (v_k depends only on
# z_0..z_k). This test verifies the per-frame (non-cumsum) ratio is used.
# --------------------------------------------------------------------------- #
def test_actor_ratio_is_per_frame_under_causal_policy() -> None:
    """Under causal_velocity, per-frame ratio is legal (logp_k is conditional).
    Verify the update used per-frame (non-cumulative) ratio."""
    torch.manual_seed(0)
    env = _NegativeRewardEnv(num_envs=2, reward=0.05)
    algo = _build_algo(env, failure_penalty=2.0, advantage_normalization="none")
    assert algo._policy.causal_velocity is True
    obs = algo.initial_reset()
    rollout = algo.collect(obs)

    raw_batch = rollout["actions"].shape[0] * rollout["actions"].shape[1]
    actor_obs = rollout["actor_obs"].reshape(raw_batch, -1)
    latent_path = rollout["latents"].reshape(raw_batch, -1, algo.chunk_dim)
    train_step_indices = rollout["train_step_indices"]
    new_log_probs = algo._compute_transition_log_probs_per_frame(
        actor_obs, latent_path, train_step_indices
    )
    old_log_probs = rollout["old_log_probs"].reshape(raw_batch, -1, 4)
    per_frame_log_ratio = (new_log_probs.sum(dim=1) - old_log_probs.sum(dim=1))
    expected_per_frame_ratio = torch.exp(per_frame_log_ratio)

    mask = rollout["valid_prefix_mask"].reshape(raw_batch, 4).to(dtype=torch.float32)
    for k in range(4):
        col = expected_per_frame_ratio[:, k][mask[:, k] > 0]
        if col.numel() == 0:
            continue
        assert torch.allclose(col, torch.exp(per_frame_log_ratio[:, k][mask[:, k] > 0]), atol=1e-5), (
            f"per-frame ratio k={k} must equal exp(per_frame_log_ratio_{k})"
        )


# --------------------------------------------------------------------------- #
# CAUSAL HARD TESTS: v_k / logp_k / a_k must NOT depend on future latents j>k.
# These fail on full-chunk MLP (causal_velocity=False) and pass on causal.
# --------------------------------------------------------------------------- #
def _causal_grad_leak(policy, target: str, frame_k: int, horizon: int = 4, action_dim: int = 3) -> float:
    """Return max |grad| of (logp_k or a_k) w.r.t. future noise frames j>k."""
    torch.manual_seed(0)
    from networks.flow_sampling import flow_grpo_step_per_frame
    obs = torch.randn(2, policy.obs_dim)
    noise = torch.randn(2, policy.chunk_dim, requires_grad=True)
    sde_noise = torch.randn(2, 3, policy.chunk_dim)
    sigma_schedule = torch.linspace(1.0, 0.0, 4, dtype=noise.dtype)
    latent = noise * 0.8
    t_batch = torch.full((2,), 1.0, dtype=noise.dtype)
    model_out = policy.velocity_field(obs, latent, t_batch)
    new_latent, logp = flow_grpo_step_per_frame(
        model_output=model_out, latents=latent, sigmas=sigma_schedule, index=0,
        eta=0.7, sample_noise=sde_noise[:, 0], horizon=horizon, action_dim=action_dim,
    )
    noise.grad = None
    if target == "logp":
        logp[0, frame_k].sum().backward(retain_graph=True)
    elif target == "action":
        actions = policy._action_transform(new_latent[0], prev_action=torch.zeros(action_dim))
        actions = actions.view(horizon, action_dim)
        actions[frame_k].sum().backward(retain_graph=True)
    else:
        raise ValueError(f"unknown target: {target}")
    g = noise.grad[0]
    # future frames = cols [(k+1)*A : ]
    future = g[(frame_k + 1) * action_dim:]
    return float(future.abs().max().item()) if future.numel() > 0 else 0.0


def test_causal_logp_no_future_gradient_leak() -> None:
    """∂logp_k / ∂z_j = 0 for j > k (causal conditional density)."""
    from networks.flow_policy import FlowMatchingPolicy
    pol = FlowMatchingPolicy(obs_dim=5, action_dim=3, horizon=4, hidden_dims=(16, 16),
                             activation="elu", action_squash_scale=5.0, causal_velocity=True)
    pol.set_action_max_delta(0.5)
    for k in range(3):  # frame 3 has no future
        leak = _causal_grad_leak(pol, "logp", k)
        assert leak <= 1e-6, f"causal logp frame {k} leaks future grad: {leak}"


def test_causal_action_no_future_gradient_leak() -> None:
    """∂a_k / ∂z_j = 0 for j > k (direct-delta is causal by construction)."""
    from networks.flow_policy import FlowMatchingPolicy
    pol = FlowMatchingPolicy(obs_dim=5, action_dim=3, horizon=4, hidden_dims=(16, 16),
                             activation="elu", action_squash_scale=5.0, causal_velocity=True)
    pol.set_action_max_delta(0.5)
    for k in range(3):
        leak = _causal_grad_leak(pol, "action", k)
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
        leak = _causal_grad_leak(pol, "action", k)
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
        leak = _causal_grad_leak(pol, "action", k)
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


def test_full_mlp_logp_DOES_leak_future_gradient() -> None:
    """Sanity: the non-causal full-MLP policy DOES leak future gradient.
    This documents why causal_velocity is necessary for per-frame PPO."""
    from networks.flow_policy import FlowMatchingPolicy
    pol = FlowMatchingPolicy(obs_dim=5, action_dim=3, horizon=4, hidden_dims=(16, 16),
                             activation="elu", action_squash_scale=5.0, causal_velocity=False)
    pol.set_action_max_delta(0.5)
    leak = _causal_grad_leak(pol, "logp", 0)
    assert leak > 1e-4, f"full MLP should leak future grad, got {leak}"


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
    algo = _build_algo(env, failure_penalty=10.0)
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
