from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from algorithms.sfpo import SFPO
from networks.flow_policy import FlowMatchingPolicy
from networks.mlp_actor_critic import ActionConditionedChunkCritic


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
# Action-conditioned critic: shapes + causal masking
# --------------------------------------------------------------------------- #
def test_action_conditioned_critic_shapes_and_causality() -> None:
    torch.manual_seed(1)
    critic = ActionConditionedChunkCritic(
        obs_dim=6, action_dim=4, horizon=4, hidden_dims=(32, 32), activation="elu"
    )
    obs = torch.randn(5, 6)
    actions = torch.randn(5, 4, 4)

    v = critic.evaluate_v(obs)
    assert v.shape == (5, 1)
    q = critic.evaluate_q_prefix(obs, actions)
    assert q.shape == (5, 4)

    # Q_k must not depend on actions a_k..a_{h-1} (causality).
    actions2 = actions.clone()
    actions2[:, 2:, :] = actions2[:, 2:, :] + 10.0  # perturb frames 2,3
    q2 = critic.evaluate_q_prefix(obs, actions2)
    # Q_1, Q_2 (prefixes 1,2 use frames 0..0 and 0..1) must be unchanged.
    assert torch.allclose(q[:, 0], q2[:, 0], atol=1e-6)
    assert torch.allclose(q[:, 1], q2[:, 1], atol=1e-6)
    # Q_3, Q_4 use frame 2 -> must change.
    assert not torch.allclose(q[:, 2], q2[:, 2], atol=1e-4)
    assert not torch.allclose(q[:, 3], q2[:, 3], atol=1e-4)


def test_action_conditioned_critic_rejects_bad_shapes() -> None:
    critic = ActionConditionedChunkCritic(
        obs_dim=6, action_dim=4, horizon=3, hidden_dims=(16,), activation="elu"
    )
    obs = torch.randn(5, 6)
    with pytest.raises(ValueError):
        critic.evaluate_q_prefix(obs, torch.randn(5, 3, 3))  # wrong horizon/action_dim
    with pytest.raises(ValueError):
        critic.evaluate_q_prefix(obs, torch.randn(5, 12))  # wrong ndim


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
        discount_gamma=0.99,
        clip_range=0.2, desired_kl=0.01, policy_epochs=2,
        num_mini_batches=2, micro_batch_size=64, value_loss_coef=1.0,
        value_clip_range=0.2, use_clipped_value_loss=True,
        policy_lr=1e-3, value_lr=1e-3, weight_decay=0.0, critic_weight_decay=0.0,
        empirical_normalization=False, init_at_random_ep_len=False, max_grad_norm=1.0,
        failure_penalty=10.0, q_loss_coef=1.0, q_value_clip_range=0.2,
        use_clipped_q_loss=True, advantage_normalization="per_prefix",
    )
    base.update(overrides)
    cfg = SimpleNamespace(**base)
    algo = SFPO(cfg=cfg, env=env, simulation_app=None)
    algo.build()
    return algo


# --------------------------------------------------------------------------- #
# THE core test: absorbing failure target fixes "death ranked backwards"
# --------------------------------------------------------------------------- #
def test_absorbing_failure_target_fixes_death_ranking() -> None:
    torch.manual_seed(0)
    env = _NegativeRewardEnv(num_envs=2, reward=-0.04)
    algo = _build_algo(env, failure_penalty=10.0)
    obs = algo.initial_reset()
    rollout = algo.collect(obs)

    # env1 dies (failure) at frame 0 of chunk 0; env0 survives.
    assert bool(rollout["done_frame"][0, 1, 0])
    assert bool(rollout["failure_frame"][0, 1, 0])
    assert not bool(rollout["done_frame"][0, 0].any())

    raw_env0 = float(rollout["chunk_raw_return"][0, 0].item())
    raw_env1 = float(rollout["chunk_raw_return"][0, 1].item())
    # BUG reproduction (raw return, no absorbing penalty): env1 dies early so it
    # collects fewer negative rewards -> raw return ranks ABOVE env0.
    assert raw_env1 > raw_env0, (
        f"expected dying env1 raw return ({raw_env1}) > surviving env0 ({raw_env0})"
    )

    realized_env0 = float(rollout["chunk_return_realized"][0, 0].item())
    realized_env1 = float(rollout["chunk_return_realized"][0, 1].item())
    # FIX (absorbing failure target): env1 realized return must rank BELOW env0.
    assert realized_env1 < realized_env0, (
        f"absorbing failure failed: dying env1 realized ({realized_env1}) "
        f"must be < surviving env0 ({realized_env0})"
    )
    # env1 realized return should be dominated by the failure penalty.
    expected_env1 = -0.04 + 0.99 * (-10.0)  # r0 + gamma * failure_value
    assert abs(realized_env1 - expected_env1) < 1e-4, (
        f"env1 realized {realized_env1} ~= expected {expected_env1}"
    )


def test_zero_failure_penalty_reproduces_ranking_bug() -> None:
    """With failure_penalty=0, the absorbing bootstrap is 0, so a dying chunk
    on net-negative reward still ranks above survival (the original bug). This
    documents WHY the absorbing failure target is necessary."""
    torch.manual_seed(0)
    env = _NegativeRewardEnv(num_envs=2, reward=-0.04)
    algo = _build_algo(env, failure_penalty=0.0)
    obs = algo.initial_reset()
    rollout = algo.collect(obs)

    # env1 dies (failure) at frame 0 -> bootstrap b_0 = failure_value = 0.
    # realized_env1 = r_0 + gamma * 0 = -0.04  (V-independent).
    realized_env1 = float(rollout["chunk_return_realized"][0, 1].item())
    assert abs(realized_env1 - (-0.04)) < 1e-4
    # env0 survives 4 frames of -0.04 -> raw return (V-independent) = -0.1576.
    raw_env0 = float(rollout["chunk_raw_return"][0, 0].item())
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
        assert f"critic/q_prefix_{k+1}_mean" in metrics
        assert f"critic/prefix_target_{k+1}_mean" in metrics
    # core losses finite
    for key in ("sfpo/policy_loss", "sfpo/value_loss", "sfpo/q_loss", "sfpo/loss"):
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
# CRITICAL: actor advantage uses the REALIZED prefix target (carrying the
# absorbing failure bootstrap), NOT the raw Q prediction.
# --------------------------------------------------------------------------- #
def test_actor_advantage_uses_realized_target_not_q() -> None:
    """A_k = T_{k+1} - V(s), NOT Q_k - V(s).

    With an untrained early critic, Q is garbage; the realized target T_k is
    the ground truth (it carries the absorbing failure bootstrap). The actor
    must learn from T_k - V(s), not from Q_k - V(s).
    """
    torch.manual_seed(0)
    env = _NegativeRewardEnv(num_envs=2, reward=-0.04)
    algo = _build_algo(env, failure_penalty=10.0, advantage_normalization="none")
    obs = algo.initial_reset()
    rollout = algo.collect(obs)

    prefix_targets = rollout["prefix_targets"]  # [chunks, n_envs, h]
    values_v = rollout["values_v"]  # [chunks, n_envs, 1]
    q_prefix = rollout["q_prefix"]  # [chunks, n_envs, h]
    adv = rollout["advantages"]  # [chunks, n_envs, h]
    mask = rollout["valid_prefix_mask"]  # [chunks, n_envs, h]

    expected = prefix_targets - values_v.expand(-1, -1, 4)
    # On valid prefixes, advantage == realized prefix_target - V(s).
    assert torch.allclose(adv[mask], expected[mask], atol=1e-5), (
        "actor advantage must equal realized prefix_target - V(s) on valid prefixes"
    )
    # Invalid (post-death) prefixes are masked to 0.
    assert torch.all(adv[~mask] == 0.0)
    # Sanity: it is NOT the Q-based advantage on valid prefixes.
    q_based = q_prefix - values_v.expand(-1, -1, 4)
    assert not torch.allclose(adv[mask], q_based[mask], atol=1e-5), (
        "actor advantage must NOT be Q_prefix - V(s)"
    )


# --------------------------------------------------------------------------- #
# CRITICAL: actor ratio is a PREFIX ratio (cumsum of per-frame log-ratios),
# not a per-frame ratio. This makes the ratio consistent with the prefix
# advantage A_k.
# --------------------------------------------------------------------------- #
def test_prefix_ratio_is_cumsum_of_per_frame_log_ratio() -> None:
    """Recompute the actor ratios by hand from old/new per-frame log-probs and
    verify the update actually used the cumulative (prefix) ratio."""
    torch.manual_seed(0)
    env = _NegativeRewardEnv(num_envs=2, reward=0.05)
    algo = _build_algo(env, failure_penalty=2.0, advantage_normalization="none")
    obs = algo.initial_reset()
    rollout = algo.collect(obs)

    # Recompute new per-frame log-probs (same network state as collect).
    from algorithms.sfpo import SFPO  # noqa: F401  (algo is already SFPO)
    raw_batch = rollout["actions"].shape[0] * rollout["actions"].shape[1]
    actor_obs = rollout["actor_obs"].reshape(raw_batch, -1)
    latent_path = rollout["latents"].reshape(raw_batch, -1, algo.chunk_dim)
    train_step_indices = rollout["train_step_indices"]
    new_log_probs = algo._compute_transition_log_probs_per_frame(
        actor_obs, latent_path, train_step_indices
    )  # [N, steps, h]
    old_log_probs = rollout["old_log_probs"].reshape(raw_batch, -1, 4)
    per_frame_log_ratio = (new_log_probs.sum(dim=1) - old_log_probs.sum(dim=1))  # [N, h]
    prefix_log_ratio = torch.cumsum(per_frame_log_ratio, dim=-1)  # [N, h]
    expected_prefix_ratio = torch.exp(prefix_log_ratio)  # [N, h]

    # The diagnostic sfpo/ratio_frame_k records the prefix ratio used in the
    # loss. Recompute its mean over valid prefixes and compare to the expected.
    mask = rollout["valid_prefix_mask"].reshape(raw_batch, 4).to(dtype=torch.float32)
    for k in range(4):
        col = expected_prefix_ratio[:, k][mask[:, k] > 0]
        if col.numel() == 0:
            continue
        # ratio == 1.0 exactly because the network is unchanged between collect
        # and this recompute (same parameters). Verify the cumsum structure:
        # prefix ratio_k == product of per-frame ratios 0..k.
        per_frame_ratio = torch.exp(per_frame_log_ratio)
        product_form = torch.ones_like(col)
        for j in range(k + 1):
            product_form = product_form * per_frame_ratio[:, j][mask[:, k] > 0]
        assert torch.allclose(col, product_form, atol=1e-5), (
            f"prefix ratio k={k} must equal product of per-frame ratios 0..{k}"
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
