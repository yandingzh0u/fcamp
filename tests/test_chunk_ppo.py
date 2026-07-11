from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from algorithms.chunk_ppo import SFPO


class _ChunkTestEnv:
    def __init__(self, num_envs=2, obs_dim=8, critic_dim=8, action_dim=3, reward=-0.04):
        self.num_envs = num_envs
        self.observation_dim = obs_dim
        self.critic_observation_dim = critic_dim
        self.action_dim = action_dim
        self.device = torch.device("cpu")
        self.max_episode_steps = 20
        self.dt = 0.02
        self.phase_steps = torch.zeros(num_envs, dtype=torch.long)
        self.episode_steps = torch.zeros(num_envs, dtype=torch.long)
        self.config = SimpleNamespace(action_rate_weight=0.1)
        self._gen = torch.Generator().manual_seed(7)
        self._call = 0
        self._reward = reward
        self._done_env1_at_call = 1

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
        return {"top_bin": 0.0, "top_prob": 0.0, "failed_sum": 0.0, "entropy": 0.0, "peak_bin": 0.0}

    def _done_terms(self, done, *, motion_complete=None, timeout=None):
        time_out = torch.zeros(self.num_envs, dtype=torch.bool) if timeout is None else timeout
        motion = torch.zeros(self.num_envs, dtype=torch.bool) if motion_complete is None else motion_complete
        return {
            "time_out": time_out,
            "motion_complete": motion,
            "anchor_pos_bad": torch.zeros(self.num_envs, dtype=torch.bool),
            "anchor_ori_bad": torch.zeros(self.num_envs, dtype=torch.bool),
            "ee_body_bad": done.clone(),
        }

    def step(self, action, auto_reset=False):
        assert not auto_reset
        self._call += 1
        reward = torch.full((self.num_envs,), float(self._reward))
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        if self._call == self._done_env1_at_call:
            done[1] = True
        info = {
            "done_terms": self._done_terms(done),
            "reward_terms": {},
            "termination_phase_steps": self.phase_steps.clone(),
        }
        self.phase_steps += 1
        return self._obs(), reward, done, info


class _MotionCompleteEnv(_ChunkTestEnv):
    def step(self, action, auto_reset=False):
        assert not auto_reset
        self._call += 1
        reward = torch.full((self.num_envs,), float(self._reward))
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        motion_complete = torch.zeros(self.num_envs, dtype=torch.bool)
        if self._call == self._done_env1_at_call:
            done[1] = True
            motion_complete[1] = True
        info = {
            "done_terms": self._done_terms(done, motion_complete=motion_complete),
            "reward_terms": {},
            "termination_phase_steps": self.phase_steps.clone(),
        }
        self.phase_steps += 1
        return self._obs(), reward, done, info


def _build_algo(env, **overrides):
    base = dict(
        horizon=4,
        actor_hidden_dims=(16, 16),
        critic_hidden_dims=(16, 16),
        activation="elu",
        init_noise_std=0.5,
        discount_gamma=0.99,
        num_steps_per_env=8,
        num_learning_epochs=2,
        num_mini_batches=2,
        gae_lambda=0.95,
        clip_range=0.2,
        value_clip_range=0.2,
        entropy_coef=0.005,
        value_loss_coef=1.0,
        desired_kl=0.01,
        actor_learning_rate=1e-3,
        critic_learning_rate=1e-3,
        weight_decay=0.0,
        critic_weight_decay=0.0,
        empirical_normalization=False,
        init_at_random_ep_len=False,
        max_grad_norm=1.0,
    )
    base.update(overrides)
    algo = SFPO(cfg=SimpleNamespace(**base), env=env, simulation_app=None)
    algo.build()
    return algo


def test_chunk_ppo_is_not_flow_sampling() -> None:
    torch.manual_seed(0)
    env = _ChunkTestEnv(num_envs=4, reward=0.05)
    algo = _build_algo(env)
    obs = algo.initial_reset()
    rollout = algo.collect(obs)

    assert algo.horizon == 4
    assert algo.kl_units == 4
    assert rollout["actions"].shape == (2, 4, 4, 3)
    assert rollout["logp"].shape == (2, 4, 1)
    assert rollout["mu"].shape[-1] == 12
    assert "latents" not in rollout
    assert "old_log_probs" not in rollout


def test_failure_has_zero_bootstrap_and_no_handwritten_penalty() -> None:
    torch.manual_seed(0)
    env = _ChunkTestEnv(num_envs=2, reward=-0.04)
    algo = _build_algo(env)
    obs = algo.initial_reset()
    rollout = algo.collect(obs)

    realized_env1 = float(rollout["chunk_return_realized"][0, 1].item())
    assert abs(realized_env1 - (-0.04)) < 1e-4
    assert torch.allclose(rollout["failure_cost_return"], torch.zeros_like(rollout["failure_cost_return"]))


def test_valid_prefix_mask_for_early_death() -> None:
    torch.manual_seed(0)
    env = _ChunkTestEnv(num_envs=2, reward=1.0)
    algo = _build_algo(env)
    obs = algo.initial_reset()
    rollout = algo.collect(obs)

    assert int(rollout["death_frame"][0, 1].item()) == 0
    assert rollout["valid_prefix_mask"][0, 1].tolist() == [True, False, False, False]
    assert int(rollout["death_frame"][0, 0].item()) == 4
    assert rollout["valid_prefix_mask"][0, 0].tolist() == [True, True, True, True]


def test_motion_complete_is_terminal_not_failure() -> None:
    torch.manual_seed(0)
    env = _MotionCompleteEnv(num_envs=2, reward=0.05)
    algo = _build_algo(env)
    obs = algo.initial_reset()
    rollout = algo.collect(obs)

    assert bool(rollout["done_frame"][0, 1, 0])
    assert bool(rollout["motion_complete_frame"][0, 1, 0])
    assert not bool(rollout["failure_frame"][0, 1, 0])
    assert float(rollout["frame_bootstrap"][0, 1, 0].item()) == 0.0


def test_chunk_advantages_are_normalized_and_raw_advantage_is_preserved() -> None:
    torch.manual_seed(0)
    env = _ChunkTestEnv(num_envs=4, reward=0.05)
    algo = _build_algo(env)
    obs = algo.initial_reset()
    rollout = algo.collect(obs)

    mask = rollout["chunk_valid_mask"]
    normalized = rollout["advantages"].squeeze(-1)[mask]
    if normalized.numel() > 1:
        assert abs(float(normalized.mean().item())) < 1e-5
    assert torch.allclose(rollout["raw_advantages"].squeeze(-1), rollout["chunk_advantages"], atol=1e-6)


def test_sfpo_collect_and_update_end_to_end() -> None:
    torch.manual_seed(0)
    env = _ChunkTestEnv(num_envs=4, reward=0.05)
    algo = _build_algo(env, num_mini_batches=2)
    obs = algo.initial_reset()
    rollout = algo.collect(obs)
    metrics = algo.update(rollout, collect_time=0.1)

    for key in ("sfpo/policy_loss", "sfpo/value_loss", "sfpo/loss", "sfpo/kl_raw"):
        assert math.isfinite(metrics[key]), f"{key}={metrics[key]}"
    assert metrics["sfpo/kl_units"] == 4.0
    assert abs(metrics["sfpo/kl_target_raw"] - 0.04) < 1e-9
    assert metrics["sfpo/grad_norm"] >= 0.0
    assert metrics["sfpo/grad_norm_critic"] >= 0.0
    assert metrics["policy/action_delta"] >= 0.0
    for k in range(4):
        assert f"sfpo/kl_frame_{k}" in metrics


def test_deterministic_actions_returns_full_chunk() -> None:
    torch.manual_seed(0)
    env = _ChunkTestEnv(num_envs=3, reward=0.05)
    algo = _build_algo(env)
    obs = algo.initial_reset()
    a1 = algo.deterministic_actions(obs)
    a2 = algo.deterministic_actions(obs)
    assert a1.shape == (3, 4, 3)
    assert torch.allclose(a1, a2, atol=1e-6)
