from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("rsl_rl")

from method.beyondmimic import Beyondmimic


class _FakeBeyondMimicEnv:
    def __init__(self, num_envs: int = 4) -> None:
        self.device = torch.device("cpu")
        self.num_envs = num_envs
        self.action_dim = 29
        self.max_episode_steps = 500
        self.episode_steps = torch.zeros(num_envs, dtype=torch.long)
        self.push_time_left = torch.linspace(1.0, 3.0, num_envs)
        self.phase_steps = torch.zeros(num_envs, dtype=torch.float32)
        self._step_index = 0
        self.reset_calls = 0

    def get_beyondmimic_policy_observation(self) -> torch.Tensor:
        return torch.full((self.num_envs, 160), float(self._step_index) * 0.01)

    def get_observation(self) -> torch.Tensor:
        return self.get_beyondmimic_policy_observation()

    def get_beyondmimic_critic_observation(self) -> torch.Tensor:
        return torch.full((self.num_envs, 286), float(self._step_index) * 0.02)

    def reset(self) -> torch.Tensor:
        self.reset_calls += 1
        self.episode_steps.zero_()
        self.phase_steps.zero_()
        self._step_index = 0
        return self.get_beyondmimic_policy_observation()

    def step(self, actions: torch.Tensor, auto_reset: bool = False):
        assert auto_reset
        assert actions.shape == (self.num_envs, self.action_dim)
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        time_out = torch.zeros_like(done)
        anchor_pos_bad = torch.zeros_like(done)
        if self._step_index == 0:
            done[:3] = True
            time_out[0] = True
            time_out[2] = True
            anchor_pos_bad[1:3] = True
        self._step_index += 1
        self.episode_steps += 1
        self.phase_steps += 1.0
        reward = torch.full((self.num_envs,), 1.5)
        zeros = torch.zeros_like(done)
        motion_resample = torch.zeros_like(done)
        if self._step_index == 2:
            motion_resample[2] = True
        info = {
            "done_terms": {
                "time_out": time_out,
                "motion_complete": zeros.clone(),
                "anchor_pos_bad": anchor_pos_bad,
                "anchor_ori_bad": zeros.clone(),
                "ee_body_bad": zeros.clone(),
            },
            "reward_terms": {"tracking": reward.clone()},
            "termination_phase_steps": self.phase_steps.clone(),
            "motion_resample_mask": motion_resample,
        }
        return self.get_beyondmimic_policy_observation(), reward, done, info

    @staticmethod
    def adaptive_sampling_stats() -> dict[str, float]:
        return {"mode": 1.0, "top_bin": 0.0, "top_prob": 1.0, "failed_sum": 0.0, "entropy": 0.0}


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        rollout_env_steps=24,
        actor_hidden_dims=(16, 8),
        critic_hidden_dims=(16, 8),
        activation="elu",
        empirical_normalization=True,
        init_noise_std=1.0,
        noise_std_type="scalar",
        state_dependent_std=False,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.005,
        learning_rate=1.0e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="adaptive",
        desired_kl=0.01,
        normalize_advantage_per_mini_batch=False,
        init_at_random_ep_len=False,
    )


def test_beyondmimic_delegates_rollout_timeout_and_update_to_rsl_rl() -> None:
    env = _FakeBeyondMimicEnv()
    algo = Beyondmimic(_config(), env, simulation_app=None)
    algo.build()
    observation = algo.initial_reset()
    assert env.reset_calls == 0

    rollout = algo.collect(observation)
    storage = algo.ppo.storage
    assert storage is not None
    assert storage.step == 24
    assert rollout["timeout"][0, 0]
    assert rollout["failure"][0, 1]
    assert not rollout["timeout"][0, 1]
    assert rollout["timeout"][0, 2]
    assert rollout["failure"][0, 2]

    # RSL-RL bootstraps truncations with the transition value, but does not
    # bootstrap actual failures.
    timeout_expected = 1.5 + algo.ppo.gamma * storage.values[0, 0, 0]
    assert torch.allclose(storage.rewards[0, 0, 0], timeout_expected)
    assert storage.rewards[0, 1, 0].item() == pytest.approx(1.5)
    overlap_expected = 1.5 + algo.ppo.gamma * storage.values[0, 2, 0]
    assert torch.allclose(storage.rewards[0, 2, 0], overlap_expected)
    assert algo.actor_obs_normalizer.count.item() == env.num_envs * 24
    assert algo.critic_obs_normalizer.count.item() == env.num_envs * 24

    # Before any optimizer step, replaying the action-time normalized storage
    # must reproduce the old policy/value outputs. This is the defining 2.3.3
    # boundary and catches RSL-RL 3.1's raw-storage behavior.
    with torch.no_grad():
        stored_obs = storage.observations.flatten(0, 1)
        algo.actor_critic.act(stored_obs)
        replay_mu = algo.actor_critic.action_mean.reshape_as(storage.mu)
        replay_values = algo.actor_critic.evaluate(stored_obs).reshape_as(storage.values)
    torch.testing.assert_close(replay_mu, storage.mu)
    torch.testing.assert_close(replay_values, storage.values)

    metrics = algo.update(rollout, collect_time=0.25)
    assert storage.step == 0
    assert metrics["method/beyondmimic"] == 1.0
    assert metrics["system/parameters_finite"] == 1.0
    assert metrics["system/buffers_finite"] == 1.0
    assert metrics["system/optimizer_finite"] == 1.0
    assert metrics["beyondmimic/optimizer_steps"] == 1.0
    assert metrics["beyondmimic/last_minibatch_clipped_grad_norm"] <= 1.0001
    assert metrics["beyondmimic/post_update_kl"] >= 0.0
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["rollout/motion_resample_count"] == 1.0
    assert metrics["rollout/motion_resample_frac"] == pytest.approx(1.0 / (env.num_envs * 24))
    assert metrics["normalizer/actor_count"] == env.num_envs * 24
    assert algo.deterministic_actions(observation).shape == (env.num_envs, env.action_dim)


def test_beyondmimic_rejects_nonofficial_observation_schema() -> None:
    env = _FakeBeyondMimicEnv()
    env.get_beyondmimic_policy_observation = lambda: torch.zeros(env.num_envs, 159)
    algo = Beyondmimic(_config(), env, simulation_app=None)
    with pytest.raises(ValueError, match="160D"):
        algo.build()


def test_random_episode_length_keeps_push_countdown_independent() -> None:
    env = _FakeBeyondMimicEnv(num_envs=4096)
    cfg = _config()
    cfg.init_at_random_ep_len = True
    algo = Beyondmimic(cfg, env, simulation_app=None)
    algo.build()
    timer_before = env.push_time_left.clone()
    algo.initial_reset()

    torch.testing.assert_close(env.push_time_left, timer_before)


def test_beyondmimic_policy_optimizer_normalizers_and_lr_roundtrip() -> None:
    env = _FakeBeyondMimicEnv()
    source = Beyondmimic(_config(), env, simulation_app=None)
    source.build()
    rollout = source.collect(source.initial_reset())
    source.update(rollout, collect_time=0.0)
    source.ppo.learning_rate = 6.667e-4
    for group in source.optimizer.param_groups:
        group["lr"] = source.ppo.learning_rate

    policy_state = {
        key: value.detach().clone() for key, value in source.policy.state_dict().items()
    }
    optimizer_state = source.optimizer.state_dict()
    extra_state = source.extra_checkpoint_state()

    restored = Beyondmimic(_config(), _FakeBeyondMimicEnv(), simulation_app=None)
    restored.build()
    restored.policy.load_state_dict(policy_state)
    restored.optimizer.load_state_dict(optimizer_state)
    restored.load_extra_checkpoint_state(extra_state)

    for key, expected in policy_state.items():
        torch.testing.assert_close(restored.policy.state_dict()[key], expected)
    assert restored.actor_obs_normalizer.count.item() == env.num_envs * 24
    assert restored.critic_obs_normalizer.count.item() == env.num_envs * 24
    assert restored.ppo.learning_rate == pytest.approx(6.667e-4)
    assert restored.optimizer.param_groups[0]["lr"] == pytest.approx(6.667e-4)
    assert restored.optimizer.state_dict()["state"]


def test_beyondmimic_checkpoint_schema_and_rsl_version_are_strict() -> None:
    algo = Beyondmimic(_config(), _FakeBeyondMimicEnv(), simulation_app=None)
    algo.build()
    state = algo.extra_checkpoint_state()

    with pytest.raises(ValueError, match="schema mismatch"):
        algo.load_extra_checkpoint_state({**state, "beyondmimic_schema_version": -1})
    with pytest.raises(ValueError, match="RSL-RL version mismatch"):
        algo.load_extra_checkpoint_state({**state, "rsl_rl_version": "different"})

    algo.optimizer.param_groups[0]["lr"] = 3.21e-4
    no_lr_state = dict(state)
    no_lr_state.pop("learning_rate")
    algo.load_extra_checkpoint_state(no_lr_state)
    assert algo.ppo.learning_rate == pytest.approx(3.21e-4)
