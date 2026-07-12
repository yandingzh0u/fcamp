from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from algorithms.sac_flow import SACFlow
from networks.sac_flow import SACFlowActor


class FakeEnv:
    def __init__(self) -> None:
        self.num_envs = 8
        self.observation_dim = 12
        self.critic_observation_dim = 16
        self.action_dim = 3
        self.device = torch.device("cpu")
        self.max_episode_steps = 20
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long)
        self._generator = torch.Generator().manual_seed(91)
        self._step = 0

    def _observation(self) -> torch.Tensor:
        return torch.randn(self.num_envs, self.observation_dim, generator=self._generator)

    def reset(self) -> torch.Tensor:
        self.episode_steps.zero_()
        self._obs = self._observation()
        return self._obs

    def step(self, action: torch.Tensor, auto_reset: bool = True):
        assert action.shape == (self.num_envs, self.action_dim)
        assert bool((action.abs() <= 1.00001).all())
        self._step += 1
        final_observation = self._observation()
        self._obs = self._observation()
        reward = 1.0 - action.square().mean(-1)
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        timeout = torch.zeros_like(done)
        if self._step % 3 == 0:
            done[0] = True
            done[1] = True
            timeout[1] = True
        info = {
            "done_terms": {
                "time_out": timeout,
                "motion_complete": torch.zeros_like(done),
                "anchor_pos_bad": torch.zeros_like(done),
                "anchor_ori_bad": torch.zeros_like(done),
                "ee_body_bad": done & ~timeout,
            },
            "reward_terms": {"tracking": reward},
            "final_observation": final_observation,
        }
        return self._obs, reward, done, info

    def adaptive_sampling_stats(self) -> dict:
        return {}


def config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        horizon=1,
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
        flow_steps=4,
        timestep_embed_dim=8,
        action_scale=1.0,
        use_batch_renorm=True,
        batch_norm_momentum=0.99,
        rollout_env_steps=6,
        discount_gamma=0.99,
        replay_capacity=128,
        replay_batch_size=16,
        gradient_steps_per_update=4,
        policy_delay=2,
        warmup_env_steps=6,
        init_alpha=0.2,
        target_entropy=0.0,
        policy_lr=3e-4,
        critic_lr=1e-3,
        alpha_lr=1e-3,
        weight_decay=0.0,
        critic_weight_decay=0.0,
        empirical_normalization=True,
        init_at_random_ep_len=True,
        max_grad_norm=1.0,
    )


def test_flow_g_path_is_bounded_and_has_exact_shape() -> None:
    torch.manual_seed(0)
    actor = SACFlowActor(7, 3, 32, 4, 1.0, time_embed_dim=8, log_std_hidden_dims=(16,))
    observation = torch.randn(13, 7)
    action, log_prob, info = actor.sample(observation)
    assert action.shape == (13, 3)
    assert log_prob.shape == (13, 1)
    assert info["path_std"].shape == (13, 4, 3)
    assert info["gate"].shape == (13, 4, 3)
    assert bool((action.abs() <= 1.0).all())
    assert bool(torch.isfinite(log_prob).all())
    assert float(info["gate"].mean()) > 0.98


def test_deterministic_path_repeats_and_stochastic_path_explores() -> None:
    torch.manual_seed(0)
    actor = SACFlowActor(5, 2, 16, 4, 1.0, time_embed_dim=8, log_std_hidden_dims=(16,))
    observation = torch.randn(6, 5)
    deterministic_1, _, _ = actor.sample(observation, deterministic=True)
    deterministic_2, _, _ = actor.sample(observation, deterministic=True)
    stochastic_1, _, _ = actor.sample(observation)
    stochastic_2, _, _ = actor.sample(observation)
    assert torch.allclose(deterministic_1, deterministic_2)
    assert not torch.allclose(stochastic_1, stochastic_2)


def test_end_to_end_rollout_and_update() -> None:
    torch.manual_seed(0)
    env = FakeEnv()
    algorithm = SACFlow(config(), env, None)
    algorithm.build()
    algorithm.initial_reset()
    rollout = algorithm.collect(algorithm.reset_for_update(1))
    assert rollout["actions"].shape == (6, 8, 3)
    assert rollout["warmup_actions"] == 48
    before = [parameter.detach().clone() for parameter in algorithm.actor.parameters()]
    metrics = algorithm.update(rollout, collect_time=0.01)
    assert any(
        not torch.allclose(old, new)
        for old, new in zip(before, algorithm.actor.parameters(), strict=True)
    )
    for key in (
        "sac_flow/q_loss",
        "sac_flow/actor_loss",
        "sac_flow/alpha",
        "sac_flow/log_prob",
        "sac_flow/path_std",
    ):
        assert math.isfinite(metrics[key]), key
    assert metrics["sac_flow/critic_updates"] == 4.0
    assert metrics["sac_flow/actor_updates"] == 2.0
    assert metrics["budget/physical_transitions"] == 48.0
    assert metrics["budget/replay_samples"] == 64.0
    deterministic_1 = algorithm.deterministic_actions(env._obs)
    deterministic_2 = algorithm.deterministic_actions(env._obs)
    assert deterministic_1.shape == (8, 1, 3)
    assert torch.allclose(deterministic_1, deterministic_2)

