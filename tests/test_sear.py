from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from algorithms.sear import SEAR, SequenceReplayBuffer, discounted_prefix_sum
from networks.sear import CausalDistributionalCritic, SEARTwinCritic


class FakeEnv:
    def __init__(self) -> None:
        self.num_envs = 8
        self.observation_dim = 12
        self.critic_observation_dim = 16
        self.action_dim = 3
        self.device = torch.device("cpu")
        self.max_episode_steps = 30
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long)
        self._generator = torch.Generator().manual_seed(17)
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
        reward = 1.0 - 0.1 * action.square().mean(-1)
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        timeout = torch.zeros_like(done)
        if self._step == 5:
            done[0] = True
        info = {
            "done_terms": {
                "time_out": timeout,
                "motion_complete": torch.zeros_like(done),
                "anchor_pos_bad": torch.zeros_like(done),
                "anchor_ori_bad": torch.zeros_like(done),
                "ee_body_bad": done,
            },
            "reward_terms": {"tracking": reward},
            "final_observation": final_observation,
        }
        return self._obs, reward, done, info

    def adaptive_sampling_stats(self) -> dict:
        return {}


def config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        horizon=4,
        actor_hidden_dim=32,
        actor_num_blocks=1,
        critic_hidden_dim=32,
        critic_num_heads=4,
        critic_num_blocks=1,
        num_value_bins=21,
        value_min=-10.0,
        value_max=50.0,
        action_scale=1.0,
        rollout_env_steps=8,
        discount_gamma=0.99,
        target_tau=0.05,
        replay_capacity=256,
        replay_batch_size=8,
        gradient_steps_per_update=2,
        init_alpha=0.1,
        target_entropy_scale=1.0,
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        weight_decay=1e-4,
        empirical_normalization=True,
        init_at_random_ep_len=True,
        max_grad_norm=1.0,
    )


def test_discounted_prefix_sum() -> None:
    values = torch.ones(2, 4)
    result = discounted_prefix_sum(values, 0.5)
    expected = torch.tensor([1.0, 1.5, 1.75, 1.875]).expand_as(result)
    assert torch.allclose(result, expected)


def test_sequence_replay_never_crosses_episode_boundary() -> None:
    replay = SequenceReplayBuffer(64, 2, 2, 1, 3, "cpu")
    for step in range(7):
        observation = torch.full((2, 2), float(step))
        action = torch.full((2, 1), float(step))
        reward = torch.full((2,), float(step))
        continuation = torch.ones(2, dtype=torch.bool)
        if step == 2:
            continuation[0] = False
        replay.add_row(
            observation,
            action,
            reward,
            observation + 1.0,
            torch.ones(2),
            continuation,
        )
    _, action_chunks, _, _, _ = replay.sample(64)
    differences = action_chunks[:, 1:, 0] - action_chunks[:, :-1, 0]
    assert torch.allclose(differences, torch.ones_like(differences))


def test_causal_critic_prefix_is_future_action_invariant() -> None:
    torch.manual_seed(0)
    critic = CausalDistributionalCritic(7, 3, 4, 32, 4, 1, 11)
    observation = torch.randn(5, 7)
    action = torch.randn(5, 4, 3)
    changed = action.clone()
    changed[:, 2:] += 100.0
    original_logits = critic(observation, action)
    changed_logits = critic(observation, changed)
    assert torch.allclose(original_logits[:, :2], changed_logits[:, :2], atol=1e-6)


def test_distributional_projection_is_normalized() -> None:
    critic = SEARTwinCritic(7, 3, 4, 32, 4, 1, 11, -10.0, 50.0)
    values = torch.tensor([[-20.0, -5.5, 7.0, 70.0]])
    distribution = critic.target_distribution(values)
    assert torch.allclose(distribution.sum(-1), torch.ones_like(values))
    projected = (distribution * critic.support).sum(-1)
    assert torch.allclose(projected, values.clamp(-10.0, 50.0), atol=1e-5)


def test_end_to_end_rollout_and_update() -> None:
    torch.manual_seed(0)
    env = FakeEnv()
    algorithm = SEAR(config(), env, None)
    algorithm.build()
    algorithm.initial_reset()
    rollout = algorithm.collect(algorithm.reset_for_update(1))
    assert rollout["actions"].shape == (8, 8, 3)
    assert 0 < rollout["policy_decisions"] <= 64
    before = [parameter.detach().clone() for parameter in algorithm.actor.parameters()]
    metrics = algorithm.update(rollout, collect_time=0.01)
    assert any(
        not torch.allclose(old, new)
        for old, new in zip(before, algorithm.actor.parameters(), strict=True)
    )
    for key in (
        "sear/critic_loss",
        "sear/actor_loss",
        "sear/alpha",
        "sear/q_mean",
        "sear/target_mean",
    ):
        assert math.isfinite(metrics[key]), key
    assert metrics["sear/critic_updates"] == 2.0
    assert metrics["sear/actor_updates"] == 2.0
    assert metrics["budget/physical_transitions"] == 64.0
    assert metrics["budget/replay_frames"] == 64.0
    assert metrics["sear/utd_frames"] == 1.0
    deterministic = algorithm.deterministic_actions(env._obs)
    assert deterministic.shape == (8, 4, 3)

