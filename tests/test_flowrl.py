from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from algorithms.flowrl import FlowRL, TensorReplayBuffer, expectile_loss, soft_update
from networks.flowrl import FlowRLActor


class FakeEnv:
    def __init__(self) -> None:
        self.num_envs = 8
        self.observation_dim = 12
        self.critic_observation_dim = 16
        self.action_dim = 3
        self.device = torch.device("cpu")
        self.max_episode_steps = 20
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long)
        self._generator = torch.Generator().manual_seed(42)
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
        terminal_obs = self._observation()
        next_obs = self._observation()
        reward = -action.pow(2).mean(dim=-1)
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        timeout = torch.zeros_like(done)
        if self._step % 3 == 0:
            done[0] = True
            done[1] = True
            timeout[1] = True
        done_terms = {
            "time_out": timeout,
            "motion_complete": torch.zeros_like(done),
            "anchor_pos_bad": torch.zeros_like(done),
            "anchor_ori_bad": torch.zeros_like(done),
            "ee_body_bad": done & ~timeout,
        }
        self._obs = next_obs
        info = {
            "done_terms": done_terms,
            "reward_terms": {"action_cost": action.pow(2).mean(dim=-1)},
            "final_observation": terminal_obs,
        }
        return next_obs, reward, done, info

    def adaptive_sampling_stats(self) -> dict:
        return {}


def config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        horizon=1,
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
        activation="elu",
        flow_steps=4,
        action_scale=1.0,
        rollout_env_steps=6,
        discount_gamma=0.99,
        target_tau=0.95,
        expectile=0.9,
        w2_lambda=0.1,
        cfm_weight_min=0.001,
        cfm_weight_max=1.0,
        replay_capacity=128,
        replay_batch_size=16,
        gradient_steps_per_update=4,
        policy_delay=2,
        warmup_env_steps=6,
        recent_fraction=0.2,
        recent_window=16,
        exploration_noise=0.0,
        policy_lr=3e-4,
        critic_lr=3e-4,
        value_lr=3e-4,
        weight_decay=0.0,
        critic_weight_decay=0.0,
        empirical_normalization=True,
        init_at_random_ep_len=True,
        max_grad_norm=1.0,
    )


def test_expectile_direction() -> None:
    error = torch.tensor([[2.0], [-2.0]])
    loss, weight = expectile_loss(error, 0.9)
    assert torch.allclose(weight, torch.tensor([[0.9], [0.1]]))
    assert torch.allclose(loss, torch.tensor(2.0))


def test_soft_update_uses_official_source_fraction() -> None:
    source = torch.nn.Linear(2, 1, bias=False)
    target = torch.nn.Linear(2, 1, bias=False)
    source.weight.data.fill_(2.0)
    target.weight.data.zero_()
    soft_update(target, source, tau=0.95)
    assert torch.allclose(target.weight, torch.full_like(target.weight, 1.9))


def test_midpoint_flow_constant_velocity() -> None:
    actor = FlowRLActor(5, 2, (16,), "elu", flow_steps=4, action_scale=1.0)
    obs = torch.randn(7, 5)
    base = torch.zeros(7, 2)
    constant = torch.tensor([[0.25, -0.5]])
    original = actor.velocity
    actor.velocity = lambda obs, action, time: constant.expand(obs.shape[0], -1)
    try:
        action, _ = actor.sample(obs, deterministic=True, base_noise=base)
    finally:
        actor.velocity = original
    assert torch.allclose(action, torch.tanh(constant).expand_as(action), atol=1e-6)


def test_replay_shapes() -> None:
    replay = TensorReplayBuffer(32, 4, 2, "cpu", recent_fraction=0.2, recent_window=8)
    replay.add_batch(
        torch.randn(20, 4),
        torch.randn(20, 2),
        torch.randn(20, 1),
        torch.randn(20, 4),
        torch.ones(20, 1),
    )
    batch = replay.sample(12)
    assert replay.size == 20
    assert replay.source_counts() == (0, 20)
    assert [tuple(item.shape) for item in batch] == [(12, 4), (12, 2), (12, 1), (12, 4), (12, 1)]


def test_end_to_end_rollout_and_update() -> None:
    torch.manual_seed(0)
    env = FakeEnv()
    algorithm = FlowRL(config(), env, None)
    algorithm.build()
    algorithm.initial_reset()
    rollout = algorithm.collect(algorithm.reset_for_update(1))
    assert env._step == 6
    assert algorithm.replay.size == 48
    assert rollout["warmup_actions"] == 48
    assert rollout["actions"].shape == (6, 8, 3)

    actor_before = [parameter.detach().clone() for parameter in algorithm.actor.parameters()]
    metrics = algorithm.update(rollout, collect_time=0.01)
    actor_changed = any(
        not torch.allclose(before, after)
        for before, after in zip(actor_before, algorithm.actor.parameters(), strict=True)
    )
    assert actor_changed
    for key in (
        "flowrl/online_q_loss",
        "flowrl/behavior_q_loss",
        "flowrl/value_loss",
        "flowrl/actor_loss",
        "flowrl/cfm_loss",
        "flowrl/q_pi",
    ):
        assert math.isfinite(metrics[key]), key
    assert metrics["flowrl/critic_updates"] == 4.0
    assert metrics["flowrl/actor_updates"] == 2.0
    assert metrics["budget/physical_transitions"] == 48.0
    assert metrics["budget/replay_samples"] == 64.0
    assert algorithm.optimizer is algorithm.actor_optimizer
    assert algorithm.online_q_optimizer is not algorithm.behavior_q_optimizer
    assert algorithm.behavior_q_optimizer is not algorithm.value_optimizer

    deterministic_1 = algorithm.deterministic_actions(env._obs)
    deterministic_2 = algorithm.deterministic_actions(env._obs)
    assert deterministic_1.shape == (8, 1, 3)
    assert torch.allclose(deterministic_1, deterministic_2)


if __name__ == "__main__":
    tests = (
        test_expectile_direction,
        test_soft_update_uses_official_source_fraction,
        test_midpoint_flow_constant_velocity,
        test_replay_shapes,
        test_end_to_end_rollout_and_update,
    )
    for test in tests:
        print(f"[{test.__name__}]")
        test()
    print(f"All {len(tests)} FlowRL tests passed.")
