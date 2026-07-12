from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from algorithms.policyflow import PolicyFlow
from networks.policyflow import PolicyFlowActor


class FakeEnv:
    def __init__(self) -> None:
        self.num_envs = 8
        self.observation_dim = 12
        self.critic_observation_dim = 16
        self.action_dim = 3
        self.device = torch.device("cpu")
        self.max_episode_steps = 20
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long)
        self._generator = torch.Generator().manual_seed(41)
        self._step = 0

    def _actor_observation(self) -> torch.Tensor:
        return torch.randn(self.num_envs, self.observation_dim, generator=self._generator)

    def _critic_observation(self) -> torch.Tensor:
        return torch.randn(self.num_envs, self.critic_observation_dim, generator=self._generator)

    def reset(self) -> torch.Tensor:
        self.episode_steps.zero_()
        self._obs = self._actor_observation()
        self._critic_obs = self._critic_observation()
        return self._obs

    def get_critic_observation(self) -> torch.Tensor:
        return self._critic_obs

    def step(self, action: torch.Tensor, auto_reset: bool = True):
        assert action.shape == (self.num_envs, self.action_dim)
        self._step += 1
        final_obs = self._actor_observation()
        final_critic_obs = self._critic_observation()
        self._obs = self._actor_observation()
        self._critic_obs = self._critic_observation()
        reward = 1.0 - 0.05 * action.square().mean(-1)
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
            "final_observation": final_obs,
            "final_critic_observation": final_critic_obs,
        }
        return self._obs, reward, done, info

    def adaptive_sampling_stats(self) -> dict:
        return {}


def config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        horizon=1,
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
        activation="elu",
        flow_steps=4,
        timestep_embed_dim=8,
        init_noise_std=0.5,
        rollout_env_steps=6,
        discount_gamma=0.99,
        gae_lambda=0.95,
        num_learning_epochs=2,
        num_mini_batches=2,
        clip_range=0.2,
        value_clip_range=0.2,
        gaussian_entropy_coef=0.002,
        brownian_reg_coef=0.006,
        value_loss_coef=1.0,
        desired_kl=0.01,
        actor_learning_rate=3e-4,
        critic_learning_rate=1e-3,
        weight_decay=0.0,
        critic_weight_decay=0.0,
        empirical_normalization=True,
        init_at_random_ep_len=True,
        max_grad_norm=1.0,
    )


def test_old_snapshot_starts_with_unit_ratio() -> None:
    torch.manual_seed(0)
    actor = PolicyFlowActor(7, 3, (16,), "elu", 4, 8, 0.5)
    observation = torch.randn(11, 7)
    with torch.no_grad():
        _, info = actor.sample_action(observation)
    delta_velocity, std, _ = actor.flow_variation(
        observation,
        info["prior"],
        info["base_noise"],
        compute_brownian=True,
        time_index=torch.arange(11) % 9,
    )
    new_log_prob = torch.distributions.Normal(delta_velocity, std).log_prob(
        info["delta"]
    ).sum(-1)
    assert torch.allclose(delta_velocity, torch.zeros_like(delta_velocity), atol=1e-7)
    assert torch.allclose(torch.exp(new_log_prob - info["log_prob"]), torch.ones(11))


def test_midpoint_integration_for_constant_velocity() -> None:
    actor = PolicyFlowActor(5, 2, (16,), "elu", 4, 8, 0.5)
    observation = torch.randn(6, 5)
    base = torch.randn(6, 2)
    constant = torch.tensor([[0.2, -0.4]])
    original = actor.current.forward
    actor.current.forward = lambda obs, action, time: constant.expand(obs.shape[0], -1)
    try:
        result, _ = actor.sample_prior(observation, base)
    finally:
        actor.current.forward = original
    assert torch.allclose(result, base + constant.expand_as(base), atol=1e-6)


def test_end_to_end_rollout_and_update() -> None:
    torch.manual_seed(0)
    env = FakeEnv()
    algorithm = PolicyFlow(config(), env, None)
    algorithm.build()
    algorithm.initial_reset()
    rollout = algorithm.collect(algorithm.reset_for_update(1))
    assert rollout["actions"].shape == (6, 8, 3)
    before = [
        parameter.detach().clone()
        for parameter in algorithm.actor.parameters()
        if parameter.requires_grad
    ]
    metrics = algorithm.update(rollout, collect_time=0.01)
    after = [parameter for parameter in algorithm.actor.parameters() if parameter.requires_grad]
    assert any(
        not torch.allclose(old, new)
        for old, new in zip(before, after, strict=True)
    )
    for key in (
        "policyflow/policy_loss",
        "policyflow/value_loss",
        "policyflow/brownian_loss",
        "policyflow/kl",
        "policyflow/noise_std",
    ):
        assert math.isfinite(metrics[key]), key
    assert metrics["policyflow/optimization_steps"] == 4.0
    assert metrics["budget/physical_transitions"] == 48.0
    deterministic_1 = algorithm.deterministic_actions(env._obs)
    deterministic_2 = algorithm.deterministic_actions(env._obs)
    assert deterministic_1.shape == (8, 1, 3)
    assert torch.allclose(deterministic_1, deterministic_2)

