from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from algorithms.reinflow import ReinFlow
from networks.reinflow import ReinFlowPolicy


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
        self._obs = self._observation()
        self._critic_obs = self._critic_observation()

    def _observation(self) -> torch.Tensor:
        return torch.randn(self.num_envs, self.observation_dim, generator=self._generator)

    def _critic_observation(self) -> torch.Tensor:
        return torch.randn(self.num_envs, self.critic_observation_dim, generator=self._generator)

    def reset(self) -> torch.Tensor:
        self.episode_steps.zero_()
        self._obs = self._observation()
        self._critic_obs = self._critic_observation()
        return self._obs

    def get_critic_observation(self) -> torch.Tensor:
        return self._critic_obs

    def sample_phase_indices(self, count: int, horizon: int) -> torch.Tensor:
        return torch.zeros(count, dtype=torch.long)

    def reset_envs(self, env_ids: torch.Tensor, phase_indices: torch.Tensor) -> torch.Tensor:
        replacement = self._observation()[env_ids]
        self._obs[env_ids] = replacement
        self._critic_obs[env_ids] = self._critic_observation()[env_ids]
        self.episode_steps[env_ids] = 0
        return replacement

    def step(self, action: torch.Tensor, auto_reset: bool = False):
        assert auto_reset is False
        assert action.shape == (self.num_envs, self.action_dim)
        self._step += 1
        self.episode_steps += 1
        reward = -0.1 * action.pow(2).mean(dim=-1)
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        timeout = torch.zeros_like(done)
        if self._step % 5 == 0:
            done[0] = True
            done[1] = True
            timeout[1] = True
        self._obs = self._observation()
        self._critic_obs = self._critic_observation()
        done_terms = {
            "time_out": timeout,
            "motion_complete": torch.zeros_like(done),
            "anchor_pos_bad": torch.zeros_like(done),
            "anchor_ori_bad": torch.zeros_like(done),
            "ee_body_bad": done & ~timeout,
        }
        return self._obs, reward, done, {
            "done_terms": done_terms,
            "reward_terms": {"action_cost": action.pow(2).mean(dim=-1)},
        }

    def adaptive_sampling_stats(self) -> dict:
        return {}


def config() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        horizon=4,
        actor_hidden_dims=(32, 16),
        noise_hidden_dims=(16,),
        critic_hidden_dims=(32, 16),
        activation="mish",
        flow_steps=4,
        timestep_embed_dim=8,
        action_scale=1.0,
        min_denoising_std=0.1,
        max_denoising_std=0.24,
        randn_clip_value=3.0,
        logprob_min=-1.0,
        logprob_max=1.0,
        account_for_initial_stochasticity=True,
        normalize_denoising_horizon=True,
        normalize_action_dimension=True,
        rollout_env_steps=8,
        discount_gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.01,
        target_kl=1.0,
        policy_epochs=2,
        num_mini_batches=2,
        entropy_coef=0.03,
        value_loss_coef=1.0,
        policy_lr=4.5e-5,
        value_lr=6.5e-4,
        weight_decay=0.0,
        critic_weight_decay=1e-5,
        empirical_normalization=True,
        init_at_random_ep_len=True,
        max_grad_norm=1.0,
        pretrained_actor_path="",
    )


def policy() -> ReinFlowPolicy:
    return ReinFlowPolicy(
        obs_dim=12,
        action_dim=3,
        horizon=4,
        hidden_dims=(32, 16),
        noise_hidden_dims=(16,),
        activation="mish",
        flow_steps=4,
        timestep_embed_dim=8,
        action_scale=1.0,
        min_std=0.1,
        max_std=0.24,
        randn_clip_value=3.0,
    )


def test_chain_onpolicy_ratio_is_one() -> None:
    torch.manual_seed(0)
    actor = policy()
    obs = torch.randn(7, 12)
    _, chains, old_log_prob, _ = actor.sample_chain(obs)
    new_log_prob, _, _ = actor.chain_log_prob(obs, chains)
    ratio = torch.exp(new_log_prob - old_log_prob)
    assert torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6)


def test_noise_bounds_and_both_heads_receive_gradients() -> None:
    torch.manual_seed(1)
    actor = policy()
    obs = torch.randn(6, 12)
    _, chains, old_log_prob, _ = actor.sample_chain(obs)
    new_log_prob, entropy, stats = actor.chain_log_prob(obs, chains)
    assert float(stats["transition_stds"].min().item()) >= 0.1 - 1e-6
    assert float(stats["transition_stds"].max().item()) <= 0.24 + 1e-6
    loss = -(new_log_prob + 0.03 * entropy).mean()
    loss.backward()
    velocity_has_grad = any(
        parameter.grad is not None and bool((parameter.grad.abs() > 0).any())
        for parameter in actor.velocity_net.parameters()
    )
    noise_has_grad = any(
        parameter.grad is not None and bool((parameter.grad.abs() > 0).any())
        for parameter in actor.noise_net.parameters()
    )
    assert velocity_has_grad
    assert noise_has_grad
    assert torch.allclose(old_log_prob, new_log_prob.detach(), atol=1e-6)


def test_deterministic_chain_is_repeatable() -> None:
    actor = policy()
    obs = torch.randn(5, 12)
    action_1, _, _, _ = actor.sample_chain(obs, deterministic=True)
    action_2, _, _, _ = actor.sample_chain(obs, deterministic=True)
    assert action_1.shape == (5, 4, 3)
    assert torch.allclose(action_1, action_2)


def test_end_to_end_h4_rollout_update() -> None:
    torch.manual_seed(2)
    env = FakeEnv()
    algorithm = ReinFlow(config(), env, None)
    algorithm.build()
    algorithm.initial_reset()
    rollout = algorithm.collect(algorithm.reset_for_update(1))
    assert env._step == 8
    assert rollout["chains"].shape == (2, 8, 5, 4, 3)
    assert rollout["actions"].shape == (2, 8, 4, 3)
    assert rollout["advantages"].shape == (2, 8)
    assert bool(torch.isfinite(rollout["advantages"]).all())

    before = [parameter.detach().clone() for parameter in algorithm.actor.parameters()]
    metrics = algorithm.update(rollout, collect_time=0.01)
    changed = any(
        not torch.allclose(old, new)
        for old, new in zip(before, algorithm.actor.parameters(), strict=True)
    )
    assert changed
    for key in (
        "reinflow/policy_loss",
        "reinflow/value_loss",
        "reinflow/ratio",
        "reinflow/approx_kl",
        "reinflow/noise_std",
    ):
        assert math.isfinite(metrics[key]), key
    assert abs(metrics["reinflow/first_ratio"] - 1.0) < 1e-5
    assert metrics["budget/physical_transitions"] == 64.0
    assert metrics["budget/policy_decisions"] == 16.0
    assert metrics["budget/chain_transitions"] == 64.0
    assert algorithm.optimizer is algorithm.actor_optimizer
    assert algorithm.actor_optimizer is not algorithm.critic_optimizer


if __name__ == "__main__":
    tests = (
        test_chain_onpolicy_ratio_is_one,
        test_noise_bounds_and_both_heads_receive_gradients,
        test_deterministic_chain_is_repeatable,
        test_end_to_end_h4_rollout_update,
    )
    for test in tests:
        print(f"[{test.__name__}]")
        test()
    print(f"All {len(tests)} ReinFlow tests passed.")
