from __future__ import annotations

from types import SimpleNamespace

import torch

from algorithms.sfpo import SFPO
from networks.flow_policy import FlowMatchingPolicy


def test_sfpo_chunk_gae_uses_horizon_scaled_discount() -> None:
    cfg = SimpleNamespace(horizon=2, discount_gamma=0.9, gae_lambda=1.0)
    algo = SFPO(cfg=cfg, env=None, simulation_app=None)
    rewards = torch.ones(3, 1, 1)
    values = torch.zeros(3, 1, 1)
    dones = torch.zeros(3, 1, 1, dtype=torch.bool)
    last_values = torch.zeros(1, 1)

    returns, advantages = algo._compute_chunk_gae(last_values, values, dones, rewards)

    chunk_gamma = 0.9 ** 2
    expected = torch.tensor([
        1.0 + chunk_gamma + chunk_gamma ** 2,
        1.0 + chunk_gamma,
        1.0,
    ]).view(3, 1, 1)
    assert torch.allclose(returns, expected, atol=1e-6)
    assert abs(float(advantages.mean().item())) < 1e-6


def test_sfpo_done_blocks_chunk_bootstrap() -> None:
    cfg = SimpleNamespace(horizon=4, discount_gamma=0.9, gae_lambda=1.0)
    algo = SFPO(cfg=cfg, env=None, simulation_app=None)
    rewards = torch.ones(3, 1, 1)
    values = torch.zeros(3, 1, 1)
    dones = torch.tensor([[[False]], [[True]], [[False]]])
    last_values = torch.zeros(1, 1)

    returns, _ = algo._compute_chunk_gae(last_values, values, dones, rewards)

    chunk_gamma = 0.9 ** 4
    assert abs(float(returns[1, 0, 0]) - 1.0) < 1e-6
    assert abs(float(returns[0, 0, 0]) - (1.0 + chunk_gamma)) < 1e-6


def test_sfpo_invalid_chunks_do_not_affect_advantage_normalization() -> None:
    cfg = SimpleNamespace(horizon=2, discount_gamma=0.9, gae_lambda=1.0)
    algo = SFPO(cfg=cfg, env=None, simulation_app=None)
    rewards = torch.tensor([[[1.0]], [[1.0]], [[1000.0]]])
    values = torch.zeros(3, 1, 1)
    dones = torch.tensor([[[False]], [[True]], [[False]]])
    valid = torch.tensor([[[True]], [[True]], [[False]]])
    last_values = torch.zeros(1, 1)

    returns, advantages = algo._compute_chunk_gae(
        last_values,
        values,
        dones,
        rewards,
        valid_mask=valid,
        alive_at_end=torch.zeros(1, 1, dtype=torch.bool),
    )

    chunk_gamma = 0.9 ** 2
    assert abs(float(returns[0, 0, 0]) - (1.0 + chunk_gamma)) < 1e-6
    assert abs(float(returns[1, 0, 0]) - 1.0) < 1e-6
    assert float(advantages[2, 0, 0]) == 0.0


def test_sfpo_flow_log_probs_are_recomputed_on_policy() -> None:
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
    obs = torch.randn(7, 5)
    initial_noise = torch.randn(7, algo.chunk_dim)
    sde_noise = torch.randn(7, cfg.flow_steps, algo.chunk_dim)

    _, latent_path, old_log_probs = algo._sde_ode_rollout_actions(
        obs,
        initial_noise=initial_noise,
        sde_noise=sde_noise,
    )
    recomputed = algo._compute_transition_log_probs(
        obs,
        latent_path,
        torch.arange(cfg.flow_steps),
    )

    assert torch.allclose(recomputed, old_log_probs, atol=1e-5)
