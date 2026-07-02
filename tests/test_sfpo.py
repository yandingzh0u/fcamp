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


class _ChunkDoneEnv:
    """Mock env: env0 dies (failure, not timeout) on the 2nd step call of the
    rollout, then is reset at chunk end. All other envs never die. Constant
    reward=1 so chunk returns are easy to check against discounted sums."""

    def __init__(self, num_envs=4, obs_dim=8, critic_dim=8, action_dim=3):
        self.num_envs = num_envs
        self.observation_dim = obs_dim
        self.critic_observation_dim = critic_dim
        self.action_dim = action_dim
        self.device = torch.device("cpu")
        self.max_episode_steps = 20
        self.dt = 0.02
        self.phase_steps = torch.zeros(num_envs, dtype=torch.long)
        self._call = 0
        self._gen = torch.Generator().manual_seed(7)
        self.step_auto_reset_flags = []
        self.reset_env_id_history = []

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
        self.reset_env_id_history.append(env_ids.detach().cpu().tolist())
        return self._obs()[env_ids]

    def step(self, action, auto_reset=False):
        self.step_auto_reset_flags.append(bool(auto_reset))
        self._call += 1
        reward = torch.ones(self.num_envs)
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        if self._call == 2:
            done[0] = True
        time_out = torch.zeros(self.num_envs, dtype=torch.bool)
        done_terms = {
            "time_out": time_out,
            "anchor_pos_bad": torch.zeros(self.num_envs, dtype=torch.bool),
            "anchor_ori_bad": torch.zeros(self.num_envs, dtype=torch.bool),
            "ee_body_bad": done.clone(),
        }
        info = {
            "done_terms": done_terms,
            "reward_terms": {},
            "termination_phase_steps": self.phase_steps.clone(),
        }
        self.phase_steps += 1
        return self._obs(), reward, done, info


def test_sfpo_h4_chunk_internal_done_mask_no_first_life() -> None:
    torch.manual_seed(0)
    env = _ChunkDoneEnv(num_envs=4, obs_dim=8, critic_dim=8, action_dim=3)
    cfg = SimpleNamespace(
        horizon=4, rollout_env_steps=8, flow_steps=2, sde_eta=0.7, init_noise_std=0.8,
        action_squash_scale=5.0, eval_initial_noise="zero",
        actor_hidden_dims=(16, 16), critic_hidden_dims=(16, 16), activation="elu",
        terminal_penalty=0.0, discount_gamma=0.99, gae_lambda=0.95,
        clip_range=0.2, adv_clip_max=5.0, desired_kl=0.01, policy_epochs=2,
        num_mini_batches=2, micro_batch_size=64, value_loss_coef=1.0,
        value_clip_range=0.2, use_clipped_value_loss=True,
        policy_lr=1e-3, value_lr=1e-3, weight_decay=0.0, critic_weight_decay=0.0,
        empirical_normalization=False, init_at_random_ep_len=False, max_grad_norm=1.0,
    )
    algo = SFPO(cfg=cfg, env=env, simulation_app=None)
    algo.build()
    obs = algo.initial_reset()
    rollout = algo.collect(obs)

    gamma = 0.99
    # env0 died at frame_idx=1 of chunk0 -> 2 live frames, dead frames masked.
    assert int(rollout["live_frames"][0, 0, 0].item()) == 2
    assert rollout["alive_frame"][0, 0].tolist() == [True, True, False, False]
    # Dead frames' reward not counted: chunk_reward == 1 + gamma.
    assert abs(float(rollout["rewards"][0, 0, 0].item()) - (1.0 + gamma)) < 1e-5
    # valid_mask all True across chunks (NO first-life truncation).
    assert bool(rollout["valid_mask"].all().item())
    # SFPO h4 should not auto-reset mid-chunk; it resets dead envs at chunk end.
    assert env.step_auto_reset_flags == [False] * 8
    assert env.reset_env_id_history == [[0]]
    # env1 unaffected: full chunk alive.
    assert int(rollout["live_frames"][0, 1, 0].item()) == 4
    assert rollout["alive_frame"][0, 1].tolist() == [True, True, True, True]
    # After chunk-end reset, env0 is alive again in chunk1 (continuous flow).
    assert bool(rollout["alive_frame"][1, 0, 0].item())
    assert rollout["alive_frame"][1, 0].tolist() == [True, True, True, True]
    assert abs(float(rollout["rewards"][1, 0, 0].item()) - (1.0 + gamma + gamma**2 + gamma**3)) < 1e-5
