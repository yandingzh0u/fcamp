from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from engine.config import AdaMimicConfig, load_config
from method import load_method_class
from method.adamimic import AdaMimic
from models.adamimic_policy import AdaMimicActorCritic


ROOT = Path(__file__).resolve().parents[1]


def test_adamimic_config_and_loader() -> None:
    cfg = load_config(ROOT / "configs" / "adamimic_stage1_largebox.yaml")
    assert cfg.method == "adamimic"
    assert isinstance(cfg.parameters, AdaMimicConfig)
    assert cfg.parameters.stage == "stage1"
    assert cfg.parameters.rollout_env_steps == 75
    assert cfg.parameters.actor_time_scale_range == (0.0, 0.0)
    assert cfg.environment.reset_phase_sampling == "rsi"
    assert cfg.environment.adaptive_motion_sampling is False
    assert load_method_class("adamimic") is AdaMimic


def test_adamimic_stage2_checkpoint_path_is_resolved() -> None:
    cfg = load_config(ROOT / "configs" / "adamimic_stage2_largebox.yaml")
    assert cfg.parameters.stage == "stage2"
    assert cfg.parameters.residual_delta is True
    assert cfg.environment.reset_phase_sampling == "zero"
    assert cfg.parameters.checkpoint_path.endswith("runs/adamimic_stage1_largebox/checkpoints/last.pt")


def test_adamimic_stage1_rejects_adaptive_time_overrides() -> None:
    with pytest.raises(ValueError, match="stage1"):
        load_config(
            ROOT / "configs" / "adamimic_stage1_largebox.yaml",
            [
                "parameters.train_time=true",
                "parameters.actor_time_scale_range=[-0.015, 0.02]",
            ],
        )


def test_adamimic_stage1_rejects_non_official_reset_sampling() -> None:
    with pytest.raises(ValueError, match="RSI"):
        load_config(
            ROOT / "configs" / "adamimic_stage1_largebox.yaml",
            [
                "environment.reset_phase_sampling=adaptive",
                "environment.adaptive_motion_sampling=true",
            ],
        )


def test_adamimic_actor_critic_two_level_action_shape() -> None:
    torch.manual_seed(0)
    model = AdaMimicActorCritic(
        actor_obs_dim=8,
        critic_obs_dim=7,
        control_action_dim=3,
        actor_hidden_dims=(16, 8),
        critic_hidden_dims=(16, 8),
        activation="elu",
        init_noise_std=0.5,
        infer_keyframe_time=True,
        actor_time_scale_range=(-0.01, 0.02),
        fixed_dt=0.02,
        time_min_std=0.005,
    )
    obs = torch.randn(5, 8)
    critic = torch.randn(5, 7)
    action = model.act(obs)
    assert action.shape == (5, 4)
    low_logp, high_logp = model.get_actions_log_prob(action)
    assert low_logp.shape == (5,)
    assert high_logp.shape == (5,)
    assert model.evaluate_low(critic, action[:, -1:]).shape == (5,)
    assert model.evaluate_high(critic).shape == (5,)
    assert model.act_inference(obs).shape == (5, 4)
    assert model.entropy.shape == (5,)
    assert torch.allclose(model.entropy, model.distribution.entropy().sum(dim=-1))


class _MockMimicEnv:
    def __init__(self, num_envs=4, obs_dim=8, critic_dim=7, action_dim=3):
        self.num_envs = num_envs
        self.observation_dim = obs_dim
        self.critic_observation_dim = critic_dim
        self.action_dim = action_dim
        self.device = torch.device("cpu")
        self.max_episode_steps = 20
        self.phase_steps = torch.zeros(num_envs, dtype=torch.float32)
        self.episode_steps = torch.zeros(num_envs, dtype=torch.long)
        self._gen = torch.Generator().manual_seed(123)
        self._step = 0

    def reset(self, phase_indices=None):
        if phase_indices is None:
            self.phase_steps.zero_()
        else:
            self.phase_steps.copy_(phase_indices.to(dtype=self.phase_steps.dtype))
        self.episode_steps.zero_()
        return self._obs()

    def get_critic_observation(self):
        return torch.randn(self.num_envs, self.critic_observation_dim, generator=self._gen)

    def adaptive_sampling_stats(self):
        return {"top_bin": 0.0, "top_prob": 0.0, "failed_sum": 0.0, "entropy": 0.0, "peak_bin": 0.0}

    def _obs(self):
        return torch.randn(self.num_envs, self.observation_dim, generator=self._gen)

    def step(self, action, auto_reset=False, reference_dt=None):
        assert action.shape == (self.num_envs, self.action_dim)
        assert reference_dt is not None
        self._step += 1
        self.phase_steps += torch.as_tensor(reference_dt).reshape(self.num_envs) / 0.02
        self.episode_steps += 1
        reward = 0.05 - 0.001 * action.square().sum(dim=-1)
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        if self._step % 3 == 0:
            done[1] = True
        time_out = torch.zeros(self.num_envs, dtype=torch.bool)
        anchor_pos_bad = torch.zeros(self.num_envs, dtype=torch.bool)
        anchor_ori_bad = torch.zeros(self.num_envs, dtype=torch.bool)
        ee_body_bad = done.clone()
        if auto_reset and bool(done.any()):
            ids = done.nonzero(as_tuple=False).squeeze(-1)
            self.phase_steps[ids] = 0
            self.episode_steps[ids] = 0
        info = {
            "done_terms": {
                "time_out": time_out,
                "motion_complete": torch.zeros_like(done),
                "anchor_pos_bad": anchor_pos_bad,
                "anchor_ori_bad": anchor_ori_bad,
                "ee_body_bad": ee_body_bad,
            },
            "reward_terms": {
                "action_rate": action.square().sum(dim=-1),
            },
            "termination_phase_steps": self.phase_steps.clone(),
            "reference_frame_delta": torch.as_tensor(reference_dt).reshape(self.num_envs) / 0.02,
        }
        return self._obs(), reward, done, info


def _adamimic_test_cfg(**overrides):
    values = dict(
        stage="stage1",
        actor_hidden_dims=(16, 8),
        critic_hidden_dims=(16, 8),
        activation="elu",
        rollout_env_steps=5,
        discount_gamma=0.99,
        time_discount_gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        desired_kl=0.01,
        policy_epochs=2,
        num_mini_batches=2,
        micro_batch_size=4,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        policy_lr=1e-3,
        weight_decay=0.01,
        empirical_normalization=False,
        init_at_random_ep_len=False,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        infer_keyframe_time=True,
        actor_time_scale_range=(0.0, 0.0),
        fixed_dt=0.02,
        time_min_std=0.005,
        init_noise_std=0.5,
        train_time=False,
        time_reward_scale=1.0,
        use_timeout_bootstrap=True,
        use_smooth=True,
        smoothness_upper_bound=1.0,
        smoothness_lower_bound=0.1,
        value_smoothness_coef=0.01,
        residual_delta=False,
        checkpoint_path="",
        freeze_base=False,
        residual_time_threshold=0.0,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_adamimic_collect_and_update_end_to_end() -> None:
    torch.manual_seed(0)
    env = _MockMimicEnv()
    algo = AdaMimic(cfg=_adamimic_test_cfg(), env=env, simulation_app=None)
    algo.build()
    obs = algo.initial_reset()
    rollout = algo.collect(obs)
    assert rollout["actions"].shape == (5, env.num_envs, env.action_dim + 1)
    metrics = algo.update(rollout, collect_time=0.1)
    assert math.isfinite(metrics["adamimic/loss"])
    assert math.isfinite(metrics["adamimic/value_low_loss"])
    assert metrics["adamimic/sample_count"] == 16.0
    assert metrics["policy/action_delta"] >= 0.0
    assert algo.deterministic_actions(rollout["next_observation"]).shape == (env.num_envs, env.action_dim)
