from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import torch
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from algorithms.fql import FQL, aggregate_twin_q
from core.offline_dataset import save_offline_transition_dataset, validate_offline_transition_payload


class FakeEnv:
    def __init__(self) -> None:
        self.num_envs = 8
        self.observation_dim = 12
        self.critic_observation_dim = 16
        self.action_dim = 3
        self.device = torch.device("cpu")
        self.max_episode_steps = 20
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long)
        self._generator = torch.Generator().manual_seed(53)
        self._step = 0
        self._obs = self._observation()

    def _observation(self) -> torch.Tensor:
        return torch.randn(self.num_envs, self.observation_dim, generator=self._generator)

    def reset(self) -> torch.Tensor:
        self.episode_steps.zero_()
        self._obs = self._observation()
        return self._obs

    def step(self, action: torch.Tensor, auto_reset: bool = True):
        assert auto_reset is True
        assert action.shape == (self.num_envs, self.action_dim)
        assert bool((action.abs() <= 1.00001).all())
        self._step += 1
        terminal_obs = self._observation()
        next_obs = self._observation()
        reward = 0.2 - action.pow(2).mean(dim=-1)
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
        return next_obs, reward, done, {
            "done_terms": done_terms,
            "reward_terms": {"action_cost": action.pow(2).mean(dim=-1)},
            "final_observation": terminal_obs,
        }

    def adaptive_sampling_stats(self) -> dict:
        return {}


def config(
    offline_dataset_path: str = "",
    offline_pretrain_gradient_steps: int = 0,
    environment_action_scale: float = 1.0,
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        horizon=1,
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(32, 16),
        activation="gelu",
        actor_layer_norm=False,
        critic_layer_norm=True,
        flow_steps=4,
        action_scale=1.0,
        environment_action_scale=environment_action_scale,
        rollout_env_steps=6,
        discount_gamma=0.99,
        target_tau=0.005,
        q_aggregation="mean",
        alpha=10.0,
        normalize_q_loss=True,
        offline_dataset_path=offline_dataset_path,
        offline_pretrain_gradient_steps=offline_pretrain_gradient_steps,
        offline_pretrain_log_every=1,
        replay_capacity=128,
        replay_batch_size=16,
        gradient_steps_per_update=4,
        warmup_env_steps=6,
        recent_fraction=0.0,
        recent_window=16,
        flow_lr=3e-4,
        policy_lr=3e-4,
        critic_lr=3e-4,
        weight_decay=0.0,
        critic_weight_decay=0.0,
        empirical_normalization=True,
        init_at_random_ep_len=True,
        max_grad_norm=1.0,
    )


def build_algorithm(cfg: types.SimpleNamespace | None = None) -> FQL:
    algorithm = FQL(cfg or config(), FakeEnv(), None)
    algorithm.build()
    return algorithm


def test_twin_q_aggregation() -> None:
    q1 = torch.tensor([[1.0], [4.0]])
    q2 = torch.tensor([[3.0], [2.0]])
    assert torch.equal(aggregate_twin_q(q1, q2, "min"), torch.tensor([[1.0], [2.0]]))
    assert torch.equal(aggregate_twin_q(q1, q2, "mean"), torch.tensor([[2.0], [3.0]]))


def test_offline_dataset_rejects_actions_outside_fql_coordinates() -> None:
    payload = {
        "format": "mimic_offline_transitions_v1",
        "metadata": {},
        "observations": torch.zeros(2, 12),
        "actions": torch.tensor([[1.1, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "rewards": torch.zeros(2, 1),
        "next_observations": torch.zeros(2, 12),
        "masks": torch.ones(2, 1),
    }
    with pytest.raises(ValueError, match="above FQL action limit"):
        validate_offline_transition_payload(payload, obs_dim=12, action_dim=3, action_limit=1.0)


def test_euler_flow_target_uses_four_steps_and_is_detached() -> None:
    algorithm = build_algorithm()
    obs = torch.randn(7, algorithm.obs_dim)
    noises = torch.zeros(7, algorithm.action_dim)
    constant = torch.tensor([[0.25, -0.5, 0.125]])
    original_forward = algorithm.bc_flow.forward
    algorithm.bc_flow.forward = lambda obs, action, time: constant.expand(obs.shape[0], -1)
    try:
        actions = algorithm.compute_flow_actions(obs, noises)
    finally:
        algorithm.bc_flow.forward = original_forward
    assert torch.allclose(actions, constant.expand_as(actions), atol=1e-6)
    assert actions.requires_grad is False


def test_end_to_end_online_replay_update() -> None:
    torch.manual_seed(4)
    algorithm = build_algorithm()
    env = algorithm.env
    algorithm.initial_reset()
    rollout = algorithm.collect(algorithm.reset_for_update(1))
    assert env._step == 6
    assert algorithm.replay.size == 48
    assert rollout["warmup_actions"] == 48
    assert rollout["policy_actions"] == 0
    assert rollout["actions"].shape == (6, 8, 3)

    flow_before = [parameter.detach().clone() for parameter in algorithm.bc_flow.parameters()]
    actor_before = [parameter.detach().clone() for parameter in algorithm.one_step_actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in algorithm.critic.parameters()]
    metrics = algorithm.update(rollout, collect_time=0.01)

    def changed(before: list[torch.Tensor], module: torch.nn.Module) -> bool:
        return any(
            not torch.allclose(old, new)
            for old, new in zip(before, module.parameters(), strict=True)
        )

    assert changed(flow_before, algorithm.bc_flow)
    assert changed(actor_before, algorithm.one_step_actor)
    assert changed(critic_before, algorithm.critic)
    for key in (
        "fql/critic_loss",
        "fql/bc_flow_loss",
        "fql/distill_loss",
        "fql/q_loss",
        "fql/actor_loss",
        "fql/q_scale",
        "fql/target_actor_gap",
    ):
        assert math.isfinite(metrics[key]), key
    assert metrics["fql/gradient_steps"] == 4.0
    assert metrics["budget/physical_transitions"] == 48.0
    assert metrics["budget/replay_samples"] == 64.0
    assert metrics["budget/flow_training_evals"] == 64.0
    assert metrics["budget/flow_target_evals"] == 256.0
    assert algorithm.optimizer is algorithm.one_step_optimizer
    assert algorithm.one_step_optimizer is not algorithm.bc_flow_optimizer
    assert algorithm.critic_optimizer is not algorithm.one_step_optimizer

    deterministic_1 = algorithm.deterministic_actions(env._obs)
    deterministic_2 = algorithm.deterministic_actions(env._obs)
    assert deterministic_1.shape == (8, 1, 3)
    assert torch.allclose(deterministic_1, deterministic_2)


def test_offline_pretraining_seeds_replay_and_skips_random_warmup(tmp_path: Path) -> None:
    torch.manual_seed(8)
    count = 32
    dataset_path = save_offline_transition_dataset(
        tmp_path / "offline.pt",
        observations=torch.randn(count, 12),
        actions=0.5 * torch.tanh(torch.randn(count, 3)),
        rewards=torch.randn(count, 1),
        next_observations=torch.randn(count, 12),
        masks=torch.ones(count, 1),
        metadata={"teacher_algorithm": "unit-test", "action_limit": 1.0},
    )
    algorithm = build_algorithm(config(str(dataset_path), offline_pretrain_gradient_steps=2))
    assert algorithm.replay.size == count
    assert algorithm.replay.source_counts() == (count, 0)
    algorithm.initial_reset()
    rollout = algorithm.collect(algorithm.reset_for_update(1))
    assert algorithm.offline_pretrain_samples == 32
    assert rollout["warmup_actions"] == 0
    assert rollout["policy_actions"] == 48
    assert algorithm.replay.source_counts() == (32, 48)
    metrics = algorithm.update(rollout, collect_time=0.02)
    assert metrics["budget/offline_seed_transitions"] == 32.0
    assert metrics["budget/offline_pretrain_samples"] == 32.0
    assert metrics["fql/replay_offline_transitions"] == 32.0
    assert metrics["fql/replay_online_transitions"] == 48.0
    assert metrics["fql_pretrain/offline_sample_fraction"] == 1.0
    assert 0.0 < metrics["fql/update_offline_sample_fraction"] < 1.0


def test_offline_dataset_environment_scale_must_match_config(tmp_path: Path) -> None:
    count = 4
    dataset_path = save_offline_transition_dataset(
        tmp_path / "scale5.pt",
        observations=torch.zeros(count, 12),
        actions=torch.zeros(count, 3),
        rewards=torch.zeros(count, 1),
        next_observations=torch.zeros(count, 12),
        masks=torch.ones(count, 1),
        metadata={
            "action_coordinate": "normalized_fql_action",
            "environment_action_scale": 5.0,
        },
    )
    with pytest.raises(ValueError, match="does not match FQL config"):
        build_algorithm(config(str(dataset_path), environment_action_scale=1.0))


if __name__ == "__main__":
    tests = (
        test_twin_q_aggregation,
        test_offline_dataset_rejects_actions_outside_fql_coordinates,
        test_euler_flow_target_uses_four_steps_and_is_detached,
        test_end_to_end_online_replay_update,
    )
    for test in tests:
        print(f"[{test.__name__}]")
        test()
    print(f"All {len(tests)} FQL tests passed.")
