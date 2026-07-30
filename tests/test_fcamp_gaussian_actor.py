from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from components.rollout.fcamp_diagnostics import (
    gaussian_action_noise_statistics,
    gaussian_action_path_noise_statistics,
)
from method.fcamp import FCAMP
from components.rollout.training_streams import PHASE0_STREAM


class _OffsetPolicy(nn.Module):
    def __init__(self, horizon: int, action_dim: int) -> None:
        super().__init__()
        self.mean = nn.Parameter(torch.zeros(horizon, action_dim))
        self.action_path_log_std = nn.Parameter(
            torch.full((horizon, action_dim), math.log(0.5))
        )


def _gaussian_actor(horizon: int = 4, action_dim: int = 2) -> FCAMP:
    algo = object.__new__(FCAMP)
    algo.env = SimpleNamespace(device=torch.device("cpu"))
    algo.horizon_h = horizon
    algo.num_act = action_dim
    algo.chunk_dim = horizon * action_dim
    algo.actor_obs_dim = 1
    algo.action_path_std_min = 0.02
    algo.action_path_std_max = 1.5
    algo.cfg = SimpleNamespace(
        clip_range=0.2,
        max_grad_norm=10.0,
        policy_epochs=1,
        desired_kl=0.0,
        kl_early_stop_factor=100.0,
        num_mini_batches=1,
        micro_batch_size=0,
        streams=SimpleNamespace(phase0_fraction=0.1),
    )
    algo._policy = _OffsetPolicy(horizon, action_dim)
    algo._flow_mean_latent = lambda obs: (
        algo._policy.mean.unsqueeze(0)
        .expand(obs.shape[0], -1, -1)
        .reshape(obs.shape[0], -1)
    )
    algo.learning_rate = 1.0e-2
    algo.min_lr = 1.0e-5
    algo.max_lr = 1.0e-2
    algo.actor_optimizer = torch.optim.SGD(
        algo._policy.parameters(),
        lr=algo.learning_rate,
    )
    return algo


def test_unchanged_gaussian_ratio_is_one_and_future_offsets_are_masked() -> None:
    algo = _gaussian_actor()
    sample = torch.tensor(
        [[[0.20, -0.10], [0.10, 0.30], [0.40, -0.20], [-0.30, 0.25]]]
    )
    old_mean = torch.zeros_like(sample)
    old_log_std = algo._bounded_action_path_log_std().detach().clone()
    old_log_prob = algo._diagonal_gaussian_log_prob(
        sample,
        old_mean,
        old_log_std,
    )
    valid = torch.tensor([[[True, True, False, False]]])
    rollout = {
        "actor_obs": torch.zeros(1, 1, 1),
        "sampled_action_paths": sample.reshape(1, 1, 4, 2),
        "old_action_path_means": old_mean.reshape(1, 1, 4, 2),
        "old_log_std": old_log_std,
        "old_log_probs": old_log_prob.reshape(1, 1, 4),
        "advantages": torch.ones(1, 1, 4),
        "valid": valid,
        "credit_valid": valid,
        "stream_ids": torch.tensor([PHASE0_STREAM], dtype=torch.int8),
    }

    metrics = algo._actor_update(rollout)

    assert metrics["fcamp/ratio"] == pytest.approx(1.0)
    assert metrics["fcamp/kl"] == pytest.approx(0.0, abs=1.0e-8)
    assert metrics["fcamp/clip_fraction"] == pytest.approx(0.0)
    assert algo._policy.mean.grad is not None
    assert algo._policy.action_path_log_std.grad is not None
    assert float(algo._policy.mean.grad[:2].abs().sum().item()) > 0.0
    assert float(
        algo._policy.action_path_log_std.grad[:2].abs().sum().item()
    ) > 0.0
    torch.testing.assert_close(
        algo._policy.mean.grad[2:],
        torch.zeros_like(algo._policy.mean.grad[2:]),
    )
    torch.testing.assert_close(
        algo._policy.action_path_log_std.grad[2:],
        torch.zeros_like(algo._policy.action_path_log_std.grad[2:]),
    )


def test_gaussian_action_noise_statistics_use_only_executed_actions() -> None:
    delta = torch.tensor(
        [
            [
                [[1.0, -1.0], [2.0, -2.0]],
                [[3.0, -3.0], [100.0, -100.0]],
            ]
        ]
    )
    executed = torch.tensor([[[True, True], [True, False]]])

    metrics = gaussian_action_noise_statistics(delta, executed)

    assert metrics["gaussian/action_noise/component_count"] == 6.0
    assert metrics["gaussian/action_noise/signed_mean"] == pytest.approx(0.0)
    assert metrics["gaussian/action_noise/rms"] == pytest.approx(
        math.sqrt(28.0 / 6.0)
    )
    assert (
        metrics["gaussian/action_noise_h1/component_count"]
        == 2.0
    )


def test_gaussian_action_path_noise_statistics_report_per_h_geometry() -> None:
    delta = torch.tensor(
        [
            [
                [[1.0, -1.0], [1.0, -1.0]],
                [[2.0, -2.0], [200.0, -200.0]],
            ]
        ]
    )
    executed = torch.tensor([[[True, True], [True, False]]])

    metrics = gaussian_action_path_noise_statistics(delta, executed)

    assert metrics["gaussian/action_path_noise/component_count"] == 6.0
    assert metrics["gaussian/action_path_noise_h0/rms"] == pytest.approx(
        math.sqrt(10.0 / 4.0)
    )
    assert metrics["gaussian/action_path_noise_h1/rms"] == pytest.approx(1.0)
    assert metrics[
        "gaussian/action_path_noise/last_to_first_rms_ratio"
    ] == pytest.approx(1.0 / math.sqrt(10.0 / 4.0))
