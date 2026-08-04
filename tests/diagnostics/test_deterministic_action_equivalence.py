from __future__ import annotations

from types import SimpleNamespace

import torch

from diagnostics.common.rollout_collector import deterministic_action_equivalence


class _Normalizer:
    def __call__(self, value: torch.Tensor, *, update: bool) -> torch.Tensor:
        assert update is False
        return 2.0 * value


class _Actor:
    std = torch.ones(2)

    @staticmethod
    def act_inference(value: torch.Tensor) -> torch.Tensor:
        return value[:, :2] + 3.0


class _Algorithm:
    actor_obs_normalizer = _Normalizer()
    actor = _Actor()

    def deterministic_action(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor.act_inference(self.actor_obs_normalizer(observation, update=False))


def test_deterministic_mean_matches_algorithm_adapter_exactly() -> None:
    observation = torch.tensor([[1.0, 2.0, 4.0]])
    result = deterministic_action_equivalence(_Algorithm(), observation)
    assert result == {"equivalent": True, "max_abs_error": 0.0, "atol": 0.0}
