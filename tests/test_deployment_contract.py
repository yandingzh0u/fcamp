from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from engine.checkpoint import (
    FCAMP_CHECKPOINT_CONTRACT,
    preflight_checkpoint_payload,
    preflight_static_checkpoint_payload,
)
from method.base import Algorithm


class _Algorithm(Algorithm):
    def __init__(self, env):
        super().__init__(cfg=SimpleNamespace(), env=env, simulation_app=None)
        self._policy = torch.nn.Linear(1, 1)
        self._optimizer = torch.optim.SGD(self._policy.parameters(), lr=0.1)
        self.last_step = None
        self.preflight_payload = None

    def build(self) -> None:
        pass

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        del update_idx
        return torch.empty(0)

    def initial_reset(self) -> torch.Tensor:
        return torch.empty(0)

    def collect(self, obs: torch.Tensor) -> dict:
        del obs
        return {}

    def update(self, rollout: dict, collect_time: float) -> dict:
        del rollout, collect_time
        return {}

    def log(self, update_idx: int, max_updates: int, metrics: dict) -> None:
        del update_idx, max_updates, metrics

    def log_banner(self) -> None:
        pass

    @property
    def policy(self) -> torch.nn.Module:
        return self._policy

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def horizon(self) -> int:
        return 4

    def extra_checkpoint_state(self) -> dict:
        return {}

    def load_extra_checkpoint_state(self, payload: dict, reset_optimizer: bool = False) -> None:
        del payload, reset_optimizer

    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        return obs + 1.0

    def evaluation_step(self, actions, reference_dt):
        self.last_step = (actions.clone(), reference_dt.clone())
        info = {"applied_action": actions.clone()}
        count = actions.shape[0]
        return (
            torch.zeros_like(actions),
            torch.zeros(count),
            torch.zeros(count, dtype=torch.bool),
            info,
        )

    def validate_checkpoint_payload(self, payload: dict) -> None:
        self.preflight_payload = payload


def _env():
    return SimpleNamespace(
        num_envs=2,
        dt=0.02,
        last_action=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )


def test_deployment_chunk_defaults_to_deterministic_policy_payload() -> None:
    algo = _Algorithm(_env())
    obs = torch.tensor([[0.0, 1.0], [2.0, 3.0]])

    torch.testing.assert_close(algo.deployment_chunk(obs), obs + 1.0)


def test_validated_deployment_chunk_enforces_full_horizon() -> None:
    algo = _Algorithm(_env())

    with pytest.raises(ValueError, match="batch/time shape"):
        algo.validated_deployment_chunk(torch.zeros(2, 2))

    expected = torch.randn(2, 4, 2)
    algo.deployment_chunk = lambda obs: expected
    torch.testing.assert_close(
        algo.validated_deployment_chunk(torch.zeros(2, 2)),
        expected,
    )


def test_default_deployment_step_holds_inactive_env_without_zero_jump() -> None:
    algo = _Algorithm(_env())
    payload = torch.tensor([[9.0, 8.0], [-7.0, -6.0]])
    reference_dt = torch.tensor([0.03, 0.04])

    _, _, _, info = algo.evaluation_step_payload(
        payload,
        reference_dt,
        active_mask=torch.tensor([True, False]),
    )

    expected_action = torch.tensor([[9.0, 8.0], [3.0, 4.0]])
    expected_dt = torch.tensor([0.03, 0.02])
    torch.testing.assert_close(algo.last_step[0], expected_action)
    torch.testing.assert_close(algo.last_step[1], expected_dt)
    torch.testing.assert_close(info["applied_action"], expected_action)


def test_applied_action_diagnostic_rejects_raw_policy_payload() -> None:
    algo = _Algorithm(_env())

    with pytest.raises(RuntimeError, match="applied_action"):
        algo.require_applied_action({"raw_z": torch.zeros(2, 2)})


def test_default_deployment_step_rejects_non_boolean_active_mask() -> None:
    algo = _Algorithm(_env())

    with pytest.raises(ValueError, match="active_mask must be bool"):
        algo.evaluation_step_payload(
            torch.zeros(2, 2),
            None,
            active_mask=torch.ones(2),
        )


def test_checkpoint_preflight_runs_method_contract_before_loading() -> None:
    algo = _Algorithm(_env())
    payload = {"policy": algo.policy.state_dict(), "algo_state": {"schema": 99}}

    preflight_checkpoint_payload(algo, payload)

    assert algo.preflight_payload is payload


def test_static_fcamp_preflight_rejects_old_schema_without_runtime() -> None:
    state = {
        **FCAMP_CHECKPOINT_CONTRACT,
        "discriminator_policy_conditioning": False,
    }
    payload = {
        "config": {"method": "fcamp"},
        "policy": {},
        "algo_state": state,
    }
    preflight_static_checkpoint_payload(payload, expected_method="fcamp")

    old_payload = {
        **payload,
        "algo_state": {
            **state,
            "fcamp_schema_version": FCAMP_CHECKPOINT_CONTRACT[
                "fcamp_schema_version"
            ]
            - 1,
        },
    }
    with pytest.raises(ValueError, match="fcamp_schema_version"):
        preflight_static_checkpoint_payload(
            old_payload,
            expected_method="fcamp",
        )


@pytest.mark.parametrize("payload", [None, [], {"policy": None}, {}])
def test_checkpoint_preflight_rejects_malformed_payload(payload) -> None:
    algo = _Algorithm(_env())

    with pytest.raises((TypeError, KeyError)):
        preflight_checkpoint_payload(algo, payload)
