from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


_STUB_NAMES = (
    "engine.validation",
    "engine.validation_logging",
    "envs.g1_mimic",
    "method.fixed_reward",
)
_SAVED_MODULES = {name: sys.modules.get(name) for name in _STUB_NAMES}

_validation = ModuleType("engine.validation")
_validation.run_validation_rollout = lambda *args, **kwargs: {}
_validation.validation_max_steps = lambda *args, **kwargs: 1
sys.modules["engine.validation"] = _validation

_validation_logging = ModuleType("engine.validation_logging")
_validation_logging.log_validation_metrics = lambda *args, **kwargs: None
sys.modules["engine.validation_logging"] = _validation_logging

_g1_mimic = ModuleType("envs.g1_mimic")
_g1_mimic.G1MimicEnv = object
sys.modules["envs.g1_mimic"] = _g1_mimic

_fixed_reward = ModuleType("method.fixed_reward")
_fixed_reward.FixedRewardPPO = object
sys.modules["method.fixed_reward"] = _fixed_reward

try:
    from engine.trainer import CoreTrainer
finally:
    for _name, _module in _SAVED_MODULES.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module


class _SimulationApp:
    def is_running(self) -> bool:
        return True


class _Environment:
    def begin_reset_phase_diagnostics(self) -> None:
        pass

    def finish_reset_phase_diagnostics(self) -> dict[str, float]:
        return {}


class _Algorithm:
    def log_banner(self) -> None:
        pass

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        assert update_idx == 1
        return torch.tensor([[2.0]])

    def collect(self, observation: torch.Tensor) -> dict:
        return {"next_observation": observation + 1.0}

    def update(self, rollout: dict, collect_time: float) -> dict[str, float]:
        del rollout, collect_time
        return {"system/parameters_finite": 1.0}


class _MetricsLogger:
    def __init__(self) -> None:
        self.metrics: dict[str, float] | None = None

    def write(self, update_idx: int, metrics: dict[str, float]) -> None:
        assert update_idx == 1
        self.metrics = dict(metrics)

    def close(self) -> None:
        pass


class _Checkpointer:
    def __init__(self) -> None:
        self.saved_updates: list[int] = []

    def save(
        self,
        update_idx: int,
        metrics: dict[str, float],
        filename=None,
    ) -> None:
        del metrics, filename
        self.saved_updates.append(update_idx)

    def target_reached(self, metrics: dict[str, float]) -> bool:
        del metrics
        return False


class _FailingValidationCheckpointer(_Checkpointer):
    pass


def test_each_update_accounts_only_its_formal_rollout() -> None:
    trainer = CoreTrainer.__new__(CoreTrainer)
    trainer.simulation_app = _SimulationApp()
    trainer.train_cfg = SimpleNamespace(
        resume="",
        max_updates=1,
        log_every=2,
        validation_every=0,
        save_every=0,
        target_validation_steps=0,
    )
    trainer.env_cfg = SimpleNamespace(num_envs=2, sim_dt=0.02)
    trainer.algo_cfg = SimpleNamespace(num_steps_per_env=4)
    trainer.env = _Environment()
    trainer.algo = _Algorithm()
    trainer.current_observation = torch.tensor([[1.0]])
    trainer.start_update = 1
    trainer.env_transitions_total = 0
    trainer.train_wall_seconds_total = 0.0
    trainer.checkpoint_dir = SimpleNamespace()
    trainer.metrics_logger = _MetricsLogger()
    trainer.checkpointer = _Checkpointer()

    trainer.train()

    metrics = trainer.metrics_logger.metrics
    assert metrics is not None
    assert trainer.checkpointer.saved_updates == [1]
    assert metrics["samples/formal_env_transitions_update"] == 8.0
    assert metrics["samples/env_transitions_update"] == 8.0
    assert metrics["samples/env_transitions_total"] == 8.0
    assert trainer.env_transitions_total == 8
    assert not any(key.startswith("warmup/") for key in metrics)


def test_periodic_checkpoint_is_saved_before_validation(monkeypatch) -> None:
    trainer = CoreTrainer.__new__(CoreTrainer)
    trainer.simulation_app = _SimulationApp()
    trainer.train_cfg = SimpleNamespace(
        resume="",
        max_updates=1,
        log_every=2,
        validation_every=1,
        save_every=1,
        validation_fixed_seed=0,
        validation_directional_start_phase=-1,
        target_validation_steps=0,
    )
    trainer.env_cfg = SimpleNamespace(num_envs=2, sim_dt=0.02)
    trainer.algo_cfg = SimpleNamespace(num_steps_per_env=4)
    trainer.env = _Environment()
    trainer.algo = _Algorithm()
    trainer.current_observation = torch.tensor([[1.0]])
    trainer.start_update = 1
    trainer.env_transitions_total = 0
    trainer.train_wall_seconds_total = 0.0
    trainer.checkpoint_dir = SimpleNamespace()
    trainer.metrics_logger = _MetricsLogger()
    trainer.checkpointer = _FailingValidationCheckpointer()

    monkeypatch.setattr("engine.trainer.validation_max_steps", lambda *_: 1)

    def fail_validation(*_args, **_kwargs):
        assert trainer.checkpointer.saved_updates == [1]
        raise RuntimeError("validation sentinel")

    monkeypatch.setattr("engine.trainer.run_validation_rollout", fail_validation)
    with pytest.raises(RuntimeError, match="validation sentinel"):
        trainer.train()
