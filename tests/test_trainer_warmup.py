from __future__ import annotations

import sys
from types import ModuleType
from types import SimpleNamespace

import torch


_STUB_NAMES = ("engine.validation", "engine.validation_logging", "envs.g1_mimic")
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


class _WarmupAlgorithm:
    def __init__(self) -> None:
        self.warmup_calls = 0

    def log_banner(self) -> None:
        pass

    def pre_training_warmup(self, observation: torch.Tensor):
        self.warmup_calls += 1
        return observation + 1.0, {"actor_optimizer_steps": 0.0}, 3

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        assert update_idx == 1
        return torch.tensor([[2.0]])

    def collect(self, observation: torch.Tensor) -> dict:
        return {"next_observation": observation + 1.0}

    def update(self, rollout: dict, collect_time: float) -> dict[str, float]:
        del rollout, collect_time
        return {}


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

    def save(self, update_idx: int, metrics: dict[str, float], filename=None) -> None:
        del metrics, filename
        self.saved_updates.append(update_idx)

    def target_reached(self, metrics: dict[str, float]) -> bool:
        del metrics
        return False


def _trainer(*, resume: str = "") -> CoreTrainer:
    trainer = CoreTrainer.__new__(CoreTrainer)
    trainer.simulation_app = _SimulationApp()
    trainer.train_cfg = SimpleNamespace(
        resume=resume,
        max_updates=1,
        log_every=2,
        validation_every=0,
        save_every=0,
        target_validation_steps=0,
    )
    trainer.env_cfg = SimpleNamespace(num_envs=2, sim_dt=0.005)
    trainer.algo_cfg = SimpleNamespace(rollout_env_steps=4)
    trainer.env = _Environment()
    trainer.algo = _WarmupAlgorithm()
    trainer.current_observation = torch.tensor([[1.0]])
    trainer.start_update = 1
    trainer.env_transitions_total = 0
    trainer.train_wall_seconds_total = 0.0
    trainer._pre_training_warmup_ran = False
    trainer.checkpoint_dir = SimpleNamespace(__str__=lambda self: "unused")
    trainer.metrics_logger = _MetricsLogger()
    trainer.checkpointer = _Checkpointer()
    return trainer


def test_warmup_is_accounted_in_first_formal_update_without_advancing_index() -> None:
    trainer = _trainer()

    trainer.train()

    metrics = trainer.metrics_logger.metrics
    assert metrics is not None
    assert trainer.algo.warmup_calls == 1
    assert trainer.checkpointer.saved_updates == [1]
    assert metrics["warmup/actor_optimizer_steps"] == 0.0
    assert metrics["samples/warmup_env_transitions"] == 3.0
    assert metrics["samples/formal_env_transitions_update"] == 8.0
    assert metrics["samples/env_transitions_update"] == 11.0
    assert metrics["samples/env_transitions_total"] == 11.0
    assert trainer.env_transitions_total == 11
    assert metrics["perf/iteration_s"] >= metrics["timing/warmup_s"]
    assert trainer._run_pre_training_warmup() == ({}, 0, 0.0)
    assert trainer.algo.warmup_calls == 1


def test_warmup_is_skipped_for_resume() -> None:
    trainer = _trainer(resume="checkpoint.pt")
    trainer.start_update = 7

    metrics, transitions, elapsed = trainer._run_pre_training_warmup()

    assert metrics == {}
    assert transitions == 0
    assert elapsed == 0.0
    assert trainer.algo.warmup_calls == 0


def test_warmup_rejects_negative_transition_count() -> None:
    trainer = _trainer()
    trainer.algo.pre_training_warmup = lambda observation: (observation, {}, -1)

    try:
        trainer._run_pre_training_warmup()
    except ValueError as exc:
        assert "must be >= 0" in str(exc)
    else:
        raise AssertionError("negative warm-up transition count was accepted")
