from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from components.rollout.amp_diagnostics import (
    AMPDiagnosticsMixin,
    _ensure_horizon_metrics,
    amp_domain_parity_statistics,
    gaussian_action_noise_statistics,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_validation_logging():
    """Load validation logging without importing the Isaac Sim environment."""

    stub = ModuleType("engine.validation")
    stub.short_body_name = lambda name: str(name).split("_link")[0]
    previous = sys.modules.get("engine.validation")
    sys.modules["engine.validation"] = stub
    try:
        spec = importlib.util.spec_from_file_location(
            "engine._validation_logging_horizon_test",
            ROOT / "engine" / "validation_logging.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if previous is None:
            sys.modules.pop("engine.validation", None)
        else:
            sys.modules["engine.validation"] = previous


def _assert_finite(metrics: dict[str, float], keys: list[str]) -> None:
    for key in keys:
        assert key in metrics
        assert math.isfinite(float(metrics[key])), key


def test_h1_noise_statistics_keep_finite_h0_through_h3_schema() -> None:
    delta = torch.tensor([[[[2.0, -2.0]]]])
    executed = torch.ones(1, 1, 1, dtype=torch.bool)

    metrics = gaussian_action_noise_statistics(delta, executed)

    assert metrics["gaussian/action_noise_h0/rms"] == pytest.approx(2.0)
    for offset in range(1, 4):
        prefix = f"gaussian/action_noise_h{offset}"
        _assert_finite(
            metrics,
            [
                f"{prefix}/component_count",
                f"{prefix}/signed_mean",
                f"{prefix}/rms",
                f"{prefix}/abs_mean",
                f"{prefix}/abs_p95",
                f"{prefix}/abs_max",
            ],
        )
        assert metrics[f"{prefix}/component_count"] == 0.0
        assert metrics[f"{prefix}/rms"] == -1.0
    assert "gaussian/action_noise_h4/rms" not in metrics


def test_h5_noise_statistics_emit_real_fifth_offset() -> None:
    delta = torch.arange(1.0, 6.0).view(1, 1, 5, 1)
    executed = torch.ones(1, 1, 5, dtype=torch.bool)

    metrics = gaussian_action_noise_statistics(delta, executed)

    for offset in range(5):
        assert metrics[f"gaussian/action_noise_h{offset}/rms"] == pytest.approx(
            float(offset + 1)
        )
    assert metrics[
        "gaussian/action_noise/last_to_first_rms_ratio"
    ] == pytest.approx(5.0)
    assert "gaussian/action_noise_h5/rms" not in metrics


def test_amp_domain_parity_statistics_is_exact_and_field_resolved() -> None:
    policy = torch.zeros(3, 239)
    expert = policy.clone()
    expert[:, 207] = 2.0

    metrics = amp_domain_parity_statistics(policy, expert)

    assert metrics["root_ang_vel_max"] == 2.0
    assert metrics["root_ang_vel_rms"] > 0.0
    assert metrics["root_lin_vel_max"] == 0.0
    assert metrics["joint_vel_max"] == 0.0
    assert metrics["full_max"] == 2.0


@pytest.mark.parametrize("horizon, expected_slots", [(1, 4), (4, 4), (5, 5)])
def test_horizon_metrics_are_dynamic_and_always_report_h0_through_h3(
    horizon: int,
    expected_slots: int,
) -> None:
    metrics: dict[str, float] = {
        f"amp_policy/offset_{offset}_kl": float(offset + 1)
        for offset in range(horizon)
    }

    _ensure_horizon_metrics(metrics, horizon)

    for offset in range(expected_slots):
        _assert_finite(
            metrics,
            [
                f"amp_policy/offset_{offset}_kl",
                f"amp_policy/offset_{offset}_clip",
                f"amp_policy/offset_{offset}_ratio",
                f"amp_policy/offset_{offset}_fixed_std",
                f"amp_policy/offset_{offset}_executed_count",
                f"gaussian/action_noise_h{offset}/rms",
            ],
        )
    assert f"amp_policy/offset_{expected_slots}_kl" not in metrics


def _amp_console_metrics(horizon: int) -> dict[str, float]:
    metrics = {
        "gaussian/action_noise/rms": 0.5,
        "gaussian/action_noise/last_to_first_rms_ratio": 1.0,
        "amp_policy/fixed_normalized_std": 0.05,
        "amp_policy/entropy": 1.0,
        "timing/collect_s": 1.0,
        "timing/actor_update_s": 1.0,
        "timing/critic_update_s": 1.0,
        "timing/disc_update_s": 1.0,
        "system/cuda_peak_allocated_gib": 1.0,
    }
    for offset in range(horizon):
        value = float(offset + 1)
        metrics.update(
            {
                f"amp_policy/offset_{offset}_fixed_std": value,
                f"gaussian/action_noise_h{offset}/rms": value,
                f"amp_policy/offset_{offset}_kl": value,
                f"amp_policy/offset_{offset}_clip": value,
                f"amp_policy/offset_{offset}_ratio": value,
            }
        )
    return metrics


@pytest.mark.parametrize("horizon, expected_slots", [(1, 4), (4, 4), (5, 5)])
def test_amp_console_prints_dynamic_finite_slots(
    horizon: int,
    expected_slots: int,
    capsys,
) -> None:
    diagnostics = object.__new__(AMPDiagnosticsMixin)
    diagnostics.horizon_h = horizon
    metrics = _amp_console_metrics(horizon)

    diagnostics.log(1, 1, metrics)

    output = capsys.readouterr().out.splitlines()
    exploration = next(
        line for line in output if line.startswith("[AMP_EXPLORATION]")
    )
    ppo = next(line for line in output if line.startswith("[AMP_PPO_H]"))
    for line in (exploration, ppo):
        assert "nan" not in line.lower()
        assert "inf" not in line.lower()
    for field in ("std_h", "noise_h"):
        values = exploration.split(f"{field}=", 1)[1].split(" ", 1)[0].split("/")
        assert len(values) == expected_slots
    for field in ("kl", "clip", "ratio"):
        values = ppo.split(f"{field}=", 1)[1].split(" ", 1)[0].split("/")
        assert len(values) == expected_slots
    if horizon == 1:
        assert "std_h=1.00000/-1.00000/-1.00000/-1.00000" in exploration
        assert "kl=1.000000/-1.000000/-1.000000/-1.000000" in ppo


def _validation_offset_metrics(horizon: int) -> dict[str, float]:
    metrics = {"validation/steps_mean": 1.0}
    for offset in range(horizon):
        for name in (
            "joint_pos_error",
            "joint_vel_error",
            "root_ang_vel_error",
        ):
            metrics.update(
                {
                    f"validation/chunk_offset{offset}_{name}_count": 1.0,
                    f"validation/chunk_offset{offset}_{name}_mean": float(
                        offset + 1
                    ),
                    f"validation/chunk_offset{offset}_{name}_p95": float(
                        offset + 1
                    ),
                    f"validation/chunk_offset{offset}_{name}_p99": float(
                        offset + 1
                    ),
                }
            )
    return metrics


@pytest.mark.parametrize("horizon, expected_slots", [(1, 4), (4, 4), (5, 5)])
def test_validation_console_preserves_four_slots_and_extends_past_h4(
    horizon: int,
    expected_slots: int,
    capsys,
) -> None:
    validation_logging = _load_validation_logging()
    metrics = _validation_offset_metrics(horizon)
    env = SimpleNamespace(track_body_names=[])

    validation_logging.log_validation_metrics(env, metrics)

    offset_lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("[VAL_OFFSET")
    ]
    assert len(offset_lines) == expected_slots
    assert offset_lines[-1].startswith(f"[VAL_OFFSET{expected_slots - 1}]")
    for line in offset_lines:
        assert "joint_pos=" in line
        assert "joint_pos_p95=" in line
        assert "joint_vel=" in line
        assert "joint_vel_p95=" in line
        assert "root_ang_vel=" in line
        assert "root_ang_vel_p95=" in line
        assert "count=" in line
        assert "nan" not in line.lower()
        assert "inf" not in line.lower()
    for offset in range(expected_slots):
        _assert_finite(
            metrics,
            [
                f"validation/chunk_offset{offset}_joint_pos_error_count",
                f"validation/chunk_offset{offset}_joint_pos_error_mean",
            ],
        )
    if horizon == 1:
        assert (
            metrics["validation/chunk_offset3_joint_pos_error_count"]
            == 0.0
        )
        assert (
            metrics["validation/chunk_offset3_joint_pos_error_mean"]
            == -1.0
        )
