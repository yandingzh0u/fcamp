from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from diagnostics.common.checkpoint_io import compare_checkpoint_payloads


def _result(*, weight: float = 1.0, elapsed: float = 3.0) -> dict:
    return {
        "update_idx": 201,
        "policy": {"actor.weight": torch.tensor([weight])},
        "optimizer": {"state": {0: {"step": torch.tensor(1)}}},
        "algo_state": {"counter": 1},
        "adaptive_sampler_state": {"counts": torch.tensor([1, 2])},
        "torch_rng_state": torch.arange(8, dtype=torch.uint8),
        "cuda_rng_state": torch.arange(8, dtype=torch.uint8),
        "env_transitions_total": 123,
        "train_wall_seconds_total": elapsed,
        "metrics": {
            "Loss/Value": 0.25,
            "Policy/mean_noise_std": 0.5,
            "perf/iteration_s": elapsed,
            "timing/collect_s": elapsed / 2.0,
            "perf/train_wall_s_total": elapsed,
        },
    }


def test_same_rng_resume_ignores_only_wall_clock_telemetry() -> None:
    comparison = compare_checkpoint_payloads(
        _result(elapsed=3.0),
        _result(elapsed=9.0),
    )

    assert comparison["all_state_exact"] is True
    assert comparison["deterministic_metrics"]["all_close"] is True
    assert comparison["nondeterministic_telemetry"]["all_close"] is False
    assert comparison["reproducible"] is True


def test_same_rng_resume_fails_on_policy_or_scientific_metric_drift() -> None:
    policy_drift = compare_checkpoint_payloads(_result(), _result(weight=2.0))
    assert policy_drift["reproducible"] is False
    assert policy_drift["exact_sections"]["policy"] is False

    metric_drift = _result()
    metric_drift["metrics"]["Loss/Value"] = 0.30
    comparison = compare_checkpoint_payloads(_result(), metric_drift)
    assert comparison["reproducible"] is False
    assert comparison["deterministic_metrics"]["all_close"] is False
