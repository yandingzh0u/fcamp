from __future__ import annotations

import csv
import sys
from types import ModuleType
from types import SimpleNamespace

import torch

from engine.metrics_logger import MetricsLogger

_saved_spec = sys.modules.get("envs.spec")
_spec = ModuleType("envs.spec")
_spec.EE_Z_TERMINATION_THRESHOLD = 0.25
sys.modules["envs.spec"] = _spec
try:
    from engine.validation import (
        classify_amp_done_terms,
        standard_amp_action_target,
    )
finally:
    if _saved_spec is None:
        sys.modules.pop("envs.spec", None)
    else:
        sys.modules["envs.spec"] = _saved_spec

from engine.validation_logging import log_validation_metrics


def test_amp_done_classification_uses_only_physical_failure_and_timeout() -> None:
    done = torch.tensor([True, True, True, True, False])
    terms = {
        "time_out": torch.tensor([True, False, True, False, False]),
        "motion_complete": torch.tensor([True, True, True, True, True]),
        "tracking_failure": torch.tensor([True, True, True, True, True]),
        "anchor_pos_bad": torch.tensor([True, True, True, True, True]),
        "illegal_contact": torch.tensor([True, False, False, False, False]),
        "numerical_failure": torch.tensor([False, True, False, False, False]),
        "physical_failure": torch.tensor([True, True, False, False, False]),
    }

    timeout, motion_complete, failure = classify_amp_done_terms(done, terms)

    assert torch.equal(timeout, torch.tensor([False, False, True, False, False]))
    assert not bool(motion_complete.any())
    assert torch.equal(failure, torch.tensor([True, True, False, False, False]))


def test_amp_action_target_is_zero_centered_absolute_and_clipped() -> None:
    action = torch.tensor([[-2.0, -0.5, 0.25, 3.0]])
    scale = torch.tensor([[0.5, 2.0, 4.0, 0.25]])

    target = standard_amp_action_target(action, scale)

    assert torch.equal(target, torch.tensor([[-0.5, -1.0, 1.0, 0.25]]))


def _physical_validation_metrics() -> dict[str, float]:
    return {
        "validation/steps_mean": 12.0,
        "validation/failure_frac": 0.25,
        "validation/physical_failure_frac": 0.25,
        "validation/illegal_contact_frac": 0.20,
        "validation/numerical_failure_frac": 0.05,
        "validation/tracking_failure_counterfactual_frac": 0.75,
        "validation/anchor_pos_bad_counterfactual_frac": 0.50,
        "validation/anchor_ori_bad_counterfactual_frac": 0.25,
        "validation/ee_body_bad_counterfactual_frac": 0.60,
        "validation/reference_motion_end_reached_frac": 0.40,
        "validation/terminal_contact_force_max_mean": 42.0,
        "validation/terminal_illegal_contact_body_count_mean": 1.5,
        "validation/physical_failure_contact_force_max_mean": 84.0,
        "validation/physical_failure_illegal_contact_body_count_mean": 2.0,
    }


def test_validation_console_adds_amp_physical_and_counterfactual_fields(
    capsys,
) -> None:
    metrics = _physical_validation_metrics()
    env = SimpleNamespace(track_body_names=[])

    log_validation_metrics(env, metrics)

    lines = capsys.readouterr().out.splitlines()
    cause = next(line for line in lines if line.startswith("[VAL_CAUSE]"))
    outcome = next(line for line in lines if line.startswith("[VAL_OUTCOME]"))
    fail = next(line for line in lines if line.startswith("[VAL_FAIL]"))
    assert "physical_failure=0.25000" in cause
    assert "illegal_contact=0.20000" in cause
    assert "numerical_failure=0.05000" in cause
    assert "tracking_cf=0.75000" in cause
    assert "physical_failure=0.2500" in outcome
    assert "reference_end=0.4000" in outcome
    assert "contact_force_max=84.00000" in fail
    assert "illegal_contact_bodies=2.000" in fail


def test_validation_csv_adds_amp_physical_and_counterfactual_fields(
    tmp_path,
) -> None:
    metrics = _physical_validation_metrics()
    logger = MetricsLogger(tmp_path)
    try:
        logger.write_validation_summary(7, metrics)
    finally:
        logger.close()

    with (tmp_path / "validation_summary.csv").open(
        encoding="utf-8",
        newline="",
    ) as handle:
        row = next(csv.DictReader(handle))

    for key, expected in metrics.items():
        if key == "validation/steps_mean":
            continue
        assert key in row
        assert float(row[key]) == expected
