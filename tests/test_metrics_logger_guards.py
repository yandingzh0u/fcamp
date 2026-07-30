from __future__ import annotations

import pytest

from engine.metrics_logger import MetricsLogger
from engine.validation_logging import log_validation_metrics


def test_nonfinite_metric_aborts_before_it_is_persisted(tmp_path) -> None:
    logger = MetricsLogger(tmp_path)
    try:
        with pytest.raises(FloatingPointError, match="loss/value"):
            logger.write(
                3,
                {
                    "samples/env_transitions_total": 24.0,
                    "loss/value": float("nan"),
                },
            )
    finally:
        logger.close()

    assert (tmp_path / "metrics.jsonl").read_text(encoding="utf-8") == ""


def test_validation_console_uses_finite_sentinels_for_absent_outcomes(
    capsys,
) -> None:
    log_validation_metrics(
        object(),
        {
            "validation/steps_mean": 324.0,
            "validation/return_mean": 20.0,
        },
    )

    output = capsys.readouterr().out.lower()
    assert "nan" not in output
    assert "-1.00000" in output
