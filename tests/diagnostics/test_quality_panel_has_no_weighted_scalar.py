from __future__ import annotations

from diagnostics.common.quality_panel import pareto_preference, strict_pareto_pairs


def _row(**overrides: float) -> dict[str, float]:
    row = {
        "motion_complete": 1.0,
        "failure": 0.0,
        "reference_progress": 0.8,
        "survival": 100.0,
        "joint_limit_incidence": 0.0,
        "undesired_contacts": 0.0,
    }
    row.update(overrides)
    return row


def test_quality_panel_keeps_tradeoffs_unordered() -> None:
    progress = _row(reference_progress=0.9, failure=0.1)
    safety = _row(reference_progress=0.8, failure=0.0)
    assert pareto_preference(progress, safety) == 0
    assert strict_pareto_pairs([progress, safety]) == []


def test_quality_panel_emits_only_strict_pareto_pairs() -> None:
    winner = _row(reference_progress=0.95, survival=120.0)
    loser = _row(reference_progress=0.80, survival=100.0)
    assert pareto_preference(winner, loser) == 1
    assert strict_pareto_pairs([loser, winner]) == [(1, 0)]
