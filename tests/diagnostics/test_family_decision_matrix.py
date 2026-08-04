from __future__ import annotations

from pathlib import Path

from diagnostics.common.family_decision import derive_variables, select_frozen_family
from diagnostics.common.reporting import load_structured_file


ROOT = Path(__file__).resolve().parents[2]


def _artifact(diagnostic_id: str, key: str, value: str, **extra):
    return {
        "diagnostic_id": diagnostic_id,
        "status": "PASS",
        "summary": "test evidence",
        "evidence": {key: value, **extra},
    }


def test_frozen_decision_selects_first_matching_rule():
    spec = load_structured_file(
        ROOT / "diagnostics/specs/largebox_discovery_v1.yaml"
    )
    artifacts = {
        "27": _artifact("27", "decision_variable_R", "+"),
        "38": _artifact(
            "38", "decision_variable_T", "UNKNOWN",
            raw_or_teacher_positive_domain_gap="persists",
        ),
        "47": _artifact("47", "decision_variable_T", "+"),
        "58": _artifact("58", "decision_variable_E", "+"),
        "60": _artifact("60", "decision_variable_L", "+"),
        "61": _artifact("61", "decision_variable_F", "+"),
        "62": _artifact("62", "decision_variable_X", "+"),
    }
    variables, auxiliaries = derive_variables(artifacts)
    decision = select_frozen_family(spec["decision_matrix"], variables, auxiliaries)
    assert decision["matched_rule_id"] == "R+_T+_E+"
    assert decision["selected_family"] == "effective_overlap_guided_local_adversarial_curriculum"


def test_conditional_policy_evidence_never_fails_open():
    spec = load_structured_file(
        ROOT / "diagnostics/specs/largebox_discovery_v1.yaml"
    )
    artifacts = {
        "27": _artifact("27", "decision_variable_R", "CONDITIONAL"),
        "38": _artifact("38", "decision_variable_T", "UNKNOWN"),
        "47": _artifact("47", "decision_variable_T", "-"),
        "58": _artifact("58", "decision_variable_E", "-"),
        "60": _artifact("60", "decision_variable_L", "-"),
        "61": _artifact("61", "decision_variable_F", "-"),
        "62": _artifact("62", "decision_variable_X", "-"),
    }
    variables, auxiliaries = derive_variables(artifacts)
    assert variables["R"].value == "UNKNOWN"
    decision = select_frozen_family(spec["decision_matrix"], variables, auxiliaries)
    assert decision["status"] == "NO_ELIGIBLE_FAMILY"
    assert decision["selected_family"] is None
