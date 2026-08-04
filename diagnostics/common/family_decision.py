"""Fail-closed derivation of the frozen R/T/E/L/F/X decision matrix."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .manifest import ProtocolError


TRISTATE = {"+", "-", "UNKNOWN"}


@dataclass(frozen=True, slots=True)
class DecisionVariable:
    value: str
    evidence_diagnostics: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if self.value not in TRISTATE:
            raise ProtocolError(f"invalid decision value {self.value!r}")
        if not self.reason:
            raise ProtocolError("decision-variable reason may not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "evidence_diagnostics": list(self.evidence_diagnostics),
            "reason": self.reason,
        }


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ProtocolError(f"artifact lacks mapping {key!r}")
    return value


def _explicit(
    payload: Mapping[str, Any],
    *,
    evidence_key: str,
    diagnostic_id: str,
) -> DecisionVariable:
    status = str(payload.get("status", ""))
    if status not in {"PASS", "FAIL"}:
        return DecisionVariable(
            "UNKNOWN", (diagnostic_id,), f"diag_{diagnostic_id} status is {status or 'missing'}",
        )
    evidence = _mapping(payload, "evidence")
    raw = str(evidence.get(evidence_key, "UNKNOWN"))
    aliases = {
        "+": "+",
        "-": "-",
        "GREEN": "+",
        "RED": "-",
        "PASS": "+",
        "FAIL": "-",
        "TRUE": "+",
        "FALSE": "-",
        "CONDITIONAL": "UNKNOWN",
        "UNKNOWN": "UNKNOWN",
    }
    value = aliases.get(raw.upper(), aliases.get(raw, "UNKNOWN"))
    reason = str(
        evidence.get("explanation")
        or evidence.get("decision_reason")
        or payload.get("summary")
        or f"diag_{diagnostic_id} supplied {evidence_key}={raw}"
    )
    return DecisionVariable(value, (diagnostic_id,), reason)


def derive_variables(artifacts: Mapping[str, Mapping[str, Any]]) -> tuple[
    dict[str, DecisionVariable], dict[str, Any]
]:
    """Derive only explicitly evidenced variables; never infer PASS from availability."""

    required = {"27", "38", "47", "58", "60", "61", "62"}
    missing = sorted(required - set(artifacts))
    if missing:
        raise ProtocolError(f"decision matrix lacks artifacts {missing}")
    variables = {
        "R": _explicit(
            artifacts["27"], evidence_key="decision_variable_R", diagnostic_id="27"
        ),
        "T": _explicit(
            artifacts["47"], evidence_key="decision_variable_T", diagnostic_id="47"
        ),
        "E": _explicit(
            artifacts["58"], evidence_key="decision_variable_E", diagnostic_id="58"
        ),
        "L": _explicit(
            artifacts["60"], evidence_key="decision_variable_L", diagnostic_id="60"
        ),
        "F": _explicit(
            artifacts["61"], evidence_key="decision_variable_F", diagnostic_id="61"
        ),
        "X": _explicit(
            artifacts["62"], evidence_key="decision_variable_X", diagnostic_id="62"
        ),
    }
    policy = _mapping(artifacts["27"], "evidence")
    triangle = _mapping(artifacts["38"], "evidence")
    executability = _mapping(artifacts["62"], "evidence")
    auxiliaries = {
        "R_memoryless": str(policy.get("decision_variable_R_memoryless", "UNKNOWN")),
        "history_or_phase_recovery": str(
            policy.get("history_or_phase_recovery", "UNKNOWN")
        ),
        "raw_or_teacher_positive_domain_gap": str(
            triangle.get("raw_or_teacher_positive_domain_gap", "UNKNOWN")
        ),
        "failure_cause": str(
            executability.get("failure_cause", "UNKNOWN")
        ),
    }
    for key in ("R_memoryless", "history_or_phase_recovery"):
        if auxiliaries[key] not in {"+", "-", "UNKNOWN"}:
            auxiliaries[key] = "UNKNOWN"
    return variables, auxiliaries


def _matches(
    when: Mapping[str, Any],
    variables: Mapping[str, DecisionVariable],
    auxiliaries: Mapping[str, Any],
) -> bool:
    for key, expected in when.items():
        actual: Any
        if key in variables:
            actual = variables[key].value
        else:
            actual = auxiliaries.get(key, "UNKNOWN")
        if actual != expected:
            return False
    return True


def select_frozen_family(
    matrix: Any,
    variables: Mapping[str, DecisionVariable],
    auxiliaries: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(matrix, list) or len(matrix) != 6:
        raise ProtocolError("frozen decision matrix must contain exactly six rows")
    for expected_priority, row in enumerate(matrix, start=1):
        if not isinstance(row, Mapping) or int(row.get("priority", -1)) != expected_priority:
            raise ProtocolError("decision-matrix priorities changed")
        when = row.get("when")
        if not isinstance(when, Mapping):
            raise ProtocolError("decision-matrix row lacks a when mapping")
        if _matches(when, variables, auxiliaries):
            return {
                "status": "SELECTED",
                "matched_rule_id": str(row["rule_id"]),
                "selected_family": str(row["family"]),
                "reason": "the first matching preregistered decision-matrix row was selected",
            }
    unknown = [name for name, value in variables.items() if value.value == "UNKNOWN"]
    return {
        "status": "NO_ELIGIBLE_FAMILY",
        "matched_rule_id": None,
        "selected_family": None,
        "reason": (
            "no preregistered row matched"
            + (f"; unresolved variables: {unknown}" if unknown else "")
        ),
    }


__all__ = [
    "DecisionVariable",
    "derive_variables",
    "select_frozen_family",
]
