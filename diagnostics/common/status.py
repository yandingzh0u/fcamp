"""Status types shared by discovery-suite entrypoints and the runner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class DiagnosticStatus(str, Enum):
    """The only four states allowed by the frozen discovery protocol.

    ``FAIL`` is a valid scientific result.  ``INVALID_PROTOCOL`` means the
    scientific result cannot be interpreted.  They must never be conflated.
    """

    PASS = "PASS"
    FAIL = "FAIL"
    INVALID_PROTOCOL = "INVALID_PROTOCOL"
    SKIPPED_DEPENDENCY = "SKIPPED_DEPENDENCY"

    @classmethod
    def parse(cls, value: "DiagnosticStatus | str") -> "DiagnosticStatus":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().upper())
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(f"unknown diagnostic status {value!r}; expected one of {allowed}") from exc


@dataclass(frozen=True)
class DiagnosticResult:
    """Serializable execution result for one diagnostic node."""

    diagnostic_id: str
    name: str
    stage: str
    status: DiagnosticStatus
    dependencies: tuple[str, ...] = ()
    command: tuple[str, ...] = ()
    required: bool = True
    blocking: bool = False
    reason: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    exit_code: int | None = None
    expected_outputs: tuple[str, ...] = ()
    observed_outputs: tuple[str, ...] = ()
    status_artifact: str | None = None
    stdout_log: str | None = None
    stderr_log: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        # JSON arrays are part of the result contract, not implementation tuples.
        for key in ("dependencies", "command", "expected_outputs", "observed_outputs"):
            payload[key] = list(payload[key])
        payload["metadata"] = dict(self.metadata)
        return payload


def aggregate_status(
    results: Sequence[DiagnosticResult | Mapping[str, Any]],
) -> DiagnosticStatus:
    """Aggregate without ever turning missing/invalid evidence into PASS.

    Protocol invalidity has highest priority, followed by a valid scientific
    failure, then an unexecuted dependency.  Only an all-PASS collection is
    reported as PASS.
    """

    statuses: list[DiagnosticStatus] = []
    for result in results:
        raw = result.status if isinstance(result, DiagnosticResult) else result["status"]
        statuses.append(DiagnosticStatus.parse(raw))
    if not statuses:
        return DiagnosticStatus.INVALID_PROTOCOL
    for candidate in (
        DiagnosticStatus.INVALID_PROTOCOL,
        DiagnosticStatus.FAIL,
        DiagnosticStatus.SKIPPED_DEPENDENCY,
    ):
        if candidate in statuses:
            return candidate
    return DiagnosticStatus.PASS


def status_counts(
    results: Sequence[DiagnosticResult | Mapping[str, Any]],
) -> dict[str, int]:
    counts = {status.value: 0 for status in DiagnosticStatus}
    for result in results:
        raw = result.status if isinstance(result, DiagnosticResult) else result["status"]
        counts[DiagnosticStatus.parse(raw).value] += 1
    return counts


def dependencies_passed(
    dependencies: Sequence[str],
    completed: Mapping[str, DiagnosticResult],
) -> tuple[bool, tuple[str, ...]]:
    """Return whether every declared prerequisite has a valid PASS result."""

    blocked = tuple(
        dependency
        for dependency in dependencies
        if dependency not in completed
        or completed[dependency].status is not DiagnosticStatus.PASS
    )
    return not blocked, blocked


__all__ = [
    "DiagnosticResult",
    "DiagnosticStatus",
    "aggregate_status",
    "dependencies_passed",
    "status_counts",
]
