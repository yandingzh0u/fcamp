"""Small dependency-free statistics used by stage-0 diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any, Iterable


def flatten_numeric(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, nested in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten_numeric(nested, path))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result[prefix] = float(value)
    return result


def compare_numeric_mappings(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> dict[str, Any]:
    left = flatten_numeric(first)
    right = flatten_numeric(second)
    keys = sorted(set(left) | set(right))
    missing_first = [key for key in keys if key not in left]
    missing_second = [key for key in keys if key not in right]
    rows: list[dict[str, Any]] = []
    all_close = not missing_first and not missing_second
    max_abs = 0.0
    max_rel = 0.0
    for key in sorted(set(left) & set(right)):
        a = left[key]
        b = right[key]
        finite = math.isfinite(a) and math.isfinite(b)
        abs_diff = abs(a - b) if finite else math.inf
        scale = max(abs(a), abs(b), 1.0e-300) if finite else 1.0
        rel_diff = abs_diff / scale
        close = finite and abs_diff <= float(atol) + float(rtol) * abs(b)
        all_close = all_close and close
        max_abs = max(max_abs, abs_diff)
        max_rel = max(max_rel, rel_diff)
        if not close:
            rows.append(
                {
                    "key": key,
                    "first": a,
                    "second": b,
                    "absolute_difference": abs_diff,
                    "relative_difference": rel_diff,
                }
            )
    return {
        "all_close": bool(all_close),
        "atol": float(atol),
        "rtol": float(rtol),
        "shared_key_count": len(set(left) & set(right)),
        "missing_from_first": missing_first,
        "missing_from_second": missing_second,
        "max_absolute_difference": max_abs,
        "max_relative_difference": max_rel,
        "differences": rows,
    }


def first_threshold_crossing(
    rows: Iterable[Mapping[str, Any]],
    field: str,
    threshold: float,
    *,
    direction: str = "above",
) -> int | None:
    if direction not in {"above", "below"}:
        raise ValueError("direction must be 'above' or 'below'")
    ordered = sorted(rows, key=lambda row: int(row["update"]))
    for row in ordered:
        value = row.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        crossed = float(value) >= threshold if direction == "above" else float(value) <= threshold
        if crossed:
            return int(row["update"])
    return None


def monotonic_violations(
    rows: Iterable[Mapping[str, Any]],
    field: str,
    *,
    tolerance: float = 0.0,
    increasing: bool = True,
) -> list[dict[str, float | int]]:
    ordered = sorted(rows, key=lambda row: int(row["update"]))
    violations: list[dict[str, float | int]] = []
    previous: tuple[int, float] | None = None
    for row in ordered:
        value = row.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        current = (int(row["update"]), float(value))
        if previous is not None:
            delta = current[1] - previous[1]
            bad = delta < -tolerance if increasing else delta > tolerance
            if bad:
                violations.append(
                    {
                        "previous_update": previous[0],
                        "update": current[0],
                        "previous_value": previous[1],
                        "value": current[1],
                        "delta": delta,
                    }
                )
        previous = current
    return violations
