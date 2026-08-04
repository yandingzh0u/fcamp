"""Leakage-proof grouped train/validation/test assignment."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Mapping, Sequence


class TrajectorySplitError(ValueError):
    pass


SPLIT_NAMES = ("train", "validation", "test")
# The row-level train/validation/test split and the checkpoint-lineage test are
# deliberately *nested*, not one flat grouping problem.  In the fully crossed
# checkpoint x snapshot design, grouping by checkpoint lineage here would join
# every snapshot into one connected component and make an inner split
# mathematically impossible.  Lineage generalization is therefore audited as a
# separate outer holdout axis below.
INNER_GROUP_FIELDS = ("trajectory_id", "snapshot_id")
OUTER_HOLDOUT_FIELD = "checkpoint_lineage_id"
# Backward-compatible public name: callers that do not specify fields get the
# inner, leakage-proof row split.
GROUP_FIELDS = INNER_GROUP_FIELDS


@dataclass(frozen=True)
class TrajectoryRecord:
    sample_id: str
    trajectory_id: str
    snapshot_id: str
    checkpoint_lineage_id: str
    checkpoint_id: str = ""
    branch_id: str = ""
    collector_mode: str = ""

    def __post_init__(self) -> None:
        for field in ("sample_id", *GROUP_FIELDS):
            if not str(getattr(self, field)):
                raise TrajectorySplitError(f"{field} must not be empty")


@dataclass(frozen=True)
class SplitAssignment:
    sample_to_split: Mapping[str, str]
    seed: int
    fractions: Mapping[str, float]

    def split_for(self, sample_id: str) -> str:
        try:
            return self.sample_to_split[sample_id]
        except KeyError as error:
            raise KeyError(f"sample {sample_id!r} has no split assignment") from error


@dataclass(frozen=True)
class SplitAudit:
    ok: bool
    leaks: Mapping[str, Mapping[str, tuple[str, ...]]]
    split_counts: Mapping[str, int]

    def require_valid(self) -> None:
        if not self.ok:
            raise TrajectorySplitError(f"split leakage detected: {self.leaks}")


@dataclass(frozen=True)
class OuterHoldoutAudit:
    """Feasibility audit for the separate checkpoint-lineage holdout axis."""

    feasible: bool
    field: str
    identities: tuple[str, ...]
    minimum_identities: int
    reason: str

    def require_feasible(self) -> None:
        if not self.feasible:
            raise TrajectorySplitError(self.reason)


class _DisjointSet:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))
        self.rank = [0] * count

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, first: int, second: int) -> None:
        left, right = self.find(first), self.find(second)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def _validate_records(records: Sequence[TrajectoryRecord]) -> None:
    if not records:
        raise TrajectorySplitError("cannot split an empty record collection")
    sample_ids = [record.sample_id for record in records]
    if len(set(sample_ids)) != len(sample_ids):
        raise TrajectorySplitError("sample_id values must be unique")


def leakage_components(
    records: Sequence[TrajectoryRecord],
    *,
    group_fields: Sequence[str] = GROUP_FIELDS,
) -> tuple[tuple[int, ...], ...]:
    """Return connected components induced by every anti-leakage identity."""

    _validate_records(records)
    unknown = [field for field in group_fields if field not in TrajectoryRecord.__dataclass_fields__]
    if unknown:
        raise TrajectorySplitError(f"unknown grouping fields: {unknown}")
    dsu = _DisjointSet(len(records))
    first_seen: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        for field in group_fields:
            key = (field, str(getattr(record, field)))
            previous = first_seen.setdefault(key, index)
            dsu.union(index, previous)
    by_root: dict[int, list[int]] = {}
    for index in range(len(records)):
        by_root.setdefault(dsu.find(index), []).append(index)
    return tuple(tuple(indices) for indices in by_root.values())


def _component_hash(records: Sequence[TrajectoryRecord], indices: Sequence[int], seed: int) -> str:
    identities = sorted(records[index].sample_id for index in indices)
    payload = f"{int(seed)}\0" + "\0".join(identities)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assign_trajectory_splits(
    records: Sequence[TrajectoryRecord],
    *,
    fractions: Mapping[str, float] | None = None,
    seed: int = 0,
    group_fields: Sequence[str] = GROUP_FIELDS,
) -> SplitAssignment:
    fractions = dict(fractions or {"train": 0.70, "validation": 0.15, "test": 0.15})
    if set(fractions) != set(SPLIT_NAMES):
        raise TrajectorySplitError(f"fractions must contain exactly {SPLIT_NAMES}")
    if any(float(value) < 0.0 for value in fractions.values()):
        raise TrajectorySplitError("split fractions must be nonnegative")
    total_fraction = sum(float(value) for value in fractions.values())
    if abs(total_fraction - 1.0) > 1.0e-9:
        raise TrajectorySplitError("split fractions must sum to one")

    components = list(leakage_components(records, group_fields=group_fields))
    active_splits = [name for name in SPLIT_NAMES if float(fractions[name]) > 0.0]
    if len(components) < len(active_splits):
        raise TrajectorySplitError(
            "anti-leakage grouping leaves fewer independent components than active splits; "
            f"components={len(components)}, active_splits={len(active_splits)}"
        )
    # Large groups are placed first; stable hashes make all ties independent of
    # input row order.
    components.sort(key=lambda part: (-len(part), _component_hash(records, part, seed)))
    targets = {name: float(fractions[name]) * len(records) for name in SPLIT_NAMES}
    counts = {name: 0 for name in SPLIT_NAMES}
    assignment: dict[str, str] = {}
    for component in components:
        # Relative deficit avoids always starving a small validation/test split.
        best = max(
            active_splits,
            key=lambda name: (
                (targets[name] - counts[name]) / max(targets[name], 1.0),
                -SPLIT_NAMES.index(name),
            ),
        )
        for index in component:
            assignment[records[index].sample_id] = best
        counts[best] += len(component)
    empty = [name for name in active_splits if counts[name] == 0]
    if empty:
        raise TrajectorySplitError(f"grouped assignment produced empty active splits: {empty}")
    result = SplitAssignment(
        sample_to_split=assignment,
        seed=int(seed),
        fractions={name: float(fractions[name]) for name in SPLIT_NAMES},
    )
    audit_trajectory_splits(records, result, group_fields=group_fields).require_valid()
    return result


def audit_trajectory_splits(
    records: Sequence[TrajectoryRecord],
    assignment: SplitAssignment | Mapping[str, str],
    *,
    group_fields: Sequence[str] = GROUP_FIELDS,
) -> SplitAudit:
    _validate_records(records)
    mapping = assignment.sample_to_split if isinstance(assignment, SplitAssignment) else assignment
    expected = {record.sample_id for record in records}
    missing = expected - set(mapping)
    extra = set(mapping) - expected
    if missing or extra:
        raise TrajectorySplitError(
            f"assignment sample mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    invalid = sorted({str(mapping[sample]) for sample in expected} - set(SPLIT_NAMES))
    if invalid:
        raise TrajectorySplitError(f"invalid split names: {invalid}")

    leaks: dict[str, dict[str, tuple[str, ...]]] = {}
    for field in group_fields:
        value_splits: dict[str, set[str]] = {}
        for record in records:
            value_splits.setdefault(str(getattr(record, field)), set()).add(
                str(mapping[record.sample_id])
            )
        field_leaks = {
            value: tuple(sorted(splits))
            for value, splits in value_splits.items()
            if len(splits) > 1
        }
        if field_leaks:
            leaks[field] = field_leaks
    counts = {
        name: sum(str(mapping[record.sample_id]) == name for record in records)
        for name in SPLIT_NAMES
    }
    return SplitAudit(ok=not leaks, leaks=leaks, split_counts=counts)


def audit_outer_lineage_holdout(
    records: Sequence[TrajectoryRecord],
    *,
    minimum_lineages: int = 2,
) -> OuterHoldoutAudit:
    """Report whether a genuine held-out checkpoint-lineage test is possible.

    This function does not assign the ordinary row split.  It is intentionally
    separate from :func:`assign_trajectory_splits`: inner evaluation groups by
    trajectory/snapshot, while an outer experiment trains on complete lineage
    sets and evaluates on a different lineage.  A single-lineage repository is
    valid for the inner split but must mark the outer test as unavailable.
    """

    _validate_records(records)
    if int(minimum_lineages) < 2:
        raise TrajectorySplitError("minimum_lineages must be at least 2")
    identities = tuple(
        sorted({str(getattr(record, OUTER_HOLDOUT_FIELD)) for record in records})
    )
    feasible = len(identities) >= int(minimum_lineages)
    reason = (
        "checkpoint-lineage outer holdout is feasible"
        if feasible
        else (
            "checkpoint-lineage outer holdout requires at least "
            f"{int(minimum_lineages)} independent lineages; found {len(identities)}"
        )
    )
    return OuterHoldoutAudit(
        feasible=feasible,
        field=OUTER_HOLDOUT_FIELD,
        identities=identities,
        minimum_identities=int(minimum_lineages),
        reason=reason,
    )
