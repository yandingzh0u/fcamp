from diagnostics.common.trajectory_split import (
    TrajectoryRecord,
    assign_trajectory_splits,
    audit_outer_lineage_holdout,
    audit_trajectory_splits,
)


def _records() -> list[TrajectoryRecord]:
    rows: list[TrajectoryRecord] = []
    for group in range(12):
        for step in range(3):
            rows.append(
                TrajectoryRecord(
                    sample_id=f"sample-{group}-{step}",
                    trajectory_id=f"trajectory-{group}",
                    snapshot_id=f"snapshot-{group}",
                    checkpoint_lineage_id=f"lineage-{group // 2}",
                    checkpoint_id=f"checkpoint-{group}",
                )
            )
    return rows


def test_complete_trajectories_and_snapshots_never_cross_inner_splits() -> None:
    records = _records()
    assignment = assign_trajectory_splits(records, seed=20260804)
    audit = audit_trajectory_splits(records, assignment)

    assert audit.ok
    assert not audit.leaks
    for field in ("trajectory_id", "snapshot_id"):
        seen: dict[str, set[str]] = {}
        for record in records:
            seen.setdefault(getattr(record, field), set()).add(
                assignment.split_for(record.sample_id)
            )
        assert all(len(splits) == 1 for splits in seen.values())


def test_split_is_deterministic_under_input_row_reordering() -> None:
    records = _records()
    forward = assign_trajectory_splits(records, seed=17)
    reverse = assign_trajectory_splits(list(reversed(records)), seed=17)
    assert forward.sample_to_split == reverse.sample_to_split


def test_one_lineage_allows_inner_split_but_outer_holdout_is_infeasible() -> None:
    records = [
        TrajectoryRecord(
            sample_id=f"sample-{index}",
            trajectory_id=f"trajectory-{index}",
            snapshot_id=f"snapshot-{index}",
            checkpoint_lineage_id="only-lineage",
        )
        for index in range(6)
    ]
    assignment = assign_trajectory_splits(records)
    assert audit_trajectory_splits(records, assignment).ok

    outer = audit_outer_lineage_holdout(records)
    assert not outer.feasible
    assert outer.identities == ("only-lineage",)
    assert "requires at least 2" in outer.reason


def test_multiple_lineages_make_outer_holdout_feasible() -> None:
    outer = audit_outer_lineage_holdout(_records())
    assert outer.feasible
    assert len(outer.identities) == 6
