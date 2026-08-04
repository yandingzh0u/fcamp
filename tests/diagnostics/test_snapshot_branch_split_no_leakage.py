import torch

from diagnostics.common.snapshot_bank import SnapshotBank
from diagnostics.common.trajectory_split import TrajectoryRecord, assign_trajectory_splits


def test_all_branches_from_one_snapshot_share_one_split() -> None:
    snapshot_ids = [f"snapshot-{index}" for index in range(9)]
    bank = SnapshotBank.from_batched_tensors(
        snapshot_ids=snapshot_ids,
        phase=torch.arange(9, dtype=torch.float32),
        state={"joint_pos": torch.randn(9, 29), "joint_vel": torch.randn(9, 29)},
        reset_randomization={"root_offset": torch.randn(9, 3)},
        physics_randomization={"friction": torch.rand(9, 1)},
        bank_seed=20260804,
    )
    records: list[TrajectoryRecord] = []
    for snapshot_id in bank.snapshot_ids:
        for checkpoint_id in ("u100", "u200", "u300"):
            branch = bank.branch(
                snapshot_id,
                branch_id=f"{snapshot_id}-{checkpoint_id}",
                checkpoint_id=checkpoint_id,
            )
            records.append(
                TrajectoryRecord(
                    sample_id=branch.branch_id,
                    trajectory_id=branch.branch_id,
                    snapshot_id=branch.snapshot_id,
                    checkpoint_lineage_id=f"lineage-{snapshot_id}",
                    checkpoint_id=branch.checkpoint_id,
                    branch_id=branch.branch_id,
                )
            )

    assignment = assign_trajectory_splits(records, seed=3)
    for snapshot_id in bank.snapshot_ids:
        branch_splits = {
            assignment.split_for(record.sample_id)
            for record in records
            if record.snapshot_id == snapshot_id
        }
        assert len(branch_splits) == 1


def test_snapshot_get_returns_a_clone_not_mutable_bank_storage() -> None:
    bank = SnapshotBank.from_batched_tensors(
        snapshot_ids=["s0"],
        phase=torch.tensor([0.0]),
        state={"joint_pos": torch.ones(1, 2)},
        bank_seed=1,
    )
    fetched = bank.get("s0")
    fetched.state["joint_pos"].zero_()
    assert torch.equal(bank.get("s0").state["joint_pos"], torch.ones(2))
