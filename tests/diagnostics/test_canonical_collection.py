from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from diagnostics.common.canonical_collection import (
    CANONICAL_INDEX_FIELDS,
    CanonicalCollectionProtocol,
    DenseCheckpointRecord,
    _engine_state_from_bank,
    _prefix_engine_state,
    evenly_spaced_phases,
    index_rows_from_tree,
    load_rollout_index,
    load_rollout_trajectory,
    switch_policy_state,
    write_rollout_index,
)
from diagnostics.common.manifest import ProtocolError, sha256_file
from diagnostics.common.noise_bank import CollectorMode
from diagnostics.common.snapshot_bank import SnapshotBank
from diagnostics.common.imitation_6901 import imitation_contract_metadata


def _spec() -> dict:
    return {
        "collection": {
            "num_envs": 4,
            "num_snapshots": 4,
            "horizon_control_steps": 5,
            "phase_strategy": "evenly_spaced",
            "snapshot_seed": 7,
            "collector_seed": 9,
            "common_action_noise_scales": [0.1, 0.25, 0.5],
            "canonical_checkpoint_updates": [10, 20],
        }
    }


def test_collection_protocol_has_exact_four_modes_and_three_common_scales() -> None:
    protocol = CanonicalCollectionProtocol.from_spec(_spec())
    assert protocol.branch_variants == (
        (CollectorMode.CLEAN_MEAN, 0.0),
        (CollectorMode.CONTROLLED_ENVIRONMENT, 0.0),
        (CollectorMode.COMMON_ACTION_NOISE, 0.1),
        (CollectorMode.COMMON_ACTION_NOISE, 0.25),
        (CollectorMode.COMMON_ACTION_NOISE, 0.5),
        (CollectorMode.NATIVE_STOCHASTIC, 0.0),
    )

    invalid = deepcopy(_spec())
    invalid["collection"]["common_action_noise_scales"] = [0.1, 0.2]
    with pytest.raises(ProtocolError, match="at least three"):
        CanonicalCollectionProtocol.from_spec(invalid)


def test_phase_grid_spans_full_motion_independent_of_rollout_horizon() -> None:
    spec = _spec()
    spec["collection"]["num_envs"] = 64
    spec["collection"]["num_snapshots"] = 64
    spec["collection"]["horizon_control_steps"] = 325
    protocol = CanonicalCollectionProtocol.from_spec(spec)
    env = SimpleNamespace(
        device="cpu",
        motion_start_phase=0,
        motion=SimpleNamespace(num_frames=325),
        # This is the bad legacy calculation: invoking it would collapse the
        # entire grid to phase zero.
        _adaptive_phase_range=lambda horizon: (0, max(0, 324 - horizon)),
    )

    phases = evenly_spaced_phases(env, protocol)

    assert phases.shape == (64,)
    assert int(phases[0]) == 0
    assert int(phases[-1]) == 323
    phase_bins = torch.div(phases * 16, 324, rounding_mode="floor").clamp(max=15)
    assert int(torch.unique(phase_bins).numel()) >= 16


def test_clean_and_controlled_engine_states_round_trip_without_aliasing() -> None:
    clean = {
        "joint_pos": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        "global_counter": torch.tensor(13),
    }
    controlled = {
        "joint_pos": clean["joint_pos"] + 0.5,
        "global_counter": torch.tensor(17),
    }
    fields = {
        **_prefix_engine_state(clean, state_name="clean", num_envs=4),
        **_prefix_engine_state(controlled, state_name="controlled", num_envs=4),
    }
    bank = SnapshotBank.from_batched_tensors(
        snapshot_ids=[f"s{i}" for i in range(4)],
        phase=torch.arange(4, dtype=torch.float32),
        state=fields,
        reset_randomization={"delta": torch.full((4, 1), 0.5)},
        physics_randomization={"startup": torch.arange(4).reshape(4, 1)},
        bank_seed=5,
    )
    restored_clean = _engine_state_from_bank(bank, state_name="clean")
    restored_controlled = _engine_state_from_bank(bank, state_name="controlled")
    torch.testing.assert_close(restored_clean["joint_pos"], clean["joint_pos"])
    torch.testing.assert_close(restored_controlled["joint_pos"], controlled["joint_pos"])
    assert restored_clean["global_counter"].item() == 13
    assert restored_controlled["global_counter"].item() == 17


class _FakePhysxView:
    @staticmethod
    def get_coms() -> torch.Tensor:
        return torch.zeros(2, 1, 7)

    @staticmethod
    def get_material_properties() -> torch.Tensor:
        return torch.ones(2, 1, 3)


class _FakeEnvironment:
    num_envs = 2

    def __init__(self) -> None:
        self.robot = SimpleNamespace(
            data=SimpleNamespace(default_joint_pos=torch.zeros(2, 3)),
            root_physx_view=_FakePhysxView(),
        )


class _FakeAlgorithm:
    def __init__(self) -> None:
        self.policy = nn.ModuleDict({"actor": nn.Linear(2, 2, bias=False)})
        self.actor = self.policy["actor"]
        self.critic = nn.Linear(1, 1)
        self.preflight_count = 0

    def validate_checkpoint_payload(self, payload) -> None:
        assert payload["update_idx"] == 10
        self.preflight_count += 1


def test_policy_switch_loads_only_policy_and_cannot_mutate_fake_environment(tmp_path: Path) -> None:
    trainer = SimpleNamespace(env=_FakeEnvironment(), algo=_FakeAlgorithm())
    target = nn.ModuleDict({"actor": nn.Linear(2, 2, bias=False)})
    with torch.no_grad():
        target["actor"].weight.fill_(3.0)
    platform = {"dataset_sha256": "a", "robot_asset_sha256": "b", "action_schema_sha256": "c"}
    path = tmp_path / "update_0010.pt"
    torch.save(
        {
            "update_idx": 10,
            "policy": target.state_dict(),
            "platform_identity": platform,
        },
        path,
    )
    environment_before = trainer.env.robot.data.default_joint_pos.clone()
    record = DenseCheckpointRecord(
        checkpoint_id="u0010",
        path=path,
        sha256=sha256_file(path),
        update=10,
        lineage_id="lineage-a",
    )
    switch_policy_state(trainer, record, expected_platform=platform)
    assert trainer.algo.preflight_count == 1
    torch.testing.assert_close(
        trainer.algo.policy["actor"].weight,
        torch.full_like(trainer.algo.policy["actor"].weight, 3.0),
    )
    torch.testing.assert_close(trainer.env.robot.data.default_joint_pos, environment_before)


def _fake_tree() -> dict:
    time, envs = 3, 2
    done = torch.tensor([[False, False], [True, False], [True, False]])
    return {
        "metadata": {
            "collector_mode": "native_stochastic",
            **imitation_contract_metadata(),
        },
        "trajectory": {
            "trajectory_id": ["t0", "t1"],
            "snapshot_id": ["s0", "s1"],
            "episode_id": torch.tensor([[1, 2], [1, 2], [1, 2]]),
            "done": done,
        },
        "observation": {"actor_full": torch.arange(time * envs * 4).reshape(time, envs, 4)},
        "state": {"joint_pos": torch.zeros(time, envs, 2)},
        "action": {"mean": torch.ones(time, envs, 2)},
        "reference": {"joint_pos": torch.zeros(time, envs, 2)},
        "outcome": {"reference_progress": torch.zeros(time, envs)},
        "imitation": {
            "agent_physx_raw_frame": torch.zeros(time, envs, 239),
            "agent_fk_aligned_raw_frame": torch.zeros(time, envs, 239),
            "reference_expert_raw_frame": torch.zeros(time, envs, 239),
            "phase_normalized_pre_step": torch.zeros(time, envs),
        },
    }


def test_parquet_index_points_to_cropped_tensor_shard(tmp_path: Path) -> None:
    tree = _fake_tree()
    shard = tmp_path / "rollouts/fake.pt"
    shard.parent.mkdir()
    torch.save(tree, shard)
    checkpoint_file = tmp_path / "checkpoint.pt"
    torch.save({"x": 1}, checkpoint_file)
    checkpoint = DenseCheckpointRecord(
        checkpoint_id="u0010",
        path=checkpoint_file,
        sha256=sha256_file(checkpoint_file),
        update=10,
        lineage_id="lineage",
    )
    rows = index_rows_from_tree(
        tree,
        checkpoint=checkpoint,
        mode="native_stochastic",
        common_sigma=0.0,
        horizon=3,
    )
    for row in rows:
        row["shard_path"] = "rollouts/fake.pt"
    index = tmp_path / "canonical_rollout_index.parquet"
    write_rollout_index(rows, index)
    loaded_rows = load_rollout_index(index)
    assert tuple(loaded_rows[0]) == CANONICAL_INDEX_FIELDS
    assert not bool(loaded_rows[0]["eligible_for_primary_overlap"])
    first = load_rollout_trajectory(index, loaded_rows[0])
    second = load_rollout_trajectory(index, loaded_rows[1])
    assert first["observation"]["actor_full"].shape == (2, 4)
    assert first["imitation"]["agent_physx_raw_frame"].shape == (2, 239)
    assert second["observation"]["actor_full"].shape == (3, 4)
    assert first["trajectory"]["trajectory_id"] == "t0"
    assert second["trajectory"]["snapshot_id"] == "s1"
