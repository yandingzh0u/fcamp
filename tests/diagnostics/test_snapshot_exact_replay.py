from __future__ import annotations

from types import SimpleNamespace

import torch

from diagnostics.common.canonical_collection import (
    _prefix_engine_state,
    restore_snapshot_bank,
)
from diagnostics.common.noise_bank import CollectorMode, NoiseBank
from diagnostics.common.rollout_collector import same_state_branch_identity
from diagnostics.common.snapshot_bank import SnapshotBank


class _Scene:
    def __init__(self) -> None:
        self.env_origins = torch.tensor([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        self.reset_calls = 0
        self.update_calls = 0

    def reset(self, *, env_ids: torch.Tensor) -> None:
        assert torch.equal(env_ids, torch.tensor([0, 1]))
        self.reset_calls += 1

    def update(self, physics_dt: float) -> None:
        assert physics_dt == 0.005
        self.update_calls += 1


class _Simulation:
    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _ReplayEnvironment:
    num_envs = 2
    device = torch.device("cpu")
    physics_dt = 0.005

    def __init__(self) -> None:
        self.scene = _Scene()
        self.sim = _Simulation()
        self.action_joint_ids = torch.tensor([0, 1])
        self.robot = SimpleNamespace(
            data=SimpleNamespace(
                root_link_pose_w=torch.zeros(2, 7),
                joint_pos=torch.zeros(2, 2),
                joint_vel=torch.zeros(2, 2),
                default_joint_pos=torch.zeros(2, 2),
            )
        )
        self.phase_steps = torch.zeros(2)
        self.episode_steps = torch.zeros(2, dtype=torch.long)
        self.episode_ids = torch.zeros(2, dtype=torch.long)
        self._next_episode_id = 0
        self.last_action = torch.zeros(2, 2)
        self.next_push_step = torch.zeros(2, dtype=torch.long)
        self.first_push_step = torch.zeros(2, dtype=torch.long)
        self.adaptive_sampler = SimpleNamespace(
            bin_failed_count=torch.zeros(3),
            current_bin_failed_count=torch.zeros(3),
        )
        self._failure_recorded = torch.zeros(2, dtype=torch.bool)
        self.contact_sensor = SimpleNamespace(
            data=SimpleNamespace(),
            _timestamp=torch.zeros(2),
            _timestamp_last_update=torch.zeros(2),
            _is_outdated=torch.zeros(2, dtype=torch.bool),
        )
        self.root_velocity = torch.zeros(2, 6)

    def _write_robot_state(
        self,
        *,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        self.robot.data.root_link_pose_w[env_ids, :3] = (
            root_pos + self.scene.env_origins.index_select(0, env_ids)
        )
        self.robot.data.root_link_pose_w[env_ids, 3:7] = root_quat
        self.root_velocity[env_ids] = torch.cat((root_lin_vel, root_ang_vel), dim=-1)
        self.robot.data.joint_pos[env_ids] = joint_pos
        self.robot.data.joint_vel[env_ids] = joint_vel


def _engine_state(offset: float) -> dict[str, torch.Tensor]:
    root_pose = torch.tensor(
        [
            [offset, 0.0, 0.8, 1.0, 0.0, 0.0, 0.0],
            [4.0 + offset, 0.0, 0.9, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    return {
        "root_pose_w": root_pose,
        "root_velocity_w": torch.full((2, 6), offset),
        "joint_pos": torch.tensor([[offset, 1.0 + offset], [2.0 + offset, 3.0 + offset]]),
        "joint_vel": torch.full((2, 2), 0.25 + offset),
        "phase_steps": torch.tensor([0.0, 12.0]),
        "episode_steps": torch.tensor([0, 3]),
        "episode_ids": torch.tensor([10, 11]),
        "next_episode_id": torch.tensor(12),
        "last_action": torch.full((2, 2), offset),
        "next_push_step": torch.tensor([30, 40]),
        "first_push_step": torch.tensor([-1, 2]),
        "adaptive_bin_failed_count": torch.tensor([1.0, 2.0, 3.0]),
        "adaptive_current_bin_failed_count": torch.tensor([4.0, 5.0, 6.0]),
        "failure_recorded": torch.tensor([False, True]),
        "contact_timestamp": torch.tensor([1.0, 2.0]),
        "contact_timestamp_last_update": torch.tensor([0.5, 1.5]),
        "contact_is_outdated": torch.tensor([False, True]),
    }


def test_same_snapshot_and_noisebank_replay_exactly_through_engine_restore() -> None:
    clean = _engine_state(0.0)
    controlled = _engine_state(0.125)
    fields = {
        **_prefix_engine_state(clean, state_name="clean", num_envs=2),
        **_prefix_engine_state(controlled, state_name="controlled", num_envs=2),
    }
    bank = SnapshotBank.from_batched_tensors(
        snapshot_ids=("snapshot-a", "snapshot-b"),
        phase=torch.tensor([0.0, 12.0]),
        state=fields,
        reset_randomization={"joint_delta": torch.full((2, 2), 0.125)},
        physics_randomization={"default_joint_pos": torch.zeros(2, 2)},
        bank_seed=20260803,
    )
    env = _ReplayEnvironment()
    noise = NoiseBank(seed=20260803).common_action_epsilon(
        bank.snapshot_ids, horizon=5, action_dim=2
    )

    def restore() -> None:
        restore_snapshot_bank(env, bank, mode=CollectorMode.CONTROLLED_ENVIRONMENT)

    def branch() -> dict[str, torch.Tensor]:
        joint_path = []
        root_path = []
        for step in range(5):
            delta = 0.01 * noise[:, step]
            env.robot.data.joint_pos.add_(delta)
            env.robot.data.root_link_pose_w[:, :2].add_(delta)
            env.last_action.copy_(noise[:, step])
            joint_path.append(env.robot.data.joint_pos.clone())
            root_path.append(env.robot.data.root_link_pose_w.clone())
        return {
            "joint_path": torch.stack(joint_path),
            "root_path": torch.stack(root_path),
            "last_action": env.last_action.clone(),
        }

    result = same_state_branch_identity(restore, branch)
    assert result == {"identical": True, "max_abs_error": 0.0}
    # The selected branch really was the controlled snapshot, not a zero-state
    # coincidence or the clean snapshot.
    restore()
    torch.testing.assert_close(env.robot.data.joint_pos, controlled["joint_pos"])
    torch.testing.assert_close(env.last_action, controlled["last_action"])
    assert env.scene.reset_calls >= 3
    assert env.scene.update_calls >= 3
    assert env.sim.reset_calls >= 3
