from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from envs.motion import MimicMotionReference

_spec = ModuleType("envs.spec")
_spec.UNDESIRED_CONTACT_THRESHOLD = 1.0
sys.modules["envs.spec"] = _spec
from envs.reward import MimicRewardMixin
sys.modules.pop("envs.spec", None)


ROOT = Path(__file__).parents[1]
LARGEBOX_MOTION = (
    ROOT.parent
    / "holosoma"
    / "src"
    / "holosoma"
    / "holosoma"
    / "data"
    / "motions"
    / "g1_29dof"
    / "whole_body_tracking"
    / "sub3_largebox_003_mj.npz"
)


def _load_motion(path: Path) -> MimicMotionReference:
    with np.load(path, allow_pickle=True) as data:
        body_names = [str(name) for name in data["body_names"]]
        joint_names = [str(name) for name in data["joint_names"]]
    track_names = ("pelvis", "torso_link")
    track_ids = torch.tensor(
        [body_names.index(name) for name in track_names], dtype=torch.long
    )
    return MimicMotionReference(
        path,
        track_ids,
        body_names.index("torso_link"),
        torch.device("cpu"),
        robot_body_names=body_names,
        action_joint_names=joint_names,
        root_body_name="pelvis",
    )


def _motion_arrays() -> dict[str, np.ndarray]:
    with np.load(LARGEBOX_MOTION, allow_pickle=True) as data:
        return {name: np.asarray(data[name]).copy() for name in data.files}


def _write_motion(
    path: Path,
    *,
    mutate=None,
    drop: tuple[str, ...] = (),
) -> None:
    arrays = _motion_arrays()
    for name in drop:
        arrays.pop(name, None)
    if mutate is not None:
        mutate(arrays)
    np.savez(path, **arrays)


def _quat_rotate_wxyz(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quat[..., 1:]
    twice_cross = 2.0 * np.cross(xyz, vector)
    return vector + quat[..., :1] * twice_cross + np.cross(xyz, twice_cross)


def test_largebox_uses_raw_joint_velocity_prefix_as_world_root_link_velocity() -> None:
    motion = _load_motion(LARGEBOX_MOTION)
    with np.load(LARGEBOX_MOTION, allow_pickle=True) as data:
        expected = np.asarray(data["joint_vel"], dtype=np.float32)[:, :6]
        root_quat = np.asarray(data["joint_pos"], dtype=np.float32)[:, 3:7]

    torch.testing.assert_close(
        motion.root_link_velocity_w,
        torch.from_numpy(expected),
        rtol=0.0,
        atol=0.0,
    )
    rotated_angular = _quat_rotate_wxyz(root_quat, expected[:, 3:])
    assert np.max(np.abs(rotated_angular - expected[:, 3:])) > 0.1
    frame = motion.get_frame(torch.tensor([0, 100, 324]))
    torch.testing.assert_close(
        frame["root_ang_vel_w"],
        torch.from_numpy(expected[[0, 100, 324], 3:]),
        rtol=0.0,
        atol=0.0,
    )
    assert "body_lin_vel_w" not in frame
    assert "body_ang_vel_w" not in frame
    assert not hasattr(motion, "body_lin_vel_full_w")
    assert not hasattr(motion, "body_ang_vel_full_w")
    assert not hasattr(motion, "get_fcamp_demo_history")
    assert not hasattr(motion, "get_imitation_frame_at_times")


@pytest.mark.parametrize("mode", ["missing", "nonfinite"])
def test_legacy_body_velocity_arrays_are_completely_ignored(
    tmp_path: Path,
    mode: str,
) -> None:
    path = tmp_path / f"legacy_velocity_{mode}.npz"
    if mode == "missing":
        _write_motion(
            path,
            drop=("body_lin_vel_w", "body_ang_vel_w"),
        )
    else:
        def poison(arrays: dict[str, np.ndarray]) -> None:
            arrays["body_lin_vel_w"].fill(np.nan)
            arrays["body_ang_vel_w"].fill(np.nan)

        _write_motion(path, mutate=poison)

    motion = _load_motion(path)
    assert motion.num_frames == 325
    assert bool(torch.isfinite(motion.root_link_velocity_w).all())


def test_rotated_or_perturbed_root_velocity_fails_before_tensor_upload(
    tmp_path: Path,
) -> None:
    perturbed = tmp_path / "perturbed_root_velocity.npz"

    def perturb(arrays: dict[str, np.ndarray]) -> None:
        arrays["joint_vel"][100, 0] += 0.01

    _write_motion(perturbed, mutate=perturb)
    with pytest.raises(ValueError, match="centered pelvis pose difference"):
        _load_motion(perturbed)

    rotated = tmp_path / "rotated_root_angular_velocity.npz"

    def rotate(arrays: dict[str, np.ndarray]) -> None:
        quat = arrays["joint_pos"][:, 3:7]
        arrays["joint_vel"][:, 3:6] = _quat_rotate_wxyz(
            quat, arrays["joint_vel"][:, 3:6]
        )

    _write_motion(rotated, mutate=rotate)
    with pytest.raises(ValueError, match="centered pelvis pose difference"):
        _load_motion(rotated)


def test_motion_loader_rejects_nonunit_quaternions_and_root_pose_mismatch(
    tmp_path: Path,
) -> None:
    nonunit = tmp_path / "nonunit.npz"

    def scale_quat(arrays: dict[str, np.ndarray]) -> None:
        arrays["body_quat_w"][10, 1] *= 2.0

    _write_motion(nonunit, mutate=scale_quat)
    with pytest.raises(ValueError, match="unit length"):
        _load_motion(nonunit)

    mismatch = tmp_path / "root_pose_mismatch.npz"

    def shift_root(arrays: dict[str, np.ndarray]) -> None:
        arrays["joint_pos"][10, 0] += 0.01

    _write_motion(mismatch, mutate=shift_root)
    with pytest.raises(ValueError, match="root DOF pose"):
        _load_motion(mismatch)


class _FixedRewardFixture(MimicRewardMixin):
    def __init__(self) -> None:
        batch = 2
        bodies = 14
        joints = 2
        identity = torch.zeros(batch, bodies, 4)
        identity[..., 0] = 1.0
        anchor_identity = identity[:, 0]
        robot_body_pos = torch.zeros(batch, bodies, 3)
        self._context = {
            "reference": {
                "anchor_pos_w": torch.zeros(batch, 3),
                "anchor_quat_w": anchor_identity,
            },
            "robot_anchor_pos_w": torch.zeros(batch, 3),
            "robot_anchor_quat_w": anchor_identity,
            "body_pos_relative_w": robot_body_pos,
            "body_quat_relative_w": identity,
            "robot_body_pos_w": robot_body_pos,
            "robot_body_quat_w": identity,
        }
        joint_pos = torch.zeros(batch, joints)
        joint_pos[0, 0] = 1.2
        limits = torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]]).expand(
            batch, -1, -1
        )
        self.robot = SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=joint_pos,
                soft_joint_pos_limits=limits,
            )
        )
        forces = torch.zeros(batch, 2, 1, 3)
        forces[0, 0, 0, 0] = 2.0
        self.contact_sensor = SimpleNamespace(
            data=SimpleNamespace(net_forces_w_history=forces)
        )
        self.action_joint_ids = torch.arange(joints)
        self.undesired_contact_body_ids = torch.tensor([0])
        self.config = SimpleNamespace(action_rate_weight=0.1)
        self.dt = 0.02

    def get_tracking_context(self) -> dict[str, torch.Tensor]:
        return self._context


def test_fixed_reward_has_exact_pose_only_weights_and_one_dt_factor() -> None:
    env = _FixedRewardFixture()
    actions = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    previous = torch.zeros_like(actions)

    reward, terms = env.compute_reward(actions, previous)

    # env 0: 5 pose - 0.1 action rate - 2.0 joint limit - 0.1 contact.
    torch.testing.assert_close(reward, torch.tensor([0.056, 0.1]))
    contribution_keys = (
        "anchor_pos_contribution",
        "anchor_ori_contribution",
        "body_pos_contribution",
        "body_ori_contribution",
        "action_rate_contribution",
        "joint_limit_contribution",
        "undesired_contacts_contribution",
    )
    reconstructed = sum(terms[key] for key in contribution_keys)
    torch.testing.assert_close(reconstructed, reward)
    assert float(terms["reward_decomposition_error"].max()) < 1.0e-6
    assert set(terms).isdisjoint(
        {
            "body_lin_vel_reward",
            "body_ang_vel_reward",
            "diag_torso_ang_vel",
        }
    )
    for key in (
        "anchor_pos_reward",
        "anchor_ori_reward",
        "body_pos_reward",
        "body_ori_reward",
    ):
        torch.testing.assert_close(terms[key], torch.ones(2))


def test_fixed_reward_rejects_nonfinite_actions_and_wrong_config_weight() -> None:
    env = _FixedRewardFixture()
    actions = torch.zeros(2, 2)
    actions[0, 0] = torch.nan
    with pytest.raises(RuntimeError, match="non-finite"):
        env.compute_reward(actions, torch.zeros_like(actions))

    env.config.action_rate_weight = 0.2
    with pytest.raises(RuntimeError, match="action_rate_weight=0.1"):
        env.compute_reward(torch.zeros(2, 2), torch.zeros(2, 2))
