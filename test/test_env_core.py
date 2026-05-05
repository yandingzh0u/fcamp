from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("isaaclab")

from env import DEFAULT_MOTION_FILE, EnvConfig, G1Env, MimicEnvConfig
from env.config import (
    ANCHOR_ORI_TERMINATION_THRESHOLD,
    ANCHOR_Z_TERMINATION_THRESHOLD,
    EE_Z_TERMINATION_THRESHOLD,
    G1_MIMIC_ACTION_SCALE_VALUES,
    MIMIC_BODY_NAMES,
    MIMIC_EE_BODY_NAMES,
    OBS_DIM,
    _match_joint_expr,
)
from env.motion import MimicMotionReference
from env.robots.g1 import G1_29DOF_ACTION_NAMES, G1_29DOF_ASSET_JOINT_NAMES, G1_LOCAL_USD_PATH, make_g1_cfg


def test_env_exports_and_config_constants_are_consistent() -> None:
    env_cfg = EnvConfig(device="cpu", num_envs=2)
    mimic_cfg = MimicEnvConfig(device="cpu", num_envs=2)

    assert env_cfg.decimation == 4
    assert mimic_cfg.max_episode_steps == 1500
    assert OBS_DIM == 154
    assert DEFAULT_MOTION_FILE.is_file()
    assert Path(mimic_cfg.motion_file).is_file()
    assert len(MIMIC_BODY_NAMES) == 14
    assert set(MIMIC_EE_BODY_NAMES).issubset(set(MIMIC_BODY_NAMES))
    assert ANCHOR_Z_TERMINATION_THRESHOLD == pytest.approx(0.25)
    assert ANCHOR_ORI_TERMINATION_THRESHOLD == pytest.approx(0.8)
    assert EE_Z_TERMINATION_THRESHOLD == pytest.approx(0.25)


def test_joint_regex_matching_and_action_scales() -> None:
    assert _match_joint_expr(".*_hip_pitch_joint", "left_hip_pitch_joint")
    assert not _match_joint_expr("waist_yaw_joint", "left_hip_pitch_joint")
    assert len(G1_MIMIC_ACTION_SCALE_VALUES) == 29
    assert all(value > 0.0 for value in G1_MIMIC_ACTION_SCALE_VALUES)


def test_g1_robot_config_metadata_without_starting_sim() -> None:
    cfg = make_g1_cfg("/World/TestRobot", fix_root_link=True)

    assert G1_LOCAL_USD_PATH.is_file()
    assert G1_29DOF_ACTION_NAMES == G1_29DOF_ASSET_JOINT_NAMES
    assert len(G1_29DOF_ACTION_NAMES) == 29
    assert cfg.prim_path == "/World/TestRobot"
    assert cfg.spawn.articulation_props.fix_root_link is True


def test_g1_env_action_joint_state_on_fake_object() -> None:
    fake = G1Env.__new__(G1Env)
    fake.action_joint_ids = torch.tensor([2, 0])
    fake.robot = type("Robot", (), {})()
    fake.robot.data = type("Data", (), {})()
    fake.robot.data.joint_pos = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    fake.robot.data.joint_vel = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    joint_pos, joint_vel = G1Env.get_action_joint_state(fake)

    assert fake.action_dim == 2
    assert torch.equal(joint_pos, torch.tensor([[3.0, 1.0], [6.0, 4.0]]))
    assert torch.equal(joint_vel, torch.tensor([[0.3, 0.1], [0.6, 0.4]]))


def test_motion_reference_clamps_and_gathers_frames(tmp_path: Path) -> None:
    motion_path = tmp_path / "motion.npz"
    frames = 3
    joints = 29
    bodies = 5
    np.savez(
        motion_path,
        fps=np.array([60]),
        joint_pos=np.arange(frames * joints, dtype=np.float32).reshape(frames, joints),
        joint_vel=np.ones((frames, joints), dtype=np.float32),
        body_pos_w=np.arange(frames * bodies * 3, dtype=np.float32).reshape(frames, bodies, 3),
        body_quat_w=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (frames, bodies, 1)),
        body_lin_vel_w=np.ones((frames, bodies, 3), dtype=np.float32) * 2.0,
        body_ang_vel_w=np.ones((frames, bodies, 3), dtype=np.float32) * 3.0,
    )
    track_ids = torch.tensor([1, 3], dtype=torch.long)
    motion = MimicMotionReference(motion_path, track_ids, anchor_body_id=2, device=torch.device("cpu"))

    clamped = motion.clamp_time_steps(torch.tensor([-10, 1, 99]))
    frame = motion.get_frame(clamped)

    assert motion.fps == 60
    assert torch.equal(clamped, torch.tensor([0, 1, 2]))
    assert frame["joint_pos"].shape == (3, joints)
    assert frame["body_pos_w"].shape == (3, 2, 3)
    assert torch.equal(frame["body_pos_w"][0, 0], motion.body_pos_full_w[0, 1])
    assert torch.equal(frame["anchor_pos_w"][1], motion.body_pos_full_w[1, 2])
    assert torch.equal(frame["root_pos_w"][2], motion.body_pos_full_w[2, 0])
