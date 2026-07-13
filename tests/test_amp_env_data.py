from __future__ import annotations

import math
import ast
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import torch

from amp.features import canonicalize_amp_window
from env.amp_data import (
    G1_AMP_FRAME_DIM,
    G1_AMP_JOINT_AXES,
    G1_AMP_NUM_JOINTS,
    build_g1_amp_frame,
    history_indices,
    quat_wxyz_to_tan_norm,
    sample_contiguous_window_indices,
)
from env.motion import MimicMotionReference


def _identity_quat(*prefix: int) -> torch.Tensor:
    quat = torch.zeros(*prefix, 4)
    quat[..., 0] = 1.0
    return quat


def test_amp_joint_axes_match_the_runtime_urdf_action_order() -> None:
    source = ast.parse((Path(__file__).parents[1] / "env/robots/g1.py").read_text())
    joint_names = None
    for node in source.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "G1_29DOF_ASSET_JOINT_NAMES"
            for target in node.targets
        ):
            joint_names = ast.literal_eval(node.value)
            break
    assert joint_names is not None
    urdf = ET.parse(
        Path(__file__).parents[1] / "assets/robots/holosoma_g1/g1_29dof.urdf"
    ).getroot()
    urdf_axes = {
        joint.attrib["name"]: tuple(float(value) for value in joint.find("axis").attrib["xyz"].split())
        for joint in urdf.findall("joint")
        if joint.find("axis") is not None
    }
    assert [urdf_axes[name] for name in joint_names] == list(G1_AMP_JOINT_AXES)


def test_wxyz_rotation_encoding_matches_mimickit_tangent_normal() -> None:
    identity = quat_wxyz_to_tan_norm(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(identity, torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))

    # +90 degrees about z in explicit wxyz order rotates x onto +y while z stays z.
    half = math.pi / 4.0
    q_z90 = torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]])
    encoded = quat_wxyz_to_tan_norm(q_z90)
    expected = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    assert torch.allclose(encoded, expected, atol=1.0e-6)


def test_amp_frame_is_233d_and_translation_invariant_except_root_position() -> None:
    batch = 3
    root_pos = torch.randn(batch, 3)
    key_pos = root_pos[:, None, :] + torch.randn(batch, 5, 3)
    kwargs = {
        "root_pos": root_pos,
        "root_quat_wxyz": _identity_quat(batch),
        "joint_pos": torch.randn(batch, G1_AMP_NUM_JOINTS),
        "key_body_pos": key_pos,
        "root_lin_vel": torch.randn(batch, 3),
        "root_ang_vel": torch.randn(batch, 3),
        "joint_vel": torch.randn(batch, G1_AMP_NUM_JOINTS),
    }
    frame = build_g1_amp_frame(**kwargs)
    assert frame.shape == (batch, G1_AMP_FRAME_DIM)
    assert G1_AMP_FRAME_DIM == 233

    shift = torch.tensor([11.0, -7.0, 2.0])
    shifted = build_g1_amp_frame(
        **{
            **kwargs,
            "root_pos": root_pos + shift,
            "key_body_pos": key_pos + shift,
        }
    )
    # The root position is intentionally global within the scene frame; all
    # other AMP features, including root-relative key positions, are invariant.
    assert torch.allclose(shifted[:, :3], frame[:, :3] + shift)
    assert torch.allclose(shifted[:, 3:], frame[:, 3:], atol=1.0e-6)


def test_demo_window_indices_are_strictly_contiguous_and_never_wrap() -> None:
    windows = sample_contiguous_window_indices(256, 16, 41, device="cpu")
    assert windows.shape == (256, 16)
    assert bool((windows[:, 1:] - windows[:, :-1] == 1).all())
    assert int(windows.min()) >= 0
    assert int(windows.max()) < 41

    with pytest.raises(ValueError, match="window_size"):
        sample_contiguous_window_indices(1, 42, 41, device="cpu")


def test_reset_history_clamps_only_the_left_boundary() -> None:
    phases = torch.tensor([0, 2, 8])
    indices = history_indices(phases, 4, 9, clamp_start=True)
    assert torch.equal(
        indices,
        torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 1, 2],
                [5, 6, 7, 8],
            ]
        ),
    )
    with pytest.raises(ValueError, match="before motion frame 0"):
        history_indices(torch.tensor([2]), 4, 9, clamp_start=False)


def _fake_motion(num_frames: int = 24) -> MimicMotionReference:
    motion = object.__new__(MimicMotionReference)
    motion.device = torch.device("cpu")
    motion.num_frames = num_frames
    motion.root_body_id = 0
    motion.amp_key_body_ids = torch.tensor([1, 2, 3, 4, 5])
    motion.joint_pos = torch.arange(num_frames, dtype=torch.float32)[:, None].repeat(1, 29)
    motion.joint_vel = torch.ones(num_frames, 29)
    frame = torch.arange(num_frames, dtype=torch.float32)
    root_trajectory = torch.stack((frame, 3.0 - 2.0 * frame, 0.70 + 0.01 * frame), dim=-1)
    motion.body_pos_full_w = root_trajectory[:, None, :].repeat(1, 6, 1)
    motion.body_pos_full_w[:, 1:, 0] += 1.0
    motion.body_quat_full_w = _identity_quat(num_frames, 6)
    motion.body_lin_vel_full_w = torch.zeros(num_frames, 6, 3)
    motion.body_ang_vel_full_w = torch.zeros(num_frames, 6, 3)
    return motion


def _write_minimal_holosoma_motion(path: Path) -> None:
    num_frames = 2
    body_pos = np.zeros((num_frames, 2, 3), dtype=np.float32)
    body_pos[:, 1] = np.array([[1.0, 2.0, 0.9], [1.5, 2.2, 0.95]], dtype=np.float32)
    body_quat = np.zeros((num_frames, 2, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    body_lin = np.zeros((num_frames, 2, 3), dtype=np.float32)
    body_lin[:, 1] = np.array([0.5, -0.2, 0.1], dtype=np.float32)
    body_ang = np.zeros((num_frames, 2, 3), dtype=np.float32)
    body_ang[:, 1] = np.array([0.0, 0.0, 2.0], dtype=np.float32)
    np.savez(
        path,
        joint_names=np.array(["joint"], dtype=object),
        body_names=np.array(["pelvis", "torso_link"], dtype=object),
        joint_pos=np.zeros((num_frames, 1), dtype=np.float32),
        joint_vel=np.zeros((num_frames, 1), dtype=np.float32),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin,
        body_ang_vel_w=body_ang,
    )


def test_holosoma_loader_reconstructs_fixed_head_body(tmp_path: Path) -> None:
    path = tmp_path / "motion.npz"
    _write_minimal_holosoma_motion(path)
    motion = MimicMotionReference(
        path,
        track_body_ids=torch.tensor([0, 1, 2]),
        anchor_body_id=1,
        device=torch.device("cpu"),
        robot_body_names=["pelvis", "torso_link", "head_link"],
        action_joint_names=["joint"],
        root_body_name="pelvis",
        amp_key_body_names=("head_link",),
    )

    offset = torch.tensor([0.0039635, 0.0, -0.044])
    torso_pos = motion.body_pos_full_w[:, 1]
    torch.testing.assert_close(motion.body_pos_full_w[:, 2], torso_pos + offset)
    torch.testing.assert_close(motion.body_quat_full_w[:, 2], motion.body_quat_full_w[:, 1])
    torch.testing.assert_close(motion.body_ang_vel_full_w[:, 2], motion.body_ang_vel_full_w[:, 1])
    expected_lin = motion.body_lin_vel_full_w[:, 1] + torch.linalg.cross(
        motion.body_ang_vel_full_w[:, 1], offset.expand(2, -1)
    )
    torch.testing.assert_close(motion.body_lin_vel_full_w[:, 2], expected_lin)


def test_holosoma_loader_rejects_unrecoverable_amp_body(tmp_path: Path) -> None:
    path = tmp_path / "motion.npz"
    _write_minimal_holosoma_motion(path)
    with pytest.raises(ValueError, match="missing from the motion"):
        MimicMotionReference(
            path,
            track_body_ids=torch.tensor([0]),
            anchor_body_id=1,
            device=torch.device("cpu"),
            robot_body_names=["pelvis", "torso_link", "unrecoverable"],
            action_joint_names=["joint"],
            root_body_name="pelvis",
            amp_key_body_names=("unrecoverable",),
        )


def test_motion_demo_history_and_samples_clamp_only_the_left_boundary() -> None:
    motion = _fake_motion()
    reset_history = motion.get_amp_demo_history(torch.tensor([0, 3]), 4, flatten=False)
    assert reset_history.shape == (2, 4, G1_AMP_FRAME_DIM)
    assert torch.equal(reset_history[0, :, 0], torch.zeros(4))
    assert torch.equal(reset_history[1, :, 0], torch.arange(4, dtype=torch.float32))

    sampled = motion.sample_amp_demo_windows(
        32,
        16,
        flatten=False,
        generator=torch.Generator().manual_seed(9),
    )
    assert sampled.shape == (32, 16, G1_AMP_FRAME_DIM)
    # MimicKit samples newest frames over the full timeline. Negative history
    # clips to frame zero, producing only a zero prefix followed by contiguous
    # +1 steps; it never wraps the motion end into the beginning.
    delta = sampled[:, 1:, 0] - sampled[:, :-1, 0]
    assert bool(((delta == 0) | (delta == 1)).all())
    assert bool((delta[:, 1:] >= delta[:, :-1]).all())
    assert bool((delta == 0).any())
    assert bool((delta == 1).any())


def test_flattened_demo_samples_use_final_frame_root_xy_canonicalization() -> None:
    motion = _fake_motion()
    raw = motion.sample_amp_demo_windows(
        8,
        6,
        flatten=False,
        generator=torch.Generator().manual_seed(17),
    )
    flattened = motion.sample_amp_demo_windows(
        8,
        6,
        flatten=True,
        generator=torch.Generator().manual_seed(17),
    )
    canonical = flattened.reshape(8, 6, G1_AMP_FRAME_DIM)

    torch.testing.assert_close(canonical, canonicalize_amp_window(raw))
    torch.testing.assert_close(canonical[:, -1, :2], torch.zeros(8, 2))
    torch.testing.assert_close(canonical[:, :, 2], raw[:, :, 2])

    translated_motion = _fake_motion()
    translated_motion.body_pos_full_w[:, :, :2] += torch.tensor([31.0, -47.0])
    translated = translated_motion.sample_amp_demo_windows(
        8,
        6,
        flatten=True,
        generator=torch.Generator().manual_seed(17),
    )
    torch.testing.assert_close(translated, flattened)
