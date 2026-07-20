from __future__ import annotations

import math
import ast
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import torch

from components.imitation.motion_features import canonicalize_imitation_window
from envs.imitation_data import (
    G1_IMITATION_FRAME_DIM,
    G1_IMITATION_JOINT_AXES,
    G1_IMITATION_NUM_JOINTS,
    build_g1_imitation_frame,
    history_indices,
    quat_wxyz_to_tan_norm,
    sample_contiguous_window_indices,
)
from envs.motion import MimicMotionReference


def _identity_quat(*prefix: int) -> torch.Tensor:
    quat = torch.zeros(*prefix, 4)
    quat[..., 0] = 1.0
    return quat


def test_imitation_joint_axes_match_the_runtime_urdf_action_order() -> None:
    source = ast.parse((Path(__file__).parents[1] / "envs/robots/g1.py").read_text())
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
    assert [urdf_axes[name] for name in joint_names] == list(G1_IMITATION_JOINT_AXES)


def test_wxyz_rotation_encoding_matches_mimickit_tangent_normal() -> None:
    identity = quat_wxyz_to_tan_norm(torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.allclose(identity, torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))

    # +90 degrees about z in explicit wxyz order rotates x onto +y while z stays z.
    half = math.pi / 4.0
    q_z90 = torch.tensor([[math.cos(half), 0.0, 0.0, math.sin(half)]])
    encoded = quat_wxyz_to_tan_norm(q_z90)
    expected = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    assert torch.allclose(encoded, expected, atol=1.0e-6)


def test_imitation_frame_is_233d_and_translation_invariant_except_root_position() -> None:
    batch = 3
    root_pos = torch.randn(batch, 3)
    key_pos = root_pos[:, None, :] + torch.randn(batch, 5, 3)
    kwargs = {
        "root_pos": root_pos,
        "root_quat_wxyz": _identity_quat(batch),
        "joint_pos": torch.randn(batch, G1_IMITATION_NUM_JOINTS),
        "key_body_pos": key_pos,
        "root_lin_vel": torch.randn(batch, 3),
        "root_ang_vel": torch.randn(batch, 3),
        "joint_vel": torch.randn(batch, G1_IMITATION_NUM_JOINTS),
    }
    frame = build_g1_imitation_frame(**kwargs)
    assert frame.shape == (batch, G1_IMITATION_FRAME_DIM)
    assert G1_IMITATION_FRAME_DIM == 233

    shift = torch.tensor([11.0, -7.0, 2.0])
    shifted = build_g1_imitation_frame(
        **{
            **kwargs,
            "root_pos": root_pos + shift,
            "key_body_pos": key_pos + shift,
        }
    )
    # The root position is intentionally global within the scene frame; all
    # other imitation features, including root-relative key positions, are invariant.
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
    motion.anchor_body_id = 0
    motion.track_body_ids = torch.arange(6)
    motion.imitation_key_body_ids = torch.tensor([1, 2, 3, 4, 5])
    motion.joint_pos = torch.arange(num_frames, dtype=torch.float32)[:, None].repeat(1, 29)
    motion.joint_vel = torch.ones(num_frames, 29)
    frame = torch.arange(num_frames, dtype=torch.float32)
    root_trajectory = torch.stack((frame, 3.0 - 2.0 * frame, 0.70 + 0.01 * frame), dim=-1)
    motion.body_pos_full_w = root_trajectory[:, None, :].repeat(1, 6, 1)
    motion.body_pos_full_w[:, 1:, 0] += 1.0
    motion.body_quat_full_w = _identity_quat(num_frames, 6)
    motion.body_lin_vel_full_w = torch.zeros(num_frames, 6, 3)
    motion.body_ang_vel_full_w = torch.zeros(num_frames, 6, 3)
    motion._fk_model = None
    motion._fcamp_expert_integer_frame_cache = None
    return motion


class _StateDependentFK:
    """Tiny FK stand-in whose body origins differ from the motion file."""

    def body_pos(
        self,
        *,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        joint_pos: torch.Tensor,
    ) -> torch.Tensor:
        del root_quat
        offsets = root_pos.new_zeros((root_pos.shape[0], 6, 3))
        offsets[:, :, 0] = torch.arange(
            6, device=root_pos.device, dtype=root_pos.dtype
        )[None, :] * 0.25
        offsets[:, 1:, 1] = 0.05 * joint_pos[:, :5]
        return root_pos[:, None, :] + offsets


def test_motion_reference_interpolates_fractional_frames() -> None:
    motion = _fake_motion(num_frames=5)
    frame = motion.get_frame(torch.tensor([1.5, 3.25]))

    torch.testing.assert_close(frame["joint_pos"][0], torch.full((29,), 1.5))
    torch.testing.assert_close(frame["joint_pos"][1], torch.full((29,), 3.25))
    torch.testing.assert_close(frame["root_pos_w"][0, 0], torch.tensor(1.5))
    torch.testing.assert_close(frame["root_pos_w"][1, 0], torch.tensor(3.25))
    torch.testing.assert_close(
        torch.linalg.norm(frame["root_quat_w"], dim=-1),
        torch.ones(2),
    )


def test_imitation_reference_interpolates_exact_fractional_phases() -> None:
    motion = _fake_motion(num_frames=5)
    frame = motion.get_imitation_frame_at_times(torch.tensor([1.5, 3.25]))

    assert frame.shape == (2, G1_IMITATION_FRAME_DIM)
    torch.testing.assert_close(frame[:, 0], torch.tensor([1.5, 3.25]))
    torch.testing.assert_close(frame[:, 1], torch.tensor([0.0, -3.5]))
    torch.testing.assert_close(frame[:, 2], torch.tensor([0.715, 0.7325]))
    # Joint positions are converted to the common rot6d representation, so
    # interpolation must change the final frame rather than floor to 1 and 3.
    floored = motion.get_imitation_frame(torch.tensor([1, 3]))
    assert not torch.allclose(frame, floored)


def test_fcamp_expert_frame_uses_fk_at_the_same_dataset_state_and_time() -> None:
    motion = _fake_motion(num_frames=6)
    motion._fk_model = _StateDependentFK()
    times = torch.tensor([[0.0, 1.5], [3.0, 4.25]])

    actual = motion.get_fcamp_expert_frame_at_times(times)
    flat_times = times.reshape(-1)
    root_pos = motion._interpolate(
        motion.body_pos_full_w[:, motion.root_body_id], flat_times
    )
    root_quat = motion._interpolate_quat_shortest(
        motion.body_quat_full_w[:, motion.root_body_id], flat_times
    )
    joint_pos = motion._interpolate(motion.joint_pos, flat_times)
    fk_body_pos = motion._fk_model.body_pos(
        root_pos=root_pos,
        root_quat=root_quat,
        joint_pos=joint_pos,
    )
    expected = build_g1_imitation_frame(
        root_pos=root_pos,
        root_quat_wxyz=root_quat,
        joint_pos=joint_pos,
        key_body_pos=fk_body_pos.index_select(1, motion.imitation_key_body_ids),
        root_lin_vel=motion._interpolate(
            motion.body_lin_vel_full_w[:, motion.root_body_id], flat_times
        ),
        root_ang_vel=motion._interpolate(
            motion.body_ang_vel_full_w[:, motion.root_body_id], flat_times
        ),
        joint_vel=motion._interpolate(motion.joint_vel, flat_times),
    ).reshape(2, 2, G1_IMITATION_FRAME_DIM)

    assert actual.shape == (2, 2, G1_IMITATION_FRAME_DIM)
    torch.testing.assert_close(actual, expected)

    # The method-independent evaluator intentionally remains dataset-body based.
    dataset_frame = motion.get_imitation_frame_at_times(flat_times).reshape_as(actual)
    key_start = 3 + 6 + 6 * G1_IMITATION_NUM_JOINTS
    key_end = key_start + 5 * 3
    torch.testing.assert_close(actual[..., :key_start], dataset_frame[..., :key_start])
    torch.testing.assert_close(actual[..., key_end:], dataset_frame[..., key_end:])
    assert not torch.allclose(
        actual[..., key_start:key_end], dataset_frame[..., key_start:key_end]
    )


def test_fcamp_expert_frame_requires_explicit_runtime_fk_model() -> None:
    motion = _fake_motion(num_frames=4)
    with pytest.raises(RuntimeError, match="kinematic_urdf_file"):
        motion.get_fcamp_expert_frame_at_times(torch.tensor([0.0]))


def test_evaluator_demo_frame_is_independent_of_add_motion_semantics() -> None:
    motion = _fake_motion(num_frames=5)
    motion.add_joint_vel = torch.full_like(motion.joint_vel, 123.0)
    motion.add_root_link_lin_vel = torch.full((5, 3), 456.0)
    motion.add_root_link_ang_vel = torch.full((5, 3), 789.0)
    # Opposite quaternion signs encode the same physical rotation. Evaluator
    # interpolation must follow the common shortest path in either mode.
    motion.body_quat_full_w[2] *= -1.0
    phases = torch.tensor([1.5, 2.25, 3.75])

    motion.motion_reference_mode = "frame"
    frame_mode = motion.get_imitation_frame_at_times(phases)
    motion.motion_reference_mode = "mimickit_add"
    add_mode = motion.get_imitation_frame_at_times(phases)

    torch.testing.assert_close(frame_mode, add_mode)
    assert bool(torch.isfinite(add_mode).all())
    # Raw dataset velocities are zero/root and one/joint in _fake_motion; the
    # ADD forward-difference buffers above must never leak into evaluation.
    torch.testing.assert_close(add_mode[:, -35:-32], torch.zeros(3, 3))
    torch.testing.assert_close(add_mode[:, -29:], torch.ones(3, 29))


def _write_minimal_holosoma_motion(path: Path) -> None:
    num_frames = 2
    body_pos = np.zeros((num_frames, 2, 3), dtype=np.float32)
    body_pos[:, 1] = np.array([[1.0, 2.0, 0.9], [1.5, 2.2, 0.95]], dtype=np.float32)
    body_quat = np.zeros((num_frames, 2, 4), dtype=np.float32)
    # Holosoma motion .npz stores raw rigid-body quaternions as wxyz.  Holosoma
    # converts to xyzw only at its simulator boundary.
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
        imitation_key_body_names=("head_link",),
    )

    offset = torch.tensor([0.0039635, 0.0, -0.044])
    torch.testing.assert_close(
        motion.body_quat_full_w[:, :2],
        torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(2, 2, 4),
    )
    torso_pos = motion.body_pos_full_w[:, 1]
    torch.testing.assert_close(motion.body_pos_full_w[:, 2], torso_pos + offset)
    torch.testing.assert_close(motion.body_quat_full_w[:, 2], motion.body_quat_full_w[:, 1])
    torch.testing.assert_close(motion.body_ang_vel_full_w[:, 2], motion.body_ang_vel_full_w[:, 1])
    expected_lin = motion.body_lin_vel_full_w[:, 1] + torch.linalg.cross(
        motion.body_ang_vel_full_w[:, 1], offset.expand(2, -1)
    )
    torch.testing.assert_close(motion.body_lin_vel_full_w[:, 2], expected_lin)


def test_holosoma_loader_rejects_unrecoverable_imitation_body(tmp_path: Path) -> None:
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
            imitation_key_body_names=("unrecoverable",),
        )


def test_motion_demo_history_and_samples_clamp_only_the_left_boundary() -> None:
    motion = _fake_motion()
    reset_history = motion.get_imitation_demo_history(torch.tensor([0, 3]), 4, flatten=False)
    assert reset_history.shape == (2, 4, G1_IMITATION_FRAME_DIM)
    assert torch.equal(reset_history[0, :, 0], torch.zeros(4))
    assert torch.equal(reset_history[1, :, 0], torch.arange(4, dtype=torch.float32))

    sampled = motion.sample_imitation_demo_windows(
        32,
        16,
        flatten=False,
        generator=torch.Generator().manual_seed(9),
    )
    assert sampled.shape == (32, 16, G1_IMITATION_FRAME_DIM)
    # MimicKit samples newest frames over the full timeline. Negative history
    # clips to frame zero, producing only a zero prefix followed by contiguous
    # +1 steps; it never wraps the motion end into the beginning.
    delta = sampled[:, 1:, 0] - sampled[:, :-1, 0]
    assert bool(((delta == 0) | (delta == 1)).all())
    assert bool((delta[:, 1:] >= delta[:, :-1]).all())
    assert bool((delta == 0).any())
    assert bool((delta == 1).any())


def test_fcamp_history_and_endpoint_windows_share_fk_and_left_boundary() -> None:
    motion = _fake_motion(num_frames=9)
    motion._fk_model = _StateDependentFK()
    endpoints = torch.tensor([0.0, 3.0])

    history = motion.get_fcamp_demo_history(endpoints, 4, flatten=False)
    assert history.shape == (2, 4, G1_IMITATION_FRAME_DIM)
    assert torch.equal(history[0, :, 0], torch.zeros(4))
    assert torch.equal(history[1, :, 0], torch.arange(4, dtype=torch.float32))
    torch.testing.assert_close(
        history[:, -1], motion.get_fcamp_expert_frame_at_times(endpoints)
    )

    specified = motion.get_fcamp_demo_windows_at_end_indices(
        endpoints.long(), 4, flatten=False
    )
    torch.testing.assert_close(specified, history)
    flattened = motion.get_fcamp_demo_windows_at_end_indices(
        endpoints.long(), 4, flatten=True
    )
    torch.testing.assert_close(
        flattened.reshape(2, 4, G1_IMITATION_FRAME_DIM),
        canonicalize_imitation_window(history),
    )
    with pytest.raises(ValueError, match="integer phases"):
        motion.get_fcamp_demo_history(torch.tensor([1.5]), 4)


def test_flattened_demo_samples_use_final_frame_root_xy_canonicalization() -> None:
    motion = _fake_motion()
    raw = motion.sample_imitation_demo_windows(
        8,
        6,
        flatten=False,
        generator=torch.Generator().manual_seed(17),
    )
    flattened = motion.sample_imitation_demo_windows(
        8,
        6,
        flatten=True,
        generator=torch.Generator().manual_seed(17),
    )
    canonical = flattened.reshape(8, 6, G1_IMITATION_FRAME_DIM)

    torch.testing.assert_close(canonical, canonicalize_imitation_window(raw))
    torch.testing.assert_close(canonical[:, -1, :2], torch.zeros(8, 2))
    torch.testing.assert_close(canonical[:, :, 2], raw[:, :, 2])

    translated_motion = _fake_motion()
    translated_motion.body_pos_full_w[:, :, :2] += torch.tensor([31.0, -47.0])
    translated = translated_motion.sample_imitation_demo_windows(
        8,
        6,
        flatten=True,
        generator=torch.Generator().manual_seed(17),
    )
    torch.testing.assert_close(translated, flattened)
