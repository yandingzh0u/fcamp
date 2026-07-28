from __future__ import annotations

import ast
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import torch

from envs.imitation_data import (
    G1_IMITATION_FRAME_DIM,
    G1_IMITATION_JOINT_AXES,
    G1_IMITATION_NUM_JOINTS,
    build_g1_imitation_frame,
    history_indices,
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


def test_reset_history_clamps_only_the_left_boundary() -> None:
    phases = torch.tensor([0, 2, 8])
    indices = history_indices(phases, 4, 9)
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
    floored = motion.get_imitation_frame_at_times(torch.tensor([1.0, 3.0]))
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


def test_fcamp_history_and_endpoint_windows_share_fk_and_left_boundary() -> None:
    motion = _fake_motion(num_frames=9)
    motion._fk_model = _StateDependentFK()
    endpoints = torch.tensor([0.0, 3.0])

    history = motion.get_fcamp_demo_history(endpoints, 4)
    assert history.shape == (2, 4, G1_IMITATION_FRAME_DIM)
    assert torch.equal(history[0, :, 0], torch.zeros(4))
    assert torch.equal(history[1, :, 0], torch.arange(4, dtype=torch.float32))
    torch.testing.assert_close(
        history[:, -1], motion.get_fcamp_expert_frame_at_times(endpoints)
    )

    specified = motion.get_fcamp_demo_windows_at_end_indices(
        endpoints.long(), 4
    )
    torch.testing.assert_close(specified, history)
    with pytest.raises(ValueError, match="integer phases"):
        motion.get_fcamp_demo_history(torch.tensor([1.5]), 4)
