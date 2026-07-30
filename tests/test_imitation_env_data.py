from __future__ import annotations

import ast
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
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


def test_imitation_frame_is_239d_and_translation_invariant_except_root_position() -> None:
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
    assert G1_IMITATION_FRAME_DIM == 239

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
    motion.root_link_vel_w = torch.stack(
        (
            frame,
            frame + 1.0,
            frame + 2.0,
            frame + 3.0,
            frame + 4.0,
            frame + 5.0,
        ),
        dim=-1,
    )
    root_trajectory = torch.stack((frame, 3.0 - 2.0 * frame, 0.70 + 0.01 * frame), dim=-1)
    motion.body_pos_full_w = root_trajectory[:, None, :].repeat(1, 6, 1)
    motion.body_pos_full_w[:, 1:, 0] += 1.0
    motion.body_quat_full_w = _identity_quat(num_frames, 6)
    motion.body_lin_vel_full_w = torch.zeros(num_frames, 6, 3)
    motion.body_ang_vel_full_w = torch.zeros(num_frames, 6, 3)
    motion._fk_model = None
    motion._amp_expert_integer_frame_cache = None
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


def test_motion_reference_interpolates_pose_and_holds_left_velocity() -> None:
    motion = _fake_motion(num_frames=5)
    frame = motion.get_frame(torch.tensor([1.5, 3.25]))

    torch.testing.assert_close(frame["joint_pos"][0], torch.full((29,), 1.5))
    torch.testing.assert_close(frame["joint_pos"][1], torch.full((29,), 3.25))
    torch.testing.assert_close(frame["root_pos_w"][0, 0], torch.tensor(1.5))
    torch.testing.assert_close(frame["root_pos_w"][1, 0], torch.tensor(3.25))
    torch.testing.assert_close(
        frame["root_lin_vel_w"],
        torch.tensor([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]),
    )
    torch.testing.assert_close(
        frame["root_ang_vel_w"],
        torch.tensor([[4.0, 5.0, 6.0], [6.0, 7.0, 8.0]]),
    )
    torch.testing.assert_close(
        torch.linalg.norm(frame["root_quat_w"], dim=-1),
        torch.ones(2),
    )


def test_motion_reference_uses_shortest_path_slerp_at_reset() -> None:
    motion = _fake_motion(num_frames=2)
    motion.body_quat_full_w[1, :, 0] = 0.0
    motion.body_quat_full_w[1, :, 3] = 1.0

    frame = motion.get_frame(torch.tensor([0.25]))
    expected = torch.tensor([0.9238795, 0.0, 0.0, 0.3826834])

    torch.testing.assert_close(frame["root_quat_w"][0], expected)


def test_imitation_reference_interpolates_exact_fractional_phases() -> None:
    motion = _fake_motion(num_frames=5)
    motion._fk_model = _StateDependentFK()
    frame = motion.get_imitation_frame_at_times(torch.tensor([1.5, 3.25]))

    assert frame.shape == (2, G1_IMITATION_FRAME_DIM)
    torch.testing.assert_close(frame[:, 0], torch.tensor([1.5, 3.25]))
    torch.testing.assert_close(frame[:, 1], torch.tensor([0.0, -3.5]))
    torch.testing.assert_close(frame[:, 2], torch.tensor([0.715, 0.7325]))
    # Joint positions are converted to the common rot6d representation, so
    # interpolation must change the final frame rather than floor to 1 and 3.
    floored = motion.get_imitation_frame_at_times(torch.tensor([1.0, 3.0]))
    assert not torch.allclose(frame, floored)


def test_amp_expert_frame_uses_fk_at_the_same_dataset_state_and_time() -> None:
    motion = _fake_motion(num_frames=6)
    motion._fk_model = _StateDependentFK()
    times = torch.tensor([[0.0, 1.5], [3.0, 4.25]])

    actual = motion.get_amp_expert_frame_at_times(times)
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
        root_lin_vel=motion._sample_left_frame(
            motion.root_link_vel_w,
            flat_times,
        )[
            :, :3
        ],
        root_ang_vel=motion._sample_left_frame(
            motion.root_link_vel_w,
            flat_times,
        )[
            :, 3:
        ],
        joint_vel=motion._sample_left_frame(motion.joint_vel, flat_times),
    ).reshape(2, 2, G1_IMITATION_FRAME_DIM)

    assert actual.shape == (2, 2, G1_IMITATION_FRAME_DIM)
    torch.testing.assert_close(actual, expected)

    evaluator_frame = motion.get_imitation_frame_at_times(flat_times).reshape_as(actual)
    torch.testing.assert_close(actual, evaluator_frame)


def test_amp_history_and_endpoint_windows_share_fk_and_left_boundary() -> None:
    motion = _fake_motion(num_frames=9)
    motion._fk_model = _StateDependentFK()
    endpoints = torch.tensor([0.0, 3.0])

    history = motion.get_amp_demo_history(endpoints, 4)
    assert history.shape == (2, 4, G1_IMITATION_FRAME_DIM)
    assert torch.equal(history[0, :, 0], torch.zeros(4))
    assert torch.equal(history[1, :, 0], torch.arange(4, dtype=torch.float32))
    torch.testing.assert_close(
        history[:, -1], motion.get_amp_expert_frame_at_times(endpoints)
    )

    specified = motion.get_amp_demo_windows_at_end_indices(
        endpoints.long(), 4
    )
    torch.testing.assert_close(specified, history)
    with pytest.raises(ValueError, match="integer phases"):
        motion.get_amp_demo_history(torch.tensor([1.5]), 4)


def test_holosoma_loader_converts_mujoco_root_angular_velocity_to_world() -> None:
    motion = object.__new__(MimicMotionReference)
    root_linear_world = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=np.float32,
    )
    root_angular_local = np.array(
        [[1.0, 2.0, 3.0], [1.0, 0.0, 0.0], [1.0, 2.0, 3.0]],
        dtype=np.float32,
    )
    root_quat_wxyz = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [2.0**-0.5, 0.0, 0.0, 2.0**-0.5],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    root_angular_world = np.array(
        [[1.0, 2.0, 3.0], [0.0, 1.0, 0.0], [-1.0, 2.0, -3.0]],
        dtype=np.float32,
    )
    root_qvel = np.concatenate(
        (root_linear_world, root_angular_local),
        axis=-1,
    )
    data = {
        "joint_names": np.array(["joint"]),
        "body_names": np.array(["pelvis"]),
        "joint_pos": np.zeros((3, 8), dtype=np.float32),
        "joint_vel": np.concatenate(
            (root_qvel, np.ones((3, 1), dtype=np.float32)), axis=1
        ),
        "body_pos_w": np.zeros((3, 1, 3), dtype=np.float32),
        "body_quat_w": root_quat_wxyz[:, None, :],
        "body_lin_vel_w": np.full((3, 1, 3), 100.0, dtype=np.float32),
        "body_ang_vel_w": root_angular_world[:, None, :],
    }

    frame, _ = motion._load_holosoma(
        data,
        ["pelvis"],
        ["joint"],
        "pelvis",
    )

    np.testing.assert_array_equal(frame[2][:, :3], root_linear_world)
    np.testing.assert_allclose(
        frame[2][:, 3:],
        root_angular_world,
        atol=1.0e-6,
    )
