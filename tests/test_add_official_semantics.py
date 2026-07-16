import math
from pathlib import Path
from types import SimpleNamespace

import torch

from engine.config import ADDConfig, load_config
from envs.motion import (
    ADD_TARGET_OBS_STEPS,
    MimicMotionReference,
    add_target_phase_offsets,
    mimickit_frame_delta,
    mimickit_full_motion_steps,
    sample_mimickit_phase,
)
from method.amp import AMP
from method.base import classify_mimickit_done_terms


ROOT = Path(__file__).resolve().parents[1]


def test_add_config_uses_common_platform_and_official_recipe() -> None:
    cfg = load_config(ROOT / "configs" / "add_largebox.yaml")
    assert isinstance(cfg.parameters, ADDConfig)
    assert cfg.environment.platform_profile == "g1_largebox_50hz"
    assert cfg.environment.termination_mode == "amp"
    assert cfg.environment.terminate_on_motion_end is True
    assert cfg.environment.motion_reference_mode == "mimickit_add"
    assert cfg.environment.root_velocity_mode == "link"
    assert cfg.environment.reset_phase_sampling == "continuous_uniform"
    assert cfg.environment.sim_dt == 0.02
    assert cfg.environment.decimation == 4
    assert cfg.environment.max_episode_steps == 500
    assert cfg.parameters.discount_gamma == 0.99
    assert cfg.parameters.gae_lambda == 0.95
    assert cfg.parameters.normalizer_samples == 100_000_000


def test_mimickit_timebase_uses_motion_fps_and_continuous_resets() -> None:
    assert mimickit_frame_delta(50.0, 1.0 / 30.0) == 50.0 / 30.0
    assert mimickit_full_motion_steps(0, 324, 50.0, 1.0 / 30.0) == 195
    torch.manual_seed(7)
    phases = sample_mimickit_phase(1024, 0, 324, device="cpu")
    assert phases.dtype == torch.float32
    assert bool(((phases > 0.0) & (phases < 324.0)).all())
    assert bool((phases != phases.round()).any())
    assert float(phases.max()) > 323.0


def test_add_preview_uses_official_control_step_offsets() -> None:
    assert ADD_TARGET_OBS_STEPS == (1, 2, 3)
    assert add_target_phase_offsets(1.0) == (1.0, 2.0, 3.0)
    assert all(
        math.isclose(actual, expected)
        for actual, expected in zip(add_target_phase_offsets(0.6), (0.6, 1.2, 1.8), strict=True)
    )


def test_mimickit_forward_velocities_are_separate_and_left_held() -> None:
    motion = MimicMotionReference.__new__(MimicMotionReference)
    motion.device = torch.device("cpu")
    motion.motion_reference_mode = "mimickit_add"
    motion.fps = 50.0
    motion.num_frames = 4
    motion.root_body_id = 0
    frame = torch.arange(4, dtype=torch.float32)
    motion.joint_pos = frame[:, None].repeat(1, 2)
    motion.joint_vel = torch.ones(4, 2)
    motion.body_pos_full_w = torch.zeros(4, 1, 3)
    motion.body_pos_full_w[:, 0, 0] = frame
    motion.body_quat_full_w = torch.zeros(4, 1, 4)
    motion.body_quat_full_w[..., 0] = 1.0
    motion.body_lin_vel_full_w = torch.zeros(4, 1, 3)
    motion.body_ang_vel_full_w = torch.zeros(4, 1, 3)

    motion._apply_mimickit_velocity_semantics()
    torch.testing.assert_close(motion.body_lin_vel_full_w, torch.zeros_like(motion.body_lin_vel_full_w))
    torch.testing.assert_close(motion.joint_vel, torch.ones_like(motion.joint_vel))
    torch.testing.assert_close(motion.add_root_link_lin_vel[:, 0], torch.full((4,), 50.0))
    expected_joint_vel = torch.full((4, 2), 50.0)
    torch.testing.assert_close(motion.add_joint_vel, expected_joint_vel)
    sampled_joint = motion._joint_velocity(torch.tensor([0.9, 1.9, 3.0]))
    torch.testing.assert_close(sampled_joint, expected_joint_vel[[0, 1, 3]])


def test_normalizer_gate_uses_transition_clock_and_allows_crossing_update() -> None:
    algo = AMP.__new__(AMP)
    algo.cfg = SimpleNamespace(normalizer_samples=100)
    algo.normalizer_sample_count = 99
    assert algo._need_normalizer_update()
    algo._advance_normalizer_sample_count(32)
    assert algo.normalizer_sample_count == 131
    assert not algo._need_normalizer_update()


def test_advantage_normalization_matches_mimickit_unbiased_std() -> None:
    algo = AMP.__new__(AMP)
    algo.cfg = SimpleNamespace(norm_adv_clip=4.0)
    advantages = torch.tensor([1.0, 2.0, 4.0, 8.0])
    expected = (advantages - advantages.mean()) / advantages.std(unbiased=True)
    torch.testing.assert_close(algo._normalize_advantages(advantages), expected)


def test_done_precedence_is_failure_then_success_then_timeout() -> None:
    done = torch.ones(4, dtype=torch.bool)
    terms = {
        "time_out": torch.tensor([True, True, True, False]),
        "motion_complete": torch.tensor([False, True, True, True]),
        "anchor_pos_bad": torch.tensor([False, False, True, False]),
        "anchor_ori_bad": torch.zeros(4, dtype=torch.bool),
        "ee_body_bad": torch.tensor([False, False, False, True]),
    }
    timeout, success, failure = classify_mimickit_done_terms(done, terms)
    torch.testing.assert_close(timeout, torch.tensor([True, False, False, False]))
    torch.testing.assert_close(success, torch.tensor([False, True, False, False]))
    torch.testing.assert_close(failure, torch.tensor([False, False, True, True]))
