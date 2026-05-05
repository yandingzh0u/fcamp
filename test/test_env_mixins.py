from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("isaaclab")

from env.observation import MimicObservationMixin
from env.reward import MimicRewardMixin
from env.step import MimicStepMixin
from env.terminal import MimicTerminationMixin


def _identity_quat(num_envs: int, bodies: int | None = None) -> torch.Tensor:
    shape = (num_envs, 4) if bodies is None else (num_envs, bodies, 4)
    quat = torch.zeros(shape)
    quat[..., 0] = 1.0
    return quat


class _ObservationHarness(MimicObservationMixin):
    def __init__(self):
        self.num_envs = 2
        self.phase_steps = torch.tensor([0, 1])
        self.track_body_names = [f"body_{i}" for i in range(14)]
        self.track_body_ids = torch.arange(14)
        self.anchor_body_id = 7
        self.action_joint_ids = torch.arange(29)
        self.default_action_joint_pos = torch.zeros(2, 29)
        self.default_action_joint_vel = torch.zeros(2, 29)
        self.last_action = torch.full((2, 29), 0.25)
        self.scene = SimpleNamespace(env_origins=torch.zeros(2, 3))
        self.robot = SimpleNamespace(data=SimpleNamespace())
        self.robot.data.body_pos_w = torch.zeros(2, 14, 3)
        self.robot.data.body_quat_w = _identity_quat(2, 14)
        self.robot.data.body_lin_vel_w = torch.zeros(2, 14, 3)
        self.robot.data.body_ang_vel_w = torch.zeros(2, 14, 3)
        self.robot.data.root_ang_vel_b = torch.zeros(2, 3)
        self.motion = self

    def get_frame(self, phase_steps: torch.Tensor) -> dict[str, torch.Tensor]:
        del phase_steps
        return {
            "joint_pos": torch.ones(2, 29),
            "joint_vel": torch.ones(2, 29) * 2.0,
            "body_pos_w": torch.zeros(2, 14, 3),
            "body_quat_w": _identity_quat(2, 14),
            "body_lin_vel_w": torch.zeros(2, 14, 3),
            "body_ang_vel_w": torch.zeros(2, 14, 3),
            "anchor_pos_w": torch.zeros(2, 3),
            "anchor_quat_w": _identity_quat(2),
            "root_pos_w": torch.zeros(2, 3),
            "root_quat_w": _identity_quat(2),
        }

    def get_action_joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.full((2, 29), 0.1), torch.full((2, 29), -0.2)

    def _add_uniform_noise(self, value: torch.Tensor, n_min: float, n_max: float) -> torch.Tensor:
        del n_min, n_max
        return value


def test_observation_builds_official_154_dim_contract() -> None:
    harness = _ObservationHarness()

    observation = harness.build_observation()

    assert observation.shape == (2, 154)
    assert torch.allclose(observation[:, :29], torch.ones(2, 29))
    assert torch.allclose(observation[:, 29:58], torch.ones(2, 29) * 2.0)
    assert torch.allclose(observation[:, -29:], torch.full((2, 29), 0.25))


def test_relative_reference_body_positions_follow_robot_anchor_frame() -> None:
    harness = _ObservationHarness()
    harness.track_body_names = ["body_0", "body_1"]
    reference = {
        "anchor_pos_w": torch.tensor([[1.0, 2.0, 0.5]]),
        "anchor_quat_w": _identity_quat(1),
        "body_pos_w": torch.tensor([[[2.0, 4.0, 0.7], [1.0, 3.0, 0.9]]]),
        "body_quat_w": _identity_quat(1, 2),
    }
    robot_anchor_pos = torch.tensor([[10.0, 20.0, 2.0]])
    robot_anchor_quat = _identity_quat(1)

    body_pos, body_quat = harness._compute_relative_reference_bodies(reference, robot_anchor_pos, robot_anchor_quat)

    assert torch.allclose(body_pos, torch.tensor([[[11.0, 22.0, 0.7], [10.0, 21.0, 0.9]]]))
    assert torch.allclose(body_quat, _identity_quat(1, 2))


class _RewardHarness(MimicRewardMixin):
    def __init__(self):
        self.dt = 0.02
        self.action_joint_ids = torch.arange(29)
        self.undesired_contact_body_ids = torch.tensor([0, 1])
        self.robot = SimpleNamespace(data=SimpleNamespace())
        self.robot.data.joint_acc = torch.zeros(2, 29)
        self.robot.data.applied_torque = torch.zeros(2, 29)
        self.robot.data.joint_pos = torch.zeros(2, 29)
        self.robot.data.soft_joint_pos_limits = torch.stack(
            [torch.full((2, 29), -1.0), torch.full((2, 29), 1.0)],
            dim=-1,
        )
        self.contact_sensor = SimpleNamespace(data=SimpleNamespace())
        self.contact_sensor.data.net_forces_w_history = torch.zeros(2, 3, 2, 3)

    def get_tracking_context(self) -> dict[str, torch.Tensor]:
        body_pos = torch.zeros(2, 14, 3)
        body_quat = _identity_quat(2, 14)
        body_vel = torch.zeros(2, 14, 3)
        return {
            "reference": {
                "anchor_pos_w": torch.zeros(2, 3),
                "anchor_quat_w": _identity_quat(2),
                "body_lin_vel_w": body_vel,
                "body_ang_vel_w": body_vel,
            },
            "robot_anchor_pos_w": torch.zeros(2, 3),
            "robot_anchor_quat_w": _identity_quat(2),
            "body_pos_relative_w": body_pos,
            "robot_body_pos_w": body_pos,
            "body_quat_relative_w": body_quat,
            "robot_body_quat_w": body_quat,
            "robot_body_lin_vel_w": body_vel,
            "robot_body_ang_vel_w": body_vel,
        }


def test_reward_exact_value_for_perfect_tracking_without_penalties() -> None:
    harness = _RewardHarness()
    action = torch.zeros(2, 29)

    reward, terms = harness.compute_reward(action, previous_action=action)

    assert torch.allclose(reward, torch.full((2,), 5.0 * harness.dt))
    assert torch.allclose(terms["anchor_pos_reward"], torch.ones(2))
    assert torch.allclose(terms["body_ang_vel_reward"], torch.ones(2))
    assert torch.allclose(terms["undesired_contacts"], torch.zeros(2))


class _TerminationHarness(MimicTerminationMixin):
    def __init__(self):
        self.ee_body_indices = [2, 3]
        self.termination_body_indices = [2, 3]
        self.episode_steps = torch.tensor([0, 0, 0, 0])
        self.task_cfg = SimpleNamespace(max_episode_steps=10)
        self.robot = SimpleNamespace(data=SimpleNamespace())
        self.robot.data.GRAVITY_VEC_W = torch.tensor([[0.0, 0.0, -1.0]]).repeat(4, 1)

    def get_tracking_context(self) -> dict[str, torch.Tensor]:
        body_pos = torch.zeros(4, 14, 3)
        robot_body_pos = body_pos.clone()
        robot_body_pos[3, 2, 2] = 0.4
        robot_anchor_pos = torch.zeros(4, 3)
        robot_anchor_pos[1, 2] = 0.3
        robot_anchor_quat = _identity_quat(4)
        robot_anchor_quat[2] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        return {
            "reference": {
                "anchor_pos_w": torch.zeros(4, 3),
                "anchor_quat_w": _identity_quat(4),
            },
            "robot_anchor_pos_w": robot_anchor_pos,
            "robot_anchor_quat_w": robot_anchor_quat,
            "body_pos_relative_w": body_pos,
            "robot_body_pos_w": robot_body_pos,
        }


def test_termination_flags_anchor_and_end_effector_failures() -> None:
    harness = _TerminationHarness()

    done, done_terms, debug_terms = harness.compute_termination()

    assert torch.equal(done, torch.tensor([False, True, True, True]))
    assert torch.equal(done_terms["anchor_pos_bad"], torch.tensor([False, True, False, False]))
    assert torch.equal(done_terms["anchor_ori_bad"], torch.tensor([False, False, True, False]))
    assert torch.equal(done_terms["ee_body_bad"], torch.tensor([False, False, False, True]))
    assert debug_terms["ee_z_error_by_body"].shape == (4, 2)
    assert debug_terms["termination_z_error_max"][3] == pytest.approx(0.4)


class _StepHarness(MimicStepMixin):
    def __init__(self):
        self.num_envs = 2
        self.action_dim = 3
        self.observation_dim = 5
        self.device = torch.device("cpu")
        self.calls = 0

    def step(self, action_offsets: torch.Tensor, auto_reset: bool = False):
        del action_offsets, auto_reset
        self.calls += 1
        obs = torch.full((2, 5), float(self.calls))
        reward = torch.tensor([float(self.calls), 10.0 + self.calls])
        done = torch.tensor([self.calls == 1, False])
        done_terms = {
            "time_out": torch.zeros(2, dtype=torch.bool),
            "anchor_pos_bad": done.clone(),
            "anchor_ori_bad": torch.zeros(2, dtype=torch.bool),
            "ee_body_bad": torch.zeros(2, dtype=torch.bool),
        }
        info = {
            "done_terms": done_terms,
            "reward_terms": {"anchor_pos_reward": torch.ones(2)},
            "termination_phase_steps": torch.zeros(2, dtype=torch.long),
        }
        return obs, reward, done, info

    def _record_adaptive_motion_failures(self, failed_env_ids: torch.Tensor, failure_phase_steps: torch.Tensor) -> None:
        del failed_env_ids, failure_phase_steps

    def _update_adaptive_motion_sampling(self) -> None:
        pass


def test_chunk_step_masks_rewards_after_first_done_without_auto_reset() -> None:
    harness = _StepHarness()
    actions = torch.zeros(2, 3, 3)

    obs_list, rewards, terminations, truncations, infos = harness.chunk_step(actions, auto_reset=False)

    assert len(obs_list) == 3
    assert len(infos) == 3
    assert torch.equal(rewards[0], torch.tensor([1.0, 0.0, 0.0]))
    assert torch.equal(rewards[1], torch.tensor([11.0, 12.0, 13.0]))
    assert torch.equal(terminations[0], torch.tensor([True, False, False]))
    assert not truncations.any()


def test_chunk_step_rejects_bad_action_shapes() -> None:
    harness = _StepHarness()

    with pytest.raises(ValueError, match="chunk_actions must have shape"):
        harness.chunk_step(torch.zeros(2, 3), auto_reset=False)
    with pytest.raises(ValueError, match="Expected chunk_actions shape"):
        harness.chunk_step(torch.zeros(3, 2, 3), auto_reset=False)
