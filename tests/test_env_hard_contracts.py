from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from envs.contracts import (
    resolve_root_velocity_frame,
    select_imitation_root_domain,
    validate_actions_in_bounds,
)

_spec = ModuleType("envs.spec")
_spec.VELOCITY_RANGE = ((0.0, 0.0),) * 6
sys.modules["envs.spec"] = _spec
from envs.step import MimicStepMixin
sys.modules.pop("envs.spec", None)


class _VelocityRecorder:
    def __init__(self) -> None:
        self.com_env_ids = torch.empty(0, dtype=torch.long)
        self.link_env_ids = torch.empty(0, dtype=torch.long)

    def write_root_velocity_to_sim(
        self, _velocity: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        self.com_env_ids = env_ids.clone()

    def write_root_link_velocity_to_sim(
        self, _velocity: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        self.link_env_ids = env_ids.clone()


def test_fcamp_policy_command_domain_is_fixed_symmetric_scale() -> None:
    low = torch.full((3,), -5.0)
    high = torch.full((3,), 5.0)
    validate_actions_in_bounds(
        torch.stack([low, torch.zeros_like(low), high]), low, high
    )
    invalid = high.clone()
    invalid[0] += 0.01
    with pytest.raises(RuntimeError, match="policy command domain"):
        validate_actions_in_bounds(
            invalid.unsqueeze(0), low, high
        )


def test_fcamp_imitation_root_link_domain_is_strictly_scoped() -> None:
    legacy = (
        torch.tensor([[1.0, 2.0, 3.0]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]]),
    )
    link = (
        torch.tensor([[4.0, 5.0, 6.0]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        torch.tensor([[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]]),
    )
    kwargs = {
        "legacy_root_pos": legacy[0],
        "legacy_root_quat": legacy[1],
        "legacy_root_velocity": legacy[2],
        "root_link_pos": link[0],
        "root_link_quat": link[1],
        "root_link_velocity": link[2],
    }

    selected_legacy = select_imitation_root_domain(
        strict_fcamp=False, **kwargs
    )
    selected_fcamp = select_imitation_root_domain(
        strict_fcamp=True, **kwargs
    )

    for actual, expected in zip(selected_legacy, legacy, strict=True):
        torch.testing.assert_close(actual, expected)
    for actual, expected in zip(selected_fcamp, link, strict=True):
        torch.testing.assert_close(actual, expected)


def test_root_velocity_frame_is_explicit_not_inferred_from_the_tensor() -> None:
    assert resolve_root_velocity_frame("com", "link") == "link"
    assert resolve_root_velocity_frame("com", None) == "com"
    with pytest.raises(ValueError, match="root velocity frame"):
        resolve_root_velocity_frame("invalid", None)


class _PushContractFixture(MimicStepMixin):
    def __init__(self) -> None:
        self.num_envs = 3
        self.device = torch.device("cpu")
        self.dt = 0.02
        self.interval_pushes = True
        self.beyondmimic_global_push_timer = False
        self.episode_steps = torch.tensor([1, 1, 1], dtype=torch.long)
        self.next_push_step = torch.tensor([1, 1, 5], dtype=torch.long)
        self.first_push_step = torch.full((3,), -1, dtype=torch.long)
        self._last_interval_push_mask = torch.zeros(3, dtype=torch.bool)
        self.config = SimpleNamespace(root_velocity_mode="com")
        self.robot = _VelocityRecorder()

    @property
    def push_interval_step_range(self) -> tuple[int, int]:
        return 50, 50

    def get_mimic_root_velocity_w(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 6)

    def write_mimic_root_velocity_to_sim(
        self, velocity: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        self.robot.write_root_velocity_to_sim(velocity, env_ids)


def test_random_episode_age_reschedules_push_relative_to_the_new_age() -> None:
    env = _PushContractFixture()
    env_ids = torch.tensor([0, 2], dtype=torch.long)
    ages = torch.tensor([100, 400], dtype=torch.long)

    env.set_episode_age(env_ids, ages)

    torch.testing.assert_close(env.episode_steps, torch.tensor([100, 1, 400]))
    torch.testing.assert_close(env.next_push_step, torch.tensor([150, 1, 450]))
    torch.testing.assert_close(env.first_push_step, torch.tensor([-1, -1, -1]))


def test_terminal_env_is_not_pushed_and_returned_mask_is_physical_truth() -> None:
    env = _PushContractFixture()
    eligible = torch.tensor([False, True, True])

    actual_mask = env._apply_interval_pushes(eligible_mask=eligible)

    torch.testing.assert_close(actual_mask, torch.tensor([False, True, False]))
    torch.testing.assert_close(env._last_interval_push_mask, actual_mask)
    torch.testing.assert_close(env.robot.com_env_ids, torch.tensor([1]))
    assert env.first_push_step[0].item() == -1
    assert env.first_push_step[1].item() == 1
