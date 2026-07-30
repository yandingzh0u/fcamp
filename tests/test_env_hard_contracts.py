from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from envs.contracts import validate_actions_in_bounds

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


def test_fixed_reward_policy_command_domain_is_fixed_symmetric_scale() -> None:
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


def _class_method(tree: ast.Module, class_name: str, method_name: str) -> ast.FunctionDef:
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def test_all_root_velocity_paths_are_link_only_without_callsite_overrides() -> None:
    root = Path(__file__).parents[1]
    robot_tree = ast.parse((root / "envs/robot.py").read_text(encoding="utf-8"))
    mimic_tree = ast.parse((root / "envs/g1_mimic.py").read_text(encoding="utf-8"))
    step_tree = ast.parse((root / "envs/step.py").read_text(encoding="utf-8"))

    for method_name in (
        "get_mimic_root_velocity_w",
        "write_mimic_root_velocity_to_sim",
        "_write_robot_state",
    ):
        method = _class_method(robot_tree, "G1Env", method_name)
        argument_names = {
            argument.arg for argument in (*method.args.args, *method.args.kwonlyargs)
        }
        assert "velocity_frame" not in argument_names
        assert "root_velocity_frame" not in argument_names
    writer = _class_method(robot_tree, "G1Env", "write_mimic_root_velocity_to_sim")
    writer_calls = {
        node.func.attr
        for node in ast.walk(writer)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "write_root_link_velocity_to_sim" in writer_calls
    assert "write_root_velocity_to_sim" not in writer_calls

    for method_name in ("reset", "reset_envs", "_reset_env_state"):
        method = _class_method(mimic_tree, "G1MimicEnv", method_name)
        argument_names = {
            argument.arg for argument in (*method.args.args, *method.args.kwonlyargs)
        }
        assert "root_velocity_frame" not in argument_names
    interval_push = _class_method(step_tree, "MimicStepMixin", "_apply_interval_pushes")
    assert not any(
        keyword.arg in {"velocity_frame", "root_velocity_frame"}
        for call in ast.walk(interval_push)
        if isinstance(call, ast.Call)
        for keyword in call.keywords
    )


class _PushContractFixture(MimicStepMixin):
    def __init__(self) -> None:
        self.num_envs = 3
        self.device = torch.device("cpu")
        self.dt = 0.02
        self.interval_pushes = True
        self.episode_steps = torch.tensor([1, 1, 1], dtype=torch.long)
        self.next_push_step = torch.tensor([1, 1, 5], dtype=torch.long)
        self.first_push_step = torch.full((3,), -1, dtype=torch.long)
        self._last_interval_push_mask = torch.zeros(3, dtype=torch.bool)
        self.config = SimpleNamespace(root_velocity_mode="link")
        self.robot = _VelocityRecorder()

    @property
    def push_interval_step_range(self) -> tuple[int, int]:
        return 50, 50

    def get_mimic_root_velocity_w(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 6)

    def write_mimic_root_velocity_to_sim(
        self, velocity: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        self.robot.write_root_link_velocity_to_sim(velocity, env_ids)


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
    assert env.robot.com_env_ids.numel() == 0
    torch.testing.assert_close(env.robot.link_env_ids, torch.tensor([1]))
    assert env.first_push_step[0].item() == -1
    assert env.first_push_step[1].item() == 1


class _NoOpScene:
    def write_data_to_sim(self) -> None:
        pass

    def update(self, _dt: float) -> None:
        pass


class _NoOpSimulation:
    def step(self, *, render: bool) -> None:
        assert render is False


class _ShortMotionFixture(MimicStepMixin):
    def __init__(self) -> None:
        self.num_envs = 1
        self.device = torch.device("cpu")
        self.decimation = 1
        self.physics_dt = 0.005
        self.scene = _NoOpScene()
        self.sim = _NoOpSimulation()
        self.render = False
        self.phase_steps = torch.tensor([4.5])
        self.motion_frame_delta = 1.0
        self.motion_end_phase = 5
        self.motion = SimpleNamespace(num_frames=20)
        self.episode_steps = torch.zeros(1, dtype=torch.long)
        self.last_action = torch.zeros(1, 1)
        self._last_interval_push_mask = torch.zeros(1, dtype=torch.bool)

    def _apply_action_targets(self, actions: torch.Tensor) -> torch.Tensor:
        return actions

    def compute_reward(
        self,
        _actions: torch.Tensor,
        _previous_action: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return torch.zeros(1), {}

    def compute_termination(
        self,
    ) -> tuple[
        torch.Tensor,
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        false = torch.zeros(1, dtype=torch.bool)
        return self._motion_end_mask.clone(), {
            "anchor_pos_bad": false,
            "anchor_ori_bad": false,
            "ee_body_bad": false,
            "motion_complete": self._motion_end_mask.clone(),
        }, {}

    def _record_adaptive_failures(self, *_args) -> None:
        pass

    def _fold_adaptive_sampler(self) -> None:
        pass

    def _apply_interval_pushes(
        self, eligible_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        del eligible_mask
        return torch.zeros(1, dtype=torch.bool)

    def get_observation(self) -> torch.Tensor:
        return self.phase_steps[:, None]


def test_short_motion_end_controls_progress_clamp_and_completion() -> None:
    env = _ShortMotionFixture()

    observation, _reward, done, info = env.step(torch.zeros(1, 1))

    torch.testing.assert_close(env.phase_steps, torch.tensor([5.0]))
    torch.testing.assert_close(observation, torch.tensor([[5.0]]))
    torch.testing.assert_close(info["reference_phase_steps"], torch.tensor([5.0]))
    assert bool(done.item())
    assert bool(info["done_terms"]["motion_complete"].item())
    assert "imitation_frame" not in info
