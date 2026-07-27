from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

import torch

from engine.env_state import restore_env_state, snapshot_env_state


def _module(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_observation_mixin(monkeypatch):
    math_module = _module(
        "isaaclab.utils.math",
        matrix_from_quat=lambda value: value,
        quat_apply=lambda _quat, value: value,
        quat_inv=lambda value: value,
        quat_mul=lambda first, _second: first,
        subtract_frame_transforms=lambda *_args: (None, None),
        yaw_quat=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "isaaclab", _module("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.utils", _module("isaaclab.utils"))
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math_module)
    monkeypatch.setitem(
        sys.modules,
        "envs.spec",
        _module("envs.spec", OBS_DIM=1, CRITIC_OBS_DIM=1),
    )
    monkeypatch.setitem(
        sys.modules,
        "envs.imitation_data",
        _module(
            "envs.imitation_data",
            build_g1_imitation_frame=lambda **_kwargs: torch.empty(0),
        ),
    )
    path = Path(__file__).parents[1] / "envs" / "observation.py"
    spec = importlib.util.spec_from_file_location(
        "envs._command_state_observation_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MimicObservationMixin


def test_actor_and_critic_command_tail_is_normalized_rate_then_last_action(
    monkeypatch,
) -> None:
    mixin = _load_observation_mixin(monkeypatch)
    observation = object.__new__(mixin)
    observation.command_rate = torch.tensor(
        [[20.0, -5.0], [-40.0, 2.5]]
    )
    raw_rate = observation.command_rate.clone()
    observation.command_rate_limit = torch.tensor([40.0, 10.0])
    observation.last_action = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    prefix = torch.tensor([[9.0], [8.0]])

    assembled = observation._append_command_state((prefix,))

    torch.testing.assert_close(
        assembled,
        torch.tensor(
            [
                [9.0, 0.5, -0.5, 1.0, 2.0],
                [8.0, -1.0, 0.25, 3.0, 4.0],
            ]
        ),
    )
    torch.testing.assert_close(assembled[:, -2:], observation.last_action)
    torch.testing.assert_close(observation.command_rate, raw_rate)


class _Scene:
    def __init__(self, num_envs: int) -> None:
        self.env_origins = torch.zeros(num_envs, 3)

    def reset(self, env_ids: torch.Tensor) -> None:
        del env_ids

    def update(self, physics_dt: float) -> None:
        del physics_dt


def _state_fixture():
    num_envs = 3
    action_dim = 2
    robot_data = SimpleNamespace(
        root_link_pose_w=torch.zeros(num_envs, 7),
        joint_pos=torch.zeros(num_envs, action_dim),
        joint_vel=torch.zeros(num_envs, action_dim),
    )
    sensor_data = SimpleNamespace()
    sensor = SimpleNamespace(
        data=sensor_data,
        _timestamp=torch.arange(num_envs, dtype=torch.float32),
        _timestamp_last_update=torch.zeros(num_envs),
        _is_outdated=torch.zeros(num_envs, dtype=torch.bool),
    )
    sampler = SimpleNamespace(
        bin_failed_count=torch.zeros(2),
        current_bin_failed_count=torch.zeros(2),
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        device=torch.device("cpu"),
        physics_dt=0.005,
        action_joint_ids=torch.arange(action_dim),
        robot=SimpleNamespace(data=robot_data),
        scene=_Scene(num_envs),
        contact_sensor=sensor,
        adaptive_sampler=sampler,
        phase_steps=torch.tensor([0.0, 4.0, 9.0]),
        episode_steps=torch.tensor([1, 2, 3]),
        episode_ids=torch.tensor([10, 11, 12]),
        _next_episode_id=13,
        last_action=torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        command_rate=torch.tensor(
            [[7.5, -2.0], [0.0, 8.25], [-3.5, 1.0]]
        ),
        next_push_step=torch.tensor([20, 21, 22]),
        push_time_left=torch.tensor([1.0, 2.0, 3.0]),
        first_push_step=torch.tensor([-1, 7, 8]),
        _failure_recorded=torch.tensor([False, True, False]),
    )
    env.get_mimic_root_velocity_w = MethodType(
        lambda self: torch.zeros(self.num_envs, 6),
        env,
    )
    env._write_robot_state = MethodType(
        lambda self, **_kwargs: None,
        env,
    )
    return env


def test_snapshot_restore_recovers_raw_command_rate_exactly() -> None:
    env = _state_fixture()
    expected_rate = env.command_rate.clone()
    expected_action = env.last_action.clone()

    snapshot = snapshot_env_state(env)
    env.command_rate.add_(100.0)
    env.last_action.zero_()

    restore_env_state(env, snapshot)

    torch.testing.assert_close(env.command_rate, expected_rate, rtol=0.0, atol=0.0)
    torch.testing.assert_close(env.last_action, expected_action, rtol=0.0, atol=0.0)
    torch.testing.assert_close(snapshot["command_rate"], expected_rate)
