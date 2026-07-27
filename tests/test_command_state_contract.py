from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

import torch

from engine.env_state import restore_env_state, snapshot_env_state
from envs.action_rate import advance_rate_servo


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
    monkeypatch.setitem(
        sys.modules,
        "envs.contracts",
        _module(
            "envs.contracts",
            select_imitation_root_domain=lambda **_kwargs: (),
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


def _load_robot_type(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "pxr",
        _module("pxr", UsdPhysics=SimpleNamespace()),
    )
    monkeypatch.setitem(sys.modules, "isaaclab", _module("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.sim", _module("isaaclab.sim"))
    monkeypatch.setitem(
        sys.modules,
        "isaaclab.assets",
        _module("isaaclab.assets", Articulation=object, AssetBaseCfg=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "isaaclab.scene",
        _module("isaaclab.scene", InteractiveScene=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "isaaclab.sim",
        _module(
            "isaaclab.sim",
            SimulationContext=object,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "envs.spec",
        _module(
            "envs.spec",
            G1SceneConfig=object,
            G1_MIMIC_ACTION_SCALE_VALUES=(),
            PUSH_INTERVAL_STEP_RANGE=(1, 1),
            STARTUP_BASE_COM_RANGE=(),
            STARTUP_JOINT_DEFAULT_POS_RANGE=(0.0, 0.0),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "envs.robots.g1",
        _module(
            "envs.robots.g1",
            G1_29DOF_ACTION_NAMES=(),
            make_g1_cfg=lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "envs.tasks",
        _module("envs.tasks", TaskSpec=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "envs.contracts",
        _module(
            "envs.contracts",
            RootVelocityFrame=str,
            resolve_root_velocity_frame=lambda *_args: "com",
            validate_actions_in_bounds=lambda *_args, **_kwargs: None,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "engine.config",
        _module("engine.config", EnvironmentConfig=object),
    )
    path = Path(__file__).parents[1] / "envs" / "robot.py"
    spec = importlib.util.spec_from_file_location(
        "envs._command_servo_environment_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.G1Env


def _servo_environment(robot_type):
    env = object.__new__(robot_type)
    env.scene = SimpleNamespace(num_envs=2)
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env.action_joint_ids = torch.arange(2)
    env.decimation = 4
    env.physics_dt = 0.005
    env.command_servo_omega = 20.0
    env._policy_action_low = torch.full((2,), -1.0)
    env._policy_action_high = torch.full((2,), 1.0)
    env.last_action = torch.tensor([[0.1, -0.2], [0.4, -0.3]])
    env.command_rate = torch.tensor([[0.2, -0.1], [0.3, -0.4]])
    env.command_acceleration = torch.tensor([[0.5, -0.2], [0.1, -0.6]])
    captured: dict[str, torch.Tensor] = {}

    def execute(
        self,
        actions,
        *,
        physics_substep_actions,
        command_state,
        **_kwargs,
    ):
        rate, acceleration = command_state
        captured["actions"] = actions.clone()
        captured["substeps"] = physics_substep_actions.clone()
        captured["rate"] = rate.clone()
        captured["acceleration"] = acceleration.clone()
        self.last_action.copy_(actions)
        self.command_rate.copy_(rate)
        self.command_acceleration.copy_(acceleration)
        return (
            torch.zeros(self.num_envs, 1),
            torch.zeros(self.num_envs),
            torch.zeros(self.num_envs, dtype=torch.bool),
            {
                "applied_action": actions.clone(),
                "command_rate": rate.clone(),
                "command_acceleration": acceleration.clone(),
            },
        )

    env.step = MethodType(execute, env)
    return env, captured


def test_command_tail_uses_servo_natural_units_then_last_action(
    monkeypatch,
) -> None:
    mixin = _load_observation_mixin(monkeypatch)
    observation = object.__new__(mixin)
    observation.command_rate = torch.tensor(
        [[20.0, -5.0], [-40.0, 2.5]]
    )
    observation.command_acceleration = torch.tensor(
        [[100.0, -25.0], [-200.0, 12.5]]
    )
    observation.command_servo_omega = 20.0
    observation.last_action = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    prefix = torch.tensor([[9.0], [8.0]])

    assembled = observation._append_command_state((prefix,))

    torch.testing.assert_close(
        assembled,
        torch.tensor(
            [
                [9.0, 1.0, -0.25, 0.25, -0.0625, 1.0, 2.0],
                [8.0, -2.0, 0.125, -0.5, 0.03125, 3.0, 4.0],
            ]
        ),
    )
    torch.testing.assert_close(assembled[:, -2:], observation.last_action)


def test_target_rate_executes_exact_substeps_and_persists_analytic_state(
    monkeypatch,
) -> None:
    robot_type = _load_robot_type(monkeypatch)
    env, captured = _servo_environment(robot_type)
    initial_action = env.last_action.clone()
    action = initial_action.clone()
    rate = env.command_rate.clone()
    acceleration = env.command_acceleration.clone()
    target_rate = torch.tensor([[1.2, -0.7], [-0.8, 0.9]])
    expected_substeps = []
    for _ in range(env.decimation):
        action, rate, acceleration = advance_rate_servo(
            target_rate,
            action,
            rate,
            acceleration,
            dt=env.physics_dt,
            omega=env.command_servo_omega,
        )
        expected_substeps.append(action)

    _, _, _, info = env.step_target_rate(target_rate)

    torch.testing.assert_close(
        captured["substeps"], torch.stack(expected_substeps)
    )
    torch.testing.assert_close(captured["actions"], action)
    torch.testing.assert_close(captured["rate"], rate)
    torch.testing.assert_close(captured["acceleration"], acceleration)
    torch.testing.assert_close(info["command_rate"], rate)
    torch.testing.assert_close(info["command_acceleration"], acceleration)
    assert not torch.allclose(
        rate,
        (action - initial_action) / (env.decimation * env.physics_dt),
    )


def test_projection_anti_windup_can_leave_bound_on_inward_target(
    monkeypatch,
) -> None:
    robot_type = _load_robot_type(monkeypatch)
    env, _ = _servo_environment(robot_type)
    env.last_action.fill_(1.0)
    env.command_rate.zero_()
    env.command_acceleration.zero_()

    _, _, _, outward_info = env.step_target_rate(
        torch.full_like(env.last_action, 10_000.0)
    )

    torch.testing.assert_close(env.last_action, torch.ones_like(env.last_action))
    torch.testing.assert_close(env.command_rate, torch.zeros_like(env.command_rate))
    torch.testing.assert_close(
        env.command_acceleration,
        torch.zeros_like(env.command_acceleration),
    )
    assert bool(outward_info["action_projection_mask"].all())

    _, _, _, inward_info = env.step_target_rate(
        torch.full_like(env.last_action, -1.0)
    )

    assert bool((env.last_action < 1.0).all())
    assert bool((env.command_rate < 0.0).all())
    assert not bool(inward_info["action_projection_mask"].any())


def test_inactive_target_rate_rows_keep_complete_command_state(
    monkeypatch,
) -> None:
    robot_type = _load_robot_type(monkeypatch)
    env, captured = _servo_environment(robot_type)
    previous_action = env.last_action.clone()
    previous_rate = env.command_rate.clone()
    previous_acceleration = env.command_acceleration.clone()
    active = torch.tensor([True, False])

    _, _, _, info = env.step_target_rate(
        torch.full_like(env.last_action, 2.0),
        active_mask=active,
    )

    torch.testing.assert_close(
        captured["substeps"][:, 1],
        previous_action[1].expand(env.decimation, -1),
    )
    torch.testing.assert_close(env.last_action[1], previous_action[1])
    torch.testing.assert_close(env.command_rate[1], previous_rate[1])
    torch.testing.assert_close(
        env.command_acceleration[1], previous_acceleration[1]
    )
    torch.testing.assert_close(
        info["target_rate"][1], torch.zeros_like(info["target_rate"][1])
    )
    assert not bool(info["action_projection_mask"][1].any())


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
        command_acceleration=torch.tensor(
            [[75.0, -20.0], [0.0, 82.5], [-35.0, 10.0]]
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


def test_snapshot_restore_recovers_complete_command_state_exactly() -> None:
    env = _state_fixture()
    expected_rate = env.command_rate.clone()
    expected_acceleration = env.command_acceleration.clone()
    expected_action = env.last_action.clone()

    snapshot = snapshot_env_state(env)
    env.command_rate.add_(100.0)
    env.command_acceleration.sub_(100.0)
    env.last_action.zero_()

    restore_env_state(env, snapshot)

    torch.testing.assert_close(env.command_rate, expected_rate, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        env.command_acceleration,
        expected_acceleration,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(env.last_action, expected_action, rtol=0.0, atol=0.0)
    torch.testing.assert_close(snapshot["command_rate"], expected_rate)
    torch.testing.assert_close(
        snapshot["command_acceleration"],
        expected_acceleration,
    )
