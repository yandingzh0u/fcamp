from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch


def _module(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


@pytest.fixture()
def mimic_env_type(monkeypatch):
    class _Observation:
        pass

    class _Robot:
        pass

    class _Step:
        pass

    class _Termination:
        pass

    math_module = _module(
        "isaaclab.utils.math",
        quat_from_euler_xyz=lambda *args: args[0],
        quat_mul=lambda first, _second: first,
    )
    monkeypatch.setitem(sys.modules, "isaaclab", _module("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.utils", _module("isaaclab.utils"))
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math_module)
    monkeypatch.setitem(
        sys.modules,
        "components.rollout.reset_diagnostics",
        _module(
            "components.rollout.reset_diagnostics",
            ResetPhaseRecorder=object,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "envs.adaptive_sampling",
        _module("envs.adaptive_sampling", AdaptiveTimestepsSampler=object),
    )
    monkeypatch.setitem(
        sys.modules,
        "envs.imitation_data",
        _module(
            "envs.imitation_data",
            G1_IMITATION_FRAME_DIM=1,
            G1_IMITATION_KEY_BODY_NAMES=(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "envs.spec",
        _module(
            "envs.spec",
            CRITIC_OBS_DIM=2,
            OBS_DIM=2,
            PROJECT_ROOT=Path("."),
            RESET_JOINT_POSITION_RANGE=(0.0, 0.0),
            RESET_ROOT_POSE_RANGE=((0.0, 0.0),) * 6,
            VELOCITY_RANGE=((0.0, 0.0),) * 6,
            MIMIC_ANCHOR_BODY_NAME="anchor",
            MIMIC_BODY_NAMES=(),
            MIMIC_EE_BODY_NAMES=(),
            MIMIC_TERMINATION_BODY_NAMES=(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "envs.motion",
        _module(
            "envs.motion",
            MimicMotionReference=object,
            mimickit_frame_delta=lambda _fps, _dt: 1.0,
            mimickit_full_motion_steps=lambda *_args: 1,
            sample_mimickit_phase=lambda *_args, **_kwargs: torch.zeros(1),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "envs.observation",
        _module("envs.observation", MimicObservationMixin=_Observation),
    )
    monkeypatch.setitem(
        sys.modules,
        "envs.robot",
        _module(
            "envs.robot",
            G1Env=_Robot,
            RootVelocityFrame=str,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "envs.robots.g1",
        _module("envs.robots.g1", G1_29DOF_ACTION_NAMES=("a", "b")),
    )
    monkeypatch.setitem(
        sys.modules,
        "envs.step",
        _module("envs.step", MimicStepMixin=_Step),
    )
    monkeypatch.setitem(
        sys.modules,
        "envs.terminal",
        _module("envs.terminal", MimicTerminationMixin=_Termination),
    )
    monkeypatch.setitem(
        sys.modules,
        "envs.tasks",
        _module("envs.tasks", resolve_task=lambda _name: None),
    )

    path = Path(__file__).parents[1] / "envs" / "g1_mimic.py"
    spec = importlib.util.spec_from_file_location("envs._atomic_reset_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.G1MimicEnv


class _Motion:
    def __init__(self, joint_positions: torch.Tensor) -> None:
        self.joint_positions = joint_positions

    def clamp_time_steps(self, phases: torch.Tensor) -> torch.Tensor:
        return phases.clamp(0, self.joint_positions.shape[0] - 1)

    def get_frame(self, phases: torch.Tensor) -> dict[str, torch.Tensor]:
        joint_pos = self.joint_positions.index_select(0, phases.long())
        count = phases.numel()
        return {
            "root_pos_w": torch.zeros(count, 3),
            "root_quat_w": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(
                count, -1
            ),
            "root_lin_vel_w": torch.zeros(count, 3),
            "root_ang_vel_w": torch.zeros(count, 3),
            "joint_pos": joint_pos,
            "joint_vel": torch.zeros_like(joint_pos),
        }


class _Scene:
    def __init__(self, owner) -> None:
        self.owner = owner
        self.events: list[str] = []
        self.anchor_at_update: torch.Tensor | None = None

    def reset(self, *, env_ids: torch.Tensor) -> None:
        del env_ids
        self.events.append("reset")

    def update(self, _physics_dt: float) -> None:
        self.events.append("update")
        self.anchor_at_update = self.owner.last_action.clone()


def _make_env(
    env_type,
    *,
    reference_joint_pos: torch.Tensor,
    contract_installed: bool,
    reset_noise: bool,
):
    env = object.__new__(env_type)
    env.num_envs = 4
    env.device = torch.device("cpu")
    env.physics_dt = 0.005
    env.motion = _Motion(reference_joint_pos)
    env.phase_steps = torch.full((4,), -1.0)
    env.episode_steps = torch.full((4,), 17, dtype=torch.long)
    env.episode_ids = torch.full((4,), -1, dtype=torch.long)
    env._next_episode_id = 10
    env._failure_recorded = torch.ones(4, dtype=torch.bool)
    env.last_action = torch.full((4, 2), -99.0)
    env.default_action_joint_pos = torch.tensor(
        [
            [0.00, 0.00],
            [0.20, -0.10],
            [-0.30, 0.40],
            [0.50, -0.20],
        ]
    )
    env.action_scale = torch.tensor([[0.50, 0.25]])
    env._policy_action_low = torch.full((2,), -5.0) if contract_installed else None
    env._policy_action_high = torch.full((2,), 5.0) if contract_installed else None
    env.reset_noise = reset_noise
    env.scene = _Scene(env)
    env.reset_phase_recorder = SimpleNamespace(record=lambda *_args: None)
    env._reset_interval_push_schedule = MethodType(
        lambda _self, _ids: None,
        env,
    )
    env.get_observation = MethodType(
        lambda self: self.last_action.clone(),
        env,
    )
    env.get_critic_observation = MethodType(
        lambda self: self.last_action.clone(),
        env,
    )
    env.written_joint_pos = None
    env._write_robot_state = MethodType(
        lambda self, **kwargs: setattr(
            self,
            "written_joint_pos",
            kwargs["joint_pos"].clone(),
        ),
        env,
    )
    env._apply_official_reset_noise = MethodType(
        lambda _self, _ids, _rp, _rq, _rlv, _rav, joint_pos: joint_pos.add_(
            100.0
        ),
        env,
    )
    env.validated_actions: list[torch.Tensor] = []

    def validate(self, actions: torch.Tensor) -> None:
        self.validated_actions.append(actions.clone())
        if bool((actions.abs() > 5.0).any()):
            raise RuntimeError("outside policy command domain")

    env.validate_policy_actions = MethodType(validate, env)
    return env


def test_atomic_reset_uses_clean_phase_reference_for_partial_envs(
    mimic_env_type,
) -> None:
    references = torch.tensor(
        [
            [0.25, -0.25],
            [0.70, 0.15],
            [-0.10, 0.80],
        ]
    )
    env = _make_env(
        mimic_env_type,
        reference_joint_pos=references,
        contract_installed=True,
        reset_noise=True,
    )
    env_ids = torch.tensor([3, 1, 2])
    phases = torch.tensor([0, 1, 2])
    expected = (
        references.index_select(0, phases)
        - env.default_action_joint_pos.index_select(0, env_ids)
    ) / env.action_scale

    observation = env.reset_envs(env_ids, phase_indices=phases)

    torch.testing.assert_close(observation, expected)
    torch.testing.assert_close(env.last_action.index_select(0, env_ids), expected)
    torch.testing.assert_close(
        env.last_action[0],
        torch.full((2,), -99.0),
    )
    torch.testing.assert_close(env.validated_actions[0], expected)
    torch.testing.assert_close(
        env.written_joint_pos,
        references.index_select(0, phases) + 100.0,
    )
    torch.testing.assert_close(
        env.scene.anchor_at_update.index_select(0, env_ids),
        expected,
    )
    assert env.scene.events == ["reset", "update"]


def test_pre_contract_reset_is_finite_and_does_not_clamp(
    mimic_env_type,
) -> None:
    references = torch.tensor(
        [
            [3.5, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
    )
    env = _make_env(
        mimic_env_type,
        reference_joint_pos=references,
        contract_installed=False,
        reset_noise=False,
    )
    env_ids = torch.tensor([0])

    observation = env.reset_envs(env_ids, phase_indices=torch.tensor([0]))

    torch.testing.assert_close(observation, torch.tensor([[7.0, 0.0]]))
    assert env.validated_actions == []

    env.action_scale[0, 0] = 0.0
    with pytest.raises(RuntimeError, match="non-finite reset policy command"):
        env.reset_envs(env_ids, phase_indices=torch.tensor([0]))
