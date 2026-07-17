from __future__ import annotations

import math
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from engine.checkpoint import _resume_signature
from engine.config import AdaMimicConfig, load_config
from envs.adamimic import (
    ADAMIMIC_BODY_NAMES,
    ADAMIMIC_CRITIC_OBSERVATION_DIM,
    ADAMIMIC_OBSERVATION_DIM,
    AdaMimicEnvironment,
)
from method import load_method_class
from method.adamimic import AdaMimic
from models.adamimic_policy import AdaMimicActorCritic


ROOT = Path(__file__).resolve().parents[1]


def test_adamimic_config_and_loader() -> None:
    cfg = load_config(ROOT / "configs" / "adamimic_stage1_largebox.yaml")
    assert cfg.method == "adamimic"
    assert isinstance(cfg.parameters, AdaMimicConfig)
    assert cfg.parameters.stage == "stage1"
    assert cfg.environment.num_envs == 4096
    assert cfg.parameters.rollout_env_steps == 75
    assert cfg.parameters.actor_time_scale_range == (0.0, 0.0)
    assert cfg.parameters.actor_observation_history == 5
    assert cfg.parameters.reward_group_weights == ((0.5, 1.0), (0.5, 1.0))
    assert cfg.parameters.domain_randomization is True
    assert not hasattr(cfg.parameters, "empirical_normalization")
    assert not hasattr(cfg.parameters, "time_reward_scale")
    assert cfg.environment.reset_phase_sampling == "rsi"
    assert cfg.environment.adaptive_motion_sampling is False
    assert cfg.environment.termination_mode == "adamimic"
    assert cfg.environment.terminate_on_motion_end is True
    assert cfg.environment.interval_pushes is False
    assert cfg.environment.adamimic_keyframe_phases == (
        10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 115, 130, 145, 160,
        175, 190, 205, 215, 225, 235, 245, 255, 265, 270, 280, 295, 310, 324,
    )
    assert cfg.environment.adamimic_special_keyframe_indices == (4, 9, 17, 23)
    assert tuple(
        cfg.environment.adamimic_keyframe_phases[index]
        for index in cfg.environment.adamimic_special_keyframe_indices
    ) == (50, 100, 215, 270)
    assert cfg.parameters.termination_initial_threshold == 1.5
    assert cfg.parameters.termination_min_threshold == 0.6
    assert cfg.parameters.limit_initial_soft_factor == 1.15
    assert cfg.parameters.penalty_initial_scale == 0.1
    assert cfg.training.max_updates == 40_000
    assert load_method_class("adamimic") is AdaMimic


def test_adamimic_stage2_checkpoint_path_is_resolved() -> None:
    cfg = load_config(ROOT / "configs" / "adamimic_stage2_largebox.yaml")
    assert cfg.parameters.stage == "stage2"
    assert cfg.environment.num_envs == 4096
    assert cfg.parameters.residual_delta is True
    assert cfg.environment.reset_phase_sampling == "zero"
    assert cfg.parameters.termination_initial_threshold == 2.0
    assert cfg.parameters.limit_initial_soft_factor == 0.98
    assert cfg.parameters.penalty_initial_scale == 0.2
    assert cfg.training.max_updates == 10_000
    assert cfg.parameters.checkpoint_path.endswith("runs/adamimic_stage1_largebox/checkpoints/last.pt")


def test_adamimic_stages_share_benchmark_identity() -> None:
    stage1 = load_config(ROOT / "configs" / "adamimic_stage1_largebox.yaml")
    stage2 = load_config(ROOT / "configs" / "adamimic_stage2_largebox.yaml")
    for name in (
        "platform_profile",
        "task",
        "sim_dt",
        "decimation",
        "adamimic_keyframe_phases",
        "adamimic_special_keyframe_indices",
    ):
        assert getattr(stage1.environment, name) == getattr(stage2.environment, name)


def test_adamimic_keyframes_are_part_of_resume_signature() -> None:
    cfg = load_config(ROOT / "configs" / "adamimic_stage1_largebox.yaml")
    config = asdict(cfg)
    baseline = _resume_signature(config)
    changed_phases = dict(config["environment"])
    changed_phases["adamimic_keyframe_phases"] = (*cfg.environment.adamimic_keyframe_phases[:-1], 323)
    changed_special = dict(config["environment"])
    changed_special["adamimic_special_keyframe_indices"] = (3, 9, 17, 23)
    assert baseline != _resume_signature({
        "method": cfg.method,
        "environment": changed_phases,
        "parameters": config["parameters"],
    })
    assert baseline != _resume_signature({
        "method": cfg.method,
        "environment": changed_special,
        "parameters": config["parameters"],
    })


def test_adamimic_stage1_rejects_adaptive_time_overrides() -> None:
    with pytest.raises(ValueError, match="stage1"):
        load_config(
            ROOT / "configs" / "adamimic_stage1_largebox.yaml",
            [
                "parameters.train_time=true",
                "parameters.actor_time_scale_range=[-0.015, 0.02]",
            ],
        )


def test_adamimic_official_budget_can_be_overridden_for_monitoring() -> None:
    cfg = load_config(
        ROOT / "configs" / "adamimic_stage1_largebox.yaml",
        ["training.max_updates=200", "training.validation_every=10"],
    )
    assert cfg.training.max_updates == 200
    assert cfg.training.validation_every == 10


def test_adamimic_stage1_rejects_non_official_reset_sampling() -> None:
    with pytest.raises(ValueError, match="RSI"):
        load_config(
            ROOT / "configs" / "adamimic_stage1_largebox.yaml",
            [
                "environment.reset_phase_sampling=adaptive",
                "environment.adaptive_motion_sampling=true",
            ],
        )


def test_adamimic_rejects_implicit_or_invalid_keyframes() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        load_config(
            ROOT / "configs" / "adamimic_stage1_largebox.yaml",
            ["environment.adamimic_keyframe_phases=[0, 10, 10]", "environment.rsi_keyframe_count=3"],
        )


def test_adamimic_actor_critic_two_level_action_shape() -> None:
    torch.manual_seed(0)
    model = AdaMimicActorCritic(
        actor_obs_dim=8,
        critic_obs_dim=7,
        control_action_dim=3,
        actor_hidden_dims=(16, 8),
        critic_hidden_dims=(16, 8),
        activation="elu",
        init_noise_std=0.5,
        infer_keyframe_time=True,
        actor_time_scale_range=(-0.01, 0.02),
        fixed_dt=0.02,
        time_min_std=0.005,
    )
    obs = torch.randn(5, 8)
    critic = torch.randn(5, 7)
    action = model.act(obs)
    assert action.shape == (5, 4)
    low_logp, high_logp = model.get_actions_log_prob(action)
    assert low_logp.shape == (5,)
    assert high_logp.shape == (5,)
    assert model.evaluate_low(critic, action[:, -1:]).shape == (5, 2)
    assert model.evaluate_high(critic).shape == (5, 2)
    assert model.act_inference(obs).shape == (5, 4)
    assert model.entropy.shape == (5,)
    assert torch.allclose(model.entropy, model.distribution.entropy().sum(dim=-1))


def test_adamimic_normalizes_each_critic_then_applies_official_weights() -> None:
    cfg = load_config(ROOT / "configs" / "adamimic_stage1_largebox.yaml").parameters
    algorithm = AdaMimic(cfg=cfg, env=None, simulation_app=None)
    advantages = torch.tensor(
        [
            [[1.0, 50.0], [2.0, 20.0]],
            [[4.0, 10.0], [8.0, -5.0]],
        ]
    )
    normalized = algorithm._normalize_advantages(advantages)
    assert torch.allclose(normalized.mean(dim=(0, 1)), torch.zeros(2), atol=1.0e-6)
    assert torch.allclose(normalized.std(dim=(0, 1)), torch.ones(2), atol=1.0e-6)
    expected = normalized[..., 0] * 0.5 + normalized[..., 1]
    assert torch.allclose(algorithm._combine_advantages(advantages, 0), expected)
    assert torch.allclose(algorithm._combine_advantages(advantages, 1), expected)


def test_adamimic_stage2_loads_only_base_actor_and_low_critics(tmp_path: Path) -> None:
    kwargs = dict(
        actor_obs_dim=8,
        critic_obs_dim=7,
        control_action_dim=3,
        actor_hidden_dims=(16, 8),
        critic_hidden_dims=(16, 8),
        activation="elu",
        infer_keyframe_time=True,
        actor_time_scale_range=(-0.015, 0.02),
        fixed_dt=0.02,
        time_min_std=0.005,
    )
    source = AdaMimicActorCritic(**kwargs, init_noise_std=0.8)
    target = AdaMimicActorCritic(
        **kwargs,
        init_noise_std=0.08,
        residual_delta=True,
    )
    with torch.no_grad():
        for parameter in source.actor.parameters():
            parameter.fill_(0.25)
        for parameter in source.critics.parameters():
            parameter.fill_(0.5)
        for parameter in source.actor_time.parameters():
            parameter.fill_(0.75)
        source.std.fill_(0.8)

    untouched = {
        "std": target.std.detach().clone(),
        "actor_time": {key: value.detach().clone() for key, value in target.actor_time.state_dict().items()},
        "actor_delta": {key: value.detach().clone() for key, value in target.actor_delta.state_dict().items()},
        "critics_time": {key: value.detach().clone() for key, value in target.critics_time.state_dict().items()},
        "critics_delta": {key: value.detach().clone() for key, value in target.critics_delta.state_dict().items()},
    }
    checkpoint = tmp_path / "stage1.pt"
    torch.save(
        {"policy": {f"model.{key}": value for key, value in source.state_dict().items()}},
        checkpoint,
    )
    algorithm = AdaMimic(cfg=None, env=None, simulation_app=None)
    algorithm._model = target
    algorithm._load_stage1_weights(checkpoint)

    assert all(torch.all(parameter == 0.25) for parameter in target.actor.parameters())
    assert all(torch.all(parameter == 0.5) for parameter in target.critics.parameters())
    assert torch.equal(target.std, untouched["std"])
    for name in ("actor_time", "actor_delta", "critics_time", "critics_delta"):
        module = getattr(target, name)
        for key, value in module.state_dict().items():
            assert torch.equal(value, untouched[name][key])


class _MockMotion:
    num_frames = 325
    fps = 50

    def __init__(self, num_bodies: int):
        self.num_bodies = num_bodies

    @staticmethod
    def _identity_quat(count: int, trailing: tuple[int, ...] = ()) -> torch.Tensor:
        quat = torch.zeros(count, *trailing, 4)
        quat[..., 0] = 1.0
        return quat

    def get_frame(self, phase: torch.Tensor) -> dict[str, torch.Tensor]:
        count = int(phase.numel())
        root_pos = torch.zeros(count, 3)
        root_pos[:, 2] = 1.0
        return {
            "joint_pos": torch.zeros(count, 29),
            "root_pos_w": root_pos,
            "root_quat_w": self._identity_quat(count),
        }

    def get_full_body_state(self, phase: torch.Tensor) -> dict[str, torch.Tensor]:
        count = int(phase.numel())
        body_pos = torch.zeros(count, self.num_bodies, 3)
        body_pos[..., 2] = 1.0
        return {
            "body_pos_w": body_pos,
            "body_quat_w": self._identity_quat(count, (self.num_bodies,)),
        }


class _MockPhysxView:
    def __init__(self, num_envs: int, num_bodies: int):
        self.masses = torch.full((num_envs, num_bodies), 10.0)
        self.inertias = torch.ones(num_envs, num_bodies, 9)
        self.coms = torch.zeros(num_envs, num_bodies, 7)
        self.materials = torch.zeros(num_envs, num_bodies, 3)

    def set_masses(self, value: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.masses[env_ids] = value.clone()

    def set_inertias(self, value: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.inertias[env_ids] = value.clone()

    def get_coms(self) -> torch.Tensor:
        return self.coms

    def set_coms(self, value: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.coms[env_ids] = value.clone()

    def get_material_properties(self) -> torch.Tensor:
        return self.materials

    def set_material_properties(self, value: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.materials[env_ids] = value.clone()


class _MockAdaBaseEnv:
    def __init__(self, environment_cfg, num_envs: int = 4):
        self.config = environment_cfg
        self.num_envs = num_envs
        self.action_dim = 29
        self.device = torch.device("cpu")
        self.dt = 0.02
        self.decimation = 4
        self.max_episode_steps = 500
        self.phase_steps = torch.zeros(num_envs)
        self.episode_steps = torch.zeros(num_envs, dtype=torch.long)
        self.action_joint_ids = torch.arange(self.action_dim)
        self.observation_noise = False
        self.interval_pushes = False

        # IsaacLab merges the inertialess fixed mid360 link into torso_link.
        body_names = ADAMIMIC_BODY_NAMES
        body_count = len(body_names)
        body_pos = torch.zeros(num_envs, body_count, 3)
        body_pos[..., 2] = 1.0
        body_quat = torch.zeros(num_envs, body_count, 4)
        body_quat[..., 0] = 1.0
        root_pos = torch.zeros(num_envs, 3)
        root_pos[:, 2] = 1.0
        root_quat = torch.zeros(num_envs, 4)
        root_quat[:, 0] = 1.0
        joint_limits = torch.empty(num_envs, self.action_dim, 2)
        joint_limits[..., 0] = -2.0
        joint_limits[..., 1] = 2.0
        data = SimpleNamespace(
            body_pos_w=body_pos,
            body_quat_w=body_quat,
            root_pos_w=root_pos,
            root_quat_w=root_quat,
            root_ang_vel_b=torch.zeros(num_envs, 3),
            root_lin_vel_b=torch.zeros(num_envs, 3),
            GRAVITY_VEC_W=torch.tensor([0.0, 0.0, -1.0]),
            joint_pos_limits=joint_limits,
            joint_vel_limits=torch.full((num_envs, self.action_dim), 10.0),
            applied_torque=torch.zeros(num_envs, self.action_dim),
            joint_effort_limits=torch.full((num_envs, self.action_dim), 100.0),
            joint_stiffness=torch.full((num_envs, self.action_dim), 20.0),
            default_mass=torch.full((num_envs, body_count), 10.0),
            default_inertia=torch.ones(num_envs, body_count, 9),
        )
        actuator = SimpleNamespace(
            joint_indices=torch.arange(self.action_dim),
            stiffness=torch.full((num_envs, self.action_dim), 20.0),
            damping=torch.full((num_envs, self.action_dim), 1.0),
        )
        self.robot = SimpleNamespace(
            body_names=list(body_names),
            num_joints=self.action_dim,
            num_bodies=body_count,
            actuators={"all": actuator},
            data=data,
            root_physx_view=_MockPhysxView(num_envs, body_count),
            set_joint_effort_target=lambda effort, joint_ids: None,
            write_root_velocity_to_sim=lambda velocity, env_ids=None: None,
            write_root_link_velocity_to_sim=lambda velocity, env_ids=None: None,
        )
        self.motion = _MockMotion(body_count)
        self.scene = SimpleNamespace(env_origins=torch.zeros(num_envs, 3))
        self._joint_pos = torch.zeros(num_envs, self.action_dim)
        self._joint_vel = torch.zeros_like(self._joint_pos)

    def reset(self, phase_indices=None):
        self.phase_steps.copy_(
            torch.zeros_like(self.phase_steps)
            if phase_indices is None
            else phase_indices.to(dtype=self.phase_steps.dtype)
        )
        self.episode_steps.zero_()
        return torch.zeros(self.num_envs, 1)

    def _reset_env_state(self, env_ids: torch.Tensor, phase_indices: torch.Tensor) -> None:
        self.phase_steps[env_ids] = phase_indices
        self.episode_steps[env_ids] = 0

    def get_action_joint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._joint_pos, self._joint_vel

    def get_mimic_root_velocity_w(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 6)

    def adaptive_sampling_stats(self) -> dict[str, float]:
        return {
            "mode": 0.0,
            "top_bin": 0.0,
            "top_prob": 0.0,
            "failed_sum": 0.0,
            "entropy": 0.0,
            "peak_bin": 0.0,
            "rsi_keyframe_count": 28.0,
        }

    def step(self, action, auto_reset=False, reference_dt=None, physics_substep_actions=None):
        del auto_reset
        assert action.shape == (self.num_envs, self.action_dim)
        if physics_substep_actions is not None:
            assert physics_substep_actions.shape == (
                self.decimation,
                self.num_envs,
                self.action_dim,
            )
            self.last_physics_substep_actions = physics_substep_actions.clone()
        frame_delta = torch.as_tensor(reference_dt).reshape(self.num_envs) * self.motion.fps
        self.phase_steps.add_(frame_delta).clamp_(max=self.motion.num_frames - 1)
        self.episode_steps += 1
        wall_timeout = self.episode_steps >= self.max_episode_steps
        zeros = torch.zeros_like(wall_timeout)
        info = {
            "done_terms": {
                "time_out": wall_timeout,
                "motion_complete": zeros,
                "anchor_pos_bad": zeros,
                "anchor_ori_bad": zeros,
                "ee_body_bad": zeros,
            },
            "reward_terms": {},
            "termination_phase_steps": self.phase_steps.clone(),
            "reference_frame_delta": frame_delta,
        }
        return (
            torch.zeros(self.num_envs, 1),
            torch.zeros(self.num_envs),
            wall_timeout,
            info,
        )


def _adamimic_adapter(stage: str = "stage1", num_envs: int = 4) -> AdaMimicEnvironment:
    config_name = f"adamimic_{stage}_largebox.yaml"
    experiment = load_config(ROOT / "configs" / config_name)
    return AdaMimicEnvironment(
        _MockAdaBaseEnv(experiment.environment, num_envs=num_envs),
        replace(experiment.parameters, domain_randomization=False),
    )


def _adamimic_test_cfg() -> AdaMimicConfig:
    cfg = load_config(ROOT / "configs" / "adamimic_stage1_largebox.yaml").parameters
    return replace(
        cfg,
        actor_hidden_dims=(16, 8),
        critic_hidden_dims=(16, 8),
        rollout_env_steps=5,
        policy_epochs=2,
        num_mini_batches=2,
        micro_batch_size=4,
        init_at_random_ep_len=False,
        domain_randomization=False,
    )


def test_adamimic_history_and_critic_shapes() -> None:
    adapter = _adamimic_adapter(num_envs=3)
    observation = adapter.reset(phase_indices=torch.zeros(3), warmup=False)
    assert observation.shape == (3, ADAMIMIC_OBSERVATION_DIM)
    assert adapter.get_critic_observation().shape == (3, ADAMIMIC_CRITIC_OBSERVATION_DIM)
    history = observation.reshape(3, 5, -1)
    assert torch.count_nonzero(history[:, :-1]) == 0
    assert torch.count_nonzero(history[:, -1]) > 0
    assert adapter.lidar_body_id is None


def test_adamimic_first_training_reset_starts_at_phase_zero_then_uses_rsi() -> None:
    adapter = _adamimic_adapter(num_envs=3)
    adapter.sample_rsi = lambda count: torch.full((count,), 50.0)
    adapter.reset(warmup=False)
    assert torch.equal(adapter.base_env.phase_steps, torch.zeros(3))
    adapter.reset(warmup=False)
    assert torch.equal(adapter.base_env.phase_steps, torch.full((3,), 50.0))


def test_adamimic_adapter_consumes_explicit_curriculum_fields() -> None:
    experiment = load_config(ROOT / "configs" / "adamimic_stage1_largebox.yaml")
    parameters = replace(
        experiment.parameters,
        domain_randomization=False,
        termination_initial_threshold=1.7,
        limit_initial_soft_factor=1.12,
        penalty_initial_scale=0.13,
    )
    adapter = AdaMimicEnvironment(_MockAdaBaseEnv(experiment.environment), parameters)
    assert adapter.curriculum.termination_threshold == 1.7
    assert adapter.curriculum.soft_pos == 1.12
    assert adapter.curriculum.soft_vel == 1.12
    assert adapter.curriculum.soft_torque == 1.12
    assert adapter.curriculum.penalty_scale == 0.13


def test_adamimic_domain_randomization_is_active_and_bounded() -> None:
    torch.manual_seed(7)
    experiment = load_config(ROOT / "configs" / "adamimic_stage1_largebox.yaml")
    base_env = _MockAdaBaseEnv(experiment.environment, num_envs=8)
    adapter = AdaMimicEnvironment(base_env, experiment.parameters)
    assert adapter.domain_randomization is True
    view = base_env.robot.root_physx_view
    assert torch.all((view.masses[:, 0] >= 8.0) & (view.masses[:, 0] <= 15.0))
    assert torch.all((view.masses[:, 1:] >= 9.0) & (view.masses[:, 1:] <= 11.0))
    assert torch.all((view.materials[..., :2] >= 0.1) & (view.materials[..., :2] <= 1.1))
    assert torch.all((view.materials[..., 2] >= 0.0) & (view.materials[..., 2] <= 0.1))
    assert torch.all((adapter._delay_index >= 0) & (adapter._delay_index <= 4))
    assert torch.all((adapter._kp_factor >= 0.85) & (adapter._kp_factor <= 1.15))
    assert torch.all((adapter._kd_factor >= 0.85) & (adapter._kd_factor <= 1.15))
    assert torch.all((adapter._motor_strength >= 0.9) & (adapter._motor_strength <= 1.1))
    assert torch.all(adapter._actuation_offset.abs() <= 3.0)


def test_adamimic_action_delay_advances_at_physics_substeps() -> None:
    adapter = _adamimic_adapter(num_envs=5)
    adapter._delay_buffer.zero_()
    adapter._delay_index.copy_(torch.arange(5))
    adapter._base_step(torch.ones(5, 29), torch.full((5,), 0.02))
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 1.0],
        ]
    )
    actual = adapter.base_env.last_physics_substep_actions[..., 0]
    assert torch.equal(actual, expected)


def test_adamimic_validation_temporarily_uses_nominal_physics() -> None:
    torch.manual_seed(7)
    experiment = load_config(ROOT / "configs" / "adamimic_stage1_largebox.yaml")
    base_env = _MockAdaBaseEnv(experiment.environment, num_envs=8)
    adapter = AdaMimicEnvironment(base_env, experiment.parameters)
    randomized_masses = adapter._training_physical_properties["masses"].clone()
    nominal_masses = adapter._nominal_physical_properties["masses"].clone()
    assert not torch.equal(randomized_masses, nominal_masses)

    adapter.reset(phase_indices=torch.zeros(8), warmup=False)
    assert torch.equal(base_env.robot.root_physx_view.masses, randomized_masses)
    assert adapter._using_nominal_physics is False

    snapshot = adapter.snapshot_runtime_state()
    adapter.reset_evaluation(torch.zeros(8))
    assert torch.equal(base_env.robot.root_physx_view.masses, nominal_masses)
    assert adapter._using_nominal_physics is True

    adapter.restore_runtime_state(snapshot)
    assert torch.equal(base_env.robot.root_physx_view.masses, randomized_masses)
    assert adapter._using_nominal_physics is False


def test_adamimic_emits_two_distinct_reward_groups() -> None:
    adapter = _adamimic_adapter(num_envs=2)
    adapter.reset(phase_indices=torch.zeros(2), warmup=False)
    observation, reward_low, reward_high, done, info = adapter.step_training(
        torch.zeros(2, 29),
        torch.full((2,), 0.02),
    )
    assert observation.shape == (2, ADAMIMIC_OBSERVATION_DIM)
    assert reward_low.shape == reward_high.shape == (2, 2)
    assert torch.any(reward_low[:, 0] != reward_low[:, 1])
    assert not torch.equal(reward_low, reward_high)
    assert info["adamimic_reward_low"].shape == (2, 2)
    assert info["adamimic_reward_high"].shape == (2, 2)
    assert not done.any()


def test_adamimic_slow_clock_ignores_shared_500_step_wall_timeout() -> None:
    adapter = _adamimic_adapter(stage="stage2", num_envs=2)
    adapter.reset(phase_indices=torch.full((2,), 100.0), warmup=False)
    adapter.base_env.episode_steps.fill_(499)
    _, _, _, done, info = adapter.step_training(
        torch.zeros(2, 29),
        torch.full((2,), 0.005),
    )
    assert torch.all(adapter.base_env.episode_steps == 500)
    assert torch.allclose(adapter.base_env.phase_steps, torch.full((2,), 100.25))
    assert not done.any()
    assert not info["done_terms"]["time_out"].any()


def test_adamimic_checkpoint_extra_restores_curriculum_and_push_clock() -> None:
    experiment = load_config(ROOT / "configs" / "adamimic_stage1_largebox.yaml")
    source = AdaMimic(
        cfg=_adamimic_test_cfg(),
        env=_MockAdaBaseEnv(experiment.environment),
        simulation_app=None,
    )
    source.build()
    source.initial_reset()
    source.learning_rate = 0.004
    source.adam_env._training_step_count = 777
    source.adam_env.curriculum.termination_threshold = 1.234
    source.adam_env.curriculum.penalty_scale = 0.123
    source.adam_env.curriculum.update_index = 321
    payload = source.extra_checkpoint_state()

    target = AdaMimic(
        cfg=_adamimic_test_cfg(),
        env=_MockAdaBaseEnv(experiment.environment),
        simulation_app=None,
    )
    target.build()
    target.load_extra_checkpoint_state(payload)
    assert target.learning_rate == 0.004
    assert target.adam_env._training_step_count == 777
    assert target.adam_env._first_reset is False
    assert target.adam_env.curriculum.termination_threshold == 1.234
    assert target.adam_env.curriculum.penalty_scale == 0.123
    assert target.adam_env.curriculum.update_index == 321


def test_adamimic_collect_and_update_end_to_end() -> None:
    torch.manual_seed(0)
    experiment = load_config(ROOT / "configs" / "adamimic_stage1_largebox.yaml")
    env = _MockAdaBaseEnv(experiment.environment)
    algo = AdaMimic(cfg=_adamimic_test_cfg(), env=env, simulation_app=None)
    algo.build()
    obs = algo.initial_reset()
    algo.reset_for_update(8_000)
    assert algo.adam_env.curriculum.update_index == 8_000
    rollout = algo.collect(obs)
    assert rollout["actions"].shape == (5, env.num_envs, env.action_dim + 1)
    assert rollout["reward_low"].shape == (5, env.num_envs, 2)
    assert rollout["reward_high"].shape == (5, env.num_envs, 2)
    metrics = algo.update(rollout, collect_time=0.1)
    assert math.isfinite(metrics["adamimic/loss"])
    assert math.isfinite(metrics["adamimic/value_low_loss"])
    # TrackRolloutStorage intentionally trains on T-1 transitions: 4 * (5-1).
    assert metrics["adamimic/sample_count"] == 16.0
    assert metrics["policy/action_delta"] >= 0.0
    assert algo.deterministic_actions(rollout["next_observation"]).shape == (
        env.num_envs,
        env.action_dim,
    )
