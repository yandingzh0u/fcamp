from __future__ import annotations

import math
import sys
from types import ModuleType
from types import SimpleNamespace

import torch

from envs.adaptive_sampling import BeyondMimicAdaptiveSampler
from envs.imitation_data import build_g1_imitation_frame

# The semantics below are pure tensor code.  Stub the simulator-heavy spec
# module during collection so these tests stay CPU-only and do not launch
# Isaac Sim.
_spec = ModuleType("envs.spec")
_spec.OBS_DIM = 171
_spec.CRITIC_OBS_DIM = 286
_spec.ANCHOR_Z_TERMINATION_THRESHOLD = 0.5
_spec.ANCHOR_ORI_TERMINATION_THRESHOLD = 0.8
_spec.EE_Z_TERMINATION_THRESHOLD = 0.25
_spec.PUSH_INTERVAL_STEP_RANGE = (50, 150)
_spec.VELOCITY_RANGE = ((0.0, 0.0),) * 6
sys.modules["envs.spec"] = _spec
from envs.observation import BEYONDMIMIC_POLICY_OBS_DIM, MimicObservationMixin
from envs.step import MimicStepMixin
from envs.terminal import MimicTerminationMixin
sys.modules.pop("envs.spec", None)

CRITIC_OBS_DIM = 286


def test_evaluator_policy_frame_is_independent_of_add_velocity_modes() -> None:
    class _EvaluatorEnv(MimicObservationMixin):
        pass

    env = _EvaluatorEnv()
    env.num_envs = 2
    env.device = torch.device("cpu")
    env.imitation_key_body_ids = torch.tensor([0, 1, 2, 3, 4])
    joint_pos = torch.randn(2, 29)
    joint_vel = torch.randn(2, 29)
    env.get_action_joint_state = lambda: (joint_pos, joint_vel)
    root_quat = torch.zeros(2, 4)
    root_quat[:, 0] = 1.0
    root_link_velocity = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
        ]
    )
    env.robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=torch.tensor(
                [[10.0, 20.0, 0.8], [30.0, 40.0, 0.9]]
            ),
            root_link_quat_w=root_quat,
            root_link_vel_w=root_link_velocity,
            # Deliberately incompatible COM velocity: method configuration
            # must never select it in the external evaluator.
            root_vel_w=torch.full((2, 6), 999.0),
            body_pos_w=torch.randn(2, 5, 3),
        )
    )
    env.scene = SimpleNamespace(
        env_origins=torch.tensor([[10.0, 20.0, 0.0], [30.0, 40.0, 0.0]])
    )
    env.config = SimpleNamespace(
        root_velocity_mode="com",
        motion_reference_mode="frame",
    )

    frame_default = env.get_evaluator_imitation_policy_frame()
    env.config.root_velocity_mode = "link"
    env.config.motion_reference_mode = "mimickit_add"
    frame_add = env.get_evaluator_imitation_policy_frame()

    torch.testing.assert_close(frame_default, frame_add)
    expected = build_g1_imitation_frame(
        root_pos=env.robot.data.root_link_pos_w - env.scene.env_origins,
        root_quat_wxyz=root_quat,
        joint_pos=joint_pos,
        key_body_pos=env.robot.data.body_pos_w - env.scene.env_origins.unsqueeze(1),
        root_lin_vel=root_link_velocity[:, :3],
        root_ang_vel=root_link_velocity[:, 3:],
        joint_vel=joint_vel,
    )
    torch.testing.assert_close(frame_add, expected)


class _BeyondMimicObservationFixture(MimicObservationMixin):
    def __init__(self) -> None:
        self.num_envs = 2
        self.device = torch.device("cpu")
        self.observation_noise = False
        self.observation_group_size = 1
        self.policy_observation_mode = "beyondmimic"
        self.default_action_joint_pos = torch.arange(29, dtype=torch.float32)
        self.default_action_joint_vel = torch.arange(29, dtype=torch.float32) * 0.5
        self.last_action = torch.arange(58, dtype=torch.float32).reshape(2, 29)
        self.robot = SimpleNamespace(
            data=SimpleNamespace(
                root_ang_vel_b=torch.tensor(
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32
                ),
                root_lin_vel_b=torch.tensor(
                    [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]], dtype=torch.float32
                ),
            )
        )
        self._reference_joint_pos = torch.arange(58, dtype=torch.float32).reshape(2, 29)
        self._reference_joint_vel = self._reference_joint_pos + 100.0
        self._robot_joint_pos = self.default_action_joint_pos + torch.tensor(
            [[1.0], [2.0]]
        )
        self._robot_joint_vel = self.default_action_joint_vel + torch.tensor(
            [[3.0], [4.0]]
        )
        self._anchor_ori = torch.arange(12, dtype=torch.float32).reshape(2, 6) + 200.0
        self._anchor_pos = torch.arange(6, dtype=torch.float32).reshape(2, 3) + 300.0

    def get_tracking_context(self) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(self.num_envs, 1)
        return {
            "reference": {
                "joint_pos": self._reference_joint_pos,
                "joint_vel": self._reference_joint_vel,
                "anchor_pos_w": torch.zeros(self.num_envs, 3),
                "anchor_quat_w": identity,
            },
            "robot_joint_pos": self._robot_joint_pos,
            "robot_joint_vel": self._robot_joint_vel,
            "robot_anchor_pos_w": torch.zeros(self.num_envs, 3),
            "robot_anchor_quat_w": identity,
        }

    def _motion_anchor_observation_terms(self, *_args):
        return self._anchor_pos, self._anchor_ori

    def build_critic_observation(self) -> torch.Tensor:
        return torch.arange(2 * CRITIC_OBS_DIM, dtype=torch.float32).reshape(
            2, CRITIC_OBS_DIM
        )


class _BeyondMimicCachedTargetFixture(MimicObservationMixin):
    def __init__(self) -> None:
        self.num_envs = 1
        self.device = torch.device("cpu")
        self.termination_mode = "beyondmimic"
        self.track_body_names = ["pelvis", "torso_link"]
        self.track_body_ids = torch.tensor([0, 1])
        self.anchor_body_id = 1
        identity = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]])
        self.robot = SimpleNamespace(
            data=SimpleNamespace(
                body_pos_w=torch.tensor([[[2.0, 0.0, 0.0], [2.0, 0.0, 1.0]]]),
                body_quat_w=identity.clone(),
                body_lin_vel_w=torch.zeros(1, 2, 3),
                body_ang_vel_w=torch.zeros(1, 2, 3),
            )
        )
        self._reference = {
            "joint_pos": torch.zeros(1, 2),
            "joint_vel": torch.zeros(1, 2),
            "body_pos_w": torch.tensor([[[10.0, 0.0, 0.0], [10.0, 0.0, 1.0]]]),
            "body_quat_w": identity.clone(),
            "body_lin_vel_w": torch.zeros(1, 2, 3),
            "body_ang_vel_w": torch.zeros(1, 2, 3),
            "anchor_pos_w": torch.tensor([[10.0, 0.0, 1.0]]),
            "anchor_quat_w": identity[:, 0].clone(),
        }
        self._beyondmimic_body_pos_relative_w = torch.full((1, 2, 3), 123.0)
        self._beyondmimic_body_quat_relative_w = torch.full((1, 2, 4), 456.0)

    def get_reference_state(self):
        return self._reference

    def get_action_joint_state(self):
        return torch.zeros(1, 2), torch.zeros(1, 2)


def test_beyondmimic_actor_observation_is_exact_default_160d() -> None:
    env = _BeyondMimicObservationFixture()
    observation = env.get_beyondmimic_policy_observation()

    assert BEYONDMIMIC_POLICY_OBS_DIM == 160
    assert CRITIC_OBS_DIM == 286
    assert observation.shape == (2, 160)
    torch.testing.assert_close(
        observation[:, :58],
        torch.cat([env._reference_joint_pos, env._reference_joint_vel], dim=-1),
    )
    torch.testing.assert_close(observation[:, 58:61], env._anchor_pos)
    torch.testing.assert_close(observation[:, 61:67], env._anchor_ori)
    torch.testing.assert_close(observation[:, 67:70], env.robot.data.root_lin_vel_b)
    torch.testing.assert_close(observation[:, 70:73], env.robot.data.root_ang_vel_b)
    torch.testing.assert_close(
        observation[:, 73:102], env._robot_joint_pos - env.default_action_joint_pos
    )
    torch.testing.assert_close(
        observation[:, 102:131], env._robot_joint_vel - env.default_action_joint_vel
    )
    torch.testing.assert_close(observation[:, 131:], env.last_action)
    torch.testing.assert_close(env.get_observation(), observation)
    torch.testing.assert_close(
        env.get_beyondmimic_policy_observation(torch.tensor([1])), observation[1:2]
    )
    critic = env.get_beyondmimic_critic_observation()
    assert critic.shape == (2, 286)
    torch.testing.assert_close(
        env.get_beyondmimic_critic_observation(torch.tensor([1])), critic[1:2]
    )


def test_beyondmimic_relative_body_targets_are_cached_until_command_advance() -> None:
    env = _BeyondMimicCachedTargetFixture()

    context = env.get_tracking_context()
    torch.testing.assert_close(
        context["body_pos_relative_w"], torch.full((1, 2, 3), 123.0)
    )
    torch.testing.assert_close(
        context["body_quat_relative_w"], torch.full((1, 2, 4), 456.0)
    )

    env._update_beyondmimic_relative_targets()
    expected_pos, expected_quat = env._compute_relative_reference_bodies(
        env._reference,
        env.robot.data.body_pos_w[:, env.anchor_body_id],
        env.robot.data.body_quat_w[:, env.anchor_body_id],
    )
    torch.testing.assert_close(env._beyondmimic_body_pos_relative_w, expected_pos)
    torch.testing.assert_close(env._beyondmimic_body_quat_relative_w, expected_quat)


def test_beyondmimic_sampler_matches_official_failure_plus_uniform_formula() -> None:
    sampler = BeyondMimicAdaptiveSampler(325, "cpu", env_fps=50)
    assert sampler.num_bins == 7
    assert sampler.stats()["bin_count"] == 7.0
    torch.testing.assert_close(
        sampler.sampling_probabilities, torch.full((7,), 1.0 / 7.0)
    )

    sampler.bin_failed_count[3] = 1.0
    score = torch.full((7,), 0.1 / 7.0)
    score[3] += 1.0
    torch.testing.assert_close(sampler.sampling_probabilities, score / score.sum())

    sampler.init_buffers()
    sampler.update_current_failure_count(torch.tensor([0, 46, 47, 324]))
    expected_current = torch.bincount(
        (torch.tensor([0, 46, 47, 324]) * 7) // 325, minlength=7
    ).float()
    torch.testing.assert_close(sampler.current_bin_failed_count, expected_current)
    sampler.update_failure_ema()
    torch.testing.assert_close(sampler.bin_failed_count, expected_current * 0.001)
    torch.testing.assert_close(sampler.current_bin_failed_count, torch.zeros(7))

    torch.manual_seed(17)
    phases = sampler.sample_frames(4096, 0, 323)
    assert phases.dtype == torch.long
    assert int(phases.min()) >= 0
    assert int(phases.max()) <= 323


class _NoContactAccess:
    @property
    def data(self):
        raise AssertionError("BeyondMimic termination must not read contact data")


class _BeyondMimicTerminationFixture(MimicTerminationMixin):
    def __init__(self) -> None:
        self.num_envs = 4
        self.device = torch.device("cpu")
        self.termination_mode = "beyondmimic"
        self.terminate_on_motion_end = True
        self._motion_end_mask = torch.ones(4, dtype=torch.bool)
        self.episode_steps = torch.ones(4, dtype=torch.long)
        self.max_episode_steps = 500
        self.ee_body_indices = [0, 1, 2, 3]
        self.termination_body_indices = [0, 1, 2, 3]
        gravity = torch.tensor([0.0, 0.0, -1.0]).repeat(4, 1)
        self.robot = SimpleNamespace(data=SimpleNamespace(GRAVITY_VEC_W=gravity))
        self.contact_sensor = _NoContactAccess()

        identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(4, 1)
        robot_quat = identity.clone()
        # 1 - cos(theta) = 0.81, just over the official 0.8 threshold.
        theta = math.acos(0.19)
        robot_quat[1] = torch.tensor(
            [math.cos(theta / 2.0), math.sin(theta / 2.0), 0.0, 0.0]
        )
        reference_anchor_pos = torch.zeros(4, 3)
        robot_anchor_pos = torch.zeros(4, 3)
        robot_anchor_pos[0, 2] = 0.251
        robot_anchor_pos[3, 2] = 0.25
        relative_body_pos = torch.zeros(4, 4, 3)
        robot_body_pos = torch.zeros(4, 4, 3)
        relative_body_pos[2, 0, 2] = 0.251
        relative_body_pos[3, 0, 2] = 0.25
        self._context = {
            "reference": {
                "anchor_pos_w": reference_anchor_pos,
                "anchor_quat_w": identity,
            },
            "robot_anchor_pos_w": robot_anchor_pos,
            "robot_anchor_quat_w": robot_quat,
            "body_pos_relative_w": relative_body_pos,
            "robot_body_pos_w": robot_body_pos,
        }

    def get_tracking_context(self):
        return self._context


def test_beyondmimic_termination_is_025_08_025_and_has_no_contact_or_motion_end() -> None:
    env = _BeyondMimicTerminationFixture()
    done, terms, debug = env.compute_termination()

    torch.testing.assert_close(done, torch.tensor([True, True, True, False]))
    torch.testing.assert_close(
        terms["anchor_pos_bad"], torch.tensor([True, False, False, False])
    )
    torch.testing.assert_close(
        terms["anchor_ori_bad"], torch.tensor([False, True, False, False])
    )
    torch.testing.assert_close(
        terms["ee_body_bad"], torch.tensor([False, False, True, False])
    )
    assert not bool(terms["motion_complete"].any())
    assert not bool(terms["fall_contact"].any())
    torch.testing.assert_close(debug["anchor_z_error"][[0, 3]], torch.tensor([0.251, 0.25]))


class _PushRobot:
    def __init__(self) -> None:
        self.written_env_ids = torch.empty(0, dtype=torch.long)

    def write_root_velocity_to_sim(
        self, _velocity: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        self.written_env_ids = env_ids.clone()


class _BeyondMimicPushFixture(MimicStepMixin):
    def __init__(self) -> None:
        self.num_envs = 3
        self.device = torch.device("cpu")
        self.dt = 0.02
        self.interval_pushes = True
        self.beyondmimic_global_push_timer = True
        self._push_interval_time_range = (1.0, 3.0)
        self.push_time_left = torch.tensor([0.02, 0.020002, 0.019999])
        self.episode_steps = torch.tensor([0, 4, 0], dtype=torch.long)
        self.first_push_step = torch.tensor([8, 9, 10], dtype=torch.long)
        self._last_interval_push_mask = torch.zeros(3, dtype=torch.bool)
        self.config = SimpleNamespace(root_velocity_mode="com")
        self.robot = _PushRobot()

    def get_mimic_root_velocity_w(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 6)


def test_beyondmimic_push_timer_is_continuous_across_reset_and_uses_v210_order() -> None:
    env = _BeyondMimicPushFixture()
    timer_before_reset = env.push_time_left.clone()
    env._reset_interval_push_schedule(torch.tensor([0, 2]))
    torch.testing.assert_close(env.push_time_left, timer_before_reset)
    torch.testing.assert_close(env.first_push_step, torch.tensor([-1, 9, -1]))

    # v2.1.0 samples the next interval before push_by_setting_velocity samples
    # its six velocity components. The strict <1e-6 edge leaves env 1 pending.
    torch.manual_seed(123)
    expected_next_intervals = 1.0 + 2.0 * torch.rand(2)
    torch.rand(2, 6)
    torch.manual_seed(123)
    env._apply_interval_pushes()

    torch.testing.assert_close(env.robot.written_env_ids, torch.tensor([0, 2]))
    torch.testing.assert_close(env.push_time_left[[0, 2]], expected_next_intervals)
    assert env.push_time_left[1].item() >= 1.0e-6
    torch.testing.assert_close(env.first_push_step, torch.tensor([0, 9, 0]))


class _FakeScene:
    def __init__(self) -> None:
        self.update_calls = 0

    def update(self, _dt: float) -> None:
        self.update_calls += 1

    def write_data_to_sim(self) -> None:
        raise AssertionError("decimation=0 should not step the simulator")


class _FakeMotion:
    num_frames = 325
    fps = 50.0

    def get_frame(self, phase_indices: torch.Tensor) -> dict[str, torch.Tensor]:
        count = phase_indices.numel()
        phase = phase_indices.float()
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(count, 1)
        return {
            "root_pos_w": torch.stack([phase, phase * 0.0, phase * 0.0], dim=-1),
            "root_quat_w": identity,
            "root_lin_vel_w": torch.zeros(count, 3),
            "root_ang_vel_w": torch.zeros(count, 3),
            "joint_pos": torch.zeros(count, 2),
            "joint_vel": torch.zeros(count, 2),
        }


class _BeyondMimicStepFixture(MimicStepMixin):
    def __init__(self) -> None:
        self.num_envs = 2
        self.device = torch.device("cpu")
        self.action_dim = 2
        self.decimation = 0
        self.render = False
        self.dt = 0.02
        self.physics_dt = 0.005
        self.motion_frame_delta = 1.0
        self.motion = _FakeMotion()
        self.scene = _FakeScene()
        self.phase_steps = torch.tensor([324.0, 100.0])
        self.episode_steps = torch.tensor([7, 9], dtype=torch.long)
        self.episode_ids = torch.tensor([41, 42], dtype=torch.long)
        self.last_action = torch.zeros(2, 2)
        self._failure_recorded = torch.ones(2, dtype=torch.bool)
        self.reset_noise = False
        self.termination_mode = "beyondmimic"
        self.terminate_on_motion_end = False
        self.motion_end_behavior = "resample_command"
        self.written_env_ids = torch.empty(0, dtype=torch.long)
        self.done_mask = torch.zeros(2, dtype=torch.bool)
        self.reward_phase_seen = torch.full((2,), -1.0)
        self.termination_phase_seen = torch.full((2,), -1.0)
        self.observation_calls = 0
        self.critic_observation_calls = 0
        self.relative_target_update_phase_seen = torch.full((2,), -1.0)
        self._last_interval_push_mask = torch.zeros(2, dtype=torch.bool)

    def _apply_action_targets(self, _actions: torch.Tensor) -> None:
        pass

    def sample_phase_indices(self, count: int, horizon: int) -> torch.Tensor:
        assert horizon == 1
        return torch.full((count,), 7, dtype=torch.long)

    def _resample_finished_motions(self):
        env_ids = torch.where(self.phase_steps >= self.motion.num_frames)[0]
        phase_indices = self.sample_phase_indices(env_ids.numel(), horizon=1)
        self.phase_steps[env_ids] = phase_indices.float()
        self._failure_recorded[env_ids] = False
        reference = self.motion.get_frame(phase_indices)
        self._write_robot_state(env_ids=env_ids, **reference)
        self.scene.update(self.physics_dt)
        return env_ids, phase_indices

    def _write_robot_state(self, *, env_ids: torch.Tensor, **_kwargs) -> None:
        self.written_env_ids = env_ids.clone()

    def compute_reward(self, _actions: torch.Tensor, _previous_action: torch.Tensor):
        self.reward_phase_seen = self.phase_steps.clone()
        return torch.zeros(2), {}

    def compute_termination(self):
        self.termination_phase_seen = self.phase_steps.clone()
        zeros = torch.zeros(2, dtype=torch.bool)
        return self.done_mask.clone(), {
            "time_out": zeros,
            "motion_complete": zeros,
            "anchor_pos_bad": self.done_mask.clone(),
            "anchor_ori_bad": zeros,
            "ee_body_bad": zeros,
        }, {}

    def get_imitation_policy_frame(self):
        return torch.zeros(2, 1)

    def get_amp_policy_observation(self):
        return torch.zeros(2, 1)

    def get_add_policy_disc_frame(self):
        return torch.zeros(2, 1)

    def get_add_demo_disc_frame(self, _phase_steps: torch.Tensor):
        return torch.zeros(2, 1)

    def get_add_policy_observation(self):
        return torch.zeros(2, 1)

    def get_observation(self):
        self.observation_calls += 1
        return torch.cat([self.phase_steps[:, None], self.last_action], dim=-1)

    def get_critic_observation(self):
        self.critic_observation_calls += 1
        return torch.zeros(2, 1)

    def _reset_env_state(
        self, env_ids: torch.Tensor, phase_indices: torch.Tensor | None = None
    ) -> None:
        assert phase_indices is not None
        self.phase_steps[env_ids] = phase_indices.float()
        self.episode_steps[env_ids] = 0
        self.episode_ids[env_ids] += 100
        self.last_action[env_ids] = 0.0
        self._failure_recorded[env_ids] = False

    def _record_adaptive_failures(self, *_args) -> None:
        pass

    def _fold_adaptive_sampler(self) -> None:
        pass

    def _apply_interval_pushes(self) -> None:
        pass

    def _update_beyondmimic_relative_targets(self) -> None:
        self.relative_target_update_phase_seen = self.phase_steps.clone()


def test_motion_end_resamples_command_without_resetting_episode_or_action_history() -> None:
    env = _BeyondMimicStepFixture()
    episode_steps = env.episode_steps.clone()
    episode_ids = env.episode_ids.clone()
    actions = torch.tensor([[0.3, 0.4], [0.5, 0.6]])

    _observation, _reward, done, info = env.step(actions)

    assert not bool(done.any())
    torch.testing.assert_close(env.phase_steps, torch.tensor([7.0, 101.0]))
    torch.testing.assert_close(env.episode_steps, episode_steps + 1)
    torch.testing.assert_close(env.episode_ids, episode_ids)
    torch.testing.assert_close(env.last_action, actions)
    torch.testing.assert_close(info["motion_wrap_env_ids"], torch.tensor([0]))
    torch.testing.assert_close(info["motion_wrap_phase_indices"], torch.tensor([7]))
    torch.testing.assert_close(
        info["motion_resample_mask"], torch.tensor([True, False])
    )
    torch.testing.assert_close(env.written_env_ids, torch.tensor([0]))
    torch.testing.assert_close(info["reference_phase_steps"], torch.tensor([324.0, 100.0]))
    torch.testing.assert_close(env.reward_phase_seen, torch.tensor([324.0, 100.0]))
    torch.testing.assert_close(env.termination_phase_seen, torch.tensor([324.0, 100.0]))
    torch.testing.assert_close(env.relative_target_update_phase_seen, torch.tensor([7.0, 101.0]))
    assert env.scene.update_calls == 1


def test_beyondmimic_done_reset_precedes_command_advance_and_samples_no_terminal_obs() -> None:
    env = _BeyondMimicStepFixture()
    env.phase_steps = torch.tensor([50.0, 100.0])
    env.done_mask[0] = True
    old_episode_ids = env.episode_ids.clone()
    actions = torch.tensor([[0.3, 0.4], [0.5, 0.6]])

    observation, _reward, done, info = env.step(actions, auto_reset=True)

    torch.testing.assert_close(done, torch.tensor([True, False]))
    torch.testing.assert_close(env.reward_phase_seen, torch.tensor([50.0, 100.0]))
    torch.testing.assert_close(info["termination_phase_steps"], torch.tensor([50.0, 100.0]))
    torch.testing.assert_close(info["reset_phase_indices"], torch.tensor([7]))
    # CommandManager.compute runs after the reset, so the newly sampled
    # command at frame 7 becomes frame 8 in the returned observation.
    torch.testing.assert_close(env.phase_steps, torch.tensor([8.0, 101.0]))
    torch.testing.assert_close(observation[:, 0], torch.tensor([8.0, 101.0]))
    assert env.episode_steps[0].item() == 0
    assert env.episode_ids[0].item() == old_episode_ids[0].item() + 100
    torch.testing.assert_close(env.last_action[0], torch.zeros(2))
    torch.testing.assert_close(env.last_action[1], actions[1])
    torch.testing.assert_close(env.relative_target_update_phase_seen, torch.tensor([8.0, 101.0]))
    # Official RSL path only generates the returned actor observation.
    assert env.observation_calls == 1
    assert env.critic_observation_calls == 0


def test_common_validation_profile_disables_beyondmimic_command_resampling() -> None:
    env = _BeyondMimicStepFixture()
    env.termination_mode = "amp"
    env.terminate_on_motion_end = True

    _observation, _reward, _done, info = env.step(torch.zeros(2, 2))

    # The common benchmark evaluator intentionally uses its post-transition
    # reference and motion-end terminal, even for a BeyondMimic-trained policy.
    torch.testing.assert_close(info["reference_phase_steps"], torch.tensor([324.0, 101.0]))
    torch.testing.assert_close(env.phase_steps, torch.tensor([324.0, 101.0]))
    assert info["motion_wrap_env_ids"].numel() == 0
    assert not bool(info["motion_resample_mask"].any())


def test_inference_rollout_keeps_last_action_resettable() -> None:
    env = _BeyondMimicStepFixture()
    with torch.inference_mode():
        env.step(torch.ones(2, 2))

    assert not torch.is_inference(env.last_action)
    env.last_action.zero_()
