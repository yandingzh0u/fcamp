from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("isaaclab")

from engine.checkpoint import CheckpointMixin
from engine.logging import LoggingMixin
from engine.mixgrpo.config import MixGRPOConfig
from engine.mixgrpo.inference import deterministic_sde_ode_actions
from engine.mixgrpo.sampling import flow_grpo_step, flow_sde_transition
from engine.mixgrpo.trainer import MixGRPOTrainer
from engine.ppo.config import OfficialPPOConfig
from engine.ppo.trainer import OfficialPPOTrainer
from engine.validation import ValidationMixin, _short_body_name
from net.ppo import GaussianActorCritic


def test_config_defaults_match_training_contract() -> None:
    cfg = MixGRPOConfig(device="cpu")

    assert cfg.chunks_per_rollout == 4
    assert cfg.flow_steps == 4
    assert cfg.actor_hidden_dims == (512, 256, 128)
    assert cfg.critic_hidden_dims == (512, 256, 128)
    assert cfg.activation == "elu"
    assert cfg.action_squash_scale == pytest.approx(5.0)
    assert cfg.init_noise_std == pytest.approx(0.8)
    assert cfg.init_same_noise is False
    assert cfg.eval_initial_noise == "zero"
    assert cfg.sde_eta == pytest.approx(0.7)
    assert cfg.clip_range == pytest.approx(0.3)
    assert cfg.rollout_segments_per_update == 32
    assert cfg.adv_clip_max == pytest.approx(5.0)
    assert cfg.desired_kl == pytest.approx(0.03)
    assert cfg.entropy_coef == pytest.approx(0.005)
    assert cfg.value_loss_coef == pytest.approx(0.0)
    assert cfg.use_clipped_value_loss is True
    assert cfg.gae_lambda == pytest.approx(0.95)
    assert cfg.policy_epochs == 5
    assert cfg.lr == pytest.approx(1.0e-3)
    assert cfg.policy_lr == pytest.approx(1.0e-3)
    assert cfg.save_every == 500
    assert cfg.max_updates == 30000
    assert cfg.max_episode_steps == 1500
    assert cfg.startup_randomization is True
    assert cfg.reset_noise is True
    assert cfg.interval_pushes is True
    assert cfg.validation_every == 0
    assert cfg.validation_fixed_seed == -1
    assert cfg.validation_preserve_state is True
    assert cfg.reset_optimizer_on_resume is False
    assert cfg.horizon == 1
    assert cfg.action_dim == 29
    assert cfg.num_generations == 4
    assert cfg.terminal_penalty == pytest.approx(50.0)
    assert cfg.debug_probe is False
    assert cfg.debug_probe_every == 1
    assert Path(cfg.motion_file).is_file()


def test_official_ppo_config_defaults_match_unitree_rsl_cfg() -> None:
    cfg = OfficialPPOConfig(device="cpu")

    assert cfg.num_steps_per_env == 24
    assert cfg.max_updates == 30000
    assert cfg.save_every == 500
    assert cfg.empirical_normalization is False
    assert cfg.actor_hidden_dims == (512, 256, 128)
    assert cfg.critic_hidden_dims == (512, 256, 128)
    assert cfg.activation == "elu"
    assert cfg.init_noise_std == pytest.approx(1.0)
    assert cfg.value_loss_coef == pytest.approx(1.0)
    assert cfg.use_clipped_value_loss is True
    assert cfg.clip_range == pytest.approx(0.2)
    assert cfg.entropy_coef == pytest.approx(0.005)
    assert cfg.policy_epochs == 5
    assert cfg.num_mini_batches == 4
    assert cfg.lr == pytest.approx(1.0e-3)
    assert cfg.schedule == "adaptive"
    assert cfg.discount_gamma == pytest.approx(0.99)
    assert cfg.gae_lambda == pytest.approx(0.95)
    assert cfg.desired_kl == pytest.approx(0.01)
    assert cfg.max_grad_norm == pytest.approx(1.0)
    assert cfg.horizon == 1
    assert cfg.action_dim == 29
    assert Path(cfg.motion_file).is_file()


def test_gaussian_actor_critic_matches_rsl_shapes_and_logprob() -> None:
    policy = GaussianActorCritic(
        obs_dim=4,
        critic_obs_dim=5,
        action_dim=3,
        actor_hidden_dims=(8,),
        critic_hidden_dims=(8,),
        activation="elu",
        init_noise_std=1.0,
    )
    obs = torch.zeros(7, 4)
    critic_obs = torch.zeros(7, 5)

    sample = policy.act(obs, critic_obs)
    log_prob, values, entropy, mean, sigma = policy.evaluate_actions(obs, critic_obs, sample["actions"])

    assert sample["actions"].shape == (7, 3)
    assert sample["values"].shape == (7,)
    assert sample["log_probs"].shape == (7,)
    assert log_prob.shape == (7,)
    assert values.shape == (7,)
    assert entropy.shape == (7,)
    assert mean.shape == (7, 3)
    assert sigma.shape == (7, 3)
    assert torch.allclose(sigma, torch.ones_like(sigma))


def test_policy_mini_batch_size_matches_rsl_rl_floor_division() -> None:
    trainer = MixGRPOTrainer.__new__(MixGRPOTrainer)
    trainer.cfg = MixGRPOConfig(device="cpu", num_mini_batches=4, mini_batch_size=0)

    assert trainer._policy_mini_batch_size(98_304) == 24_576
    assert trainer._policy_mini_batch_size(10) == 2

    ppo_trainer = OfficialPPOTrainer.__new__(OfficialPPOTrainer)
    ppo_trainer.cfg = OfficialPPOConfig(device="cpu", num_mini_batches=4)
    assert ppo_trainer._policy_mini_batch_size(98_304) == 24_576
    assert ppo_trainer._policy_mini_batch_size(10) == 2


def test_mixgrpo_update_horizon_accumulates_multiple_rollout_segments() -> None:
    trainer = MixGRPOTrainer.__new__(MixGRPOTrainer)
    trainer.cfg = MixGRPOConfig(device="cpu", chunks_per_rollout=4, rollout_segments_per_update=8, horizon=1)

    assert trainer._chunks_per_grpo_update() == 32
    assert trainer._training_rollout_horizon() == 32


def test_mixgrpo_adaptive_lr_updates_policy_group_only() -> None:
    trainer = MixGRPOTrainer.__new__(MixGRPOTrainer)
    trainer.cfg = MixGRPOConfig(device="cpu", policy_lr=3.0e-4, lr=1.0e-3, desired_kl=0.01)
    trainer.policy = torch.nn.Linear(2, 2)
    trainer.critic = torch.nn.Linear(2, 1)
    trainer.optimizer = torch.optim.AdamW(
        [
            {"params": trainer.policy.parameters(), "lr": trainer.cfg.policy_lr},
            {"params": trainer.critic.parameters(), "lr": trainer.cfg.lr},
        ]
    )
    trainer.learning_rate = float(trainer.cfg.policy_lr)

    trainer._update_adaptive_learning_rate(0.03)

    assert trainer.learning_rate == pytest.approx(2.0e-4)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(2.0e-4)
    assert trainer.optimizer.param_groups[1]["lr"] == pytest.approx(1.0e-3)

    trainer._update_adaptive_learning_rate(torch.tensor(0.001))

    assert trainer.learning_rate == pytest.approx(3.0e-4)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(3.0e-4)


def test_compute_path_log_probs_matches_sde_transition_score() -> None:
    trainer = MixGRPOTrainer.__new__(MixGRPOTrainer)
    trainer.cfg = MixGRPOConfig(device="cpu", flow_steps=2, sde_eta=0.3, init_noise_std=0.5)
    trainer.chunk_dim = 2
    velocity = torch.tensor([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.25]])
    seen_times: list[torch.Tensor] = []

    class _Policy:
        def _prepare_observation(self, observation: torch.Tensor) -> torch.Tensor:
            return observation

        def velocity_field(self, observation: torch.Tensor, noisy_actions: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            del observation, noisy_actions
            seen_times.append(time.detach().clone())
            return velocity

    trainer.policy = _Policy()
    obs = torch.zeros(3, 4)
    latent_t = torch.tensor([[1.0, -2.0], [0.5, 0.25], [-0.3, 0.7]])
    sigma_schedule = torch.linspace(1.0, 0.0, 3)
    mean_next, expected_log_probs = flow_grpo_step(
        model_output=velocity,
        latents=latent_t,
        sigmas=sigma_schedule,
        index=0,
        eta=0.3,
        prev_sample=None,
        deterministic=False,
        sample_noise=torch.zeros_like(latent_t),
    )
    recorded_next = mean_next + torch.tensor([[0.1, -0.2], [0.3, 0.0], [-0.4, 0.2]])
    latent_path = torch.stack([latent_t, recorded_next, torch.zeros_like(latent_t)], dim=1)
    expected_log_probs = flow_grpo_step(
        model_output=velocity,
        latents=latent_t,
        sigmas=sigma_schedule,
        index=0,
        eta=0.3,
        prev_sample=recorded_next,
        deterministic=False,
    )[1]

    log_probs, kl = trainer._compute_path_log_probs_and_kl(obs, latent_path, torch.tensor([0]))

    assert torch.allclose(log_probs.squeeze(-1), expected_log_probs)
    assert torch.equal(kl, torch.zeros_like(log_probs))
    assert torch.allclose(seen_times[0], torch.ones(3))


def test_compute_gae_returns_matches_official_rsl_rl_formula() -> None:
    trainer = MixGRPOTrainer.__new__(MixGRPOTrainer)
    trainer.cfg = MixGRPOConfig(device="cpu", discount_gamma=0.5, gae_lambda=0.25)

    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    dones = torch.tensor([[False, True, False]])
    values = torch.tensor([[0.1, 0.2, 0.3]])
    last_values = torch.tensor([0.4])

    returns, advantages = trainer._compute_gae_returns(rewards, dones, values, last_values)

    raw_returns = torch.empty_like(rewards)
    advantage = torch.zeros(1)
    for step in reversed(range(3)):
        next_value = last_values if step == 2 else values[:, step + 1]
        next_is_not_terminal = 1.0 - dones[:, step].float()
        delta = rewards[:, step] + 0.5 * next_is_not_terminal * next_value - values[:, step]
        advantage = delta + 0.5 * 0.25 * next_is_not_terminal * advantage
        raw_returns[:, step] = advantage + values[:, step]
    expected_advantages = raw_returns - values
    expected_advantages = (expected_advantages - expected_advantages.mean()) / (expected_advantages.std() + 1e-8)

    assert torch.allclose(returns, raw_returns)
    assert torch.allclose(advantages, expected_advantages)


def test_flow_grpo_step_sde_transition_with_recorded_next_sample() -> None:
    latents = torch.tensor([[1.0, -2.0], [0.5, 0.25]])
    model_output = torch.tensor([[0.2, -0.4], [1.0, -2.0]])
    sigmas = torch.tensor([1.0, 0.5, 0.0])
    eta = 0.3
    expected_mean, std, _ = flow_sde_transition(model_output, latents, sigmas, index=0, eta=eta)
    recorded_next = expected_mean + torch.tensor([[1.0, -1.0], [2.0, 0.0]])

    next_sample, log_prob = flow_grpo_step(
        model_output=model_output,
        latents=latents,
        sigmas=sigmas,
        index=0,
        eta=eta,
        prev_sample=recorded_next,
    )

    residual = recorded_next - expected_mean
    expected_log_prob = -residual.square() / (2.0 * std.square())
    expected_log_prob = expected_log_prob - torch.log(std) - 0.5 * math.log(2.0 * math.pi)
    expected_log_prob = expected_log_prob.sum(dim=-1)
    assert torch.allclose(next_sample, recorded_next)
    assert torch.allclose(log_prob, expected_log_prob)


def test_flow_grpo_step_deterministic_uses_ode_update() -> None:
    latents = torch.tensor([[1.0, 2.0, 3.0]])
    model_output = torch.tensor([[0.5, -1.0, 2.0]])
    sigmas = torch.tensor([0.75, 0.25])
    eta = 0.3

    next_sample, log_prob = flow_grpo_step(
        model_output=model_output,
        latents=latents,
        sigmas=sigmas,
        index=0,
        eta=eta,
        deterministic=True,
    )

    expected = latents + (sigmas[1] - sigmas[0]) * model_output
    assert torch.allclose(next_sample, expected)
    expected_mean, std, _ = flow_sde_transition(model_output, latents, sigmas, index=0, eta=eta)
    residual = expected - expected_mean
    expected_log_prob = -residual.square() / (2.0 * std.square())
    expected_log_prob = expected_log_prob - torch.log(std) - 0.5 * math.log(2.0 * math.pi)
    expected_log_prob = expected_log_prob.sum(dim=-1)
    assert torch.allclose(log_prob, expected_log_prob)


def test_deterministic_sde_ode_actions_matches_zero_noise_sde_mean_path() -> None:
    seen_times: list[torch.Tensor] = []

    class _Policy:
        chunk_dim = 2
        horizon = 1
        action_dim = 2

        def _validate_inputs(self, observation: torch.Tensor, flow_noise: torch.Tensor, steps: int) -> None:
            assert observation.shape[0] == flow_noise.shape[0]
            assert steps == 2

        def _prepare_observation(self, observation: torch.Tensor) -> torch.Tensor:
            return observation

        def velocity_field(self, observation: torch.Tensor, noisy_actions: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
            del observation
            seen_times.append(time.detach().clone())
            return torch.ones_like(noisy_actions) * 0.25

        def _action_transform(self, action_value: torch.Tensor) -> torch.Tensor:
            return action_value

    policy = _Policy()
    observation = torch.zeros(3, 4)
    actions = deterministic_sde_ode_actions(policy, observation, steps=2, sde_eta=0.3)

    sigma_schedule = torch.linspace(1.0, 0.0, 3)
    latent = torch.zeros(3, 2)
    for step_index in range(2):
        latent, _ = flow_grpo_step(
            model_output=torch.full_like(latent, 0.25),
            latents=latent,
            sigmas=sigma_schedule,
            index=step_index,
            eta=0.3,
            deterministic=False,
            sample_noise=torch.zeros_like(latent),
        )

    assert torch.allclose(actions, latent.view(3, 1, 2))
    assert torch.allclose(seen_times[0], torch.ones(3))
    assert torch.allclose(seen_times[1], torch.full((3,), 0.5))


def test_collect_rollout_records_sde_step_log_probs_without_critic_values() -> None:
    trainer = MixGRPOTrainer.__new__(MixGRPOTrainer)
    trainer.cfg = SimpleNamespace(
        chunks_per_rollout=1,
        horizon=1,
        flow_steps=4,
        action_dim=2,
        num_generations=1,
    )
    trainer.chunk_dim = 2
    trainer.env = SimpleNamespace(num_envs=2, device=torch.device("cpu"))
    step_auto_reset_flags: list[bool] = []

    def _step(actions: torch.Tensor, auto_reset: bool = False, reset_horizon: int = 1):
        del actions
        step_auto_reset_flags.append(auto_reset)
        return (
            torch.full((2, 3), 5.0),
            torch.ones(2),
            torch.zeros(2, dtype=torch.bool),
            {
                "done_terms": {
                    "time_out": torch.zeros(2, dtype=torch.bool),
                    "anchor_pos_bad": torch.zeros(2, dtype=torch.bool),
                    "anchor_ori_bad": torch.zeros(2, dtype=torch.bool),
                    "ee_body_bad": torch.zeros(2, dtype=torch.bool),
                },
                "reward_terms": {},
                "debug_terms": {},
                "termination_phase_steps": torch.zeros(2, dtype=torch.long),
            },
        )

    trainer.env.step = _step
    trainer.env.get_observation = lambda: (_ for _ in ()).throw(AssertionError("unexpected get_observation"))

    def _sample(
        obs: torch.Tensor,
        noise: torch.Tensor,
        sde_noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del obs, noise, sde_noise
        return {
            "actions": torch.zeros(2, 1, 2),
            "all_latents": torch.zeros(2, 5, 2),
            "log_probs": torch.full((2, 2), -0.25),
            "train_step_indices": torch.tensor([0, 1]),
            "sigma_schedule": torch.linspace(1.0, 0.0, 5),
        }

    trainer._sde_sample_with_logprobs = _sample

    data = trainer._collect_rollout(torch.zeros(2, 3))

    assert data["old_log_probs"].shape == (2, 1, 1, 2)
    assert torch.allclose(data["old_log_probs"], torch.full((2, 1, 1, 2), -0.25))
    assert torch.equal(data["train_step_indices"], torch.tensor([0, 1]))
    assert torch.equal(data["next_observation"], torch.full((2, 3), 5.0))
    assert data["critic_obs"].shape == (2, 1, 1, 0)
    assert torch.equal(data["values"].squeeze(1), torch.zeros(2, 1))
    assert step_auto_reset_flags == [True]


def test_collect_rollout_uses_post_rollout_observation_without_extra_reset() -> None:
    trainer = MixGRPOTrainer.__new__(MixGRPOTrainer)
    trainer.cfg = SimpleNamespace(
        chunks_per_rollout=1,
        horizon=1,
        flow_steps=1,
        action_dim=2,
        num_generations=1,
    )
    trainer.chunk_dim = 2
    trainer.env = SimpleNamespace(num_envs=2, device=torch.device("cpu"))
    trainer.env.get_critic_observation = lambda: torch.zeros(2, 4)
    trainer.critic = lambda critic_obs: torch.zeros(critic_obs.shape[0])
    trainer.env.step = lambda actions, auto_reset=True, reset_horizon=1: (
        torch.tensor([[7.0, 7.0, 7.0], [8.0, 8.0, 8.0]]),
        torch.ones(2),
        torch.tensor([True, False]),
        {
            "done_terms": {
                "time_out": torch.zeros(2, dtype=torch.bool),
                "anchor_pos_bad": torch.tensor([True, False]),
                "anchor_ori_bad": torch.zeros(2, dtype=torch.bool),
                "ee_body_bad": torch.zeros(2, dtype=torch.bool),
            },
            "reward_terms": {},
            "debug_terms": {},
            "termination_phase_steps": torch.zeros(2, dtype=torch.long),
        },
    )
    trainer.env.get_observation = lambda: (_ for _ in ()).throw(AssertionError("unexpected get_observation"))
    trainer.env.reset_envs = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected reset"))

    trainer._sde_sample_with_logprobs = lambda obs, noise, sde_noise=None: {
        "actions": torch.zeros(2, 1, 2),
        "all_latents": torch.zeros(2, 2, 2),
        "log_probs": torch.full((2, 1), -0.25),
        "train_step_indices": torch.tensor([0]),
        "sigma_schedule": torch.linspace(1.0, 0.0, 2),
    }

    data = trainer._collect_rollout(torch.zeros(2, 3))

    assert torch.equal(data["next_observation"], torch.tensor([[7.0, 7.0, 7.0], [8.0, 8.0, 8.0]]))
    assert torch.equal(data["first_done_chunk"].squeeze(1), torch.tensor([0, 1]))


def test_replicate_group_state_does_not_reset_source_branches() -> None:
    trainer = MixGRPOTrainer.__new__(MixGRPOTrainer)

    class _Scene:
        def __init__(self):
            self.env_origins = torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [20.0, 0.0, 0.0],
                    [30.0, 0.0, 0.0],
                ]
            )
            self.reset_env_ids = None

        def reset(self, env_ids):
            self.reset_env_ids = env_ids.clone()

        def update(self, dt):
            self.updated_dt = dt

    class _RootPhysxView:
        def __init__(self):
            self.coms = torch.arange(12, dtype=torch.float32).view(4, 3)
            self.materials = torch.arange(8, dtype=torch.float32).view(4, 2)
            self.com_set_ids = None
            self.material_set_ids = None

        def get_coms(self):
            return self.coms

        def set_coms(self, coms, env_ids):
            self.coms = coms
            self.com_set_ids = env_ids.clone()

        def get_material_properties(self):
            return self.materials

        def set_material_properties(self, materials, env_ids):
            self.materials = materials
            self.material_set_ids = env_ids.clone()

    root_state = torch.zeros(4, 13)
    root_state[:, :3] = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [11.0, 12.0, 13.0],
            [24.0, 5.0, 6.0],
            [31.0, 32.0, 33.0],
        ]
    )
    root_state[:, 3] = 1.0
    joint_pos = torch.arange(12, dtype=torch.float32).view(4, 3)
    joint_vel = joint_pos + 100.0
    scene = _Scene()
    root_physx_view = _RootPhysxView()

    env = SimpleNamespace(
        num_envs=4,
        device=torch.device("cpu"),
        physics_dt=0.02,
        scene=scene,
        action_joint_ids=torch.tensor([0, 2]),
        robot=SimpleNamespace(
            data=SimpleNamespace(root_state_w=root_state, joint_pos=joint_pos, joint_vel=joint_vel),
            root_physx_view=root_physx_view,
        ),
        default_root_state=torch.arange(52, dtype=torch.float32).view(4, 13).clone(),
        default_joint_pos=torch.arange(12, dtype=torch.float32).view(4, 3).clone(),
        default_joint_vel=torch.arange(12, dtype=torch.float32).view(4, 3).clone() + 10.0,
        default_action_joint_pos=torch.arange(12, dtype=torch.float32).view(4, 3).clone() + 20.0,
        default_action_joint_vel=torch.arange(12, dtype=torch.float32).view(4, 3).clone() + 30.0,
        phase_steps=torch.tensor([10, 99, 20, 88]),
        episode_steps=torch.tensor([3, 9, 7, 11]),
        last_action=torch.arange(8, dtype=torch.float32).view(4, 2),
        next_push_step=torch.tensor([100, 900, 200, 800]),
    )
    write_call = {}

    def _write_robot_state(**kwargs):
        write_call.update({key: value.clone() if torch.is_tensor(value) else value for key, value in kwargs.items()})

    env._write_robot_state = _write_robot_state
    trainer.env = env

    trainer._replicate_group_reset_state(group_count=2, generation_count=2)

    assert torch.equal(scene.reset_env_ids, torch.tensor([1, 3]))
    assert torch.equal(write_call["env_ids"], torch.tensor([1, 3]))
    assert torch.allclose(write_call["root_pos"], torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert torch.equal(env.phase_steps, torch.tensor([10, 10, 20, 20]))
    assert torch.equal(env.episode_steps, torch.tensor([3, 3, 7, 7]))
    assert torch.equal(env.last_action, torch.tensor([[0.0, 1.0], [0.0, 1.0], [4.0, 5.0], [4.0, 5.0]]))
    assert torch.equal(env.next_push_step, torch.tensor([100, 100, 200, 200]))
    assert torch.equal(root_physx_view.com_set_ids, torch.tensor([1, 3]))
    assert torch.equal(root_physx_view.material_set_ids, torch.tensor([1, 3]))


@dataclass
class _CheckpointConfig:
    lr: float = 1e-3
    target_validation_steps: int = 10
    max_episode_steps: int = 100


class _CheckpointHarness(CheckpointMixin):
    def __init__(self, checkpoint_dir: Path):
        self.cfg = _CheckpointConfig()
        self.checkpoint_dir = checkpoint_dir
        self.policy = torch.nn.Linear(3, 2)
        self.critic = torch.nn.Linear(4, 1)
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.critic.parameters()),
            lr=self.cfg.lr,
        )
        self.env = SimpleNamespace(device=torch.device("cpu"))
        self.start_update = 1


def test_checkpoint_save_load_and_target_gate(tmp_path: Path) -> None:
    trainer = _CheckpointHarness(tmp_path)
    metrics = {"validation/steps_min": 11.0, "val_fixed/steps_min": 11.0}

    assert trainer._target_validation_reached(metrics)
    assert not trainer._target_validation_reached({"validation/steps_min": 10.0, "val_fixed/steps_min": 11.0})
    trainer._save_checkpoint(7, {"score": 1.25})

    step_path = tmp_path / "update_0007.pt"
    last_path = tmp_path / "last.pt"
    assert step_path.is_file()
    assert last_path.is_file()

    restored = _CheckpointHarness(tmp_path)
    restored.cfg.lr = 5e-4
    restored._load_checkpoint(step_path)

    assert restored.start_update == 8
    assert restored.optimizer.param_groups[0]["lr"] == pytest.approx(5e-4)
    assert torch.allclose(restored.critic.weight, trainer.critic.weight)


class _LoggingHarness(LoggingMixin):
    def __init__(self):
        self.cfg = SimpleNamespace(max_updates=3)
        self.env = SimpleNamespace(ee_body_names=["left_wrist_yaw_link"])


def test_logging_update_emits_expected_sections(capsys: pytest.CaptureFixture[str]) -> None:
    harness = _LoggingHarness()
    metrics = {
        "algo/name": "mixgrpo",
        "group/reward_mean": 1.0,
        "group/reward_std": 0.5,
        "rollout/chunk_return_mean": 0.25,
        "policy/loss": 0.1,
        "policy/policy_loss": 0.2,
        "policy/ratio": 1.0,
        "policy/clip_frac": 0.0,
        "policy/grad_norm": 0.3,
        "timing/collect_s": 0.4,
        "timing/update_s": 0.5,
    }

    harness._log_update(1, metrics)
    out = capsys.readouterr().out

    assert "[UPDATE] 1/3" in out
    assert "[POLICY]" in out
    assert "[TRACK_CHUNK0]" in out
    assert "[DONE]" in out


def test_logging_update_emits_ppo_specific_sections(capsys: pytest.CaptureFixture[str]) -> None:
    harness = _LoggingHarness()
    metrics = {
        "algo/name": "ppo",
        "group/objective_reward_raw_mean": 1.0,
        "group/objective_reward_raw_std": 0.1,
        "group/objective_reward_raw_min": 0.8,
        "group/objective_reward_raw_max": 1.2,
        "group/reward_raw_mean": 1.5,
        "group/reward_raw_std": 0.2,
        "group/reward_raw_min": 1.1,
        "group/reward_raw_max": 1.8,
        "rollout/chunk_return_mean": 0.25,
        "rollout/max_episode_return_projection": 10.0,
        "rollout/return_mean": 1.0,
        "rollout/raw_return_mean": 1.5,
        "policy/loss": 0.1,
        "policy/policy_loss": 0.2,
        "policy/ratio": 1.0,
        "policy/clip_frac": 0.0,
        "policy/grad_norm": 0.3,
        "timing/collect_s": 0.4,
        "timing/update_s": 0.5,
    }

    harness._log_update(2, metrics)
    out = capsys.readouterr().out

    assert "[PPO_UPDATE] 2/3" in out
    assert "[PPO_POLICY]" in out
    assert "[PPO_FIRST_FAILURE]" in out
    assert "[TRAIN_BODY]" in out
    assert "group_reward" not in out
    assert "group_std" not in out


def test_logging_validation_metrics_emits_without_full_update(capsys: pytest.CaptureFixture[str]) -> None:
    harness = _LoggingHarness()
    metrics = {
        "validation/steps_mean": 12.0,
        "validation/steps_min": 5.0,
        "validation/steps_max": 20.0,
        "validation/return_mean": 1.5,
        "validation/done_frac": 0.25,
    }

    harness._log_validation_metrics(metrics)
    out = capsys.readouterr().out

    assert "[VAL] steps_mean=12.00" in out
    assert "[UPDATE]" not in out


def test_trainer_build_metrics_uses_recorded_rollout_actions_and_valid_samples() -> None:
    trainer = MixGRPOTrainer.__new__(MixGRPOTrainer)
    trainer.cfg = SimpleNamespace(horizon=4, max_episode_steps=1500)
    trainer.env = SimpleNamespace(dt=0.02)

    valid_mask = torch.tensor(
        [
            [[True, False], [True, True]],
            [[False, False], [True, True]],
        ]
    )
    advantages = torch.arange(1, 9, dtype=torch.float32).view(2, 2, 2)

    latents = torch.zeros(2, 2, 2, 5, 4)
    final_latents = latents[..., -1, :]
    final_latents[:] = 100.0
    final_latents[valid_mask] = 2.0

    metric_actions_first = torch.full((2, 4, 29), 0.25)
    metric_actions_last = torch.full((2, 4, 29), 0.75)
    metric_chunk_return_first = torch.tensor([1.0, 3.0])
    metric_chunk_return_last = torch.tensor([5.0, 7.0])

    group_data = {
        "rewards": torch.tensor([[10.0, 20.0], [30.0, 40.0]]),
        "metric_chunk_return": metric_chunk_return_first,
        "metric_chunk_return_first": metric_chunk_return_first,
        "metric_chunk_return_last": metric_chunk_return_last,
        "metric_actions": metric_actions_first,
        "metric_actions_first": metric_actions_first,
        "metric_actions_last": metric_actions_last,
        "latents": latents,
        "valid_mask": valid_mask,
        "metric_infos_list": [
            {
                "reward_terms": {
                    "action_rate": torch.tensor([2.0, 2.0]),
                    "anchor_pos_reward": torch.tensor([1.0, 1.0]),
                },
                "done_terms": {"anchor_pos_bad": torch.tensor([False, False])},
            },
            {
                "reward_terms": {
                    "action_rate": torch.tensor([6.0, 6.0]),
                    "anchor_pos_reward": torch.tensor([1.0, 1.0]),
                },
                "done_terms": {"anchor_pos_bad": torch.tensor([True, False])},
            },
        ],
    }

    metrics = trainer._build_metrics(group_data, advantages, {"policy/loss": 0.0}, 0.1, 0.2)

    assert metrics["group/reward_mean"] == pytest.approx(7500.0)
    assert metrics["group/reward_raw_mean"] == pytest.approx(25.0)
    assert metrics["rollout/return_mean"] == pytest.approx(25.0)
    assert metrics["rollout/chunk_return_mean"] == pytest.approx(2.0)
    assert metrics["rollout/chunk_return_last_mean"] == pytest.approx(6.0)
    assert metrics["rollout/live_steps_mean"] == pytest.approx(5.0)
    assert metrics["rollout/reward_per_live_step"] == pytest.approx(5.0)
    assert metrics["rollout/reward_per_live_second"] == pytest.approx(250.0)
    assert metrics["rollout/max_episode_return_projection"] == pytest.approx(7500.0)
    assert metrics["act/abs_mean"] == pytest.approx(0.25)
    assert metrics["act/first_abs_mean"] == pytest.approx(0.25)
    assert metrics["act/last_abs_mean"] == pytest.approx(0.75)
    assert metrics["latent/final_abs_mean"] == pytest.approx(2.0)
    assert metrics["latent/final_abs_max"] == pytest.approx(2.0)
    assert metrics["group/advantage_abs_mean"] == pytest.approx(float(advantages[valid_mask].abs().mean().item()))
    assert metrics["reward/action_rate_mean"] == pytest.approx(4.0)
    assert metrics["done/anchor_pos_bad_frac"] == pytest.approx(0.5)
    assert metrics["reward_weighted/action_rate"] == pytest.approx(-0.008)


def test_short_body_name_removes_expected_suffixes() -> None:
    assert _short_body_name("left_wrist_yaw_link") == "left_wrist"
    assert _short_body_name("right_ankle_roll_link") == "right_ankle"
    assert _short_body_name("torso_link") == "torso"


class _FakePolicy:
    def __init__(self):
        self.training = True
        self.eval_called = False
        self.train_called = False
        self.chunk_dim = 29
        self.horizon = 1
        self.action_dim = 29

    def eval(self) -> None:
        self.training = False
        self.eval_called = True

    def train(self) -> None:
        self.training = True
        self.train_called = True

    def _validate_inputs(self, observation: torch.Tensor, flow_noise: torch.Tensor, steps: int) -> None:
        del steps
        assert observation.shape[0] == flow_noise.shape[0]

    def _prepare_observation(self, observation: torch.Tensor) -> torch.Tensor:
        return observation

    def velocity_field(self, observation: torch.Tensor, noisy_actions: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        del observation, time
        return torch.zeros_like(noisy_actions)

    def _action_transform(self, action_value: torch.Tensor) -> torch.Tensor:
        return action_value


class _FakeValidationEnv:
    def __init__(self):
        self.device = torch.device("cpu")
        self.ee_body_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
        self.step_count = 0

    def reset(self, phase_indices: torch.Tensor) -> torch.Tensor:
        return torch.zeros(phase_indices.shape[0], 154, device=self.device)

    def step(self, action: torch.Tensor, auto_reset: bool):
        del auto_reset
        self.step_count += 1
        num_envs = action.shape[0]
        done = torch.zeros(num_envs, dtype=torch.bool)
        if self.step_count >= 2:
            done[0] = True
        if self.step_count >= 3:
            done[1] = True
        reward = torch.ones(num_envs)
        done_terms = {
            "time_out": torch.zeros(num_envs, dtype=torch.bool),
            "anchor_pos_bad": torch.zeros(num_envs, dtype=torch.bool),
            "anchor_ori_bad": torch.zeros(num_envs, dtype=torch.bool),
            "ee_body_bad": done.clone(),
        }
        debug_terms = {
            "ee_z_error_max": torch.where(done, torch.full((num_envs,), 0.3), torch.zeros(num_envs)),
            "ee_z_error_mean": torch.where(done, torch.full((num_envs,), 0.15), torch.zeros(num_envs)),
            "anchor_z_error": torch.zeros(num_envs),
            "anchor_gravity_z_error": torch.zeros(num_envs),
            "ee_z_error_by_body": torch.where(
                done[:, None],
                torch.full((num_envs, 2), 0.3),
                torch.zeros(num_envs, 2),
            ),
        }
        reward_terms = {
            "diag_torso_ori_deg": torch.ones(num_envs),
            "diag_left_wrist_ori_deg": torch.ones(num_envs),
            "diag_right_wrist_ori_deg": torch.ones(num_envs),
            "diag_left_elbow_ori_deg": torch.ones(num_envs),
            "diag_right_elbow_ori_deg": torch.ones(num_envs),
            "diag_left_shoulder_ori_deg": torch.ones(num_envs),
            "diag_right_shoulder_ori_deg": torch.ones(num_envs),
            "diag_torso_ang_vel": torch.ones(num_envs),
            "diag_left_wrist_ang_vel": torch.ones(num_envs),
            "diag_right_wrist_ang_vel": torch.ones(num_envs),
            "diag_left_elbow_ang_vel": torch.ones(num_envs),
            "diag_right_elbow_ang_vel": torch.ones(num_envs),
            "diag_left_shoulder_ang_vel": torch.ones(num_envs),
            "diag_right_shoulder_ang_vel": torch.ones(num_envs),
        }
        info = {"done_terms": done_terms, "debug_terms": debug_terms, "reward_terms": reward_terms}
        return torch.zeros(num_envs, 154), reward, done, info


class _FakeApp:
    def is_running(self) -> bool:
        return True


class _ValidationHarness(ValidationMixin):
    def __init__(self):
        self.cfg = SimpleNamespace(
            num_envs=2,
            validation_start_phase=4,
            horizon=1,
            flow_steps=1,
            validation_max_steps=5,
        )
        self.chunk_dim = 29
        self.policy = _FakePolicy()
        self.env = _FakeValidationEnv()
        self.simulation_app = _FakeApp()
        self.current_observation = torch.full((2, 154), 7.0)
        self.reset_training_envs_called = False

    def _reset_training_envs(self) -> torch.Tensor:
        self.reset_training_envs_called = True
        return torch.full((2, 154), 9.0)


def test_validation_rollout_metrics_and_policy_mode_restore() -> None:
    harness = _ValidationHarness()

    metrics = harness.run_validation_rollout(fixed_seed=123)

    assert metrics["validation/steps_mean"] == pytest.approx(2.5)
    assert metrics["validation/done_frac"] == pytest.approx(1.0)
    assert metrics["validation/ee_body_bad_frac"] == pytest.approx(1.0)
    assert metrics["validation/ee_left_wrist_bad_frac"] == pytest.approx(1.0)
    assert harness.policy.eval_called
    assert harness.policy.train_called
    assert not harness.reset_training_envs_called
    assert torch.equal(harness.current_observation, torch.full((2, 154), 7.0))
