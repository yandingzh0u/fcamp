from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("isaaclab")

from engine.checkpoint import CheckpointMixin
from engine.config import MixGRPOConfig
from engine.logging import LoggingMixin
from engine.sampling import flow_grpo_step
from engine.validation import ValidationMixin, _short_body_name


def test_config_defaults_match_training_contract() -> None:
    cfg = MixGRPOConfig(device="cpu")

    assert cfg.group_size == 4
    assert cfg.chunks_per_rollout == 24
    assert cfg.flow_steps == 4
    assert cfg.cps_eta == pytest.approx(0.7)
    assert cfg.max_episode_steps == 1500
    assert cfg.action_dim == 29
    assert Path(cfg.motion_file).is_file()


def test_flow_grpo_step_cps_transition_with_recorded_next_sample() -> None:
    latents = torch.tensor([[1.0, -2.0], [0.5, 0.25]])
    model_output = torch.tensor([[0.2, -0.4], [1.0, -2.0]])
    sigmas = torch.tensor([1.0, 0.5, 0.0])

    sigma = sigmas[0]
    sigma_prev = sigmas[1]
    pred_original = latents - sigma * model_output
    noise_estimate = latents + model_output * (1 - sigma)
    expected_mean = pred_original * (1 - sigma_prev) + noise_estimate * sigma_prev
    recorded_next = expected_mean + torch.tensor([[1.0, -1.0], [2.0, 0.0]])

    next_sample, log_prob = flow_grpo_step(
        model_output=model_output,
        latents=latents,
        sigmas=sigmas,
        index=0,
        prev_sample=recorded_next,
        noise_level=0.0,
    )

    expected_log_prob = -torch.mean((recorded_next - expected_mean) ** 2, dim=-1)
    assert torch.allclose(next_sample, recorded_next)
    assert torch.allclose(log_prob, expected_log_prob)


def test_flow_grpo_step_deterministic_uses_ode_update() -> None:
    latents = torch.tensor([[1.0, 2.0, 3.0]])
    model_output = torch.tensor([[0.5, -1.0, 2.0]])
    sigmas = torch.tensor([0.75, 0.25])

    next_sample, _ = flow_grpo_step(
        model_output=model_output,
        latents=latents,
        sigmas=sigmas,
        index=0,
        deterministic=True,
        noise_level=0.3,
    )

    expected = latents + (sigmas[1] - sigmas[0]) * model_output
    assert torch.allclose(next_sample, expected)


@dataclass
class _CheckpointConfig:
    lr: float = 1e-3
    target_validation_steps: int = 10


class _CheckpointHarness(CheckpointMixin):
    def __init__(self, checkpoint_dir: Path):
        self.cfg = _CheckpointConfig()
        self.checkpoint_dir = checkpoint_dir
        self.policy = torch.nn.Linear(3, 2)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.cfg.lr)
        self.env = SimpleNamespace(device=torch.device("cpu"))
        self.start_update = 1


def test_checkpoint_save_load_and_target_gate(tmp_path: Path) -> None:
    trainer = _CheckpointHarness(tmp_path)
    metrics = {"validation/steps_min": 10.0, "val_fixed/steps_min": 11.0}

    assert trainer._target_validation_reached(metrics)
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


class _LoggingHarness(LoggingMixin):
    def __init__(self):
        self.cfg = SimpleNamespace(max_updates=3)
        self.env = SimpleNamespace(ee_body_names=["left_wrist_yaw_link"])


def test_logging_update_emits_expected_sections(capsys: pytest.CaptureFixture[str]) -> None:
    harness = _LoggingHarness()
    metrics = {
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
    assert "[TRACK]" in out
    assert "[DONE]" in out


def test_short_body_name_removes_expected_suffixes() -> None:
    assert _short_body_name("left_wrist_yaw_link") == "left_wrist"
    assert _short_body_name("right_ankle_roll_link") == "right_ankle"
    assert _short_body_name("torso_link") == "torso"


class _FakePolicy:
    def __init__(self):
        self.training = True
        self.eval_called = False
        self.train_called = False

    def eval(self) -> None:
        self.training = False
        self.eval_called = True

    def train(self) -> None:
        self.training = True
        self.train_called = True

    def __call__(self, observation: torch.Tensor, noise: torch.Tensor, steps: int) -> torch.Tensor:
        del observation, steps
        return torch.zeros(noise.shape[0], 1, 29, device=noise.device)


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
        self.current_observation = torch.empty(0)

    def _reset_training_envs(self) -> torch.Tensor:
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
    assert torch.equal(harness.current_observation, torch.full((2, 154), 9.0))
