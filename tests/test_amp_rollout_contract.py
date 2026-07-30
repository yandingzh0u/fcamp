from __future__ import annotations

from collections import deque
import math
from types import SimpleNamespace

import pytest
import torch

from components.imitation.temporal_history import TemporalFeatureHistory
from components.rollout.amp_gaussian_base import GaussianPolicySample
from method.amp import AMP
from models.amp_actor_critic import DiagonalGaussian


class _EvalOnlyDiscriminator:
    def eval(self) -> "_EvalOnlyDiscriminator":
        return self


class _ScriptedAMPEnv:
    """Small tensor-only environment for exercising AMP rollout semantics."""

    def __init__(
        self,
        *,
        num_envs: int,
        observation_dim: int,
        frame_dim: int,
        done_script: torch.Tensor,
        numerical_script: torch.Tensor | None = None,
    ) -> None:
        if done_script.ndim != 2 or done_script.shape[1] != num_envs:
            raise ValueError("done_script must have shape [step, num_envs]")
        self.device = torch.device("cpu")
        self.num_envs = int(num_envs)
        self.observation_dim = int(observation_dim)
        self.imitation_frame_dim = int(frame_dim)
        self.phase_steps = torch.arange(num_envs, dtype=torch.long)
        self.done_script = done_script.bool().clone()
        self.numerical_script = (
            torch.zeros_like(self.done_script)
            if numerical_script is None
            else numerical_script.bool().clone()
        )
        self._step_index = 0
        self.actions_seen: list[torch.Tensor] = []

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        step = self._step_index
        if step >= self.done_script.shape[0]:
            raise RuntimeError("script exhausted")
        self.actions_seen.append(actions.detach().clone())
        self.phase_steps += 1

        env_id = torch.arange(self.num_envs, dtype=torch.float32)
        observation = torch.stack(
            [
                torch.full((self.num_envs,), float(step + 1)),
                env_id,
                torch.full((self.num_envs,), float(step) + 0.25),
            ],
            dim=-1,
        )
        if self.observation_dim != 3:
            raise AssertionError("test environment assumes observation_dim=3")

        done = self.done_script[step].clone()
        numerical = done & self.numerical_script[step]
        observation[numerical] = float("nan")
        physical_failure = done.clone()
        imitation_frame = (
            torch.arange(
                self.num_envs * self.imitation_frame_dim,
                dtype=torch.float32,
            ).reshape(self.num_envs, self.imitation_frame_dim)
            + 1000.0 * float(step + 1)
        )
        imitation_frame[numerical] = float("nan")
        false = torch.zeros(self.num_envs, dtype=torch.bool)
        info = {
            "imitation_frame": imitation_frame,
            "imitation_frame_phase_steps": self.phase_steps.float(),
            "done_terms": {
                "time_out": false.clone(),
                "physical_failure": physical_failure,
                "illegal_contact": done & ~numerical,
                "numerical_failure": numerical,
                "tracking_failure": false.clone(),
                "motion_complete": false.clone(),
            },
            "debug_terms": {
                "contact_force_max": torch.zeros(self.num_envs),
            },
        }
        self._step_index += 1
        return observation, done, info


def _make_rollout_algorithm(
    *,
    horizon: int,
    done_script: torch.Tensor,
    numerical_script: torch.Tensor | None = None,
) -> tuple[
    AMP,
    _ScriptedAMPEnv,
    torch.Tensor,
    torch.Tensor,
    list[dict[str, torch.Tensor]],
]:
    num_steps, num_envs = done_script.shape
    observation_dim = 3
    frame_dim = 4
    history_steps = 3
    action_dim = 2
    env = _ScriptedAMPEnv(
        num_envs=num_envs,
        observation_dim=observation_dim,
        frame_dim=frame_dim,
        done_script=done_script,
        numerical_script=numerical_script,
    )

    algo = AMP.__new__(AMP)
    algo.cfg = SimpleNamespace(
        rollout_env_steps=num_steps,
        discount_gamma=0.9,
        gae_lambda=0.95,
        advantage_clip=4.0,
        style_prior=SimpleNamespace(
            reward_eval_batch_size=16,
        ),
    )
    algo.env = env
    algo.horizon_h = int(horizon)
    algo.num_act = action_dim
    algo.actor_obs_dim = observation_dim
    algo.imitation_history_steps = history_steps
    algo.imitation_frame_dim = frame_dim
    algo.imitation_history = TemporalFeatureHistory(
        num_envs,
        history_steps,
        frame_dim,
        device="cpu",
    )
    seed_windows = (
        torch.arange(
            num_envs * history_steps * frame_dim,
            dtype=torch.float32,
        ).reshape(num_envs, history_steps, frame_dim)
        + 10.0
    )
    algo.imitation_history.reset_seeded(seed_windows)
    algo.discriminator = _EvalOnlyDiscriminator()
    algo.disc_normalizer = SimpleNamespace(count=torch.tensor(7.0))
    algo.disc_version = 3
    algo._episode_reward = torch.zeros(num_envs)
    algo._episode_length = torch.zeros(num_envs)
    algo._train_reward_buffer = deque(maxlen=100)
    algo._train_length_buffer = deque(maxlen=100)

    algo.normalize_actor_observation = lambda observations: observations
    samples: list[dict[str, torch.Tensor]] = []

    def sample_chunk(observations: torch.Tensor) -> GaussianPolicySample:
        batch = observations.shape[0]
        serial = float(len(samples) + 1)
        mean = (
            observations[:, :1, None]
            + serial * 10.0
            + torch.arange(horizon, dtype=torch.float32)[None, :, None]
            + torch.arange(action_dim, dtype=torch.float32)[None, None, :]
            / 10.0
        )
        noise = (
            torch.arange(
                1,
                batch * horizon * action_dim + 1,
                dtype=torch.float32,
            ).reshape(batch, horizon, action_dim)
            / 10.0
        )
        distribution = DiagonalGaussian(
            mean,
            torch.full((horizon, action_dim), math.log(0.05)),
        )
        actions = distribution.sample(noise)
        samples.append(
            {
                "mean": mean.detach().clone(),
                "noise": noise.detach().clone(),
                "actions": actions.detach().clone(),
                "std": distribution.stddev.detach().clone(),
            }
        )
        return GaussianPolicySample(
            actions=actions,
            mean=mean,
            log_std=distribution.log_std,
            log_prob=distribution.log_prob(actions),
            entropy=distribution.entropy(),
        )

    algo.sample_normalized_action_chunk = sample_chunk

    def value(observations: torch.Tensor) -> torch.Tensor:
        finite = torch.isfinite(observations).all(dim=-1, keepdim=True)
        return torch.where(
            finite,
            torch.full_like(observations[:, :1], 0.25),
            torch.full_like(observations[:, :1], float("nan")),
        )

    algo.value_from_normalized_observation = value
    algo._amp_reward_from_windows = lambda windows: (
        torch.zeros(windows.shape[0]),
        torch.ones(windows.shape[0]),
    )

    def reset_done(env_ids: torch.Tensor) -> torch.Tensor:
        env.phase_steps[env_ids] = 0
        algo.imitation_history.reset_seeded(
            seed_windows.index_select(0, env_ids),
            env_ids=env_ids,
        )
        return torch.zeros(env_ids.numel(), observation_dim)

    algo._uniform_reset = reset_done
    initial_observation = torch.stack(
        [
            torch.zeros(num_envs),
            torch.arange(num_envs, dtype=torch.float32),
            torch.full((num_envs,), -0.25),
        ],
        dim=-1,
    )
    return algo, env, initial_observation, seed_windows, samples


@pytest.mark.parametrize("horizon", [1, 4])
def test_collect_executes_direct_fixed_std_chunks_and_rewards_from_step_one(
    horizon: int,
) -> None:
    num_steps = 5
    (
        algo,
        env,
        initial_observation,
        seed_before_push,
        samples,
    ) = _make_rollout_algorithm(
        horizon=horizon,
        done_script=torch.zeros(num_steps, 2, dtype=torch.bool),
    )

    rollout = algo.collect(initial_observation)

    # Every action passed to the environment is the exact sampled absolute
    # Gaussian action at that decision/offset; there is no previous-action
    # anchor, temporal accumulation, squashing, or other hidden transform.
    decision_slot = [0 for _ in range(env.num_envs)]
    for step in range(num_steps):
        for env_id in range(env.num_envs):
            offset = int(rollout["primitive"]["offsets"][step, env_id])
            if step > 0 and offset == 0:
                decision_slot[env_id] += 1
            torch.testing.assert_close(
                rollout["primitive"]["actions"][step, env_id],
                rollout["actions"][
                    decision_slot[env_id], env_id, offset
                ],
            )
        torch.testing.assert_close(
            env.actions_seen[step],
            rollout["primitive"]["actions"][step],
        )

    for sample in samples:
        torch.testing.assert_close(
            sample["std"],
            torch.full_like(sample["std"], 0.05),
            atol=1.0e-8,
            rtol=1.0e-6,
        )
        torch.testing.assert_close(
            sample["actions"],
            sample["mean"] + 0.05 * sample["noise"],
            atol=1.0e-7,
            rtol=1.0e-6,
        )

    assert int(rollout["action_mask"].sum()) == num_steps * env.num_envs
    assert rollout["current_windows"].shape == (
        num_steps * env.num_envs,
        3,
        4,
    )
    # Demo-seeded history makes the first post-action transition immediately
    # legal: W-1 demo predecessor frames followed by the real policy frame.
    first_policy_frame = (
        torch.arange(env.num_envs * 4, dtype=torch.float32)
        .reshape(env.num_envs, 4)
        + 1000.0
    )
    expected_first = torch.cat(
        (seed_before_push[:, 1:], first_policy_frame[:, None]),
        dim=1,
    )
    actual_first = rollout["current_windows"][: env.num_envs]
    torch.testing.assert_close(actual_first, expected_first)
    torch.testing.assert_close(
        rollout["primitive"]["rewards"][0],
        torch.ones(env.num_envs),
    )


def test_h4_terminal_prefix_restarts_chunk_and_packs_decisions_contiguously() -> None:
    done = torch.zeros(6, 2, dtype=torch.bool)
    done[1, 0] = True
    (
        algo,
        _env,
        initial_observation,
        _seed,
        _samples,
    ) = _make_rollout_algorithm(
        horizon=4,
        done_script=done,
    )

    rollout = algo.collect(initial_observation)

    torch.testing.assert_close(
        rollout["primitive"]["offsets"][:, 0],
        torch.tensor([0, 1, 0, 1, 2, 3]),
    )
    torch.testing.assert_close(
        rollout["primitive"]["offsets"][:, 1],
        torch.tensor([0, 1, 2, 3, 0, 1]),
    )
    torch.testing.assert_close(
        rollout["decision_count"],
        torch.tensor([2, 2]),
    )
    assert bool(rollout["valid"][:2].all())
    assert not bool(rollout["valid"][2:].any())
    torch.testing.assert_close(
        rollout["durations"][:2],
        torch.tensor([[2, 4], [4, 2]]),
    )
    torch.testing.assert_close(
        rollout["action_mask"][0, 0],
        torch.tensor([True, True, False, False]),
    )
    torch.testing.assert_close(
        rollout["action_mask"][1, 0],
        torch.tensor([True, True, True, True]),
    )
    assert bool(rollout["failure"][0, 0])
    assert not bool(rollout["bootstrap_mask"][0, 0])
    assert not bool(rollout["trace_mask"][0, 0])
    assert int(rollout["action_mask"].sum()) == 12


def test_numerical_failure_cannot_put_nan_bootstrap_into_returns() -> None:
    done = torch.ones(1, 1, dtype=torch.bool)
    numerical = torch.ones_like(done)
    (
        algo,
        _env,
        initial_observation,
        _seed,
        _samples,
    ) = _make_rollout_algorithm(
        horizon=4,
        done_script=done,
        numerical_script=numerical,
    )

    rollout = algo.collect(initial_observation)

    assert bool(rollout["failure"][0, 0])
    assert not bool(rollout["bootstrap_mask"][0, 0])
    assert rollout["current_windows"].shape[0] == 0
    assert rollout["current_endpoints"].shape[0] == 0
    assert rollout["primitive"]["rewards"][0, 0].item() == 0.0
    assert not bool(rollout["primitive"]["amp_window_valid"][0, 0])
    assert torch.isfinite(rollout["next_values"][0, 0])
    assert rollout["next_values"][0, 0].item() == 0.0

    algo._compute_credit(rollout)
    assert bool(torch.isfinite(rollout["advantages"][rollout["valid"]]).all())
    assert bool(
        torch.isfinite(rollout["value_targets"][rollout["valid"]]).all()
    )
