from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from algorithms.mixgrpo import MixGRPO
from core.checkpoint import Checkpointer


class _FakeSampler:
    def __init__(self) -> None:
        self.init_calls = 0
        self.load_calls = 0

    def init_buffers(self) -> None:
        self.init_calls += 1

    def load_state_dict(self, _state) -> bool:
        self.load_calls += 1
        return True


class _FakeAlgo:
    def __init__(self) -> None:
        self.policy = nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=0.1)
        self.extra_load_args: tuple[dict, bool] | None = None

    def load_extra_checkpoint_state(self, payload: dict, reset_optimizer: bool = False) -> None:
        self.extra_load_args = (payload, reset_optimizer)


def test_reset_sampler_on_resume_skips_compatible_checkpoint_state(tmp_path) -> None:
    algo = _FakeAlgo()
    sampler = _FakeSampler()
    trainer = SimpleNamespace(
        algo=algo,
        env=SimpleNamespace(
            device="cpu",
            adaptive_sampler=sampler,
            _failure_recorded=torch.ones(2, dtype=torch.bool),
        ),
        train_cfg=SimpleNamespace(reset_optimizer_on_resume=False, reset_sampler_on_resume=True),
        start_update=1,
    )
    checkpoint = tmp_path / "resume.pt"
    torch.save(
        {
            "update_idx": 12,
            "policy": algo.policy.state_dict(),
            "optimizer": algo.optimizer.state_dict(),
            "algo_state": {},
            "adaptive_sampler_state": {"version": 3},
        },
        checkpoint,
    )

    Checkpointer(trainer).load(checkpoint)

    assert sampler.init_calls == 1
    assert sampler.load_calls == 0
    assert not bool(trainer.env._failure_recorded.any())
    assert trainer.start_update == 13


class _TailEnv:
    def __init__(self) -> None:
        self.record_motion_failures = True
        self.record_flags_seen: list[bool] = []

    def step(self, action: torch.Tensor, auto_reset: bool = False):
        del auto_reset
        self.record_flags_seen.append(self.record_motion_failures)
        count = action.shape[0]
        reward = torch.ones(count, device=action.device)
        done = torch.zeros(count, dtype=torch.bool, device=action.device)
        info = {"done_terms": {"time_out": done.clone()}}
        return action, reward, done, info


def test_mixgrpo_tail_bootstrap_disables_and_restores_sampler_recording() -> None:
    algo = object.__new__(MixGRPO)
    algo._policy = nn.Identity()
    algo._policy.train()
    algo.env = _TailEnv()
    algo.cfg = SimpleNamespace(horizon=1)
    algo.deterministic_actions = lambda obs: obs.unsqueeze(1)

    obs = torch.zeros(4, 3)
    alive = torch.ones(4, dtype=torch.bool)
    result = algo._compute_tail_bootstrap(
        obs,
        alive,
        tail_steps=3,
        gamma=0.99,
        terminal_penalty=1.0,
    )

    assert result.shape == (4,)
    assert algo.env.record_flags_seen == [False, False, False]
    assert algo.env.record_motion_failures is True
    assert algo._policy.training is True
