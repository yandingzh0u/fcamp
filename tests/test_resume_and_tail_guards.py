from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from engine.checkpoint import Checkpointer, _resume_signature
from engine.config import load_config


ROOT = Path(__file__).resolve().parents[1]


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


def test_resume_signature_contains_the_complete_physical_decoder() -> None:
    config = asdict(
        load_config(ROOT / "configs" / "fcamp_largebox.yaml")
    )
    signature = _resume_signature(config)

    for name in (
        "policy_action_bound",
        "command_rate_limit",
        "rate_half_life_seconds",
    ):
        assert signature["environment"][name] == config["environment"][name]


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
