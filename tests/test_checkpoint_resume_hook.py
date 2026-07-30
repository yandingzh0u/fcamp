from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from engine.checkpoint import Checkpointer


class _OrderedSampler:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def load_state_dict(self, _state) -> bool:
        self.events.append("sampler")
        return True


class _ResumeAwareAlgorithm:
    def __init__(
        self,
        events: list[str],
        expected_rng_state: torch.Tensor,
        resumed_observation: torch.Tensor,
    ) -> None:
        self.events = events
        self.expected_rng_state = expected_rng_state
        self.resumed_observation = resumed_observation
        self.policy = nn.Linear(1, 1)
        self.optimizer = torch.optim.SGD(self.policy.parameters(), lr=0.1)

    def load_extra_checkpoint_state(self, _payload, reset_optimizer: bool = False) -> None:
        del reset_optimizer
        self.events.append("algorithm")

    def reset_after_resume(self) -> torch.Tensor:
        assert torch.equal(torch.random.get_rng_state(), self.expected_rng_state)
        assert self.events == ["algorithm", "sampler"]
        self.events.append("reset_after_resume")
        return self.resumed_observation


def test_resume_hook_runs_after_sampler_and_rng_restore(tmp_path) -> None:
    events: list[str] = []
    resumed_observation = torch.tensor([[42.0]])

    torch.manual_seed(8128)
    checkpoint_rng_state = torch.random.get_rng_state()
    algo = _ResumeAwareAlgorithm(events, checkpoint_rng_state, resumed_observation)
    trainer = SimpleNamespace(
        algo=algo,
        env=SimpleNamespace(
            device="cpu",
            adaptive_sampler=_OrderedSampler(events),
        ),
        train_cfg=SimpleNamespace(
            reset_optimizer_on_resume=False,
            reset_sampler_on_resume=False,
        ),
        current_observation=torch.tensor([[-1.0]]),
        start_update=1,
    )
    checkpoint = tmp_path / "resume.pt"
    torch.save(
        {
            "update_idx": 7,
            "policy": algo.policy.state_dict(),
            "optimizer": algo.optimizer.state_dict(),
            "algo_state": {},
            "adaptive_sampler_state": {},
            "torch_rng_state": checkpoint_rng_state,
        },
        checkpoint,
    )

    torch.manual_seed(123)
    Checkpointer(trainer).load(checkpoint)

    assert events == ["algorithm", "sampler", "reset_after_resume"]
    assert trainer.current_observation is resumed_observation
    assert trainer.start_update == 8


def test_target_reached_uses_directional_validation_key() -> None:
    trainer = SimpleNamespace(
        train_cfg=SimpleNamespace(target_validation_steps=100),
        env=SimpleNamespace(max_episode_steps=500),
    )
    checkpointer = Checkpointer(trainer)

    assert checkpointer.target_reached(
        {
            "validation/steps_min": 101.0,
            "validation_directional/steps_min": 101.0,
        }
    )
    assert not checkpointer.target_reached(
        {
            "validation/steps_min": 101.0,
            "validation_directional/steps_min": 100.0,
        }
    )
