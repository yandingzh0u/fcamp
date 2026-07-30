from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from engine.checkpoint import Checkpointer


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
        assert self.events == ["algorithm"]
        self.events.append("reset_after_resume")
        return self.resumed_observation


def test_resume_hook_runs_after_algorithm_and_rng_restore(tmp_path) -> None:
    events: list[str] = []
    resumed_observation = torch.tensor([[42.0]])

    torch.manual_seed(8128)
    checkpoint_rng_state = torch.random.get_rng_state()
    algo = _ResumeAwareAlgorithm(events, checkpoint_rng_state, resumed_observation)
    trainer = SimpleNamespace(
        algo=algo,
        env=SimpleNamespace(device="cpu"),
        train_cfg=SimpleNamespace(
            reset_optimizer_on_resume=False,
        ),
        current_observation=torch.tensor([[-1.0]]),
        start_update=1,
    )
    checkpoint = tmp_path / "resume.pt"
    platform_identity = {
        "dataset_sha256": "dataset",
        "robot_asset_sha256": "robot",
        "action_schema_sha256": "actions",
    }
    torch.save(
        {
            "update_idx": 7,
            "policy": algo.policy.state_dict(),
            "optimizer": algo.optimizer.state_dict(),
            "algo_state": {},
            "torch_rng_state": checkpoint_rng_state,
            "platform_identity": platform_identity,
        },
        checkpoint,
    )

    torch.manual_seed(123)
    checkpointer = Checkpointer(trainer)
    checkpointer.platform_identity = platform_identity
    checkpointer.load(checkpoint)

    assert events == ["algorithm", "reset_after_resume"]
    assert trainer.current_observation is resumed_observation
    assert trainer.start_update == 8


def test_resume_rejects_checkpoint_without_platform_identity(tmp_path) -> None:
    algo = _ResumeAwareAlgorithm([], torch.random.get_rng_state(), torch.zeros(1, 1))
    trainer = SimpleNamespace(
        algo=algo,
        env=SimpleNamespace(device="cpu"),
        train_cfg=SimpleNamespace(reset_optimizer_on_resume=False),
    )
    checkpoint = tmp_path / "missing-platform-identity.pt"
    torch.save(
        {
            "update_idx": 1,
            "policy": algo.policy.state_dict(),
            "optimizer": algo.optimizer.state_dict(),
            "algo_state": {},
        },
        checkpoint,
    )

    checkpointer = Checkpointer(trainer)
    checkpointer.platform_identity = {
        "dataset_sha256": "dataset",
        "robot_asset_sha256": "robot",
        "action_schema_sha256": "actions",
    }
    with pytest.raises(ValueError, match="dataset/robot/action-schema identity"):
        checkpointer.load(checkpoint)
