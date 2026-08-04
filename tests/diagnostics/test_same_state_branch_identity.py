from __future__ import annotations

import torch

from diagnostics.common.rollout_collector import same_state_branch_identity
from diagnostics.common.stage4_branch_collection import replay_max_abs_error


def test_same_state_branch_replay_is_exact() -> None:
    state = {"value": torch.tensor([3.0])}

    def restore() -> None:
        state["value"] = torch.tensor([3.0])

    def branch() -> dict[str, torch.Tensor]:
        state["value"] += 2.0
        return {"state": state["value"]}

    assert same_state_branch_identity(restore, branch)["identical"] is True


def _stage4_tree(done: torch.Tensor, values: torch.Tensor) -> dict[str, object]:
    return {
        "trajectory": {"done": done},
        "state": {"root_pos": values, "body_pos": values.unsqueeze(-2)},
        "action": {"applied": values},
        "imitation": {"agent_physx_raw_frame": values},
    }


def test_stage4_replay_ignores_only_post_terminal_auto_reset_samples() -> None:
    done = torch.tensor([[False], [True], [False], [False]])
    first_values = torch.tensor([[[0.0]], [[1.0]], [[10.0]], [[11.0]]])
    replay_values = torch.tensor([[[0.0]], [[1.0]], [[20.0]], [[21.0]]])
    assert replay_max_abs_error(
        _stage4_tree(done, first_values), _stage4_tree(done, replay_values)
    ) == 0.0


def test_stage4_replay_still_rejects_first_terminal_or_active_drift() -> None:
    done = torch.tensor([[False], [True], [False]])
    later_done = torch.tensor([[False], [False], [True]])
    values = torch.tensor([[[0.0]], [[1.0]], [[2.0]]])
    assert replay_max_abs_error(
        _stage4_tree(done, values), _stage4_tree(later_done, values)
    ) == float("inf")
    changed = values.clone()
    changed[1] += 1.0e-4
    assert replay_max_abs_error(
        _stage4_tree(done, values), _stage4_tree(done, changed)
    ) >= 1.0e-4
