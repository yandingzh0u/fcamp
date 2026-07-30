from __future__ import annotations

import pytest
import torch

from envs.contracts import (
    resolve_root_velocity_frame,
    validate_actions_in_bounds,
)

def test_amp_policy_command_domain_is_fixed_symmetric_scale() -> None:
    low = torch.full((3,), -5.0)
    high = torch.full((3,), 5.0)
    validate_actions_in_bounds(
        torch.stack([low, torch.zeros_like(low), high]), low, high
    )
    invalid = high.clone()
    invalid[0] += 0.01
    with pytest.raises(RuntimeError, match="policy command domain"):
        validate_actions_in_bounds(
            invalid.unsqueeze(0), low, high
        )


def test_root_velocity_frame_is_explicit_not_inferred_from_the_tensor() -> None:
    assert resolve_root_velocity_frame("com", "link") == "link"
    assert resolve_root_velocity_frame("com", None) == "com"
    with pytest.raises(ValueError, match="root velocity frame"):
        resolve_root_velocity_frame("invalid", None)
