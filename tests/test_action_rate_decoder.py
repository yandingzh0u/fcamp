from __future__ import annotations

import pytest
import torch

from envs.action_rate import (
    command_rate_decay,
    decode_raw_target_rate,
    normalize_command_rate,
)


DT = 0.02
HALF_LIFE = 0.08


def _decode(
    raw: torch.Tensor,
    action: torch.Tensor,
    rate: torch.Tensor,
    *,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    action_dim = action.shape[-1]
    return decode_raw_target_rate(
        raw,
        action,
        rate,
        torch.full((action_dim,), 10.0),
        torch.full((action_dim,), -5.0),
        torch.full((action_dim,), 5.0),
        control_dt=DT,
        decay=command_rate_decay(DT, HALF_LIFE),
        active_mask=active_mask,
    )


def _rollout(
    raw_frames: torch.Tensor,
    initial_action: torch.Tensor,
    initial_rate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    action = initial_action
    rate = initial_rate
    actions = []
    for raw in raw_frames:
        next_action = _decode(raw, action, rate)
        rate = (next_action - action) / DT
        action = next_action
        actions.append(action)
    return torch.stack(actions), action, rate


def test_decay_is_a_physical_time_constant_not_a_horizon_constant() -> None:
    decay = command_rate_decay(DT, HALF_LIFE)
    assert decay == pytest.approx(2.0 ** -0.25)
    assert decay**4 == pytest.approx(0.5)
    assert command_rate_decay(DT, HALF_LIFE) == decay


def test_policy_observes_dimensionless_rate_while_state_stays_physical() -> None:
    command_rate = torch.tensor(
        [[20.0, -5.0, 0.0], [-40.0, 2.5, 15.0]]
    )
    original = command_rate.clone()
    rate_limit = torch.tensor([40.0, 10.0, 30.0])

    observed = normalize_command_rate(command_rate, rate_limit)

    torch.testing.assert_close(
        observed,
        torch.tensor([[0.5, -0.5, 0.0], [-1.0, 0.25, 0.5]]),
    )
    torch.testing.assert_close(command_rate, original)


def test_reblocking_does_not_change_the_eight_frame_trajectory() -> None:
    generator = torch.Generator().manual_seed(7)
    raw = torch.randn(8, 3, 2, generator=generator)
    initial_action = torch.tensor(
        [[0.2, -0.3], [0.1, 0.4], [-0.2, 0.0]]
    )
    initial_rate = torch.tensor(
        [[1.0, -2.0], [0.5, 0.25], [-1.0, 1.5]]
    )

    continuous, _, _ = _rollout(raw, initial_action, initial_rate)
    first, carried_action, carried_rate = _rollout(
        raw[:4], initial_action, initial_rate
    )
    second, _, _ = _rollout(raw[4:], carried_action, carried_rate)

    torch.testing.assert_close(continuous, torch.cat((first, second), dim=0))


def test_zero_target_rate_decays_to_hold_without_hidden_windup() -> None:
    raw = torch.zeros(40, 1, 1)
    actions, _, final_rate = _rollout(
        raw,
        torch.zeros(1, 1),
        torch.full((1, 1), 4.0),
    )
    expected_rate = 4.0 * command_rate_decay(DT, HALF_LIFE) ** torch.arange(
        1, 41
    )
    actual_rate = torch.diff(
        torch.cat((torch.zeros(1, 1, 1), actions), dim=0),
        dim=0,
    ).flatten() / DT

    torch.testing.assert_close(actual_rate, expected_rate)
    assert float(final_rate.abs().max()) < 0.004
    assert float(actions[-1].abs().max()) < 0.5


def test_projection_is_reconciled_by_the_applied_action_difference() -> None:
    action = torch.tensor([[4.99]])
    rate = torch.tensor([[10.0]])
    next_action = _decode(torch.full_like(action, 100.0), action, rate)
    torch.testing.assert_close(next_action, torch.tensor([[5.0]]))

    reconciled_rate = (next_action - action) / DT
    assert reconciled_rate.item() == pytest.approx(0.5, abs=2.0e-5)
    held_action = _decode(
        torch.zeros_like(action),
        next_action,
        torch.zeros_like(reconciled_rate),
    )
    torch.testing.assert_close(held_action, next_action)


def test_one_pole_prediction_equals_unprojected_action_second_difference() -> None:
    action = torch.tensor([[0.25, -0.50], [-0.75, 0.40]])
    previous_rate = torch.tensor([[1.50, -2.00], [0.25, 1.00]])
    raw = torch.tensor([[0.30, -0.20], [0.10, 0.40]])
    decay = command_rate_decay(DT, HALF_LIFE)

    next_action = _decode(raw, action, previous_rate)
    actual_d2 = (next_action - action) - DT * previous_rate
    target_rate = 10.0 * torch.tanh(raw)
    predicted_d2 = DT * (1.0 - decay) * (
        target_rate - previous_rate
    )

    assert bool((next_action.abs() < 5.0).all())
    torch.testing.assert_close(actual_d2, predicted_d2, atol=1.0e-7, rtol=1.0e-6)


def test_inactive_environment_holds_its_actual_command() -> None:
    action = torch.tensor([[1.0, -1.0], [2.0, -2.0]])
    rate = torch.tensor([[3.0, -3.0], [4.0, -4.0]])
    raw = torch.full_like(action, 2.0)
    decoded = _decode(
        raw,
        action,
        rate,
        active_mask=torch.tensor([True, False]),
    )

    torch.testing.assert_close(decoded[1], action[1])
    assert not torch.equal(decoded[0], action[0])


def test_decoder_rejects_nonfinite_raw_policy_values() -> None:
    with pytest.raises(ValueError, match="raw_target_rate must be finite"):
        _decode(
            torch.tensor([[float("nan")]]),
            torch.zeros(1, 1),
            torch.zeros(1, 1),
        )
