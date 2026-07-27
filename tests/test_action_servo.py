from __future__ import annotations

import pytest
import torch

from envs.action_servo import advance_position_servo


OMEGA = 20.0


def _state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[1.2, -0.4], [0.3, 0.8]], dtype=torch.float64),
        torch.tensor([[0.7, -1.1], [0.2, 1.4]], dtype=torch.float64),
        torch.tensor([[-0.5, 0.9], [1.3, -0.2]], dtype=torch.float64),
        torch.tensor([[0.5, 0.1], [-0.8, 1.0]], dtype=torch.float64),
    )


def test_exact_step_is_invariant_to_splitting_the_interval() -> None:
    action, rate, acceleration, target_action = _state()

    whole = advance_position_servo(
        target_action,
        action,
        rate,
        acceleration,
        dt=0.037,
        omega=OMEGA,
    )
    first = advance_position_servo(
        target_action,
        action,
        rate,
        acceleration,
        dt=0.011,
        omega=OMEGA,
    )
    split = advance_position_servo(
        target_action,
        *first,
        dt=0.026,
        omega=OMEGA,
    )

    for whole_value, split_value in zip(whole, split, strict=True):
        torch.testing.assert_close(
            whole_value, split_value, atol=2.0e-14, rtol=3.0e-15
        )


def test_target_jump_enters_through_jerk_without_state_jump() -> None:
    zeros = torch.zeros(1, 1, dtype=torch.float64)
    target_action = torch.ones_like(zeros)
    expected_jerk = OMEGA**3 * target_action

    coarse = advance_position_servo(
        target_action, zeros, zeros, zeros, dt=2.0e-5, omega=OMEGA
    )
    fine = advance_position_servo(
        target_action, zeros, zeros, zeros, dt=1.0e-5, omega=OMEGA
    )

    torch.testing.assert_close(
        fine[2] / 1.0e-5,
        expected_jerk,
        atol=2.5,
        rtol=0.0,
    )
    assert coarse[2].item() / fine[2].item() == pytest.approx(2.0, rel=5.0e-4)
    assert coarse[1].item() / fine[1].item() == pytest.approx(4.0, rel=5.0e-4)
    assert coarse[0].item() / fine[0].item() == pytest.approx(8.0, rel=5.0e-4)


def test_constant_target_stably_settles_the_complete_state() -> None:
    action = torch.tensor([[0.4, -1.0]], dtype=torch.float64)
    rate = torch.tensor([[3.0, -2.0]], dtype=torch.float64)
    acceleration = torch.tensor([[-5.0, 4.0]], dtype=torch.float64)
    target_action = torch.tensor([[0.2, 0.7]], dtype=torch.float64)

    for _ in range(500):
        action, rate, acceleration = advance_position_servo(
            target_action,
            action,
            rate,
            acceleration,
            dt=0.01,
            omega=OMEGA,
        )

    torch.testing.assert_close(action, target_action)
    assert float(rate.abs().max()) < 1.0e-13
    assert float(acceleration.abs().max()) < 1.0e-12


def test_inactive_environment_holds_all_carried_state() -> None:
    action, rate, acceleration, target_action = _state()
    next_state = advance_position_servo(
        target_action,
        action,
        rate,
        acceleration,
        dt=0.02,
        omega=OMEGA,
        active_mask=torch.tensor([True, False]),
    )

    for next_value, previous_value in zip(
        next_state, (action, rate, acceleration), strict=True
    ):
        torch.testing.assert_close(next_value[1], previous_value[1])
        assert not torch.equal(next_value[0], previous_value[0])


def test_exact_solution_matches_augmented_matrix_exponential() -> None:
    action, rate, acceleration, target_action = _state()
    dt = 0.073
    exact = advance_position_servo(
        target_action,
        action,
        rate,
        acceleration,
        dt=dt,
        omega=OMEGA,
    )

    generator = torch.zeros(4, 4, dtype=torch.float64)
    generator[0, 1] = 1.0
    generator[1, 2] = 1.0
    generator[2, 0] = -(OMEGA**3)
    generator[2, 1] = -(3.0 * OMEGA**2)
    generator[2, 2] = -(3.0 * OMEGA)
    generator[2, 3] = OMEGA**3
    transition = torch.matrix_exp(generator * dt)
    augmented = torch.stack(
        (action, rate, acceleration, target_action), dim=-1
    )
    expected = augmented @ transition.T

    for index, exact_value in enumerate(exact):
        torch.testing.assert_close(
            exact_value, expected[..., index], atol=2.0e-11, rtol=2.0e-12
        )
    torch.testing.assert_close(expected[..., 3], target_action)


def test_exact_solution_matches_fine_runge_kutta_integration() -> None:
    action, rate, acceleration, target_action = _state()
    exact = advance_position_servo(
        target_action,
        action,
        rate,
        acceleration,
        dt=0.073,
        omega=OMEGA,
    )

    numerical = (action, rate, acceleration)
    step = 0.073 / 10_000

    def derivative(
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, v, a = state
        jerk = (
            OMEGA**3 * (target_action - x)
            - 3.0 * OMEGA**2 * v
            - 3.0 * OMEGA * a
        )
        return v, a, jerk

    def add(
        state: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        slope: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return tuple(
            value + scale * delta
            for value, delta in zip(state, slope, strict=True)
        )

    for _ in range(10_000):
        k1 = derivative(numerical)
        k2 = derivative(add(numerical, k1, step / 2.0))
        k3 = derivative(add(numerical, k2, step / 2.0))
        k4 = derivative(add(numerical, k3, step))
        numerical = tuple(
            value
            + step
            * (d1 + 2.0 * d2 + 2.0 * d3 + d4)
            / 6.0
            for value, d1, d2, d3, d4 in zip(
                numerical, k1, k2, k3, k4, strict=True
            )
        )

    for exact_value, numerical_value in zip(exact, numerical, strict=True):
        torch.testing.assert_close(
            exact_value, numerical_value, atol=2.0e-13, rtol=2.0e-13
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dt": 0.0, "omega": OMEGA}, "dt must be finite and positive"),
        ({"dt": float("nan"), "omega": OMEGA}, "dt must be finite and positive"),
        ({"dt": 0.02, "omega": 0.0}, "omega must be finite and positive"),
        (
            {"dt": 0.02, "omega": float("inf")},
            "omega must be finite and positive",
        ),
    ],
)
def test_servo_rejects_invalid_scalar_parameters(
    kwargs: dict[str, float],
    message: str,
) -> None:
    action, rate, acceleration, target_action = _state()
    with pytest.raises(ValueError, match=message):
        advance_position_servo(
            target_action, action, rate, acceleration, **kwargs
        )


def test_servo_rejects_invalid_state_or_mask() -> None:
    action, rate, acceleration, target_action = _state()

    with pytest.raises(ValueError, match="identical shapes"):
        advance_position_servo(
            target_action[:, :1],
            action,
            rate,
            acceleration,
            dt=0.02,
            omega=OMEGA,
        )
    with pytest.raises(ValueError, match="target_action must be finite"):
        advance_position_servo(
            target_action.fill_(float("nan")),
            action,
            rate,
            acceleration,
            dt=0.02,
            omega=OMEGA,
        )
    with pytest.raises(ValueError, match="active_mask must have shape"):
        advance_position_servo(
            torch.zeros_like(action),
            action,
            rate,
            acceleration,
            dt=0.02,
            omega=OMEGA,
            active_mask=torch.ones(2, 1, dtype=torch.bool),
        )
