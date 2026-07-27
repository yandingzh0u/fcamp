from __future__ import annotations

import pytest
import torch

from envs.action_rate import advance_rate_servo


OMEGA = 20.0


def _state() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[1.2, -0.4], [0.3, 0.8]], dtype=torch.float64),
        torch.tensor([[0.7, -1.1], [0.2, 1.4]], dtype=torch.float64),
        torch.tensor([[-0.5, 0.9], [1.3, -0.2]], dtype=torch.float64),
        torch.tensor([[1.5, 0.1], [-0.8, 2.0]], dtype=torch.float64),
    )


def test_exact_step_is_invariant_to_splitting_the_interval() -> None:
    action, rate, acceleration, target_rate = _state()

    whole = advance_rate_servo(
        target_rate,
        action,
        rate,
        acceleration,
        dt=0.037,
        omega=OMEGA,
    )
    first = advance_rate_servo(
        target_rate,
        action,
        rate,
        acceleration,
        dt=0.011,
        omega=OMEGA,
    )
    split = advance_rate_servo(
        target_rate,
        *first,
        dt=0.026,
        omega=OMEGA,
    )

    for whole_value, split_value in zip(whole, split, strict=True):
        torch.testing.assert_close(
            whole_value, split_value, atol=2.0e-15, rtol=2.0e-15
        )


def test_target_rate_step_enters_through_jerk_without_state_jump() -> None:
    zeros = torch.zeros(1, 1, dtype=torch.float64)
    target_rate = torch.ones_like(zeros)
    expected_jerk = OMEGA**2 * target_rate

    coarse = advance_rate_servo(
        target_rate, zeros, zeros, zeros, dt=2.0e-5, omega=OMEGA
    )
    fine = advance_rate_servo(
        target_rate, zeros, zeros, zeros, dt=1.0e-5, omega=OMEGA
    )

    torch.testing.assert_close(
        fine[2] / 1.0e-5,
        expected_jerk,
        atol=0.17,
        rtol=0.0,
    )
    assert coarse[2].item() / fine[2].item() == pytest.approx(2.0, rel=5.0e-4)
    assert coarse[1].item() / fine[1].item() == pytest.approx(4.0, rel=5.0e-4)
    assert coarse[0].item() / fine[0].item() == pytest.approx(8.0, rel=5.0e-4)


def test_zero_target_rate_stably_settles_rate_and_acceleration() -> None:
    action = torch.tensor([[0.4, -1.0]], dtype=torch.float64)
    rate = torch.tensor([[3.0, -2.0]], dtype=torch.float64)
    acceleration = torch.tensor([[-5.0, 4.0]], dtype=torch.float64)
    target_rate = torch.zeros_like(action)

    initial_action = action
    for _ in range(500):
        action, rate, acceleration = advance_rate_servo(
            target_rate,
            action,
            rate,
            acceleration,
            dt=0.01,
            omega=OMEGA,
        )

    expected_final_action = (
        initial_action
        + torch.tensor([[3.0, -2.0]], dtype=torch.float64) * (2.0 / OMEGA)
        + torch.tensor([[-5.0, 4.0]], dtype=torch.float64) / OMEGA**2
    )
    torch.testing.assert_close(action, expected_final_action)
    assert float(rate.abs().max()) < 1.0e-38
    assert float(acceleration.abs().max()) < 1.0e-36


def test_inactive_environment_holds_all_carried_state() -> None:
    action, rate, acceleration, target_rate = _state()
    next_state = advance_rate_servo(
        target_rate,
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


def test_exact_solution_matches_fine_runge_kutta_integration() -> None:
    action, rate, acceleration, target_rate = _state()
    exact = advance_rate_servo(
        target_rate,
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
        return v, a, OMEGA**2 * (target_rate - v) - 2.0 * OMEGA * a

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
    action, rate, acceleration, target_rate = _state()
    with pytest.raises(ValueError, match=message):
        advance_rate_servo(
            target_rate, action, rate, acceleration, **kwargs
        )


def test_servo_rejects_invalid_state_or_mask() -> None:
    action, rate, acceleration, target_rate = _state()

    with pytest.raises(ValueError, match="identical shapes"):
        advance_rate_servo(
            target_rate[:, :1],
            action,
            rate,
            acceleration,
            dt=0.02,
            omega=OMEGA,
        )
    with pytest.raises(ValueError, match="target_rate must be finite"):
        advance_rate_servo(
            target_rate.fill_(float("nan")),
            action,
            rate,
            acceleration,
            dt=0.02,
            omega=OMEGA,
        )
    with pytest.raises(ValueError, match="active_mask must have shape"):
        advance_rate_servo(
            torch.zeros_like(action),
            action,
            rate,
            acceleration,
            dt=0.02,
            omega=OMEGA,
            active_mask=torch.ones(2, 1, dtype=torch.bool),
        )
