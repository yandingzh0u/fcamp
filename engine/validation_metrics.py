from __future__ import annotations

import torch


class StepDiagnostics:
    """Accumulate true one-step control and tracking diagnostics."""

    _DYNAMICS = (
        "action_delta",
        "action_d2",
        "joint_vel_jump",
        "root_lin_vel_jump",
        "root_ang_vel_jump",
    )
    _TRACKING = (
        "joint_pos_mae",
        "joint_vel_mae",
        "root_lin_vel_mae",
        "root_ang_vel_mae",
    )

    def __init__(
        self,
        *,
        initial_action: torch.Tensor,
        initial_joint_vel: torch.Tensor,
        initial_root_lin_vel: torch.Tensor,
        initial_root_ang_vel: torch.Tensor,
    ) -> None:
        if initial_action.ndim != 2 or initial_joint_vel.ndim != 2:
            raise ValueError(
                "initial action and joint velocity must be rank-2 tensors"
            )
        batch = int(initial_action.shape[0])
        if initial_joint_vel.shape[0] != batch:
            raise ValueError("initial state tensors must have the same batch size")
        for name, value in (
            ("initial_root_lin_vel", initial_root_lin_vel),
            ("initial_root_ang_vel", initial_root_ang_vel),
        ):
            if value.shape != (batch, 3):
                raise ValueError(
                    f"{name} must have shape ({batch}, 3), got {tuple(value.shape)}"
                )

        self.num_envs = batch
        self.device = initial_action.device
        self._previous_action = initial_action.detach().clone()
        self._previous_action_delta = torch.zeros_like(initial_action)
        self._previous_joint_vel = initial_joint_vel.detach().clone()
        self._previous_root_lin_vel = initial_root_lin_vel.detach().clone()
        self._previous_root_ang_vel = initial_root_ang_vel.detach().clone()
        self._episode_steps = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )
        self._dynamics: dict[str, list[torch.Tensor]] = {
            name: [] for name in self._DYNAMICS
        }
        self._tracking: dict[str, list[torch.Tensor]] = {
            name: [] for name in self._TRACKING
        }
        self._initial_action_delta: list[torch.Tensor] = []

    def reset(
        self,
        mask: torch.Tensor,
        *,
        initial_action: torch.Tensor,
        initial_joint_vel: torch.Tensor,
        initial_root_lin_vel: torch.Tensor,
        initial_root_ang_vel: torch.Tensor,
    ) -> None:
        selected = self._validate_mask(mask)
        for name, value, expected in (
            ("initial_action", initial_action, self._previous_action.shape),
            (
                "initial_joint_vel",
                initial_joint_vel,
                self._previous_joint_vel.shape,
            ),
            (
                "initial_root_lin_vel",
                initial_root_lin_vel,
                self._previous_root_lin_vel.shape,
            ),
            (
                "initial_root_ang_vel",
                initial_root_ang_vel,
                self._previous_root_ang_vel.shape,
            ),
        ):
            if value.shape != expected:
                raise ValueError(
                    f"{name} must have shape {tuple(expected)}, got {tuple(value.shape)}"
                )
        self._previous_action[selected] = initial_action[selected]
        self._previous_action_delta[selected] = 0.0
        self._previous_joint_vel[selected] = initial_joint_vel[selected]
        self._previous_root_lin_vel[selected] = initial_root_lin_vel[selected]
        self._previous_root_ang_vel[selected] = initial_root_ang_vel[selected]
        self._episode_steps[selected] = 0

    def update(
        self,
        *,
        active_mask: torch.Tensor,
        action: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        reference_joint_pos: torch.Tensor,
        reference_joint_vel: torch.Tensor,
        reference_root_lin_vel: torch.Tensor,
        reference_root_ang_vel: torch.Tensor,
    ) -> None:
        active = self._validate_mask(active_mask)
        expected_action = self._previous_action.shape
        expected_joint = self._previous_joint_vel.shape
        if action.shape != expected_action:
            raise ValueError(
                f"action must have shape {tuple(expected_action)}, got {tuple(action.shape)}"
            )
        for name, value in (
            ("joint_pos", joint_pos),
            ("joint_vel", joint_vel),
            ("reference_joint_pos", reference_joint_pos),
            ("reference_joint_vel", reference_joint_vel),
        ):
            if value.shape != expected_joint:
                raise ValueError(
                    f"{name} must have shape {tuple(expected_joint)}, got {tuple(value.shape)}"
                )
        expected_root = self._previous_root_lin_vel.shape
        for name, value in (
            ("root_lin_vel", root_lin_vel),
            ("root_ang_vel", root_ang_vel),
            ("reference_root_lin_vel", reference_root_lin_vel),
            ("reference_root_ang_vel", reference_root_ang_vel),
        ):
            if value.shape != expected_root:
                raise ValueError(
                    f"{name} must have shape {tuple(expected_root)}, got {tuple(value.shape)}"
                )

        first = active & (self._episode_steps == 0)
        consecutive = active & (self._episode_steps > 0)
        action_delta_vector = action - self._previous_action
        if bool(first.any()):
            self._initial_action_delta.append(
                action_delta_vector.abs().mean(dim=-1)[first].detach().clone()
            )
        dynamics = {
            "action_delta": action_delta_vector.abs().mean(dim=-1),
            "action_d2": (
                action_delta_vector - self._previous_action_delta
            ).abs().mean(dim=-1),
            "joint_vel_jump": (
                joint_vel - self._previous_joint_vel
            ).abs().mean(dim=-1),
            "root_lin_vel_jump": (
                root_lin_vel - self._previous_root_lin_vel
            ).abs().mean(dim=-1),
            "root_ang_vel_jump": (
                root_ang_vel - self._previous_root_ang_vel
            ).abs().mean(dim=-1),
        }
        if bool(consecutive.any()):
            for name, value in dynamics.items():
                self._dynamics[name].append(
                    value[consecutive].detach().clone()
                )
        tracking = {
            "joint_pos_mae": (
                joint_pos - reference_joint_pos
            ).abs().mean(dim=-1),
            "joint_vel_mae": (
                joint_vel - reference_joint_vel
            ).abs().mean(dim=-1),
            "root_lin_vel_mae": (
                root_lin_vel - reference_root_lin_vel
            ).abs().mean(dim=-1),
            "root_ang_vel_mae": (
                root_ang_vel - reference_root_ang_vel
            ).abs().mean(dim=-1),
        }
        if bool(active.any()):
            for name, value in tracking.items():
                self._tracking[name].append(value[active].detach().clone())

        self._previous_action[active] = action[active]
        self._previous_action_delta[active] = action_delta_vector[active]
        self._previous_joint_vel[active] = joint_vel[active]
        self._previous_root_lin_vel[active] = root_lin_vel[active]
        self._previous_root_ang_vel[active] = root_ang_vel[active]
        self._episode_steps[active] += 1

    def metrics(self, prefix: str = "validation") -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name, parts in self._dynamics.items():
            for statistic, value in _distribution_summary(parts).items():
                metrics[f"{prefix}/dynamics/{name}/{statistic}"] = value
        for name, parts in self._tracking.items():
            for statistic, value in _distribution_summary(parts).items():
                metrics[f"{prefix}/tracking/{name}/{statistic}"] = value
        for statistic, value in _distribution_summary(
            self._initial_action_delta
        ).items():
            metrics[f"{prefix}/initial/action_delta/{statistic}"] = value
        return metrics

    def _validate_mask(self, mask: torch.Tensor) -> torch.Tensor:
        selected = mask.to(device=self.device, dtype=torch.bool).reshape(-1)
        if selected.shape != (self.num_envs,):
            raise ValueError(
                f"mask must contain {self.num_envs} values, got {tuple(selected.shape)}"
            )
        return selected


def _distribution_summary(parts: list[torch.Tensor]) -> dict[str, float]:
    if not parts:
        return {"count": 0.0, "mean": -1.0, "p95": -1.0, "p99": -1.0}
    values = torch.cat(parts).float()
    if values.numel() == 0:
        return {"count": 0.0, "mean": -1.0, "p95": -1.0, "p99": -1.0}
    return {
        "count": float(values.numel()),
        "mean": float(values.mean().item()),
        "p95": float(torch.quantile(values, 0.95).item()),
        "p99": float(torch.quantile(values, 0.99).item()),
    }


def terminal_phase_metrics(
    prefix: str,
    phases: torch.Tensor,
    mask: torch.Tensor,
    *,
    motion_end_phase: float,
) -> dict[str, float]:
    selected = phases[mask.bool()].float()
    metrics = {f"{prefix}/count": float(selected.numel())}
    if selected.numel() == 0:
        for name in ("min", "mean", "p50", "p95", "max", "progress_mean"):
            metrics[f"{prefix}/phase_{name}"] = -1.0
        return metrics
    metrics.update(
        {
            f"{prefix}/phase_min": float(selected.min().item()),
            f"{prefix}/phase_mean": float(selected.mean().item()),
            f"{prefix}/phase_p50": float(
                torch.quantile(selected, 0.50).item()
            ),
            f"{prefix}/phase_p95": float(
                torch.quantile(selected, 0.95).item()
            ),
            f"{prefix}/phase_max": float(selected.max().item()),
            f"{prefix}/phase_progress_mean": float(
                (selected / max(float(motion_end_phase), 1.0)).mean().item()
            ),
        }
    )
    return metrics
