from __future__ import annotations

import math

import torch

from envs.action_servo import advance_position_servo


class ChunkBoundaryDiagnostics:
    """Collect primitive-level chunk diagnostics without changing control.

    A sample is classified by the offset of the action that caused the
    transition.  The first complete chunk after reset is warm-up for both
    boundary and internal distributions, so their ratio always compares
    identically aged transitions.  The caller excludes the current terminal
    transition, as well as every later inactive step, from ``active_mask``.
    """

    _DISTRIBUTIONS = (
        "action_delta",
        "action_d2",
        "action_d3",
        "joint_vel_jump",
        "root_ang_vel_jump",
    )
    _CATEGORIES = ("boundary", "internal", "reset_first")
    _TRACKING = (
        "joint_pos_error",
        "joint_vel_error",
        "root_ang_vel_error",
    )

    def __init__(
        self,
        *,
        horizon: int,
        initial_action: torch.Tensor,
        initial_joint_vel: torch.Tensor,
        initial_root_ang_vel: torch.Tensor,
    ) -> None:
        if int(horizon) <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if initial_action.ndim != 2 or initial_joint_vel.ndim != 2:
            raise ValueError("initial action and joint velocity must be rank-2 tensors")
        if initial_root_ang_vel.shape != (initial_action.shape[0], 3):
            raise ValueError(
                "initial_root_ang_vel must have shape "
                f"({initial_action.shape[0]}, 3), got {tuple(initial_root_ang_vel.shape)}"
            )
        if initial_joint_vel.shape[0] != initial_action.shape[0]:
            raise ValueError("initial state tensors must have the same batch size")

        self.horizon = int(horizon)
        self.num_envs = int(initial_action.shape[0])
        self.device = initial_action.device
        self._previous_action = initial_action.detach().clone()
        self._previous_action_delta = torch.zeros_like(initial_action)
        self._previous_action_d2 = torch.zeros_like(initial_action)
        self._previous_joint_vel = initial_joint_vel.detach().clone()
        self._previous_root_ang_vel = initial_root_ang_vel.detach().clone()
        self._has_previous_action_delta = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._has_previous_action_d2 = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._episode_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._values: dict[str, dict[str, list[torch.Tensor]]] = {
            name: {category: [] for category in self._CATEGORIES}
            for name in self._DISTRIBUTIONS
        }
        # FCAMP uses H=4.  Keeping at least four slots makes the output schema
        # stable for shorter-horizon smoke tests as well.
        self._tracked_offsets = max(4, self.horizon)
        self._offset_values: dict[str, list[list[torch.Tensor]]] = {
            name: [[] for _ in range(self._tracked_offsets)]
            for name in self._TRACKING
        }
        self._reset_tracking_values: dict[str, list[torch.Tensor]] = {
            name: [] for name in self._TRACKING
        }

    def reset(
        self,
        mask: torch.Tensor,
        *,
        initial_action: torch.Tensor,
        initial_joint_vel: torch.Tensor,
        initial_root_ang_vel: torch.Tensor,
    ) -> None:
        """Start new episodes for selected environments.

        Validation currently performs one reset before the rollout, but this
        explicit operation prevents a future auto-reset caller from joining
        measurements across two episodes.
        """

        selected = self._validate_mask(mask)
        if initial_action.shape != self._previous_action.shape:
            raise ValueError("reset initial_action shape does not match accumulator state")
        if initial_joint_vel.shape != self._previous_joint_vel.shape:
            raise ValueError("reset initial_joint_vel shape does not match accumulator state")
        if initial_root_ang_vel.shape != self._previous_root_ang_vel.shape:
            raise ValueError("reset initial_root_ang_vel shape does not match accumulator state")
        self._previous_action[selected] = initial_action[selected]
        self._previous_action_delta[selected] = 0.0
        self._previous_action_d2[selected] = 0.0
        self._previous_joint_vel[selected] = initial_joint_vel[selected]
        self._previous_root_ang_vel[selected] = initial_root_ang_vel[selected]
        self._has_previous_action_delta[selected] = False
        self._has_previous_action_d2[selected] = False
        self._episode_steps[selected] = 0

    def update(
        self,
        *,
        active_mask: torch.Tensor,
        chunk_offset: int | torch.Tensor,
        action: torch.Tensor,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
        reference_joint_pos: torch.Tensor,
        reference_joint_vel: torch.Tensor,
        reference_root_ang_vel: torch.Tensor,
    ) -> None:
        """Record one post-action transition for every active environment."""

        active = self._validate_mask(active_mask)
        expected_action_shape = self._previous_action.shape
        expected_joint_shape = self._previous_joint_vel.shape
        if action.shape != expected_action_shape:
            raise ValueError(
                f"action must have shape {tuple(expected_action_shape)}, got {tuple(action.shape)}"
            )
        for name, value in (
            ("joint_pos", joint_pos),
            ("joint_vel", joint_vel),
            ("reference_joint_pos", reference_joint_pos),
            ("reference_joint_vel", reference_joint_vel),
        ):
            if value.shape != expected_joint_shape:
                raise ValueError(
                    f"{name} must have shape {tuple(expected_joint_shape)}, got {tuple(value.shape)}"
                )
        for name, value in (
            ("root_ang_vel", root_ang_vel),
            ("reference_root_ang_vel", reference_root_ang_vel),
        ):
            if value.shape != self._previous_root_ang_vel.shape:
                raise ValueError(
                    f"{name} must have shape {tuple(self._previous_root_ang_vel.shape)}, "
                    f"got {tuple(value.shape)}"
                )

        if torch.is_tensor(chunk_offset):
            offsets = chunk_offset.to(device=self.device, dtype=torch.long)
            if offsets.ndim == 0:
                offsets = offsets.expand(self.num_envs)
            else:
                offsets = offsets.reshape(-1)
            if offsets.shape != (self.num_envs,):
                raise ValueError(
                    f"chunk_offset tensor must contain {self.num_envs} values, got {tuple(offsets.shape)}"
                )
        else:
            offsets = torch.full(
                (self.num_envs,), int(chunk_offset), dtype=torch.long, device=self.device
            )
        if bool(((offsets < 0) | (offsets >= self.horizon)).any()):
            raise ValueError(f"chunk offsets must lie in [0, {self.horizon - 1}]")

        reset_first = active & (self._episode_steps == 0)
        ratio_ready = active & (self._episode_steps >= self.horizon)
        boundary = ratio_ready & (offsets == 0)
        internal = ratio_ready & (offsets != 0)

        previous_delta_valid = self._has_previous_action_delta.clone()
        action_delta_vector = action - self._previous_action
        action_delta = action_delta_vector.abs().mean(dim=-1)
        action_d2_vector = action_delta_vector - self._previous_action_delta
        action_d2 = action_d2_vector.abs().mean(dim=-1)
        action_d3 = (
            action_d2_vector - self._previous_action_d2
        ).abs().mean(dim=-1)
        joint_vel_jump = (joint_vel - self._previous_joint_vel).abs().mean(dim=-1)
        root_ang_vel_jump = (root_ang_vel - self._previous_root_ang_vel).abs().mean(dim=-1)
        values = {
            "action_delta": action_delta,
            "action_d2": action_d2,
            "action_d3": action_d3,
            "joint_vel_jump": joint_vel_jump,
            "root_ang_vel_jump": root_ang_vel_jump,
        }
        category_masks = {
            "boundary": boundary,
            "internal": internal,
            "reset_first": reset_first,
        }
        for name, value in values.items():
            for category, mask in category_masks.items():
                # A second difference is undefined on the first transition of
                # an episode because there is no preceding action delta.
                if name == "action_d2":
                    mask = mask & self._has_previous_action_delta
                elif name == "action_d3":
                    mask = mask & self._has_previous_action_d2
                if bool(mask.any()):
                    self._values[name][category].append(value[mask].detach().clone())

        tracking = {
            "joint_pos_error": (joint_pos - reference_joint_pos).abs().mean(dim=-1),
            "joint_vel_error": (joint_vel - reference_joint_vel).abs().mean(dim=-1),
            "root_ang_vel_error": (root_ang_vel - reference_root_ang_vel).abs().mean(dim=-1),
        }
        if bool(reset_first.any()):
            for name, value in tracking.items():
                self._reset_tracking_values[name].append(
                    value[reset_first].detach().clone()
                )
        for offset in range(self._tracked_offsets):
            offset_mask = active & (offsets == offset)
            if not bool(offset_mask.any()):
                continue
            for name, value in tracking.items():
                self._offset_values[name][offset].append(value[offset_mask].detach().clone())

        self._previous_action[active] = action[active]
        self._previous_action_delta[active] = action_delta_vector[active]
        self._previous_action_d2[active] = action_d2_vector[active]
        self._previous_joint_vel[active] = joint_vel[active]
        self._previous_root_ang_vel[active] = root_ang_vel[active]
        self._has_previous_action_delta[active] = True
        self._has_previous_action_d2[active] = previous_delta_valid[active]
        self._episode_steps[active] += 1

    def metrics(self, prefix: str = "validation") -> dict[str, float]:
        metrics: dict[str, float] = {}
        for name in self._DISTRIBUTIONS:
            summaries: dict[str, dict[str, float]] = {}
            for category in self._CATEGORIES:
                summary = _distribution_summary(self._values[name][category])
                summaries[category] = summary
                for statistic, value in summary.items():
                    metrics[f"{prefix}/chunk_{name}_{category}_{statistic}"] = value
            boundary_mean = summaries["boundary"]["mean"]
            internal_mean = summaries["internal"]["mean"]
            if (
                summaries["boundary"]["count"] > 0.0
                and summaries["internal"]["count"] > 0.0
                and internal_mean > 1.0e-12
            ):
                ratio = boundary_mean / internal_mean
            else:
                ratio = -1.0
            metrics[f"{prefix}/chunk_{name}_boundary_internal_ratio"] = float(ratio)

        for name in self._TRACKING:
            reset_summary = _distribution_summary(self._reset_tracking_values[name])
            for statistic, value in reset_summary.items():
                metrics[f"{prefix}/chunk_{name}_reset_first_{statistic}"] = value
            for offset in range(self._tracked_offsets):
                summary = _distribution_summary(self._offset_values[name][offset])
                for statistic, value in summary.items():
                    metrics[f"{prefix}/chunk_offset{offset}_{name}_{statistic}"] = value
        return metrics

    def _validate_mask(self, mask: torch.Tensor) -> torch.Tensor:
        selected = mask.to(device=self.device, dtype=torch.bool).reshape(-1)
        if selected.shape != (self.num_envs,):
            raise ValueError(
                f"mask must contain {self.num_envs} values, got {tuple(selected.shape)}"
            )
        return selected


class C2ServoDiagnostics:
    """Measure the exact C2 target-action servo and its chunk-offset behavior."""

    _VALUES = (
        "target_action_abs",
        "target_increment_abs",
        "target_increment_d2_abs",
        "previous_command_rate_abs",
        "previous_command_acceleration_abs",
        "actual_command_acceleration_abs",
        "position_error_abs",
        "initial_jerk_abs",
        "command_acceleration_delta_abs",
        "predicted_action_d2_abs",
        "actual_action_d2_abs",
        "reference_action_delta_abs",
        "reference_action_d2_abs",
        "predicted_action_d3_abs",
        "actual_action_d3_abs",
        "reference_action_d3_abs",
        "prediction_residual_abs",
        "rate_prediction_residual_abs",
        "acceleration_prediction_residual_abs",
        "projection_joint_fraction",
    )

    def __init__(
        self,
        *,
        horizon: int,
        control_dt: float,
        omega: float,
        initial_action: torch.Tensor,
        initial_reference_action: torch.Tensor,
        transition_end_phase_window: tuple[float, float] = (280.0, 310.0),
    ) -> None:
        if int(horizon) <= 0:
            raise ValueError(f"horizon must be positive, got {horizon}")
        if initial_action.ndim != 2:
            raise ValueError("initial_action must be rank 2")
        if initial_reference_action.shape != initial_action.shape:
            raise ValueError(
                "initial_reference_action must match initial_action"
            )
        control_dt = float(control_dt)
        omega = float(omega)
        if not math.isfinite(control_dt) or control_dt <= 0.0:
            raise ValueError("control_dt must be finite and positive")
        if not math.isfinite(omega) or omega <= 0.0:
            raise ValueError("omega must be finite and positive")
        phase_low, phase_high = map(float, transition_end_phase_window)
        if (
            not math.isfinite(phase_low)
            or not math.isfinite(phase_high)
            or phase_low > phase_high
        ):
            raise ValueError(
                "transition_end_phase_window must be finite and ordered"
            )

        self.horizon = int(horizon)
        self.control_dt = control_dt
        self.omega = omega
        self.num_envs, self.action_dim = initial_action.shape
        self.device = initial_action.device
        self.phase_low = phase_low
        self.phase_high = phase_high
        self._phase_label = (
            f"transition_end_phase{_metric_number(phase_low)}_{_metric_number(phase_high)}"
        )
        self._previous_action = initial_action.detach().clone()
        self._previous_action_delta = torch.zeros_like(initial_action)
        self._previous_action_d2 = torch.zeros_like(initial_action)
        self._previous_reference_action = (
            initial_reference_action.detach().clone()
        )
        self._previous_reference_delta = torch.zeros_like(initial_action)
        self._previous_reference_d2 = torch.zeros_like(initial_action)
        self._previous_target_increment = torch.zeros_like(initial_action)
        self._has_previous_action_delta = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._has_previous_action_d2 = torch.zeros_like(
            self._has_previous_action_delta
        )
        self._has_previous_reference_delta = torch.zeros_like(
            self._has_previous_action_delta
        )
        self._has_previous_reference_d2 = torch.zeros_like(
            self._has_previous_action_delta
        )
        self._has_previous_target_increment = torch.zeros_like(
            self._has_previous_action_delta
        )
        self._episode_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._tracked_offsets = max(4, self.horizon)
        self._offset_values = {
            name: [[] for _ in range(self._tracked_offsets)]
            for name in self._VALUES
        }
        self._phase_offset_values = {
            name: [[] for _ in range(self._tracked_offsets)]
            for name in self._VALUES
        }

    def update(
        self,
        *,
        active_mask: torch.Tensor,
        chunk_offset: int | torch.Tensor,
        transition_end_phase_steps: torch.Tensor,
        target_action: torch.Tensor,
        previous_target_action: torch.Tensor,
        previous_action: torch.Tensor,
        previous_command_rate: torch.Tensor,
        previous_command_acceleration: torch.Tensor,
        actual_action: torch.Tensor,
        actual_command_rate: torch.Tensor,
        actual_command_acceleration: torch.Tensor,
        action_projection_mask: torch.Tensor,
        reference_action: torch.Tensor,
    ) -> None:
        active = self._validate_mask(active_mask)
        expected = (self.num_envs, self.action_dim)
        for name, value in (
            ("target_action", target_action),
            ("previous_target_action", previous_target_action),
            ("previous_action", previous_action),
            ("previous_command_rate", previous_command_rate),
            (
                "previous_command_acceleration",
                previous_command_acceleration,
            ),
            ("actual_action", actual_action),
            ("actual_command_rate", actual_command_rate),
            (
                "actual_command_acceleration",
                actual_command_acceleration,
            ),
            ("reference_action", reference_action),
        ):
            if value.shape != expected:
                raise ValueError(
                    f"{name} must have shape {expected}, got {tuple(value.shape)}"
                )
        if action_projection_mask.shape != expected:
            raise ValueError(
                "action_projection_mask must have shape "
                f"{expected}, got {tuple(action_projection_mask.shape)}"
            )
        if not torch.equal(
            previous_action[active], self._previous_action[active]
        ):
            raise RuntimeError(
                "validation previous_action differs from servo history"
            )

        phases = transition_end_phase_steps.to(
            device=self.device, dtype=torch.float32
        ).reshape(-1)
        if phases.shape != (self.num_envs,):
            raise ValueError(
                "transition_end_phase_steps must contain "
                f"{self.num_envs} values, got {tuple(phases.shape)}"
            )

        predicted_action, predicted_rate, predicted_acceleration = (
            advance_position_servo(
                target_action,
                previous_action,
                previous_command_rate,
                previous_command_acceleration,
                dt=self.control_dt,
                omega=self.omega,
            )
        )
        predicted_delta = predicted_action - previous_action
        actual_delta = actual_action - previous_action
        predicted_d2 = predicted_delta - self._previous_action_delta
        actual_d2 = actual_delta - self._previous_action_delta
        predicted_d3 = predicted_d2 - self._previous_action_d2
        actual_d3 = actual_d2 - self._previous_action_d2
        prediction_residual = actual_action - predicted_action

        reference_delta = reference_action - self._previous_reference_action
        reference_d2 = reference_delta - self._previous_reference_delta
        reference_d3 = reference_d2 - self._previous_reference_d2
        target_increment = target_action - previous_target_action
        target_increment_d2 = (
            target_increment - self._previous_target_increment
        )
        position_error = target_action - previous_action
        initial_jerk = (
            self.omega**3 * position_error
            - 3.0 * self.omega**2 * previous_command_rate
            - 3.0 * self.omega * previous_command_acceleration
        )
        values = {
            "target_action_abs": target_action.abs().mean(dim=-1),
            "target_increment_abs": target_increment.abs().mean(dim=-1),
            "target_increment_d2_abs": (
                target_increment_d2.abs().mean(dim=-1)
            ),
            "previous_command_rate_abs": (
                previous_command_rate.abs().mean(dim=-1)
            ),
            "previous_command_acceleration_abs": (
                previous_command_acceleration.abs().mean(dim=-1)
            ),
            "actual_command_acceleration_abs": (
                actual_command_acceleration.abs().mean(dim=-1)
            ),
            "position_error_abs": position_error.abs().mean(dim=-1),
            "initial_jerk_abs": initial_jerk.abs().mean(dim=-1),
            "command_acceleration_delta_abs": (
                actual_command_acceleration
                - previous_command_acceleration
            ).abs().mean(dim=-1),
            "predicted_action_d2_abs": predicted_d2.abs().mean(dim=-1),
            "actual_action_d2_abs": actual_d2.abs().mean(dim=-1),
            "reference_action_delta_abs": (
                reference_delta.abs().mean(dim=-1)
            ),
            "reference_action_d2_abs": reference_d2.abs().mean(dim=-1),
            "predicted_action_d3_abs": predicted_d3.abs().mean(dim=-1),
            "actual_action_d3_abs": actual_d3.abs().mean(dim=-1),
            "reference_action_d3_abs": reference_d3.abs().mean(dim=-1),
            "prediction_residual_abs": (
                prediction_residual.abs().mean(dim=-1)
            ),
            "rate_prediction_residual_abs": (
                actual_command_rate - predicted_rate
            ).abs().mean(dim=-1),
            "acceleration_prediction_residual_abs": (
                actual_command_acceleration - predicted_acceleration
            ).abs().mean(dim=-1),
            "projection_joint_fraction": action_projection_mask.to(
                device=self.device, dtype=torch.float32
            ).mean(dim=-1),
        }

        eligible = (
            active
            & (self._episode_steps >= self.horizon)
            & self._has_previous_action_d2
            & self._has_previous_reference_d2
            & self._has_previous_target_increment
        )
        in_phase_window = (
            eligible
            & (phases >= self.phase_low)
            & (phases <= self.phase_high)
        )
        offsets = self._offset_tensor(chunk_offset)
        for offset in range(self._tracked_offsets):
            offset_mask = eligible & (offsets == offset)
            phase_offset_mask = in_phase_window & (offsets == offset)
            if not bool(offset_mask.any()) and not bool(
                phase_offset_mask.any()
            ):
                continue
            for name, value in values.items():
                if bool(offset_mask.any()):
                    self._offset_values[name][offset].append(
                        value[offset_mask].detach().clone()
                    )
                if bool(phase_offset_mask.any()):
                    self._phase_offset_values[name][offset].append(
                        value[phase_offset_mask].detach().clone()
                    )

        had_action_delta = self._has_previous_action_delta.clone()
        had_reference_delta = self._has_previous_reference_delta.clone()
        self._previous_action[active] = actual_action[active]
        self._previous_action_delta[active] = actual_delta[active]
        self._previous_action_d2[active] = actual_d2[active]
        self._previous_reference_action[active] = reference_action[active]
        self._previous_reference_delta[active] = reference_delta[active]
        self._previous_reference_d2[active] = reference_d2[active]
        self._previous_target_increment[active] = target_increment[active]
        self._has_previous_action_delta[active] = True
        self._has_previous_action_d2[active] = had_action_delta[active]
        self._has_previous_reference_delta[active] = True
        self._has_previous_reference_d2[active] = had_reference_delta[active]
        self._has_previous_target_increment[active] = True
        self._episode_steps[active] += 1

    def metrics(self, prefix: str = "validation") -> dict[str, float]:
        metrics: dict[str, float] = {}
        self._add_group_metrics(
            metrics,
            prefix=prefix,
            group_prefix="servo",
            values=self._offset_values,
        )
        self._add_group_metrics(
            metrics,
            prefix=prefix,
            group_prefix=f"servo_{self._phase_label}",
            values=self._phase_offset_values,
        )
        return metrics

    def _add_group_metrics(
        self,
        metrics: dict[str, float],
        *,
        prefix: str,
        group_prefix: str,
        values: dict[str, list[list[torch.Tensor]]],
    ) -> None:
        for name in self._VALUES:
            for offset in range(self._tracked_offsets):
                summary = _distribution_summary(values[name][offset])
                for statistic, value in summary.items():
                    metrics[
                        f"{prefix}/{group_prefix}_offset{offset}_{name}_{statistic}"
                    ] = value

            boundary_summary = _distribution_summary(values[name][0])
            internal_parts = [
                part
                for offset in range(1, self.horizon)
                for part in values[name][offset]
            ]
            internal_summary = _distribution_summary(internal_parts)
            for category, summary in (
                ("boundary", boundary_summary),
                ("internal", internal_summary),
            ):
                for statistic, value in summary.items():
                    metrics[
                        f"{prefix}/{group_prefix}_{name}_{category}_{statistic}"
                    ] = value
            if (
                boundary_summary["count"] > 0.0
                and internal_summary["count"] > 0.0
                and internal_summary["mean"] > 1.0e-12
            ):
                ratio = boundary_summary["mean"] / internal_summary["mean"]
            else:
                ratio = -1.0
            metrics[
                f"{prefix}/{group_prefix}_{name}_boundary_internal_ratio"
            ] = float(ratio)

    def _validate_mask(self, mask: torch.Tensor) -> torch.Tensor:
        selected = mask.to(device=self.device, dtype=torch.bool).reshape(-1)
        if selected.shape != (self.num_envs,):
            raise ValueError(
                f"mask must contain {self.num_envs} values, got {tuple(selected.shape)}"
            )
        return selected

    def _offset_tensor(self, chunk_offset: int | torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(chunk_offset):
            offsets = chunk_offset.to(device=self.device, dtype=torch.long)
            if offsets.ndim == 0:
                offsets = offsets.expand(self.num_envs)
            else:
                offsets = offsets.reshape(-1)
            if offsets.shape != (self.num_envs,):
                raise ValueError(
                    "chunk_offset tensor must contain "
                    f"{self.num_envs} values, got {tuple(offsets.shape)}"
                )
        else:
            offsets = torch.full(
                (self.num_envs,),
                int(chunk_offset),
                dtype=torch.long,
                device=self.device,
            )
        if bool(((offsets < 0) | (offsets >= self.horizon)).any()):
            raise ValueError(f"chunk offsets must lie in [0, {self.horizon - 1}]")
        return offsets


def _metric_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}".replace("-", "m").replace(".", "p")


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
            f"{prefix}/phase_p50": float(torch.quantile(selected, 0.50).item()),
            f"{prefix}/phase_p95": float(torch.quantile(selected, 0.95).item()),
            f"{prefix}/phase_max": float(selected.max().item()),
            f"{prefix}/phase_progress_mean": float(
                (selected / max(float(motion_end_phase), 1.0)).mean().item()
            ),
        }
    )
    return metrics
