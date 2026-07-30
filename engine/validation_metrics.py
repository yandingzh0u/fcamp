from __future__ import annotations

import torch


class ChunkBoundaryDiagnostics:
    """Collect primitive-level chunk diagnostics without changing control.

    A sample is classified by the offset of the action that caused the
    transition.  Offset zero is a chunk boundary only after an environment has
    already executed at least one action; the first post-reset transition is
    reported separately.  Terminal transitions are legal samples, while an
    environment must be omitted from ``active_mask`` on every later step.
    """

    _DISTRIBUTIONS = (
        "action_delta",
        "action_d2",
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
        self._previous_joint_vel = initial_joint_vel.detach().clone()
        self._previous_root_ang_vel = initial_root_ang_vel.detach().clone()
        self._has_previous_action_delta = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._episode_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._values: dict[str, dict[str, list[torch.Tensor]]] = {
            name: {category: [] for category in self._CATEGORIES}
            for name in self._DISTRIBUTIONS
        }
        # Keep H0-H3 visible in every run while supporting any AMP horizon.
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
        self._previous_joint_vel[selected] = initial_joint_vel[selected]
        self._previous_root_ang_vel[selected] = initial_root_ang_vel[selected]
        self._has_previous_action_delta[selected] = False
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
        boundary = active & (self._episode_steps > 0) & (offsets == 0)
        internal = active & (self._episode_steps > 0) & (offsets != 0)

        action_delta_vector = action - self._previous_action
        action_delta = action_delta_vector.abs().mean(dim=-1)
        action_d2 = (action_delta_vector - self._previous_action_delta).abs().mean(dim=-1)
        joint_vel_jump = (joint_vel - self._previous_joint_vel).abs().mean(dim=-1)
        root_ang_vel_jump = (root_ang_vel - self._previous_root_ang_vel).abs().mean(dim=-1)
        values = {
            "action_delta": action_delta,
            "action_d2": action_d2,
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
        self._previous_joint_vel[active] = joint_vel[active]
        self._previous_root_ang_vel[active] = root_ang_vel[active]
        self._has_previous_action_delta[active] = True
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
