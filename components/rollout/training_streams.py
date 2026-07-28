"""Deterministic phase-zero attempt/curriculum stream support for FCAMP."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


PHASE0_STREAM = 0
CURRICULUM_STREAM = 1


@dataclass(frozen=True)
class Phase0CurriculumStreams:
    """A persistent environment partition with a fixed optimization mixture.

    The phase-zero side is a full-horizon *attempt* stream.  Environments stay
    on the same episode across optimizer updates and reset to phase zero only
    after a real terminal.  Starting at phase zero never implies success.
    """

    stream_ids: torch.Tensor
    phase0_fraction: float
    phase0_start: int

    @classmethod
    def create(
        cls,
        num_envs: int,
        *,
        phase0_fraction: float,
        phase0_start: int,
        device: torch.device | str,
    ) -> "Phase0CurriculumStreams":
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        if not 0.0 < float(phase0_fraction) < 1.0:
            raise ValueError("phase0_fraction must be in (0, 1)")
        if num_envs == 1:
            phase0_count = 1
        else:
            phase0_count = min(
                num_envs - 1,
                max(1, int(round(float(phase0_fraction) * num_envs))),
            )
        stream_ids = torch.full(
            (num_envs,),
            CURRICULUM_STREAM,
            dtype=torch.int8,
            device=device,
        )
        stream_ids[:phase0_count] = PHASE0_STREAM
        return cls(
            stream_ids=stream_ids,
            phase0_fraction=float(phase0_fraction),
            phase0_start=int(phase0_start),
        )

    @property
    def phase0_mask(self) -> torch.Tensor:
        return self.stream_ids == PHASE0_STREAM

    @property
    def curriculum_mask(self) -> torch.Tensor:
        return self.stream_ids == CURRICULUM_STREAM

    @property
    def phase0_ids(self) -> torch.Tensor:
        return self.phase0_mask.nonzero(as_tuple=False).squeeze(-1)

    @property
    def curriculum_ids(self) -> torch.Tensor:
        return self.curriculum_mask.nonzero(as_tuple=False).squeeze(-1)

    def reset_phases(
        self,
        reset_ids: torch.Tensor,
        sample_curriculum: Callable[[int], torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reset phases and per-reset stream ids in ``reset_ids`` order."""

        reset_ids = reset_ids.to(
            device=self.stream_ids.device,
            dtype=torch.long,
        )
        reset_streams = self.stream_ids.index_select(0, reset_ids)
        phases = torch.full(
            (reset_ids.numel(),),
            self.phase0_start,
            dtype=torch.long,
            device=reset_ids.device,
        )
        curriculum_positions = (
            reset_streams == CURRICULUM_STREAM
        ).nonzero(as_tuple=False).squeeze(-1)
        if curriculum_positions.numel() > 0:
            sampled = sample_curriculum(int(curriculum_positions.numel())).to(
                device=reset_ids.device,
                dtype=torch.long,
            )
            if sampled.shape != curriculum_positions.shape:
                raise ValueError(
                    "curriculum phase sampler returned an incompatible shape"
                )
            phases.index_copy_(0, curriculum_positions, sampled)
        return phases, reset_streams

class Phase0AttemptTracker:
    """Track complete phase-zero attempt lifecycles across rollout updates."""

    _OUTCOMES = ("failed", "succeeded", "timed_out", "interrupted")

    def __init__(self, stream_ids: torch.Tensor) -> None:
        self.phase0_mask = (
            stream_ids.detach().to(dtype=torch.int8) == PHASE0_STREAM
        )
        self.active = torch.zeros_like(self.phase0_mask, dtype=torch.bool)
        self.ages = torch.zeros_like(stream_ids, dtype=torch.long)
        self.cumulative = {
            "started": 0,
            **{name: 0 for name in self._OUTCOMES},
        }
        self.current_update = {
            "started": 0,
            **{name: 0 for name in self._OUTCOMES},
        }

    def begin_update(self) -> None:
        for name in self.current_update:
            self.current_update[name] = 0

    def _phase0_ids(self, env_ids: torch.Tensor) -> torch.Tensor:
        ids = torch.as_tensor(
            env_ids,
            device=self.active.device,
            dtype=torch.long,
        )
        if ids.numel() == 0:
            return ids
        return ids[self.phase0_mask.index_select(0, ids)]

    def start(self, env_ids: torch.Tensor) -> None:
        ids = self._phase0_ids(env_ids)
        if ids.numel() == 0:
            return
        if bool(self.active.index_select(0, ids).any()):
            raise RuntimeError("cannot restart an in-flight phase0 attempt")
        self.active[ids] = True
        self.ages[ids] = 0
        count = int(ids.numel())
        self.cumulative["started"] += count
        self.current_update["started"] += count

    def observe_step(
        self,
        alive_before: torch.Tensor,
        done: torch.Tensor,
        failure: torch.Tensor,
        timeout: torch.Tensor,
        motion_complete: torch.Tensor,
    ) -> None:
        participating = alive_before.bool() & self.phase0_mask
        if bool((participating & ~self.active).any()):
            raise RuntimeError("phase0 transition has no active attempt")
        self.ages[participating] += 1

        terminal = done.bool() & self.phase0_mask
        classified = (
            failure.bool() | timeout.bool() | motion_complete.bool()
        ) & self.phase0_mask
        if not torch.equal(terminal, classified):
            raise RuntimeError("phase0 terminal outcome is not uniquely classified")
        if not bool(terminal.any()):
            return

        outcomes = {
            "failed": failure.bool() & terminal,
            "timed_out": timeout.bool() & terminal,
            "succeeded": motion_complete.bool() & terminal,
        }
        if any(
            bool((mask & ~self.active).any())
            for mask in outcomes.values()
        ):
            raise RuntimeError("resolved phase0 attempt is not active")
        for name, mask in outcomes.items():
            count = int(mask.sum().item())
            self.cumulative[name] += count
            self.current_update[name] += count
        self.active[terminal] = False

    def interrupt_inflight(self) -> None:
        count = int((self.active & self.phase0_mask).sum().item())
        if count > 0:
            self.cumulative["interrupted"] += count
            self.current_update["interrupted"] += count
        self.active[self.phase0_mask] = False
        self.ages[self.phase0_mask] = 0

    def state_dict(self) -> dict:
        return {
            "phase0_mask": self.phase0_mask.detach().to("cpu"),
            "active": self.active.detach().to("cpu"),
            "ages": self.ages.detach().to("cpu"),
            "cumulative": dict(self.cumulative),
        }

    def load_state_dict(self, payload: dict | None) -> None:
        if not isinstance(payload, dict):
            raise ValueError("FCAMP checkpoint has no phase0 attempt tracker")
        saved_mask = payload.get("phase0_mask")
        saved_active = payload.get("active")
        saved_ages = payload.get("ages")
        if (
            not torch.is_tensor(saved_mask)
            or not torch.is_tensor(saved_active)
            or not torch.is_tensor(saved_ages)
            or saved_mask.shape != self.phase0_mask.shape
            or saved_active.shape != self.active.shape
            or saved_ages.shape != self.ages.shape
            or not torch.equal(
                saved_mask.to(dtype=torch.bool, device="cpu"),
                self.phase0_mask.detach().to("cpu"),
            )
        ):
            raise ValueError("FCAMP phase0 attempt tracker is incompatible")
        self.active.copy_(
            saved_active.to(device=self.active.device, dtype=torch.bool)
        )
        self.ages.copy_(
            saved_ages.to(device=self.ages.device, dtype=torch.long)
        )
        cumulative = payload.get("cumulative")
        required = {"started", *self._OUTCOMES}
        if not isinstance(cumulative, dict) or set(cumulative) != required:
            raise ValueError("FCAMP phase0 attempt counters are incompatible")
        self.cumulative = {
            name: int(cumulative[name])
            for name in ("started", *self._OUTCOMES)
        }
        self.begin_update()
        self._validate_conservation()

    def _validate_conservation(self) -> None:
        inflight = int((self.active & self.phase0_mask).sum().item())
        resolved = sum(
            self.cumulative[name]
            for name in ("failed", "succeeded", "timed_out")
        )
        expected = resolved + self.cumulative["interrupted"] + inflight
        if self.cumulative["started"] != expected:
            raise RuntimeError(
                "phase0 attempt conservation failed: "
                f"started={self.cumulative['started']} expected={expected}"
            )

    def metrics(self) -> dict[str, float]:
        self._validate_conservation()
        active_ages = self.ages[self.active & self.phase0_mask].float()
        if active_ages.numel() > 0:
            quantiles = torch.quantile(
                active_ages,
                torch.tensor(
                    [0.5, 0.95],
                    device=active_ages.device,
                ),
            )
            age_mean = float(active_ages.mean().item())
            age_p50 = float(quantiles[0].item())
            age_p95 = float(quantiles[1].item())
            age_max = float(active_ages.max().item())
        else:
            age_mean = age_p50 = age_p95 = age_max = -1.0
        cumulative_resolved = sum(
            self.cumulative[name]
            for name in ("failed", "succeeded", "timed_out")
        )
        update_resolved = sum(
            self.current_update[name]
            for name in ("failed", "succeeded", "timed_out")
        )
        prefix = "stream/phase0_attempt"
        metrics = {
            f"{prefix}/inflight_count": float(active_ages.numel()),
            f"{prefix}/inflight_age_mean": age_mean,
            f"{prefix}/inflight_age_p50": age_p50,
            f"{prefix}/inflight_age_p95": age_p95,
            f"{prefix}/inflight_age_max": age_max,
            f"{prefix}/update_resolved": float(update_resolved),
            f"{prefix}/update_success_rate": float(
                self.current_update["succeeded"] / max(update_resolved, 1)
            ),
            f"{prefix}/cumulative_resolved": float(cumulative_resolved),
            f"{prefix}/cumulative_success_rate": float(
                self.cumulative["succeeded"] / max(cumulative_resolved, 1)
            ),
            f"{prefix}/conservation_error": 0.0,
        }
        for name, value in self.current_update.items():
            metrics[f"{prefix}/update_{name}"] = float(value)
        for name, value in self.cumulative.items():
            metrics[f"{prefix}/cumulative_{name}"] = float(value)
        return metrics
