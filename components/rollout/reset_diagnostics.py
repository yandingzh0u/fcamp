"""Exact diagnostics for uniform demonstration-state resets."""

from __future__ import annotations

import math

import torch


class ResetPhaseRecorder:
    """Accumulate the single population of uniform AMP reset phases."""

    def __init__(
        self,
        num_frames: int,
        *,
        start_phase: int,
        end_phase: int | None = None,
        device: torch.device | str,
        log_num_bins: int = 0,
    ) -> None:
        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        if log_num_bins < 0:
            raise ValueError("log_num_bins cannot be negative")
        self.num_frames = int(num_frames)
        self.start_phase = int(start_phase)
        self.end_phase = (
            self.num_frames - 1
            if end_phase is None
            else int(end_phase)
        )
        if not (
            0 <= self.start_phase <= self.end_phase < self.num_frames
        ):
            raise ValueError(
                "reset phase support must satisfy "
                "0 <= start_phase <= end_phase < num_frames"
            )
        self.device = torch.device(device)
        self.num_bins = int(log_num_bins)
        self.histogram = torch.zeros(
            self.num_frames,
            dtype=torch.long,
            device=self.device,
        )
        self.count = torch.zeros((), dtype=torch.long, device=self.device)
        self.phase_sum = torch.zeros(
            (),
            dtype=torch.float64,
            device=self.device,
        )
        self.phase_min = torch.zeros(
            (),
            dtype=torch.float32,
            device=self.device,
        )
        self.phase_max = torch.zeros(
            (),
            dtype=torch.float32,
            device=self.device,
        )
        self.active = False

    @torch.no_grad()
    def begin(self) -> None:
        self.histogram.zero_()
        self.count.zero_()
        self.phase_sum.zero_()
        self.phase_min.fill_(float("inf"))
        self.phase_max.fill_(float("-inf"))
        self.active = True

    @torch.no_grad()
    def record(self, phases: torch.Tensor) -> None:
        if not self.active or phases.numel() == 0:
            return
        values = phases.detach().to(
            device=self.device,
            dtype=torch.float32,
        ).reshape(-1)
        if not bool(torch.isfinite(values).all()):
            raise ValueError("reset phases must be finite")
        frame_indices = values.floor().long().clamp(0, self.num_frames - 1)
        self.histogram += torch.bincount(
            frame_indices,
            minlength=self.num_frames,
        )
        self.count += values.numel()
        self.phase_sum += values.double().sum()
        self.phase_min.copy_(torch.minimum(self.phase_min, values.min()))
        self.phase_max.copy_(torch.maximum(self.phase_max, values.max()))

    @staticmethod
    def _histogram_quantile(histogram: torch.Tensor, q: float) -> float:
        count = int(histogram.sum().item())
        if count == 0:
            return -1.0
        rank = max(1, int(math.ceil(float(q) * count)))
        index = torch.searchsorted(
            histogram.cumsum(dim=0),
            torch.tensor(rank, device=histogram.device, dtype=histogram.dtype),
        )
        return float(index.item())

    def _bin_metrics(
        self,
        prefix: str,
        histogram: torch.Tensor,
        count: int,
    ) -> dict[str, float]:
        if self.num_bins == 0:
            return {}
        frame_ids = torch.arange(
            self.num_frames,
            dtype=torch.long,
            device=self.device,
        )
        bin_ids = torch.clamp(
            frame_ids * self.num_bins // self.num_frames,
            max=self.num_bins - 1,
        )
        bin_counts = torch.zeros(
            self.num_bins,
            dtype=torch.long,
            device=self.device,
        )
        bin_counts.scatter_add_(0, bin_ids, histogram)
        metrics: dict[str, float] = {}
        for index, bin_count in enumerate(bin_counts):
            value = float(bin_count.item())
            metrics[f"{prefix}/bin_{index}_count"] = value
            metrics[f"{prefix}/bin_{index}_fraction"] = (
                value / float(count) if count > 0 else 0.0
            )
        return metrics

    def _metrics(
        self,
        prefix: str,
        *,
        histogram: torch.Tensor,
        count: int,
        phase_sum: float,
        phase_min: float,
        phase_max: float,
    ) -> dict[str, float]:
        first_index = max(
            0,
            min(self.start_phase, self.num_frames - 1),
        )
        first_bin_count = float(histogram[first_index].item())
        metrics = {
            f"{prefix}/count": float(count),
            f"{prefix}/first_frame_bin_count": first_bin_count,
            f"{prefix}/first_frame_bin_fraction": (
                first_bin_count / float(count) if count > 0 else 0.0
            ),
            f"{prefix}/other_frame_bin_count": (
                float(count) - first_bin_count
            ),
        }
        if count == 0:
            for name in ("min", "mean", "p50", "p95", "max"):
                metrics[f"{prefix}/phase_{name}"] = -1.0
        else:
            metrics.update(
                {
                    f"{prefix}/phase_min": phase_min,
                    f"{prefix}/phase_mean": phase_sum / float(count),
                    f"{prefix}/phase_p50": self._histogram_quantile(
                        histogram,
                        0.50,
                    ),
                    f"{prefix}/phase_p95": self._histogram_quantile(
                        histogram,
                        0.95,
                    ),
                    f"{prefix}/phase_max": phase_max,
                }
            )
        metrics.update(self._bin_metrics(prefix, histogram, count))
        return metrics

    @torch.no_grad()
    def finish(self) -> dict[str, float]:
        self.active = False
        count = int(self.count.item())
        metrics = self._metrics(
            "train_reset/all",
            histogram=self.histogram,
            count=count,
            phase_sum=float(self.phase_sum.item()),
            phase_min=(
                float(self.phase_min.item())
                if count > 0
                else -1.0
            ),
            phase_max=(
                float(self.phase_max.item())
                if count > 0
                else -1.0
            ),
        )
        metrics.update(
            {
                "train_reset/uniform_contract": 1.0,
                "train_reset/uniform_low": float(self.start_phase),
                "train_reset/uniform_high_exclusive": float(
                    self.end_phase
                ),
                "train_reset/uniform_floor_bin_count": float(
                    max(1, self.end_phase - self.start_phase)
                ),
            }
        )
        return metrics
