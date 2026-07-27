"""Exact training-reset phase diagnostics shared by every algorithm."""

from __future__ import annotations

import math

import torch

from .training_streams import CURRICULUM_STREAM, PHASE0_STREAM


class ResetPhaseRecorder:
    """Accumulate true reset events at the environment reset boundary."""

    _ALL = 0
    _PHASE0 = 1
    _CURRICULUM = 2

    def __init__(
        self,
        num_frames: int,
        *,
        start_phase: int,
        device: torch.device | str,
    ) -> None:
        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        self.num_frames = int(num_frames)
        self.start_phase = int(start_phase)
        self.device = torch.device(device)
        self.histogram = torch.zeros(
            3,
            self.num_frames,
            dtype=torch.long,
            device=self.device,
        )
        self.count = torch.zeros(3, dtype=torch.long, device=self.device)
        self.phase_sum = torch.zeros(
            3,
            dtype=torch.float64,
            device=self.device,
        )
        self.phase_min = torch.full(
            (3,),
            float("inf"),
            dtype=torch.float32,
            device=self.device,
        )
        self.phase_max = torch.full(
            (3,),
            float("-inf"),
            dtype=torch.float32,
            device=self.device,
        )
        self.active = False
        self.has_stream_labels = False

    @torch.no_grad()
    def begin(self) -> None:
        self.histogram.zero_()
        self.count.zero_()
        self.phase_sum.zero_()
        self.phase_min.fill_(float("inf"))
        self.phase_max.fill_(float("-inf"))
        self.has_stream_labels = False
        self.active = True

    @torch.no_grad()
    def record(
        self,
        phases: torch.Tensor,
        stream_ids: torch.Tensor | None = None,
    ) -> None:
        if not self.active or phases.numel() == 0:
            return
        values = phases.detach().to(
            device=self.device,
            dtype=torch.float32,
        ).reshape(-1)
        frame_indices = values.floor().long().clamp(0, self.num_frames - 1)
        self._record_group(self._ALL, values, frame_indices)
        if stream_ids is None:
            return
        streams = stream_ids.detach().to(
            device=self.device,
            dtype=torch.int8,
        ).reshape(-1)
        if streams.shape != values.shape:
            raise ValueError("reset phases and stream ids must have the same shape")
        self.has_stream_labels = True
        for stream_id, group in (
            (PHASE0_STREAM, self._PHASE0),
            (CURRICULUM_STREAM, self._CURRICULUM),
        ):
            mask = streams == stream_id
            if bool(mask.any()):
                self._record_group(group, values[mask], frame_indices[mask])

    def _record_group(
        self,
        group: int,
        values: torch.Tensor,
        frame_indices: torch.Tensor,
    ) -> None:
        self.histogram[group] += torch.bincount(
            frame_indices,
            minlength=self.num_frames,
        )
        self.count[group] += values.numel()
        self.phase_sum[group] += values.double().sum()
        self.phase_min[group] = torch.minimum(
            self.phase_min[group],
            values.min(),
        )
        self.phase_max[group] = torch.maximum(
            self.phase_max[group],
            values.max(),
        )

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

    def _group_metrics(
        self,
        group: int,
        prefix: str,
        sampler,
    ) -> dict[str, float]:
        count = int(self.count[group].item())
        histogram = self.histogram[group]
        metrics = {
            f"{prefix}/count": float(count),
            f"{prefix}/start_count": float(
                histogram[
                    max(0, min(self.start_phase, self.num_frames - 1))
                ].item()
            ),
        }
        metrics[f"{prefix}/start_fraction"] = (
            metrics[f"{prefix}/start_count"] / float(count)
            if count > 0
            else 0.0
        )
        metrics[f"{prefix}/nonstart_count"] = (
            float(count) - metrics[f"{prefix}/start_count"]
        )
        if count == 0:
            for name in ("min", "mean", "p50", "p95", "max"):
                metrics[f"{prefix}/phase_{name}"] = -1.0
        else:
            metrics.update(
                {
                    f"{prefix}/phase_min": float(self.phase_min[group].item()),
                    f"{prefix}/phase_mean": float(
                        self.phase_sum[group].item() / count
                    ),
                    f"{prefix}/phase_p50": self._histogram_quantile(
                        histogram, 0.50
                    ),
                    f"{prefix}/phase_p95": self._histogram_quantile(
                        histogram, 0.95
                    ),
                    f"{prefix}/phase_max": float(self.phase_max[group].item()),
                }
            )
        frame_ids = torch.arange(
            self.num_frames,
            dtype=torch.long,
            device=self.device,
        )
        bin_ids = sampler.frames_to_bins(frame_ids).to(self.device)
        num_bins = int(sampler.num_bins)
        bin_counts = torch.zeros(
            num_bins,
            dtype=torch.long,
            device=self.device,
        )
        bin_counts.scatter_add_(0, bin_ids, histogram)
        for index, bin_count in enumerate(bin_counts):
            value = float(bin_count.item())
            metrics[f"{prefix}/bin_{index}_count"] = value
            metrics[f"{prefix}/bin_{index}_fraction"] = (
                value / float(count) if count > 0 else 0.0
            )
        return metrics

    @torch.no_grad()
    def finish(self, sampler) -> dict[str, float]:
        self.active = False
        metrics = self._group_metrics(
            self._ALL,
            "train_reset/all",
            sampler,
        )
        if self.has_stream_labels:
            metrics.update(
                self._group_metrics(
                    self._PHASE0,
                    "train_reset/phase0",
                    sampler,
                )
            )
            metrics.update(
                self._group_metrics(
                    self._CURRICULUM,
                    "train_reset/curriculum",
                    sampler,
                )
            )
        return metrics
