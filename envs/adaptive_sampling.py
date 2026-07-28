from __future__ import annotations

import torch


ADAPTIVE_SAMPLER_VERSION = 5


class AdaptiveTimestepsSampler:
    """Failure-biased sampler restored for the existing mimic environment.

    The sampler keeps the historical behaviour of the last Flow-CPS-compatible
    implementation: failures are accumulated per simulation step, folded into
    an EMA, and sampling is biased towards predecessor bins.
    """

    def __init__(
        self,
        motion_time_step_total: int,
        device: torch.device | str,
        *,
        num_bins: int = 0,
        env_fps: int = 50,
        adaptive_alpha: float = 0.001,
        adaptive_predecessor_ratio: float = 0.8,
        adaptive_predecessor_lookback_bins: int = 1,
    ):
        self.device = device
        self.motion_time_step_total = int(max(1, motion_time_step_total))
        self.num_frames = self.motion_time_step_total
        if int(num_bins) <= 0:
            num_bins = self.num_frames // int(max(1, env_fps)) + 1
        self.num_bins = int(max(1, num_bins))
        self.adaptive_alpha = min(1.0, max(0.0, float(adaptive_alpha)))
        self.adaptive_predecessor_ratio = min(1.0, max(0.0, float(adaptive_predecessor_ratio)))
        self.adaptive_predecessor_lookback_bins = max(1, int(adaptive_predecessor_lookback_bins))
        self.init_buffers()

    def init_buffers(self) -> None:
        self.current_bin_failed_count = torch.zeros(self.num_bins, dtype=torch.float32, device=self.device)
        self.bin_failed_count = torch.zeros(self.num_bins, dtype=torch.float32, device=self.device)

    def frames_to_bins(self, failed_at_time_step: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            (failed_at_time_step.reshape(-1).long() * self.num_bins) // self.motion_time_step_total,
            0,
            self.num_bins - 1,
        )

    def update_current_failure_count(self, failed_at_time_step: torch.Tensor) -> None:
        if failed_at_time_step.numel() == 0:
            return
        idx = torch.clamp(failed_at_time_step.reshape(-1).long(), 0, self.num_frames - 1)
        failed_bin = self.frames_to_bins(idx)
        self.current_bin_failed_count += torch.bincount(failed_bin, minlength=self.num_bins).to(
            self.current_bin_failed_count
        )

    def update_failure_ema(self) -> None:
        a = self.adaptive_alpha
        self.bin_failed_count = a * self.current_bin_failed_count + (1.0 - a) * self.bin_failed_count
        self.current_bin_failed_count.zero_()

    @property
    def sampling_probabilities(self) -> torch.Tensor:
        uniform = torch.full(
            (self.num_bins,), 1.0 / float(self.num_bins), dtype=torch.float32, device=self.device
        )
        failure = self.bin_failed_count.clamp_min(0.0)
        failure_total = failure.sum()
        if not bool(torch.isfinite(failure_total)) or float(failure_total.item()) <= 0.0:
            return uniform
        death_bins = torch.arange(self.num_bins, device=self.device)
        predecessor_bins = torch.clamp(death_bins - self.adaptive_predecessor_lookback_bins, min=0)
        predecessor = torch.zeros_like(failure)
        predecessor.scatter_add_(0, predecessor_bins, failure)
        predecessor /= predecessor.sum()
        r = self.adaptive_predecessor_ratio
        return r * predecessor + (1.0 - r) * uniform

    def bin_frame_bounds(self, bins: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lo = (bins * self.motion_time_step_total) // self.num_bins
        hi = ((bins + 1) * self.motion_time_step_total) // self.num_bins
        hi = torch.maximum(hi, lo + 1)
        return lo, hi

    def _sample_predecessor(self, num_samples: int, min_phase: int, max_phase: int) -> torch.Tensor:
        if num_samples <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        all_bins = torch.arange(self.num_bins, device=self.device)
        lo, hi = self.bin_frame_bounds(all_bins)
        full_span = (hi - lo).clamp_min(1).to(torch.float32)
        lo_eff = torch.clamp(lo, min=min_phase)
        hi_eff = torch.clamp(hi, max=max_phase + 1)
        span_eff = (hi_eff - lo_eff).clamp_min(0)
        probs = self.sampling_probabilities / full_span * span_eff.to(torch.float32)
        total = probs.sum()
        if not bool(torch.isfinite(total)) or float(total.item()) <= 0.0:
            probs = span_eff.to(torch.float32)
            total = probs.sum()
            if float(total.item()) <= 0.0:
                return torch.full((num_samples,), min_phase, dtype=torch.long, device=self.device)
        probs = probs / total
        bins = torch.multinomial(probs, num_samples, replacement=True)
        bin_lo = lo_eff.index_select(0, bins)
        bin_span = span_eff.index_select(0, bins).clamp_min(1)
        offset = (torch.rand(num_samples, device=self.device) * bin_span.to(torch.float32)).long()
        frames = bin_lo + offset
        return torch.clamp(frames, min=min_phase, max=max_phase)

    def sample_frames(self, num_samples: int, min_phase: int, max_phase: int) -> torch.Tensor:
        if num_samples <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        min_phase = max(0, int(min_phase))
        max_phase = min(self.num_frames - 1, int(max_phase))
        if max_phase < min_phase:
            return torch.full((num_samples,), min_phase, dtype=torch.long, device=self.device)
        return self._sample_predecessor(num_samples, min_phase, max_phase)

    def state_dict(self) -> dict:
        return {
            "version": ADAPTIVE_SAMPLER_VERSION,
            "num_bins": self.num_bins,
            "num_frames": self.num_frames,
            "bin_failed_count": self.bin_failed_count.detach().cpu(),
            "current_bin_failed_count": self.current_bin_failed_count.detach().cpu(),
        }

    def load_state_dict(self, state: dict | None) -> bool:
        if not state:
            return False
        if int(state.get("version", -1)) != ADAPTIVE_SAMPLER_VERSION:
            return False
        bfc = state.get("bin_failed_count")
        if bfc is None or tuple(bfc.shape) != (self.num_bins,):
            return False
        self.bin_failed_count.copy_(bfc.to(self.bin_failed_count))
        cbfc = state.get("current_bin_failed_count")
        if cbfc is not None and tuple(cbfc.shape) == (self.num_bins,):
            self.current_bin_failed_count.copy_(cbfc.to(self.current_bin_failed_count))
        return True

    def stats(self) -> dict[str, float]:
        probabilities = self.sampling_probabilities
        top_prob, top_bin = probabilities.max(dim=0)
        entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
        if self.num_bins > 1:
            entropy = entropy / torch.log(torch.tensor(float(self.num_bins), device=self.device))
        else:
            entropy = torch.ones_like(entropy)
        failed_sum = float(self.bin_failed_count.sum().item())
        peak_bin = int(torch.argmax(self.bin_failed_count).item()) if failed_sum > 0.0 else -1
        stats = {
            "bin_count": float(self.num_bins),
            "top_bin": float(top_bin.item()),
            "top_prob": float(top_prob.item()),
            "failed_sum": failed_sum,
            "entropy": float(entropy.item()),
            "peak_bin": float(peak_bin),
            "predecessor_ratio": float(self.adaptive_predecessor_ratio),
            "predecessor_lookback_bins": float(self.adaptive_predecessor_lookback_bins),
        }
        for index, (probability, failure) in enumerate(
            zip(probabilities, self.bin_failed_count, strict=True)
        ):
            stats[f"bin_{index}_prob"] = float(probability.item())
            stats[f"bin_{index}_failure_ema"] = float(failure.item())
        return stats
