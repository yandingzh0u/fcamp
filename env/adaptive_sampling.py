from __future__ import annotations

import torch

# Bumped whenever the on-disk sampler state layout changes. Old checkpoints that carry the
# pre-Holosoma failure/exposure stats (version mismatch / absent) are NOT restored.
ADAPTIVE_SAMPLER_VERSION = 2


class AdaptiveTimestepsSampler:
    """Holosoma death-frame adaptive motion sampler.

    Bins motion failures over the GLOBAL motion-frame axis and oversamples the frames the
    robot dies at most. The death frame maps directly to a bin:

        failed_bin = clamp((failed_at_time_step * num_bins) // motion_time_step_total,
                           0, num_bins - 1)

    The sampler is maintained entirely by ``env.step()``: each rl-environment step the
    current step's failures are folded into an exponential moving average and the per-step
    accumulator is zeroed (``update_bin_failed_count``). There is no exposure tracking and no
    start-frame bookkeeping -- the bin counts are the EMA of raw failure counts.
    """

    def __init__(
        self,
        motion_time_step_total: int,
        device: torch.device | str,
        *,
        num_bins: int,
        adaptive_kernel_size: int = 1,
        adaptive_lambda: float = 0.8,
        adaptive_uniform_ratio: float = 0.1,
        adaptive_alpha: float = 0.001,
    ):
        self.device = device
        self.motion_time_step_total = int(max(1, motion_time_step_total))
        self.num_bins = int(max(1, num_bins))
        self.adaptive_kernel_size = int(max(1, adaptive_kernel_size))
        self.adaptive_lambda = float(adaptive_lambda)
        self.adaptive_uniform_ratio = min(1.0, max(0.0, float(adaptive_uniform_ratio)))
        self.adaptive_alpha = min(1.0, max(0.0, float(adaptive_alpha)))
        kernel = torch.tensor(
            [self.adaptive_lambda**i for i in range(self.adaptive_kernel_size)],
            dtype=torch.float32,
            device=self.device,
        )
        self.kernel = kernel / kernel.sum()
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

    def update_current_bin_failed_count(self, failed_at_time_step: torch.Tensor) -> None:
        """Accumulate this step's death frames into the per-step failure counter."""
        if failed_at_time_step.numel() == 0:
            return
        failed_bin = self.frames_to_bins(failed_at_time_step)
        self.current_bin_failed_count += torch.bincount(failed_bin, minlength=self.num_bins).to(
            self.current_bin_failed_count
        )

    def update_bin_failed_count(self) -> None:
        """Fold the per-step failure counter into the EMA, then zero it. Called every step."""
        self.bin_failed_count = (self.adaptive_alpha * self.current_bin_failed_count) + (
            1.0 - self.adaptive_alpha
        ) * self.bin_failed_count
        self.current_bin_failed_count.zero_()

    @property
    def sampling_probabilities(self) -> torch.Tensor:
        # p_b proportional to (failure EMA + uniform floor), smoothed by the non-causal kernel.
        probabilities = self.bin_failed_count + self.adaptive_uniform_ratio / float(self.num_bins)
        probabilities = torch.nn.functional.pad(
            probabilities.view(1, 1, -1),
            (0, self.adaptive_kernel_size - 1),
            mode="replicate",
        )
        probabilities = torch.nn.functional.conv1d(probabilities, self.kernel.view(1, 1, -1)).view(-1)
        total = probabilities.sum()
        if not bool(torch.isfinite(total)) or float(total.item()) <= 0.0:
            return torch.full((self.num_bins,), 1.0 / float(self.num_bins), device=self.device)
        return probabilities / total

    def bin_frame_bounds(self, bins: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lo = (bins * self.motion_time_step_total) // self.num_bins
        hi = ((bins + 1) * self.motion_time_step_total) // self.num_bins
        hi = torch.maximum(hi, lo + 1)
        return lo, hi

    def sample_frames(self, num_samples: int, min_phase: int, max_phase: int) -> torch.Tensor:
        """Conditionally sample global motion frames inside [min_phase, max_phase].

        This is true conditional sampling, NOT a post-hoc clamp: each bin is intersected with
        [min_phase, max_phase], bins with empty intersection are masked out, the remaining
        bin probabilities are renormalized, and offsets are drawn only within each bin's
        intersection. Clamping instead would pile every out-of-range sample onto the two
        boundary frames and create spurious edge spikes."""
        if num_samples <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        min_phase = int(min_phase)
        max_phase = int(max_phase)
        if max_phase < min_phase:
            return torch.full((num_samples,), min_phase, dtype=torch.long, device=self.device)

        all_bins = torch.arange(self.num_bins, device=self.device)
        lo, hi = self.bin_frame_bounds(all_bins)  # global [lo, hi) per bin
        full_span = (hi - lo).clamp_min(1).to(torch.float32)
        # Intersection of each bin with the inclusive range [min_phase, max_phase].
        lo_eff = torch.clamp(lo, min=min_phase)
        hi_eff = torch.clamp(hi, max=max_phase + 1)
        span_eff = (hi_eff - lo_eff).clamp_min(0)  # 0 => no overlap with the range

        # Bin-selection weight = (global frame density P_b / full_span_b) * in-range frame count.
        # This makes the per-FRAME sampling density the exact conditional restriction of the
        # global distribution to [min_phase, max_phase]; without the span_eff/full_span factor a
        # partially-covered boundary bin would be over-/under-sampled per frame.
        probs = self.sampling_probabilities / full_span * span_eff.to(torch.float32)
        total = probs.sum()
        if not bool(torch.isfinite(total)) or float(total.item()) <= 0.0:
            # No in-range failure mass (or degenerate): fall back to per-frame-uniform in range,
            # i.e. bin weight proportional to the number of in-range frames it contributes.
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

    def state_dict(self) -> dict:
        return {
            "version": ADAPTIVE_SAMPLER_VERSION,
            "num_bins": self.num_bins,
            "bin_failed_count": self.bin_failed_count.detach().cpu(),
            "current_bin_failed_count": self.current_bin_failed_count.detach().cpu(),
        }

    def load_state_dict(self, state: dict | None) -> bool:
        """Restore the EMA bins. Returns False (and keeps the fresh zeros) on version /
        shape mismatch -- old failure/exposure stats are intentionally NOT restored."""
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
        return {
            "top_bin": float(top_bin.item()),
            "top_prob": float(top_prob.item()),
            "failed_sum": float(self.bin_failed_count.sum().item()),
            "entropy": float(entropy.item()),
        }
