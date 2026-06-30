from __future__ import annotations

import torch

# Bumped whenever the on-disk sampler state layout changes. Old checkpoints that carry an
# incompatible sampler state (version mismatch / absent) are NOT restored -- resuming
# update_1000.pt therefore starts the sampler fresh, as required.
#   v2: per-bin death-frame EMA (~1s bins).
#   v3: per-FRAME failure EMA + causal lookback start sampling (current design).
ADAPTIVE_SAMPLER_VERSION = 3


class AdaptiveTimestepsSampler:
    """Holosoma causal-lookback motion start sampler.

    Maintains a per-FRAME failure EMA over the global motion-frame axis (length
    ``num_frames``). Each real tracking failure at frame ``f`` votes ONLY for START frames in
    the causal lookback window ``[f - lookback_max, f - lookback_min]`` -- i.e. an env is
    spawned ``lookback_min..lookback_max`` frames (0.4..1.6 s at 50 Hz) BEFORE the frame it
    dies at, giving the policy a roll-in to learn the dynamic transition INTO the failure.
    Sampling never lands on or after a failure frame.

        p(s) = hard_ratio * p_failure_lookback(s) + uniform_ratio * p_uniform(s)

    The sampler is maintained entirely by ``env.step()``: each rl-environment step the current
    step's tracking-failure frames are accumulated (``update_current_failure_count``) and folded
    into the EMA at the end of the step (``update_failure_ema``). Failure counts are normalized
    by ``num_envs`` so an 8192-env run is not twice as peaked as an official 4096-env run. All
    three algorithms only consume the env; none write to the sampler.
    """

    def __init__(
        self,
        motion_time_step_total: int,
        device: torch.device | str,
        *,
        num_envs: int,
        lookback_min: int = 20,
        lookback_max: int = 80,
        hard_ratio: float = 0.7,
        uniform_ratio: float = 0.3,
        adaptive_alpha: float = 0.001,
    ):
        self.device = device
        self.num_frames = int(max(1, motion_time_step_total))
        self.num_envs = int(max(1, num_envs))
        self.lookback_min = int(max(1, lookback_min))
        self.lookback_max = int(max(self.lookback_min, lookback_max))
        self.hard_ratio = max(0.0, float(hard_ratio))
        self.uniform_ratio = max(0.0, float(uniform_ratio))
        self.adaptive_alpha = min(1.0, max(0.0, float(adaptive_alpha)))
        self.init_buffers()

    def init_buffers(self) -> None:
        # Raw per-step failure accumulator (zeroed each fold) and the per-frame failure EMA.
        self.current_failure_count = torch.zeros(self.num_frames, dtype=torch.float32, device=self.device)
        self.failure_ema = torch.zeros(self.num_frames, dtype=torch.float32, device=self.device)

    # --------------------------------------------------------------------- failure recording
    def update_current_failure_count(self, failed_at_time_step: torch.Tensor) -> None:
        """Accumulate this step's tracking-failure death frames into the per-step counter."""
        if failed_at_time_step.numel() == 0:
            return
        idx = torch.clamp(failed_at_time_step.reshape(-1).long(), 0, self.num_frames - 1)
        self.current_failure_count += torch.bincount(idx, minlength=self.num_frames).to(
            self.current_failure_count
        )

    def update_failure_ema(self) -> None:
        """Fold the per-step accumulator into the per-frame EMA, then zero it. Called every step.

        The per-step count is normalized by ``num_envs`` BEFORE folding so the EMA magnitude is
        independent of the number of parallel envs (an 8192-env run produces the same sampler
        sharpness as the official 4096-env run)."""
        normalized = self.current_failure_count / float(self.num_envs)
        self.failure_ema = self.adaptive_alpha * normalized + (1.0 - self.adaptive_alpha) * self.failure_ema
        self.current_failure_count.zero_()

    # --------------------------------------------------------------------- sampling
    def _lookback_scores(self) -> torch.Tensor:
        """For each START frame s, the failure mass in its causal lookahead window
        ``[s + lookback_min, s + lookback_max]`` (the failures that this start gives a roll-in
        to). Computed as a sliding window sum of ``failure_ema`` via a prefix sum."""
        ema = self.failure_ema
        n = self.num_frames
        prefix = torch.cat(
            [torch.zeros(1, dtype=torch.float32, device=self.device), torch.cumsum(ema, dim=0)]
        )
        starts = torch.arange(n, device=self.device)
        lo = torch.clamp(starts + self.lookback_min, 0, n)
        hi = torch.clamp(starts + self.lookback_max + 1, 0, n)  # inclusive window upper bound
        return prefix.index_select(0, hi) - prefix.index_select(0, lo)

    def start_probabilities(self, min_phase: int, max_phase: int) -> torch.Tensor:
        """Per-frame start-sampling distribution over the inclusive valid range
        ``[min_phase, max_phase]``: ``hard_ratio`` * (failure-lookback, renormalized in range)
        + ``uniform_ratio`` * (uniform in range). Falls back to pure uniform-in-range when no
        failure mass yet lies ahead of any in-range start."""
        n = self.num_frames
        min_phase = max(0, int(min_phase))
        max_phase = min(n - 1, int(max_phase))
        probs = torch.zeros(n, dtype=torch.float32, device=self.device)
        if max_phase < min_phase:
            probs[min_phase] = 1.0
            return probs
        valid = torch.zeros(n, dtype=torch.float32, device=self.device)
        valid[min_phase : max_phase + 1] = 1.0
        uniform = valid / valid.sum()

        hard = self._lookback_scores() * valid
        hard_total = hard.sum()
        hr = self.hard_ratio
        ur = self.uniform_ratio
        if float(hard_total.item()) > 0.0 and hr > 0.0:
            hard = hard / hard_total
            probs = hr * hard + ur * uniform
        else:
            # No failure mass ahead of any valid start yet -> uniform exploration over the range.
            probs = uniform.clone()
        total = probs.sum()
        if not bool(torch.isfinite(total)) or float(total.item()) <= 0.0:
            return uniform
        return probs / total

    def sample_frames(self, num_samples: int, min_phase: int, max_phase: int) -> torch.Tensor:
        """Sample ``num_samples`` global start frames from ``start_probabilities``. Hard
        (failure-lookback) starts are guaranteed to lie strictly BEFORE the failure frames
        (a start s only earns hard mass from failures at f >= s + lookback_min > s)."""
        if num_samples <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        probs = self.start_probabilities(min_phase, max_phase)
        return torch.multinomial(probs, num_samples, replacement=True).long()

    # --------------------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        return {
            "version": ADAPTIVE_SAMPLER_VERSION,
            "num_frames": self.num_frames,
            "failure_ema": self.failure_ema.detach().cpu(),
            "current_failure_count": self.current_failure_count.detach().cpu(),
        }

    def load_state_dict(self, state: dict | None) -> bool:
        """Restore the per-frame failure EMA. Returns False (keeping fresh zeros) on version /
        shape mismatch -- incompatible (e.g. pre-v3 bin) stats are intentionally NOT restored."""
        if not state:
            return False
        if int(state.get("version", -1)) != ADAPTIVE_SAMPLER_VERSION:
            return False
        ema = state.get("failure_ema")
        if ema is None or tuple(ema.shape) != (self.num_frames,):
            return False
        self.failure_ema.copy_(ema.to(self.failure_ema))
        cur = state.get("current_failure_count")
        if cur is not None and tuple(cur.shape) == (self.num_frames,):
            self.current_failure_count.copy_(cur.to(self.current_failure_count))
        return True

    # --------------------------------------------------------------------- diagnostics
    def _weighted_percentile(self, cdf: torch.Tensor, q: float) -> float:
        idx = torch.searchsorted(cdf, torch.tensor(float(q), device=self.device))
        return float(torch.clamp(idx, 0, self.num_frames - 1).item())

    def stats(self, min_phase: int = 0, max_phase: int | None = None) -> dict[str, float]:
        """Start-distribution diagnostics for the [SAMPLER] log line, computed over the current
        valid range. ``frac_before_peak`` is the share of start mass that lands before the
        dominant failure frame (the "pre-wall" sampling ratio)."""
        if max_phase is None:
            max_phase = self.num_frames - 1
        probs = self.start_probabilities(min_phase, max_phase)
        cdf = torch.cumsum(probs, dim=0)
        failed_sum = float(self.failure_ema.sum().item())
        if failed_sum > 0.0:
            peak_fail_frame = int(torch.argmax(self.failure_ema).item())
            frac_before_peak = float(probs[:peak_fail_frame].sum().item()) if peak_fail_frame > 0 else 0.0
        else:
            peak_fail_frame = -1
            frac_before_peak = float("nan")
        entropy = -(probs * probs.clamp_min(1.0e-12).log()).sum()
        entropy = entropy / torch.log(torch.tensor(float(self.num_frames), device=self.device))
        return {
            "start_p10": self._weighted_percentile(cdf, 0.10),
            "start_p50": self._weighted_percentile(cdf, 0.50),
            "start_p90": self._weighted_percentile(cdf, 0.90),
            "peak_fail_frame": float(peak_fail_frame),
            "frac_before_peak": frac_before_peak,
            "failed_sum": failed_sum,
            "entropy": float(entropy.item()),
        }
