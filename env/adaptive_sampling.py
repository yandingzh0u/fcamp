from __future__ import annotations

import torch

# Bumped whenever the on-disk sampler state layout changes. Old checkpoints that carry an
# incompatible sampler state (version mismatch / absent) are NOT restored -- resuming an older
# checkpoint therefore starts the sampler fresh.
#   v2: per-bin death-frame EMA (~1s bins), additive uniform floor (the official design).
#   v3: per-FRAME failure EMA + arbitrary [f-80,f-20] causal lookback (REMOVED -- the lookback
#       window was decoupled from the rollout credit horizon and never saw the death frame).
#   v4: official per-bin EMA (direct sampler + readiness) PLUS a separate per-FRAME EMA used only
#       by an optional second-stage causal predecessor sampler bound to the rollout credit window.
ADAPTIVE_SAMPLER_VERSION = 4


class AdaptiveTimestepsSampler:
    """Holosoma adaptive motion start sampler (failure-bin base + optional causal second stage).

    Two failure statistics are maintained, both fed only by ``env.step()``:

    * ``bin_failed_count`` -- the OFFICIAL ~1s-bin death-frame EMA. A death at frame ``f`` maps to
      ``bin = clamp(f * num_bins // num_frames)``. The direct sampler draws starts from
      ``p_bin proportional to bin_failed_count + uniform_ratio / num_bins`` (an ADDITIVE uniform
      floor, NOT a fixed mixture weight). This is verbatim the v2 official sampler and is also the
      readiness signal.
    * ``frame_failed_count`` -- a per-FRAME death-frame EMA, used ONLY by the causal sampler. It is
      kept per-frame (not the bin EMA expanded) so a death at ``f`` cannot fabricate failure mass
      at neighbouring frames inside the same bin.

    The start distribution is a two-way blend (NO extra uniform term -- the official floor already
    lives inside ``p_official``)::

        p(s) = (1 - a_cau) * p_official(s) + a_cau * p_causal(s)

    ``a_cau = beta * causal_max_ratio`` where ``beta`` ramps with the *bottleneck concentration*
    (how concentrated deaths are around the dominant failure bin). Early in training deaths are
    spread out (beta == 0 -> pure official failure-bin, stage one); once the policy reliably reaches
    the wall and deaths concentrate there (beta -> 1) the causal predecessor sampler is mixed in,
    while the official direct sampler and its uniform floor are always retained. With
    ``causal_max_ratio == 0`` the sampler is byte-for-byte the official v2 design.

    The causal predecessor score for a START frame ``s`` is bound to the rollout CREDIT window
    ``[f - (H-1), f - 1]`` and GAE/return weighted::

        p_causal(s) proportional to sum_{k=1}^{H-1} frame_failed_count[s + k] * decay^k

    so a start only earns causal mass if it can actually SEE the death within one ``H``-step rollout
    (H = num_steps_per_env, decay = gamma*lambda for GAE algorithms, gamma for the GRPO 48-frame
    rollout). ``H`` and ``decay`` are injected by the algorithm via :meth:`configure_credit` (the
    YAML keeps them at 0 so changing the rollout length never silently misconfigures the sampler).
    A single death at ``f`` therefore produces causal support strictly on ``[f-(H-1), f-1]`` and
    exactly zero on/after ``f``.
    """

    def __init__(
        self,
        motion_time_step_total: int,
        device: torch.device | str,
        *,
        num_bins: int = 0,
        env_fps: int = 50,
        adaptive_kernel_size: int = 1,
        adaptive_lambda: float = 0.8,
        adaptive_uniform_ratio: float = 0.1,
        adaptive_alpha: float = 0.001,
        causal_max_ratio: float = 0.5,
        causal_horizon: int = 0,
        causal_decay: float = 0.0,
        causal_arm_lo: float = 0.5,
        causal_arm_hi: float = 0.8,
    ):
        self.device = device
        self.motion_time_step_total = int(max(1, motion_time_step_total))
        self.num_frames = self.motion_time_step_total
        if int(num_bins) <= 0:
            # ~1 bin per second of motion (Holosoma default); auto-derived so it never needs
            # retuning when the motion clip changes.
            num_bins = self.num_frames // int(max(1, env_fps)) + 1
        self.num_bins = int(max(1, num_bins))
        self.adaptive_kernel_size = int(max(1, adaptive_kernel_size))
        self.adaptive_lambda = float(adaptive_lambda)
        self.adaptive_uniform_ratio = min(1.0, max(0.0, float(adaptive_uniform_ratio)))
        self.adaptive_alpha = min(1.0, max(0.0, float(adaptive_alpha)))
        self.causal_max_ratio = min(1.0, max(0.0, float(causal_max_ratio)))
        self.causal_horizon = int(max(0, causal_horizon))
        self.causal_decay = float(causal_decay)
        self.causal_arm_lo = float(causal_arm_lo)
        self.causal_arm_hi = float(causal_arm_hi)
        kernel = torch.tensor(
            [self.adaptive_lambda**i for i in range(self.adaptive_kernel_size)],
            dtype=torch.float32,
            device=self.device,
        )
        self.kernel = kernel / kernel.sum()
        self.init_buffers()

    def init_buffers(self) -> None:
        # Official per-bin EMA (direct sampler + readiness).
        self.current_bin_failed_count = torch.zeros(self.num_bins, dtype=torch.float32, device=self.device)
        self.bin_failed_count = torch.zeros(self.num_bins, dtype=torch.float32, device=self.device)
        # Per-frame EMA (causal predecessor sampler only).
        self.current_frame_failed_count = torch.zeros(self.num_frames, dtype=torch.float32, device=self.device)
        self.frame_failed_count = torch.zeros(self.num_frames, dtype=torch.float32, device=self.device)

    def configure_credit(self, horizon: int, decay: float) -> None:
        """Bind the causal predecessor window to the algorithm's rollout credit horizon. Called by
        the algorithm at init: PPO/FPO pass ``H = num_steps_per_env, decay = gamma * lambda``;
        MixGRPO passes ``H = 48, decay = gamma`` (48-frame rollout, no GAE)."""
        self.causal_horizon = int(max(0, horizon))
        self.causal_decay = float(decay)

    # --------------------------------------------------------------------- failure recording
    def frames_to_bins(self, failed_at_time_step: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            (failed_at_time_step.reshape(-1).long() * self.num_bins) // self.motion_time_step_total,
            0,
            self.num_bins - 1,
        )

    def update_current_failure_count(self, failed_at_time_step: torch.Tensor) -> None:
        """Accumulate this step's tracking-failure death frames into BOTH per-step accumulators."""
        if failed_at_time_step.numel() == 0:
            return
        idx = torch.clamp(failed_at_time_step.reshape(-1).long(), 0, self.num_frames - 1)
        self.current_frame_failed_count += torch.bincount(idx, minlength=self.num_frames).to(
            self.current_frame_failed_count
        )
        failed_bin = self.frames_to_bins(idx)
        self.current_bin_failed_count += torch.bincount(failed_bin, minlength=self.num_bins).to(
            self.current_bin_failed_count
        )

    def update_failure_ema(self) -> None:
        """Fold both per-step accumulators into their EMAs, then zero them. Called every step."""
        a = self.adaptive_alpha
        self.bin_failed_count = a * self.current_bin_failed_count + (1.0 - a) * self.bin_failed_count
        self.frame_failed_count = a * self.current_frame_failed_count + (1.0 - a) * self.frame_failed_count
        self.current_bin_failed_count.zero_()
        self.current_frame_failed_count.zero_()

    # --------------------------------------------------------------------- official direct sampler
    @property
    def sampling_probabilities(self) -> torch.Tensor:
        # p_b proportional to (failure EMA + ADDITIVE uniform floor), smoothed by the kernel.
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

    def _sample_official(self, num_samples: int, min_phase: int, max_phase: int) -> torch.Tensor:
        """Official failure-bin conditional sampling inside [min_phase, max_phase].

        True conditional sampling (NOT a post-hoc clamp): each bin is intersected with the range,
        empty-intersection bins are masked, the remaining bin probabilities are renormalized by the
        per-frame density, and offsets are drawn only within each bin's intersection."""
        if num_samples <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        all_bins = torch.arange(self.num_bins, device=self.device)
        lo, hi = self.bin_frame_bounds(all_bins)  # global [lo, hi) per bin
        full_span = (hi - lo).clamp_min(1).to(torch.float32)
        lo_eff = torch.clamp(lo, min=min_phase)
        hi_eff = torch.clamp(hi, max=max_phase + 1)
        span_eff = (hi_eff - lo_eff).clamp_min(0)  # 0 => no overlap with the range

        # Bin weight = (global frame density P_b / full_span_b) * in-range frame count, so the
        # per-FRAME density is the exact conditional restriction of the global distribution.
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

    # --------------------------------------------------------------------- causal predecessor sampler
    def _causal_frame_scores(self) -> torch.Tensor:
        """Per START frame ``s``: ``sum_{k=1}^{H-1} frame_failed_count[s+k] * decay^k`` -- failure
        mass reachable within the rollout credit window ``[s+1, s+H-1]``, GAE/return discounted.
        Strictly causal: only k >= 1, so a death at ``f`` never gives mass to ``s >= f``."""
        n = self.num_frames
        scores = torch.zeros(n, dtype=torch.float32, device=self.device)
        H = self.causal_horizon
        d = self.causal_decay
        if H <= 1 or d <= 0.0:
            return scores
        G = self.frame_failed_count
        for k in range(1, H):
            w = d**k
            if w <= 0.0:
                break
            scores[: n - k] += G[k:] * w
        return scores

    def _sample_causal(self, num_samples: int, min_phase: int, max_phase: int) -> torch.Tensor:
        """Sample predecessor starts from ``_causal_frame_scores`` restricted to [min_phase,
        max_phase]. Returns an EMPTY tensor when there is no in-range causal mass (the caller then
        falls back to the official direct sampler for those slots)."""
        if num_samples <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        scores = self._causal_frame_scores()
        valid = torch.zeros(self.num_frames, dtype=torch.float32, device=self.device)
        valid[min_phase : max_phase + 1] = 1.0
        probs = scores * valid
        total = probs.sum()
        if not bool(torch.isfinite(total)) or float(total.item()) <= 0.0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        probs = probs / total
        return torch.multinomial(probs, num_samples, replacement=True).long()

    # --------------------------------------------------------------------- readiness / blend
    def bottleneck_concentration(self) -> float:
        """Readiness signal: share of the bin failure EMA within +-1 bin of the dominant failure
        bin. Low while deaths are spread (policy not reaching the wall) -> high once deaths
        concentrate at the bottleneck. Self-locating; never hard-codes the wall phase."""
        ema = self.bin_failed_count
        total = float(ema.sum().item())
        if total <= 0.0:
            return 0.0
        peak = int(torch.argmax(ema).item())
        lo = max(0, peak - 1)
        hi = min(self.num_bins - 1, peak + 1)
        return float(ema[lo : hi + 1].sum().item() / total)

    def causal_weight(self) -> float:
        """``a_cau`` = how much of the blend is the causal predecessor sampler this update."""
        if self.causal_max_ratio <= 0.0 or self.causal_horizon <= 1 or self.causal_decay <= 0.0:
            return 0.0
        r = self.bottleneck_concentration()
        if self.causal_arm_hi <= self.causal_arm_lo:
            beta = 1.0 if r >= self.causal_arm_hi else 0.0
        else:
            beta = (r - self.causal_arm_lo) / (self.causal_arm_hi - self.causal_arm_lo)
            beta = min(1.0, max(0.0, beta))
        return beta * self.causal_max_ratio

    def sample_frames(self, num_samples: int, min_phase: int, max_phase: int) -> torch.Tensor:
        """Draw ``num_samples`` start frames from ``(1 - a_cau) * p_official + a_cau * p_causal``,
        conditioned on the valid range [min_phase, max_phase]."""
        if num_samples <= 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        min_phase = max(0, int(min_phase))
        max_phase = min(self.num_frames - 1, int(max_phase))
        if max_phase < min_phase:
            return torch.full((num_samples,), min_phase, dtype=torch.long, device=self.device)

        a_cau = self.causal_weight()
        n_cau = int(round(num_samples * a_cau))
        causal = self._sample_causal(n_cau, min_phase, max_phase) if n_cau > 0 else None
        n_causal = 0 if causal is None else int(causal.numel())
        # Any causal slot with no in-range mass falls back to the official direct sampler.
        n_dir = num_samples - n_causal
        direct = self._sample_official(n_dir, min_phase, max_phase)
        if n_causal == 0:
            return direct
        out = torch.cat([direct, causal])
        return out[torch.randperm(out.numel(), device=self.device)]

    # --------------------------------------------------------------------- persistence
    def state_dict(self) -> dict:
        return {
            "version": ADAPTIVE_SAMPLER_VERSION,
            "num_bins": self.num_bins,
            "num_frames": self.num_frames,
            "bin_failed_count": self.bin_failed_count.detach().cpu(),
            "current_bin_failed_count": self.current_bin_failed_count.detach().cpu(),
            "frame_failed_count": self.frame_failed_count.detach().cpu(),
            "current_frame_failed_count": self.current_frame_failed_count.detach().cpu(),
        }

    def load_state_dict(self, state: dict | None) -> bool:
        """Restore both EMAs. Returns False (keeping fresh zeros) on version / shape mismatch --
        incompatible (e.g. pre-v4) stats are intentionally NOT restored."""
        if not state:
            return False
        if int(state.get("version", -1)) != ADAPTIVE_SAMPLER_VERSION:
            return False
        bfc = state.get("bin_failed_count")
        ffc = state.get("frame_failed_count")
        if bfc is None or tuple(bfc.shape) != (self.num_bins,):
            return False
        if ffc is None or tuple(ffc.shape) != (self.num_frames,):
            return False
        self.bin_failed_count.copy_(bfc.to(self.bin_failed_count))
        self.frame_failed_count.copy_(ffc.to(self.frame_failed_count))
        cbfc = state.get("current_bin_failed_count")
        if cbfc is not None and tuple(cbfc.shape) == (self.num_bins,):
            self.current_bin_failed_count.copy_(cbfc.to(self.current_bin_failed_count))
        cffc = state.get("current_frame_failed_count")
        if cffc is not None and tuple(cffc.shape) == (self.num_frames,):
            self.current_frame_failed_count.copy_(cffc.to(self.current_frame_failed_count))
        return True

    # --------------------------------------------------------------------- diagnostics
    def stats(self, min_phase: int = 0, max_phase: int | None = None) -> dict[str, float]:
        """Diagnostics for the [SAMPLER] log line: official bin distribution + causal blend state."""
        probabilities = self.sampling_probabilities
        top_prob, top_bin = probabilities.max(dim=0)
        entropy = -(probabilities * probabilities.clamp_min(1.0e-12).log()).sum()
        if self.num_bins > 1:
            entropy = entropy / torch.log(torch.tensor(float(self.num_bins), device=self.device))
        else:
            entropy = torch.ones_like(entropy)
        failed_sum = float(self.bin_failed_count.sum().item())
        peak_bin = int(torch.argmax(self.bin_failed_count).item()) if failed_sum > 0.0 else -1
        frame_sum = float(self.frame_failed_count.sum().item())
        peak_fail_frame = int(torch.argmax(self.frame_failed_count).item()) if frame_sum > 0.0 else -1
        return {
            "top_bin": float(top_bin.item()),
            "top_prob": float(top_prob.item()),
            "failed_sum": failed_sum,
            "entropy": float(entropy.item()),
            "peak_bin": float(peak_bin),
            "peak_fail_frame": float(peak_fail_frame),
            "bottleneck_concentration": self.bottleneck_concentration(),
            "causal_beta": self.causal_weight(),
        }
