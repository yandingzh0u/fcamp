"""External temporal-motion distribution metric.

The metric deliberately consumes only the common raw imitation-frame schema.
It never reads a method's discriminator, reward, observation, or critic.  A
policy window is compared with the demonstration frames at the exact reference
phases traversed by that environment, so variable reference-time policies and
fixed-rate policies share the same evaluator.

MMD is estimated with the standard linear-time paired estimator and a fixed
mixture of RBF kernels. Pair members from one trajectory never share primitive
frames; under the usual iid sample assumption this is the unbiased linear-time
MMD estimator. Distances are mean squared distances in demo-standardized
feature space, which keeps the bandwidths independent of the window
dimensionality. The public non-negative MMD^2 clamps the finite-sample signed
estimate at zero; the signed estimate is retained for diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from components.imitation.motion_features import canonicalize_imitation_window


def sanitize_reference_phases(
    phases: torch.Tensor,
    *,
    num_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return safe demo-query phases and a strict validity mask.

    Finished, wrapped, or malformed trajectories must not make the shared
    evaluator throw while querying a reference frame. Invalid values are
    clamped only for the query; callers use the returned mask to exclude them
    from temporal windows.
    """

    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    values = phases.to(dtype=torch.float32)
    maximum = float(num_frames - 1)
    valid = torch.isfinite(values) & (values >= 0.0) & (values <= maximum)
    safe = torch.nan_to_num(values, nan=0.0, posinf=maximum, neginf=0.0).clamp(
        min=0.0,
        max=maximum,
    )
    return safe, valid


@dataclass(frozen=True)
class DemoFeatureNormalizer:
    """Fixed normalization fitted exclusively to demonstration frames."""

    mean: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(
        cls,
        demo_frames: torch.Tensor,
        *,
        minimum_scale: float = 1.0e-3,
    ) -> "DemoFeatureNormalizer":
        if demo_frames.ndim != 2 or demo_frames.shape[0] < 1:
            raise ValueError(
                "demo_frames must have shape [num_demo_frames, frame_dim], "
                f"got {tuple(demo_frames.shape)}"
            )
        if minimum_scale <= 0.0:
            raise ValueError("minimum_scale must be positive")
        frames = demo_frames.detach().to(dtype=torch.float32)
        if not bool(torch.isfinite(frames).all()):
            raise ValueError("demo_frames contain non-finite values")
        return cls(
            mean=frames.mean(dim=0),
            scale=frames.std(dim=0, unbiased=False).clamp_min(float(minimum_scale)),
        )

    @property
    def frame_dim(self) -> int:
        return int(self.mean.numel())

    def normalize(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.shape[-1] != self.frame_dim:
            raise ValueError(
                f"expected frame dimension {self.frame_dim}, got {frames.shape[-1]}"
            )
        mean = self.mean.to(device=frames.device, dtype=frames.dtype)
        scale = self.scale.to(device=frames.device, dtype=frames.dtype)
        return (frames - mean) / scale


class _LinearTimeMultiKernelMMD:
    """Streaming linear-time unbiased MMD estimator over equal-size samples."""

    def __init__(self, bandwidths: tuple[float, ...]) -> None:
        if not bandwidths or any(value <= 0.0 for value in bandwidths):
            raise ValueError("bandwidths must contain positive values")
        self.bandwidths = tuple(float(value) for value in bandwidths)
        self._sum = 0.0
        self._sum_sq = 0.0
        self._pairs = 0

    def _kernel(self, lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        mean_sq = (lhs - rhs).square().mean(dim=-1)
        kernels = [
            torch.exp(-mean_sq / (2.0 * bandwidth * bandwidth))
            for bandwidth in self.bandwidths
        ]
        return torch.stack(kernels, dim=0).mean(dim=0)

    def update_pairs(
        self,
        policy_0: torch.Tensor,
        policy_1: torch.Tensor,
        demo_0: torch.Tensor,
        demo_1: torch.Tensor,
    ) -> None:
        if policy_0.ndim != 2 or not (
            policy_1.shape == demo_0.shape == demo_1.shape == policy_0.shape
        ):
            raise ValueError(
                "all paired samples must share shape [pairs, features], got "
                f"{tuple(policy_0.shape)}, {tuple(policy_1.shape)}, "
                f"{tuple(demo_0.shape)}, {tuple(demo_1.shape)}"
            )
        if policy_0.shape[0] == 0:
            return
        policy_0 = policy_0.detach().to(dtype=torch.float32)
        policy_1 = policy_1.detach().to(device=policy_0.device, dtype=torch.float32)
        demo_0 = demo_0.detach().to(device=policy_0.device, dtype=torch.float32)
        demo_1 = demo_1.detach().to(device=policy_0.device, dtype=torch.float32)
        estimate = (
            self._kernel(policy_0, policy_1)
            + self._kernel(demo_0, demo_1)
            - self._kernel(policy_0, demo_1)
            - self._kernel(policy_1, demo_0)
        )
        self._sum += float(estimate.sum().item())
        self._sum_sq += float(estimate.square().sum().item())
        self._pairs += int(estimate.numel())

    @property
    def pairs(self) -> int:
        return self._pairs

    def signed_estimate(self) -> float | None:
        if self._pairs == 0:
            return None
        return self._sum / self._pairs

    def standard_error(self) -> float | None:
        if self._pairs < 2:
            return None
        mean = self._sum / self._pairs
        variance = max(0.0, (self._sum_sq - self._pairs * mean * mean) / (self._pairs - 1))
        return (variance / self._pairs) ** 0.5


class PhaseMatchedWindowMMD:
    """Track phase-matched MMD for several causal alive-only window sizes.

    To bound validation overhead independently of the training environment
    count, a deterministic evenly-spaced subset of environments is selected.
    A ring entry is legal only after the same environment has supplied W
    consecutive post-transition frames marked alive by the caller.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        normalizer: DemoFeatureNormalizer,
        device: torch.device | str,
        window_sizes: tuple[int, ...] = (16, 32),
        bandwidths: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0),
        max_envs: int = 128,
        reference_phase_start: float = 0.0,
        reference_phase_end: float | None = None,
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive")
        if max_envs <= 0:
            raise ValueError("max_envs must be positive")
        if not window_sizes or any(size <= 0 for size in window_sizes):
            raise ValueError("window_sizes must contain positive values")
        if len(set(window_sizes)) != len(window_sizes):
            raise ValueError("window_sizes must be unique")
        self.window_sizes = tuple(sorted(int(size) for size in window_sizes))
        self.max_window = max(self.window_sizes)
        self.ring_size = 2 * self.max_window
        self.normalizer = normalizer
        self.device = torch.device(device)
        self.reference_phase_start = float(reference_phase_start)
        self.reference_phase_end = (
            None if reference_phase_end is None else float(reference_phase_end)
        )
        if (
            self.reference_phase_end is not None
            and self.reference_phase_end <= self.reference_phase_start
        ):
            raise ValueError("reference_phase_end must exceed reference_phase_start")
        selected_count = min(int(num_envs), int(max_envs))
        # Integer arithmetic gives a deterministic, unique, evenly-spaced set
        # including env zero without depending on any RNG state.
        self.env_ids = torch.div(
            torch.arange(selected_count, device=self.device, dtype=torch.long) * int(num_envs),
            selected_count,
            rounding_mode="floor",
        )
        # Two maximum-length windows are sufficient to reconstruct the current
        # window and the previous same-lane window without storing W flattened
        # pending windows per environment.
        shape = (selected_count, self.ring_size, normalizer.frame_dim)
        self._policy_ring = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self._demo_ring = torch.zeros_like(self._policy_ring)
        self._consecutive = torch.zeros(selected_count, dtype=torch.long, device=self.device)
        self._cursor = 0
        self._estimators = {
            size: _LinearTimeMultiKernelMMD(bandwidths) for size in self.window_sizes
        }
        self._legal_windows = {size: 0 for size in self.window_sizes}
        self._sampled_windows = {size: 0 for size in self.window_sizes}
        self._pending_valid = {
            size: torch.zeros(
                selected_count,
                size,
                dtype=torch.bool,
                device=self.device,
            )
            for size in self.window_sizes
        }
        self._pending_endpoint = {
            size: torch.full(
                (selected_count, size),
                -1,
                dtype=torch.long,
                device=self.device,
            )
            for size in self.window_sizes
        }
        self._pair_gap_min = {size: None for size in self.window_sizes}
        self._pair_gap_max = {size: None for size in self.window_sizes}
        self._phase_count = {size: 0 for size in self.window_sizes}
        self._phase_sum = {size: 0.0 for size in self.window_sizes}
        self._phase_min = {size: None for size in self.window_sizes}
        self._phase_max = {size: None for size in self.window_sizes}
        self._endpoint_step = 0

    @property
    def selected_env_count(self) -> int:
        return int(self.env_ids.numel())

    def update_selected(
        self,
        policy_frames: torch.Tensor,
        demo_frames: torch.Tensor,
        alive: torch.Tensor,
        endpoint_phases: torch.Tensor,
    ) -> None:
        expected = (self.selected_env_count, self.normalizer.frame_dim)
        if tuple(policy_frames.shape) != expected or tuple(demo_frames.shape) != expected:
            raise ValueError(
                f"selected frames must have shape {expected}, got "
                f"{tuple(policy_frames.shape)} and {tuple(demo_frames.shape)}"
            )
        if tuple(alive.shape) != (self.selected_env_count,):
            raise ValueError(
                f"alive must have shape {(self.selected_env_count,)}, got {tuple(alive.shape)}"
            )
        if tuple(endpoint_phases.shape) != (self.selected_env_count,):
            raise ValueError(
                "endpoint_phases must have shape "
                f"{(self.selected_env_count,)}, got {tuple(endpoint_phases.shape)}"
            )
        policy = self.normalizer.normalize(
            policy_frames.to(device=self.device, dtype=torch.float32)
        )
        demo = self.normalizer.normalize(
            demo_frames.to(device=self.device, dtype=torch.float32)
        )
        valid = (
            alive.to(device=self.device, dtype=torch.bool)
            & torch.isfinite(policy).all(dim=-1)
            & torch.isfinite(demo).all(dim=-1)
            & torch.isfinite(endpoint_phases.to(device=self.device))
        )
        self._policy_ring[:, self._cursor] = torch.where(
            valid[:, None], policy, torch.zeros_like(policy)
        )
        self._demo_ring[:, self._cursor] = torch.where(
            valid[:, None], demo, torch.zeros_like(demo)
        )
        self._consecutive = torch.where(valid, self._consecutive + 1, torch.zeros_like(self._consecutive))

        for size in self.window_sizes:
            self._pending_valid[size][~valid, :] = False
            self._pending_endpoint[size][~valid, :] = -1
            ready = self._consecutive >= size
            if not bool(ready.any()):
                continue
            self._legal_windows[size] += int(ready.sum().item())
            # Every legal overlapping W-frame window enters the stream once.
            # endpoint mod W defines independent pending lanes. Consecutive
            # samples in one lane are W primitive steps apart, so pair members
            # contain no shared underlying frames. Pairing consumes the pending
            # sample; the next same-lane window starts a fresh pair.
            ready_ids = ready.nonzero(as_tuple=False).squeeze(-1)
            ready_phases = endpoint_phases.to(
                device=self.device, dtype=torch.float32
            ).index_select(0, ready_ids)
            phase_min = float(ready_phases.min().item())
            phase_max = float(ready_phases.max().item())
            self._phase_count[size] += int(ready_phases.numel())
            self._phase_sum[size] += float(ready_phases.sum().item())
            current_phase_min = self._phase_min[size]
            current_phase_max = self._phase_max[size]
            self._phase_min[size] = (
                phase_min if current_phase_min is None else min(current_phase_min, phase_min)
            )
            self._phase_max[size] = (
                phase_max if current_phase_max is None else max(current_phase_max, phase_max)
            )
            lane = self._endpoint_step % size
            current_indices = (
                torch.arange(size, device=self.device) + self._cursor - size + 1
            ) % self.ring_size
            policy_window = self._policy_ring.index_select(0, ready_ids).index_select(
                1, current_indices
            )
            demo_window = self._demo_ring.index_select(0, ready_ids).index_select(
                1, current_indices
            )
            policy_flat = canonicalize_imitation_window(policy_window).flatten(start_dim=1)
            demo_flat = canonicalize_imitation_window(demo_window).flatten(start_dim=1)
            had_pending = self._pending_valid[size][ready_ids, lane]
            if bool(had_pending.any()):
                pair_ids = ready_ids[had_pending]
                previous_indices = (
                    torch.arange(size, device=self.device) + self._cursor - 2 * size + 1
                ) % self.ring_size
                previous_policy = canonicalize_imitation_window(
                    self._policy_ring.index_select(0, pair_ids).index_select(
                        1, previous_indices
                    )
                ).flatten(start_dim=1)
                previous_demo = canonicalize_imitation_window(
                    self._demo_ring.index_select(0, pair_ids).index_select(
                        1, previous_indices
                    )
                ).flatten(start_dim=1)
                gaps = self._endpoint_step - self._pending_endpoint[size][pair_ids, lane]
                if not bool((gaps == size).all()):
                    raise RuntimeError(
                        f"W={size} MMD pair endpoints must be exactly {size} steps apart; "
                        f"got min={int(gaps.min().item())} max={int(gaps.max().item())}"
                    )
                self._estimators[size].update_pairs(
                    previous_policy,
                    policy_flat[had_pending],
                    previous_demo,
                    demo_flat[had_pending],
                )
                gap_min = int(gaps.min().item())
                gap_max = int(gaps.max().item())
                current_min = self._pair_gap_min[size]
                current_max = self._pair_gap_max[size]
                self._pair_gap_min[size] = gap_min if current_min is None else min(current_min, gap_min)
                self._pair_gap_max[size] = gap_max if current_max is None else max(current_max, gap_max)
                self._pending_valid[size][pair_ids, lane] = False
                self._pending_endpoint[size][pair_ids, lane] = -1
            needs_pending = ~had_pending
            if bool(needs_pending.any()):
                pending_ids = ready_ids[needs_pending]
                self._pending_valid[size][pending_ids, lane] = True
                self._pending_endpoint[size][pending_ids, lane] = self._endpoint_step
            self._sampled_windows[size] += int(ready_ids.numel())
        self._cursor = (self._cursor + 1) % self.ring_size
        self._endpoint_step += 1

    def metrics(self, prefix: str = "validation") -> dict[str, float]:
        metrics: dict[str, float] = {
            f"{prefix}/window_mmd_selected_envs": float(self.selected_env_count),
        }
        for size in self.window_sizes:
            estimator = self._estimators[size]
            metrics[f"{prefix}/window_mmd_windows_w{size}"] = float(self._legal_windows[size])
            metrics[f"{prefix}/window_mmd_samples_w{size}"] = float(self._sampled_windows[size])
            metrics[f"{prefix}/window_mmd_pairs_w{size}"] = float(estimator.pairs)
            phase_count = self._phase_count[size]
            if phase_count > 0:
                phase_min = float(self._phase_min[size])
                phase_max = float(self._phase_max[size])
                phase_mean = self._phase_sum[size] / phase_count
                metrics[f"{prefix}/window_phase_endpoint_count_w{size}"] = float(
                    phase_count
                )
                metrics[f"{prefix}/window_phase_endpoint_min_w{size}"] = phase_min
                metrics[f"{prefix}/window_phase_endpoint_mean_w{size}"] = phase_mean
                metrics[f"{prefix}/window_phase_endpoint_max_w{size}"] = phase_max
                metrics[f"{prefix}/window_phase_endpoint_span_w{size}"] = phase_max - phase_min
                if self.reference_phase_end is not None:
                    span = self.reference_phase_end - self.reference_phase_start
                    progress_min = max(0.0, min(1.0, (phase_min - self.reference_phase_start) / span))
                    progress_mean = max(0.0, min(1.0, (phase_mean - self.reference_phase_start) / span))
                    progress_max = max(0.0, min(1.0, (phase_max - self.reference_phase_start) / span))
                    metrics[f"{prefix}/window_reference_progress_min_w{size}"] = progress_min
                    metrics[f"{prefix}/window_reference_progress_mean_w{size}"] = progress_mean
                    metrics[f"{prefix}/window_reference_progress_max_w{size}"] = progress_max
                    metrics[f"{prefix}/window_reference_progress_span_w{size}"] = (
                        progress_max - progress_min
                    )
            if self._pair_gap_min[size] is not None:
                metrics[f"{prefix}/window_mmd_pair_gap_min_w{size}"] = float(
                    self._pair_gap_min[size]
                )
                metrics[f"{prefix}/window_mmd_pair_gap_max_w{size}"] = float(
                    self._pair_gap_max[size]
                )
            signed = estimator.signed_estimate()
            if signed is not None:
                metrics[f"{prefix}/window_mmd2_raw_w{size}"] = float(signed)
                metrics[f"{prefix}/window_mmd2_w{size}"] = float(max(0.0, signed))
            stderr = estimator.standard_error()
            if stderr is not None:
                metrics[f"{prefix}/window_mmd2_stderr_w{size}"] = float(stderr)
        return metrics
