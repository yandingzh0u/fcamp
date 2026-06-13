from __future__ import annotations

import torch


def compute_failure_rates(
    failure_counts: torch.Tensor,
    exposure_counts: torch.Tensor,
) -> torch.Tensor:
    if failure_counts.shape != exposure_counts.shape or failure_counts.ndim != 1:
        raise ValueError("failure_counts and exposure_counts must be matching 1D tensors")

    failures = torch.nan_to_num(failure_counts, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    exposures = torch.nan_to_num(exposure_counts, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    total_exposure = exposures.sum()
    if float(total_exposure.item()) <= 0.0:
        return torch.ones_like(exposures)

    global_rate = failures.sum() / total_exposure
    return torch.where(exposures > 1.0e-8, failures / exposures.clamp_min(1.0e-8), global_rate)


def build_adaptive_phase_probabilities(
    failure_rates: torch.Tensor,
    candidate_phases: torch.Tensor,
    *,
    motion_num_frames: int,
) -> torch.Tensor:
    """Map per-start-bin failure rates to a normalized phase distribution."""
    if failure_rates.ndim != 1 or failure_rates.numel() == 0:
        raise ValueError("failure_rates must be a non-empty 1D tensor")
    if candidate_phases.ndim != 1 or candidate_phases.numel() == 0:
        raise ValueError("candidate_phases must be a non-empty 1D tensor")
    if motion_num_frames <= 0:
        raise ValueError(f"motion_num_frames must be positive, got {motion_num_frames}")

    rates = torch.nan_to_num(failure_rates, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    candidate_bins = torch.clamp(
        (candidate_phases * rates.numel()) // motion_num_frames,
        0,
        rates.numel() - 1,
    )
    phase_scores = rates.to(device=candidate_phases.device).index_select(0, candidate_bins)
    score_sum = phase_scores.sum()
    if not bool(torch.isfinite(score_sum)) or float(score_sum.item()) <= 0.0:
        return torch.full_like(phase_scores, 1.0 / float(phase_scores.numel()))
    return phase_scores / score_sum


def allocate_sampling_counts(
    num_samples: int,
    *,
    uniform_ratio: float,
    start_phase_ratio: float,
) -> tuple[int, int, int]:
    """Allocate exact per-batch start, uniform, and adaptive quotas."""
    if num_samples < 0:
        raise ValueError(f"num_samples must be >= 0, got {num_samples}")
    start_phase_ratio = min(1.0, max(0.0, float(start_phase_ratio)))
    uniform_ratio = min(1.0 - start_phase_ratio, max(0.0, float(uniform_ratio)))

    start_count = min(num_samples, int(round(num_samples * start_phase_ratio)))
    uniform_count = min(num_samples - start_count, int(round(num_samples * uniform_ratio)))
    adaptive_count = num_samples - start_count - uniform_count
    return start_count, uniform_count, adaptive_count


def stratified_uniform_offsets(
    num_candidates: int,
    num_samples: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample one random offset per equal-width stratum across the full range."""
    if num_candidates <= 0:
        raise ValueError(f"num_candidates must be positive, got {num_candidates}")
    if num_samples < 0:
        raise ValueError(f"num_samples must be >= 0, got {num_samples}")
    if num_samples == 0:
        return torch.empty(0, dtype=torch.long, device=device)

    positions = (
        (torch.arange(num_samples, dtype=torch.float32, device=device) + torch.rand(num_samples, device=device))
        * (float(num_candidates) / float(num_samples))
    )
    offsets = positions.long().clamp(max=num_candidates - 1)
    return offsets.index_select(0, torch.randperm(num_samples, device=device))
