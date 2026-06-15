from __future__ import annotations

import torch


def allocate_sampling_counts(
    num_samples: int,
    *,
    start_phase_ratio: float,
) -> tuple[int, int]:
    """Split a batch into (phase-0 count, stratified-uniform count).

    A fixed fraction of every batch is pinned to the clip start (phase 0) so the motion
    opening always gets coverage; the remainder is stratified-uniform over the full valid
    phase range. There is no adaptive / failure-weighted quota.
    """
    if num_samples < 0:
        raise ValueError(f"num_samples must be >= 0, got {num_samples}")
    start_phase_ratio = min(1.0, max(0.0, float(start_phase_ratio)))
    start_count = min(num_samples, int(round(num_samples * start_phase_ratio)))
    uniform_count = num_samples - start_count
    return start_count, uniform_count


def stratified_uniform_offsets(
    num_candidates: int,
    num_samples: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Sample one random offset per equal-width stratum across the full range.

    With ~2048 samples over a ~959-frame clip this gives near-complete coverage of the whole
    motion every update, avoiding the gaps a plain torch.randint draw would leave.
    """
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
