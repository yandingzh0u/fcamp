"""KL-based adaptive learning-rate controller for clipped Flow-CPS updates."""
from __future__ import annotations

import math

import torch


def adaptive_lr_from_kl(
    raw_kl,
    kl_units: int,
    target_per_step: float,
    lr: float,
    min_lr: float,
    max_lr: float,
    factor: float = 1.5,
) -> tuple[float, float]:
    """Scale ``lr`` based on a per-step KL comparison.

    Compares ``raw_kl / kl_units`` against ``target_per_step``:
      * per-step KL > 2 * target  -> lr /= factor  (capped at min_lr)
      * 0 < per-step KL < 0.5 * target -> lr *= factor (capped at max_lr)
      * otherwise -> unchanged

    Returns ``(new_lr, kl_per_step)``. A non-positive / non-finite ``raw_kl`` or
    ``target_per_step <= 0`` leaves ``lr`` unchanged (per-step KL still reported
    when finite).
    """
    units = max(1, int(kl_units))
    value = float(raw_kl.item() if torch.is_tensor(raw_kl) else raw_kl)
    if not math.isfinite(value):
        return float(lr), 0.0
    kl_per_step = value / units
    if target_per_step <= 0.0:
        return float(lr), kl_per_step
    if value <= 0.0:
        return float(lr), kl_per_step
    if kl_per_step > 2.0 * target_per_step:
        return max(float(min_lr), float(lr) / factor), kl_per_step
    if 0.0 < kl_per_step < 0.5 * target_per_step:
        return min(float(max_lr), float(lr) * factor), kl_per_step
    return float(lr), kl_per_step
