"""Shared KL-based adaptive learning-rate controller.

The contract across PPO / SFPO / MixGRPO:

* ``desired_kl`` in the config is a **per-env-control-step KL budget**, NOT a
  per-sample / per-chunk budget. This makes ``desired_kl: 0.01`` mean the same
  thing regardless of the action-chunk horizon ``h``.

* The raw KL observed during the update is algorithm-specific:

  - **PPO / MixGRPO** use a *chunk*-level (joint) KL: for a horizon-``h``
    policy the joint log-prob ratio is roughly ``h`` times a single-step
    ratio, so the raw KL scales with ``h``. To compare apples to apples we
    normalize by ``kl_units`` (= ``h``) before comparing against
    ``desired_kl``.

  - **SFPO** uses per-frame / per-prefix ratios and the adaptive-LR KL is the
    masked MEAN per-frame KL (a per-control-step quantity), so ``kl_units=1``
    and ``desired_kl`` is compared directly.

  Examples (``desired_kl = 0.01``)::

      PPO        h=1  -> kl_units=1  -> raw_target = 0.01 * 1   = 0.01
      MixGRPO    h=4  -> kl_units=4  -> raw_target = 0.01 * 4   = 0.04
      SFPO       h=4  -> kl_units=1  -> raw_target = 0.01 * 1   = 0.01

Only ``horizon`` is used as the normalizer for the chunk-level algorithms.
``flow_steps`` is NOT used because the per-SDE-step log-prob is already
averaged inside ``kl_loss``. Group counts are NOT used because they are a
sampling/sorting structure, not a control-step length.

FPO is intentionally NOT routed through here: its ``kl`` is a prediction MSE,
not a log-prob KL, so it cannot share the ``desired_kl`` semantics.
"""
from __future__ import annotations

import math

import torch


def normalized_kl(raw_kl, kl_units: int) -> float:
    """Convert a raw (chunk-level) KL into a per-control-step KL."""
    units = max(1, int(kl_units))
    value = float(raw_kl.item() if torch.is_tensor(raw_kl) else raw_kl)
    return value / units


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
