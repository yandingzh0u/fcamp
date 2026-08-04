from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
from scipy.stats import spearmanr


def _seed_score_matrix(seed_scores: Sequence[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(scores, dtype=np.float64).reshape(-1) for scores in seed_scores]
    if len(arrays) < 2:
        raise ValueError("at least two held-out reward seeds are required")
    if len({array.shape for array in arrays}) != 1 or arrays[0].size < 2:
        raise ValueError("reward seeds must share a non-trivial aligned state bank")
    matrix = np.stack(arrays, axis=1)
    if not np.isfinite(matrix).all():
        raise FloatingPointError("reward scores contain NaN or Inf")
    return matrix


def icc_two_way_consistency(seed_scores: Sequence[np.ndarray]) -> float:
    """Compute ICC(C,1) over aligned samples × discriminator seeds."""

    values = _seed_score_matrix(seed_scores)
    sample_count, seed_count = values.shape
    grand = float(values.mean())
    row_means = values.mean(axis=1)
    column_means = values.mean(axis=0)
    ss_rows = seed_count * float(np.square(row_means - grand).sum())
    ss_columns = sample_count * float(np.square(column_means - grand).sum())
    residual = values - row_means[:, None] - column_means[None, :] + grand
    ss_error = float(np.square(residual).sum())
    ms_rows = ss_rows / float(sample_count - 1)
    ms_error = ss_error / float((sample_count - 1) * (seed_count - 1))
    denominator = ms_rows + (seed_count - 1) * ms_error
    if denominator <= 0.0:
        return 1.0 if ms_error == 0.0 else float("nan")
    return float((ms_rows - ms_error) / denominator)


def reward_seed_agreement(seed_scores: Sequence[np.ndarray]) -> dict[str, float]:
    values = _seed_score_matrix(seed_scores)
    correlations: list[float] = []
    sign_disagreements: list[float] = []
    centered = values - np.median(values, axis=0, keepdims=True)
    for left, right in combinations(range(values.shape[1]), 2):
        correlations.append(float(spearmanr(values[:, left], values[:, right]).statistic))
        sign_disagreements.append(float(np.mean(np.sign(centered[:, left]) != np.sign(centered[:, right]))))
    return {
        "icc_consistency": icc_two_way_consistency(
            [values[:, index] for index in range(values.shape[1])]
        ),
        "pairwise_spearman_mean": float(np.mean(correlations)),
        "pairwise_spearman_min": float(np.min(correlations)),
        "sign_disagreement_mean": float(np.mean(sign_disagreements)),
    }


def strict_pairwise_accuracy(
    scores: np.ndarray,
    winner_loser_pairs: Sequence[tuple[int, int]],
) -> dict[str, float]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(scores).all():
        raise FloatingPointError("scores contain NaN or Inf")
    if not winner_loser_pairs:
        return {"pair_count": 0.0, "accuracy": float("nan"), "tie_fraction": float("nan")}
    correct = 0
    ties = 0
    for winner, loser in winner_loser_pairs:
        if not 0 <= int(winner) < scores.size or not 0 <= int(loser) < scores.size:
            raise IndexError("preference pair is outside the score bank")
        delta = float(scores[int(winner)] - scores[int(loser)])
        correct += int(delta > 0.0)
        ties += int(delta == 0.0)
    count = len(winner_loser_pairs)
    return {
        "pair_count": float(count),
        "accuracy": float(correct / count),
        "tie_fraction": float(ties / count),
    }


def failed_top_decile_fraction(scores: np.ndarray, failed: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    failed = np.asarray(failed, dtype=bool).reshape(-1)
    if scores.shape != failed.shape or scores.size < 10:
        raise ValueError("scores/failed must align and contain at least ten samples")
    cutoff = np.quantile(scores, 0.9)
    selected = scores >= cutoff
    return float(failed[selected].mean())
