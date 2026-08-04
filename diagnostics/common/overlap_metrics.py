from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .noise_bank import is_overlap_eligible, parse_collector_mode

def require_primary_overlap_mode(mode: str) -> None:
    parsed = parse_collector_mode(mode)
    if not is_overlap_eligible(parsed):
        raise ValueError(
            "native_stochastic is fingerprint-only and is excluded from overlap"
        )


def posterior_overlap_fraction(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    tau: float,
) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if probabilities.shape != labels.shape or probabilities.size == 0:
        raise ValueError("probabilities and labels must be aligned and non-empty")
    if not 0.0 < float(tau) < 0.5:
        raise ValueError("tau must lie in (0,0.5)")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("labels must contain both classes")
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("probabilities contain NaN or Inf")
    central = (probabilities > tau) & (probabilities < 1.0 - tau)
    class_zero = float(central[labels == 0].mean())
    class_one = float(central[labels == 1].mean())
    return {
        "tau": float(tau),
        "class_0": class_zero,
        "class_1": class_one,
        "balanced": 0.5 * (class_zero + class_one),
    }


def _finite_matrix(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape [N,D]")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")
    return array


def bidirectional_knn_diagnostics(
    source: np.ndarray,
    target: np.ndarray,
    *,
    k: int = 5,
    radius_quantile: float = 0.95,
) -> dict[str, float]:
    """Compute explicitly directional empirical coverage and label mixing.

    A within-domain kNN distance quantile defines each domain's empirical
    radius.  The result is intentionally named coverage, never mathematical
    support.
    """

    source = _finite_matrix("source", source)
    target = _finite_matrix("target", target)
    if source.shape[1] != target.shape[1]:
        raise ValueError("source/target feature dimensions differ")
    if not 0.0 < float(radius_quantile) < 1.0:
        raise ValueError("radius_quantile must lie in (0,1)")
    k_source = min(max(1, int(k)), max(1, source.shape[0] - 1))
    k_target = min(max(1, int(k)), max(1, target.shape[0] - 1))

    def within_radius(data: np.ndarray, count: int) -> float:
        if data.shape[0] == 1:
            return 0.0
        neighbors = NearestNeighbors(n_neighbors=count + 1).fit(data)
        distances = neighbors.kneighbors(data, return_distance=True)[0][:, -1]
        return float(np.quantile(distances, radius_quantile))

    source_radius = within_radius(source, k_source)
    target_radius = within_radius(target, k_target)
    target_index = NearestNeighbors(n_neighbors=1).fit(target)
    source_to_target = target_index.kneighbors(source, return_distance=True)[0][:, 0]
    source_index = NearestNeighbors(n_neighbors=1).fit(source)
    target_to_source = source_index.kneighbors(target, return_distance=True)[0][:, 0]

    combined = np.concatenate((source, target), axis=0)
    labels = np.concatenate(
        (np.zeros(source.shape[0], dtype=np.int64), np.ones(target.shape[0], dtype=np.int64))
    )
    mix_k = min(max(1, int(k)), combined.shape[0] - 1)
    neighborhood = NearestNeighbors(n_neighbors=mix_k + 1).fit(combined)
    indices = neighborhood.kneighbors(combined, return_distance=False)[:, 1:]
    mixing = np.mean(labels[indices] != labels[:, None], axis=1)
    return {
        "source_to_target_coverage": float(np.mean(source_to_target <= target_radius)),
        "target_to_source_coverage": float(np.mean(target_to_source <= source_radius)),
        "source_radius": source_radius,
        "target_radius": target_radius,
        "source_label_mixing": float(mixing[: source.shape[0]].mean()),
        "target_label_mixing": float(mixing[source.shape[0] :].mean()),
    }


def ratio_ess(
    weights: np.ndarray,
    *,
    clip: float | None = None,
) -> dict[str, float | None]:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.size == 0 or not np.isfinite(weights).all():
        raise ValueError("weights must be non-empty and finite")
    if np.any(weights < 0.0):
        raise ValueError("density-ratio weights must be non-negative")
    if clip is not None:
        if float(clip) <= 0.0:
            raise ValueError("clip must be positive")
        weights = np.minimum(weights, float(clip))
    denominator = float(np.square(weights).sum())
    ess = 0.0 if denominator == 0.0 else float(weights.sum() ** 2 / denominator)
    return {
        "sample_count": float(weights.size),
        "ess": ess,
        "ess_fraction": ess / float(weights.size),
        "weight_mean": float(weights.mean()),
        "weight_max": float(weights.max()),
        "clip": None if clip is None else float(clip),
    }


def classifier_density_ratios(
    probabilities: np.ndarray,
    *,
    direction: str,
    epsilon: float = 1.0e-6,
) -> np.ndarray:
    """Convert balanced posterior probabilities into directional ratios."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("probabilities contain NaN or Inf")
    p = np.clip(probabilities, epsilon, 1.0 - epsilon)
    if direction == "source_to_target":
        return p / (1.0 - p)
    if direction == "target_to_source":
        return (1.0 - p) / p
    raise ValueError("direction must be source_to_target or target_to_source")


def bootstrap_ess_fraction(
    weights: np.ndarray,
    *,
    seed: int,
    replicates: int = 500,
    confidence: float = 0.95,
    clip: float | None = None,
) -> dict[str, float]:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if weights.size < 2 or int(replicates) < 2:
        raise ValueError("bootstrap requires at least two weights and replicates")
    rng = np.random.default_rng(int(seed))
    estimates = np.empty(int(replicates), dtype=np.float64)
    for index in range(int(replicates)):
        sample = weights[rng.integers(0, weights.size, size=weights.size)]
        estimates[index] = ratio_ess(sample, clip=clip)["ess_fraction"]
    alpha = 0.5 * (1.0 - float(confidence))
    return {
        "mean": float(estimates.mean()),
        "lower": float(np.quantile(estimates, alpha)),
        "upper": float(np.quantile(estimates, 1.0 - alpha)),
        "confidence": float(confidence),
        "replicates": float(replicates),
    }
