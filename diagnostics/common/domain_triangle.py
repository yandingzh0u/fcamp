from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from .classifiers import fit_balanced_source_classifier
from .overlap_metrics import (
    bidirectional_knn_diagnostics,
    bootstrap_ess_fraction,
    classifier_density_ratios,
    posterior_overlap_fraction,
    ratio_ess,
)
from .reward_validity import reward_seed_agreement


HELD_OUT_PAIR_SEED = 0x444F4D41494E


@dataclass(frozen=True, slots=True)
class DomainSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray

    def validate(self, name: str) -> None:
        dimensions: set[int] = set()
        for split_name in ("train", "validation", "test"):
            value = np.asarray(getattr(self, split_name), dtype=np.float64)
            if value.ndim != 2 or value.shape[0] < 2 or value.shape[1] < 1:
                raise ValueError(f"{name}.{split_name} must have shape [N,D]")
            if not np.isfinite(value).all():
                raise FloatingPointError(f"{name}.{split_name} contains NaN or Inf")
            dimensions.add(int(value.shape[1]))
        if len(dimensions) != 1:
            raise ValueError(f"{name} split dimensions differ")


def _balanced_pair(first: np.ndarray, second: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    count = min(first.shape[0], second.shape[0])
    if count < 2:
        raise ValueError("balanced domain pair needs at least two samples per class")
    rng = np.random.default_rng(int(seed))
    left = first[rng.choice(first.shape[0], size=count, replace=False)]
    right = second[rng.choice(second.shape[0], size=count, replace=False)]
    features = np.concatenate((left, right), axis=0)
    labels = np.concatenate((np.zeros(count, dtype=np.int64), np.ones(count, dtype=np.int64)))
    order = rng.permutation(features.shape[0])
    return features[order], labels[order]


def pairwise_source_matrix(
    domains: Mapping[str, DomainSplit],
    *,
    seeds: Sequence[int],
    calibrate: bool = True,
) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], list[dict[str, object]]]]:
    names = sorted(domains)
    if len(names) < 2:
        raise ValueError("at least two domains are required")
    for name in names:
        domains[name].validate(name)
    dimensions = {domains[name].train.shape[1] for name in names}
    if len(dimensions) != 1:
        raise ValueError("domain feature dimensions differ")
    matrix = {name: {other: (0.5 if name == other else float("nan")) for other in names} for name in names}
    details: dict[tuple[str, str], list[dict[str, object]]] = {}
    for first, second in combinations(names, 2):
        records: list[dict[str, object]] = []
        # Every discriminator seed is evaluated on the exact same held-out
        # state bank.  Changing the held-out rows with the model seed makes
        # ICC/Spearman undefined and was explicitly forbidden by the suite.
        test_x, test_y = _balanced_pair(
            domains[first].test,
            domains[second].test,
            seed=HELD_OUT_PAIR_SEED,
        )
        for seed in seeds:
            train_x, train_y = _balanced_pair(
                domains[first].train, domains[second].train, seed=int(seed)
            )
            result, _ = fit_balanced_source_classifier(
                train_x,
                train_y,
                test_x,
                test_y,
                seed=int(seed),
                calibrate=calibrate,
            )
            records.append(
                {
                    **result.as_dict(),
                    "probabilities": result.probabilities,
                    "labels": result.labels,
                }
            )
        mean_auc = float(np.mean([float(record["auc"]) for record in records]))
        matrix[first][second] = mean_auc
        matrix[second][first] = mean_auc
        details[(first, second)] = records
    return matrix, details


def reward_ordering_stability(seed_scores: Sequence[np.ndarray]) -> dict[str, float]:
    """Agreement of discriminator rewards on one aligned held-out bank."""

    arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in seed_scores]
    agreement = reward_seed_agreement(arrays)
    matrix = np.stack(arrays, axis=1)
    count = matrix.shape[0]
    decile_count = max(1, int(np.ceil(0.10 * count)))
    top_sets = [set(np.argsort(values)[-decile_count:].tolist()) for values in matrix.T]
    bottom_sets = [set(np.argsort(values)[:decile_count].tolist()) for values in matrix.T]

    def mean_jaccard(groups: Sequence[set[int]]) -> float:
        scores: list[float] = []
        for left in range(len(groups)):
            for right in range(left + 1, len(groups)):
                union = groups[left] | groups[right]
                scores.append(len(groups[left] & groups[right]) / max(1, len(union)))
        return float(np.mean(scores))

    return {
        **agreement,
        "top_decile_jaccard_mean": mean_jaccard(top_sets),
        "bottom_decile_jaccard_mean": mean_jaccard(bottom_sets),
        "held_out_sample_count": float(count),
        "seed_count": float(len(arrays)),
    }


def causal_group_swap_audit(
    first: DomainSplit,
    second: DomainSplit,
    *,
    groups: Mapping[str, tuple[int, int]],
    seed: int,
) -> dict[str, object]:
    """Diagnose source features with held-out bidirectional group swaps.

    This is only a source-difference intervention.  It is deliberately not a
    motion-quality score, a mask proposal, or evidence of reachability.
    """

    first.validate("first")
    second.validate("second")
    width = first.train.shape[1]
    if second.train.shape[1] != width:
        raise ValueError("domain feature dimensions differ")
    covered: set[int] = set()
    for name, (start, stop) in groups.items():
        if not name or not 0 <= int(start) < int(stop) <= width:
            raise ValueError(f"invalid feature group {name!r}: {(start, stop)}")
        indices = set(range(int(start), int(stop)))
        if covered & indices:
            raise ValueError("feature groups overlap")
        covered |= indices
    if covered != set(range(width)):
        raise ValueError("feature groups must partition the complete feature vector")

    train_x, train_y = _balanced_pair(first.train, second.train, seed=int(seed))
    test_x, test_y = _balanced_pair(first.test, second.test, seed=HELD_OUT_PAIR_SEED)
    baseline, model = fit_balanced_source_classifier(
        train_x, train_y, test_x, test_y, seed=int(seed), calibrate=True
    )
    first_rows = np.flatnonzero(test_y == 0)
    second_rows = np.flatnonzero(test_y == 1)
    if first_rows.size != second_rows.size:
        raise RuntimeError("balanced held-out source bank lost class balance")
    pooled_mean = train_x.mean(axis=0)
    records: dict[str, object] = {}
    for name, (start, stop) in groups.items():
        swapped = test_x.copy()
        swapped[first_rows, int(start) : int(stop)] = test_x[
            second_rows, int(start) : int(stop)
        ]
        swapped[second_rows, int(start) : int(stop)] = test_x[
            first_rows, int(start) : int(stop)
        ]
        probabilities = np.asarray(model.predict_proba(swapped)[:, 1], dtype=np.float64)
        # Positive means the intervention moved each row toward the donor's
        # source label, which is the causal quantity relevant to a shortcut.
        toward_donor = np.concatenate(
            (
                probabilities[first_rows] - baseline.probabilities[first_rows],
                baseline.probabilities[second_rows] - probabilities[second_rows],
            )
        )
        erased = test_x.copy()
        erased[:, int(start) : int(stop)] = pooled_mean[int(start) : int(stop)]
        erased_probabilities = np.asarray(
            model.predict_proba(erased)[:, 1], dtype=np.float64
        )
        group_only = np.broadcast_to(pooled_mean, test_x.shape).copy()
        group_only[:, int(start) : int(stop)] = test_x[:, int(start) : int(stop)]
        group_only_probabilities = np.asarray(
            model.predict_proba(group_only)[:, 1], dtype=np.float64
        )
        erased_auc = float(roc_auc_score(test_y, erased_probabilities))
        records[name] = {
            "start": int(start),
            "stop": int(stop),
            "width": int(stop) - int(start),
            "group_only_frozen_auc": float(roc_auc_score(test_y, group_only_probabilities)),
            "erase_group_frozen_auc": erased_auc,
            "baseline_auc_minus_erasure_auc": baseline.auc - erased_auc,
            "swap_auc": float(roc_auc_score(test_y, probabilities)),
            "swap_toward_donor_probability_mean": float(np.mean(toward_donor)),
        }
    return {
        "baseline": baseline.as_dict(),
        "groups": records,
        "scope": "source-difference diagnosis only; never a candidate mask or reachability claim",
    }


def effective_overlap_audit(
    source_validation: np.ndarray,
    target_validation: np.ndarray,
    source_test: np.ndarray,
    target_test: np.ndarray,
    *,
    source_train: np.ndarray | None = None,
    target_train: np.ndarray | None = None,
    seed: int,
    taus: Sequence[float] = (0.05, 0.10, 0.20),
    ratio_clips: Sequence[float] = (10.0, 20.0, 100.0),
    k: int = 5,
) -> dict[str, object]:
    source_validation = np.asarray(source_validation, dtype=np.float64)
    target_validation = np.asarray(target_validation, dtype=np.float64)
    source_test = np.asarray(source_test, dtype=np.float64)
    target_test = np.asarray(target_test, dtype=np.float64)
    if (source_train is None) != (target_train is None):
        raise ValueError("source_train and target_train must be supplied together")
    classifier_source = source_validation if source_train is None else np.asarray(source_train, dtype=np.float64)
    classifier_target = target_validation if target_train is None else np.asarray(target_train, dtype=np.float64)
    pooled_validation = np.concatenate((source_validation, target_validation), axis=0)
    knn_mean = pooled_validation.mean(axis=0)
    knn_std = pooled_validation.std(axis=0)
    knn_std = np.maximum(knn_std, 1.0e-8)
    train_x, train_y = _balanced_pair(classifier_source, classifier_target, seed=seed)
    test_x, test_y = _balanced_pair(source_test, target_test, seed=HELD_OUT_PAIR_SEED)
    classifier, _ = fit_balanced_source_classifier(
        train_x, train_y, test_x, test_y, seed=seed, calibrate=True
    )
    posterior = {
        str(float(tau)): posterior_overlap_fraction(
            classifier.probabilities, classifier.labels, tau=float(tau)
        )
        for tau in taus
    }
    source_mask = classifier.labels == 0
    target_mask = classifier.labels == 1
    forward_weights = classifier_density_ratios(
        classifier.probabilities[source_mask], direction="source_to_target"
    )
    reverse_weights = classifier_density_ratios(
        classifier.probabilities[target_mask], direction="target_to_source"
    )
    ess: dict[str, object] = {"forward": {}, "reverse": {}}
    ess["forward"]["unclipped"] = {
        **ratio_ess(forward_weights),
        "bootstrap": bootstrap_ess_fraction(forward_weights, seed=seed),
    }
    ess["reverse"]["unclipped"] = {
        **ratio_ess(reverse_weights),
        "bootstrap": bootstrap_ess_fraction(reverse_weights, seed=seed + 1),
    }
    for clip in ratio_clips:
        label = str(float(clip))
        ess["forward"][label] = {
            **ratio_ess(forward_weights, clip=float(clip)),
            "bootstrap": bootstrap_ess_fraction(
                forward_weights, seed=seed, clip=float(clip)
            ),
        }
        ess["reverse"][label] = {
            **ratio_ess(reverse_weights, clip=float(clip)),
            "bootstrap": bootstrap_ess_fraction(
                reverse_weights, seed=seed + 1, clip=float(clip)
            ),
        }
    return {
        "classifier": classifier.as_dict(),
        "posterior_overlap": posterior,
        "knn": bidirectional_knn_diagnostics(
            (source_test - knn_mean) / knn_std,
            (target_test - knn_mean) / knn_std,
            k=k,
        ),
        "ratio_ess": ess,
        "knn_normalization": {
            "fit_split": "pooled_validation_only",
            "minimum_std": 1.0e-8,
        },
        "classifier_fit_split": "train" if source_train is not None else "validation_legacy_call",
        "interpretation": "empirical effective overlap; not mathematical support or policy reachability",
    }


def normalized_gap(first: np.ndarray, second: np.ndarray, *, scale: np.ndarray) -> dict[str, float]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)
    if first.shape != second.shape or first.ndim != 2 or scale.shape != (first.shape[1],):
        raise ValueError("gap inputs must align as [N,D] with scale [D]")
    residual = (first - second) / np.maximum(scale, 1.0e-8)
    return {
        "rms": float(np.sqrt(np.mean(np.square(residual)))),
        "mean_l2": float(np.mean(np.linalg.norm(residual, axis=1) / np.sqrt(residual.shape[1]))),
        "max_abs": float(np.max(np.abs(residual))),
    }
