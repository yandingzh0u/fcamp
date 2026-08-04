from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True, slots=True)
class SourceClassifierResult:
    """Held-out outputs from one balanced source-classification probe."""

    seed: int
    probabilities: np.ndarray
    labels: np.ndarray
    auc: float
    accuracy: float
    nll: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "seed": self.seed,
            "auc": self.auc,
            "accuracy": self.accuracy,
            "nll": self.nll,
            "sample_count": int(self.labels.size),
        }


def _matrix(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape [N,D], got {array.shape}")
    if not np.isfinite(array).all():
        raise FloatingPointError(f"{name} contains NaN or Inf")
    return array


def _labels(name: str, value: np.ndarray, count: int) -> np.ndarray:
    labels = np.asarray(value, dtype=np.int64).reshape(-1)
    if labels.shape != (count,):
        raise ValueError(f"{name} must have shape {(count,)}, got {labels.shape}")
    if set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError(f"{name} must contain both binary classes")
    return labels


def fit_balanced_source_classifier(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    seed: int,
    calibrate: bool = True,
    calibration_folds: int = 3,
    max_iter: int = 1000,
) -> tuple[SourceClassifierResult, object]:
    """Fit a reproducible linear probe without touching the held-out split.

    The deliberately simple probe is an audit instrument, not a certificate of
    distributional support.  Calibration is learned exclusively on ``train_x``.
    """

    train_x = _matrix("train_x", train_x)
    test_x = _matrix("test_x", test_x)
    if train_x.shape[1] != test_x.shape[1]:
        raise ValueError("train/test feature dimensions differ")
    train_y = _labels("train_y", train_y, train_x.shape[0])
    test_y = _labels("test_y", test_y, test_x.shape[0])

    base = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=int(max_iter),
                    random_state=int(seed),
                    solver="lbfgs",
                ),
            ),
        ]
    )
    if calibrate:
        counts = np.bincount(train_y, minlength=2)
        folds = min(int(calibration_folds), int(counts.min()))
        if folds < 2:
            raise ValueError("calibration requires at least two samples per class")
        model: object = CalibratedClassifierCV(base, method="sigmoid", cv=folds)
    else:
        model = base
    model.fit(train_x, train_y)  # type: ignore[attr-defined]
    probabilities = np.asarray(
        model.predict_proba(test_x)[:, 1],  # type: ignore[attr-defined]
        dtype=np.float64,
    )
    if not np.isfinite(probabilities).all():
        raise FloatingPointError("classifier emitted non-finite probabilities")
    predictions = probabilities >= 0.5
    result = SourceClassifierResult(
        seed=int(seed),
        probabilities=probabilities,
        labels=test_y,
        auc=float(roc_auc_score(test_y, probabilities)),
        accuracy=float(accuracy_score(test_y, predictions)),
        nll=float(log_loss(test_y, probabilities, labels=(0, 1))),
    )
    return result, model


def seed_probability_agreement(
    probabilities: Iterable[np.ndarray],
) -> dict[str, float]:
    """Report seed agreement without treating consistency as correctness."""

    arrays = [np.asarray(value, dtype=np.float64).reshape(-1) for value in probabilities]
    if len(arrays) < 2:
        raise ValueError("at least two discriminator seeds are required")
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("seed probability arrays are not aligned")
    matrix = np.stack(arrays)
    if not np.isfinite(matrix).all():
        raise FloatingPointError("seed probabilities contain NaN or Inf")
    correlations = np.corrcoef(matrix)
    upper = correlations[np.triu_indices(len(arrays), k=1)]
    signs = matrix >= 0.5
    disagreement = np.mean(np.any(signs != signs[:1], axis=0))
    return {
        "pairwise_pearson_mean": float(np.nanmean(upper)),
        "sign_disagreement": float(disagreement),
    }
