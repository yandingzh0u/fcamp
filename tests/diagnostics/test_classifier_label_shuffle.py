from __future__ import annotations

import numpy as np

from diagnostics.common.classifiers import fit_balanced_source_classifier


def test_label_shuffle_returns_chance_source_auc() -> None:
    rng = np.random.default_rng(21)
    train_x = rng.normal(size=(800, 6))
    test_x = rng.normal(size=(400, 6))
    train_y = rng.integers(0, 2, size=800)
    test_y = rng.integers(0, 2, size=400)
    result, _ = fit_balanced_source_classifier(
        train_x, train_y, test_x, test_y, seed=9, calibrate=False
    )
    assert 0.40 <= result.auc <= 0.60
