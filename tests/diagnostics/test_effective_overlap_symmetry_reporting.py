from __future__ import annotations

import numpy as np

from diagnostics.common.overlap_metrics import bidirectional_knn_diagnostics


def test_effective_overlap_reports_both_directions() -> None:
    rng = np.random.default_rng(3)
    source = rng.normal(size=(120, 2))
    target = np.concatenate((rng.normal(size=(100, 2)), 20.0 + rng.normal(size=(20, 2))))
    result = bidirectional_knn_diagnostics(source, target, k=3)
    assert "source_to_target_coverage" in result
    assert "target_to_source_coverage" in result
    assert result["source_to_target_coverage"] != result["target_to_source_coverage"]
