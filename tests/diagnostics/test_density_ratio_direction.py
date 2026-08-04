from __future__ import annotations

import numpy as np

from diagnostics.common.overlap_metrics import classifier_density_ratios


def test_density_ratio_directions_are_reciprocal() -> None:
    posterior = np.asarray([0.2, 0.4, 0.8])
    forward = classifier_density_ratios(
        posterior, direction="source_to_target"
    )
    reverse = classifier_density_ratios(
        posterior, direction="target_to_source"
    )
    assert np.allclose(forward * reverse, 1.0)
