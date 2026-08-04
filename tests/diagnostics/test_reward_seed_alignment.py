from __future__ import annotations

import numpy as np
import pytest

from diagnostics.common.reward_validity import reward_seed_agreement


def test_reward_seed_alignment_requires_same_state_bank() -> None:
    with pytest.raises(ValueError, match="aligned"):
        reward_seed_agreement([np.arange(5.0), np.arange(6.0)])


def test_reward_seed_alignment_is_high_for_monotone_seeds() -> None:
    base = np.linspace(-2.0, 2.0, 100)
    result = reward_seed_agreement([base, base + 3.0, base - 1.0])
    assert result["pairwise_spearman_min"] > 0.999
    assert result["icc_consistency"] > 0.999
