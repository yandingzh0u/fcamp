import pytest

from diagnostics.common.noise_bank import (
    CollectorMode,
    NoiseProtocolError,
    is_overlap_eligible,
    overlap_eligible_indices,
    require_overlap_eligible,
)


def test_native_checkpoint_noise_is_never_overlap_eligible() -> None:
    assert not is_overlap_eligible(CollectorMode.NATIVE_STOCHASTIC)
    assert all(
        is_overlap_eligible(mode)
        for mode in (
            CollectorMode.CLEAN_MEAN,
            CollectorMode.CONTROLLED_ENVIRONMENT,
            CollectorMode.COMMON_ACTION_NOISE,
        )
    )
    with pytest.raises(NoiseProtocolError, match="excluded from effective-overlap"):
        require_overlap_eligible(
            [CollectorMode.CLEAN_MEAN, CollectorMode.NATIVE_STOCHASTIC]
        )


def test_overlap_filter_drops_native_stochastic_rows_explicitly() -> None:
    modes = [
        "clean_mean",
        "native_stochastic",
        "common_action_noise",
        "native_stochastic",
        "controlled_environment",
    ]
    assert overlap_eligible_indices(modes) == (0, 2, 4)
