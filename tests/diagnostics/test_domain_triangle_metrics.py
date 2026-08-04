from __future__ import annotations

import numpy as np

from diagnostics.common.domain_triangle import DomainSplit, effective_overlap_audit, pairwise_source_matrix


def _domain(rng: np.random.Generator, shift: float) -> DomainSplit:
    return DomainSplit(
        train=shift + rng.normal(size=(120, 3)),
        validation=shift + rng.normal(size=(80, 3)),
        test=shift + rng.normal(size=(80, 3)),
    )


def test_pairwise_source_matrix_is_explicitly_symmetric() -> None:
    rng = np.random.default_rng(17)
    matrix, _ = pairwise_source_matrix(
        {"K": _domain(rng, 0.0), "T": _domain(rng, 2.5)},
        seeds=(1,),
        calibrate=False,
    )
    assert matrix["K"]["T"] == matrix["T"]["K"]
    assert matrix["K"]["T"] > 0.9


def test_effective_overlap_audit_never_claims_reachability() -> None:
    rng = np.random.default_rng(5)
    source_validation = rng.normal(size=(100, 2))
    target_validation = rng.normal(size=(100, 2))
    source_test = rng.normal(size=(80, 2))
    target_test = rng.normal(size=(80, 2))
    result = effective_overlap_audit(
        source_validation,
        target_validation,
        source_test,
        target_test,
        seed=3,
        ratio_clips=(10.0,),
        taus=(0.1,),
        k=3,
    )
    assert "not mathematical support" in result["interpretation"]
    assert "forward" in result["ratio_ess"] and "reverse" in result["ratio_ess"]
