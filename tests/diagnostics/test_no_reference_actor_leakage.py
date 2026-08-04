from pathlib import Path

import torch

from diagnostics.common.feature_views import build_actor_feature_views, validate_actor_partition
from diagnostics.common.observation_spec import ObservationCardinality, derive_observation_spec


ROOT = Path(__file__).resolve().parents[2]


def _spec():
    return derive_observation_spec(
        ROOT / "envs" / "observation.py",
        stream="actor",
        cardinality=ObservationCardinality(29, 4, 4, 2, 14),
    )


def test_no_reference_view_is_invariant_to_every_reference_term() -> None:
    spec = _spec()
    validate_actor_partition(spec)
    baseline = torch.arange(2 * spec.total_dim, dtype=torch.float32).reshape(2, spec.total_dim)
    counterfactual = baseline.clone()
    for term in spec.reference_terms:
        counterfactual[..., term.slice] += 10_000.0

    original_views = build_actor_feature_views(baseline, spec)
    changed_views = build_actor_feature_views(counterfactual, spec)
    torch.testing.assert_close(
        original_views.actor_no_reference,
        changed_views.actor_no_reference,
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(
        original_views.actor_reference_terms,
        changed_views.actor_reference_terms,
    )


def test_reference_and_proprio_views_are_an_exact_partition() -> None:
    spec = _spec()
    observation = torch.randn(3, spec.total_dim)
    views = build_actor_feature_views(observation, spec)
    reference_width = sum(term.width for term in spec.reference_terms)
    proprio_width = sum(term.width for term in spec.proprio_terms)

    assert views.actor_reference_terms.shape[-1] == reference_width
    assert views.actor_no_reference.shape[-1] == proprio_width
    assert reference_width + proprio_width == spec.total_dim
    torch.testing.assert_close(views.actor_no_reference, views.actor_proprio_terms)
