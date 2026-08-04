"""Named feature projections built from :mod:`observation_spec`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .observation_spec import FeatureRole, ObservationSpec, ObservationSpecError


@dataclass(frozen=True)
class ActorFeatureViews:
    actor_full: torch.Tensor
    actor_reference_terms: torch.Tensor
    actor_proprio_terms: torch.Tensor
    actor_no_reference: torch.Tensor


def _indices_for_role(spec: ObservationSpec, role: FeatureRole) -> torch.Tensor:
    indices = [
        index
        for term in spec.terms
        if term.role is role
        for index in range(term.start, term.stop)
    ]
    return torch.tensor(indices, dtype=torch.long)


def select_named_terms(
    observation: torch.Tensor,
    spec: ObservationSpec,
    names: Iterable[str],
) -> torch.Tensor:
    if observation.shape[-1] != spec.total_dim:
        raise ObservationSpecError(
            f"{spec.stream} observation has width {observation.shape[-1]}, "
            f"expected {spec.total_dim}"
        )
    selected = tuple(names)
    if len(set(selected)) != len(selected):
        raise ObservationSpecError("named feature selection contains duplicates")
    slices = spec.named_slices
    missing = [name for name in selected if name not in slices]
    if missing:
        raise ObservationSpecError(f"unknown observation terms: {missing}")
    parts = [observation[..., slices[name]] for name in selected]
    if not parts:
        return observation[..., :0]
    return torch.cat(parts, dim=-1)


def validate_actor_partition(spec: ObservationSpec) -> None:
    if spec.stream != "actor":
        raise ObservationSpecError("actor feature views require an actor observation spec")
    reference = set(_indices_for_role(spec, FeatureRole.REFERENCE).tolist())
    proprio = set(_indices_for_role(spec, FeatureRole.PROPRIO).tolist())
    if reference & proprio:
        raise ObservationSpecError("reference and proprio observation indices overlap")
    expected = set(range(spec.total_dim))
    if reference | proprio != expected:
        raise ObservationSpecError("reference/proprio partition does not cover actor observation")


def build_actor_feature_views(
    observation: torch.Tensor,
    spec: ObservationSpec,
) -> ActorFeatureViews:
    """Build leakage-safe actor views using named terms, never offsets."""

    validate_actor_partition(spec)
    if observation.shape[-1] != spec.total_dim:
        raise ObservationSpecError(
            f"actor observation has width {observation.shape[-1]}, expected {spec.total_dim}"
        )
    reference_names = [term.name for term in spec.reference_terms]
    proprio_names = [term.name for term in spec.proprio_terms]
    reference = select_named_terms(observation, spec, reference_names)
    proprio = select_named_terms(observation, spec, proprio_names)
    # Deliberately return a distinct field: downstream schemas call this view
    # actor_no_reference, while actor_proprio_terms names its provenance.
    no_reference = proprio.clone()
    return ActorFeatureViews(
        actor_full=observation,
        actor_reference_terms=reference,
        actor_proprio_terms=proprio,
        actor_no_reference=no_reference,
    )
