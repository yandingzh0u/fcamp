from pathlib import Path

import pytest

from diagnostics.common.observation_spec import (
    ObservationCardinality,
    ObservationSpecError,
    derive_observation_spec,
    derive_observation_specs,
    observation_term_order,
)


ROOT = Path(__file__).resolve().parents[2]
OBSERVATION_SOURCE = ROOT / "envs" / "observation.py"


def _cardinality() -> ObservationCardinality:
    return ObservationCardinality(
        action_joint_count=29,
        termination_body_count=4,
        termination_contact_body_count=4,
        foot_body_count=2,
        track_body_count=14,
    )


def test_named_slices_follow_production_torch_cat_order() -> None:
    specs = derive_observation_specs(OBSERVATION_SOURCE, _cardinality())
    actor_order = tuple(name for name, _ in observation_term_order(OBSERVATION_SOURCE, stream="actor"))
    critic_order = tuple(name for name, _ in observation_term_order(OBSERVATION_SOURCE, stream="critic"))

    assert tuple(specs["actor"].named_slices) == actor_order
    assert tuple(specs["critic"].named_slices) == critic_order
    assert specs["actor"].total_dim == 171
    assert specs["critic"].total_dim == 286

    for spec in specs.values():
        assert spec.terms[0].start == 0
        assert spec.terms[-1].stop == spec.total_dim
        assert all(left.stop == right.start for left, right in zip(spec.terms, spec.terms[1:]))


def test_source_parser_tracks_reordered_terms_without_importing_env(tmp_path: Path) -> None:
    source = OBSERVATION_SOURCE.read_text(encoding="utf-8")
    source = source.replace(
        "reference_joint_state,\n                motion_anchor_pos_b,",
        "motion_anchor_pos_b,\n                reference_joint_state,",
        1,
    )
    reordered = tmp_path / "observation.py"
    reordered.write_text(source, encoding="utf-8")

    order = observation_term_order(reordered, stream="actor")
    assert [name for name, _ in order[:2]] == ["motion_anchor_pos_b", "reference_joint_state"]


def test_unreviewed_observation_term_cannot_silently_enter_proprio(tmp_path: Path) -> None:
    source = OBSERVATION_SOURCE.read_text(encoding="utf-8")
    source = source.replace(
        "reference_joint_state,\n                motion_anchor_pos_b,",
        "reference_joint_state,\n                unknown_future_context,\n                motion_anchor_pos_b,",
        1,
    )
    changed = tmp_path / "observation.py"
    changed.write_text(source, encoding="utf-8")
    widths = _cardinality().term_widths()
    widths["unknown_future_context"] = 7

    with pytest.raises(ObservationSpecError, match="no audited reference/proprio role"):
        derive_observation_spec(changed, stream="actor", term_widths=widths)
