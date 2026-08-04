from __future__ import annotations

import json
from pathlib import Path

import pytest

from diagnostics.common.edge_causality import (
    LocalEdgeProtocol,
    candidate_chains,
    evaluate_single_seed,
    load_real_edge_backend,
    select_preregistered_edges,
)
from diagnostics.common.manifest import DependencyUnavailable


SPEC = (
    Path(__file__).resolve().parents[2]
    / "diagnostics"
    / "specs"
    / "largebox_discovery_v1.yaml"
)


def _protocol() -> LocalEdgeProtocol:
    return LocalEdgeProtocol.from_spec(json.loads(SPEC.read_text(encoding="utf-8")))


def _candidate(source: int, target: int, overlap: float, *, eligible: bool = True) -> dict:
    return {
        "edge_id": f"e{source}_{target}",
        "source_update": source,
        "target_update": target,
        "outcome": {"strict_pareto_target_over_source": True},
        "overlap": {
            "eligible": eligible,
            "overlap_order_key": [overlap, overlap, overlap, overlap, -0.5],
        },
    }


def test_edge_selection_uses_only_recorded_overlap_and_pareto_candidates() -> None:
    candidates = [
        _candidate(100, 200, 0.7),
        _candidate(150, 200, 1.2),
        _candidate(200, 300, 1.1),
        _candidate(300, 500, 0.9),
        _candidate(50, 500, 0.5),
        _candidate(150, 250, 5.0, eligible=False),
    ]
    selected, audit = select_preregistered_edges(
        candidates,
        crossings={0.1: 200, 0.5: 300, 0.9: 500},
        final_update=500,
    )
    ids = {edge["edge_id"] for edge in selected}
    assert "e150_250" not in ids
    assert "e50_500" in ids  # frozen earliest-lower-quality direct control
    assert all(item.get("selected") for item in audit if "transition_level" in item)


def test_consecutive_chain_requires_exact_target_to_next_source_link() -> None:
    linked = [_candidate(100, 200, 1), _candidate(200, 300, 1), _candidate(300, 400, 1)]
    assert candidate_chains(linked, length=3) == [["e100_200", "e200_300", "e300_400"]]
    unlinked = [_candidate(100, 200, 1), _candidate(210, 300, 1), _candidate(300, 400, 1)]
    assert candidate_chains(unlinked, length=3) == []


def _online_result(training_delta: float, target_final: float) -> dict:
    protocol = _protocol()
    rewards0 = {str(seed): 0.2 for seed in protocol.held_out_critic_seeds}
    rewards1 = {str(seed): 0.3 for seed in protocol.held_out_critic_seeds}
    return {
        "seed": protocol.ppo_seeds[0],
        "evaluations": [
            {
                "distance_to_target": 1.0,
                "distance_to_source": 0.0,
                "failure_rate": 0.10,
                "task_or_progress": 0.8,
                "training_critic_score": 0.0,
                "held_out_rewards": rewards0,
            },
            {
                "distance_to_target": target_final,
                "distance_to_source": 0.5,
                "failure_rate": 0.10,
                "task_or_progress": 0.8,
                "training_critic_score": training_delta,
                "held_out_rewards": rewards1,
            },
        ],
    }


def test_training_critic_gain_alone_never_passes_an_edge() -> None:
    protocol = _protocol()
    audit = evaluate_single_seed(_online_result(1000.0, 0.95), protocol=protocol)
    assert audit["training_critic_score_delta"] == 1000.0
    assert audit["training_critic_score_used_for_pass"] is False
    assert audit["passed"] is False


def test_absent_real_backend_has_no_fixed_reward_fallback(tmp_path: Path) -> None:
    with pytest.raises(DependencyUnavailable, match="fixed_reward PPO"):
        load_real_edge_backend(tmp_path)
