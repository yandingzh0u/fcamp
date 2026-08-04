from __future__ import annotations

import pytest

from diagnostics.common.edge_causality import (
    EVALUATION_CRITIC_ROLE,
    TRAINING_CRITIC_ROLE,
    validate_evaluation_critics,
)
from diagnostics.common.manifest import ProtocolError


SEEDS = (20260803, 20260804, 20260805, 20260806, 20260807)


def _evaluation() -> dict:
    return {
        "role": EVALUATION_CRITIC_ROLE,
        "artifacts": [
            {
                "seed": seed,
                "path": f"/models/held_out_{seed}.pt",
                "sha256": f"{index + 1:064x}",
                "role": EVALUATION_CRITIC_ROLE,
            }
            for index, seed in enumerate(SEEDS)
        ],
    }


def test_five_independent_diag35_critics_are_accepted() -> None:
    training = [
        {
            "path": "/edge/training.pt",
            "sha256": "f" * 64,
            "role": TRAINING_CRITIC_ROLE,
        }
    ]
    validate_evaluation_critics(
        _evaluation(), training_critic_artifacts=training, expected_seeds=SEEDS
    )


def test_training_critic_path_or_hash_cannot_be_reused_for_evaluation() -> None:
    evaluation = _evaluation()
    leaked = dict(evaluation["artifacts"][2])
    leaked["role"] = TRAINING_CRITIC_ROLE
    with pytest.raises(ProtocolError, match="leaked"):
        validate_evaluation_critics(
            evaluation,
            training_critic_artifacts=[leaked],
            expected_seeds=SEEDS,
        )

    same_hash_different_path = {
        "path": "/edge/copied_training.pt",
        "sha256": evaluation["artifacts"][0]["sha256"],
        "role": TRAINING_CRITIC_ROLE,
    }
    with pytest.raises(ProtocolError, match="leaked"):
        validate_evaluation_critics(
            evaluation,
            training_critic_artifacts=[same_hash_different_path],
            expected_seeds=SEEDS,
        )


def test_missing_seed_or_wrong_role_fails_closed() -> None:
    missing = _evaluation()
    missing["artifacts"] = missing["artifacts"][:-1]
    with pytest.raises(ProtocolError, match="seeds"):
        validate_evaluation_critics(
            missing, training_critic_artifacts=[], expected_seeds=SEEDS
        )
    wrong_role = _evaluation()
    wrong_role["role"] = "training"
    with pytest.raises(ProtocolError, match="held out"):
        validate_evaluation_critics(
            wrong_role, training_critic_artifacts=[], expected_seeds=SEEDS
        )
