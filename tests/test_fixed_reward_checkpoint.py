from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from components.rollout.fixed_reward_contract import FIXED_REWARD_CHECKPOINT_CONTRACT
from engine.checkpoint import (
    FIXED_REWARD_SCHEMA_VERSION,
    audit_fixed_reward_checkpoint_payload,
)


def _payload() -> dict:
    return {
        "update_idx": 3,
        "config": {"method": "fixed_reward"},
        "policy": {
            "actor.actor_module.0.weight": torch.zeros(1),
            "actor.std": torch.ones(1),
            "actor_obs_normalizer._mean": torch.zeros(1),
            "critic.critic_module.0.weight": torch.zeros(1),
            "critic_obs_normalizer._mean": torch.zeros(1),
        },
        "optimizer": {"state": {}, "param_groups": []},
        "metrics": {},
        "algo_state": {
            **FIXED_REWARD_CHECKPOINT_CONTRACT,
            "critic_optimizer": {"state": {}, "param_groups": []},
            "actor_learning_rate": 0.001,
            "critic_learning_rate": 0.001,
            "stream_ids": torch.tensor([0, 1]),
            "phase0_stream_count": 1,
            "phase0_stream_fraction": 0.1,
            "phase0_attempt_tracker": {},
            "actor_optimizer_steps_total": 60,
            "critic_optimizer_steps_total": 60,
        },
        "env_transitions_total": 9_216,
        "train_wall_seconds_total": 1.0,
        "platform_identity": {},
        "adaptive_sampler_state": {},
        "torch_rng_state": torch.random.get_rng_state(),
    }


def test_schema_17_plain_ppo_payload_is_accepted() -> None:
    assert FIXED_REWARD_SCHEMA_VERSION == 17
    audit_fixed_reward_checkpoint_payload(_payload())


@pytest.mark.parametrize("old_schema", range(1, 17))
def test_schema_1_through_16_are_rejected_before_restore(old_schema: int) -> None:
    payload = _payload()
    payload["algo_state"]["fixed_reward_schema_version"] = old_schema
    with pytest.raises(ValueError, match="schema version mismatch"):
        audit_fixed_reward_checkpoint_payload(payload)


def test_checkpoint_requires_exact_holosoma_contract() -> None:
    for key in FIXED_REWARD_CHECKPOINT_CONTRACT:
        if key == "fixed_reward_schema_version":
            continue
        payload = _payload()
        payload["algo_state"][key] = "wrong"
        with pytest.raises(ValueError, match="semantic contract mismatch"):
            audit_fixed_reward_checkpoint_payload(payload)


@pytest.mark.parametrize(
    "path",
    [
        ("policy", "actor.cps_diag_raw"),
        ("policy", "actor.flow_velocity.weight"),
        ("algo_state", "exploration_optimizer"),
        ("algo_state", "latent_path"),
        ("algo_state", "discriminator"),
    ],
)
def test_removed_algorithm_state_is_rejected(path: tuple[str, str]) -> None:
    payload = _payload()
    payload[path[0]][path[1]] = torch.zeros(1)
    with pytest.raises(ValueError):
        audit_fixed_reward_checkpoint_payload(payload)


def test_checkpoint_has_only_actor_and_critic_optimizers() -> None:
    payload = _payload()
    audit_fixed_reward_checkpoint_payload(payload)
    assert "optimizer" in payload
    assert "critic_optimizer" in payload["algo_state"]
    assert "exploration_optimizer" not in payload["algo_state"]
    assert "flow_optimizer" not in payload["algo_state"]


def test_unknown_top_level_state_is_rejected() -> None:
    payload = deepcopy(_payload())
    payload["world_model"] = {}
    with pytest.raises(ValueError):
        audit_fixed_reward_checkpoint_payload(payload)
