from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from engine.checkpoint import (
    Checkpointer,
    audit_fixed_reward_checkpoint_payload,
)
from engine.config import load_config
from components.rollout.fixed_reward_contract import (
    FIXED_REWARD_CHECKPOINT_CONTRACT,
)


ROOT = Path(__file__).resolve().parents[1]


def _payload() -> dict:
    return {
        "update_idx": 50,
        "config": {"method": "fixed_reward"},
        "policy": {
            "actor.weight": torch.zeros(1, 1),
            "actor_obs_normalizer.count": torch.tensor(0),
            "critic.weight": torch.zeros(1, 1),
            "prefix_context_normalizer.count": torch.tensor(0),
        },
        "optimizer": {"state": {}, "param_groups": []},
        "metrics": {"reward/mean": 0.05},
        "algo_state": {
            **FIXED_REWARD_CHECKPOINT_CONTRACT,
            "critic_optimizer": {"state": {}, "param_groups": []},
            "learning_rate": 3.0e-4,
            "critic_learning_rate": 3.0e-4,
            "actor_obs_normalizer": {},
            "stream_ids": torch.tensor([0, 1]),
            "phase0_stream_count": 1,
            "phase0_stream_fraction": 0.1,
            "phase0_attempt_tracker": {
                "phase0_mask": torch.tensor([True, False]),
                "active": torch.tensor([True, False]),
                "ages": torch.tensor([0, 0]),
                "cumulative": {
                    "started": 1,
                    "failed": 0,
                    "timed_out": 0,
                    "succeeded": 0,
                    "interrupted": 0,
                },
            },
        },
        "env_transitions_total": 9_830_400,
        "train_wall_seconds_total": 12.0,
        "platform_identity": {
            "dataset_sha256": "dataset",
            "robot_asset_sha256": "robot",
            "action_schema_sha256": "actions",
        },
        "adaptive_sampler_state": {"version": 3},
        "torch_rng_state": torch.random.get_rng_state(),
    }


def test_schema_four_checkpoint_round_trip_passes_recursive_audit(
    tmp_path,
) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save(_payload(), path)

    restored = torch.load(path, map_location="cpu", weights_only=False)
    audit_fixed_reward_checkpoint_payload(restored)

    assert set(restored["policy"]) == {
        "actor.weight",
        "actor_obs_normalizer.count",
        "critic.weight",
        "prefix_context_normalizer.count",
    }
    assert set(restored["algo_state"]) == {
        "critic_optimizer",
        "learning_rate",
        "critic_learning_rate",
        "actor_obs_normalizer",
        "fixed_reward_schema_version",
        "policy_semantics",
        "action_semantics",
        "cps_semantics",
        "gae_semantics",
        "stream_ids",
        "phase0_stream_count",
        "phase0_stream_fraction",
        "phase0_attempt_tracker",
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("algo_state", "disc_optimizer"), {}),
        (
            ("algo_state", "phase0_attempt_tracker", "replay_state"),
            {},
        ),
        (("policy", "discriminator.weight"), torch.zeros(1)),
        (("metrics", "channel_task_reward"), 0.0),
    ],
)
def test_checkpoint_audit_rejects_removed_state_recursively(
    path: tuple[str, ...],
    value,
) -> None:
    payload = deepcopy(_payload())
    node = payload
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value

    with pytest.raises(ValueError, match="removed subsystem state"):
        audit_fixed_reward_checkpoint_payload(payload)


def test_checkpoint_audit_rejects_schema_one_before_restore() -> None:
    payload = _payload()
    payload["algo_state"]["fixed_reward_schema_version"] = 1

    with pytest.raises(ValueError, match="schema version mismatch"):
        audit_fixed_reward_checkpoint_payload(payload)


@pytest.mark.parametrize(
    "semantic_key",
    [
        "policy_semantics",
        "action_semantics",
        "cps_semantics",
        "gae_semantics",
    ],
)
def test_checkpoint_audit_rejects_semantic_mismatch_before_restore(
    semantic_key: str,
) -> None:
    payload = _payload()
    payload["algo_state"][semantic_key] = "wrong_semantics"

    with pytest.raises(ValueError, match="semantic contract mismatch"):
        audit_fixed_reward_checkpoint_payload(payload)


def test_schema_four_checkpoint_cannot_omit_its_config() -> None:
    payload = _payload()
    del payload["config"]

    with pytest.raises(ValueError, match="top-level schema mismatch"):
        audit_fixed_reward_checkpoint_payload(payload)


class _LoadRecorder:
    def __init__(self) -> None:
        self.loaded = False

    def load_state_dict(self, state) -> None:
        del state
        self.loaded = True


class _PreflightFailureAlgorithm:
    def __init__(self) -> None:
        self.policy = _LoadRecorder()
        self.optimizer = _LoadRecorder()

    def validate_checkpoint_payload(self, payload: dict) -> None:
        del payload
        raise ValueError("preflight sentinel")


class _AuditMustFailAlgorithm:
    def __init__(self) -> None:
        self.policy = _LoadRecorder()
        self.optimizer = _LoadRecorder()


def test_algorithm_preflight_precedes_all_weight_and_optimizer_loads(
    tmp_path,
) -> None:
    cfg = load_config(ROOT / "configs" / "fixed_reward_largebox.yaml")
    payload = _payload()
    payload["config"] = asdict(cfg)
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)

    algo = _PreflightFailureAlgorithm()
    trainer = SimpleNamespace(
        cfg=cfg,
        algo=algo,
        env_cfg=SimpleNamespace(
            num_envs=cfg.environment.num_envs,
            sim_dt=cfg.environment.sim_dt,
        ),
        algo_cfg=cfg.parameters,
        train_cfg=cfg.training,
        env=SimpleNamespace(device="cpu"),
    )

    with pytest.raises(ValueError, match="preflight sentinel"):
        Checkpointer(trainer).load(path)

    assert not algo.policy.loaded
    assert not algo.optimizer.loaded


@pytest.mark.parametrize(
    ("field", "bad_value", "error"),
    [
        ("fixed_reward_schema_version", 1, "schema version mismatch"),
        (
            "cps_semantics",
            "legacy_path_smoothed_cps_v1",
            "semantic contract mismatch",
        ),
    ],
)
def test_schema_and_semantics_fail_before_weight_or_optimizer_restore(
    tmp_path,
    field: str,
    bad_value,
    error: str,
) -> None:
    cfg = load_config(ROOT / "configs" / "fixed_reward_largebox.yaml")
    payload = _payload()
    payload["config"] = asdict(cfg)
    payload["algo_state"][field] = bad_value
    path = tmp_path / "incompatible.pt"
    torch.save(payload, path)

    algo = _AuditMustFailAlgorithm()
    trainer = SimpleNamespace(
        cfg=cfg,
        algo=algo,
        env_cfg=SimpleNamespace(
            num_envs=cfg.environment.num_envs,
            sim_dt=cfg.environment.sim_dt,
        ),
        algo_cfg=cfg.parameters,
        train_cfg=cfg.training,
        env=SimpleNamespace(device="cpu"),
    )

    with pytest.raises(ValueError, match=error):
        Checkpointer(trainer).load(path)

    assert not algo.policy.loaded
    assert not algo.optimizer.loaded
