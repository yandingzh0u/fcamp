from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from diagnostics.common.canonical_collection import (
    CANONICAL_INDEX_FIELDS,
    write_rollout_index,
)
from diagnostics.common.policy_class_probe import (
    PolicyClassProtocol,
    FrozenCheckpointPolicy,
    canonical_index,
    derive_repository_observation_specs,
    deterministic_group_splits,
    load_bc_inference,
    load_canonical_arrays,
    make_probe_dataset,
    save_model_bundle,
    select_policy_class_rows,
    train_action_phase_probe,
    _checkpoint_candidates,
)


ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "diagnostics" / "specs" / "largebox_discovery_v1.yaml"


def _spec() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


def test_policy_class_protocol_is_fully_frozen_in_spec() -> None:
    protocol = PolicyClassProtocol.from_spec(_spec())
    assert protocol.primary_update == 500
    assert protocol.seeds == (20260803, 20260804, 20260805)
    assert protocol.alias_histories == (1, 2, 4, 8, 16, 32)
    assert protocol.bc_histories == (1, 4, 8, 16, 32)
    assert protocol.mlp_hidden_dims == (256, 256)
    assert protocol.open_loop_perturbation_fractions == (0.001, 0.005, 0.01)
    assert protocol.open_loop_horizon == 50
    assert (protocol.b_model, protocol.b_history, protocol.b_seed) == (
        "GRU", 32, 20260803
    )
    assert protocol.b_collector_modes == (
        "clean_mean", "controlled_environment", "common_action_noise"
    )
    assert protocol.b_common_sigmas == (0.10, 0.25, 0.50)
    assert protocol.b_output_name == "canonical_B_rollout_index.parquet"


def test_stage2_observation_partition_is_ast_derived() -> None:
    specs, provenance = derive_repository_observation_specs(ROOT)
    assert specs["actor"].total_dim == 171
    assert specs["critic"].total_dim == 286
    assert [term.name for term in specs["actor"].terms][:2] == [
        "reference_joint_state",
        "motion_anchor_pos_b",
    ]
    assert provenance["cardinality"]["action_joint_count"] == 29


def _synthetic_index(tmp_path: Path) -> Path:
    time, envs, actor_dim, reference_dim, proprio_dim, action_dim = 6, 3, 7, 2, 5, 2
    actor = torch.arange(time * envs * actor_dim, dtype=torch.float32).reshape(time, envs, actor_dim)
    tree = {
        "metadata": {},
        "trajectory": {
            "phase_continuous": torch.arange(time, dtype=torch.float32)[:, None].repeat(1, envs),
            "phase": torch.arange(time)[:, None].repeat(1, envs),
            "contact_mode": torch.zeros(time, envs, dtype=torch.long),
            "step": torch.arange(time)[:, None].repeat(1, envs),
        },
        "observation": {
            "actor_full": actor,
            "actor_reference_terms": actor[..., :reference_dim],
            "actor_proprio_terms": actor[..., reference_dim:],
            "actor_no_reference": actor[..., reference_dim:],
            "critic_full": actor,
        },
        "state": {},
        "action": {
            "mean": actor[..., :action_dim] / 10.0,
            "std": torch.ones(time, envs, action_dim),
        },
        "reference": {},
        "outcome": {},
        "imitation": {
            "agent_physx_raw_frame": torch.zeros(time, envs, 239),
            "agent_fk_aligned_raw_frame": torch.zeros(time, envs, 239),
            "reference_expert_raw_frame": torch.zeros(time, envs, 239),
            "phase_normalized_pre_step": torch.linspace(0.0, 1.0, time)[:, None].repeat(1, envs),
        },
    }
    rollout_dir = tmp_path / "rollouts"
    rollout_dir.mkdir()
    torch.save(tree, rollout_dir / "u500.pt")
    rows = []
    for env_index in range(envs):
        rows.append(
            {
                "sample_id": f"sample-{env_index}",
                "trajectory_id": f"trajectory-{env_index}",
                "snapshot_id": f"snapshot-{env_index}",
                "env_id": env_index,
                "episode_id": env_index,
                "checkpoint_id": "u0500-test",
                "checkpoint_path": "/unavailable/u500.pt",
                "checkpoint_sha256": "a" * 64,
                "checkpoint_update": 500,
                "checkpoint_lineage_id": "lineage-test",
                "policy_domain": "teacher_fixed_reward",
                "collector_mode": "clean_mean",
                "common_sigma": 0.0,
                "eligible_for_primary_overlap": True,
                "shard_path": "rollouts/u500.pt",
                "shard_env_index": env_index,
                "num_steps": time,
            }
        )
    assert tuple(rows[0]) == CANONICAL_INDEX_FIELDS
    index = tmp_path / "canonical_rollout_index.parquet"
    write_rollout_index(rows, index)
    return index


def test_stage2_reads_only_canonical_index_and_shards(tmp_path: Path) -> None:
    index = _synthetic_index(tmp_path)
    rows = select_policy_class_rows(canonical_index(index), checkpoint_update=500)
    arrays = load_canonical_arrays(index, rows)
    assert arrays.actor_full.shape == (18, 7)
    assert arrays.actor_no_reference.shape == (18, 5)
    assert arrays.action_mean.shape == (18, 2)
    assert arrays.metadata["checkpoint_update"] == 500
    assert set(arrays.trajectory_ids) == {
        "trajectory-0", "trajectory-1", "trajectory-2"
    }


def test_group_split_never_fragments_snapshot_or_trajectory() -> None:
    trajectories = np.repeat([f"t{i}" for i in range(12)], 4)
    snapshots = np.repeat([f"s{i}" for i in range(12)], 4)
    split = deterministic_group_splits(trajectories, snapshots, seed=7)
    for identity in set(trajectories):
        assert len(set(split[trajectories == identity])) == 1
    for identity in set(snapshots):
        assert len(set(split[snapshots == identity])) == 1
    assert set(split) == {"train", "validation", "test"}


def test_frozen_checkpoint_policy_uses_checkpoint_actor_and_normalizer(tmp_path: Path) -> None:
    from models.holosoma_ppo import EmpiricalNormalization, PPOActor

    torch.manual_seed(7)
    actor = PPOActor(
        observation_dim=5,
        hidden_dims=(6, 4),
        activation="ELU",
        num_actions=2,
        init_noise_std=0.3,
    )
    normalizer = EmpiricalNormalization(5, "cpu")
    normalizer._mean.copy_(torch.linspace(-0.2, 0.2, 5).unsqueeze(0))
    normalizer._std.copy_(torch.linspace(0.8, 1.2, 5).unsqueeze(0))
    state = {
        **{f"actor.{name}": value for name, value in actor.state_dict().items()},
        **{
            f"actor_obs_normalizer.{name}": value
            for name, value in normalizer.state_dict().items()
        },
    }
    checkpoint = tmp_path / "update_0500.pt"
    torch.save(
        {
            "update_idx": 500,
            "config": {"parameters": {"activation": "ELU"}},
            "policy": state,
        },
        checkpoint,
    )
    observation = torch.randn(9, 5)
    with torch.no_grad():
        expected = actor.act_inference(normalizer(observation, update=False)).numpy()
    loaded = FrozenCheckpointPolicy(checkpoint)
    np.testing.assert_allclose(loaded.mean(observation.numpy()), expected, atol=1.0e-7)
    np.testing.assert_allclose(loaded.std, actor.std.detach().numpy(), atol=0.0)


def test_saved_bc_bundle_has_closed_loop_inference_interface(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    trajectories = np.repeat([f"t{i}" for i in range(12)], 8)
    snapshots = np.repeat([f"s{i}" for i in range(12)], 8)
    features = rng.normal(size=(96, 3)).astype(np.float32)
    actions = np.stack((features[:, 0] + features[:, 1], features[:, 2]), axis=1).astype(np.float32)
    from diagnostics.common.policy_class_probe import CanonicalArrays

    arrays = CanonicalArrays(
        actor_full=features,
        actor_no_reference=features,
        actor_reference_terms=features[:, :1],
        actor_proprio_terms=features,
        action_mean=actions,
        action_std=np.ones_like(actions),
        phases=np.tile(np.arange(8), 12).astype(float),
        contact_modes=np.zeros(96),
        trajectory_ids=trajectories,
        snapshot_ids=snapshots,
        steps=np.tile(np.arange(8), 12),
        sample_ids=np.repeat([f"x{i}" for i in range(12)], 8),
        metadata={},
    )
    splits = deterministic_group_splits(trajectories, snapshots, seed=9)
    dataset = make_probe_dataset(features, arrays, splits, history=1)
    metrics, bundle = train_action_phase_probe(
        dataset,
        kind="mlp",
        seed=11,
        epochs=2,
        batch_size=16,
        hidden_dim=8,
        max_train_samples=100,
        patience=2,
        predict_phase=False,
    )
    assert np.isfinite(metrics["action"]["nrmse"])
    path = tmp_path / "bc.pt"
    save_model_bundle(path, bundle)
    policy = load_bc_inference(path)
    policy.reset()
    prediction = policy.act(torch.as_tensor(features[:5]))
    assert tuple(prediction.shape) == (5, 2)
    assert torch.isfinite(prediction).all()


def test_policy_class_summary_does_not_turn_missing_assets_into_fail(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "diag_27_policy_class_summary.py"),
            "--spec",
            str(SPEC),
            "--output-dir",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    assert "SKIPPED_DEPENDENCY" in completed.stdout
    result = json.loads((tmp_path / "policy_class_gate.json").read_text())
    assert result["status"] == "SKIPPED_DEPENDENCY"
    assert result["evidence"]["gate"] == "UNKNOWN"


def test_checkpoint_candidates_accept_both_declared_inventory_path_columns(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    dense = output / "checkpoints"
    dense.mkdir(parents=True)
    full_path = tmp_path / "full.pt"
    dense_path = tmp_path / "dense.pt"
    pd.DataFrame(
        [{"checkpoint_sha256": "a" * 64, "path": str(full_path)}]
    ).to_csv(output / "checkpoint_inventory.csv", index=False)
    pd.DataFrame(
        [{"checkpoint_sha256": "a" * 64, "checkpoint_path": str(dense_path)}]
    ).to_csv(dense / "dense_checkpoint_inventory.csv", index=False)

    candidates = _checkpoint_candidates(
        tmp_path,
        output,
        {"checkpoint_sha256": "a" * 64, "checkpoint_update": 500},
        None,
    )
    assert full_path.resolve() in candidates
    assert dense_path.resolve() in candidates
