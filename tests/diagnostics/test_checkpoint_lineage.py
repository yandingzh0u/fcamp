from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from diagnostics.common.checkpoint_io import inventory_checkpoints


def _config(*, resume: str = "") -> dict:
    return {
        "method": "fixed_reward",
        "environment": {"task": "largebox_plane", "sim_dt": 0.02},
        "parameters": {"gamma": 0.99},
        "training": {"seed": 42, "resume": resume, "max_updates": 500},
    }


def _write_checkpoint(run: Path, update: int, *, resume: str = "", policy_value: float = 0.0) -> Path:
    checkpoint_dir = run / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = _config(resume=resume)
    (run / "resolved_config.json").write_text(
        json.dumps({**config, "source_snapshot_sha256": "c" * 64}),
        encoding="utf-8",
    )
    payload = {
        "update_idx": update,
        "config": config,
        "policy": {
            "actor.std": torch.ones(2) * policy_value,
            "actor_obs_normalizer.count": torch.tensor(update),
            "critic_obs_normalizer.count": torch.tensor(update),
        },
        "optimizer": {"state": {}, "param_groups": []},
        "metrics": {},
        "algo_state": {
            "fixed_reward_schema_version": 17,
            "critic_optimizer": {"state": {}, "param_groups": []},
            "actor_optimizer_steps_total": update,
            "critic_optimizer_steps_total": update,
        },
        "env_transitions_total": update * 10,
        "platform_identity": {
            "dataset_sha256": "d" * 64,
            "robot_asset_sha256": "e" * 64,
            "action_schema_sha256": "f" * 64,
        },
        "adaptive_sampler_state": {"version": 1},
        "torch_rng_state": torch.random.get_rng_state(),
    }
    path = checkpoint_dir / f"update_{update:04d}.pt"
    torch.save(payload, path)
    return path


def test_checkpoints_in_one_run_share_lineage_and_updates_match(tmp_path) -> None:
    first = _write_checkpoint(tmp_path / "run_a", 50)
    second = _write_checkpoint(tmp_path / "run_a", 100, policy_value=1.0)
    records = inventory_checkpoints([first, second])

    assert len({record["checkpoint_lineage_id"] for record in records}) == 1
    assert len({record["checkpoint_branch_id"] for record in records}) == 1
    assert all(record["lineage_coherent"] for record in records)
    assert all(record["update_match"] for record in records)


def test_fresh_runs_are_not_silently_merged_but_resume_inherits_ancestry(tmp_path) -> None:
    parent = _write_checkpoint(tmp_path / "parent", 200)
    independent = _write_checkpoint(tmp_path / "independent", 200, policy_value=2.0)
    child = _write_checkpoint(tmp_path / "child", 201, resume=str(parent), policy_value=1.0)
    records = inventory_checkpoints([parent, independent, child])
    by_run = {Path(record["run_dir"]).name: record for record in records}

    assert by_run["parent"]["checkpoint_lineage_id"] == by_run["child"]["checkpoint_lineage_id"]
    assert by_run["parent"]["checkpoint_branch_id"] != by_run["child"]["checkpoint_branch_id"]
    assert by_run["independent"]["checkpoint_lineage_id"] != by_run["parent"]["checkpoint_lineage_id"]
    assert by_run["child"]["lineage_parent_resolved"] is True


def test_filename_payload_update_mismatch_is_visible(tmp_path) -> None:
    path = _write_checkpoint(tmp_path / "run", 7)
    wrong_name = path.with_name("update_0008.pt")
    path.rename(wrong_name)

    [record] = inventory_checkpoints([wrong_name])
    assert record["update_filename"] == 8
    assert record["update_payload"] == 7
    assert record["update_match"] is False
