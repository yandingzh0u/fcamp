from __future__ import annotations

import json

import pytest

from diagnostics.common.manifest import (
    PASS,
    complete_manifest_errors,
    create_source_snapshot,
    write_json_exclusive,
)


def _complete_manifest() -> dict:
    digest = "a" * 64
    return {
        "status": PASS,
        "git_commit": "b" * 40,
        "source_snapshot_sha256": digest,
        "resolved_config_sha256": digest,
        "checkpoint_sha256": digest,
        "checkpoint_update": 200,
        "checkpoint_lineage_id": digest,
        "robot_asset_sha256": digest,
        "motion_sha256": digest,
        "action_schema_sha256": digest,
        "task_name": "largebox_plane",
    }


def test_manifest_complete_rejects_unknown_or_missing_identity() -> None:
    manifest = _complete_manifest()
    assert complete_manifest_errors(manifest) == []

    manifest["git_commit"] = "unknown"
    manifest["motion_sha256"] = "missing"
    errors = complete_manifest_errors(manifest)
    assert any("git_commit" in error for error in errors)
    assert any("motion_sha256" in error for error in errors)


def test_result_writes_are_exclusive_and_preserve_first_result(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    write_json_exclusive(path, _complete_manifest())
    original = path.read_bytes()

    with pytest.raises(FileExistsError):
        write_json_exclusive(path, {**_complete_manifest(), "checkpoint_update": 201})

    assert path.read_bytes() == original
    assert json.loads(path.read_text(encoding="utf-8"))["checkpoint_update"] == 200


def test_source_snapshot_includes_dirty_and_untracked_source_deterministically(tmp_path) -> None:
    repo = tmp_path / "repo"
    (repo / "components").mkdir(parents=True)
    (repo / "components" / "tracked.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "tools").mkdir()
    (repo / "tools" / "untracked_probe.py").write_text("VALUE = 2\n", encoding="utf-8")
    first = create_source_snapshot(repo, tmp_path / "first.tar.gz")
    second = create_source_snapshot(repo, tmp_path / "second.tar.gz")

    assert first["file_count"] == 2
    assert first["sha256"] == second["sha256"]
    assert first["members_sha256"] == second["members_sha256"]
