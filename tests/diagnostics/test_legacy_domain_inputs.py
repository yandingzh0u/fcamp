import json
from pathlib import Path

from diagnostics.common.manifest import sha256_file


ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / "runs" / "discovery_legacy_inputs"


def test_legacy_domains_match_largebox_motion_robot_and_action_contract() -> None:
    manifest = json.loads((LEGACY / "manifest.json").read_text(encoding="utf-8"))
    identity = manifest["shared_task_identity"]
    for domain_name in ("A_amp", "A_mix"):
        domain = manifest["domains"][domain_name]
        config_path = LEGACY / Path(domain["checkpoint"]).parents[1] / "resolved_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["environment"]["task"] == identity["task_name"]
        assert config["dataset_sha256"] == identity["motion_sha256"]
        assert config["robot_asset_sha256"] == identity["robot_asset_sha256"]
        assert config["action_schema_sha256"] == identity["action_schema_sha256"]


def test_dirty_legacy_domains_keep_the_exact_source_snapshots() -> None:
    manifest = json.loads((LEGACY / "manifest.json").read_text(encoding="utf-8"))
    for domain in manifest["domains"].values():
        assert domain["git_dirty"] is True
        snapshot = LEGACY / domain["source_snapshot"]
        assert sha256_file(snapshot) == domain["source_snapshot_sha256"]
        assert (LEGACY / domain["checkpoint"]).is_file()
