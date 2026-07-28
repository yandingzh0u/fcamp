from __future__ import annotations

import argparse
import hashlib
import os
import sys
import traceback
import json
import subprocess
import tarfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher
from engine.config import load_config

parser = argparse.ArgumentParser(description="Train a humanoid imitation method.")
parser.add_argument("--config", type=str, required=True, help="Path to the YAML config (the single source of defaults).")
parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override an existing config leaf, e.g. --set parameters.horizon=1. Repeatable.")
parser.add_argument("--run_name", type=str, default="", help="Run name; sets checkpoint/log dirs under runs/.")
parser.add_argument("--validate_only", action="store_true", default=False, help="Load a checkpoint and run one validation rollout.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

cfg = load_config(args_cli.config, args_cli.overrides)
args_cli.headless = True
args_cli.device = cfg.environment.device
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    from engine.trainer import CoreTrainer
    from envs.robots.g1 import G1_29DOF_ACTION_NAMES, G1_LOCAL_URDF_PATH
    from envs.tasks import resolve_task

    run_name = args_cli.run_name or f"{cfg.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = REPO_ROOT / "runs" / run_name
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train.log"

    # Persist the fully-resolved experiment identity before Isaac Sim starts.
    resolved = asdict(cfg)
    dataset_path = resolve_task(cfg.environment.task).motion_file.resolve()
    robot_path = G1_LOCAL_URDF_PATH.resolve()
    resolved["dataset_path"] = str(dataset_path)
    resolved["dataset_sha256"] = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    resolved["robot_asset_path"] = str(robot_path)
    resolved["robot_asset_sha256"] = hashlib.sha256(robot_path.read_bytes()).hexdigest()
    resolved["action_schema_sha256"] = hashlib.sha256(
        json.dumps(G1_29DOF_ACTION_NAMES, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    try:
        resolved["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        resolved["git_commit"] = "unknown"
    try:
        git_status = subprocess.check_output(
            ["git", "status", "--short"], cwd=REPO_ROOT, text=True
        )
    except Exception:
        git_status = "unavailable\n"
    (run_dir / "git_status.txt").write_text(git_status, encoding="utf-8")

    # A dirty/untracked research tree cannot be reproduced from git_commit.
    # Archive the actual executable source used by every run, excluding assets
    # and prior run outputs. This is intentionally small enough to keep beside
    # the resolved config and metrics.
    snapshot_path = run_dir / "source_snapshot.tar.gz"
    source_roots = (
        "components",
        "configs",
        "engine",
        "envs",
        "method",
        "models",
        "tests",
    )
    with tarfile.open(snapshot_path, "w:gz") as archive:
        for source_root in source_roots:
            root = REPO_ROOT / source_root
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                    continue
                archive.add(path, arcname=path.relative_to(REPO_ROOT))
        for path in sorted(REPO_ROOT.iterdir()):
            if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".toml"}:
                archive.add(path, arcname=path.name)
    snapshot_sha256 = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    resolved["git_dirty"] = bool(git_status.strip())
    resolved["source_snapshot"] = snapshot_path.name
    resolved["source_snapshot_sha256"] = snapshot_sha256
    config_identity = json.dumps(resolved, sort_keys=True, default=str).encode("utf-8")
    resolved["resolved_config_sha256"] = hashlib.sha256(config_identity).hexdigest()
    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump(resolved, handle, indent=2, sort_keys=True)


    class _Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                s.write(data)
        def flush(self):
            for s in self.streams:
                s.flush()

    log_handle = open(log_file, "a", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_handle)
    sys.stderr = _Tee(sys.__stderr__, log_handle)
    print(f"[RUN] dir={run_dir}", flush=True)
    print(f"[RUN] log_file={log_file}", flush=True)
    print(f"[RUN] method={cfg.method} task={cfg.environment.task} seed={cfg.training.seed}", flush=True)
    print(f"[RUN] resolved_config_sha256={resolved['resolved_config_sha256']}", flush=True)
    print(f"[RUN] source_snapshot_sha256={snapshot_sha256}", flush=True)

    trainer = CoreTrainer(simulation_app, cfg, run_dir / "checkpoints")
    try:
        if args_cli.validate_only:
            trainer.validate_only()
        else:
            trainer.train()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
