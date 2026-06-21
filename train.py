"""Unified training entry point.

    python train.py --config configs/mixgrpo_crawl.yaml [--set algo.horizon=1 ...] \
        [--run_name NAME] [--validate_only]

All defaults live in the YAML. `--set a.b=c` overrides any leaf. The algorithm is selected by
`algo_name` in the config (or `--set algo_name=ppo` later).
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train a humanoid imitation policy (multi-algorithm).")
parser.add_argument("--config", type=str, required=True, help="Path to the YAML config (the single source of defaults).")
parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override a config leaf, e.g. --set algo.horizon=1. Repeatable.")
parser.add_argument("--run_name", type=str, default="", help="Run name; sets checkpoint/log dirs under runs/.")
parser.add_argument("--validate_only", action="store_true", default=False, help="Load a checkpoint and run one validation rollout.")
parser.add_argument("--log_file", type=str, default="", help="Optional explicit stdout/stderr log path.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main() -> None:
    from core.config import load_config
    from core.trainer import CoreTrainer
    from algorithms import make_algorithm

    cfg = load_config(args_cli.config, args_cli.overrides)

    run_name = args_cli.run_name or f"{cfg.algo_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = REPO_ROOT / "runs" / run_name
    if not cfg.train.checkpoint_dir:
        cfg.train.checkpoint_dir = str(run_dir / "checkpoints")
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(args_cli.log_file) if args_cli.log_file else (log_dir / "train.log")

    # Tee stdout/stderr to the run log.
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
    print(f"[INFO] run_dir={run_dir}", flush=True)
    print(f"[INFO] log_file={log_file}", flush=True)
    print(f"[INFO] algo={cfg.algo_name}", flush=True)

    trainer = CoreTrainer(simulation_app, cfg, make_algorithm(cfg.algo_name))
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
