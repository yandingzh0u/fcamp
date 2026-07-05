from __future__ import annotations

import time
from pathlib import Path

import torch

from .checkpoint import Checkpointer
from .config import ExperimentConfig
from .validation_logging import log_validation_metrics
from .validation import run_validation_rollout, validation_max_steps
from env.mimic import G1MimicEnv


class CoreTrainer:
    def __init__(self, simulation_app, cfg: ExperimentConfig, algo_factory, checkpoint_dir: Path):
        self.simulation_app = simulation_app
        self.cfg = cfg
        self.env_cfg = cfg.environment
        self.algo_cfg = cfg.parameters
        self.train_cfg = cfg.training
        self.start_update = 1

        torch.manual_seed(cfg.training.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.training.seed)

        self.env = G1MimicEnv(cfg.environment, cfg.observation_group_size)
        self.checkpoint_dir = checkpoint_dir.resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.algo = algo_factory(self.algo_cfg, self.env, simulation_app)
        self.algo.build()
        self.checkpointer = Checkpointer(self)

        self.current_observation = self.algo.initial_reset()

        if cfg.training.resume:
            self.checkpointer.load(Path(cfg.training.resume))

    def train(self) -> None:
        tcfg = self.train_cfg
        self.algo.log_banner()
        print(f"[INFO] checkpoint_dir={self.checkpoint_dir}", flush=True)
        if tcfg.resume:
            print(f"[INFO] resumed_from={tcfg.resume}", flush=True)

        for update_idx in range(self.start_update, tcfg.max_updates + 1):
            if not self.simulation_app.is_running():
                break
            t0 = time.perf_counter()
            current_obs = self.algo.reset_for_update(update_idx)
            self.current_observation = current_obs
            rollout = self.algo.collect(current_obs)
            self.current_observation = rollout["next_observation"]
            collect_time = time.perf_counter() - t0

            metrics = self.algo.update(rollout, collect_time)


            del rollout
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if update_idx <= 3 or update_idx % tcfg.log_every == 0:
                self.algo.log(update_idx, tcfg.max_updates, metrics)

            if tcfg.validation_every > 0 and update_idx % tcfg.validation_every == 0:
                fixed_seed = tcfg.validation_fixed_seed if tcfg.validation_fixed_seed >= 0 else None
                print(
                    f"[VALIDATION_START] update={update_idx} "
                    f"max_steps={validation_max_steps(tcfg, self.env)} envs={self.env_cfg.num_envs} "
                    f"fixed_seed={fixed_seed if fixed_seed is not None else 'disabled'}",
                    flush=True,
                )
                vt0 = time.perf_counter()
                metrics.update(run_validation_rollout(self))
                dir_phase = tcfg.validation_directional_start_phase
                if dir_phase >= 0:
                    dir_metrics = run_validation_rollout(self, start_phase_override=dir_phase)
                    for key, value in dir_metrics.items():
                        metrics[key.replace("validation/", "validation_directional/")] = value
                if fixed_seed is not None:
                    fixed_metrics = run_validation_rollout(self, fixed_seed=fixed_seed)
                    for key, value in fixed_metrics.items():
                        metrics[key.replace("validation/", "val_fixed/")] = value
                metrics["timing/validation_s"] = time.perf_counter() - vt0
                print(f"[VALIDATION_DONE] update={update_idx} time={metrics['timing/validation_s']:.3f}s", flush=True)
                if update_idx <= 3 or update_idx % tcfg.log_every == 0:
                    log_validation_metrics(self.env, metrics)

            if (
                update_idx == tcfg.max_updates or (tcfg.save_every > 0 and update_idx % tcfg.save_every == 0)
            ):
                self.checkpointer.save(update_idx, metrics)
            if self.checkpointer.target_reached(metrics):
                self.checkpointer.save(update_idx, metrics, filename="success.pt")
                print(
                    f"[SUCCESS] validation reached {tcfg.target_validation_steps} steps; "
                    "saved success.pt",
                    flush=True,
                )
                break

        print("[INFO] Training finished.", flush=True)

    def validate_only(self) -> None:
        metrics = run_validation_rollout(self)
        log_validation_metrics(self.env, metrics)
