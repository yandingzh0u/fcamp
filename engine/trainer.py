from __future__ import annotations

import time
from pathlib import Path

import torch

from .checkpoint import Checkpointer
from .config import ExperimentConfig
from .validation_logging import log_validation_metrics
from .validation import run_validation_rollout, validation_max_steps
from .metrics_logger import MetricsLogger
from envs.g1_mimic import G1MimicEnv


class CoreTrainer:
    def __init__(self, simulation_app, cfg: ExperimentConfig, algo_factory, checkpoint_dir: Path):
        self.simulation_app = simulation_app
        self.cfg = cfg
        self.env_cfg = cfg.environment
        self.algo_cfg = cfg.parameters
        self.train_cfg = cfg.training
        self.start_update = 1
        self.env_transitions_total = 0
        self.train_wall_seconds_total = 0.0

        torch.manual_seed(cfg.training.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.training.seed)

        self.env = G1MimicEnv(cfg.environment, cfg.observation_group_size)
        self.checkpoint_dir = checkpoint_dir.resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_logger = MetricsLogger(self.checkpoint_dir.parent / "logs")

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
            iteration_s = time.perf_counter() - t0
            transitions_update = int(self.env_cfg.num_envs) * int(self.algo_cfg.rollout_env_steps)
            self.env_transitions_total += transitions_update
            self.train_wall_seconds_total += iteration_s
            metrics.update(
                {
                    "samples/env_transitions_update": float(transitions_update),
                    "samples/env_transitions_total": float(self.env_transitions_total),
                    "progress/control_seconds_per_env": float(
                        self.env_transitions_total * self.env_cfg.sim_dt / self.env_cfg.num_envs
                    ),
                    "perf/iteration_s": float(iteration_s),
                    "perf/train_wall_s_total": float(self.train_wall_seconds_total),
                    "perf/env_transitions_per_s": float(transitions_update / max(iteration_s, 1.0e-9)),
                    "health/parameters_finite": float(metrics.get("system/parameters_finite", 1.0)),
                }
            )


            del rollout
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if update_idx % tcfg.log_every == 0:
                self.algo.log(update_idx, tcfg.max_updates, metrics)
                print(
                    f"[PROGRESS] env_steps={self.env_transitions_total} "
                    f"iteration={iteration_s:.3f}s throughput={metrics['perf/env_transitions_per_s']:.1f}",
                    flush=True,
                )

            if tcfg.validation_every > 0 and update_idx % tcfg.validation_every == 0:
                fixed_seed = (
                    tcfg.validation_fixed_seed
                    if tcfg.validation_fixed_seed >= 0
                    else tcfg.seed
                )
                val_max_steps = validation_max_steps(tcfg, self.env)
                print(
                    f"[VALIDATION_START] update={update_idx} "
                    f"max_steps={val_max_steps} envs={self.env_cfg.num_envs} "
                    f"fixed_seed={fixed_seed}",
                    flush=True,
                )
                vt0 = time.perf_counter()
                metrics["validation/fixed_seed"] = float(fixed_seed)
                metrics["validation/protocol_version"] = 3.0
                metrics["validation/max_steps"] = float(val_max_steps)
                metrics.update(run_validation_rollout(self, fixed_seed=fixed_seed))
                dir_phase = tcfg.validation_directional_start_phase
                if dir_phase >= 0:
                    dir_metrics = run_validation_rollout(
                        self,
                        fixed_seed=fixed_seed,
                        start_phase_override=dir_phase,
                    )
                    for key, value in dir_metrics.items():
                        metrics[key.replace("validation/", "validation_directional/")] = value
                metrics["timing/validation_s"] = time.perf_counter() - vt0
                print(f"[VALIDATION_DONE] update={update_idx} time={metrics['timing/validation_s']:.3f}s", flush=True)
                if update_idx % tcfg.log_every == 0:
                    log_validation_metrics(self.env, metrics)
                self.metrics_logger.write_validation_summary(update_idx, metrics)

            # Structured metrics are written every iteration, after optional
            # validation has appended its metrics.
            self.metrics_logger.write(update_idx, metrics)

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
            if self._official_output_reset_due(update_idx):
                self.current_observation = self.algo.initial_reset()
                print(
                    f"[OFFICIAL_RESET] update={update_idx} every={tcfg.official_reset_every}",
                    flush=True,
                )

        print("[INFO] Training finished.", flush=True)
        self.metrics_logger.close()

    def _official_output_reset_due(self, update_idx: int) -> bool:
        every = int(self.train_cfg.official_reset_every)
        if every <= 0:
            return False
        return update_idx >= 1 and (update_idx - 1) % every == 0

    def validate_only(self) -> None:
        fixed_seed = (
            self.train_cfg.validation_fixed_seed
            if self.train_cfg.validation_fixed_seed >= 0
            else self.train_cfg.seed
        )
        metrics = run_validation_rollout(self, fixed_seed=fixed_seed)
        metrics["validation/fixed_seed"] = float(fixed_seed)
        metrics["validation/protocol_version"] = 3.0
        metrics["validation/max_steps"] = float(validation_max_steps(self.train_cfg, self.env))
        log_validation_metrics(self.env, metrics)
        self.metrics_logger.write_validation_summary(self.start_update - 1, metrics)
        self.metrics_logger.write(self.start_update - 1, metrics)
        self.metrics_logger.close()
