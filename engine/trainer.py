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
from method.amp import AMP


VALIDATION_PROTOCOL_VERSION = 6.0


class CoreTrainer:
    def __init__(self, simulation_app, cfg: ExperimentConfig, checkpoint_dir: Path):
        self.simulation_app = simulation_app
        self.cfg = cfg
        self.env_cfg = cfg.environment
        self.algo_cfg = cfg.parameters
        self.train_cfg = cfg.training
        self.start_update = 1
        self.env_transitions_total = 0
        self.train_wall_seconds_total = 0.0
        self._pre_training_warmup_ran = False

        torch.manual_seed(cfg.training.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.training.seed)

        self.env = G1MimicEnv(cfg.environment)
        self.checkpoint_dir = checkpoint_dir.resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_logger = MetricsLogger(self.checkpoint_dir.parent / "logs")

        self.algo = AMP(self.algo_cfg, self.env)
        self.algo.build()
        self.checkpointer = Checkpointer(self)

        # Include the actual initial RSI population in update-one diagnostics.
        # Starting the recorder only inside the update loop would make a
        # healthy first rollout misleadingly report reset count zero.
        self.env.begin_reset_phase_diagnostics()
        self._reset_phase_diagnostics_open = True
        self.current_observation = self.algo.initial_reset()

        if cfg.training.resume:
            self.checkpointer.load(Path(cfg.training.resume))

    def train(self) -> None:
        tcfg = self.train_cfg
        self.algo.log_banner()
        print(f"[INFO] checkpoint_dir={self.checkpoint_dir}", flush=True)
        if tcfg.resume:
            print(f"[INFO] resumed_from={tcfg.resume}", flush=True)

        warmup_metrics, warmup_transitions, warmup_seconds = self._run_pre_training_warmup()

        for update_idx in range(self.start_update, tcfg.max_updates + 1):
            if not self.simulation_app.is_running():
                break
            t0 = time.perf_counter()
            if not getattr(
                self,
                "_reset_phase_diagnostics_open",
                False,
            ):
                self.env.begin_reset_phase_diagnostics()
                self._reset_phase_diagnostics_open = True
            current_obs = self.algo.reset_for_update(update_idx)
            self.current_observation = current_obs
            rollout = self.algo.collect(current_obs)
            reset_metrics = self.env.finish_reset_phase_diagnostics()
            self._reset_phase_diagnostics_open = False
            self.current_observation = rollout["next_observation"]
            collect_time = time.perf_counter() - t0

            metrics = self.algo.update(rollout, collect_time)
            metrics.update(reset_metrics)
            formal_iteration_s = time.perf_counter() - t0
            formal_transitions = int(self.env_cfg.num_envs) * int(self.algo_cfg.rollout_env_steps)
            transitions_update = formal_transitions + warmup_transitions
            iteration_s = formal_iteration_s + warmup_seconds
            self.env_transitions_total += formal_transitions
            self.train_wall_seconds_total += formal_iteration_s
            metrics.update(warmup_metrics)
            metrics.update(
                {
                    "samples/env_transitions_update": float(transitions_update),
                    "samples/formal_env_transitions_update": float(formal_transitions),
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
            # Warm-up belongs only to the first fresh-run accounting interval.
            warmup_metrics = {}
            warmup_transitions = 0
            warmup_seconds = 0.0


            del rollout
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if update_idx % tcfg.log_every == 0:
                self.algo.log(update_idx, tcfg.max_updates, metrics)
                print(
                    "[TRAIN_RESET] "
                    f"count={metrics.get('train_reset/all/count', 0.0):.0f} "
                    "first_frame_bin_fraction="
                    f"{metrics.get('train_reset/all/first_frame_bin_fraction', 0.0):.4f} "
                    f"p50={metrics.get('train_reset/all/phase_p50', -1.0):.1f} "
                    f"p95={metrics.get('train_reset/all/phase_p95', -1.0):.1f}",
                    flush=True,
                )
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
                metrics["validation/protocol_version"] = VALIDATION_PROTOCOL_VERSION
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
        print("[INFO] Training finished.", flush=True)
        self.metrics_logger.close()

    def _run_pre_training_warmup(self) -> tuple[dict[str, float], int, float]:
        """Run the optional warm-up once on a fresh run and account its cost."""
        if (
            self._pre_training_warmup_ran
            or self.start_update != 1
            or bool(self.train_cfg.resume)
        ):
            return {}, 0, 0.0
        started = time.perf_counter()
        result = self.algo.pre_training_warmup(self.current_observation)
        elapsed = time.perf_counter() - started
        if not isinstance(result, tuple) or len(result) != 3:
            raise TypeError(
                "pre_training_warmup must return "
                "(observation, metrics, env_transition_count)"
            )
        observation, method_metrics, transition_count = result
        if not isinstance(method_metrics, dict):
            raise TypeError("pre_training_warmup metrics must be a dict")
        if isinstance(transition_count, bool) or not isinstance(transition_count, int):
            raise TypeError("pre_training_warmup transition count must be an int")
        if transition_count < 0:
            raise ValueError("pre_training_warmup transition count must be >= 0")

        self._pre_training_warmup_ran = True
        self.current_observation = observation
        if transition_count == 0 and not method_metrics:
            return {}, 0, 0.0

        self.env_transitions_total += transition_count
        self.train_wall_seconds_total += elapsed
        metrics = {f"warmup/{key}": value for key, value in method_metrics.items()}
        metrics.update(
            {
                "samples/warmup_env_transitions": float(transition_count),
                "timing/warmup_s": float(elapsed),
            }
        )
        print(
            f"[WARMUP] env_transitions={transition_count} time={elapsed:.3f}s",
            flush=True,
        )
        return metrics, transition_count, elapsed

    def validate_only(self) -> None:
        fixed_seed = (
            self.train_cfg.validation_fixed_seed
            if self.train_cfg.validation_fixed_seed >= 0
            else self.train_cfg.seed
        )
        metrics = run_validation_rollout(self, fixed_seed=fixed_seed)
        metrics["validation/fixed_seed"] = float(fixed_seed)
        metrics["validation/protocol_version"] = VALIDATION_PROTOCOL_VERSION
        metrics["validation/max_steps"] = float(validation_max_steps(self.train_cfg, self.env))
        log_validation_metrics(self.env, metrics)
        self.metrics_logger.write_validation_summary(self.start_update - 1, metrics)
        self.metrics_logger.write(self.start_update - 1, metrics)
        self.metrics_logger.close()
