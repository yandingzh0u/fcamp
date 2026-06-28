"""Algorithm-agnostic training loop.

Owns: the env, the per-update schedule (collect -> update -> log -> validate -> checkpoint),
and resume. Knows nothing about flow matching, GRPO groups, chunks, or critics — all of that
lives behind the Algorithm interface.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from .checkpoint import Checkpointer
from .config import Config
from .env_factory_adapter import build_env
from .logging import log_validation_metrics
from .validation import run_validation_rollout, validation_max_steps


class CoreTrainer:
    def __init__(self, simulation_app, cfg: Config, algo_factory):
        self.simulation_app = simulation_app
        self.cfg = cfg
        self.env_cfg = cfg.env
        self.algo_cfg = cfg.algo
        self.train_cfg = cfg.train
        self.start_update = 1

        torch.manual_seed(cfg.train.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.train.seed)

        self.env = build_env(cfg)
        # The env resolves max_episode_steps<=0 to the motion clip length; mirror it back so
        # reward projection / validation horizon use the real cap.
        resolved = int(getattr(self.env.task_cfg, "max_episode_steps", cfg.env.max_episode_steps))
        if resolved > 0:
            cfg.env.max_episode_steps = resolved

        self.checkpoint_dir = (
            Path(cfg.train.checkpoint_dir).expanduser().resolve() if cfg.train.checkpoint_dir else None
        )
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.algo = algo_factory(self.algo_cfg, self.env, simulation_app)
        self.algo.build()
        self.checkpointer = Checkpointer(self)

        self.current_observation = self.algo.initial_reset()

        if cfg.train.resume:
            self.checkpointer.load(Path(cfg.train.resume).expanduser().resolve())

    def train(self) -> None:
        tcfg = self.train_cfg
        self.algo.log_banner()
        if self.checkpoint_dir is not None:
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
            self.current_observation = rollout.get("next_observation", current_obs)
            collect_time = time.perf_counter() - t0

            metrics = self.algo.update(rollout, collect_time)

            # Free the rollout BEFORE the next iteration's collect() runs. Otherwise the old
            # rollout's large MC buffers (cfm_eps / x1_pred, ~1.4GiB at 8192x48x16) stay alive
            # while the next collect() allocates a fresh rollout + CFM activations, doubling peak
            # memory and OOM-ing on the 2nd collect. empty_cache() returns the freed blocks to the
            # CUDA allocator so Isaac/PhysX (non-PyTorch) can reuse that VRAM.
            del rollout
            torch.cuda.empty_cache()

            if update_idx % tcfg.log_every == 0:
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
                if fixed_seed is not None:
                    fixed_metrics = run_validation_rollout(self, fixed_seed=fixed_seed)
                    for key, value in fixed_metrics.items():
                        metrics[key.replace("validation/", "val_fixed/")] = value
                metrics["timing/validation_s"] = time.perf_counter() - vt0
                print(f"[VALIDATION_DONE] update={update_idx} time={metrics['timing/validation_s']:.3f}s", flush=True)
                if update_idx % tcfg.log_every == 0:
                    log_validation_metrics(self.env, metrics)

            if self.checkpoint_dir is not None and (
                update_idx == tcfg.max_updates or (tcfg.save_every > 0 and update_idx % tcfg.save_every == 0)
            ):
                self.checkpointer.save(update_idx, metrics)
            if self.checkpointer.target_reached(metrics):
                if self.checkpoint_dir is not None:
                    self.checkpointer.save(update_idx, metrics, filename=tcfg.success_checkpoint_name)
                print(
                    f"[SUCCESS] validation reached {tcfg.target_validation_steps} steps; "
                    f"saved {tcfg.success_checkpoint_name}",
                    flush=True,
                )
                break

        print("[INFO] Training finished.", flush=True)

    def validate_only(self) -> None:
        metrics = run_validation_rollout(self)
        log_validation_metrics(self.env, metrics)
