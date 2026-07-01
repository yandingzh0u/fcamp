from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch


class Checkpointer:
    def __init__(self, trainer):
        self.t = trainer

    def target_reached(self, metrics: dict[str, float]) -> bool:
        tcfg = self.t.train_cfg
        if tcfg.target_validation_steps <= 0:
            return False
        target = float(tcfg.target_validation_steps)
        max_episode_steps = float(self.t.env.max_episode_steps)
        random_min = metrics.get("validation/steps_min")
        fixed_min = metrics.get("val_fixed/steps_min")
        if random_min is None:
            return False

        def reached(steps: float) -> bool:
            return steps > target if target < max_episode_steps else steps >= target

        if fixed_min is not None:
            return reached(random_min) and reached(fixed_min)
        return reached(random_min)

    def save(self, update_idx: int, metrics: dict[str, float], filename: str | None = None) -> None:
        t = self.t
        payload = {
            "update_idx": update_idx,
            "config": asdict(t.cfg),
            "policy": t.algo.policy.state_dict(),
            "optimizer": t.algo.optimizer.state_dict(),
            "metrics": metrics,
            "algo_state": t.algo.extra_checkpoint_state(),
        }
        payload["adaptive_sampler_state"] = t.env.adaptive_sampler.state_dict()
        payload["torch_rng_state"] = torch.random.get_rng_state()
        if torch.device(t.env.device).type == "cuda":
            payload["cuda_rng_state"] = torch.cuda.get_rng_state(t.env.device)
        step_path = t.checkpoint_dir / (filename if filename is not None else f"update_{update_idx:04d}.pt")
        torch.save(payload, step_path)
        if filename is None:
            torch.save(payload, t.checkpoint_dir / "last.pt")
        print(f"[CHECKPOINT] saved {step_path}", flush=True)

    def load(self, checkpoint_path: Path) -> None:
        t = self.t
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location=t.env.device)
        t.algo.policy.load_state_dict(payload["policy"])
        reset_optimizer = bool(t.train_cfg.reset_optimizer_on_resume)
        if reset_optimizer:
            print(f"[CHECKPOINT] loaded policy from {checkpoint_path}; fresh optimizer by request.", flush=True)
        elif "optimizer" in payload:
            t.algo.optimizer.load_state_dict(payload["optimizer"])
        else:
            raise KeyError(f"Checkpoint {checkpoint_path} has no optimizer state.")

        t.algo.load_extra_checkpoint_state(payload.get("algo_state", {}), reset_optimizer=reset_optimizer)


        reset_sampler = bool(t.train_cfg.reset_sampler_on_resume)
        if reset_sampler:
            t.env.adaptive_sampler.init_buffers()
            t.env._failure_recorded.zero_()
            print("[CHECKPOINT] adaptive sampler reset by request; starting fresh.", flush=True)
        elif t.env.adaptive_sampler.load_state_dict(payload.get("adaptive_sampler_state")):
            print("[CHECKPOINT] restored adaptive sampler state.", flush=True)
        else:
            print("[CHECKPOINT] adaptive sampler state absent/incompatible; starting fresh.", flush=True)

        try:
            if "torch_rng_state" in payload:
                torch.random.set_rng_state(payload["torch_rng_state"].cpu())
            if "cuda_rng_state" in payload and torch.cuda.is_available():
                torch.cuda.set_rng_state(payload["cuda_rng_state"].cpu(), t.env.device)
        except Exception as exc:
            print(f"[CHECKPOINT] WARN: could not restore RNG state: {exc}", flush=True)

        t.start_update = int(payload.get("update_idx", 0)) + 1
        print(f"[CHECKPOINT] loaded {checkpoint_path}, resuming from update {t.start_update}.", flush=True)
