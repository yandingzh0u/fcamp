from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch


class CheckpointMixin:
    def _target_validation_reached(self, metrics: dict[str, float]) -> bool:
        if self.cfg.target_validation_steps <= 0:
            return False
        target = float(self.cfg.target_validation_steps)
        max_episode_steps = float(getattr(self.cfg, "max_episode_steps", target))
        random_min = metrics.get("validation/steps_min")
        fixed_min = metrics.get("val_fixed/steps_min")
        if random_min is None:
            return False
        def reached(steps: float) -> bool:
            if target < max_episode_steps:
                return steps > target
            return steps >= target

        if fixed_min is not None:
            return reached(random_min) and reached(fixed_min)
        return reached(random_min)

    def _save_checkpoint(self, update_idx: int, metrics: dict[str, float], filename: str | None = None) -> None:
        if self.checkpoint_dir is None:
            return
        payload = {
            "update_idx": update_idx,
            "config": asdict(self.cfg),
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "metrics": metrics,
        }
        step_path = self.checkpoint_dir / (filename if filename is not None else f"update_{update_idx:04d}.pt")
        last_path = self.checkpoint_dir / "last.pt"
        torch.save(payload, step_path)
        if filename is None:
            torch.save(payload, last_path)
        print(f"[CHECKPOINT] saved {step_path}", flush=True)

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location=self.env.device)
        self.policy.load_state_dict(payload["policy"])
        if bool(getattr(self.cfg, "reset_optimizer_on_resume", False)):
            print(
                f"[CHECKPOINT] loaded policy from {checkpoint_path}; starting a fresh optimizer by request.",
                flush=True,
            )
        elif "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        else:
            raise KeyError(f"Checkpoint {checkpoint_path} has no optimizer state.")
        policy_lr = float(self.cfg.policy_lr)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = policy_lr
        if hasattr(self, "learning_rate"):
            self.learning_rate = policy_lr
        self.start_update = int(payload.get("update_idx", 0)) + 1
        print(
            f"[CHECKPOINT] loaded {checkpoint_path}, resuming from update {self.start_update}.",
            flush=True,
        )
