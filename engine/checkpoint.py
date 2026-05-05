from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch


class CheckpointMixin:
    def _target_validation_reached(self, metrics: dict[str, float]) -> bool:
        if self.cfg.target_validation_steps <= 0:
            return False
        target = float(self.cfg.target_validation_steps)
        random_min = metrics.get("validation/steps_min")
        fixed_min = metrics.get("val_fixed/steps_min")
        if random_min is None or fixed_min is None:
            return False
        return random_min >= target and fixed_min >= target

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
        load_result = self.policy.load_state_dict(payload["policy"], strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                "[CHECKPOINT] loaded with compatible key filtering: "
                f"missing={load_result.missing_keys} unexpected={load_result.unexpected_keys}",
                flush=True,
            )
        if "optimizer" in payload:
            try:
                self.optimizer.load_state_dict(payload["optimizer"])
            except ValueError as exc:
                print(
                    f"[CHECKPOINT] optimizer state is incompatible after simplification ({exc}); "
                    "starting a fresh optimizer.",
                    flush=True,
                )
        else:
            print(
                f"[CHECKPOINT] {checkpoint_path} has no optimizer state; starting a fresh optimizer.",
                flush=True,
            )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.cfg.lr
        self.start_update = int(payload.get("update_idx", 0)) + 1
        print(
            f"[CHECKPOINT] loaded {checkpoint_path}, resuming from update {self.start_update}.",
            flush=True,
        )
