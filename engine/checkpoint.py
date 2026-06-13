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
            # Resume continuity: the adaptive KL-controlled LR and the adaptive phase sampler
            # state must survive a resume, otherwise the curriculum restarts from scratch and
            # the LR snaps back to the initial value.
            "learning_rate": float(getattr(self, "learning_rate", self.cfg.policy_lr)),
        }
        if hasattr(self.env, "bin_failed_count"):
            payload["adaptive_bin_failed_count"] = self.env.bin_failed_count.detach().cpu()
        if hasattr(self.env, "bin_exposure_count"):
            payload["adaptive_bin_exposure_count"] = self.env.bin_exposure_count.detach().cpu()
        try:
            payload["torch_rng_state"] = torch.random.get_rng_state()
            if torch.cuda.is_available():
                payload["cuda_rng_state"] = torch.cuda.get_rng_state(self.env.device)
        except Exception:
            pass
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
        reset_optimizer = bool(getattr(self.cfg, "reset_optimizer_on_resume", False))
        if reset_optimizer:
            print(
                f"[CHECKPOINT] loaded policy from {checkpoint_path}; starting a fresh optimizer by request.",
                flush=True,
            )
        elif "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        else:
            raise KeyError(f"Checkpoint {checkpoint_path} has no optimizer state.")

        # Learning rate: only force back to the configured initial LR when explicitly resetting
        # the optimizer. Otherwise restore the adaptive (KL-controlled) LR so the resume is
        # seamless instead of slamming the LR ~25x higher and shocking the policy.
        if reset_optimizer:
            resume_lr = float(self.cfg.policy_lr)
        else:
            resume_lr = float(payload.get("learning_rate", self.cfg.policy_lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = resume_lr
        if hasattr(self, "learning_rate"):
            self.learning_rate = resume_lr

        # Restore adaptive phase-sampler history so the failure-weighted curriculum continues
        # instead of restarting from a near-uniform cold state.
        has_rate_state = "adaptive_bin_failed_count" in payload and "adaptive_bin_exposure_count" in payload
        if not reset_optimizer and has_rate_state and hasattr(self.env, "bin_exposure_count"):
            saved_bins = payload["adaptive_bin_failed_count"].to(self.env.bin_failed_count)
            saved_exposure = payload["adaptive_bin_exposure_count"].to(self.env.bin_exposure_count)
            if saved_bins.shape == self.env.bin_failed_count.shape and saved_exposure.shape == self.env.bin_exposure_count.shape:
                self.env.bin_failed_count.copy_(saved_bins)
                self.env.bin_exposure_count.copy_(saved_exposure)
                print("[CHECKPOINT] restored adaptive sampler failure/exposure state.", flush=True)
            else:
                print(
                    f"[CHECKPOINT] adaptive bin shape mismatch (saved {tuple(saved_bins.shape)} vs "
                    f"env {tuple(self.env.bin_failed_count.shape)}); keeping fresh sampler state.",
                    flush=True,
                )
        elif not reset_optimizer and "adaptive_bin_failed_count" in payload and hasattr(self.env, "bin_exposure_count"):
            self.env.bin_failed_count.zero_()
            self.env.bin_exposure_count.zero_()
            print(
                "[CHECKPOINT] legacy adaptive sampler state has no exposure counts; "
                "starting the failure-rate sampler fresh.",
                flush=True,
            )

        # Restore RNG streams for reproducible continuation.
        if not reset_optimizer:
            try:
                if "torch_rng_state" in payload:
                    torch.random.set_rng_state(payload["torch_rng_state"].cpu())
                if "cuda_rng_state" in payload and torch.cuda.is_available():
                    torch.cuda.set_rng_state(payload["cuda_rng_state"].cpu(), self.env.device)
            except Exception as exc:
                print(f"[CHECKPOINT] WARN: could not restore RNG state: {exc}", flush=True)

        self.start_update = int(payload.get("update_idx", 0)) + 1
        print(
            f"[CHECKPOINT] loaded {checkpoint_path}, resuming from update {self.start_update} "
            f"(lr={resume_lr:.3e}).",
            flush=True,
        )
