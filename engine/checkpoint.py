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
            # Resume continuity: the adaptive KL-controlled LR must survive a resume, otherwise
            # the LR snaps back to the initial value on reload.
            "learning_rate": float(getattr(self, "learning_rate", self.cfg.policy_lr)),
        }
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
        try:
            self.policy.load_state_dict(payload["policy"])
        except RuntimeError as exc:
            # Checkpoints from before the current architecture are NOT resumable and must be
            # retrained from scratch. Several changes broke weight/layout compatibility:
            #   * the actor observation gained the multi-contact support term (OBS_DIM 163->167),
            #     and the contact features were inserted mid-vector (forcing policy_obs_dim back
            #     would mis-align every downstream feature);
            #   * the action parametrization changed to the anchored incremental-trajectory
            #     coefficient latent (different velocity_net output semantics and basis).
            raise RuntimeError(
                f"Failed to load policy weights from {checkpoint_path}. This is expected for "
                "checkpoints trained before the current architecture (support-contact observation "
                "AND anchored incremental-trajectory action parametrization). The observation "
                "layout and the action latent both changed, so old checkpoints are not resumable "
                f"and must be retrained from scratch. Original error: {exc}"
            ) from exc
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
