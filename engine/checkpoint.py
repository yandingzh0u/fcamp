from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import os
import shutil

import torch


_RESUME_ENV_KEYS = (
    "platform_profile",
    "task",
    "sim_dt",
    "decimation",
    "fix_root_link",
    "max_episode_steps",
    "motion_start_phase",
    "motion_end_phase",
    "reset_phase_sampling",
    "rsi_keyframe_count",
    "startup_randomization",
    "reset_noise",
    "interval_pushes",
    "observation_noise",
    "adaptive_motion_sampling",
    "adaptive_num_bins",
    "adaptive_alpha",
    "adaptive_predecessor_ratio",
    "adaptive_predecessor_lookback_bins",
    "adaptive_uniform_ratio",
    "adaptive_kernel_size",
    "adaptive_lambda",
    "termination_mode",
    "terminate_on_motion_end",
    "motion_reference_mode",
    "root_velocity_mode",
    "policy_observation_mode",
    "motion_end_behavior",
    "action_rate_weight",
    "physics_material_combine_mode",
    "contact_sensor_update_period",
)

_RESUME_ENV_DEFAULTS = {
    "adaptive_uniform_ratio": 0.1,
    "adaptive_kernel_size": 1,
    "adaptive_lambda": 0.8,
    "policy_observation_mode": "tracking",
    "motion_end_behavior": "hold_last",
    "physics_material_combine_mode": "average",
    "contact_sensor_update_period": "control",
}


def _resume_signature(config: dict) -> dict:
    environment = config.get("environment", {})
    return {
        "method": config.get("method", config.get("algorithm")),
        "environment": {
            key: environment.get(key, _RESUME_ENV_DEFAULTS.get(key))
            for key in _RESUME_ENV_KEYS
        },
        "parameters": config.get("parameters"),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _platform_identity(task_name: str) -> dict[str, str]:
    from envs.robots.g1 import G1_29DOF_ACTION_NAMES, G1_LOCAL_URDF_PATH
    from envs.tasks import resolve_task

    dataset_path = resolve_task(task_name).motion_file.resolve()
    robot_path = G1_LOCAL_URDF_PATH.resolve()
    action_schema = json.dumps(
        G1_29DOF_ACTION_NAMES,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "dataset_sha256": _file_sha256(dataset_path),
        "robot_asset_sha256": _file_sha256(robot_path),
        "action_schema_sha256": hashlib.sha256(action_schema).hexdigest(),
    }


class Checkpointer:
    def __init__(self, trainer):
        self.t = trainer
        task_name = getattr(getattr(trainer, "env_cfg", None), "task", None)
        self.platform_identity = (
            _platform_identity(str(task_name)) if task_name is not None else None
        )

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
            "env_transitions_total": int(t.env_transitions_total),
            "train_wall_seconds_total": float(t.train_wall_seconds_total),
            "platform_identity": self.platform_identity,
        }
        payload["adaptive_sampler_state"] = t.env.adaptive_sampler.state_dict()
        payload["torch_rng_state"] = torch.random.get_rng_state()
        if torch.device(t.env.device).type == "cuda":
            payload["cuda_rng_state"] = torch.cuda.get_rng_state(t.env.device)
        step_path = t.checkpoint_dir / (filename if filename is not None else f"update_{update_idx:04d}.pt")
        torch.save(payload, step_path)
        if filename is None:
            # Replay-complete FC-AMP checkpoints can exceed 1 GiB. Keep last.pt
            # as a hard link instead of serializing the same payload twice.
            last_path = t.checkpoint_dir / "last.pt"
            last_path.unlink(missing_ok=True)
            try:
                os.link(step_path, last_path)
            except OSError:
                shutil.copy2(step_path, last_path)
        print(f"[CHECKPOINT] saved {step_path}", flush=True)

    def load(self, checkpoint_path: Path) -> None:
        t = self.t
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        # Load through CPU so a large discriminator replay sidecar does not
        # transiently consume GPU memory before being copied back to its CPU ring.
        payload = torch.load(checkpoint_path, map_location="cpu")
        saved_config = payload.get("config")
        if saved_config is not None:
            current_signature = _resume_signature(asdict(t.cfg))
            saved_signature = _resume_signature(saved_config)
            if saved_signature != current_signature:
                raise ValueError(
                    "Checkpoint training semantics do not match the current config; "
                    "start a fresh run instead of crossing dataset/platform/recipe profiles."
                )
        saved_platform_identity = payload.get("platform_identity")
        if saved_platform_identity is not None and self.platform_identity is not None:
            if saved_platform_identity != self.platform_identity:
                raise ValueError(
                    "Checkpoint dataset/robot/action schema does not match the current platform."
                )
        else:
            print(
                "[CHECKPOINT] WARN: legacy checkpoint has no dataset/robot/action-schema hashes.",
                flush=True,
            )
        preflight = getattr(t.algo, "validate_checkpoint_payload", None)
        if callable(preflight):
            preflight(payload)
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

        reset_after_resume = getattr(t.algo, "reset_after_resume", None)
        if callable(reset_after_resume):
            resumed_observation = reset_after_resume()
            if resumed_observation is not None:
                t.current_observation = resumed_observation

        t.start_update = int(payload.get("update_idx", 0)) + 1
        completed_updates = t.start_update - 1
        if hasattr(t, "env_cfg") and hasattr(t, "algo_cfg"):
            transitions_per_update = int(t.env_cfg.num_envs) * int(t.algo_cfg.rollout_env_steps)
            fallback_transitions = completed_updates * transitions_per_update
            t.env_transitions_total = int(payload.get("env_transitions_total", fallback_transitions))
            t.train_wall_seconds_total = float(payload.get("train_wall_seconds_total", 0.0))
        print(f"[CHECKPOINT] loaded {checkpoint_path}, resuming from update {t.start_update}.", flush=True)
