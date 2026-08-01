from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import os
import shutil

import torch

from components.rollout.fixed_reward_contract import (
    FIXED_REWARD_CHECKPOINT_CONTRACT,
)

from .config import config_from_checkpoint_dict


FIXED_REWARD_SCHEMA_VERSION = int(
    FIXED_REWARD_CHECKPOINT_CONTRACT["fixed_reward_schema_version"]
)
_CHECKPOINT_TOP_LEVEL_KEYS = frozenset(
    {
        "update_idx",
        "config",
        "policy",
        "optimizer",
        "metrics",
        "algo_state",
        "env_transitions_total",
        "train_wall_seconds_total",
        "platform_identity",
        "adaptive_sampler_state",
        "torch_rng_state",
        "cuda_rng_state",
    }
)
_CHECKPOINT_REQUIRED_KEYS = _CHECKPOINT_TOP_LEVEL_KEYS - {"cuda_rng_state"}
_POLICY_MODULE_NAMES = frozenset(
    {
        "actor",
        "actor_obs_normalizer",
        "critic",
        "critic_obs_normalizer",
    }
)
_ALGO_STATE_KEYS = frozenset(
    {
        "critic_optimizer",
        "learning_rate",
        "critic_learning_rate",
        "stream_ids",
        "phase0_stream_count",
        "phase0_stream_fraction",
        "phase0_attempt_tracker",
    }
) | frozenset(FIXED_REWARD_CHECKPOINT_CONTRACT)
_REMOVED_STATE_KEY_MARKERS = (
    "amp_",
    "disc_",
    "discriminator",
    "mixed_reward",
    "channel_",
    "history",
    "replay",
    "mmd",
)


def _removed_state_key_paths(
    value,
    prefix: str = "",
) -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            lowered = key_text.lower()
            if any(marker in lowered for marker in _REMOVED_STATE_KEY_MARKERS):
                found.append(path)
            found.extend(_removed_state_key_paths(nested, path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            path = f"{prefix}[{index}]"
            found.extend(_removed_state_key_paths(nested, path))
    return found


def audit_fixed_reward_checkpoint_payload(payload: dict) -> None:
    """Reject schema drift and removed subsystem state before restoration."""

    if not isinstance(payload, dict):
        raise ValueError("Checkpoint payload must be a mapping.")
    config = payload.get("config")
    algo_state = payload.get("algo_state")
    is_new_schema = (
        isinstance(config, dict)
        and config.get("method") == "fixed_reward"
    ) or (
        isinstance(algo_state, dict)
        and "fixed_reward_schema_version" in algo_state
    )
    if not is_new_schema:
        # Legacy payloads are rejected by config/method preflight. Keeping this
        # audit scoped to the new schema preserves a clear legacy error.
        return

    # Check the complete algorithm contract before inspecting or restoring
    # any policy or optimizer state.
    if not isinstance(algo_state, Mapping):
        raise ValueError("fixed_reward checkpoint algo_state must be a mapping.")
    schema_version = algo_state.get("fixed_reward_schema_version")
    if (
        type(schema_version) is not int
        or schema_version != FIXED_REWARD_SCHEMA_VERSION
    ):
        raise ValueError(
            "fixed_reward checkpoint schema version mismatch: "
            f"expected={FIXED_REWARD_SCHEMA_VERSION}, "
            f"actual={schema_version!r}"
        )
    for key, expected in FIXED_REWARD_CHECKPOINT_CONTRACT.items():
        actual = algo_state.get(key)
        if actual != expected:
            raise ValueError(
                "fixed_reward checkpoint semantic contract mismatch: "
                f"{key} expected={expected!r}, actual={actual!r}"
            )

    top_level_keys = set(payload)
    missing = _CHECKPOINT_REQUIRED_KEYS - top_level_keys
    unknown = top_level_keys - _CHECKPOINT_TOP_LEVEL_KEYS
    if missing or unknown:
        raise ValueError(
            "fixed_reward checkpoint top-level schema mismatch: "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )

    removed_paths = _removed_state_key_paths(payload)
    if removed_paths:
        raise ValueError(
            "fixed_reward checkpoint contains removed subsystem state: "
            f"{removed_paths}"
        )

    policy_state = payload.get("policy")
    if not isinstance(policy_state, Mapping):
        raise ValueError("fixed_reward checkpoint policy must be a state dict.")
    policy_modules = {
        str(key).split(".", 1)[0]
        for key in policy_state
    }
    if policy_modules != _POLICY_MODULE_NAMES:
        raise ValueError(
            "fixed_reward checkpoint policy modules mismatch: "
            f"expected={sorted(_POLICY_MODULE_NAMES)}, "
            f"actual={sorted(policy_modules)}"
        )

    if set(algo_state) != _ALGO_STATE_KEYS:
        raise ValueError(
            "fixed_reward checkpoint algo_state schema mismatch: "
            f"expected={sorted(_ALGO_STATE_KEYS)}, "
            f"actual={sorted(algo_state)}"
        )


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
    "adaptive_num_bins",
    "adaptive_alpha",
    "adaptive_predecessor_ratio",
    "adaptive_predecessor_lookback_bins",
    "root_velocity_mode",
    "action_rate_weight",
)

def _resume_signature(config: dict) -> dict:
    environment = config.get("environment", {})
    return {
        "method": config.get("method"),
        "environment": {
            key: environment.get(key) for key in _RESUME_ENV_KEYS
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
        directional_min = metrics.get("validation_directional/steps_min")
        if random_min is None:
            return False

        def reached(steps: float) -> bool:
            return steps > target if target < max_episode_steps else steps >= target

        if directional_min is not None:
            return reached(random_min) and reached(directional_min)
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
        audit_fixed_reward_checkpoint_payload(payload)
        step_path = t.checkpoint_dir / (filename if filename is not None else f"update_{update_idx:04d}.pt")
        torch.save(payload, step_path)
        if filename is None:
            # Keep last.pt as a hard link instead of serializing the same
            # training state twice.
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
        # Load through CPU so checkpoint restoration has predictable GPU memory
        # use before individual modules are copied to their target device.
        payload = torch.load(checkpoint_path, map_location="cpu")
        audit_fixed_reward_checkpoint_payload(payload)
        saved_config = payload.get("config")
        if saved_config is not None:
            current_signature = _resume_signature(asdict(t.cfg))
            saved_signature = _resume_signature(
                asdict(config_from_checkpoint_dict(saved_config))
            )
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

        t.current_observation = t.algo.reset_after_resume()

        t.start_update = int(payload.get("update_idx", 0)) + 1
        completed_updates = t.start_update - 1
        if hasattr(t, "env_cfg") and hasattr(t, "algo_cfg"):
            transitions_per_update = int(t.env_cfg.num_envs) * int(t.algo_cfg.rollout_env_steps)
            fallback_transitions = completed_updates * transitions_per_update
            t.env_transitions_total = int(payload.get("env_transitions_total", fallback_transitions))
            t.train_wall_seconds_total = float(payload.get("train_wall_seconds_total", 0.0))
        print(f"[CHECKPOINT] loaded {checkpoint_path}, resuming from update {t.start_update}.", flush=True)
