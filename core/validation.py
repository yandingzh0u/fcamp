"""Deterministic validation rollout. Algorithm-agnostic.

Resets all envs to validation_start_phase and rolls the algorithm's greedy action chunk
continuously, recording survival steps and termination causes. Preserves and restores the
training env state + RNG so validation never perturbs training.
"""
from __future__ import annotations

import time

import torch

from env.config import EE_Z_TERMINATION_THRESHOLD
from .env_state import restore_env_state, snapshot_env_state


def short_body_name(body_name: str) -> str:
    name = body_name.removesuffix("_link")
    for suffix in ("_yaw", "_roll"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def validation_max_steps(train_cfg, env) -> int:
    """VAL horizon = the full motion clip length (validate over the whole dataset), unless the
    user explicitly requests a longer survival target. validation_max_steps in the config is
    only a floor/explicit override; it must NOT cap VAL below the clip length (that would stop
    validation before the motion ends)."""
    motion = getattr(env, "motion", None)
    motion_frames = int(getattr(motion, "num_frames", 0) or 0)
    steps = int(train_cfg.validation_max_steps)
    if motion_frames > 0:
        steps = max(steps, motion_frames)
    target = int(train_cfg.target_validation_steps)
    if target > 0:
        steps = max(steps, target + 1)
    return max(1, steps)


def run_validation_rollout(
    trainer, fixed_seed: int | None = None, start_phase_override: int | None = None
) -> dict[str, float]:
    algo = trainer.algo
    env = trainer.env
    policy = algo.policy
    tcfg = trainer.train_cfg
    horizon = int(algo.cfg.horizon)
    num_envs = env.num_envs

    was_training = policy.training
    policy.eval()
    preserve_state = bool(tcfg.validation_preserve_state)
    training_snapshot = snapshot_env_state(env) if preserve_state else None
    training_observation = getattr(trainer, "current_observation", None)
    env_device = torch.device(env.device)
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = None
    original_obs_noise = getattr(env.task_cfg, "observation_noise", True)
    env.task_cfg.observation_noise = bool(tcfg.validation_observation_noise)
    # Eval deaths must not feed the training adaptive sampler.
    original_record_failures = getattr(env, "record_motion_failures", True)
    env.record_motion_failures = False
    # Validation scores survival to motion end: reaching the final frame is a clean stop
    # recorded as motion_complete (success), NOT a teleport roll-in and NOT disguised as a
    # time_out / tracking failure. A directional run (e.g. start_phase 800) therefore stops at
    # the final frame (~159 steps) instead of resampling and continuing.
    original_terminate_on_motion_end = getattr(env, "terminate_on_motion_end", False)
    env.terminate_on_motion_end = True
    # Validate over the WHOLE motion clip: lift the training episode-length time-out so envs
    # are not force-timed-out at max_episode_steps (e.g. 500) before the motion ends (959).
    # Survival is then bounded only by the real tracking-failure terminations + motion end.
    original_max_episode_steps = int(getattr(env.task_cfg, "max_episode_steps", -1))
    motion_frames = int(getattr(getattr(env, "motion", None), "num_frames", 0) or 0)
    if motion_frames > 0:
        # The full phase-0 validation scores the final reference frame on control step
        # ``motion_frames``. With a cap equal to motion_frames, ``episode_steps >= cap`` and
        # motion_complete become true on the same step, falsely labelling every successful run
        # as a timeout too. One extra step keeps motion_complete as the sole clean-success cause.
        env.task_cfg.max_episode_steps = motion_frames + 1
    if torch.cuda.is_available() and env_device.type == "cuda":
        cuda_rng_state = torch.cuda.get_rng_state(env_device)
    if fixed_seed is not None:
        if torch.cuda.is_available() and env_device.type == "cuda":
            torch.cuda.manual_seed_all(fixed_seed)
        torch.manual_seed(fixed_seed)

    start_phase = tcfg.validation_start_phase if start_phase_override is None else start_phase_override
    validation_phase = torch.full(
        (num_envs,), max(0, int(start_phase)), dtype=torch.long, device=env.device
    )
    reset_t0 = time.perf_counter()
    print("[VALIDATION_RESET_START]", flush=True)
    current_obs = env.reset(phase_indices=validation_phase)
    print(f"[VALIDATION_RESET_DONE] time={time.perf_counter() - reset_t0:.3f}s", flush=True)

    cached_chunk: torch.Tensor | None = None
    chunk_index = horizon
    done = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    survived_steps = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    # Actual motion frame the env was scored at when it died (info["termination_phase_steps"],
    # the pre-advance phase). NOT start_phase + survived_steps, which is off-by-one after the
    # Holosoma phase-timing change and ignores adaptive reset start frames.
    death_phase_record = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    cumulative_reward = torch.zeros(num_envs, device=env.device)
    done_term_record = {
        name: torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        for name in ("time_out", "motion_complete", "anchor_pos_bad", "anchor_ori_bad", "ee_body_bad")
    }
    done_debug_record = {
        name: torch.zeros(num_envs, device=env.device)
        for name in ("ee_z_error_max", "ee_z_error_mean", "anchor_z_error", "anchor_gravity_z_error")
    }
    ee_body_count = len(env.ee_body_names)
    done_ee_z_error_record = torch.zeros(num_envs, ee_body_count, device=env.device)
    done_ee_bad_record = torch.zeros(num_envs, ee_body_count, dtype=torch.bool, device=env.device)
    diag_keys = [
        "diag_torso_ori_deg", "diag_left_wrist_ori_deg", "diag_right_wrist_ori_deg",
        "diag_left_elbow_ori_deg", "diag_right_elbow_ori_deg",
        "diag_left_shoulder_ori_deg", "diag_right_shoulder_ori_deg",
        "diag_torso_ang_vel", "diag_left_wrist_ang_vel", "diag_right_wrist_ang_vel",
        "diag_left_elbow_ang_vel", "diag_right_elbow_ang_vel",
        "diag_left_shoulder_ang_vel", "diag_right_shoulder_ang_vel",
    ]
    diag_accum = {key: torch.zeros(num_envs, device=env.device) for key in diag_keys}
    diag_steps = torch.zeros(num_envs, device=env.device)

    max_steps = validation_max_steps(tcfg, env)
    try:
        rollout_t0 = time.perf_counter()
        with torch.no_grad():
            for step_idx in range(max_steps):
                if not trainer.simulation_app.is_running():
                    break
                if cached_chunk is None or chunk_index >= horizon:
                    cached_chunk = algo.deterministic_actions(current_obs)
                    chunk_index = 0
                action = cached_chunk[:, chunk_index, :]
                if bool(done.any()):
                    action = torch.where(done.unsqueeze(-1), torch.zeros_like(action), action)
                chunk_index += 1

                current_obs, reward, step_done, info = env.step(action, auto_reset=False)
                active_mask = ~done
                new_done = active_mask & step_done
                if bool(new_done.any()):
                    done_terms = info["done_terms"]
                    debug_terms = info["debug_terms"]
                    death_phase_record[new_done] = info["termination_phase_steps"][new_done]
                    for name in done_term_record:
                        done_term_record[name][new_done] = done_terms[name][new_done]
                    for name in done_debug_record:
                        done_debug_record[name][new_done] = debug_terms[name][new_done]
                    ee_z_error_by_body = debug_terms["ee_z_error_by_body"][new_done]
                    done_ee_z_error_record[new_done] = ee_z_error_by_body
                    done_ee_bad_record[new_done] = ee_z_error_by_body > EE_Z_TERMINATION_THRESHOLD
                survived_steps += active_mask.to(dtype=torch.long)
                cumulative_reward += active_mask.float() * reward
                reward_terms = info.get("reward_terms", {})
                active_f = active_mask.float()
                for key in diag_keys:
                    if key in reward_terms:
                        diag_accum[key] += active_f * reward_terms[key]
                diag_steps += active_f
                done |= step_done
                done_frac = done.float().mean().item()
                if step_idx == 0 or (step_idx + 1) % 50 == 0 or step_idx + 1 == max_steps or bool(done.all()):
                    print(
                        f"[VALIDATION_PROGRESS] step={step_idx + 1}/{max_steps} "
                        f"done={done_frac:.5f} alive={(~done).float().mean().item():.5f} "
                        f"elapsed={time.perf_counter() - rollout_t0:.3f}s",
                        flush=True,
                    )
                if done_frac >= float(tcfg.validation_done_frac_early_stop):
                    print(f"[VALIDATION_EARLY_STOP] step={step_idx + 1} done_frac={done_frac:.4f}", flush=True)
                    break
                if bool(done.all()):
                    break
    finally:
        env.task_cfg.observation_noise = original_obs_noise
        env.record_motion_failures = original_record_failures
        env.terminate_on_motion_end = original_terminate_on_motion_end
        env.task_cfg.max_episode_steps = original_max_episode_steps
        if training_snapshot is not None:
            restore_env_state(env, training_snapshot)
            trainer.current_observation = training_observation if training_observation is not None else env.get_observation()
        else:
            trainer.current_observation = env.get_observation()
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, env_device)
        if was_training:
            policy.train()

    metrics: dict[str, float] = {
        "validation/steps_mean": float(survived_steps.float().mean().item()),
        "validation/steps_min": float(survived_steps.min().item()),
        "validation/steps_p50": float(torch.quantile(survived_steps.float(), 0.50).item()),
        "validation/steps_p95": float(torch.quantile(survived_steps.float(), 0.95).item()),
        "validation/steps_max": float(survived_steps.max().item()),
        "validation/return_mean": float(cumulative_reward.mean().item()),
        "validation/done_frac": float(done.float().mean().item()),
    }
    # Phase-absolute success/failure rates over ALL initial envs (death_phase_record holds the
    # actual motion phase scored at death). alive_at_phase_850: share that did not terminate
    # before phase 850. wrist_fail_825_840: share that died of a wrist z-gate inside [825, 840].
    # motion_complete_rate: share that reached the final frame.
    alive_phase = 850
    wall_lo, wall_hi = 825, 840
    died = done & (~done_term_record["motion_complete"])
    died_before_alive = died & (death_phase_record < alive_phase)
    metrics["validation/alive_at_phase_850"] = float(1.0 - died_before_alive.float().mean().item())
    metrics["validation/motion_complete_rate"] = float(done_term_record["motion_complete"].float().mean().item())
    wrist_cols = [i for i, n in enumerate(env.ee_body_names) if "wrist" in n]
    if wrist_cols:
        wrist_bad_any = done_ee_bad_record[:, wrist_cols].any(dim=1)
    else:
        wrist_bad_any = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    in_wall = (death_phase_record >= wall_lo) & (death_phase_record <= wall_hi)
    wrist_fail = done_term_record["ee_body_bad"] & wrist_bad_any & in_wall
    metrics["validation/wrist_fail_825_840"] = float(wrist_fail.float().mean().item())

    if bool(done.any()):
        failed_phases = death_phase_record[done]
        metrics.update({
            "validation/fail_phase_mean": float(failed_phases.float().mean().item()),
            "validation/fail_phase_min": float(failed_phases.min().item()),
            "validation/fail_phase_max": float(failed_phases.max().item()),
            "validation/time_out_frac": float(done_term_record["time_out"].float().mean().item()),
            "validation/motion_complete_frac": float(done_term_record["motion_complete"].float().mean().item()),
            "validation/anchor_pos_bad_frac": float(done_term_record["anchor_pos_bad"].float().mean().item()),
            "validation/anchor_ori_bad_frac": float(done_term_record["anchor_ori_bad"].float().mean().item()),
            "validation/ee_body_bad_frac": float(done_term_record["ee_body_bad"].float().mean().item()),
            "validation/ee_z_max": float(done_debug_record["ee_z_error_max"][done].mean().item()),
            "validation/ee_z_mean": float(done_debug_record["ee_z_error_mean"][done].mean().item()),
            "validation/anchor_z": float(done_debug_record["anchor_z_error"][done].mean().item()),
            "validation/anchor_gravity": float(done_debug_record["anchor_gravity_z_error"][done].mean().item()),
        })
        for index, body_name in enumerate(env.ee_body_names):
            sname = short_body_name(body_name)
            metrics[f"validation/ee_{sname}_bad_frac"] = float(done_ee_bad_record[done, index].float().mean().item())
            metrics[f"validation/ee_{sname}_z_error"] = float(done_ee_z_error_record[done, index].mean().item())

    safe_steps = diag_steps.clamp(min=1.0)
    for key in diag_keys:
        metrics[f"validation/{key}"] = float((diag_accum[key] / safe_steps).mean().item())
    return metrics
