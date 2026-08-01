from __future__ import annotations

import time

import torch
from isaaclab.utils.math import quat_error_magnitude

from envs.spec import EE_Z_TERMINATION_THRESHOLD
from .env_state import restore_env_state, snapshot_env_state
from .restore_tolerance import float_restore_error
from .validation_metrics import (
    StepDiagnostics,
    terminal_phase_metrics,
)


def classify_mimickit_done_terms(
    done: torch.Tensor,
    done_terms: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply MimicKit's TIME -> SUCC -> FAIL overwrite precedence."""

    done = done.bool()
    failure = done & (
        done_terms["anchor_pos_bad"].bool()
        | done_terms["anchor_ori_bad"].bool()
        | done_terms["ee_body_bad"].bool()
    )
    motion_complete = done & done_terms["motion_complete"].bool() & ~failure
    timeout = done & done_terms["time_out"].bool() & ~motion_complete & ~failure
    return timeout, motion_complete, failure


def short_body_name(body_name: str) -> str:
    name = body_name.removesuffix("_link")
    for suffix in ("_yaw", "_roll"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def validation_max_steps(train_cfg, env) -> int:
    motion_steps = env.full_motion_control_steps()
    steps = int(train_cfg.validation_max_steps)
    steps = max(steps, motion_steps)
    target = int(train_cfg.target_validation_steps)
    if target > 0:
        steps = max(steps, target + 1)
    return max(1, steps)


def _reset_alignment_metrics(env, prefix: str) -> dict[str, float]:
    reference = env.get_reference_state()
    robot_body_pos = env.robot.data.body_pos_w[:, env.track_body_ids]
    robot_body_quat = env.robot.data.body_quat_w[:, env.track_body_ids]
    body_pos_err = torch.linalg.norm(robot_body_pos - reference["body_pos_w"], dim=-1)
    body_ori_deg = (
        quat_error_magnitude(reference["body_quat_w"], robot_body_quat)
        * (180.0 / 3.141592653589793)
    )
    body_pos_err_mean_by_body = body_pos_err.mean(dim=0)
    body_ori_deg_mean_by_body = body_ori_deg.mean(dim=0)
    body_pos_top_idx = int(torch.argmax(body_pos_err_mean_by_body).item())
    body_ori_top_idx = int(torch.argmax(body_ori_deg_mean_by_body).item())
    joint_pos, joint_vel = env.get_action_joint_state()
    # The environment configuration owns the root-link velocity contract for
    # reset, training, evaluation, and diagnostics alike.
    root_velocity = env.get_mimic_root_velocity_w()
    root_ori_deg = (
        quat_error_magnitude(reference["root_quat_w"], env.robot.data.root_quat_w)
        * (180.0 / 3.141592653589793)
    )
    anchor_ori_deg = (
        quat_error_magnitude(reference["anchor_quat_w"], env.robot.data.body_quat_w[:, env.anchor_body_id])
        * (180.0 / 3.141592653589793)
    )
    return {
        f"{prefix}/reset_root_pos_err": float(
            torch.linalg.norm(env.robot.data.root_pos_w - reference["root_pos_w"], dim=-1).mean().item()
        ),
        f"{prefix}/reset_root_ori_deg": float(root_ori_deg.mean().item()),
        f"{prefix}/reset_anchor_pos_err": float(
            torch.linalg.norm(env.robot.data.body_pos_w[:, env.anchor_body_id] - reference["anchor_pos_w"], dim=-1)
            .mean()
            .item()
        ),
        f"{prefix}/reset_anchor_ori_deg": float(anchor_ori_deg.mean().item()),
        f"{prefix}/reset_joint_pos_err": float(torch.abs(joint_pos - reference["joint_pos"]).mean().item()),
        f"{prefix}/reset_joint_vel_err": float(torch.abs(joint_vel - reference["joint_vel"]).mean().item()),
        f"{prefix}/reset_root_lin_vel_err": float(
            torch.abs(root_velocity[:, :3] - reference["root_lin_vel_w"]).mean().item()
        ),
        f"{prefix}/reset_root_ang_vel_err": float(
            torch.abs(root_velocity[:, 3:] - reference["root_ang_vel_w"]).mean().item()
        ),
        f"{prefix}/reset_body_pos_err": float(body_pos_err.mean().item()),
        f"{prefix}/reset_body_pos_max": float(body_pos_err_mean_by_body[body_pos_top_idx].item()),
        f"{prefix}/reset_body_pos_top_index": float(body_pos_top_idx),
        f"{prefix}/reset_body_ori_deg": float(body_ori_deg.mean().item()),
        f"{prefix}/reset_body_ori_max": float(body_ori_deg_mean_by_body[body_ori_top_idx].item()),
        f"{prefix}/reset_body_ori_top_index": float(body_ori_top_idx),
    }


@torch.no_grad()
def run_validation_rollout(
    trainer,
    fixed_seed: int | None = None,
    start_phase_override: int | None = None,
    *,
    restore_training_state: bool = True,
) -> dict[str, float]:
    algo = trainer.algo
    env = trainer.env
    policy = algo.policy
    tcfg = trainer.train_cfg
    num_envs = env.num_envs

    was_training = policy.training
    policy.eval()
    training_snapshot = snapshot_env_state(env)
    algorithm_snapshot = algo.snapshot_runtime_state()
    training_observation = trainer.current_observation.detach().clone()
    training_restore_observation_error = float("inf")
    training_restore_state_error = float("inf")
    env_device = torch.device(env.device)
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = None
    original_obs_noise = env.observation_noise
    env.observation_noise = False
    training_clean_observation = env.get_observation().detach().clone()
    original_reset_noise = env.reset_noise
    original_interval_pushes = env.interval_pushes
    env.reset_noise = False
    env.interval_pushes = False

    original_record_failures = env.record_motion_failures
    env.record_motion_failures = False

    original_max_episode_steps = env.max_episode_steps
    max_steps = validation_max_steps(tcfg, env)
    env.max_episode_steps = env.full_motion_control_steps() + 1
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
    # Validation must not inherit solver warm-start or contact-history state
    # from the preceding training rollout.  A full physics reset makes the
    # fixed phase/seed contract identical in-process and after checkpoint load.
    env.sim.reset()
    current_obs = algo.evaluation_reset(validation_phase)
    print(f"[VALIDATION_RESET_DONE] time={time.perf_counter() - reset_t0:.3f}s", flush=True)
    reset_metrics = _reset_alignment_metrics(env, "validation")
    _, initial_joint_vel = env.get_action_joint_state()
    initial_root_velocity = env.get_mimic_root_velocity_w()
    step_diagnostics = StepDiagnostics(
        initial_action=env.last_action,
        initial_joint_vel=initial_joint_vel,
        initial_root_lin_vel=initial_root_velocity[:, :3],
        initial_root_ang_vel=initial_root_velocity[:, 3:],
    )
    done = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    survived_steps = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    latest_phase_steps = validation_phase.float().clone()


    death_phase_record = torch.zeros(num_envs, dtype=torch.float32, device=env.device)
    cumulative_reward = torch.zeros(num_envs, device=env.device)
    done_term_names = [
        "time_out",
        "motion_complete",
        "anchor_pos_bad",
        "anchor_ori_bad",
        "ee_body_bad",
    ]
    done_term_record = {
        name: torch.zeros(num_envs, dtype=torch.bool, device=env.device)
        for name in done_term_names
    }
    done_debug_record = {
        name: torch.zeros(num_envs, device=env.device)
        for name in ("ee_z_error_max", "ee_z_error_mean", "anchor_z_error", "anchor_gravity_z_error")
    }
    ee_body_count = len(env.ee_body_names)
    done_ee_z_error_record = torch.zeros(num_envs, ee_body_count, device=env.device)
    done_ee_bad_record = torch.zeros(num_envs, ee_body_count, dtype=torch.bool, device=env.device)
    track_body_count = len(env.track_body_names)
    done_action_abs_record = torch.zeros(num_envs, device=env.device)
    done_action_max_record = torch.zeros(num_envs, device=env.device)
    done_root_pos_err_record = torch.zeros(num_envs, device=env.device)
    done_root_ori_deg_record = torch.zeros(num_envs, device=env.device)
    done_anchor_pos_err_record = torch.zeros(num_envs, device=env.device)
    done_anchor_ori_deg_record = torch.zeros(num_envs, device=env.device)
    done_joint_pos_err_record = torch.zeros(num_envs, device=env.device)
    done_joint_vel_err_record = torch.zeros(num_envs, device=env.device)
    done_body_pos_err_record = torch.zeros(num_envs, track_body_count, device=env.device)
    done_body_z_err_record = torch.zeros(num_envs, track_body_count, device=env.device)
    done_pd_target_ref_now_record = torch.zeros(num_envs, device=env.device)
    done_pd_target_ref_next_record = torch.zeros(num_envs, device=env.device)
    pd_target_ref_now_accum = torch.zeros(num_envs, device=env.device)
    pd_target_ref_next_accum = torch.zeros(num_envs, device=env.device)
    pd_target_ref_next_joint_accum = torch.zeros(env.action_dim, device=env.device)
    pd_target_ref_next_joint_count = torch.zeros((), device=env.device)
    done_pd_target_ref_next_joint_record = torch.zeros(
        num_envs, env.action_dim, device=env.device
    )
    pd_target_ref_steps = torch.zeros(num_envs, device=env.device)
    diag_keys = [
        "diag_torso_ori_deg", "diag_left_wrist_ori_deg", "diag_right_wrist_ori_deg",
        "diag_left_elbow_ori_deg", "diag_right_elbow_ori_deg",
        "diag_left_shoulder_ori_deg", "diag_right_shoulder_ori_deg",
    ]
    diag_accum = {key: torch.zeros(num_envs, device=env.device) for key in diag_keys}
    diag_steps = torch.zeros(num_envs, device=env.device)

    try:
        rollout_t0 = time.perf_counter()
        with torch.no_grad():
            for step_idx in range(max_steps):
                if not trainer.simulation_app.is_running():
                    break
                active_mask = ~done
                action = algo.deterministic_action(current_obs)
                if bool(done.any()):
                    action = torch.where(done.unsqueeze(-1), torch.zeros_like(action), action)
                action_target = env.default_action_joint_pos + env.action_scale * torch.clamp(action, -100.0, 100.0)

                current_obs, reward, step_done, info = algo.evaluation_step(action)
                reference_post = env.motion.get_frame(info["reference_phase_steps"])
                robot_joint_pos, robot_joint_vel = env.get_action_joint_state()
                robot_root_velocity = env.get_mimic_root_velocity_w()
                step_diagnostics.update(
                    active_mask=active_mask,
                    action=action,
                    joint_pos=robot_joint_pos,
                    joint_vel=robot_joint_vel,
                    root_lin_vel=robot_root_velocity[:, :3],
                    root_ang_vel=robot_root_velocity[:, 3:],
                    reference_joint_pos=reference_post["joint_pos"],
                    reference_joint_vel=reference_post["joint_vel"],
                    reference_root_lin_vel=reference_post["root_lin_vel_w"],
                    reference_root_ang_vel=reference_post["root_ang_vel_w"],
                )
                latest_phase_steps = info["termination_phase_steps"].float().clone()
                ref_now = env.motion.get_frame(info["phase_start_steps"])["joint_pos"]
                ref_next = env.motion.get_frame(info["reference_phase_steps"])["joint_pos"]
                pd_target_ref_now_by_joint = torch.abs(action_target - ref_now)
                pd_target_ref_next_by_joint = torch.abs(action_target - ref_next)
                pd_target_ref_now = pd_target_ref_now_by_joint.mean(dim=-1)
                pd_target_ref_next = pd_target_ref_next_by_joint.mean(dim=-1)
                active_f = active_mask.float()
                pd_target_ref_now_accum += active_f * pd_target_ref_now
                pd_target_ref_next_accum += active_f * pd_target_ref_next
                if bool(active_mask.any()):
                    pd_target_ref_next_joint_accum += pd_target_ref_next_by_joint[
                        active_mask
                    ].sum(dim=0)
                    pd_target_ref_next_joint_count += active_mask.float().sum()
                pd_target_ref_steps += active_f
                new_done = active_mask & step_done
                if bool(new_done.any()):
                    done_terms = info["done_terms"]
                    debug_terms = info["debug_terms"]
                    death_phase_record[new_done] = info["termination_phase_steps"].float()[new_done]
                    for name in done_term_record:
                        if name in done_terms:
                            done_term_record[name][new_done] = done_terms[name][new_done]
                    for name in done_debug_record:
                        done_debug_record[name][new_done] = debug_terms[name][new_done]
                    ee_z_error_by_body = debug_terms["ee_z_error_by_body"][new_done]
                    done_ee_z_error_record[new_done] = ee_z_error_by_body
                    done_ee_bad_record[new_done] = ee_z_error_by_body > EE_Z_TERMINATION_THRESHOLD
                    reference = env.get_reference_state()
                    robot_joint_pos, robot_joint_vel = env.get_action_joint_state()
                    robot_body_pos = env.robot.data.body_pos_w[:, env.track_body_ids]
                    root_ori_deg = (
                        quat_error_magnitude(reference["root_quat_w"], env.robot.data.root_quat_w)
                        * (180.0 / 3.141592653589793)
                    )
                    anchor_ori_deg = (
                        quat_error_magnitude(
                            reference["anchor_quat_w"],
                            env.robot.data.body_quat_w[:, env.anchor_body_id],
                        )
                        * (180.0 / 3.141592653589793)
                    )
                    body_pos_err = torch.linalg.norm(robot_body_pos - reference["body_pos_w"], dim=-1)
                    body_z_err = torch.abs(robot_body_pos[..., 2] - reference["body_pos_w"][..., 2])
                    done_action_abs_record[new_done] = action[new_done].abs().mean(dim=-1)
                    done_action_max_record[new_done] = action[new_done].abs().max(dim=-1).values
                    done_pd_target_ref_now_record[new_done] = pd_target_ref_now[new_done]
                    done_pd_target_ref_next_record[new_done] = pd_target_ref_next[new_done]
                    done_pd_target_ref_next_joint_record[new_done] = (
                        pd_target_ref_next_by_joint[new_done]
                    )
                    done_root_pos_err_record[new_done] = torch.linalg.norm(
                        env.robot.data.root_pos_w[new_done] - reference["root_pos_w"][new_done],
                        dim=-1,
                    )
                    done_root_ori_deg_record[new_done] = root_ori_deg[new_done]
                    done_anchor_pos_err_record[new_done] = torch.linalg.norm(
                        env.robot.data.body_pos_w[new_done, env.anchor_body_id]
                        - reference["anchor_pos_w"][new_done],
                        dim=-1,
                    )
                    done_anchor_ori_deg_record[new_done] = anchor_ori_deg[new_done]
                    done_joint_pos_err_record[new_done] = torch.abs(
                        robot_joint_pos[new_done] - reference["joint_pos"][new_done]
                    ).mean(dim=-1)
                    done_joint_vel_err_record[new_done] = torch.abs(
                        robot_joint_vel[new_done] - reference["joint_vel"][new_done]
                    ).mean(dim=-1)
                    done_body_pos_err_record[new_done] = body_pos_err[new_done]
                    done_body_z_err_record[new_done] = body_z_err[new_done]
                survived_steps += active_mask.to(dtype=torch.long)
                cumulative_reward += active_mask.float() * reward
                reward_terms = info["reward_terms"]
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
                if bool(done.all()):
                    break
        final_phase_record = torch.where(done, death_phase_record, latest_phase_steps)
    finally:
        if restore_training_state:
            env.sim.reset()
            restore_env_state(env, training_snapshot)
            state_pairs = {
                "root_pose_w": (
                    env.robot.data.root_link_pose_w,
                    training_snapshot["root_pose_w"],
                ),
                "root_velocity_w": (
                    env.get_mimic_root_velocity_w(),
                    training_snapshot["root_velocity_w"],
                ),
                "joint_pos": (
                    env.robot.data.joint_pos,
                    training_snapshot["joint_pos"],
                ),
                "joint_vel": (
                    env.robot.data.joint_vel,
                    training_snapshot["joint_vel"],
                ),
                "last_action": (
                    env.last_action,
                    training_snapshot["last_action"],
                ),
            }
            restore_measurements = {
                name: float_restore_error(restored, expected)
                for name, (restored, expected) in state_pairs.items()
            }
            training_restore_state_error = max(
                measurement[0]
                for measurement in restore_measurements.values()
            )
            training_restore_state_ulp_ratio = max(
                measurement[1]
                for measurement in restore_measurements.values()
            )
            training_restore_state_tolerance = max(
                measurement[2]
                for measurement in restore_measurements.values()
            )
            failed_restore = {
                name: measurement
                for name, measurement in restore_measurements.items()
                if measurement[1] > 1.0
            }
            if failed_restore:
                raise RuntimeError(
                    "validation failed to restore raw training state within "
                    f"two ULPs: {failed_restore}"
                )
            restored_training_observation = env.get_observation()
            (
                training_restore_observation_error,
                training_restore_observation_bound_ratio,
                training_restore_observation_tolerance,
            ) = float_restore_error(
                restored_training_observation,
                training_clean_observation,
                # The observation contains FK quantities derived from the
                # restored world pose.  Its absolute representable floor must
                # therefore be no tighter than the accepted source-state
                # round-trip bound.
                absolute_floor=max(
                    1.0e-5, training_restore_state_tolerance
                ),
            )
            if training_restore_observation_bound_ratio > 1.0:
                raise RuntimeError(
                    "validation failed to restore the training observation "
                    "within its source-state float bound: "
                    f"error={training_restore_observation_error:.9g}, "
                    f"tolerance={training_restore_observation_tolerance:.9g}"
                )
        else:
            training_restore_state_error = 0.0
            training_restore_state_ulp_ratio = 0.0
            training_restore_state_tolerance = 0.0
            training_restore_observation_error = 0.0
            training_restore_observation_bound_ratio = 0.0
            training_restore_observation_tolerance = 0.0
        env.observation_noise = original_obs_noise
        env.reset_noise = original_reset_noise
        env.interval_pushes = original_interval_pushes
        env.record_motion_failures = original_record_failures
        env.max_episode_steps = original_max_episode_steps
        algo.restore_runtime_state(algorithm_snapshot)
        trainer.current_observation = training_observation
        torch.random.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, env_device)
        if was_training:
            policy.train()

    motion_span = max(1.0, float(env.motion_end_phase - max(0, int(start_phase))))
    reference_progress = torch.clamp(
        (final_phase_record - float(max(0, int(start_phase)))) / motion_span,
        min=0.0,
        max=1.0,
    )
    metrics: dict[str, float] = {
        "validation/steps_mean": float(survived_steps.float().mean().item()),
        "validation/steps_min": float(survived_steps.min().item()),
        "validation/steps_p50": float(torch.quantile(survived_steps.float(), 0.50).item()),
        "validation/steps_p95": float(torch.quantile(survived_steps.float(), 0.95).item()),
        "validation/steps_max": float(survived_steps.max().item()),
        "validation/return_mean": float(cumulative_reward.mean().item()),
        "validation/done_frac": float(done.float().mean().item()),
        "validation/survival_seconds_mean": float(survived_steps.float().mean().item() * env.dt),
        "validation/survival_seconds_p50": float(
            torch.quantile(survived_steps.float(), 0.50).item() * env.dt
        ),
        "validation/survival_seconds_p95": float(
            torch.quantile(survived_steps.float(), 0.95).item() * env.dt
        ),
        "validation/reference_progress_mean": float(reference_progress.mean().item()),
        "validation/reference_progress_p50": float(torch.quantile(reference_progress, 0.50).item()),
        "validation/reference_progress_p95": float(torch.quantile(reference_progress, 0.95).item()),
        "validation/full_motion_control_steps": float(env.full_motion_control_steps()),
        "validation/training_restore_clean_observation_error_max": (
            training_restore_observation_error
        ),
        "validation/training_restore_clean_observation_bound_ratio_max": (
            training_restore_observation_bound_ratio
        ),
        "validation/training_restore_clean_observation_tolerance_max": (
            training_restore_observation_tolerance
        ),
        "validation/training_restore_raw_state_error_max": (
            training_restore_state_error
        ),
        "validation/training_restore_raw_state_bound_ratio_max": (
            training_restore_state_ulp_ratio
        ),
        "validation/training_restore_raw_state_tolerance_max": (
            training_restore_state_tolerance
        ),
    }
    metrics.update(step_diagnostics.metrics())
    metrics.update(reset_metrics)
    timeout, motion_complete, failure = classify_mimickit_done_terms(done, done_term_record)
    metrics.update({
        "validation/time_out_frac": float(timeout.float().mean().item()),
        "validation/motion_complete_frac": float(motion_complete.float().mean().item()),
        "validation/failure_frac": float(failure.float().mean().item()),
        "validation/censored_frac": float((~done).float().mean().item()),
        "validation/anchor_pos_bad_frac": float(done_term_record["anchor_pos_bad"].float().mean().item()),
        "validation/anchor_ori_bad_frac": float(done_term_record["anchor_ori_bad"].float().mean().item()),
        "validation/ee_body_bad_frac": float(done_term_record["ee_body_bad"].float().mean().item()),
    })
    outcome_masks = {
        "failure": failure,
        "time_out": timeout,
        "motion_complete": motion_complete,
    }
    cause_masks = {
        name: done_term_record[name]
        for name in (
            "anchor_pos_bad",
            "anchor_ori_bad",
            "ee_body_bad",
            "pose_fail",
        )
        if name in done_term_record
    }
    for name, mask in {**outcome_masks, **cause_masks}.items():
        metrics.update(
            terminal_phase_metrics(
                f"validation/terminal/{name}",
                death_phase_record,
                mask,
                motion_end_phase=float(env.motion_end_phase),
            )
        )
    if "pose_fail" in done_term_record:
        metrics["validation/pose_fail_frac"] = float(done_term_record["pose_fail"].float().mean().item())
    if bool((pd_target_ref_steps > 0).any()):
        safe_action_steps = pd_target_ref_steps.clamp(min=1.0)
        metrics["validation/pd_target_vs_reference_joint_mae_now"] = float(
            (pd_target_ref_now_accum / safe_action_steps).mean().item()
        )
        metrics["validation/pd_target_vs_reference_joint_mae_next"] = float(
            (pd_target_ref_next_accum / safe_action_steps).mean().item()
        )
        if float(pd_target_ref_next_joint_count.item()) > 0.0:
            action_joint_mean = (
                pd_target_ref_next_joint_accum
                / pd_target_ref_next_joint_count.clamp(min=1.0)
            )
            top_count = min(3, int(env.action_dim))
            top_indices = torch.argsort(action_joint_mean, descending=True)[:top_count]
            for rank, joint_index in enumerate(top_indices, start=1):
                idx = int(joint_index.item())
                metrics[
                    "validation/pd_target_vs_reference_next_top"
                    f"{rank}_joint_index"
                ] = float(idx)
                metrics[
                    "validation/pd_target_vs_reference_next_top"
                    f"{rank}_joint_mae"
                ] = float(action_joint_mean[idx].item())


    if bool(failure.any()):
        metrics.update({
            "validation/ee_z_max": float(done_debug_record["ee_z_error_max"][failure].mean().item()),
            "validation/ee_z_mean": float(done_debug_record["ee_z_error_mean"][failure].mean().item()),
            "validation/anchor_z": float(done_debug_record["anchor_z_error"][failure].mean().item()),
            "validation/anchor_gravity": float(done_debug_record["anchor_gravity_z_error"][failure].mean().item()),
            "validation/fail_action_abs": float(done_action_abs_record[failure].mean().item()),
            "validation/fail_action_max": float(done_action_max_record[failure].mean().item()),
            "validation/fail_pd_target_vs_reference_joint_mae_now": float(
                done_pd_target_ref_now_record[failure].mean().item()
            ),
            "validation/fail_pd_target_vs_reference_joint_mae_next": float(
                done_pd_target_ref_next_record[failure].mean().item()
            ),
            "validation/fail_root_pos_err": float(done_root_pos_err_record[failure].mean().item()),
            "validation/fail_root_ori_deg": float(done_root_ori_deg_record[failure].mean().item()),
            "validation/fail_anchor_pos_err": float(done_anchor_pos_err_record[failure].mean().item()),
            "validation/fail_anchor_ori_deg": float(done_anchor_ori_deg_record[failure].mean().item()),
            "validation/fail_joint_pos_err": float(done_joint_pos_err_record[failure].mean().item()),
            "validation/fail_joint_vel_err": float(done_joint_vel_err_record[failure].mean().item()),
            "validation/fail_body_pos_err": float(done_body_pos_err_record[failure].mean().item()),
            "validation/fail_body_z_err": float(done_body_z_err_record[failure].mean().item()),
        })
        fail_action_joint_mean = done_pd_target_ref_next_joint_record[
            failure
        ].mean(dim=0)
        top_count = min(3, int(env.action_dim))
        top_indices = torch.argsort(fail_action_joint_mean, descending=True)[:top_count]
        for rank, joint_index in enumerate(top_indices, start=1):
            idx = int(joint_index.item())
            metrics[
                "validation/fail_pd_target_vs_reference_next_top"
                f"{rank}_joint_index"
            ] = float(idx)
            metrics[
                "validation/fail_pd_target_vs_reference_next_top"
                f"{rank}_joint_mae"
            ] = float(fail_action_joint_mean[idx].item())
        body_pos_mean = done_body_pos_err_record[failure].mean(dim=0)
        body_z_mean = done_body_z_err_record[failure].mean(dim=0)
        body_pos_top_idx = int(torch.argmax(body_pos_mean).item())
        body_z_top_idx = int(torch.argmax(body_z_mean).item())
        metrics["validation/fail_body_pos_top_index"] = float(body_pos_top_idx)
        metrics["validation/fail_body_pos_top_err"] = float(body_pos_mean[body_pos_top_idx].item())
        metrics["validation/fail_body_z_top_index"] = float(body_z_top_idx)
        metrics["validation/fail_body_z_top_err"] = float(body_z_mean[body_z_top_idx].item())
        for index, body_name in enumerate(env.ee_body_names):
            sname = short_body_name(body_name)
            metrics[f"validation/ee_{sname}_bad_frac"] = float(done_ee_bad_record[failure, index].float().mean().item())
            metrics[f"validation/ee_{sname}_z_error"] = float(done_ee_z_error_record[failure, index].mean().item())

    safe_steps = diag_steps.clamp(min=1.0)
    for key in diag_keys:
        metrics[f"validation/{key}"] = float((diag_accum[key] / safe_steps).mean().item())
    return metrics
