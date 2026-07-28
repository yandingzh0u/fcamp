from __future__ import annotations

import time

import torch
from isaaclab.utils.math import quat_error_magnitude

from components.evaluation import (
    DemoFeatureNormalizer,
    PhaseMatchedWindowMMD,
    sanitize_reference_phases,
)
from envs.spec import EE_Z_TERMINATION_THRESHOLD
from .env_state import restore_env_state, snapshot_env_state
from .validation_metrics import ChunkBoundaryDiagnostics, terminal_phase_metrics


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
    # FCAMP writes and discriminates root-link velocity. Read the same frame so
    # the reset diagnostic cannot become a COM-vs-link comparison artifact.
    root_velocity = env.get_mimic_root_velocity_w(velocity_frame="link")
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


@torch.inference_mode()
def run_validation_rollout(
    trainer, fixed_seed: int | None = None, start_phase_override: int | None = None
) -> dict[str, float]:
    algo = trainer.algo
    env = trainer.env
    policy = algo.policy
    tcfg = trainer.train_cfg
    horizon = algo.horizon
    num_envs = env.num_envs

    was_training = policy.training
    policy.eval()
    training_snapshot = snapshot_env_state(env)
    algorithm_snapshot = algo.snapshot_runtime_state()
    training_observation = trainer.current_observation
    env_device = torch.device(env.device)
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = None
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
    current_obs = algo.evaluation_reset(validation_phase)
    print(f"[VALIDATION_RESET_DONE] time={time.perf_counter() - reset_t0:.3f}s", flush=True)
    reset_metrics = _reset_alignment_metrics(env, "validation")
    _, initial_joint_vel = env.get_action_joint_state()
    initial_root_ang_vel = env.get_mimic_root_velocity_w()[:, 3:]
    chunk_diagnostics = ChunkBoundaryDiagnostics(
        horizon=horizon,
        initial_action=env.last_action,
        initial_joint_vel=initial_joint_vel,
        initial_root_ang_vel=initial_root_ang_vel,
    )

    demo_frame_indices = torch.arange(
        env.motion.num_frames, dtype=torch.long, device=env.device
    )
    motion_metric = PhaseMatchedWindowMMD(
        num_envs=num_envs,
        normalizer=DemoFeatureNormalizer.fit(
            env.motion.get_imitation_frame_at_times(demo_frame_indices.float())
        ),
        device=env.device,
        reference_phase_start=float(max(0, int(start_phase))),
        reference_phase_end=float(env.motion_end_phase),
    )

    cached_chunk: torch.Tensor | None = None
    chunk_index = horizon
    done = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    survived_steps = torch.zeros(num_envs, dtype=torch.long, device=env.device)
    latest_phase_steps = validation_phase.float().clone()


    death_phase_record = torch.zeros(num_envs, dtype=torch.float32, device=env.device)
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
    done_action_ref_now_record = torch.zeros(num_envs, device=env.device)
    done_action_ref_next_record = torch.zeros(num_envs, device=env.device)
    action_ref_now_accum = torch.zeros(num_envs, device=env.device)
    action_ref_next_accum = torch.zeros(num_envs, device=env.device)
    action_ref_next_joint_accum = torch.zeros(env.action_dim, device=env.device)
    action_ref_next_joint_count = torch.zeros((), device=env.device)
    done_action_ref_next_joint_record = torch.zeros(num_envs, env.action_dim, device=env.device)
    action_ref_steps = torch.zeros(num_envs, device=env.device)
    try:
        rollout_t0 = time.perf_counter()
        with torch.no_grad():
            for step_idx in range(max_steps):
                if not trainer.simulation_app.is_running():
                    break
                if cached_chunk is None or chunk_index >= cached_chunk.shape[1]:
                    cached_chunk = algo.deterministic_actions(current_obs)
                    chunk_index = 0
                primitive_offset = step_idx % horizon
                action = cached_chunk[:, chunk_index, :]
                if bool(done.any()):
                    action = torch.where(done.unsqueeze(-1), torch.zeros_like(action), action)
                chunk_index += 1
                active_mask = ~done
                action_target = env.default_action_joint_pos + env.action_scale * torch.clamp(action, -100.0, 100.0)

                current_obs, step_done, info = algo.evaluation_step(action)
                reference_post = env.motion.get_frame(info["reference_phase_steps"])
                robot_joint_pos, robot_joint_vel = env.get_action_joint_state()
                robot_root_ang_vel = env.get_mimic_root_velocity_w()[:, 3:]
                chunk_diagnostics.update(
                    active_mask=active_mask,
                    chunk_offset=primitive_offset,
                    action=action,
                    joint_pos=robot_joint_pos,
                    joint_vel=robot_joint_vel,
                    root_ang_vel=robot_root_ang_vel,
                    reference_joint_pos=reference_post["joint_pos"],
                    reference_joint_vel=reference_post["joint_vel"],
                    reference_root_ang_vel=reference_post["root_ang_vel_w"],
                )
                metric_env_ids = motion_metric.env_ids
                policy_imitation_frame = env.get_evaluator_imitation_policy_frame(
                    metric_env_ids
                )
                metric_phases_raw = info["imitation_frame_phase_steps"].index_select(
                    0, metric_env_ids
                )
                metric_phases, phase_valid = sanitize_reference_phases(
                    metric_phases_raw,
                    num_frames=env.motion.num_frames,
                )
                demo_imitation_frame = env.motion.get_imitation_frame_at_times(metric_phases)
                # Terminal post-action states are excluded. Every legal window
                # therefore contains W consecutive states that remained alive,
                # in range, and on the same unwrapped motion trajectory.
                motion_metric.update_selected(
                    policy_imitation_frame,
                    demo_imitation_frame,
                    (active_mask & ~step_done).index_select(0, metric_env_ids) & phase_valid,
                    metric_phases,
                )
                latest_phase_steps = info["termination_phase_steps"].float().clone()
                ref_now = env.motion.get_frame(info["phase_start_steps"])["joint_pos"]
                ref_next = env.motion.get_frame(info["reference_phase_steps"])["joint_pos"]
                action_ref_now_by_joint = torch.abs(action_target - ref_now)
                action_ref_next_by_joint = torch.abs(action_target - ref_next)
                action_ref_now = action_ref_now_by_joint.mean(dim=-1)
                action_ref_next = action_ref_next_by_joint.mean(dim=-1)
                active_f = active_mask.float()
                action_ref_now_accum += active_f * action_ref_now
                action_ref_next_accum += active_f * action_ref_next
                if bool(active_mask.any()):
                    action_ref_next_joint_accum += action_ref_next_by_joint[active_mask].sum(dim=0)
                    action_ref_next_joint_count += active_mask.float().sum()
                action_ref_steps += active_f
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
                    done_action_ref_now_record[new_done] = action_ref_now[new_done]
                    done_action_ref_next_record[new_done] = action_ref_next[new_done]
                    done_action_ref_next_joint_record[new_done] = action_ref_next_by_joint[new_done]
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
        validation_first_push_step = env.first_push_step.clone()
        final_phase_record = torch.where(done, death_phase_record, latest_phase_steps)
    finally:
        env.reset_noise = original_reset_noise
        env.interval_pushes = original_interval_pushes
        env.record_motion_failures = original_record_failures
        env.max_episode_steps = original_max_episode_steps
        restore_env_state(env, training_snapshot)
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
    }
    metrics.update(motion_metric.metrics())
    metrics.update(chunk_diagnostics.metrics())
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
    _pushed = validation_first_push_step >= 0
    _died = failure
    metrics["validation/push_applied_frac"] = float(_pushed.float().mean().item())
    metrics["validation/died_before_push_frac"] = float((_died & ~_pushed).float().mean().item())
    metrics["validation/pushed_then_died_frac"] = float((_died & _pushed).float().mean().item())
    _pushed_steps = validation_first_push_step[_pushed]
    metrics["validation/first_push_step_count"] = float(_pushed_steps.numel())
    metrics["validation/first_push_step_mean"] = (
        float(_pushed_steps.float().mean().item()) if _pushed_steps.numel() > 0 else -1.0
    )
    if bool((action_ref_steps > 0).any()):
        safe_action_steps = action_ref_steps.clamp(min=1.0)
        metrics["validation/action_target_ref_now_abs"] = float(
            (action_ref_now_accum / safe_action_steps).mean().item()
        )
        metrics["validation/action_target_ref_next_abs"] = float(
            (action_ref_next_accum / safe_action_steps).mean().item()
        )
        if float(action_ref_next_joint_count.item()) > 0.0:
            action_joint_mean = action_ref_next_joint_accum / action_ref_next_joint_count.clamp(min=1.0)
            top_count = min(3, int(env.action_dim))
            top_indices = torch.argsort(action_joint_mean, descending=True)[:top_count]
            for rank, joint_index in enumerate(top_indices, start=1):
                idx = int(joint_index.item())
                metrics[f"validation/action_target_ref_next_top{rank}_joint_index"] = float(idx)
                metrics[f"validation/action_target_ref_next_top{rank}_joint_err"] = float(
                    action_joint_mean[idx].item()
                )


    if bool(failure.any()):
        failed_phases = death_phase_record[failure]
        metrics.update({
            "validation/fail_phase_mean": float(failed_phases.float().mean().item()),
            "validation/fail_phase_min": float(failed_phases.min().item()),
            "validation/fail_phase_max": float(failed_phases.max().item()),
            "validation/ee_z_max": float(done_debug_record["ee_z_error_max"][failure].mean().item()),
            "validation/ee_z_mean": float(done_debug_record["ee_z_error_mean"][failure].mean().item()),
            "validation/anchor_z": float(done_debug_record["anchor_z_error"][failure].mean().item()),
            "validation/anchor_gravity": float(done_debug_record["anchor_gravity_z_error"][failure].mean().item()),
            "validation/fail_action_abs": float(done_action_abs_record[failure].mean().item()),
            "validation/fail_action_max": float(done_action_max_record[failure].mean().item()),
            "validation/fail_action_target_ref_now_abs": float(done_action_ref_now_record[failure].mean().item()),
            "validation/fail_action_target_ref_next_abs": float(done_action_ref_next_record[failure].mean().item()),
            "validation/fail_root_pos_err": float(done_root_pos_err_record[failure].mean().item()),
            "validation/fail_root_ori_deg": float(done_root_ori_deg_record[failure].mean().item()),
            "validation/fail_anchor_pos_err": float(done_anchor_pos_err_record[failure].mean().item()),
            "validation/fail_anchor_ori_deg": float(done_anchor_ori_deg_record[failure].mean().item()),
            "validation/fail_joint_pos_err": float(done_joint_pos_err_record[failure].mean().item()),
            "validation/fail_joint_vel_err": float(done_joint_vel_err_record[failure].mean().item()),
            "validation/fail_body_pos_err": float(done_body_pos_err_record[failure].mean().item()),
            "validation/fail_body_z_err": float(done_body_z_err_record[failure].mean().item()),
        })
        fail_action_joint_mean = done_action_ref_next_joint_record[failure].mean(dim=0)
        top_count = min(3, int(env.action_dim))
        top_indices = torch.argsort(fail_action_joint_mean, descending=True)[:top_count]
        for rank, joint_index in enumerate(top_indices, start=1):
            idx = int(joint_index.item())
            metrics[f"validation/fail_action_target_ref_next_top{rank}_joint_index"] = float(idx)
            metrics[f"validation/fail_action_target_ref_next_top{rank}_joint_err"] = float(
                fail_action_joint_mean[idx].item()
            )
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
    return metrics
