from __future__ import annotations

import time

import torch
from isaaclab.utils.math import quat_error_magnitude

from envs.spec import EE_Z_TERMINATION_THRESHOLD
from .env_state import restore_env_state, snapshot_env_state


def short_body_name(body_name: str) -> str:
    name = body_name.removesuffix("_link")
    for suffix in ("_yaw", "_roll"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def validation_max_steps(train_cfg, env) -> int:

    motion_frames = env.motion.num_frames
    steps = int(train_cfg.validation_max_steps)
    if motion_frames > 0:
        steps = max(steps, motion_frames)
    target = int(train_cfg.target_validation_steps)
    if target > 0:
        steps = max(steps, target + 1)
    return max(1, steps)


def _deployment_action_chunk(algo, obs: torch.Tensor) -> torch.Tensor:
    payload = algo.deployment_actions(obs)
    if payload.dim() == 2:
        return payload.unsqueeze(1)
    if payload.dim() == 3:
        return payload
    raise ValueError(f"deployment_actions must return [N,D] or [N,H,D], got shape={tuple(payload.shape)}")


def _split_reference_action(algo, env, payload: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not bool(getattr(algo, "uses_reference_dt", False)):
        return payload, None
    expected_dim = int(env.action_dim) + 1
    if payload.shape[-1] != expected_dim:
        raise ValueError(
            f"{algo.__class__.__name__} uses reference_dt but returned action dim "
            f"{payload.shape[-1]}, expected {expected_dim}"
        )
    return payload[..., : env.action_dim], payload[..., -1]


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
        f"{prefix}/reset_body_pos_err": float(body_pos_err.mean().item()),
        f"{prefix}/reset_body_pos_max": float(body_pos_err_mean_by_body[body_pos_top_idx].item()),
        f"{prefix}/reset_body_pos_top_index": float(body_pos_top_idx),
        f"{prefix}/reset_body_ori_deg": float(body_ori_deg.mean().item()),
        f"{prefix}/reset_body_ori_max": float(body_ori_deg_mean_by_body[body_ori_top_idx].item()),
        f"{prefix}/reset_body_ori_top_index": float(body_ori_top_idx),
    }


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
    training_observation = trainer.current_observation
    env_device = torch.device(env.device)
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = None
    original_obs_noise = env.observation_noise
    env.observation_noise = False

    original_record_failures = env.record_motion_failures
    env.record_motion_failures = False


    original_terminate_on_motion_end = env.terminate_on_motion_end
    env.terminate_on_motion_end = getattr(env, "termination_mode", "tracking") != "amp"


    original_max_episode_steps = env.max_episode_steps
    motion_frames = env.motion.num_frames
    if motion_frames > 0:


        env.max_episode_steps = motion_frames + 1
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
    reset_metrics = _reset_alignment_metrics(env, "validation")

    cached_chunk: torch.Tensor | None = None
    chunk_index = horizon
    done = torch.zeros(num_envs, dtype=torch.bool, device=env.device)
    survived_steps = torch.zeros(num_envs, dtype=torch.long, device=env.device)


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
    amp_contact_body_ids = getattr(
        env,
        "amp_undesired_contact_body_ids",
        torch.empty(0, dtype=torch.long, device=env.device),
    )
    amp_contact_body_ids = amp_contact_body_ids.to(device=env.device, dtype=torch.long)
    done_amp_contact_force_record = torch.zeros(
        num_envs,
        int(amp_contact_body_ids.numel()),
        device=env.device,
    )
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
                if cached_chunk is None or chunk_index >= cached_chunk.shape[1]:
                    cached_chunk = _deployment_action_chunk(algo, current_obs)
                    chunk_index = 0
                action_payload = cached_chunk[:, chunk_index, :]
                action, reference_dt = _split_reference_action(algo, env, action_payload)
                if bool(done.any()):
                    action = torch.where(done.unsqueeze(-1), torch.zeros_like(action), action)
                    if reference_dt is not None:
                        reference_dt = torch.where(done, torch.full_like(reference_dt, float(env.dt)), reference_dt)
                chunk_index += 1

                current_obs, reward, step_done, info = env.step(
                    action,
                    auto_reset=False,
                    reference_dt=reference_dt,
                )
                active_mask = ~done
                new_done = active_mask & step_done
                if bool(new_done.any()):
                    done_terms = info["done_terms"]
                    debug_terms = info["debug_terms"]
                    death_phase_record[new_done] = info["termination_phase_steps"].long()[new_done]
                    for name in done_term_record:
                        done_term_record[name][new_done] = done_terms[name][new_done]
                    for name in done_debug_record:
                        done_debug_record[name][new_done] = debug_terms[name][new_done]
                    ee_z_error_by_body = debug_terms["ee_z_error_by_body"][new_done]
                    done_ee_z_error_record[new_done] = ee_z_error_by_body
                    done_ee_bad_record[new_done] = ee_z_error_by_body > EE_Z_TERMINATION_THRESHOLD
                    amp_contact_force = debug_terms.get("amp_undesired_contact_force_by_body")
                    if torch.is_tensor(amp_contact_force) and amp_contact_force.shape[1:] == done_amp_contact_force_record.shape[1:]:
                        done_amp_contact_force_record[new_done] = amp_contact_force[new_done]
                    reference = env.get_reference_state()
                    robot_joint_pos, robot_joint_vel = env.get_action_joint_state()
                    robot_body_pos = env.robot.data.body_pos_w[:, env.track_body_ids]
                    robot_body_quat = env.robot.data.body_quat_w[:, env.track_body_ids]
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
        validation_first_push_step = env.first_push_step.clone()
    finally:
        env.observation_noise = original_obs_noise
        env.record_motion_failures = original_record_failures
        env.terminate_on_motion_end = original_terminate_on_motion_end
        env.max_episode_steps = original_max_episode_steps
        restore_env_state(env, training_snapshot)
        trainer.current_observation = training_observation
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
    metrics.update(reset_metrics)
    _pushed = validation_first_push_step >= 0
    _died = done & (~done_term_record["motion_complete"])
    metrics["validation/push_applied_frac"] = float(_pushed.float().mean().item())
    metrics["validation/died_before_push_frac"] = float((_died & ~_pushed).float().mean().item())
    metrics["validation/pushed_then_died_frac"] = float((_died & _pushed).float().mean().item())
    _pushed_steps = validation_first_push_step[_pushed]
    metrics["validation/first_push_step_count"] = float(_pushed_steps.numel())
    metrics["validation/first_push_step_mean"] = (
        float(_pushed_steps.float().mean().item()) if _pushed_steps.numel() > 0 else -1.0
    )


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
            "validation/fail_action_abs": float(done_action_abs_record[done].mean().item()),
            "validation/fail_action_max": float(done_action_max_record[done].mean().item()),
            "validation/fail_root_pos_err": float(done_root_pos_err_record[done].mean().item()),
            "validation/fail_root_ori_deg": float(done_root_ori_deg_record[done].mean().item()),
            "validation/fail_anchor_pos_err": float(done_anchor_pos_err_record[done].mean().item()),
            "validation/fail_anchor_ori_deg": float(done_anchor_ori_deg_record[done].mean().item()),
            "validation/fail_joint_pos_err": float(done_joint_pos_err_record[done].mean().item()),
            "validation/fail_joint_vel_err": float(done_joint_vel_err_record[done].mean().item()),
            "validation/fail_body_pos_err": float(done_body_pos_err_record[done].mean().item()),
            "validation/fail_body_z_err": float(done_body_z_err_record[done].mean().item()),
        })
        body_pos_mean = done_body_pos_err_record[done].mean(dim=0)
        body_z_mean = done_body_z_err_record[done].mean(dim=0)
        body_pos_top_idx = int(torch.argmax(body_pos_mean).item())
        body_z_top_idx = int(torch.argmax(body_z_mean).item())
        metrics["validation/fail_body_pos_top_index"] = float(body_pos_top_idx)
        metrics["validation/fail_body_pos_top_err"] = float(body_pos_mean[body_pos_top_idx].item())
        metrics["validation/fail_body_z_top_index"] = float(body_z_top_idx)
        metrics["validation/fail_body_z_top_err"] = float(body_z_mean[body_z_top_idx].item())
        for index, body_name in enumerate(env.ee_body_names):
            sname = short_body_name(body_name)
            metrics[f"validation/ee_{sname}_bad_frac"] = float(done_ee_bad_record[done, index].float().mean().item())
            metrics[f"validation/ee_{sname}_z_error"] = float(done_ee_z_error_record[done, index].mean().item())
        if amp_contact_body_ids.numel() > 0:
            failed_contact = done_amp_contact_force_record[done]
            contact_frac = (failed_contact > 0.1).float().mean(dim=0)
            contact_force = failed_contact.mean(dim=0)
            score = contact_frac * 1000.0 + contact_force
            top_count = min(3, int(amp_contact_body_ids.numel()))
            top_indices = torch.argsort(score, descending=True)[:top_count]
            for rank, contact_index in enumerate(top_indices, start=1):
                idx = int(contact_index.item())
                metrics[f"validation/contact_top{rank}_index"] = float(idx)
                metrics[f"validation/contact_top{rank}_frac"] = float(contact_frac[idx].item())
                metrics[f"validation/contact_top{rank}_force"] = float(contact_force[idx].item())

    safe_steps = diag_steps.clamp(min=1.0)
    for key in diag_keys:
        metrics[f"validation/{key}"] = float((diag_accum[key] / safe_steps).mean().item())
    return metrics
