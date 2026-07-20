from __future__ import annotations

from .validation import short_body_name


def log_validation_metrics(env, metrics: dict[str, float]) -> None:
    if "validation/steps_mean" in metrics:
        _log_validation_block(env, "VAL", "validation", metrics)
    if "validation_directional/steps_mean" in metrics:
        _log_validation_block(env, "VAL_DIR", "validation_directional", metrics)
    if "val_fixed/steps_mean" in metrics:
        _log_validation_block(env, "VAL_FIXED", "val_fixed", metrics)


def _log_validation_block(env, label: str, prefix: str, metrics: dict[str, float]) -> None:
    print(
        f"[{label}] steps_mean={metrics[f'{prefix}/steps_mean']:.2f} "
        f"steps_min={metrics.get(f'{prefix}/steps_min', float('nan')):.0f} "
        f"steps_p50={metrics.get(f'{prefix}/steps_p50', float('nan')):.0f} "
        f"steps_p95={metrics.get(f'{prefix}/steps_p95', float('nan')):.0f} "
        f"steps_max={metrics.get(f'{prefix}/steps_max', float('nan')):.0f} "
        f"fail_phase_mean={metrics.get(f'{prefix}/fail_phase_mean', float('nan')):.1f} "
        f"return={metrics[f'{prefix}/return_mean']:.5f} "
        f"done={metrics.get(f'{prefix}/done_frac', float('nan')):.5f}",
        flush=True,
    )
    print(
        f"[{label}_CAUSE] "
        f"anchor_pos_bad={metrics.get(f'{prefix}/anchor_pos_bad_frac', float('nan')):.5f} "
        f"anchor_ori_bad={metrics.get(f'{prefix}/anchor_ori_bad_frac', float('nan')):.5f} "
        f"ee_body_bad={metrics.get(f'{prefix}/ee_body_bad_frac', float('nan')):.5f} "
        f"fall_contact={metrics.get(f'{prefix}/fall_contact_frac', float('nan')):.5f} "
        f"pose_fail={metrics.get(f'{prefix}/pose_fail_frac', float('nan')):.5f} "
        f"time_out={metrics.get(f'{prefix}/time_out_frac', float('nan')):.5f} "
        f"anchor_z={metrics.get(f'{prefix}/anchor_z', float('nan')):.5f} "
        f"anchor_grav={metrics.get(f'{prefix}/anchor_gravity', float('nan')):.5f} "
        f"ee_z_max={metrics.get(f'{prefix}/ee_z_max', float('nan')):.5f} "
        f"ee_z_mean={metrics.get(f'{prefix}/ee_z_mean', float('nan')):.5f}",
        flush=True,
    )
    fail_pos_idx = int(metrics.get(f"{prefix}/fail_body_pos_top_index", -1.0))
    fail_z_idx = int(metrics.get(f"{prefix}/fail_body_z_top_index", -1.0))
    fail_pos_body = (
        short_body_name(env.track_body_names[fail_pos_idx])
        if 0 <= fail_pos_idx < len(env.track_body_names)
        else "n/a"
    )
    fail_z_body = (
        short_body_name(env.track_body_names[fail_z_idx])
        if 0 <= fail_z_idx < len(env.track_body_names)
        else "n/a"
    )
    print(
        f"[{label}_FAIL] "
        f"act_abs={metrics.get(f'{prefix}/fail_action_abs', float('nan')):.5f} "
        f"act_max={metrics.get(f'{prefix}/fail_action_max', float('nan')):.5f} "
        f"root_pos={metrics.get(f'{prefix}/fail_root_pos_err', float('nan')):.5f} "
        f"root_ori={metrics.get(f'{prefix}/fail_root_ori_deg', float('nan')):.3f}deg "
        f"anchor_pos={metrics.get(f'{prefix}/fail_anchor_pos_err', float('nan')):.5f} "
        f"anchor_ori={metrics.get(f'{prefix}/fail_anchor_ori_deg', float('nan')):.3f}deg "
        f"joint_pos={metrics.get(f'{prefix}/fail_joint_pos_err', float('nan')):.5f} "
        f"joint_vel={metrics.get(f'{prefix}/fail_joint_vel_err', float('nan')):.5f} "
        f"body_pos={metrics.get(f'{prefix}/fail_body_pos_err', float('nan')):.5f} "
        f"body_pos_top={fail_pos_body}:{metrics.get(f'{prefix}/fail_body_pos_top_err', float('nan')):.5f} "
        f"body_z={metrics.get(f'{prefix}/fail_body_z_err', float('nan')):.5f} "
        f"body_z_top={fail_z_body}:{metrics.get(f'{prefix}/fail_body_z_top_err', float('nan')):.5f}",
        flush=True,
    )
    print(
        f"[{label}_OUTCOME] "
        f"failure={metrics.get(f'{prefix}/failure_frac', float('nan')):.4f} "
        f"motion_complete={metrics.get(f'{prefix}/motion_complete_frac', float('nan')):.4f} "
        f"time_out={metrics.get(f'{prefix}/time_out_frac', float('nan')):.4f} "
        f"steps_p50={metrics.get(f'{prefix}/steps_p50', float('nan')):.0f}",
        flush=True,
    )
    print(
        f"[{label}_PHASE] "
        f"failure_p50={metrics.get(f'{prefix}/terminal/failure/phase_p50', float('nan')):.1f} "
        f"failure_p95={metrics.get(f'{prefix}/terminal/failure/phase_p95', float('nan')):.1f} "
        f"anchor_pos_p50={metrics.get(f'{prefix}/terminal/anchor_pos_bad/phase_p50', float('nan')):.1f} "
        f"anchor_ori_p50={metrics.get(f'{prefix}/terminal/anchor_ori_bad/phase_p50', float('nan')):.1f} "
        f"ee_body_p50={metrics.get(f'{prefix}/terminal/ee_body_bad/phase_p50', float('nan')):.1f} "
        f"fall_contact_p50={metrics.get(f'{prefix}/terminal/fall_contact/phase_p50', float('nan')):.1f} "
        f"motion_complete_p50={metrics.get(f'{prefix}/terminal/motion_complete/phase_p50', float('nan')):.1f}",
        flush=True,
    )
    print(
        f"[{label}_MOTION] "
        f"window_mmd2_w16={metrics.get(f'{prefix}/window_mmd2_w16', float('nan')):.7f} "
        f"window_mmd2_w32={metrics.get(f'{prefix}/window_mmd2_w32', float('nan')):.7f} "
        f"raw_w16={metrics.get(f'{prefix}/window_mmd2_raw_w16', float('nan')):.7f} "
        f"raw_w32={metrics.get(f'{prefix}/window_mmd2_raw_w32', float('nan')):.7f} "
        f"pairs_w16={metrics.get(f'{prefix}/window_mmd_pairs_w16', 0.0):.0f} "
        f"pairs_w32={metrics.get(f'{prefix}/window_mmd_pairs_w32', 0.0):.0f} "
        f"gap_w16={metrics.get(f'{prefix}/window_mmd_pair_gap_min_w16', 0.0):.0f} "
        f"gap_w32={metrics.get(f'{prefix}/window_mmd_pair_gap_min_w32', 0.0):.0f} "
        f"phase_mean_w16={metrics.get(f'{prefix}/window_phase_endpoint_mean_w16', float('nan')):.2f} "
        f"phase_max_w16={metrics.get(f'{prefix}/window_phase_endpoint_max_w16', float('nan')):.2f} "
        f"phase_mean_w32={metrics.get(f'{prefix}/window_phase_endpoint_mean_w32', float('nan')):.2f} "
        f"phase_max_w32={metrics.get(f'{prefix}/window_phase_endpoint_max_w32', float('nan')):.2f} "
        f"progress_max_w16={metrics.get(f'{prefix}/window_reference_progress_max_w16', float('nan')):.4f} "
        f"progress_max_w32={metrics.get(f'{prefix}/window_reference_progress_max_w32', float('nan')):.4f} "
        f"progress_span_w16={metrics.get(f'{prefix}/window_reference_progress_span_w16', float('nan')):.4f} "
        f"progress_span_w32={metrics.get(f'{prefix}/window_reference_progress_span_w32', float('nan')):.4f} "
        f"samples_w16={metrics.get(f'{prefix}/window_mmd_samples_w16', 0.0):.0f} "
        f"samples_w32={metrics.get(f'{prefix}/window_mmd_samples_w32', 0.0):.0f} "
        f"selected_envs={metrics.get(f'{prefix}/window_mmd_selected_envs', 0.0):.0f}",
        flush=True,
    )
    print(
        f"[{label}_CHUNK_ACTION] "
        f"delta_boundary={metrics.get(f'{prefix}/chunk_action_delta_boundary_mean', float('nan')):.6f} "
        f"delta_internal={metrics.get(f'{prefix}/chunk_action_delta_internal_mean', float('nan')):.6f} "
        f"delta_ratio={metrics.get(f'{prefix}/chunk_action_delta_boundary_internal_ratio', float('nan')):.4f} "
        f"delta_boundary_p95={metrics.get(f'{prefix}/chunk_action_delta_boundary_p95', float('nan')):.6f} "
        f"delta_boundary_p99={metrics.get(f'{prefix}/chunk_action_delta_boundary_p99', float('nan')):.6f} "
        f"delta_internal_p95={metrics.get(f'{prefix}/chunk_action_delta_internal_p95', float('nan')):.6f} "
        f"delta_internal_p99={metrics.get(f'{prefix}/chunk_action_delta_internal_p99', float('nan')):.6f} "
        f"d2_boundary={metrics.get(f'{prefix}/chunk_action_d2_boundary_mean', float('nan')):.6f} "
        f"d2_internal={metrics.get(f'{prefix}/chunk_action_d2_internal_mean', float('nan')):.6f} "
        f"d2_ratio={metrics.get(f'{prefix}/chunk_action_d2_boundary_internal_ratio', float('nan')):.4f} "
        f"d2_boundary_p95={metrics.get(f'{prefix}/chunk_action_d2_boundary_p95', float('nan')):.6f} "
        f"d2_boundary_p99={metrics.get(f'{prefix}/chunk_action_d2_boundary_p99', float('nan')):.6f} "
        f"d2_internal_p95={metrics.get(f'{prefix}/chunk_action_d2_internal_p95', float('nan')):.6f} "
        f"d2_internal_p99={metrics.get(f'{prefix}/chunk_action_d2_internal_p99', float('nan')):.6f}",
        flush=True,
    )
    print(
        f"[{label}_CHUNK_STATE] "
        f"joint_vel_boundary={metrics.get(f'{prefix}/chunk_joint_vel_jump_boundary_mean', float('nan')):.6f} "
        f"joint_vel_internal={metrics.get(f'{prefix}/chunk_joint_vel_jump_internal_mean', float('nan')):.6f} "
        f"joint_vel_ratio={metrics.get(f'{prefix}/chunk_joint_vel_jump_boundary_internal_ratio', float('nan')):.4f} "
        f"joint_vel_boundary_p95={metrics.get(f'{prefix}/chunk_joint_vel_jump_boundary_p95', float('nan')):.6f} "
        f"joint_vel_boundary_p99={metrics.get(f'{prefix}/chunk_joint_vel_jump_boundary_p99', float('nan')):.6f} "
        f"joint_vel_internal_p95={metrics.get(f'{prefix}/chunk_joint_vel_jump_internal_p95', float('nan')):.6f} "
        f"joint_vel_internal_p99={metrics.get(f'{prefix}/chunk_joint_vel_jump_internal_p99', float('nan')):.6f} "
        f"root_ang_boundary={metrics.get(f'{prefix}/chunk_root_ang_vel_jump_boundary_mean', float('nan')):.6f} "
        f"root_ang_internal={metrics.get(f'{prefix}/chunk_root_ang_vel_jump_internal_mean', float('nan')):.6f} "
        f"root_ang_ratio={metrics.get(f'{prefix}/chunk_root_ang_vel_jump_boundary_internal_ratio', float('nan')):.4f} "
        f"root_ang_boundary_p95={metrics.get(f'{prefix}/chunk_root_ang_vel_jump_boundary_p95', float('nan')):.6f} "
        f"root_ang_boundary_p99={metrics.get(f'{prefix}/chunk_root_ang_vel_jump_boundary_p99', float('nan')):.6f} "
        f"root_ang_internal_p95={metrics.get(f'{prefix}/chunk_root_ang_vel_jump_internal_p95', float('nan')):.6f} "
        f"root_ang_internal_p99={metrics.get(f'{prefix}/chunk_root_ang_vel_jump_internal_p99', float('nan')):.6f}",
        flush=True,
    )
    print(
        f"[{label}_CHUNK_RESET] "
        f"action_delta={metrics.get(f'{prefix}/chunk_action_delta_reset_first_mean', float('nan')):.6f} "
        f"action_delta_p95={metrics.get(f'{prefix}/chunk_action_delta_reset_first_p95', float('nan')):.6f} "
        f"joint_vel_jump={metrics.get(f'{prefix}/chunk_joint_vel_jump_reset_first_mean', float('nan')):.6f} "
        f"joint_vel_jump_p95={metrics.get(f'{prefix}/chunk_joint_vel_jump_reset_first_p95', float('nan')):.6f} "
        f"root_ang_vel_jump={metrics.get(f'{prefix}/chunk_root_ang_vel_jump_reset_first_mean', float('nan')):.6f} "
        f"root_ang_vel_jump_p95={metrics.get(f'{prefix}/chunk_root_ang_vel_jump_reset_first_p95', float('nan')):.6f} "
        f"joint_pos_error={metrics.get(f'{prefix}/chunk_joint_pos_error_reset_first_mean', float('nan')):.6f} "
        f"joint_vel_error={metrics.get(f'{prefix}/chunk_joint_vel_error_reset_first_mean', float('nan')):.6f} "
        f"root_ang_vel_error={metrics.get(f'{prefix}/chunk_root_ang_vel_error_reset_first_mean', float('nan')):.6f}",
        flush=True,
    )
    for offset in range(4):
        print(
            f"[{label}_OFFSET{offset}] "
            f"joint_pos={metrics.get(f'{prefix}/chunk_offset{offset}_joint_pos_error_mean', float('nan')):.6f} "
            f"joint_pos_p95={metrics.get(f'{prefix}/chunk_offset{offset}_joint_pos_error_p95', float('nan')):.6f} "
            f"joint_vel={metrics.get(f'{prefix}/chunk_offset{offset}_joint_vel_error_mean', float('nan')):.6f} "
            f"joint_vel_p95={metrics.get(f'{prefix}/chunk_offset{offset}_joint_vel_error_p95', float('nan')):.6f} "
            f"root_ang_vel={metrics.get(f'{prefix}/chunk_offset{offset}_root_ang_vel_error_mean', float('nan')):.6f} "
            f"root_ang_vel_p95={metrics.get(f'{prefix}/chunk_offset{offset}_root_ang_vel_error_p95', float('nan')):.6f} "
            f"count={metrics.get(f'{prefix}/chunk_offset{offset}_joint_pos_error_count', 0.0):.0f}",
            flush=True,
        )
