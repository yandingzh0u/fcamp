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
