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
    ee_parts: list[str] = []
    for body_name in env.ee_body_names:
        sname = short_body_name(body_name)
        bad_key = f"{prefix}/ee_{sname}_bad_frac"
        err_key = f"{prefix}/ee_{sname}_z_error"
        if bad_key in metrics or err_key in metrics:
            ee_parts.append(
                f"{sname}_bad={metrics.get(bad_key, float('nan')):.5f} "
                f"{sname}_z={metrics.get(err_key, float('nan')):.5f}"
            )
    if ee_parts:
        print(f"[{label}_EE] " + " ".join(ee_parts), flush=True)
    print(
        f"[{label}_CAUSE] "
        f"anchor_pos_bad={metrics.get(f'{prefix}/anchor_pos_bad_frac', float('nan')):.5f} "
        f"anchor_ori_bad={metrics.get(f'{prefix}/anchor_ori_bad_frac', float('nan')):.5f} "
        f"ee_body_bad={metrics.get(f'{prefix}/ee_body_bad_frac', float('nan')):.5f} "
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
    reset_pos_idx = int(metrics.get(f"{prefix}/reset_body_pos_top_index", -1.0))
    reset_ori_idx = int(metrics.get(f"{prefix}/reset_body_ori_top_index", -1.0))
    reset_pos_body = (
        short_body_name(env.track_body_names[reset_pos_idx])
        if 0 <= reset_pos_idx < len(env.track_body_names)
        else "n/a"
    )
    reset_ori_body = (
        short_body_name(env.track_body_names[reset_ori_idx])
        if 0 <= reset_ori_idx < len(env.track_body_names)
        else "n/a"
    )
    print(
        f"[{label}_RESET] "
        f"root_pos={metrics.get(f'{prefix}/reset_root_pos_err', float('nan')):.5f} "
        f"root_ori={metrics.get(f'{prefix}/reset_root_ori_deg', float('nan')):.3f}deg "
        f"anchor_pos={metrics.get(f'{prefix}/reset_anchor_pos_err', float('nan')):.5f} "
        f"anchor_ori={metrics.get(f'{prefix}/reset_anchor_ori_deg', float('nan')):.3f}deg "
        f"joint_pos={metrics.get(f'{prefix}/reset_joint_pos_err', float('nan')):.5f} "
        f"body_pos={metrics.get(f'{prefix}/reset_body_pos_err', float('nan')):.5f} "
        f"body_pos_top={reset_pos_body}:{metrics.get(f'{prefix}/reset_body_pos_max', float('nan')):.5f} "
        f"body_ori={metrics.get(f'{prefix}/reset_body_ori_deg', float('nan')):.3f}deg "
        f"body_ori_top={reset_ori_body}:{metrics.get(f'{prefix}/reset_body_ori_max', float('nan')):.3f}deg",
        flush=True,
    )
    contact_parts: list[str] = []
    contact_ids = getattr(env, "amp_undesired_contact_body_ids", None)
    if contact_ids is not None:
        contact_ids_list = contact_ids.detach().cpu().tolist()
        for rank in range(1, 4):
            idx = int(metrics.get(f"{prefix}/contact_top{rank}_index", -1.0))
            if idx < 0 or idx >= len(contact_ids_list):
                continue
            body_id = int(contact_ids_list[idx])
            body_name = short_body_name(env.contact_sensor.body_names[body_id])
            contact_parts.append(
                f"{body_name}_frac={metrics.get(f'{prefix}/contact_top{rank}_frac', float('nan')):.5f} "
                f"{body_name}_force={metrics.get(f'{prefix}/contact_top{rank}_force', float('nan')):.3f}"
            )
    if contact_parts:
        print(f"[{label}_CONTACT] " + " ".join(contact_parts), flush=True)

    print(
        f"[{label}_WALL] "
        f"alive@850={metrics.get(f'{prefix}/alive_at_phase_850', float('nan')):.4f} "
        f"motion_complete={metrics.get(f'{prefix}/motion_complete_rate', float('nan')):.4f} "
        f"wrist_fail_825_840={metrics.get(f'{prefix}/wrist_fail_825_840', float('nan')):.4f} "
        f"steps_p50={metrics.get(f'{prefix}/steps_p50', float('nan')):.0f}",
        flush=True,
    )
    print(
        f"[{label}_PUSH] "
        f"push_applied={metrics.get(f'{prefix}/push_applied_frac', float('nan')):.4f} "
        f"died_before_push={metrics.get(f'{prefix}/died_before_push_frac', float('nan')):.4f} "
        f"pushed_then_died={metrics.get(f'{prefix}/pushed_then_died_frac', float('nan')):.4f} "
        f"first_push_step={metrics.get(f'{prefix}/first_push_step_mean', float('nan')):.1f}",
        flush=True,
    )
