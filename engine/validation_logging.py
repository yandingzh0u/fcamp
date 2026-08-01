from __future__ import annotations


_MISSING_METRIC = -1.0


def _value(
    metrics: dict[str, float],
    prefix: str,
    name: str,
) -> float:
    return float(metrics.get(f"{prefix}/{name}", _MISSING_METRIC))


def log_validation_metrics(
    env,
    metrics: dict[str, float],
) -> None:
    del env
    if "validation/steps_mean" in metrics:
        _log_validation_block("VAL", "validation", metrics)
    if "validation_directional/steps_mean" in metrics:
        _log_validation_block(
            "VAL_DIR",
            "validation_directional",
            metrics,
        )


def _log_validation_block(
    label: str,
    prefix: str,
    metrics: dict[str, float],
) -> None:
    print(
        f"[{label}] "
        f"steps_mean={_value(metrics, prefix, 'steps_mean'):.2f} "
        f"steps_min={_value(metrics, prefix, 'steps_min'):.0f} "
        f"steps_p50={_value(metrics, prefix, 'steps_p50'):.0f} "
        f"steps_p95={_value(metrics, prefix, 'steps_p95'):.0f} "
        f"steps_max={_value(metrics, prefix, 'steps_max'):.0f} "
        f"return={_value(metrics, prefix, 'return_mean'):.5f} "
        f"done={_value(metrics, prefix, 'done_frac'):.5f}",
        flush=True,
    )
    print(
        f"[{label}_OUTCOME] "
        f"failure={_value(metrics, prefix, 'failure_frac'):.5f} "
        f"motion_complete={_value(metrics, prefix, 'motion_complete_frac'):.5f} "
        f"time_out={_value(metrics, prefix, 'time_out_frac'):.5f} "
        f"censored={_value(metrics, prefix, 'censored_frac'):.5f} "
        f"reference_progress={_value(metrics, prefix, 'reference_progress_mean'):.5f}",
        flush=True,
    )
    print(
        f"[{label}_CAUSE] "
        f"anchor_pos_bad={_value(metrics, prefix, 'anchor_pos_bad_frac'):.5f} "
        f"anchor_ori_bad={_value(metrics, prefix, 'anchor_ori_bad_frac'):.5f} "
        f"ee_body_bad={_value(metrics, prefix, 'ee_body_bad_frac'):.5f} "
        f"failure_phase_p50="
        f"{_value(metrics, prefix, 'terminal/failure/phase_p50'):.1f} "
        f"completion_phase_p50="
        f"{_value(metrics, prefix, 'terminal/motion_complete/phase_p50'):.1f}",
        flush=True,
    )
    print(
        f"[{label}_RESET] "
        f"root_pos={_value(metrics, prefix, 'reset_root_pos_err'):.6f} "
        f"root_ori={_value(metrics, prefix, 'reset_root_ori_deg'):.3f}deg "
        f"anchor_pos={_value(metrics, prefix, 'reset_anchor_pos_err'):.6f} "
        f"anchor_ori={_value(metrics, prefix, 'reset_anchor_ori_deg'):.3f}deg "
        f"joint_pos={_value(metrics, prefix, 'reset_joint_pos_err'):.6f} "
        f"joint_vel={_value(metrics, prefix, 'reset_joint_vel_err'):.6f} "
        f"body_pos={_value(metrics, prefix, 'reset_body_pos_err'):.6f} "
        f"body_ori={_value(metrics, prefix, 'reset_body_ori_deg'):.3f}deg",
        flush=True,
    )
    print(
        f"[{label}_TRACKING] "
        f"joint_pos_mae="
        f"{_value(metrics, prefix, 'tracking/joint_pos_mae/mean'):.6f} "
        f"joint_vel_mae="
        f"{_value(metrics, prefix, 'tracking/joint_vel_mae/mean'):.6f} "
        f"root_lin_mae="
        f"{_value(metrics, prefix, 'tracking/root_lin_vel_mae/mean'):.6f} "
        f"root_ang_mae="
        f"{_value(metrics, prefix, 'tracking/root_ang_vel_mae/mean'):.6f} "
        f"pd_target_ref_now="
        f"{_value(metrics, prefix, 'pd_target_vs_reference_joint_mae_now'):.6f} "
        f"pd_target_ref_next="
        f"{_value(metrics, prefix, 'pd_target_vs_reference_joint_mae_next'):.6f} "
        f"torso_ori={_value(metrics, prefix, 'diag_torso_ori_deg'):.3f}deg "
        f"left_wrist_ori="
        f"{_value(metrics, prefix, 'diag_left_wrist_ori_deg'):.3f}deg "
        f"right_wrist_ori="
        f"{_value(metrics, prefix, 'diag_right_wrist_ori_deg'):.3f}deg",
        flush=True,
    )
    print(
        f"[{label}_DYNAMICS] "
        f"action_delta="
        f"{_value(metrics, prefix, 'dynamics/action_delta/mean'):.6f} "
        f"action_delta_p95="
        f"{_value(metrics, prefix, 'dynamics/action_delta/p95'):.6f} "
        f"action_d2={_value(metrics, prefix, 'dynamics/action_d2/mean'):.6f} "
        f"action_d2_p95="
        f"{_value(metrics, prefix, 'dynamics/action_d2/p95'):.6f} "
        f"joint_vel_jump="
        f"{_value(metrics, prefix, 'dynamics/joint_vel_jump/mean'):.6f} "
        f"root_lin_jump="
        f"{_value(metrics, prefix, 'dynamics/root_lin_vel_jump/mean'):.6f} "
        f"root_ang_jump="
        f"{_value(metrics, prefix, 'dynamics/root_ang_vel_jump/mean'):.6f} "
        f"initial_action_delta="
        f"{_value(metrics, prefix, 'initial/action_delta/mean'):.6f}",
        flush=True,
    )
