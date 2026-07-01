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

    print(
        f"[{label}_WALL] "
        f"alive@850={metrics.get(f'{prefix}/alive_at_phase_850', float('nan')):.4f} "
        f"motion_complete={metrics.get(f'{prefix}/motion_complete_rate', float('nan')):.4f} "
        f"wrist_fail_825_840={metrics.get(f'{prefix}/wrist_fail_825_840', float('nan')):.4f} "
        f"steps_p50={metrics.get(f'{prefix}/steps_p50', float('nan')):.0f}",
        flush=True,
    )
