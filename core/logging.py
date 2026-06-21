"""Task-level logging shared by all algorithms (mimic tracking + validation).

Algorithm-specific blocks ([POLICY], [OBJECTIVE], [CHUNK_REWARD], ...) live with the
algorithm. These blocks describe the mimic task / env outcome and are reusable.
"""
from __future__ import annotations

from .validation import short_body_name


def log_shared_update_diagnostics(metrics: dict[str, float], *, failure_label: str, index_name: str) -> None:
    print(
        f"[TRAIN] mean_reward={metrics.get('train/mean_reward', float('nan')):.5f} "
        f"mean_len={metrics.get('train/mean_episode_length', float('nan')):.2f} "
        f"recent_eps={metrics.get('train/recent_episode_count', 0.0):.0f} "
        f"completed_eps={metrics.get('train/completed_episodes', 0.0):.0f}",
        flush=True,
    )
    print(
        f"[UPDATE_EFFECT] action_delta={metrics.get('policy/action_delta', float('nan')):.8f} "
        f"param_rms_delta={metrics.get('policy/param_rms_delta', float('nan')):.8f}",
        flush=True,
    )
    print(
        f"[TIME] collect={metrics['timing/collect_s']:.3f}s update={metrics['timing/update_s']:.3f}s",
        flush=True,
    )
    print(
        f"[DONE] timeout={metrics.get('done/time_out_frac', 0.0):.5f} "
        f"anchor_pos={metrics.get('done/anchor_pos_bad_frac', 0.0):.5f} "
        f"anchor_ori={metrics.get('done/anchor_ori_bad_frac', 0.0):.5f} "
        f"ee_body={metrics.get('done/ee_body_bad_frac', 0.0):.5f}",
        flush=True,
    )
    print(
        f"[DONE_ROLLOUT] timeout={metrics.get('done_rollout/time_out_frac', 0.0):.5f} "
        f"anchor_pos={metrics.get('done_rollout/anchor_pos_bad_frac', 0.0):.5f} "
        f"anchor_ori={metrics.get('done_rollout/anchor_ori_bad_frac', 0.0):.5f} "
        f"ee_body={metrics.get('done_rollout/ee_body_bad_frac', 0.0):.5f}",
        flush=True,
    )
    print(
        f"[{failure_label}] "
        f"{index_name}_mean={metrics.get('rollout/first_failure_chunk_mean', float('nan')):.2f} "
        f"{index_name}_min={metrics.get('rollout/first_failure_chunk_min', float('nan')):.0f} "
        f"{index_name}_max={metrics.get('rollout/first_failure_chunk_max', float('nan')):.0f} "
        f"phase_mean={metrics.get('rollout/first_failure_phase_mean', float('nan')):.2f} "
        f"phase_min={metrics.get('rollout/first_failure_phase_min', float('nan')):.0f} "
        f"phase_max={metrics.get('rollout/first_failure_phase_max', float('nan')):.0f} "
        f"anchor_pos={metrics.get('rollout/first_failure_anchor_pos_frac', 0.0):.5f} "
        f"anchor_ori={metrics.get('rollout/first_failure_anchor_ori_frac', 0.0):.5f} "
        f"ee_body={metrics.get('rollout/first_failure_ee_body_frac', 0.0):.5f} "
        f"timeout={metrics.get('rollout/first_failure_timeout_frac', 0.0):.5f}",
        flush=True,
    )


def log_shared_tracking(metrics: dict[str, float]) -> None:
    print(
        f"[TRACK_CHUNK0] "
        f"anchor_pos={metrics.get('reward/anchor_pos_reward_mean', float('nan')):.5f} "
        f"anchor_ori={metrics.get('reward/anchor_ori_reward_mean', float('nan')):.5f} "
        f"body_pos={metrics.get('reward/body_pos_reward_mean', float('nan')):.5f} "
        f"body_ori={metrics.get('reward/body_ori_reward_mean', float('nan')):.5f} "
        f"body_lin={metrics.get('reward/body_lin_vel_reward_mean', float('nan')):.5f} "
        f"body_ang={metrics.get('reward/body_ang_vel_reward_mean', float('nan')):.5f}",
        flush=True,
    )
    print(
        f"[TRACK_ROLLOUT] "
        f"anchor_pos={metrics.get('reward_rollout/anchor_pos_reward_mean', float('nan')):.5f} "
        f"anchor_ori={metrics.get('reward_rollout/anchor_ori_reward_mean', float('nan')):.5f} "
        f"body_pos={metrics.get('reward_rollout/body_pos_reward_mean', float('nan')):.5f} "
        f"body_ori={metrics.get('reward_rollout/body_ori_reward_mean', float('nan')):.5f} "
        f"body_lin={metrics.get('reward_rollout/body_lin_vel_reward_mean', float('nan')):.5f} "
        f"body_ang={metrics.get('reward_rollout/body_ang_vel_reward_mean', float('nan')):.5f}",
        flush=True,
    )
    print(
        f"[TRAIN_COST] "
        f"action_rate={metrics.get('reward/action_rate_mean', float('nan')):.5f} "
        f"action_accel={metrics.get('reward/action_accel_mean', float('nan')):.5f} "
        f"action_l2={metrics.get('reward/action_l2_mean', float('nan')):.5f} "
        f"joint_torque={metrics.get('reward/joint_torque_mean', float('nan')):.5f} "
        f"joint_limit={metrics.get('reward/joint_limit_mean', float('nan')):.5f} "
        f"undesired_contacts={metrics.get('reward/undesired_contacts_mean', float('nan')):.5f}",
        flush=True,
    )
    print(
        f"[REWARD_WEIGHTED] "
        f"pos={metrics.get('reward_weighted/positive', float('nan')):.5f} "
        f"penalty={metrics.get('reward_weighted/penalty', float('nan')):.5f} "
        f"total={metrics.get('reward_weighted/total', float('nan')):.5f} "
        f"act_rate={metrics.get('reward_weighted/action_rate', float('nan')):.5f} "
        f"act_accel={metrics.get('reward_weighted/action_accel', float('nan')):.5f} "
        f"act_l2={metrics.get('reward_weighted/action_l2', float('nan')):.5f} "
        f"torque={metrics.get('reward_weighted/joint_torque', float('nan')):.5f} "
        f"contacts={metrics.get('reward_weighted/undesired_contacts', float('nan')):.5f}",
        flush=True,
    )
    print(
        f"[ACT_SUMMARY] "
        f"abs_mean={metrics.get('act/abs_mean', float('nan')):.4f} "
        f"abs_p95={metrics.get('act/abs_p95', float('nan')):.4f} "
        f"abs_p99={metrics.get('act/abs_p99', float('nan')):.4f} "
        f"abs_max={metrics.get('act/abs_max', float('nan')):.4f} "
        f"abs_max_all={metrics.get('act/abs_max_all', float('nan')):.4f} "
        f"legs={metrics.get('act/legs_abs', float('nan')):.4f} "
        f"waist={metrics.get('act/waist_abs', float('nan')):.4f} "
        f"arms={metrics.get('act/arms_abs', float('nan')):.4f} "
        f"first={metrics.get('act/first_abs_mean', float('nan')):.4f} "
        f"last={metrics.get('act/last_abs_mean', float('nan')):.4f} "
        f"latent_abs={metrics.get('latent/final_abs_mean', float('nan')):.4f} "
        f"latent_max={metrics.get('latent/final_abs_max', float('nan')):.4f}",
        flush=True,
    )
    print(
        f"[FRAME_DIAG] "
        f"frame0_abs={metrics.get('act/frame0_abs_mean', float('nan')):.4f} "
        f"frame1_abs={metrics.get('act/frame1_abs_mean', float('nan')):.4f} "
        f"in_chunk_delta={metrics.get('act/in_chunk_delta_abs', float('nan')):.4f} "
        f"cross_chunk_delta={metrics.get('act/cross_chunk_delta_abs', float('nan')):.4f} "
        f"frame0_reward={metrics.get('reward/frame0_mean', float('nan')):.5f} "
        f"frame1_reward={metrics.get('reward/frame1_mean', float('nan')):.5f}",
        flush=True,
    )
    print(
        f"[TRAIN_BODY] "
        f"torso_ori={metrics.get('reward/diag_torso_ori_deg_mean', float('nan')):.2f}deg "
        f"l_wrist_ori={metrics.get('reward/diag_left_wrist_ori_deg_mean', float('nan')):.2f}deg "
        f"r_wrist_ori={metrics.get('reward/diag_right_wrist_ori_deg_mean', float('nan')):.2f}deg "
        f"l_elbow_ori={metrics.get('reward/diag_left_elbow_ori_deg_mean', float('nan')):.2f}deg "
        f"r_elbow_ori={metrics.get('reward/diag_right_elbow_ori_deg_mean', float('nan')):.2f}deg "
        f"l_shoulder_ori={metrics.get('reward/diag_left_shoulder_ori_deg_mean', float('nan')):.2f}deg "
        f"r_shoulder_ori={metrics.get('reward/diag_right_shoulder_ori_deg_mean', float('nan')):.2f}deg",
        flush=True,
    )
    print(
        f"[TRAIN_ANG] "
        f"torso={metrics.get('reward/diag_torso_ang_vel_mean', float('nan')):.3f} "
        f"l_wrist={metrics.get('reward/diag_left_wrist_ang_vel_mean', float('nan')):.3f} "
        f"r_wrist={metrics.get('reward/diag_right_wrist_ang_vel_mean', float('nan')):.3f} "
        f"l_elbow={metrics.get('reward/diag_left_elbow_ang_vel_mean', float('nan')):.3f} "
        f"r_elbow={metrics.get('reward/diag_right_elbow_ang_vel_mean', float('nan')):.3f} "
        f"l_shoulder={metrics.get('reward/diag_left_shoulder_ang_vel_mean', float('nan')):.3f} "
        f"r_shoulder={metrics.get('reward/diag_right_shoulder_ang_vel_mean', float('nan')):.3f}",
        flush=True,
    )
    print(
        f"[TRAIN_ACT] "
        f"l_wrist_r={metrics.get('act/l_wrist_roll', float('nan')):.4f} "
        f"l_wrist_p={metrics.get('act/l_wrist_pitch', float('nan')):.4f} "
        f"l_wrist_y={metrics.get('act/l_wrist_yaw', float('nan')):.4f} "
        f"r_wrist_r={metrics.get('act/r_wrist_roll', float('nan')):.4f} "
        f"r_wrist_p={metrics.get('act/r_wrist_pitch', float('nan')):.4f} "
        f"r_wrist_y={metrics.get('act/r_wrist_yaw', float('nan')):.4f} "
        f"l_elbow={metrics.get('act/l_elbow', float('nan')):.4f} "
        f"r_elbow={metrics.get('act/r_elbow', float('nan')):.4f}",
        flush=True,
    )


def log_validation_metrics(env, metrics: dict[str, float]) -> None:
    if "validation/steps_mean" in metrics:
        _log_validation_block(env, "VAL", "validation", metrics)
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
