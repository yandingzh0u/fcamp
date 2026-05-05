from __future__ import annotations

from .validation import _short_body_name


class LoggingMixin:
    def _log_update(self, update_idx: int, metrics: dict[str, float]) -> None:
        print(
            f"[UPDATE] {update_idx}/{self.cfg.max_updates} "
            f"group_reward={metrics['group/reward_mean']:.5f} "
            f"group_std={metrics['group/reward_std']:.5f} "
            f"rollout_ret={metrics['rollout/chunk_return_mean']:.5f}",
            flush=True,
        )
        print(
            f"[POLICY] loss={metrics['policy/loss']:.5f} "
            f"policy_loss={metrics['policy/policy_loss']:.5f} "
            f"latent_reg={metrics.get('policy/latent_reg_loss', 0.0):.5f} "
            f"sat={metrics.get('policy/action_sat_loss', 0.0):.5f} "
            f"ratio={metrics['policy/ratio']:.4f} "
            f"clip_frac={metrics['policy/clip_frac']:.4f} "
            f"grad={metrics['policy/grad_norm']:.5f}",
            flush=True,
        )
        print(
            f"[POLICY_DETAIL] "
            f"ratio_min={metrics.get('policy/ratio_min', float('nan')):.4f} "
            f"ratio_max={metrics.get('policy/ratio_max', float('nan')):.4f} "
            f"logp_delta_abs={metrics.get('policy/logprob_delta_abs', float('nan')):.5f} "
            f"adv_abs={metrics.get('group/advantage_abs_mean', float('nan')):.5f} "
            f"valid={metrics.get('rollout/valid_frac', float('nan')):.5f}",
            flush=True,
        )
        print(
            f"[UPDATE_EFFECT] "
            f"action_delta={metrics.get('policy/action_delta', float('nan')):.8f} "
            f"param_rms_delta={metrics.get('policy/param_rms_delta', float('nan')):.8f}",
            flush=True,
        )
        print(
            f"[TRACK] "
            f"anchor_pos={metrics.get('reward/anchor_pos_reward_mean', float('nan')):.5f} "
            f"anchor_ori={metrics.get('reward/anchor_ori_reward_mean', float('nan')):.5f} "
            f"body_pos={metrics.get('reward/body_pos_reward_mean', float('nan')):.5f} "
            f"body_ori={metrics.get('reward/body_ori_reward_mean', float('nan')):.5f} "
            f"body_lin={metrics.get('reward/body_lin_vel_reward_mean', float('nan')):.5f} "
            f"body_ang={metrics.get('reward/body_ang_vel_reward_mean', float('nan')):.5f}",
            flush=True,
        )
        print(
            f"[TIME] collect={metrics['timing/collect_s']:.3f}s "
            f"update={metrics['timing/update_s']:.3f}s",
            flush=True,
        )
        print(
            f"[TRAIN_COST] "
            f"action_rate={metrics.get('reward/action_rate_mean', float('nan')):.5f} "
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
            f"torque={metrics.get('reward_weighted/joint_torque', float('nan')):.5f} "
            f"contacts={metrics.get('reward_weighted/undesired_contacts', float('nan')):.5f}",
            flush=True,
        )
        print(
            f"[ACT_SUMMARY] "
            f"abs_mean={metrics.get('act/abs_mean', float('nan')):.4f} "
            f"abs_p95={metrics.get('act/abs_p95', float('nan')):.4f} "
            f"abs_max={metrics.get('act/abs_max', float('nan')):.4f} "
            f"legs={metrics.get('act/legs_abs', float('nan')):.4f} "
            f"waist={metrics.get('act/waist_abs', float('nan')):.4f} "
            f"arms={metrics.get('act/arms_abs', float('nan')):.4f} "
            f"latent_abs={metrics.get('latent/final_abs_mean', float('nan')):.4f} "
            f"latent_max={metrics.get('latent/final_abs_max', float('nan')):.4f}",
            flush=True,
        )
        print(
            f"[DONE] "
            f"timeout={metrics.get('done/time_out_frac', 0.0):.5f} "
            f"anchor_pos={metrics.get('done/anchor_pos_bad_frac', 0.0):.5f} "
            f"anchor_ori={metrics.get('done/anchor_ori_bad_frac', 0.0):.5f} "
            f"ee_body={metrics.get('done/ee_body_bad_frac', 0.0):.5f}",
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
        if "validation/steps_mean" in metrics:
            self._log_validation_block("VAL", "validation", metrics)
        if "val_fixed/steps_mean" in metrics:
            self._log_validation_block("VAL_FIXED", "val_fixed", metrics)

    def _log_validation_block(self, label: str, prefix: str, metrics: dict[str, float]) -> None:
        print(
            f"[{label}] steps_mean={metrics[f'{prefix}/steps_mean']:.2f} "
            f"steps_min={metrics.get(f'{prefix}/steps_min', float('nan')):.0f} "
            f"steps_max={metrics.get(f'{prefix}/steps_max', float('nan')):.0f} "
            f"return={metrics[f'{prefix}/return_mean']:.5f} "
            f"done={metrics.get(f'{prefix}/done_frac', float('nan')):.5f}",
            flush=True,
        )
        ee_parts: list[str] = []
        for body_name in self.env.ee_body_names:
            short_name = _short_body_name(body_name)
            bad_key = f"{prefix}/ee_{short_name}_bad_frac"
            err_key = f"{prefix}/ee_{short_name}_z_error"
            if bad_key in metrics or err_key in metrics:
                ee_parts.append(
                    f"{short_name}_bad={metrics.get(bad_key, float('nan')):.5f} "
                    f"{short_name}_z={metrics.get(err_key, float('nan')):.5f}"
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
