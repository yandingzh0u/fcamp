from __future__ import annotations

import torch


def _short_body_name(body_name: str) -> str:
    short_name = body_name.removesuffix("_link")
    for suffix in ("_yaw", "_roll"):
        if short_name.endswith(suffix):
            short_name = short_name[: -len(suffix)]
    return short_name


class ValidationMixin:
    def run_validation_rollout(self, fixed_seed: int | None = None) -> dict[str, float]:
        del fixed_seed
        was_training = self.policy.training
        self.policy.eval()

        validation_phase = torch.full(
            (self.cfg.num_envs,),
            max(0, self.cfg.validation_start_phase),
            dtype=torch.long,
            device=self.env.device,
        )
        current_obs = self.env.reset(phase_indices=validation_phase)
        cached_chunk: torch.Tensor | None = None
        chunk_index = self.cfg.horizon
        done = torch.zeros(self.cfg.num_envs, dtype=torch.bool, device=self.env.device)
        survived_steps = torch.zeros(self.cfg.num_envs, dtype=torch.long, device=self.env.device)
        cumulative_reward = torch.zeros(self.cfg.num_envs, device=self.env.device)
        done_term_record = {
            "time_out": torch.zeros(self.cfg.num_envs, dtype=torch.bool, device=self.env.device),
            "anchor_pos_bad": torch.zeros(self.cfg.num_envs, dtype=torch.bool, device=self.env.device),
            "anchor_ori_bad": torch.zeros(self.cfg.num_envs, dtype=torch.bool, device=self.env.device),
            "ee_body_bad": torch.zeros(self.cfg.num_envs, dtype=torch.bool, device=self.env.device),
        }
        done_debug_record = {
            "ee_z_error_max": torch.zeros(self.cfg.num_envs, device=self.env.device),
            "ee_z_error_mean": torch.zeros(self.cfg.num_envs, device=self.env.device),
            "anchor_z_error": torch.zeros(self.cfg.num_envs, device=self.env.device),
            "anchor_gravity_z_error": torch.zeros(self.cfg.num_envs, device=self.env.device),
        }
        ee_body_count = len(self.env.ee_body_names)
        done_ee_z_error_record = torch.zeros(self.cfg.num_envs, ee_body_count, device=self.env.device)
        done_ee_bad_record = torch.zeros(self.cfg.num_envs, ee_body_count, dtype=torch.bool, device=self.env.device)
        diag_keys = [
            "diag_torso_ori_deg", "diag_left_wrist_ori_deg", "diag_right_wrist_ori_deg",
            "diag_left_elbow_ori_deg", "diag_right_elbow_ori_deg",
            "diag_left_shoulder_ori_deg", "diag_right_shoulder_ori_deg",
            "diag_torso_ang_vel", "diag_left_wrist_ang_vel", "diag_right_wrist_ang_vel",
            "diag_left_elbow_ang_vel", "diag_right_elbow_ang_vel",
            "diag_left_shoulder_ang_vel", "diag_right_shoulder_ang_vel",
        ]
        diag_accum = {key: torch.zeros(self.cfg.num_envs, device=self.env.device) for key in diag_keys}
        diag_steps = torch.zeros(self.cfg.num_envs, device=self.env.device)

        try:
            with torch.no_grad():
                for _ in range(self.cfg.validation_max_steps):
                    if not self.simulation_app.is_running():
                        break
                    if cached_chunk is None or chunk_index >= self.cfg.horizon:
                        noise = torch.zeros(self.cfg.num_envs, self.chunk_dim, device=self.env.device)
                        cached_chunk = self.policy(
                            current_obs,
                            noise,
                            steps=self.cfg.flow_steps,
                        )
                        chunk_index = 0

                    action = cached_chunk[:, chunk_index, :]
                    if bool(done.any()):
                        action = torch.where(done.unsqueeze(-1), torch.zeros_like(action), action)
                    chunk_index += 1

                    current_obs, reward, step_done, info = self.env.step(action, auto_reset=True)
                    active_mask = ~done
                    new_done = active_mask & step_done
                    if bool(new_done.any()):
                        done_terms = info["done_terms"]
                        debug_terms = info["debug_terms"]
                        for name in done_term_record:
                            done_term_record[name][new_done] = done_terms[name][new_done]
                        for name in done_debug_record:
                            done_debug_record[name][new_done] = debug_terms[name][new_done]
                        ee_z_error_by_body = debug_terms["ee_z_error_by_body"][new_done]
                        done_ee_z_error_record[new_done] = ee_z_error_by_body
                        done_ee_bad_record[new_done] = ee_z_error_by_body > 0.25
                    survived_steps += active_mask.to(dtype=torch.long)
                    cumulative_reward += active_mask.float() * reward
                    reward_terms = info.get("reward_terms", {})
                    active_f = active_mask.float()
                    for key in diag_keys:
                        if key in reward_terms:
                            diag_accum[key] += active_f * reward_terms[key]
                    diag_steps += active_f
                    done |= step_done
                    if bool(done.all()):
                        break
        finally:
            self.current_observation = self._reset_training_envs()
            if was_training:
                self.policy.train()

        metrics: dict[str, float] = {
            "validation/steps_mean": float(survived_steps.float().mean().item()),
            "validation/steps_min": float(survived_steps.min().item()),
            "validation/steps_max": float(survived_steps.max().item()),
            "validation/return_mean": float(cumulative_reward.mean().item()),
            "validation/done_frac": float(done.float().mean().item()),
        }
        if bool(done.any()):
            metrics.update({
                "validation/time_out_frac": float(done_term_record["time_out"].float().mean().item()),
                "validation/anchor_pos_bad_frac": float(done_term_record["anchor_pos_bad"].float().mean().item()),
                "validation/anchor_ori_bad_frac": float(done_term_record["anchor_ori_bad"].float().mean().item()),
                "validation/ee_body_bad_frac": float(done_term_record["ee_body_bad"].float().mean().item()),
                "validation/ee_z_max": float(done_debug_record["ee_z_error_max"][done].mean().item()),
                "validation/ee_z_mean": float(done_debug_record["ee_z_error_mean"][done].mean().item()),
                "validation/anchor_z": float(done_debug_record["anchor_z_error"][done].mean().item()),
                "validation/anchor_gravity": float(done_debug_record["anchor_gravity_z_error"][done].mean().item()),
            })
            for index, body_name in enumerate(self.env.ee_body_names):
                short_name = _short_body_name(body_name)
                metrics[f"validation/ee_{short_name}_bad_frac"] = float(
                    done_ee_bad_record[done, index].float().mean().item()
                )
                metrics[f"validation/ee_{short_name}_z_error"] = float(
                    done_ee_z_error_record[done, index].mean().item()
                )

        safe_steps = diag_steps.clamp(min=1.0)
        for key in diag_keys:
            metrics[f"validation/{key}"] = float((diag_accum[key] / safe_steps).mean().item())
        return metrics
