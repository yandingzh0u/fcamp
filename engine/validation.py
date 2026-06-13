from __future__ import annotations

import time

import torch

from env.config import EE_Z_TERMINATION_THRESHOLD

def _short_body_name(body_name: str) -> str:
    short_name = body_name.removesuffix("_link")
    for suffix in ("_yaw", "_roll"):
        if short_name.endswith(suffix):
            short_name = short_name[: -len(suffix)]
    return short_name


class ValidationMixin:
    def _validation_max_steps(self) -> int:
        validation_max_steps = int(self.cfg.validation_max_steps)
        target_steps = int(getattr(self.cfg, "target_validation_steps", 0))
        max_episode_steps = int(getattr(self.cfg, "max_episode_steps", validation_max_steps))
        # The episode can never outlive the motion clip (motion-end is a terminal), so there is
        # no point looping validation past it. Cap the validation horizon at the clip length
        # unless the user explicitly wants to probe a longer survival target.
        if max_episode_steps > 0:
            validation_max_steps = min(validation_max_steps, max_episode_steps)
        if 0 < target_steps:
            validation_max_steps = max(validation_max_steps, target_steps + 1)
        return max(1, validation_max_steps)

    def run_validation_rollout(self, fixed_seed: int | None = None) -> dict[str, float]:
        was_training = self.policy.training
        self.policy.eval()
        preserve_state = bool(getattr(self.cfg, "validation_preserve_state", False))
        training_snapshot = (
            self._snapshot_env_state()
            if preserve_state and hasattr(self, "_snapshot_env_state")
            else None
        )
        training_observation = getattr(self, "current_observation", None)
        env_device = torch.device(self.env.device)
        cpu_rng_state = torch.random.get_rng_state()
        cuda_rng_state = None
        original_observation_noise = getattr(self.env.task_cfg, "observation_noise", True)
        self.env.task_cfg.observation_noise = bool(getattr(self.cfg, "validation_observation_noise", False))
        if torch.cuda.is_available() and env_device.type == "cuda":
            cuda_rng_state = torch.cuda.get_rng_state(env_device)
        if fixed_seed is not None and torch.cuda.is_available() and env_device.type == "cuda":
            torch.cuda.manual_seed_all(fixed_seed)
        if fixed_seed is not None:
            torch.manual_seed(fixed_seed)

        validation_phase = torch.full(
            (self.cfg.num_envs,),
            max(0, self.cfg.validation_start_phase),
            dtype=torch.long,
            device=self.env.device,
        )
        reset_t0 = time.perf_counter()
        print("[VALIDATION_RESET_START]", flush=True)
        current_obs = self.env.reset(phase_indices=validation_phase)
        print(f"[VALIDATION_RESET_DONE] time={time.perf_counter() - reset_t0:.3f}s", flush=True)
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

        validation_max_steps = self._validation_max_steps()

        try:
            rollout_t0 = time.perf_counter()
            with torch.no_grad():
                for step_idx in range(validation_max_steps):
                    if not self.simulation_app.is_running():
                        break
                    if cached_chunk is None or chunk_index >= self.cfg.horizon:
                        if hasattr(self, "_deterministic_actions"):
                            cached_chunk = self._deterministic_actions(current_obs)
                        else:
                            from .mixgrpo.inference import deterministic_sde_ode_actions

                            cached_chunk = deterministic_sde_ode_actions(
                                self.policy,
                                current_obs,
                                steps=self.cfg.flow_steps,
                                sde_eta=getattr(self.cfg, "sde_eta", 0.7),
                            )
                        chunk_index = 0

                    action = cached_chunk[:, chunk_index, :]
                    if bool(done.any()):
                        action = torch.where(done.unsqueeze(-1), torch.zeros_like(action), action)
                    chunk_index += 1

                    current_obs, reward, step_done, info = self.env.step(action, auto_reset=False)
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
                        done_ee_bad_record[new_done] = ee_z_error_by_body > EE_Z_TERMINATION_THRESHOLD
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
                    if (
                        step_idx == 0
                        or (step_idx + 1) % 50 == 0
                        or step_idx + 1 == validation_max_steps
                        or bool(done.all())
                    ):
                        print(
                            f"[VALIDATION_PROGRESS] step={step_idx + 1}/{validation_max_steps} "
                            f"done={done.float().mean().item():.5f} "
                            f"alive={(~done).float().mean().item():.5f} "
                            f"elapsed={time.perf_counter() - rollout_t0:.3f}s",
                            flush=True,
                        )
                    # Early exit: once the overwhelming majority of envs have terminated, the
                    # remaining survivors add wall-time for negligible statistical value
                    # (deterministic envs die in near-lockstep). Stops the long thin tail.
                    if done_frac >= float(getattr(self.cfg, "validation_done_frac_early_stop", 0.98)):
                        print(
                            f"[VALIDATION_EARLY_STOP] step={step_idx + 1} done_frac={done_frac:.4f}",
                            flush=True,
                        )
                        break
                    if bool(done.all()):
                        break
        finally:
            self.env.task_cfg.observation_noise = original_observation_noise
            if training_snapshot is not None and hasattr(self, "_restore_env_state"):
                self._restore_env_state(training_snapshot)
                if training_observation is not None:
                    self.current_observation = training_observation
                else:
                    self.current_observation = self.env.get_observation()
            else:
                if hasattr(self.env, "get_observation"):
                    self.current_observation = self.env.get_observation()
                elif training_observation is not None:
                    self.current_observation = training_observation
            torch.random.set_rng_state(cpu_rng_state)
            if cuda_rng_state is not None:
                torch.cuda.set_rng_state(cuda_rng_state, env_device)
            if was_training:
                self.policy.train()

        metrics: dict[str, float] = {
            "validation/steps_mean": float(survived_steps.float().mean().item()),
            "validation/steps_min": float(survived_steps.min().item()),
            "validation/steps_p50": float(torch.quantile(survived_steps.float(), 0.50).item()),
            "validation/steps_p95": float(torch.quantile(survived_steps.float(), 0.95).item()),
            "validation/steps_max": float(survived_steps.max().item()),
            "validation/return_mean": float(cumulative_reward.mean().item()),
            "validation/done_frac": float(done.float().mean().item()),
        }
        if bool(done.any()):
            failed_phases = self.cfg.validation_start_phase + survived_steps[done]
            metrics.update({
                "validation/fail_phase_mean": float(failed_phases.float().mean().item()),
                "validation/fail_phase_min": float(failed_phases.min().item()),
                "validation/fail_phase_max": float(failed_phases.max().item()),
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
