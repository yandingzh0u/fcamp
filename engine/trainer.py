from __future__ import annotations

import time
from pathlib import Path

import torch

from env import G1MimicEnv, MimicEnvConfig
from net import FlowMatchingPolicy
from .checkpoint import CheckpointMixin
from .config import MixGRPOConfig
from .logging import LoggingMixin
from .sampling import flow_grpo_step
from .validation import ValidationMixin


class MixGRPOTrainer(ValidationMixin, CheckpointMixin, LoggingMixin):
    def __init__(self, simulation_app, cfg: MixGRPOConfig):
        self.simulation_app = simulation_app
        self.cfg = cfg
        self.start_update = 1
        self.chunk_dim = cfg.horizon * cfg.action_dim
        self.checkpoint_dir = Path(cfg.checkpoint_dir).expanduser().resolve() if cfg.checkpoint_dir else None

        self.env = G1MimicEnv(
            MimicEnvConfig(
                device=cfg.device,
                num_envs=cfg.num_envs,
                sim_dt=cfg.sim_dt,
                fix_root_link=cfg.fix_root_link,
                motion_start_phase=cfg.motion_start_phase,
                motion_end_phase=cfg.motion_end_phase,
                motion_file=cfg.motion_file,
                max_episode_steps=cfg.max_episode_steps,
            )
        )
        if self.env.action_dim != cfg.action_dim:
            raise ValueError(f"Expected env action_dim {self.env.action_dim}, got {cfg.action_dim}")
        self.num_grpo_groups = self.env.num_envs

        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        policy_obs_dim = cfg.policy_obs_dim if cfg.policy_obs_dim > 0 else self.env.observation_dim
        self.policy = FlowMatchingPolicy(
            obs_dim=policy_obs_dim,
            action_dim=cfg.action_dim,
            horizon=cfg.horizon,
            hidden_dim=cfg.hidden_dim,
            time_embed_dim=cfg.time_embed_dim,
            depth=cfg.depth,
            action_limit=cfg.action_limit,
        ).to(self.env.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)
        self.current_observation = self._reset_training_envs()

        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if cfg.resume:
            self._load_checkpoint(Path(cfg.resume).expanduser().resolve())

    def _snapshot_env_state(self) -> dict[str, torch.Tensor]:
        robot = self.env.robot
        return {
            "root_state_w": robot.data.root_state_w.clone(),
            "joint_pos": robot.data.joint_pos.clone(),
            "joint_vel": robot.data.joint_vel.clone(),
            "phase_steps": self.env.phase_steps.clone(),
            "episode_steps": self.env.episode_steps.clone(),
            "last_action": self.env.last_action.clone(),
            "next_push_step": self.env.next_push_step.clone(),
            "bin_failed_count": self.env.bin_failed_count.clone(),
        }

    def _restore_env_state(self, snapshot: dict[str, torch.Tensor]) -> None:
        env_ids = torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        root_state = snapshot["root_state_w"]
        root_pos_local = root_state[:, :3] - self.env.scene.env_origins
        self.env._write_robot_state(
            root_pos=root_pos_local,
            root_quat=root_state[:, 3:7],
            root_lin_vel=root_state[:, 7:10],
            root_ang_vel=root_state[:, 10:13],
            joint_pos=snapshot["joint_pos"][:, self.env.action_joint_ids],
            joint_vel=snapshot["joint_vel"][:, self.env.action_joint_ids],
            env_ids=env_ids,
        )
        self.env.phase_steps = snapshot["phase_steps"].clone()
        self.env.episode_steps = snapshot["episode_steps"].clone()
        self.env.last_action = snapshot["last_action"].clone()
        self.env.next_push_step = snapshot["next_push_step"].clone()
        self.env.bin_failed_count = snapshot["bin_failed_count"].clone()
        self.env._current_bin_failed.zero_()
        self.env.scene.reset(env_ids=env_ids)
        self.env.scene.update(self.env.physics_dt)

    def _cps_sample_with_logprobs(self, obs: torch.Tensor, noise: torch.Tensor) -> dict[str, torch.Tensor]:
        self.policy._validate_inputs(obs, noise, self.cfg.flow_steps)
        obs_prep = self.policy._prepare_observation(obs)
        batch_size = noise.shape[0]
        steps = self.cfg.flow_steps
        sigma_schedule = torch.linspace(1.0, 0.0, steps + 1, device=noise.device)

        latent = noise
        all_latents = [latent.detach()]
        all_log_probs = []

        for step_index in range(steps):
            sigma = sigma_schedule[step_index]
            s_val = 1.0 - sigma.item()
            s_batch = torch.full((batch_size,), s_val, device=noise.device, dtype=noise.dtype)

            with torch.no_grad():
                velocity = self.policy.velocity_field(obs_prep, latent, s_batch)
            model_output = -velocity
            deterministic = step_index == steps - 1
            latent, log_prob = flow_grpo_step(
                model_output=model_output,
                latents=latent,
                sigmas=sigma_schedule,
                index=step_index,
                deterministic=deterministic,
                noise_level=self.cfg.cps_eta,
            )
            all_latents.append(latent.detach())
            all_log_probs.append(log_prob.detach())

        pre_tanh = torch.clamp(latent, -10.0, 10.0)
        actions = self.policy._action_transform(pre_tanh)
        return {
            "actions": actions.view(batch_size, self.policy.horizon, self.policy.action_dim),
            "all_latents": torch.stack(all_latents, dim=1),
            "all_log_probs": torch.stack(all_log_probs, dim=1),
            "sigma_schedule": sigma_schedule,
        }

    def _recompute_logprob_one_step(
        self,
        obs_prep: torch.Tensor,
        latents: torch.Tensor,
        next_latents: torch.Tensor,
        sigma_schedule: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        batch_size = latents.shape[0]
        sigma = sigma_schedule[step_index]
        s_val = 1.0 - sigma.item()
        s_batch = torch.full((batch_size,), s_val, device=latents.device, dtype=latents.dtype)
        velocity = self.policy.velocity_field(obs_prep, latents, s_batch)
        _, log_prob = flow_grpo_step(
            model_output=-velocity,
            latents=latents,
            sigmas=sigma_schedule,
            index=step_index,
            prev_sample=next_latents,
            deterministic=False,
            noise_level=self.cfg.cps_eta,
        )
        return log_prob

    def _collect_groups(self, current_obs: torch.Tensor) -> dict[str, torch.Tensor]:
        total_envs = self.env.num_envs
        group_size = self.cfg.group_size
        chunks_per_rollout = self.cfg.chunks_per_rollout
        snapshot = self._snapshot_env_state()
        sigma_schedule = None

        all_group_rollout_rewards = []
        all_group_chunk_rewards = []
        all_group_obs = []
        all_group_latents = []
        all_group_log_probs = []
        all_group_valid_masks = []

        metric_chunk_return: torch.Tensor | None = None
        metric_actions: torch.Tensor | None = None
        metric_infos_list: list[dict[str, torch.Tensor]] = []

        for group_index in range(group_size):
            self._restore_env_state(snapshot)
            obs_t = current_obs
            done_mask = torch.zeros(total_envs, dtype=torch.bool, device=self.env.device)

            group_chunk_rewards = []
            group_chunk_obs = []
            group_chunk_latents = []
            group_chunk_log_probs = []
            group_chunk_valid_masks = []

            for chunk_index in range(chunks_per_rollout):
                chunk_valid_mask = ~done_mask
                noise = torch.zeros(total_envs, self.chunk_dim, device=self.env.device)

                with torch.no_grad():
                    sample = self._cps_sample_with_logprobs(obs_t, noise)

                sigma_schedule = sample["sigma_schedule"]
                _, step_rewards, terminations, truncations, infos_list = self.env.chunk_step(
                    sample["actions"],
                    auto_reset=False,
                )
                chunk_reward = step_rewards.sum(dim=1).masked_fill(done_mask, 0.0)
                done_mask = done_mask | (terminations | truncations).any(dim=1)
                next_obs_t = self.env.get_observation()

                if group_index == 0 and chunk_index == 0:
                    metric_chunk_return = chunk_reward.detach()
                    metric_actions = sample["actions"].clone()
                    metric_infos_list = infos_list

                group_chunk_obs.append(obs_t)
                group_chunk_latents.append(sample["all_latents"])
                group_chunk_log_probs.append(sample["all_log_probs"][:, :-1])
                group_chunk_rewards.append(chunk_reward)
                group_chunk_valid_masks.append(chunk_valid_mask)
                obs_t = next_obs_t

            group_chunk_rewards_t = torch.stack(group_chunk_rewards, dim=1)
            all_group_rollout_rewards.append(group_chunk_rewards_t.sum(dim=1))
            all_group_chunk_rewards.append(group_chunk_rewards_t)
            all_group_obs.append(torch.stack(group_chunk_obs, dim=1))
            all_group_latents.append(torch.stack(group_chunk_latents, dim=1))
            all_group_log_probs.append(torch.stack(group_chunk_log_probs, dim=1))
            all_group_valid_masks.append(torch.stack(group_chunk_valid_masks, dim=1))

        rollout_rewards = torch.stack(all_group_rollout_rewards, dim=1)
        chunk_rewards = torch.stack(all_group_chunk_rewards, dim=1)
        if metric_chunk_return is None:
            metric_chunk_return = torch.zeros(total_envs, device=self.env.device)
        if metric_actions is None:
            metric_actions = torch.zeros(total_envs, self.cfg.horizon, self.cfg.action_dim, device=self.env.device)

        return {
            "obs": torch.stack(all_group_obs, dim=1),
            "rewards": rollout_rewards,
            "chunk_rewards": chunk_rewards,
            "latents": torch.stack(all_group_latents, dim=1),
            "log_probs": torch.stack(all_group_log_probs, dim=1),
            "valid_mask": torch.stack(all_group_valid_masks, dim=1),
            "sigma_schedule": sigma_schedule,
            "metric_chunk_return": metric_chunk_return,
            "metric_actions": metric_actions,
            "metric_infos_list": metric_infos_list,
        }

    def _compute_grpo_advantages(self, chunk_rewards: torch.Tensor) -> torch.Tensor:
        returns = torch.zeros_like(chunk_rewards)
        running = torch.zeros_like(chunk_rewards[:, :, 0])
        for chunk_index in range(chunk_rewards.shape[-1] - 1, -1, -1):
            running = chunk_rewards[:, :, chunk_index] + self.cfg.discount_gamma * running
            returns[:, :, chunk_index] = running
        mean = returns.mean(dim=1, keepdim=True)
        std = returns.std(dim=1, keepdim=True) + 1e-8
        return (returns - mean) / std

    def _policy_update(
        self,
        obs: torch.Tensor,
        latents: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        sigma_schedule: torch.Tensor,
    ) -> dict[str, float]:
        sample_count = obs.shape[0]
        step_count = old_log_probs.shape[1]
        clipped_advantages = torch.clamp(advantages, -self.cfg.adv_clip_max, self.cfg.adv_clip_max)
        obs_prep = self.policy._prepare_observation(obs)
        probe_count = min(128, sample_count)
        probe_obs = obs[:probe_count]
        probe_noise = torch.zeros(probe_count, self.chunk_dim, device=obs.device, dtype=obs.dtype)
        with torch.no_grad():
            probe_action_before = self.policy(probe_obs, probe_noise, steps=self.cfg.flow_steps)
            params_before = [param.detach().clone() for param in self.policy.parameters()]

        totals = {
            "loss": 0.0,
            "policy_loss": 0.0,
            "latent_reg_loss": 0.0,
            "action_sat_loss": 0.0,
            "clip_frac": 0.0,
            "ratio": 0.0,
            "ratio_min": float("inf"),
            "ratio_max": 0.0,
            "logprob_delta_abs": 0.0,
        }
        update_count = 0
        grad_norm = 0.0

        for _ in range(self.cfg.policy_epochs):
            perm = torch.randperm(sample_count, device=obs.device)
            for start in range(0, sample_count, self.cfg.mini_batch_size):
                mb = perm[start : start + self.cfg.mini_batch_size]
                mb_obs = obs_prep[mb]
                mb_adv = clipped_advantages[mb]
                mb_latents = latents[mb]
                mb_policy_loss = torch.tensor(0.0, device=obs.device)
                mb_clip_frac = 0.0
                mb_ratio_sum = 0.0

                for step_index in range(step_count):
                    step_latent = mb_latents[:, step_index, :]
                    step_next = mb_latents[:, step_index + 1, :]
                    step_old_lp = old_log_probs[mb, step_index]
                    new_log_prob = self._recompute_logprob_one_step(
                        mb_obs,
                        step_latent,
                        step_next,
                        sigma_schedule,
                        step_index,
                    )
                    ratio = torch.exp(new_log_prob - step_old_lp)
                    logprob_delta = new_log_prob - step_old_lp
                    unclipped_loss = -mb_adv * ratio
                    clipped_loss = -mb_adv * torch.clamp(
                        ratio,
                        1.0 - self.cfg.clip_range,
                        1.0 + self.cfg.clip_range,
                    )
                    mb_policy_loss = mb_policy_loss + torch.maximum(unclipped_loss, clipped_loss).mean()
                    with torch.no_grad():
                        mb_clip_frac += (torch.abs(ratio - 1.0) > self.cfg.clip_range).float().mean().item() / step_count
                        mb_ratio_sum += ratio.mean().item() / step_count
                        totals["ratio_min"] = min(totals["ratio_min"], float(ratio.min().item()))
                        totals["ratio_max"] = max(totals["ratio_max"], float(ratio.max().item()))
                        totals["logprob_delta_abs"] += float(logprob_delta.abs().mean().item()) / step_count

                mb_policy_loss = mb_policy_loss / max(step_count, 1)
                pred_pre_tanh = self.policy._integrate_flow(
                    mb_obs,
                    mb_latents[:, 0, :],
                    steps=self.cfg.flow_steps,
                )
                pred_action = self.policy._action_transform(pred_pre_tanh)
                latent_excess = torch.relu(pred_pre_tanh.abs() - self.cfg.latent_soft_limit)
                mb_latent_reg_loss = torch.mean(latent_excess**2)
                sat_threshold = min(max(self.cfg.action_saturation_threshold, 0.0), self.cfg.action_limit)
                action_excess = torch.relu(pred_action.abs() - sat_threshold)
                mb_action_sat_loss = torch.mean(action_excess**2)

                loss = (
                    mb_policy_loss
                    + self.cfg.latent_reg_coeff * mb_latent_reg_loss
                    + self.cfg.action_saturation_coeff * mb_action_sat_loss
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                totals["loss"] += loss.item()
                totals["policy_loss"] += mb_policy_loss.item()
                totals["latent_reg_loss"] += mb_latent_reg_loss.item()
                totals["action_sat_loss"] += mb_action_sat_loss.item()
                totals["clip_frac"] += mb_clip_frac
                totals["ratio"] += mb_ratio_sum
                update_count += 1

        denom = max(update_count, 1)
        with torch.no_grad():
            probe_action_after = self.policy(probe_obs, probe_noise, steps=self.cfg.flow_steps)
            action_delta = torch.mean(torch.abs(probe_action_after - probe_action_before))
            param_delta_sq = torch.tensor(0.0, device=obs.device)
            param_count = 0
            for param, before in zip(self.policy.parameters(), params_before, strict=True):
                delta = param.detach() - before
                param_delta_sq = param_delta_sq + torch.sum(delta * delta)
                param_count += delta.numel()
            param_rms_delta = torch.sqrt(param_delta_sq / max(param_count, 1))
        return {
            "policy/loss": totals["loss"] / denom,
            "policy/policy_loss": totals["policy_loss"] / denom,
            "policy/latent_reg_loss": totals["latent_reg_loss"] / denom,
            "policy/action_sat_loss": totals["action_sat_loss"] / denom,
            "policy/clip_frac": totals["clip_frac"] / denom,
            "policy/ratio": totals["ratio"] / denom,
            "policy/ratio_min": totals["ratio_min"] if totals["ratio_min"] != float("inf") else 0.0,
            "policy/ratio_max": totals["ratio_max"],
            "policy/logprob_delta_abs": totals["logprob_delta_abs"] / denom,
            "policy/grad_norm": float(grad_norm),
            "policy/action_delta": float(action_delta.item()),
            "policy/param_rms_delta": float(param_rms_delta.item()),
        }

    def _training_rollout_horizon(self) -> int:
        return max(1, self.cfg.horizon * self.cfg.chunks_per_rollout)

    def _reset_training_envs(self) -> torch.Tensor:
        phase_indices = self.env.sample_phase_indices(
            self.num_grpo_groups,
            horizon=self._training_rollout_horizon(),
        )
        return self.env.reset(phase_indices=phase_indices)

    def train(self) -> None:
        print("[INFO] Starting MixGRPO training", flush=True)
        print(f"[INFO] motion_file={self.cfg.motion_file}", flush=True)
        print(
            f"[INFO] obs_dim={self.env.observation_dim} "
            f"action_chunk=({self.cfg.horizon}, {self.cfg.action_dim}) "
            f"num_envs={self.cfg.num_envs} group_size={self.cfg.group_size} "
            f"chunks_per_rollout={self.cfg.chunks_per_rollout} "
            f"cps_eta={self.cfg.cps_eta} flow_steps={self.cfg.flow_steps} "
            f"action_limit={self.cfg.action_limit}",
            flush=True,
        )
        print(
            f"[INFO] policy_epochs={self.cfg.policy_epochs} "
            f"clip_range={self.cfg.clip_range} "
            f"latent_reg_coeff={self.cfg.latent_reg_coeff} "
            f"action_saturation_coeff={self.cfg.action_saturation_coeff} "
            f"mini_batch={self.cfg.mini_batch_size} lr={self.cfg.lr}",
            flush=True,
        )
        if self.checkpoint_dir is not None:
            print(f"[INFO] checkpoint_dir={self.checkpoint_dir}", flush=True)
        if self.cfg.resume:
            print(f"[INFO] resumed_from={self.cfg.resume}", flush=True)

        for update_idx in range(self.start_update, self.cfg.max_updates + 1):
            if not self.simulation_app.is_running():
                break

            t0 = time.perf_counter()
            current_obs = self._reset_training_envs()
            group_data = self._collect_groups(current_obs)
            collect_time = time.perf_counter() - t0

            advantages = self._compute_grpo_advantages(group_data["chunk_rewards"])
            env_count = self.num_grpo_groups
            group_size = self.cfg.group_size
            chunks = self.cfg.chunks_per_rollout
            steps = self.cfg.flow_steps
            train_steps = max(steps - 1, 0)

            valid_flat = group_data["valid_mask"].reshape(env_count * group_size * chunks)
            obs_flat = group_data["obs"].reshape(env_count * group_size * chunks, -1)[valid_flat]
            latents_flat = group_data["latents"].reshape(env_count * group_size * chunks, steps + 1, -1)[valid_flat]
            log_probs_flat = group_data["log_probs"].reshape(env_count * group_size * chunks, train_steps)[valid_flat]
            adv_flat = advantages.reshape(env_count * group_size * chunks)[valid_flat]

            t1 = time.perf_counter()
            update_metrics = self._policy_update(
                obs_flat,
                latents_flat,
                log_probs_flat,
                adv_flat,
                group_data["sigma_schedule"],
            )
            update_time = time.perf_counter() - t1
            metrics = self._build_metrics(group_data, advantages, update_metrics, collect_time, update_time)

            if self.cfg.validation_every > 0 and update_idx % self.cfg.validation_every == 0:
                metrics.update(self.run_validation_rollout())
                fixed_metrics = self.run_validation_rollout(fixed_seed=42)
                for key, value in fixed_metrics.items():
                    metrics[key.replace("validation/", "val_fixed/")] = value

            if update_idx % self.cfg.log_every == 0:
                self._log_update(update_idx, metrics)

            if self.checkpoint_dir is not None and (
                update_idx == self.cfg.max_updates
                or (self.cfg.save_every > 0 and update_idx % self.cfg.save_every == 0)
            ):
                self._save_checkpoint(update_idx, metrics)

            if self._target_validation_reached(metrics):
                if self.checkpoint_dir is not None:
                    self._save_checkpoint(update_idx, metrics, filename=self.cfg.success_checkpoint_name)
                print(
                    f"[SUCCESS] validation reached {self.cfg.target_validation_steps} steps; "
                    f"saved {self.cfg.success_checkpoint_name}",
                    flush=True,
                )
                break

        self.current_observation = self._reset_training_envs()
        print("[INFO] Training finished.", flush=True)

    def _build_metrics(
        self,
        group_data: dict[str, torch.Tensor],
        advantages: torch.Tensor,
        update_metrics: dict[str, float],
        collect_time: float,
        update_time: float,
    ) -> dict[str, float]:
        group_rewards = group_data["rewards"]
        metric_chunk_return = group_data["metric_chunk_return"]
        metric_actions = group_data["metric_actions"]
        act_abs_tensor = metric_actions.abs()
        act_abs = act_abs_tensor.mean(dim=(0, 1))
        final_latents = group_data["latents"][..., -1, :]
        valid_mask = group_data["valid_mask"]
        metrics = {
            **update_metrics,
            "group/reward_mean": float(group_rewards.mean().item()),
            "group/reward_std": float(group_rewards.std().item()),
            "group/reward_min": float(group_rewards.min().item()),
            "group/reward_max": float(group_rewards.max().item()),
            "group/advantage_abs_mean": float(advantages.abs().mean().item()),
            "rollout/valid_frac": float(valid_mask.float().mean().item()),
            "rollout/chunk_return_mean": float(metric_chunk_return.mean().item()),
            "rollout/chunk_return_std": float(metric_chunk_return.std().item()),
            "timing/collect_s": collect_time,
            "timing/update_s": update_time,
            "act/abs_mean": float(act_abs_tensor.mean().item()),
            "act/abs_max": float(act_abs_tensor.max().item()),
            "act/abs_p95": float(torch.quantile(act_abs_tensor.flatten(), 0.95).item()),
            "act/legs_abs": float(act_abs[[0, 1, 3, 4, 6, 7, 9, 10, 13, 14, 17, 18]].mean().item()),
            "act/waist_abs": float(act_abs[[2, 5, 8]].mean().item()),
            "act/arms_abs": float(act_abs[[11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]].mean().item()),
            "latent/final_abs_mean": float(final_latents.abs().mean().item()),
            "latent/final_abs_max": float(final_latents.abs().max().item()),
            "act/l_shoulder_pitch": float(act_abs[11].item()),
            "act/r_shoulder_pitch": float(act_abs[12].item()),
            "act/l_shoulder_roll": float(act_abs[15].item()),
            "act/r_shoulder_roll": float(act_abs[16].item()),
            "act/l_shoulder_yaw": float(act_abs[19].item()),
            "act/r_shoulder_yaw": float(act_abs[20].item()),
            "act/l_elbow": float(act_abs[21].item()),
            "act/r_elbow": float(act_abs[22].item()),
            "act/l_wrist_roll": float(act_abs[23].item()),
            "act/r_wrist_roll": float(act_abs[24].item()),
            "act/l_wrist_pitch": float(act_abs[25].item()),
            "act/r_wrist_pitch": float(act_abs[26].item()),
            "act/l_wrist_yaw": float(act_abs[27].item()),
            "act/r_wrist_yaw": float(act_abs[28].item()),
        }
        done_union: dict[str, torch.Tensor] = {}
        for step_info in group_data["metric_infos_list"]:
            for key, value in step_info["reward_terms"].items():
                metric_key = f"reward/{key}_mean"
                metrics[metric_key] = metrics.get(metric_key, 0.0) + float(value.mean().item()) / self.cfg.horizon
            for key, value in step_info["done_terms"].items():
                if key not in done_union:
                    done_union[key] = value.bool().clone()
                else:
                    done_union[key] |= value.bool()
        for key, union_mask in done_union.items():
            metrics[f"done/{key}_frac"] = float(union_mask.float().mean().item())
        reward_weights = {
            "joint_acc": -2.5e-7,
            "joint_torque": -1.0e-5,
            "action_rate": -1.0e-1,
            "joint_limit": -10.0,
            "anchor_pos_reward": 0.5,
            "anchor_ori_reward": 0.5,
            "body_pos_reward": 1.0,
            "body_ori_reward": 1.0,
            "body_lin_vel_reward": 1.0,
            "body_ang_vel_reward": 1.0,
            "undesired_contacts": -0.1,
        }
        weighted_positive = 0.0
        weighted_penalty = 0.0
        for reward_name, weight in reward_weights.items():
            raw_key = f"reward/{reward_name}_mean"
            if raw_key not in metrics:
                continue
            contribution = weight * metrics[raw_key] * self.env.dt
            metrics[f"reward_weighted/{reward_name}"] = contribution
            if contribution >= 0.0:
                weighted_positive += contribution
            else:
                weighted_penalty += contribution
        metrics["reward_weighted/positive"] = weighted_positive
        metrics["reward_weighted/penalty"] = weighted_penalty
        metrics["reward_weighted/total"] = weighted_positive + weighted_penalty
        return metrics
