from __future__ import annotations

import math
from collections import deque

import torch

from algorithms.base import Algorithm
from networks.flow_inference import deterministic_sde_ode_actions
from networks.flow_sampling import flow_grpo_step
from networks.flow_policy import FlowMatchingPolicy


class MixGRPO(Algorithm):

    def build(self) -> None:
        cfg = self.cfg
        env = self.env
        if cfg.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {cfg.horizon}")
        if cfg.num_generations < 1:
            raise ValueError(f"num_generations must be >= 1, got {cfg.num_generations}")
        if env.num_envs % cfg.num_generations != 0:
            raise ValueError(
                f"num_envs ({env.num_envs}) must be divisible by num_generations ({cfg.num_generations})"
            )
        self.num_act = env.action_dim
        self.num_grpo_groups = env.num_envs // cfg.num_generations
        self.action_chunk_dim = cfg.horizon * self.num_act

        self._policy = FlowMatchingPolicy(
            obs_dim=env.observation_dim,
            action_dim=self.num_act,
            horizon=cfg.horizon,
            hidden_dims=tuple(cfg.actor_hidden_dims),
            activation=cfg.activation,
            action_squash_scale=cfg.action_squash_scale,
        ).to(env.device)
        self.chunk_dim = self._policy.chunk_dim

        self._optimizer = torch.optim.Adam(
            self._policy.parameters(), lr=cfg.policy_lr, betas=(0.9, 0.999), eps=1.0e-8
        )
        self.learning_rate = float(cfg.policy_lr)

        self._init_train_episode_stats()

        self.max_episode_steps = env.max_episode_steps

    @property
    def policy(self) -> torch.nn.Module:
        return self._policy

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def horizon(self) -> int:
        return self.cfg.horizon

    def extra_checkpoint_state(self) -> dict:
        return {"learning_rate": float(self.learning_rate)}

    def load_extra_checkpoint_state(self, payload: dict, reset_optimizer: bool = False) -> None:
        if reset_optimizer:


            resume_lr = float(self.cfg.policy_lr)
        else:
            checkpoint_lr = float(payload.get("learning_rate", self.cfg.policy_lr))
            resume_lr = min(float(self.cfg.policy_lr), checkpoint_lr)
        for group in self._optimizer.param_groups:
            group["lr"] = resume_lr
        self.learning_rate = resume_lr


    def _chunks_per_grpo_update(self) -> int:
        rollout_env_steps = int(self.cfg.rollout_env_steps)
        if rollout_env_steps <= 0:
            raise ValueError(f"rollout_env_steps must be > 0, got {rollout_env_steps}")
        horizon = max(1, int(self.cfg.horizon))
        if rollout_env_steps % horizon != 0:
            raise ValueError(
                f"rollout_env_steps ({rollout_env_steps}) must be divisible by horizon ({horizon})."
            )
        return max(1, rollout_env_steps // horizon)

    def _training_rollout_horizon(self) -> int:
        return max(1, self.cfg.horizon * self._chunks_per_grpo_update())

    def _train_step_indices(self, device) -> torch.Tensor:
        return torch.arange(int(self.cfg.flow_steps), device=device, dtype=torch.long)


    def _init_train_episode_stats(self) -> None:
        env = self.env
        self._train_reward_sum = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._train_episode_length = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)
        self._train_reward_buffer: deque[float] = deque(maxlen=100)
        self._train_length_buffer: deque[float] = deque(maxlen=100)
        self._train_completed_episodes = 0

    def _record_train_episode_stats(self, rewards, dones, step_counts=None) -> None:
        self._train_reward_sum += rewards.to(dtype=torch.float32)
        if step_counts is None:
            self._train_episode_length += 1.0
        else:
            self._train_episode_length += step_counts.to(dtype=torch.float32)
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() == 0:
            return
        self._train_reward_buffer.extend(self._train_reward_sum.index_select(0, done_ids).detach().cpu().tolist())
        self._train_length_buffer.extend(self._train_episode_length.index_select(0, done_ids).detach().cpu().tolist())
        self._train_completed_episodes += int(done_ids.numel())
        self._train_reward_sum[done_ids] = 0.0
        self._train_episode_length[done_ids] = 0.0


    def _sde_ode_rollout_actions(self, obs, *, initial_noise, sde_noise=None):
        self._policy._validate_inputs(obs, initial_noise, self.cfg.flow_steps)
        obs_prep = self._policy._prepare_observation(obs)
        batch_size = obs.shape[0]
        steps = int(self.cfg.flow_steps)
        sigma_schedule = torch.linspace(1.0, 0.0, steps + 1, device=obs.device, dtype=obs.dtype)
        latent = initial_noise.to(device=obs.device, dtype=obs.dtype) * float(self.cfg.init_noise_std)
        all_latents = [latent.detach()]
        if sde_noise is None:
            sde_noise = torch.randn(batch_size, steps, self.chunk_dim, device=obs.device, dtype=obs.dtype)
        elif sde_noise.shape != (batch_size, steps, self.chunk_dim):
            raise ValueError(
                f"sde_noise must have shape {(batch_size, steps, self.chunk_dim)}, got {tuple(sde_noise.shape)}"
            )
        step_log_probs = []
        for step_index in range(steps):
            sigma = sigma_schedule[step_index]
            timestep_batch = torch.full((batch_size,), float(sigma.item()), device=obs.device, dtype=obs.dtype)
            model_output = self._policy.velocity_field(obs_prep, latent, timestep_batch)
            latent, log_prob = flow_grpo_step(
                model_output=model_output,
                latents=latent,
                sigmas=sigma_schedule,
                index=step_index,
                eta=float(self.cfg.sde_eta),
                sample_noise=sde_noise[:, step_index],
            )
            all_latents.append(latent.detach())
            step_log_probs.append(log_prob)
        if not step_log_probs:
            raise RuntimeError("SDE-ODE rollout produced no trainable transition log-probs.")
        actions = self._policy._action_transform(latent)
        stacked_log_probs = torch.stack(step_log_probs, dim=1)
        return actions, torch.stack(all_latents, dim=1), stacked_log_probs

    def _sample_policy_with_logprobs(self, obs, noise, sde_noise=None):
        train_step_indices = self._train_step_indices(obs.device)
        actions, latent_path, step_log_probs = self._sde_ode_rollout_actions(
            obs, initial_noise=noise, sde_noise=sde_noise
        )
        return {
            "actions": actions.view(obs.shape[0], self._policy.horizon, self._policy.action_dim),
            "all_latents": latent_path.detach(),
            "log_probs": step_log_probs.detach(),
            "train_step_indices": train_step_indices.detach().clone(),
        }

    def deterministic_actions(self, obs: torch.Tensor) -> torch.Tensor:
        initial_noise = None
        if str(self.cfg.eval_initial_noise) == "random":
            initial_noise = torch.randn(obs.shape[0], self.chunk_dim, device=obs.device, dtype=obs.dtype)
        return deterministic_sde_ode_actions(
            self._policy,
            obs,
            steps=self.cfg.flow_steps,
            sde_eta=float(self.cfg.sde_eta),
            initial_noise=initial_noise,
        )

    def _compute_transition_log_probs(self, obs, latent_path, step_indices):
        if latent_path.ndim != 3:
            raise ValueError(f"latent_path must be (batch, steps+1, dim), got {tuple(latent_path.shape)}")
        sample_count, path_steps, latent_dim = latent_path.shape
        steps = int(self.cfg.flow_steps)
        if obs.shape[0] != sample_count:
            raise ValueError("obs and latent_path batch sizes must match")
        if path_steps != steps + 1 or latent_dim != self.chunk_dim:
            raise ValueError(
                f"latent_path must be {(sample_count, steps + 1, self.chunk_dim)}, got {tuple(latent_path.shape)}"
            )
        obs_prep = self._policy._prepare_observation(obs)
        sigma_schedule = torch.linspace(1.0, 0.0, steps + 1, device=obs.device, dtype=obs.dtype)
        log_probs = []
        for step_tensor in step_indices.to(device=obs.device, dtype=torch.long):
            step_index = int(step_tensor.item())
            if step_index < 0 or step_index >= steps:
                raise ValueError(f"MixGRPO step index {step_index} is outside [0, {steps})")
            latent_t = latent_path[:, step_index].to(dtype=obs.dtype)
            next_latent = latent_path[:, step_index + 1].to(dtype=obs.dtype)
            sigma = sigma_schedule[step_index]
            timestep_batch = torch.full((sample_count,), float(sigma.item()), device=obs.device, dtype=obs.dtype)
            model_output = self._policy.velocity_field(obs_prep, latent_t, timestep_batch)
            _, log_prob = flow_grpo_step(
                model_output=model_output,
                latents=latent_t,
                sigmas=sigma_schedule,
                index=step_index,
                eta=float(self.cfg.sde_eta),
                prev_sample=next_latent,
            )
            log_probs.append(log_prob)
        return torch.stack(log_probs, dim=1)


    def _training_anchor_phases(self) -> torch.Tensor:
        anchors = self.env.sample_phase_indices(self.num_grpo_groups, self._training_rollout_horizon())
        return anchors.to(device=self.env.device, dtype=torch.long)

    def initial_reset(self) -> torch.Tensor:

        generation_count = int(self.cfg.num_generations)
        phase_indices = self._training_anchor_phases()
        reset_phases = phase_indices.repeat_interleave(generation_count)
        self.env.reset(phase_indices=reset_phases)
        self._replicate_group_reset_state(self.num_grpo_groups, generation_count)
        return self.env.get_observation()

    def reset_for_update(self, update_idx: int) -> torch.Tensor:
        generation_count = int(self.cfg.num_generations)
        phase_indices = self._training_anchor_phases()
        reset_phases = phase_indices.repeat_interleave(generation_count)
        env_ids = torch.arange(self.env.num_envs, device=self.env.device, dtype=torch.long)
        self.env.reset_envs(env_ids, phase_indices=reset_phases)

        self._train_reward_sum.zero_()
        self._train_episode_length.zero_()
        self._replicate_group_reset_state(self.num_grpo_groups, generation_count)
        return self.env.get_observation()

    def _replicate_group_reset_state(self, group_count, generation_count, group_ids=None) -> None:
        if generation_count <= 1:
            return
        total_envs = group_count * generation_count
        if total_envs != self.env.num_envs:
            raise ValueError(
                f"group_count * generation_count must equal num_envs, got {group_count} * {generation_count}"
            )
        device = self.env.device
        if group_ids is None:
            env_ids = torch.arange(total_envs, device=device, dtype=torch.long)
            target_env_ids = env_ids[(env_ids % generation_count) != 0]
            source_for_target = (target_env_ids // generation_count) * generation_count
        else:
            group_ids = group_ids.to(device=device, dtype=torch.long).reshape(-1)
            if group_ids.numel() == 0:
                return
            branch_offsets = torch.arange(1, generation_count, device=device, dtype=torch.long)
            target_env_ids = (group_ids[:, None] * generation_count + branch_offsets[None, :]).reshape(-1)
            source_for_target = (group_ids * generation_count).repeat_interleave(generation_count - 1)
        if target_env_ids.numel() == 0:
            return
        env = self.env
        root_state = env.robot.data.root_state_w.index_select(0, source_for_target)
        joint_pos = env.robot.data.joint_pos.index_select(0, source_for_target)
        joint_vel = env.robot.data.joint_vel.index_select(0, source_for_target)
        root_pos_local = root_state[:, :3] - env.scene.env_origins.index_select(0, source_for_target)

        env.default_root_state[target_env_ids] = env.default_root_state.index_select(0, source_for_target)
        env.default_joint_pos[target_env_ids] = env.default_joint_pos.index_select(0, source_for_target)
        env.default_joint_vel[target_env_ids] = env.default_joint_vel.index_select(0, source_for_target)
        env.default_action_joint_pos[target_env_ids] = env.default_action_joint_pos.index_select(0, source_for_target)
        env.default_action_joint_vel[target_env_ids] = env.default_action_joint_vel.index_select(0, source_for_target)

        target_cpu = target_env_ids.detach().cpu()
        source_cpu = source_for_target.detach().cpu()
        try:
            coms = env.robot.root_physx_view.get_coms().clone()
            coms[target_cpu] = coms[source_cpu]
            env.robot.root_physx_view.set_coms(coms, target_cpu)
        except Exception as exc:
            print(f"[WARN] Failed to replicate group torso COM randomization: {exc}", flush=True)
        try:
            materials = env.robot.root_physx_view.get_material_properties().clone()
            materials[target_cpu] = materials[source_cpu]
            env.robot.root_physx_view.set_material_properties(materials, target_cpu)
        except Exception as exc:
            print(f"[WARN] Failed to replicate group material randomization: {exc}", flush=True)

        env.scene.reset(env_ids=target_env_ids)
        env._write_robot_state(
            root_pos=root_pos_local,
            root_quat=root_state[:, 3:7],
            root_lin_vel=root_state[:, 7:10],
            root_ang_vel=root_state[:, 10:13],
            joint_pos=joint_pos[:, env.action_joint_ids],
            joint_vel=joint_vel[:, env.action_joint_ids],
            env_ids=target_env_ids,
        )
        env.phase_steps[target_env_ids] = env.phase_steps.index_select(0, source_for_target)
        env.episode_steps[target_env_ids] = env.episode_steps.index_select(0, source_for_target)
        env.last_action[target_env_ids] = env.last_action.index_select(0, source_for_target)
        env.next_push_step[target_env_ids] = env.next_push_step.index_select(0, source_for_target)
        self._replicate_group_contact_history(target_env_ids, source_for_target)
        env.scene.update(env.physics_dt)

    def _replicate_group_contact_history(self, target_env_ids, source_for_target) -> None:
        contact_sensor = getattr(self.env, "contact_sensor", None)
        if contact_sensor is None:
            return
        data = contact_sensor.data
        for attr in (
            "net_forces_w", "net_forces_w_history", "force_matrix_w", "force_matrix_w_history",
            "last_air_time", "current_air_time", "last_contact_time", "current_contact_time",
        ):
            buffer = getattr(data, attr, None)
            if buffer is None:
                continue
            buffer[target_env_ids] = buffer.index_select(0, source_for_target)

    def _compute_group_relative_advantages(self, rewards, valid_mask):
        if rewards.ndim != 2:
            raise ValueError(f"rewards must be (groups, generations), got {tuple(rewards.shape)}")
        if valid_mask.shape != rewards.shape:
            raise ValueError("valid_mask must match rewards")
        rewards_float = rewards.to(torch.float32)
        valid_float = valid_mask.to(dtype=torch.float32)
        counts = valid_float.sum(dim=1, keepdim=True).clamp(min=1.0)
        means = (rewards_float * valid_float).sum(dim=1, keepdim=True) / counts
        centered = (rewards_float - means) * valid_float
        denom = (counts - 1.0).clamp(min=1.0)
        variances = centered.square().sum(dim=1, keepdim=True) / denom
        stds = torch.sqrt(variances + 1e-8)
        advantages = (rewards_float - means) / stds
        return torch.where(valid_mask, advantages.to(dtype=rewards.dtype), torch.zeros_like(rewards))


    def _record_first_done(
        self, *, done_mask, terminations, truncations, infos_list, chunk_index,
        first_done_chunk, first_done_phase, first_done_anchor_pos, first_done_anchor_ori,
        first_done_ee_body, first_done_timeout,
    ) -> None:
        done_steps = terminations | truncations
        new_done_any = (~done_mask) & done_steps.any(dim=1)
        if not bool(new_done_any.any()):
            return
        env_ids = new_done_any.nonzero(as_tuple=False).squeeze(-1)
        first_done_chunk[env_ids] = int(chunk_index)
        if infos_list and all("termination_phase_steps" in si for si in infos_list):
            phase_by_step = torch.stack([si["termination_phase_steps"] for si in infos_list], dim=1)
            first_offsets = done_steps.to(dtype=torch.long).argmax(dim=1)
            first_done_phase[env_ids] = phase_by_step[env_ids, first_offsets[env_ids]]
        anchor_pos = torch.zeros_like(first_done_anchor_pos)
        anchor_ori = torch.zeros_like(first_done_anchor_ori)
        ee_body = torch.zeros_like(first_done_ee_body)
        timeout = torch.zeros_like(first_done_timeout)
        for si in infos_list:
            dterms = si["done_terms"]
            anchor_pos |= dterms["anchor_pos_bad"].bool()
            anchor_ori |= dterms["anchor_ori_bad"].bool()
            ee_body |= dterms["ee_body_bad"].bool()
            timeout |= dterms["time_out"].bool()
        first_done_anchor_pos[env_ids] = anchor_pos[env_ids]
        first_done_anchor_ori[env_ids] = anchor_ori[env_ids]
        first_done_ee_body[env_ids] = ee_body[env_ids]
        first_done_timeout[env_ids] = timeout[env_ids]

    def collect(self, current_obs: torch.Tensor) -> dict:
        env = self.env
        total_envs = env.num_envs
        generation_count = int(self.cfg.num_generations)
        group_count = total_envs // generation_count
        chunks_per_rollout = self._chunks_per_grpo_update()
        collection_start_phases = (
            env.phase_steps.detach().clone() if hasattr(env, "phase_steps")
            else torch.zeros(total_envs, dtype=torch.long, device=env.device)
        )
        obs_t = current_obs
        metric_action_abs_max_all = 0.0

        first_done_chunk = torch.full((total_envs,), chunks_per_rollout, dtype=torch.long, device=env.device)
        first_done_phase = torch.full((total_envs,), -1, dtype=torch.long, device=env.device)
        first_done_anchor_pos = torch.zeros(total_envs, dtype=torch.bool, device=env.device)
        first_done_anchor_ori = torch.zeros(total_envs, dtype=torch.bool, device=env.device)
        first_done_ee_body = torch.zeros(total_envs, dtype=torch.bool, device=env.device)
        first_done_timeout = torch.zeros(total_envs, dtype=torch.bool, device=env.device)
        first_done_frame = torch.full((total_envs,), -1, dtype=torch.long, device=env.device)
        ever_done = torch.zeros(total_envs, dtype=torch.bool, device=env.device)

        rollout_obs, rollout_latents, rollout_old_log_probs = [], [], []
        rollout_train_step_indices = None
        rollout_actions, rollout_rewards, rollout_dones = [], [], []
        rollout_timeouts, rollout_valid, rollout_live_frames = [], [], []
        rollout_reward_frame, rollout_done_frame, rollout_alive_frame = [], [], []
        rollout_infos = []
        metric_first_chunk_infos: list[dict] = []
        metric_rollout_info_items: list[tuple] = []
        metric_chunk_return_first = None
        metric_chunk_return_last = None
        metric_actions_first = None
        metric_actions_last = None
        metric_cross_chunk_delta_sum = torch.zeros((), device=env.device, dtype=obs_t.dtype)
        metric_cross_chunk_delta_count = torch.zeros((), device=env.device, dtype=obs_t.dtype)
        prev_chunk_last_action = None
        horizon = int(self.cfg.horizon)
        gamma = float(self.cfg.discount_gamma)

        for chunk_index in range(chunks_per_rollout):
            active_before_step = ~ever_done
            if bool(self.cfg.init_same_noise):
                noise = torch.randn(group_count, self.chunk_dim, device=env.device).repeat_interleave(
                    generation_count, dim=0
                )
            else:
                noise = torch.randn(total_envs, self.chunk_dim, device=env.device)
            sde_noise = torch.randn(
                total_envs, int(self.cfg.flow_steps), self.chunk_dim, device=env.device, dtype=obs_t.dtype
            )
            if bool(self.cfg.first_generation_zero_noise):
                zero_branch_ids = torch.arange(0, total_envs, generation_count, device=env.device)
                noise[zero_branch_ids] = 0.0
                sde_noise[zero_branch_ids] = 0.0
            with torch.no_grad():
                sample = self._sample_policy_with_logprobs(obs_t, noise, sde_noise=sde_noise)
            rollout_train_step_indices = sample["train_step_indices"]
            action_chunk = sample["actions"]
            expected = (total_envs, horizon, self.num_act)
            if action_chunk.shape != expected:
                raise RuntimeError(f"policy produced action_chunk {tuple(action_chunk.shape)}, expected {expected}")
            metric_action_abs_max_all = max(metric_action_abs_max_all, float(action_chunk.abs().max().item()))

            chunk_reward = torch.zeros(total_envs, device=env.device, dtype=obs_t.dtype)
            chunk_done = torch.zeros(total_envs, dtype=torch.bool, device=env.device)
            chunk_timeout = torch.zeros(total_envs, dtype=torch.bool, device=env.device)
            chunk_live_frames = torch.zeros(total_envs, device=env.device, dtype=obs_t.dtype)
            alive_in_chunk = active_before_step.clone()
            last_info = None
            chunk_reward_frames, chunk_done_frames, chunk_alive_frames = [], [], []
            chunk_start_obs = obs_t
            for frame_idx in range(horizon):
                alive_before_frame = alive_in_chunk.clone()
                action_t = action_chunk[:, frame_idx, :]
                if bool((~alive_before_frame).any()):
                    action_t = torch.where(alive_before_frame.unsqueeze(-1), action_t, torch.zeros_like(action_t))
                in_chunk_auto_reset = (frame_idx == horizon - 1)
                next_obs_t, reward_t, done_t, info_t = env.step(
                    action_t,
                    auto_reset=in_chunk_auto_reset,
                    reset_horizon=max(1, (chunks_per_rollout - chunk_index - 1) * horizon + (horizon - frame_idx)),
                )
                if chunk_index == 0:
                    metric_first_chunk_infos.append(info_t)
                contrib = alive_before_frame.to(dtype=chunk_reward.dtype)
                chunk_reward = chunk_reward + (gamma ** frame_idx) * reward_t.to(dtype=chunk_reward.dtype) * contrib
                chunk_live_frames = chunk_live_frames + contrib
                chunk_reward_frames.append(reward_t.detach().to(dtype=chunk_reward.dtype))
                timeout_frame = info_t["done_terms"]["time_out"].bool()
                chunk_done_frames.append((alive_before_frame & done_t & ~timeout_frame).detach())
                chunk_alive_frames.append(alive_before_frame.detach())
                new_done_in_chunk = alive_before_frame & done_t
                if bool(new_done_in_chunk.any()):
                    chunk_done = chunk_done | new_done_in_chunk
                    chunk_timeout = chunk_timeout | (new_done_in_chunk & timeout_frame)
                    first_done_frame[new_done_in_chunk] = int(frame_idx)
                    self._record_first_done(
                        done_mask=ever_done,
                        terminations=(new_done_in_chunk & ~timeout_frame)[:, None],
                        truncations=(new_done_in_chunk & timeout_frame)[:, None],
                        infos_list=[info_t],
                        chunk_index=chunk_index,
                        first_done_chunk=first_done_chunk,
                        first_done_phase=first_done_phase,
                        first_done_anchor_pos=first_done_anchor_pos,
                        first_done_anchor_ori=first_done_anchor_ori,
                        first_done_ee_body=first_done_ee_body,
                        first_done_timeout=first_done_timeout,
                    )
                metric_rollout_info_items.append((info_t, alive_before_frame.detach()))
                alive_in_chunk = alive_before_frame & ~done_t
                ever_done = ever_done | done_t
                obs_t = next_obs_t
                last_info = info_t
            info_t = last_info
            done_t = chunk_done
            timeout_t = chunk_timeout
            reward_t = chunk_reward

            if chunk_index == 0:
                metric_chunk_return_first = reward_t.detach()
                metric_actions_first = action_chunk.detach().clone()
            if chunk_index == chunks_per_rollout - 1:
                metric_chunk_return_last = reward_t.detach()
                metric_actions_last = action_chunk.detach().clone()

            chunk_first_action = action_chunk[:, 0, :].detach()
            chunk_last_action = action_chunk[:, horizon - 1, :].detach()
            if prev_chunk_last_action is not None:
                boundary_alive = active_before_step.to(dtype=obs_t.dtype)
                if bool(boundary_alive.sum() > 0):
                    delta = (chunk_first_action - prev_chunk_last_action).abs().mean(dim=-1)
                    metric_cross_chunk_delta_sum = metric_cross_chunk_delta_sum + (delta * boundary_alive).sum()
                    metric_cross_chunk_delta_count = metric_cross_chunk_delta_count + boundary_alive.sum()
            prev_chunk_last_action = chunk_last_action

            rollout_obs.append(chunk_start_obs)
            rollout_latents.append(sample["all_latents"])
            rollout_old_log_probs.append(sample["log_probs"].detach())
            rollout_actions.append(action_chunk.detach())
            rollout_rewards.append(reward_t.detach())
            rollout_dones.append(done_t.detach())
            rollout_timeouts.append(timeout_t.detach())
            rollout_valid.append(active_before_step.detach())
            rollout_live_frames.append(chunk_live_frames.detach())
            rollout_reward_frame.append(torch.stack(chunk_reward_frames, dim=1))
            rollout_done_frame.append(torch.stack(chunk_done_frames, dim=1))
            rollout_alive_frame.append(torch.stack(chunk_alive_frames, dim=1))
            rollout_infos.append(info_t)
            self._record_train_episode_stats(reward_t.detach(), done_t.detach(), step_counts=chunk_live_frames.detach())

        last_values = torch.zeros(total_envs, device=env.device)
        tail_steps = max(0, int(self.cfg.tail_bootstrap_steps))
        tail_alive_at_end = (~ever_done).clone()
        if tail_steps > 0 and bool(tail_alive_at_end.any()):
            last_values = self._compute_tail_bootstrap(
                obs_t, tail_alive_at_end, tail_steps=tail_steps,
                gamma=gamma, terminal_penalty=float(self.cfg.terminal_penalty),
            )

        chunk_rewards = torch.stack(rollout_rewards, dim=1)
        valid_steps = torch.stack(rollout_valid, dim=1)
        live_frame_counts = torch.stack(rollout_live_frames, dim=1)
        reward_frame = torch.stack(rollout_reward_frame, dim=1)
        done_frame = torch.stack(rollout_done_frame, dim=1)
        alive_frame = torch.stack(rollout_alive_frame, dim=1)
        reward_frame_g = reward_frame.view(group_count, generation_count, chunks_per_rollout, horizon)
        done_frame_g = done_frame.view(group_count, generation_count, chunks_per_rollout, horizon)
        alive_frame_g = alive_frame.view(group_count, generation_count, chunks_per_rollout, horizon)
        objective_chunk_rewards = chunk_rewards * valid_steps.to(dtype=chunk_rewards.dtype)
        score_denominator = live_frame_counts.to(dtype=chunk_rewards.dtype).sum(dim=1).clamp(min=1.0)
        score_rewards = objective_chunk_rewards.sum(dim=1) / score_denominator
        actions = torch.stack(rollout_actions, dim=1)
        latents = torch.stack(rollout_latents, dim=1)
        old_log_probs = torch.stack(rollout_old_log_probs, dim=1)
        if rollout_train_step_indices is None:
            rollout_train_step_indices = self._train_step_indices(env.device)
        valid_mask = valid_steps.view(group_count, generation_count, chunks_per_rollout)
        metric_chunk_return = metric_chunk_return_first if metric_chunk_return_first is not None else torch.zeros(total_envs, device=env.device)
        metric_actions = metric_actions_first if metric_actions_first is not None else torch.zeros(total_envs, horizon, self.num_act, device=env.device)
        return {
            "obs": torch.stack(rollout_obs, dim=1).view(group_count, generation_count, chunks_per_rollout, -1),
            "rewards": objective_chunk_rewards.sum(dim=1).view(group_count, generation_count),
            "raw_rewards": chunk_rewards.sum(dim=1).view(group_count, generation_count),
            "score_rewards": score_rewards.view(group_count, generation_count),
            "chunk_rewards": objective_chunk_rewards.view(group_count, generation_count, chunks_per_rollout),
            "raw_chunk_rewards": chunk_rewards.view(group_count, generation_count, chunks_per_rollout),
            "train_chunk_rewards": chunk_rewards.view(group_count, generation_count, chunks_per_rollout),
            "dones": torch.stack(rollout_dones, dim=1).view(group_count, generation_count, chunks_per_rollout),
            "timeouts": torch.stack(rollout_timeouts, dim=1).view(group_count, generation_count, chunks_per_rollout),
            "live_frames": live_frame_counts.view(group_count, generation_count, chunks_per_rollout),
            "reward_frame": reward_frame_g,
            "done_frame": done_frame_g,
            "alive_frame": alive_frame_g,
            "last_values": last_values,
            "latents": latents.view(group_count, generation_count, chunks_per_rollout, self.cfg.flow_steps + 1, self.chunk_dim),
            "old_log_probs": old_log_probs.view(group_count, generation_count, *old_log_probs.shape[1:]),
            "train_step_indices": rollout_train_step_indices,
            "actions": actions.view(group_count, generation_count, chunks_per_rollout, horizon, self.num_act),
            "valid_mask": valid_mask,
            "first_done_chunk": first_done_chunk.view(group_count, generation_count),
            "first_done_phase": first_done_phase.view(group_count, generation_count),
            "first_done_anchor_pos": first_done_anchor_pos.view(group_count, generation_count),
            "first_done_anchor_ori": first_done_anchor_ori.view(group_count, generation_count),
            "first_done_ee_body": first_done_ee_body.view(group_count, generation_count),
            "first_done_timeout": first_done_timeout.view(group_count, generation_count),
            "first_done_frame": first_done_frame.view(group_count, generation_count),
            "collection_start_phases": collection_start_phases.view(group_count, generation_count),
            "metric_chunk_return": metric_chunk_return,
            "metric_actions": metric_actions,
            "metric_infos_list": metric_first_chunk_infos if metric_first_chunk_infos else rollout_infos[:1],
            "metric_chunk_return_first": metric_chunk_return_first,
            "metric_chunk_return_last": metric_chunk_return_last,
            "metric_actions_first": metric_actions_first,
            "metric_actions_last": metric_actions_last,
            "metric_cross_chunk_delta_sum": metric_cross_chunk_delta_sum,
            "metric_cross_chunk_delta_count": metric_cross_chunk_delta_count,
            "metric_action_abs_max_all": metric_action_abs_max_all,
            "metric_rollout_info_items": metric_rollout_info_items,
            "next_observation": obs_t.detach().clone(),
        }

    def _compute_tail_bootstrap(self, obs_start, alive_mask, *, tail_steps, gamma, terminal_penalty):
        if tail_steps <= 0:
            return torch.zeros(obs_start.shape[0], device=obs_start.device, dtype=obs_start.dtype)
        was_training = self._policy.training
        self._policy.eval()


        prev_record = self.env.record_motion_failures
        self.env.record_motion_failures = False
        try:
            with torch.no_grad():
                obs_t = obs_start
                tail_return = torch.zeros(obs_t.shape[0], device=obs_t.device, dtype=obs_t.dtype)
                still_alive = alive_mask.clone()
                if not bool(still_alive.any()):
                    return tail_return
                horizon = int(self.cfg.horizon)
                cached_chunk = None
                chunk_index = horizon
                discount = torch.ones(obs_t.shape[0], device=obs_t.device, dtype=obs_t.dtype)
                for _ in range(tail_steps):
                    if cached_chunk is None or chunk_index >= horizon:
                        cached_chunk = self.deterministic_actions(obs_t)
                        chunk_index = 0
                    action = cached_chunk[:, chunk_index, :]
                    if bool((~still_alive).any()):
                        action = torch.where(still_alive.unsqueeze(-1), action, torch.zeros_like(action))
                    chunk_index += 1
                    obs_t, reward, step_done, _info = self.env.step(action, auto_reset=False)
                    contrib_mask = still_alive.to(dtype=tail_return.dtype)
                    tail_return = tail_return + discount * reward.to(dtype=tail_return.dtype) * contrib_mask
                    new_done = still_alive & step_done
                    if bool(new_done.any()):
                        timeout_done = _info.get("done_terms", {}).get("time_out")
                        failure_done = new_done & (~timeout_done.bool()) if timeout_done is not None else new_done
                        if bool(failure_done.any()):
                            tail_return = tail_return - discount * float(terminal_penalty) * failure_done.to(dtype=tail_return.dtype)
                    still_alive = still_alive & ~step_done
                    if not bool(still_alive.any()):
                        break
                    discount = discount * float(gamma)
        finally:
            self.env.record_motion_failures = prev_record
            if was_training:
                self._policy.train()
        return (tail_return * alive_mask.to(dtype=tail_return.dtype)).detach()


    def update(self, group_data: dict, collect_time: float) -> dict:
        import time as _time
        env = self.env


        env_count = self.num_grpo_groups
        rollout_branch_count = int(self.cfg.num_generations)
        chunks = int(group_data["chunk_rewards"].shape[-1])
        frame_gamma = float(self.cfg.discount_gamma)
        chunk_gamma = frame_gamma ** max(1, int(self.cfg.horizon))
        train_chunk_rewards = group_data.get("train_chunk_rewards", group_data["chunk_rewards"])
        live_frames = group_data.get(
            "live_frames", group_data["valid_mask"].float() * float(max(1, int(self.cfg.horizon)))
        ).sum(dim=-1)
        early_stop_penalty = (
            (float(chunks * max(1, int(self.cfg.horizon))) - live_frames)
            / max(float(chunks * max(1, int(self.cfg.horizon))), 1.0)
            * float(self.cfg.terminal_penalty)
        )
        sample_rewards = group_data["rewards"] - early_stop_penalty
        train_chunk_rewards_g = train_chunk_rewards.view(env_count, rollout_branch_count, chunks)
        valid_mask_g = group_data["valid_mask"]
        valid_float = valid_mask_g.to(dtype=train_chunk_rewards_g.dtype)
        first_life_rewards = train_chunk_rewards_g * valid_float
        first_done_chunk_g = group_data["first_done_chunk"].to(device=first_life_rewards.device)
        first_done_timeout_g = group_data.get(
            "first_done_timeout", torch.zeros_like(first_done_chunk_g, dtype=torch.bool)
        ).to(device=first_life_rewards.device).bool()
        died_mask = (first_done_chunk_g < chunks) & (~first_done_timeout_g)
        if bool(died_mask.any()):
            death_idx = first_done_chunk_g.clamp(max=chunks - 1)
            death_one_hot = torch.nn.functional.one_hot(death_idx, num_classes=chunks).to(dtype=first_life_rewards.dtype)
            first_done_frame_g = group_data.get("first_done_frame", torch.zeros_like(first_done_chunk_g)).to(device=first_life_rewards.device)
            death_frame = first_done_frame_g.clamp(min=0, max=max(0, int(self.cfg.horizon) - 1))
            death_discount = (frame_gamma ** death_frame.to(dtype=first_life_rewards.dtype)).unsqueeze(-1)
            first_life_rewards = first_life_rewards - (
                died_mask.to(dtype=first_life_rewards.dtype).unsqueeze(-1)
                * death_one_hot * death_discount * float(self.cfg.terminal_penalty)
            )
        reward_to_go = torch.zeros_like(first_life_rewards)
        tail_bootstrap_g = group_data["last_values"].view(env_count, rollout_branch_count).to(
            device=first_life_rewards.device, dtype=first_life_rewards.dtype
        )
        alive_at_end_g = (~died_mask).to(dtype=first_life_rewards.dtype)
        running_return = tail_bootstrap_g * alive_at_end_g
        for chunk_idx in reversed(range(chunks)):
            running_return = first_life_rewards[:, :, chunk_idx] + chunk_gamma * running_return
            reward_to_go[:, :, chunk_idx] = running_return
            running_return = running_return * valid_float[:, :, chunk_idx]
        chunk_scores = reward_to_go
        group_baseline_mask = valid_mask_g
        advantages_by_chunk = self._compute_group_relative_advantages(
            chunk_scores.permute(0, 2, 1).reshape(env_count * chunks, rollout_branch_count),
            group_baseline_mask.permute(0, 2, 1).reshape(env_count * chunks, rollout_branch_count),
        )
        advantages_per_chunk = advantages_by_chunk.view(env_count, chunks, rollout_branch_count).permute(0, 2, 1)
        valid_flat = valid_mask_g.reshape(env_count * rollout_branch_count * chunks)
        update_flat = valid_flat
        adv_flat = advantages_per_chunk.reshape(env_count * rollout_branch_count * chunks)[update_flat]
        grpo_adv_flat = advantages_per_chunk.reshape(env_count * rollout_branch_count * chunks)[valid_flat]
        obs_flat = group_data["obs"].reshape(env_count * rollout_branch_count * chunks, -1)[update_flat]

        t1 = _time.perf_counter()
        update_metrics = self._policy_update(
            obs_flat,
            group_data["actions"].reshape(env_count * rollout_branch_count * chunks, self.action_chunk_dim)[update_flat],
            group_data["latents"].reshape(env_count * rollout_branch_count * chunks, self.cfg.flow_steps + 1, self.chunk_dim)[update_flat],
            group_data["old_log_probs"].reshape(env_count * rollout_branch_count * chunks, *group_data["old_log_probs"].shape[3:])[update_flat],
            group_data["train_step_indices"],
            adv_flat,
        )
        update_time = _time.perf_counter() - t1
        metrics = self._build_metrics(group_data, advantages_per_chunk, update_metrics, collect_time, update_time)
        metrics["group/grpo_advantage_abs_mean"] = float(grpo_adv_flat.abs().mean().item()) if grpo_adv_flat.numel() > 0 else 0.0
        metrics["group/grpo_score_mean"] = float(sample_rewards.mean().item())
        metrics["group/early_stop_penalty_mean"] = float(early_stop_penalty.mean().item())
        valid_reward_to_go = reward_to_go[valid_mask_g]
        metrics["group/reward_to_go_mean"] = float(valid_reward_to_go.mean().item()) if valid_reward_to_go.numel() > 0 else 0.0
        return metrics

    def _policy_mini_batch_size(self, sample_count: int) -> int:
        num_mini_batches = max(1, int(self.cfg.num_mini_batches))
        return max(1, sample_count // num_mini_batches)

    def _policy_micro_batch_size(self, batch_size: int) -> int:
        if self.cfg.micro_batch_size <= 0:
            return max(1, batch_size)
        return max(1, min(batch_size, int(self.cfg.micro_batch_size)))

    def _update_adaptive_learning_rate(self, observed_kl: float) -> None:
        desired_kl = float(self.cfg.desired_kl)
        if desired_kl <= 0.0:
            return
        kl_value = float(observed_kl.item() if torch.is_tensor(observed_kl) else observed_kl)
        if not math.isfinite(kl_value) or kl_value <= 0.0:
            return
        if kl_value > 2.0 * desired_kl:
            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
        elif kl_value < 0.5 * desired_kl:
            self.learning_rate = min(float(self.cfg.policy_lr), self.learning_rate * 1.5)
        self._optimizer.param_groups[0]["lr"] = self.learning_rate


    def _policy_update(self, obs, actions, latent_path, old_log_probs, train_step_indices, advantages):
        sample_count = obs.shape[0]
        if actions.ndim != 2:
            raise ValueError("MixGRPO PPO update expects one action sample per env step.")
        if latent_path.ndim != 3:
            raise ValueError("MixGRPO PPO update expects one SDE-ODE latent path per env step.")
        if old_log_probs.ndim != 2 or old_log_probs.shape != (sample_count, train_step_indices.numel()):
            raise ValueError("old_log_probs must contain per-SDE-step joint chunk scores.")
        if sample_count == 0:
            return {k: v for k, v in self._empty_policy_metrics().items()}

        mini_batch_size = self._policy_mini_batch_size(sample_count)
        advantage_abs_mean = float(advantages.detach().abs().mean().item())
        probe_count = min(128, sample_count)
        probe_obs = obs[:probe_count]
        with torch.no_grad():
            probe_action_before = self.deterministic_actions(probe_obs)
            params_before = [p.detach().clone() for p in self._policy.parameters()]

        totals = {
            "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
            "pre_clip_frac": 0.0, "pre_ratio": 0.0, "pre_ratio_min": float("inf"), "pre_ratio_max": 0.0,
            "step_clip_frac": 0.0, "step_ratio": 0.0, "step_ratio_min": float("inf"), "step_ratio_max": 0.0,
            "step_logprob_delta_abs": 0.0, "step_kl_loss": 0.0,
            "logprob_delta_abs": 0.0, "old_log_prob": 0.0, "new_log_prob": 0.0, "kl_loss": 0.0,
            "grad_norm": 0.0,
        }
        mini_batch_update_count = 0
        grad_update_count = 0
        micro_batch_count = 0
        clip_range = float(self.cfg.clip_range)

        permutation_count = max(mini_batch_size, (sample_count // mini_batch_size) * mini_batch_size)
        perm = torch.randperm(permutation_count, device=obs.device)
        for _ in range(self.cfg.policy_epochs):
            for start in range(0, permutation_count, mini_batch_size):
                mb = perm[start: start + mini_batch_size]
                mb_obs = obs[mb]
                mb_latent_path = latent_path[mb]
                mb_old_log_probs = old_log_probs[mb]
                mb_adv = advantages[mb]
                micro_batch_size = self._policy_micro_batch_size(mb.numel())
                self._optimizer.zero_grad(set_to_none=True)
                mb_size = max(1, mb.numel())
                mb_policy_loss_value = 0.0
                mb_entropy_value = 0.0
                mb_kl_loss_value = 0.0
                mb_clip_weighted = 0.0
                mb_ratio_weighted = 0.0
                mb_logprob_delta_weighted = 0.0
                mb_old_logprob_weighted = 0.0
                mb_new_logprob_weighted = 0.0
                mb_ratio_min = float("inf")
                mb_ratio_max = 0.0

                for micro_start in range(0, mb_size, micro_batch_size):
                    micro_end = min(micro_start + micro_batch_size, mb_size)
                    weight = (micro_end - micro_start) / mb_size
                    micro_old_log_probs = mb_old_log_probs[micro_start:micro_end]
                    micro_adv = mb_adv[micro_start:micro_end]
                    new_log_probs = self._compute_transition_log_probs(
                        mb_obs[micro_start:micro_end], mb_latent_path[micro_start:micro_end], train_step_indices
                    )
                    log_ratio = new_log_probs - micro_old_log_probs
                    ratio = torch.exp(log_ratio)
                    micro_adv_steps = torch.clamp(micro_adv, -float(self.cfg.adv_clip_max), float(self.cfg.adv_clip_max)).unsqueeze(-1)
                    unclipped = -micro_adv_steps * ratio
                    clipped = -micro_adv_steps * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                    micro_policy_loss = torch.maximum(unclipped, clipped).mean()
                    micro_kl_loss = 0.5 * log_ratio.square().mean()
                    micro_loss = micro_policy_loss
                    (micro_loss * weight).backward()
                    with torch.no_grad():
                        mb_policy_loss_value += float(micro_policy_loss.item() * weight)
                        mb_kl_loss_value += float(micro_kl_loss.item() * weight)
                        mb_clip_weighted += (torch.abs(ratio - 1.0) > clip_range).float().mean().item() * weight
                        mb_ratio_weighted += float(ratio.mean().item() * weight)
                        mb_logprob_delta_weighted += float(log_ratio.abs().mean().item() * weight)
                        mb_old_logprob_weighted += float(micro_old_log_probs.mean().item() * weight)
                        mb_new_logprob_weighted += float(new_log_probs.mean().item() * weight)
                        mb_ratio_min = min(mb_ratio_min, float(ratio.min().item()))
                        mb_ratio_max = max(mb_ratio_max, float(ratio.max().item()))
                    micro_batch_count += 1

                self._update_adaptive_learning_rate(mb_kl_loss_value)
                grad_norm = torch.nn.utils.clip_grad_norm_(list(self._policy.parameters()), self.cfg.max_grad_norm)
                self._optimizer.step()
                totals["policy_loss"] += mb_policy_loss_value
                totals["entropy"] += mb_entropy_value
                totals["kl_loss"] += mb_kl_loss_value
                totals["pre_clip_frac"] += mb_clip_weighted
                totals["pre_ratio"] += mb_ratio_weighted
                totals["pre_ratio_min"] = min(totals["pre_ratio_min"], mb_ratio_min)
                totals["pre_ratio_max"] = max(totals["pre_ratio_max"], mb_ratio_max)
                totals["logprob_delta_abs"] += mb_logprob_delta_weighted
                totals["old_log_prob"] += mb_old_logprob_weighted
                totals["new_log_prob"] += mb_new_logprob_weighted
                totals["grad_norm"] += float(grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm)
                mini_batch_update_count += 1
                grad_update_count += 1

                with torch.no_grad():
                    for micro_start in range(0, mb_size, micro_batch_size):
                        micro_end = min(micro_start + micro_batch_size, mb_size)
                        weight = (micro_end - micro_start) / mb_size
                        step_new_log_probs = self._compute_transition_log_probs(
                            mb_obs[micro_start:micro_end], mb_latent_path[micro_start:micro_end], train_step_indices
                        )
                        step_log_ratio = step_new_log_probs - mb_old_log_probs[micro_start:micro_end]
                        step_ratio = torch.exp(step_log_ratio)
                        totals["step_clip_frac"] += (torch.abs(step_ratio - 1.0) > clip_range).float().mean().item() * weight
                        totals["step_ratio"] += float(step_ratio.mean().item() * weight)
                        totals["step_ratio_min"] = min(totals["step_ratio_min"], float(step_ratio.min().item()))
                        totals["step_ratio_max"] = max(totals["step_ratio_max"], float(step_ratio.max().item()))
                        totals["step_logprob_delta_abs"] += float(step_log_ratio.abs().mean().item() * weight)
                        totals["step_kl_loss"] += float((0.5 * step_log_ratio.square()).mean().item() * weight)

        with torch.no_grad():
            ratio_probe_count = min(256, sample_count)
            post_new_log_probs = self._compute_transition_log_probs(
                obs[:ratio_probe_count], latent_path[:ratio_probe_count], train_step_indices
            )
            post_log_ratio = post_new_log_probs - old_log_probs[:ratio_probe_count]
            post_ratio_tensor = torch.exp(post_log_ratio)
            post_ratio = float(post_ratio_tensor.mean().item())
            post_ratio_min = float(post_ratio_tensor.min().item())
            post_ratio_max = float(post_ratio_tensor.max().item())
            post_logprob_delta_abs = float(post_log_ratio.abs().mean().item())
            post_kl_loss = float((0.5 * post_log_ratio.square()).mean().item())
            post_clip_frac = float((torch.abs(post_ratio_tensor - 1.0) > clip_range).float().mean().item())

        update_denom = max(mini_batch_update_count, 1)
        grad_denom = max(grad_update_count, 1)
        with torch.no_grad():
            probe_action_after = self.deterministic_actions(probe_obs)
            action_delta = torch.mean(torch.abs(probe_action_after - probe_action_before))
            param_delta_sq = torch.tensor(0.0, device=obs.device)
            param_count = 0
            for param, before in zip(self._policy.parameters(), params_before, strict=True):
                delta = param.detach() - before
                param_delta_sq = param_delta_sq + torch.sum(delta * delta)
                param_count += delta.numel()
            param_rms_delta = torch.sqrt(param_delta_sq / max(param_count, 1))

        policy_loss = totals["policy_loss"] / update_denom
        entropy = totals["entropy"] / update_denom
        kl_loss = totals["kl_loss"] / update_denom
        total_loss = policy_loss - float(self.cfg.entropy_coef) * entropy
        current_policy_lr = float(self._optimizer.param_groups[0]["lr"])
        return {
            "policy/loss": total_loss,
            "policy/policy_loss": policy_loss,
            "policy/value_loss": 0.0,
            "policy/entropy": entropy,
            "policy/kl_loss": kl_loss,
            "policy/joint_kl": kl_loss,
            "policy/joint_ratio": totals["pre_ratio"] / update_denom,
            "policy/joint_clip_frac": totals["pre_clip_frac"] / update_denom,
            "policy/clip_frac": totals["pre_clip_frac"] / update_denom,
            "policy/ratio": totals["pre_ratio"] / update_denom,
            "policy/ratio_min": totals["pre_ratio_min"] if totals["pre_ratio_min"] != float("inf") else 0.0,
            "policy/ratio_max": totals["pre_ratio_max"],
            "policy/logprob_delta_abs": totals["logprob_delta_abs"] / update_denom,
            "policy/old_log_prob": totals["old_log_prob"] / update_denom,
            "policy/new_log_prob": totals["new_log_prob"] / update_denom,
            "policy/step_ratio": totals["step_ratio"] / update_denom,
            "policy/step_ratio_min": totals["step_ratio_min"] if totals["step_ratio_min"] != float("inf") else 0.0,
            "policy/step_ratio_max": totals["step_ratio_max"],
            "policy/step_clip_frac": totals["step_clip_frac"] / update_denom,
            "policy/step_logprob_delta_abs": totals["step_logprob_delta_abs"] / update_denom,
            "policy/step_kl_loss": totals["step_kl_loss"] / update_denom,
            "policy/post_ratio": post_ratio,
            "policy/post_ratio_min": post_ratio_min,
            "policy/post_ratio_max": post_ratio_max,
            "policy/post_clip_frac": post_clip_frac,
            "policy/post_logprob_delta_abs": post_logprob_delta_abs,
            "policy/post_kl_loss": post_kl_loss,
            "policy/grad_norm": totals["grad_norm"] / grad_denom,
            "policy/action_delta": float(action_delta.item()),
            "policy/param_rms_delta": float(param_rms_delta.item()),
            "policy/sample_count": float(sample_count),
            "policy/advantage_abs_mean": advantage_abs_mean,
            "policy/effective_mini_batch_size": float(mini_batch_size),
            "policy/micro_batch_size": float(self._policy_micro_batch_size(mini_batch_size)),
            "policy/micro_batches": float(micro_batch_count),
            "policy/optimizer_steps": float(grad_update_count),
            "policy/lr": current_policy_lr,
            "policy/critic_lr": 0.0,
            "policy/sde_train_steps": float(train_step_indices.numel()),
        }

    def _empty_policy_metrics(self) -> dict:
        keys = [
            "policy/loss", "policy/policy_loss", "policy/value_loss", "policy/entropy",
            "policy/clip_frac", "policy/ratio", "policy/ratio_min", "policy/ratio_max",
            "policy/logprob_delta_abs", "policy/old_log_prob", "policy/new_log_prob", "policy/kl_loss",
            "policy/joint_kl", "policy/joint_ratio", "policy/joint_clip_frac",
            "policy/step_ratio", "policy/step_ratio_min", "policy/step_ratio_max", "policy/step_clip_frac",
            "policy/step_logprob_delta_abs", "policy/step_kl_loss",
            "policy/post_ratio", "policy/post_ratio_min", "policy/post_ratio_max", "policy/post_clip_frac",
            "policy/post_logprob_delta_abs", "policy/post_kl_loss",
            "policy/grad_norm", "policy/action_delta", "policy/param_rms_delta", "policy/sample_count",
            "policy/advantage_abs_mean", "policy/effective_mini_batch_size", "policy/micro_batch_size",
            "policy/micro_batches", "policy/optimizer_steps", "policy/sde_train_steps",
        ]
        out = {k: 0.0 for k in keys}
        out["policy/ratio"] = 1.0
        out["policy/joint_ratio"] = 1.0
        out["policy/lr"] = float(self._optimizer.param_groups[0]["lr"])
        out["policy/critic_lr"] = 0.0
        return out


    def _build_metrics(self, group_data, advantages, update_metrics, collect_time, update_time) -> dict:
        env = self.env
        group_rewards = group_data["rewards"]
        raw_group_rewards = group_data.get("raw_rewards", group_rewards)
        score_group_rewards = group_data.get("score_rewards")
        metric_chunk_return = group_data["metric_chunk_return"]
        metric_chunk_return_first = group_data.get("metric_chunk_return_first")
        metric_chunk_return_last = group_data.get("metric_chunk_return_last")
        metric_actions = group_data["metric_actions"]
        act_abs_tensor = metric_actions.abs()
        metric_actions_first = group_data.get("metric_actions_first")
        metric_actions_last = group_data.get("metric_actions_last")
        act_first_mean = float(metric_actions_first.abs().mean().item()) if metric_actions_first is not None else 0.0
        act_last_mean = float(metric_actions_last.abs().mean().item()) if metric_actions_last is not None else 0.0

        horizon_dim = int(self.cfg.horizon)
        actions_full = group_data.get("actions")
        alive_full = group_data.get("alive_frame")
        reward_full = group_data.get("reward_frame")

        def _alive_weighted_mean(per_value, alive_w):
            denom = alive_w.sum()
            if float(denom.item()) <= 0.0:
                return float("nan")
            return float(((per_value * alive_w).sum() / denom).item())

        frame_abs_means, frame_reward_means = [], []
        in_chunk_delta_abs = float("nan")
        if actions_full is not None and alive_full is not None:
            alive_w = alive_full.to(dtype=actions_full.dtype)
            per_frame_abs = actions_full.abs().mean(dim=-1)
            for f in range(horizon_dim):
                frame_abs_means.append(_alive_weighted_mean(per_frame_abs[..., f], alive_w[..., f]))
            if horizon_dim > 1:
                delta = (actions_full[..., 1:, :] - actions_full[..., :-1, :]).abs().mean(dim=-1)
                in_chunk_delta_abs = _alive_weighted_mean(delta, alive_w[..., 1:])
            if reward_full is not None:
                rw = reward_full.to(dtype=actions_full.dtype)
                for f in range(horizon_dim):
                    frame_reward_means.append(_alive_weighted_mean(rw[..., f], alive_w[..., f]))
        frame0_abs_mean = frame_abs_means[0] if frame_abs_means else 0.0
        frame1_abs_mean = frame_abs_means[1] if len(frame_abs_means) > 1 else float("nan")
        frame0_reward_mean = frame_reward_means[0] if frame_reward_means else float("nan")
        frame1_reward_mean = frame_reward_means[1] if len(frame_reward_means) > 1 else float("nan")

        cross_delta_sum = group_data.get("metric_cross_chunk_delta_sum")
        cross_delta_count = group_data.get("metric_cross_chunk_delta_count")
        if cross_delta_sum is not None and cross_delta_count is not None and float(cross_delta_count.item()) > 0.0:
            cross_chunk_delta_abs = float((cross_delta_sum / cross_delta_count).item())
        else:
            cross_chunk_delta_abs = float("nan")
        chunk_first_mean = float(metric_chunk_return_first.mean().item()) if metric_chunk_return_first is not None else float(metric_chunk_return.mean().item())
        chunk_last_mean = float(metric_chunk_return_last.mean().item()) if metric_chunk_return_last is not None else float("nan")

        valid_mask = group_data["valid_mask"]
        valid_advantages = advantages[valid_mask]
        act_abs = act_abs_tensor.mean(dim=(0, 1))
        final_latents = group_data["latents"][..., -1, :]
        valid_final_latents = final_latents[valid_mask]
        live_frame_counts = group_data.get("live_frames")
        if live_frame_counts is None:
            live_steps = valid_mask.float().sum(dim=-1) * self.cfg.horizon
        else:
            live_steps = live_frame_counts.sum(dim=-1)
        live_steps_flat = live_steps.flatten()
        first_done_chunk = group_data.get("first_done_chunk")
        first_done_phase = group_data.get("first_done_phase")
        first_done_anchor_pos = group_data.get("first_done_anchor_pos")
        first_done_anchor_ori = group_data.get("first_done_anchor_ori")
        first_done_ee_body = group_data.get("first_done_ee_body")
        first_done_timeout = group_data.get("first_done_timeout")
        collection_start_phases = group_data.get("collection_start_phases")
        failed_mask = first_done_phase >= 0 if first_done_phase is not None else None
        failure_phase_values = first_done_phase[failed_mask] if failed_mask is not None and bool(failed_mask.any()) else None
        failure_chunk_values = first_done_chunk[failed_mask] if failed_mask is not None and bool(failed_mask.any()) else None
        if first_done_phase is not None and failed_mask is not None and collection_start_phases is not None and bool(failed_mask.any()):
            start_phase_by_group = collection_start_phases if collection_start_phases.shape == first_done_phase.shape else collection_start_phases.reshape_as(first_done_phase)
            failure_relative_phase_values = first_done_phase[failed_mask] - start_phase_by_group[failed_mask]
        else:
            failure_relative_phase_values = None
        total_live_steps = live_steps.sum().clamp(min=1.0)
        reward_per_live_step = group_rewards.sum() / total_live_steps
        raw_reward_per_live_step = raw_group_rewards.sum() / total_live_steps
        official_scale_reward = reward_per_live_step * self.max_episode_steps
        chunk_rewards_for_metrics = group_data.get("chunk_rewards", advantages)
        chunk_objective_means = chunk_rewards_for_metrics.mean(dim=(0, 1))
        chunk_raw_means = group_data.get("raw_chunk_rewards", chunk_rewards_for_metrics).mean(dim=(0, 1))
        chunk_count_for_metrics = int(chunk_objective_means.shape[0])
        mid_chunk_index = min(max(chunk_count_for_metrics // 2, 0), chunk_count_for_metrics - 1)
        sampler_stats = env.adaptive_sampling_stats()
        legs_idx = list(range(0, 12)); waist_idx = [12, 13, 14]; arms_idx = list(range(15, 29))

        metrics = {
            **update_metrics,
            "algo/name": "mixgrpo",
            "group/reward_mean": float(official_scale_reward.item()),
            "group/reward_raw_mean": float(raw_group_rewards.mean().item()),
            "group/reward_raw_std": float(raw_group_rewards.std().item()),
            "group/reward_raw_min": float(raw_group_rewards.min().item()),
            "group/reward_raw_max": float(raw_group_rewards.max().item()),
            "group/score_reward_mean": float(score_group_rewards.mean().item()) if score_group_rewards is not None else float("nan"),
            "group/score_reward_std": float(score_group_rewards.std().item()) if score_group_rewards is not None else float("nan"),
            "group/objective_reward_raw_mean": float(group_rewards.mean().item()),
            "group/objective_reward_raw_std": float(group_rewards.std().item()),
            "group/objective_reward_raw_min": float(group_rewards.min().item()),
            "group/objective_reward_raw_max": float(group_rewards.max().item()),
            "group/reward_std": float((group_rewards / live_steps.clamp(min=1.0) * self.max_episode_steps).std().item()),
            "group/reward_min": float((group_rewards / live_steps.clamp(min=1.0) * self.max_episode_steps).min().item()),
            "group/reward_max": float((group_rewards / live_steps.clamp(min=1.0) * self.max_episode_steps).max().item()),
            "group/advantage_abs_mean": float(valid_advantages.abs().mean().item()) if valid_advantages.numel() > 0 else 0.0,
            "rollout/valid_frac": float(valid_mask.float().mean().item()),
            "rollout/return_mean": float(group_rewards.mean().item()),
            "rollout/return_std": float(group_rewards.std().item()),
            "rollout/raw_return_mean": float(raw_group_rewards.mean().item()),
            "rollout/raw_return_std": float(raw_group_rewards.std().item()),
            "rollout/chunk_return_mean": float(metric_chunk_return.mean().item()),
            "rollout/chunk_return_std": float(metric_chunk_return.std().item()),
            "rollout/chunk_return_first_mean": chunk_first_mean,
            "rollout/chunk_return_last_mean": chunk_last_mean,
            "rollout/chunk_objective_first_mean": float(chunk_objective_means[0].item()),
            "rollout/chunk_objective_mid_mean": float(chunk_objective_means[mid_chunk_index].item()),
            "rollout/chunk_objective_last_mean": float(chunk_objective_means[-1].item()),
            "rollout/chunk_raw_first_mean": float(chunk_raw_means[0].item()),
            "rollout/chunk_raw_mid_mean": float(chunk_raw_means[mid_chunk_index].item()),
            "rollout/chunk_raw_last_mean": float(chunk_raw_means[-1].item()),
            "rollout/live_steps_mean": float(live_steps.mean().item()),
            "rollout/live_steps_min": float(live_steps_flat.min().item()),
            "rollout/live_steps_p50": float(torch.quantile(live_steps_flat, 0.50).item()),
            "rollout/live_steps_p95": float(torch.quantile(live_steps_flat, 0.95).item()),
            "rollout/live_steps_max": float(live_steps_flat.max().item()),
            "rollout/success_frac": float((~failed_mask).float().mean().item()) if failed_mask is not None else 0.0,
            "rollout/first_failure_chunk_mean": float(failure_chunk_values.float().mean().item()) if failure_chunk_values is not None else float("nan"),
            "rollout/first_failure_chunk_min": float(failure_chunk_values.min().item()) if failure_chunk_values is not None else float("nan"),
            "rollout/first_failure_chunk_max": float(failure_chunk_values.max().item()) if failure_chunk_values is not None else float("nan"),
            "rollout/first_failure_phase_mean": float(failure_phase_values.float().mean().item()) if failure_phase_values is not None else float("nan"),
            "rollout/first_failure_phase_min": float(failure_phase_values.min().item()) if failure_phase_values is not None else float("nan"),
            "rollout/first_failure_phase_max": float(failure_phase_values.max().item()) if failure_phase_values is not None else float("nan"),
            "rollout/first_failure_relative_phase_mean": float(failure_relative_phase_values.float().mean().item()) if failure_relative_phase_values is not None else float("nan"),
            "rollout/first_failure_anchor_pos_frac": float((first_done_anchor_pos & failed_mask).float().mean().item()) if first_done_anchor_pos is not None and failed_mask is not None else 0.0,
            "rollout/first_failure_anchor_ori_frac": float((first_done_anchor_ori & failed_mask).float().mean().item()) if first_done_anchor_ori is not None and failed_mask is not None else 0.0,
            "rollout/first_failure_ee_body_frac": float((first_done_ee_body & failed_mask).float().mean().item()) if first_done_ee_body is not None and failed_mask is not None else 0.0,
            "rollout/first_failure_timeout_frac": float((first_done_timeout & failed_mask).float().mean().item()) if first_done_timeout is not None and failed_mask is not None else 0.0,
            "rollout/reward_per_live_step": float(reward_per_live_step.item()),
            "rollout/reward_per_live_second": float((reward_per_live_step / env.dt).item()),
            "rollout/raw_reward_per_live_step": float(raw_reward_per_live_step.item()),
            "rollout/max_episode_return_projection": float(official_scale_reward.item()),
            "phase/start_mean": float(collection_start_phases.float().mean().item()) if collection_start_phases is not None else float("nan"),
            "phase/start_min": float(collection_start_phases.min().item()) if collection_start_phases is not None else float("nan"),
            "phase/start_max": float(collection_start_phases.max().item()) if collection_start_phases is not None else float("nan"),
            "phase/start_at_min_frac": float((collection_start_phases == env.motion_start_phase).float().mean().item()) if collection_start_phases is not None else float("nan"),
            "sampler/top_bin": sampler_stats.get("top_bin", float("nan")),
            "sampler/top_prob": sampler_stats.get("top_prob", float("nan")),
            "sampler/peak_bin": sampler_stats.get("peak_bin", float("nan")),
            "sampler/failed_sum": sampler_stats.get("failed_sum", float("nan")),
            "sampler/entropy": sampler_stats.get("entropy", float("nan")),
            "timing/collect_s": collect_time,
            "timing/update_s": update_time,
            "act/abs_mean": float(act_abs_tensor.mean().item()),
            "act/abs_max": float(act_abs_tensor.max().item()),
            "act/abs_max_all": float(group_data.get("metric_action_abs_max_all", 0.0)),
            "act/abs_p95": float(torch.quantile(act_abs_tensor.flatten(), 0.95).item()),
            "act/abs_p99": float(torch.quantile(act_abs_tensor.flatten(), 0.99).item()),
            "act/legs_abs": float(act_abs[legs_idx].mean().item()),
            "act/waist_abs": float(act_abs[waist_idx].mean().item()),
            "act/arms_abs": float(act_abs[arms_idx].mean().item()),
            "latent/final_abs_mean": float(valid_final_latents.abs().mean().item()) if valid_final_latents.numel() > 0 else 0.0,
            "latent/final_abs_max": float(valid_final_latents.abs().max().item()) if valid_final_latents.numel() > 0 else 0.0,
            "act/l_shoulder_pitch": float(act_abs[15].item()), "act/r_shoulder_pitch": float(act_abs[22].item()),
            "act/l_shoulder_roll": float(act_abs[16].item()), "act/r_shoulder_roll": float(act_abs[23].item()),
            "act/l_shoulder_yaw": float(act_abs[17].item()), "act/r_shoulder_yaw": float(act_abs[24].item()),
            "act/l_elbow": float(act_abs[18].item()), "act/r_elbow": float(act_abs[25].item()),
            "act/l_wrist_roll": float(act_abs[19].item()), "act/r_wrist_roll": float(act_abs[26].item()),
            "act/l_wrist_pitch": float(act_abs[20].item()), "act/r_wrist_pitch": float(act_abs[27].item()),
            "act/l_wrist_yaw": float(act_abs[21].item()), "act/r_wrist_yaw": float(act_abs[28].item()),
            "act/first_abs_mean": act_first_mean, "act/last_abs_mean": act_last_mean,
            "act/frame0_abs_mean": frame0_abs_mean, "act/frame1_abs_mean": frame1_abs_mean,
            "act/in_chunk_delta_abs": in_chunk_delta_abs, "act/cross_chunk_delta_abs": cross_chunk_delta_abs,
            "reward/frame0_mean": frame0_reward_mean, "reward/frame1_mean": frame1_reward_mean,
        }
        done_union: dict[str, torch.Tensor] = {}
        metric_infos_list = group_data["metric_infos_list"]
        info_count = max(len(metric_infos_list), 1)
        for step_info in metric_infos_list:
            for key, value in step_info["reward_terms"].items():
                mk = f"reward/{key}_mean"
                metrics[mk] = metrics.get(mk, 0.0) + float(value.mean().item()) / info_count
            for key, value in step_info["done_terms"].items():
                done_union[key] = value.bool().clone() if key not in done_union else (done_union[key] | value.bool())
        for key, union_mask in done_union.items():
            metrics[f"done/{key}_frac"] = float(union_mask.float().mean().item())
        rollout_info_items = group_data.get("metric_rollout_info_items", [])
        rollout_reward_sums: dict[str, float] = {}
        rollout_weight_sum = 0.0


        done_rollout_sums: dict[str, float] = {}
        done_rollout_steps = 0
        for step_info, valid_mask_for_step in rollout_info_items:
            done_rollout_steps += 1
            for key, value in step_info["done_terms"].items():
                done_rollout_sums[key] = done_rollout_sums.get(key, 0.0) + float(value.float().mean().item())
            valid_mask_f = valid_mask_for_step.float()
            valid_weight = float(valid_mask_f.sum().item())
            if valid_weight <= 0.0:
                continue
            rollout_weight_sum += valid_weight
            for key, value in step_info["reward_terms"].items():
                rollout_reward_sums[key] = rollout_reward_sums.get(key, 0.0) + float((value * valid_mask_f).sum().item())
        if rollout_weight_sum > 0.0:
            for key, value_sum in rollout_reward_sums.items():
                metrics[f"reward_rollout/{key}_mean"] = value_sum / rollout_weight_sum
        if done_rollout_steps > 0:
            for key, value_sum in done_rollout_sums.items():
                metrics[f"done_rollout/{key}_frac"] = value_sum / done_rollout_steps
        train_reward_buffer = self._train_reward_buffer
        train_length_buffer = self._train_length_buffer
        if train_reward_buffer:
            metrics["train/mean_reward"] = float(sum(train_reward_buffer) / len(train_reward_buffer))
            metrics["train/mean_episode_length"] = float(sum(train_length_buffer) / len(train_length_buffer))
        else:
            metrics["train/mean_reward"] = float("nan")
            metrics["train/mean_episode_length"] = float("nan")
        metrics["train/recent_episode_count"] = float(len(train_reward_buffer))
        metrics["train/completed_episodes"] = float(self._train_completed_episodes)
        reward_weights = {
            "action_rate": -self.env.config.action_rate_weight,
            "joint_limit": -10.0, "anchor_pos_reward": 0.5, "anchor_ori_reward": 0.5,
            "body_pos_reward": 1.0, "body_ori_reward": 1.0, "body_lin_vel_reward": 1.0,
            "body_ang_vel_reward": 1.0,
            "undesired_contacts": -0.1,
        }
        weighted_positive = 0.0
        weighted_penalty = 0.0
        for reward_name, weight in reward_weights.items():
            raw_key = f"reward/{reward_name}_mean"
            if raw_key not in metrics:
                continue
            contribution = weight * metrics[raw_key] * env.dt
            metrics[f"reward_weighted/{reward_name}"] = contribution
            if contribution >= 0.0:
                weighted_positive += contribution
            else:
                weighted_penalty += contribution
        metrics["reward_weighted/positive"] = weighted_positive
        metrics["reward_weighted/penalty"] = weighted_penalty
        metrics["reward_weighted/total"] = weighted_positive + weighted_penalty
        last_values = group_data.get("last_values")
        if last_values is not None and last_values.numel() > 0:
            metrics["group/tail_return_mean"] = float(last_values.mean().item())
            metrics["group/tail_return_alive_frac"] = float((last_values != 0.0).to(dtype=torch.float32).mean().item())
        return metrics


    def log(self, update_idx: int, max_updates: int, metrics: dict) -> None:
        print(
            f"[UPDATE] {update_idx}/{max_updates} "
            f"group_reward={metrics['group/reward_mean']:.5f} "
            f"group_std={metrics['group/reward_std']:.5f} "
            f"group_raw={metrics.get('group/reward_raw_mean', float('nan')):.5f} "
            f"rollout_ret={metrics.get('rollout/return_mean', metrics['rollout/chunk_return_mean']):.5f} "
            f"chunk0_ret={metrics.get('rollout/chunk_return_first_mean', metrics['rollout/chunk_return_mean']):.5f} "
            f"chunk_last_ret={metrics.get('rollout/chunk_return_last_mean', float('nan')):.5f}",
            flush=True,
        )
        print(
            f"[REWARD_SCALE] "
            f"live_steps={metrics.get('rollout/live_steps_mean', float('nan')):.2f} "
            f"min={metrics.get('rollout/live_steps_min', float('nan')):.0f} "
            f"p50={metrics.get('rollout/live_steps_p50', float('nan')):.0f} "
            f"p95={metrics.get('rollout/live_steps_p95', float('nan')):.0f} "
            f"max={metrics.get('rollout/live_steps_max', float('nan')):.0f} "
            f"survive={metrics.get('rollout/success_frac', float('nan')):.5f} "
            f"per_step={metrics.get('rollout/reward_per_live_step', float('nan')):.5f} "
            f"per_second={metrics.get('rollout/reward_per_live_second', float('nan')):.3f} "
            f"episode30s_projection={metrics.get('rollout/max_episode_return_projection', float('nan')):.2f}",
            flush=True,
        )
        print(
            f"[OBJECTIVE] "
            f"obj_mean={metrics.get('group/objective_reward_raw_mean', float('nan')):.5f} "
            f"obj_std={metrics.get('group/objective_reward_raw_std', float('nan')):.5f} "
            f"obj_min={metrics.get('group/objective_reward_raw_min', float('nan')):.5f} "
            f"obj_max={metrics.get('group/objective_reward_raw_max', float('nan')):.5f} "
            f"raw_mean={metrics.get('group/reward_raw_mean', float('nan')):.5f} "
            f"raw_std={metrics.get('group/reward_raw_std', float('nan')):.5f} "
            f"raw_min={metrics.get('group/reward_raw_min', float('nan')):.5f} "
            f"raw_max={metrics.get('group/reward_raw_max', float('nan')):.5f} "
            f"score_mean={metrics.get('group/score_reward_mean', float('nan')):.5f} "
            f"rtg_mean={metrics.get('group/reward_to_go_mean', float('nan')):.5f} "
            f"grpo_score={metrics.get('group/grpo_score_mean', float('nan')):.5f} "
            f"early_penalty={metrics.get('group/early_stop_penalty_mean', float('nan')):.5f}",
            flush=True,
        )
        print(
            f"[PHASE] "
            f"start_mean={metrics.get('phase/start_mean', float('nan')):.2f} "
            f"start_min={metrics.get('phase/start_min', float('nan')):.0f} "
            f"start_max={metrics.get('phase/start_max', float('nan')):.0f} "
            f"start_at_min={metrics.get('phase/start_at_min_frac', float('nan')):.5f} "
            f"fail_rel_mean={metrics.get('rollout/first_failure_relative_phase_mean', float('nan')):.2f} "
            f"sampler_top_bin={metrics.get('sampler/top_bin', float('nan')):.0f} "
            f"sampler_top_prob={metrics.get('sampler/top_prob', float('nan')):.3f} "
            f"sampler_peak_bin={metrics.get('sampler/peak_bin', float('nan')):.0f} "
            f"sampler_failed_sum={metrics.get('sampler/failed_sum', float('nan')):.4f} "
            f"sampler_entropy={metrics.get('sampler/entropy', float('nan')):.3f}",
            flush=True,
        )
        print(
            f"[CHUNK_REWARD] "
            f"obj_first={metrics.get('rollout/chunk_objective_first_mean', float('nan')):.5f} "
            f"obj_mid={metrics.get('rollout/chunk_objective_mid_mean', float('nan')):.5f} "
            f"obj_last={metrics.get('rollout/chunk_objective_last_mean', float('nan')):.5f} "
            f"raw_first={metrics.get('rollout/chunk_raw_first_mean', float('nan')):.5f} "
            f"raw_mid={metrics.get('rollout/chunk_raw_mid_mean', float('nan')):.5f} "
            f"raw_last={metrics.get('rollout/chunk_raw_last_mean', float('nan')):.5f}",
            flush=True,
        )
        print(
            f"[POLICY] loss={metrics['policy/loss']:.5f} "
            f"policy_loss={metrics['policy/policy_loss']:.5f} "
            f"value_loss={metrics.get('policy/value_loss', float('nan')):.5f} "
            f"entropy={metrics.get('policy/entropy', float('nan')):.5f} "
            f"kl={metrics.get('policy/kl_loss', float('nan')):.5f} "
            f"step_ratio={metrics.get('policy/step_ratio', float('nan')):.4f} "
            f"step_clip={metrics.get('policy/step_clip_frac', float('nan')):.4f} "
            f"post_ratio={metrics.get('policy/post_ratio', metrics['policy/ratio']):.4f} "
            f"post_clip={metrics.get('policy/post_clip_frac', metrics['policy/clip_frac']):.4f} "
            f"grad={metrics['policy/grad_norm']:.5f} "
            f"lr={metrics.get('policy/lr', float('nan')):.6f} "
            f"critic_lr={metrics.get('policy/critic_lr', float('nan')):.6f} "
            f"sde_steps={metrics.get('policy/sde_train_steps', float('nan')):.0f} "
            f"opt_steps={metrics.get('policy/optimizer_steps', float('nan')):.0f}",
            flush=True,
        )
        print(
            f"[JOINT_KL] joint_kl={metrics.get('policy/joint_kl', float('nan')):.5f} "
            f"joint_ratio={metrics.get('policy/joint_ratio', float('nan')):.4f} "
            f"joint_clip={metrics.get('policy/joint_clip_frac', float('nan')):.4f}",
            flush=True,
        )
        print(
            f"[POLICY_DETAIL] "
            f"samples={metrics.get('policy/sample_count', float('nan')):.0f} "
            f"mb={metrics.get('policy/effective_mini_batch_size', float('nan')):.0f} "
            f"micro_mb={metrics.get('policy/micro_batch_size', float('nan')):.0f} "
            f"micro_steps={metrics.get('policy/micro_batches', float('nan')):.0f} "
            f"batch_ratio={metrics.get('policy/ratio', float('nan')):.4f} "
            f"batch_min={metrics.get('policy/ratio_min', float('nan')):.4f} "
            f"batch_max={metrics.get('policy/ratio_max', float('nan')):.4f} "
            f"post_ratio={metrics.get('policy/post_ratio', float('nan')):.4f} "
            f"post_min={metrics.get('policy/post_ratio_min', float('nan')):.4f} "
            f"post_max={metrics.get('policy/post_ratio_max', float('nan')):.4f} "
            f"post_clip={metrics.get('policy/post_clip_frac', float('nan')):.4f} "
            f"logp_delta_abs={metrics.get('policy/logprob_delta_abs', float('nan')):.5f} "
            f"post_logp_delta_abs={metrics.get('policy/post_logprob_delta_abs', float('nan')):.5f} "
            f"post_kl={metrics.get('policy/post_kl_loss', float('nan')):.5f} "
            f"old_logp={metrics.get('policy/old_log_prob', float('nan')):.5f} "
            f"new_logp={metrics.get('policy/new_log_prob', float('nan')):.5f} "
            f"adv_abs={metrics.get('policy/advantage_abs_mean', float('nan')):.5f} "
            f"grpo_adv_abs={metrics.get('group/grpo_advantage_abs_mean', float('nan')):.5f} "
            f"valid={metrics.get('rollout/valid_frac', float('nan')):.5f}",
            flush=True,
        )

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
        print(f"[TIME] collect={metrics['timing/collect_s']:.3f}s update={metrics['timing/update_s']:.3f}s", flush=True)
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
            f"[FIRST_FAILURE] "
            f"chunk_mean={metrics.get('rollout/first_failure_chunk_mean', float('nan')):.2f} "
            f"chunk_min={metrics.get('rollout/first_failure_chunk_min', float('nan')):.0f} "
            f"chunk_max={metrics.get('rollout/first_failure_chunk_max', float('nan')):.0f} "
            f"phase_mean={metrics.get('rollout/first_failure_phase_mean', float('nan')):.2f} "
            f"phase_min={metrics.get('rollout/first_failure_phase_min', float('nan')):.0f} "
            f"phase_max={metrics.get('rollout/first_failure_phase_max', float('nan')):.0f} "
            f"anchor_pos={metrics.get('rollout/first_failure_anchor_pos_frac', 0.0):.5f} "
            f"anchor_ori={metrics.get('rollout/first_failure_anchor_ori_frac', 0.0):.5f} "
            f"ee_body={metrics.get('rollout/first_failure_ee_body_frac', 0.0):.5f} "
            f"timeout={metrics.get('rollout/first_failure_timeout_frac', 0.0):.5f}",
            flush=True,
        )
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

    def log_banner(self) -> None:
        cfg = self.cfg
        env = self.env
        print("[INFO] Starting MixGRPO training", flush=True)
        print(
            f"[INFO] task={env.task.name} terrain={env.task.terrain} motion_file={env.task.motion_file}",
            flush=True,
        )
        print(
            f"[INFO] obs_dim={env.observation_dim} actor_type=mix_sde_ode "
            f"policy_horizon={cfg.horizon} action_dim={self.num_act} single_action_mode=True "
            f"rollout_chunks={self._chunks_per_grpo_update()} "
            f"rollout_env_steps={self._training_rollout_horizon()} "
            f"reset_noise={env.config.reset_noise} interval_pushes={env.config.interval_pushes} "
            f"observation_noise={env.config.observation_noise} "
            f"future_ref_steps=0 "
            f"phase_sampler={'adaptive' if env.config.adaptive_motion_sampling else 'uniform'} "
            f"adaptive_predecessor_ratio={env.config.adaptive_predecessor_ratio} "
            f"num_envs={env.num_envs} rollout_env_steps_target={int(cfg.rollout_env_steps)} "
            f"tail_bootstrap_steps={int(cfg.tail_bootstrap_steps)} "
            f"terminal_penalty={cfg.terminal_penalty} num_generations={cfg.num_generations} "
            f"grpo_groups={self.num_grpo_groups} init_noise_std={cfg.init_noise_std} "
            f"init_same_noise={cfg.init_same_noise} "
            f"first_generation_zero_noise={cfg.first_generation_zero_noise} "
            f"eval_initial_noise={cfg.eval_initial_noise} sde_eta={cfg.sde_eta} "
            f"flow_steps={cfg.flow_steps} actor_hidden_dims={list(cfg.actor_hidden_dims)} "
            f"activation={cfg.activation} action_squash_scale={cfg.action_squash_scale}",
            flush=True,
        )
        print(
            f"[INFO] reward=official_holosoma_9term action_rate={env.config.action_rate_weight}",
            flush=True,
        )
        _chunks = self._chunks_per_grpo_update()
        _env_frames = _chunks * int(cfg.horizon)
        _chunk_samples = int(env.num_envs) * _chunks
        print(
            f"[INFO] env_frames_per_update={_env_frames} chunk_samples_per_update={_chunk_samples} "
            f"frame_samples_per_update={_chunk_samples * int(cfg.horizon)} "
            f"sde_logprob_terms={_chunk_samples * int(cfg.flow_steps)} "
            f"(note: POLICY_DETAIL 'samples' == chunk_samples_per_update)",
            flush=True,
        )
        print(
            f"[INFO] ppo_objective=chunk_level(joint_sample) "
            f"latent_dim={self._policy.chunk_dim} "
            f"action_chunk_dim={self.action_chunk_dim}",
            flush=True,
        )
        print(
            f"[INFO] policy_epochs={cfg.policy_epochs} clip_range={cfg.clip_range} "
            f"adv_clip_max={cfg.adv_clip_max} desired_kl={cfg.desired_kl} gamma={cfg.discount_gamma} "
            f"entropy_coef={cfg.entropy_coef} value_loss_coef=0.0 "
            f"num_mini_batches={cfg.num_mini_batches} mini_batch_override=0 "
            f"micro_batch={cfg.micro_batch_size} policy_lr={cfg.policy_lr} critic_lr=0.0",
            flush=True,
        )
        print(
            "[INFO] critic_free=True actor_std_trainable=False exploration=four_step_sde_noise "
            f"first_generation_zero_sde_noise={cfg.first_generation_zero_noise}",
            flush=True,
        )
