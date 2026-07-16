from __future__ import annotations

import torch

from .spec import PUSH_INTERVAL_STEP_RANGE, VELOCITY_RANGE


class MimicStepMixin:
    def step(
        self,
        actions: torch.Tensor,
        auto_reset: bool = False,
        reset_horizon: int = 1,
        reference_dt: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        previous_action = self.last_action.clone()
        self._apply_action_targets(actions)
        for _ in range(self.decimation):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
        if self.render:
            self._render_step_index += 1
            if self._render_step_index % self.render_every == 0:
                self.sim.render()

        phase_start_steps = self.phase_steps.clone()
        if reference_dt is None:
            reference_frame_delta = torch.full_like(
                phase_start_steps,
                float(self.motion_frame_delta),
                dtype=torch.float32,
            )
            reference_dt_tensor = torch.full_like(reference_frame_delta, float(self.dt))
        else:
            reference_dt_tensor = torch.as_tensor(reference_dt, dtype=torch.float32, device=self.device)
            if reference_dt_tensor.ndim == 0:
                reference_dt_tensor = reference_dt_tensor.expand(self.num_envs)
            reference_dt_tensor = reference_dt_tensor.reshape(self.num_envs).clamp(min=1.0e-6)
            if self.uses_mimickit_motion_reference:
                reference_frame_delta = reference_dt_tensor * float(self.motion.fps)
            else:
                reference_frame_delta = reference_dt_tensor / float(self.dt)
        next_phase_steps = phase_start_steps + reference_frame_delta.to(dtype=phase_start_steps.dtype)
        reference_phase_steps = torch.clamp(
            next_phase_steps, max=self.motion.num_frames - 1
        )
        # Single time-state machine:
        #   (s_t, tau_t) --a_t--> (s_{t+1}, tau_{t+1})
        # Reward, terminal diagnostics and imitation alignment all use tau_{t+1}.
        self.phase_steps = reference_phase_steps
        self.episode_steps += 1

        self._motion_end_mask = next_phase_steps >= (self.motion.num_frames - 1)
        termination_phase_steps = reference_phase_steps.clone()
        reward, reward_terms = self.compute_reward(actions, previous_action)
        if reference_dt is not None:
            reward = reward * reference_frame_delta.to(dtype=reward.dtype)
            reward_terms = {
                **reward_terms,
                "reference_dt": reference_dt_tensor,
                "reference_frame_delta": reference_frame_delta,
            }
        done, done_terms, debug_terms = self.compute_termination()
        terminal_observation = None
        terminal_critic_observation = None
        # Capture the true post-action state before any optional reset.  FCAMP
        # normally disables auto-reset within a chunk, while this also makes the
        # semantics correct for evaluation code that uses auto_reset=True.
        imitation_frame = self.get_imitation_policy_frame()
        amp_policy_observation = self.get_amp_policy_observation()
        add_policy_disc_frame = self.get_add_policy_disc_frame()
        add_demo_disc_frame = self.get_add_demo_disc_frame(reference_phase_steps)
        add_policy_observation = self.get_add_policy_observation()
        reset_env_ids = torch.empty(0, dtype=torch.long, device=self.device)
        reset_phase_indices = torch.empty(0, dtype=torch.long, device=self.device)


        tracking_failure = done_terms["anchor_pos_bad"] | done_terms["anchor_ori_bad"] | done_terms["ee_body_bad"]
        self._record_adaptive_failures(tracking_failure, termination_phase_steps)

        if auto_reset and bool(done.any()):

            terminal_observation = self.get_observation().clone()
            terminal_critic_observation = self.get_critic_observation().clone()
            env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            reset_phases = self.sample_phase_indices(env_ids.numel(), horizon=max(1, reset_horizon))
            self.reset_envs(env_ids, phase_indices=reset_phases)
            reset_env_ids = env_ids
            reset_phase_indices = reset_phases

        motion_wrap_env_ids = torch.empty(0, dtype=torch.long, device=self.device)
        motion_wrap_phase_indices = torch.empty(0, dtype=torch.long, device=self.device)
        if not self.terminate_on_motion_end:
            not_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            if reset_env_ids.numel() > 0:
                not_reset[reset_env_ids] = False
            # MimicKit AMP disables motion-end termination without silently
            # reinitializing the character; reference queries clamp at the end.
            self.phase_steps[not_reset] = next_phase_steps[not_reset]


        self._fold_adaptive_sampler()

        self.last_action = actions.clone()
        if auto_reset and bool(done.any()):
            self.last_action[done] = 0.0
        self._apply_interval_pushes()
        observation = self.get_observation()
        info = {
            "reward_terms": reward_terms,
            "done_terms": done_terms,
            "debug_terms": debug_terms,
            "phase_start_steps": phase_start_steps,
            "reference_phase_steps": reference_phase_steps,
            "termination_phase_steps": termination_phase_steps,
            "imitation_frame_phase_steps": reference_phase_steps,
            "reference_dt": reference_dt_tensor,
            "reference_frame_delta": reference_frame_delta,
            "imitation_frame": imitation_frame,
            "amp_policy_observation": amp_policy_observation,
            "add_policy_disc_frame": add_policy_disc_frame,
            "add_demo_disc_frame": add_demo_disc_frame,
            "add_policy_observation": add_policy_observation,
            "reset_env_ids": reset_env_ids,
            "reset_phase_indices": reset_phase_indices,
            "motion_wrap_env_ids": motion_wrap_env_ids,
            "motion_wrap_phase_indices": motion_wrap_phase_indices,
        }
        if terminal_observation is not None:
            info["final_observation"] = terminal_observation
        if terminal_critic_observation is not None:
            info["final_critic_observation"] = terminal_critic_observation
        return observation, reward, done, info

    def _apply_interval_pushes(self) -> None:
        if not self.config.interval_pushes:
            return
        due_env_ids = torch.where(self.episode_steps >= self.next_push_step)[0]
        if due_env_ids.numel() == 0:
            return
        # record each env's first-push episode-step for validation diagnostics.
        first_timers = due_env_ids[self.first_push_step[due_env_ids] < 0]
        if first_timers.numel() > 0:
            self.first_push_step[first_timers] = self.episode_steps[first_timers]

        velocity_range = torch.tensor(VELOCITY_RANGE, dtype=torch.float32, device=self.device)
        low = velocity_range[:, 0].unsqueeze(0)
        high = velocity_range[:, 1].unsqueeze(0)
        velocity_delta = low + (high - low) * torch.rand((due_env_ids.numel(), 6), device=self.device)
        root_velocity = self.get_mimic_root_velocity_w().index_select(0, due_env_ids) + velocity_delta
        if self.uses_mimickit_motion_reference:
            self.robot.write_root_link_velocity_to_sim(root_velocity, env_ids=due_env_ids)
        else:
            self.robot.write_root_velocity_to_sim(root_velocity, env_ids=due_env_ids)

        min_interval, max_interval = PUSH_INTERVAL_STEP_RANGE
        self.next_push_step[due_env_ids] = self.episode_steps[due_env_ids] + torch.randint(
            min_interval,
            max_interval + 1,
            (due_env_ids.numel(),),
            dtype=torch.long,
            device=self.device,
        )
