from __future__ import annotations

import torch

from .spec import VELOCITY_RANGE


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
            reference_frame_delta = reference_dt_tensor * float(self.motion.fps)
        next_phase_steps = phase_start_steps + reference_frame_delta.to(dtype=phase_start_steps.dtype)
        motion_wrap_env_ids = torch.empty(0, dtype=torch.long, device=self.device)
        motion_wrap_phase_indices = torch.empty(0, dtype=torch.long, device=self.device)
        motion_resample_mask = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        beyondmimic_command_order = (
            getattr(self, "termination_mode", "tracking") == "beyondmimic"
            and getattr(self, "motion_end_behavior", "hold_last")
            == "resample_command"
            and not self.terminate_on_motion_end
        )
        self.episode_steps += 1
        if beyondmimic_command_order:
            # Official ManagerBasedRLEnv computes termination/reward against
            # the command observed by the actor. MotionCommand advances only
            # afterwards, immediately before the next observation.
            self._motion_end_mask = phase_start_steps >= (
                self.motion.num_frames - 1
            )
            reference_phase_steps = phase_start_steps
        else:
            # Existing common MimicKit path uses the post-transition reference.
            self._motion_end_mask = next_phase_steps >= (
                self.motion.num_frames - 1
            )
            reference_phase_steps = torch.clamp(
                next_phase_steps, max=self.motion.num_frames - 1
            )
            self.phase_steps = reference_phase_steps

        termination_phase_steps = reference_phase_steps.clone()
        if beyondmimic_command_order:
            done, done_terms, debug_terms = self.compute_termination()
            reward, reward_terms = self.compute_reward(actions, previous_action)
        else:
            reward, reward_terms = self.compute_reward(actions, previous_action)
            done, done_terms, debug_terms = self.compute_termination()
        if reference_dt is not None:
            reward = reward * reference_frame_delta.to(dtype=reward.dtype)
            reward_terms = {
                **reward_terms,
                "reference_dt": reference_dt_tensor,
                "reference_frame_delta": reference_frame_delta,
            }
        terminal_observation = None
        terminal_critic_observation = None
        # Capture the true post-action state before any optional reset.  FCAMP
        # normally disables auto-reset within a chunk, while this also makes the
        # semantics correct for evaluation code that uses auto_reset=True.
        if beyondmimic_command_order:
            imitation_frame = None
            amp_policy_observation = None
            add_policy_disc_frame = None
            add_demo_disc_frame = None
            add_policy_observation = None
        else:
            imitation_frame = self.get_imitation_policy_frame()
            amp_policy_observation = self.get_amp_policy_observation()
            add_policy_disc_frame = self.get_add_policy_disc_frame()
            add_demo_disc_frame = self.get_add_demo_disc_frame(
                reference_phase_steps
            )
            add_policy_observation = self.get_add_policy_observation()
        reset_env_ids = torch.empty(0, dtype=torch.long, device=self.device)
        reset_phase_indices = torch.empty(0, dtype=torch.long, device=self.device)


        tracking_failure = done_terms["anchor_pos_bad"] | done_terms["anchor_ori_bad"] | done_terms["ee_body_bad"]
        self._record_adaptive_failures(tracking_failure, termination_phase_steps)

        if auto_reset and bool(done.any()):
            if not beyondmimic_command_order:
                terminal_observation = self.get_observation().clone()
                terminal_critic_observation = self.get_critic_observation().clone()
            env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            reset_phases = self.sample_phase_indices(env_ids.numel(), horizon=max(1, reset_horizon))
            if beyondmimic_command_order:
                self._reset_env_state(env_ids, phase_indices=reset_phases)
            else:
                self.reset_envs(env_ids, phase_indices=reset_phases)
            reset_env_ids = env_ids
            reset_phase_indices = reset_phases

        if beyondmimic_command_order:
            # This mirrors CommandManager.compute after reset handling. Done
            # envs advance from their newly sampled command; surviving envs
            # advance from the command used for this transition.
            self.phase_steps += reference_frame_delta.to(
                dtype=self.phase_steps.dtype
            )
            motion_wrap_env_ids, motion_wrap_phase_indices = (
                self._resample_finished_motions()
            )
            motion_resample_mask[motion_wrap_env_ids] = True
            self._update_beyondmimic_relative_targets()

        if (
            not self.terminate_on_motion_end
            and getattr(self, "motion_end_behavior", "hold_last")
            != "resample_command"
        ):
            not_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            if reset_env_ids.numel() > 0:
                not_reset[reset_env_ids] = False
            # MimicKit AMP disables motion-end termination without silently
            # reinitializing the character; reference queries clamp at the end.
            self.phase_steps[not_reset] = next_phase_steps[not_reset]


        self._fold_adaptive_sampler()

        # Keep the persistent environment buffer allocated outside any
        # inference-mode rollout. Rebinding it to actions.clone() would turn it
        # into an inference tensor and make later validation/reset writes fail.
        self.last_action.copy_(actions)
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
            "reset_env_ids": reset_env_ids,
            "reset_phase_indices": reset_phase_indices,
            "motion_wrap_env_ids": motion_wrap_env_ids,
            "motion_wrap_phase_indices": motion_wrap_phase_indices,
            "motion_resample_mask": motion_resample_mask,
            "interval_push_mask": self._last_interval_push_mask.clone(),
        }
        if imitation_frame is not None:
            info["imitation_frame"] = imitation_frame
            info["amp_policy_observation"] = amp_policy_observation
            info["add_policy_disc_frame"] = add_policy_disc_frame
            info["add_demo_disc_frame"] = add_demo_disc_frame
            info["add_policy_observation"] = add_policy_observation
        if terminal_observation is not None:
            info["final_observation"] = terminal_observation
        if terminal_critic_observation is not None:
            info["final_critic_observation"] = terminal_critic_observation
        return observation, reward, done, info

    def _apply_interval_pushes(self) -> None:
        self._last_interval_push_mask.zero_()
        if not self.interval_pushes:
            return
        if self.beyondmimic_global_push_timer:
            # IsaacLab 2.1.0 keeps plain-function interval event timers across
            # episode resets. EventManager subtracts first, then uses a strict
            # < 1e-6 comparison and samples the next interval before invoking
            # push_by_setting_velocity.
            self.push_time_left -= self.dt
            due_env_ids = torch.where(self.push_time_left < 1.0e-6)[0]
            if due_env_ids.numel() > 0:
                low, high = self._push_interval_time_range
                self.push_time_left[due_env_ids] = low + (high - low) * torch.rand(
                    due_env_ids.numel(), device=self.device
                )
        else:
            due_env_ids = torch.where(self.episode_steps >= self.next_push_step)[0]
        if due_env_ids.numel() == 0:
            return
        self._last_interval_push_mask[due_env_ids] = True
        # record each env's first-push episode-step for validation diagnostics.
        first_timers = due_env_ids[self.first_push_step[due_env_ids] < 0]
        if first_timers.numel() > 0:
            self.first_push_step[first_timers] = self.episode_steps[first_timers]

        velocity_range = torch.tensor(VELOCITY_RANGE, dtype=torch.float32, device=self.device)
        low = velocity_range[:, 0].unsqueeze(0)
        high = velocity_range[:, 1].unsqueeze(0)
        velocity_delta = low + (high - low) * torch.rand((due_env_ids.numel(), 6), device=self.device)
        root_velocity = self.get_mimic_root_velocity_w().index_select(0, due_env_ids) + velocity_delta
        if self.config.root_velocity_mode == "link":
            self.robot.write_root_link_velocity_to_sim(root_velocity, env_ids=due_env_ids)
        else:
            self.robot.write_root_velocity_to_sim(root_velocity, env_ids=due_env_ids)

        if not self.beyondmimic_global_push_timer:
            min_interval, max_interval = self.push_interval_step_range
            self.next_push_step[due_env_ids] = self.episode_steps[due_env_ids] + torch.randint(
                min_interval,
                max_interval + 1,
                (due_env_ids.numel(),),
                dtype=torch.long,
                device=self.device,
            )

    def _reset_interval_push_schedule(self, env_ids: torch.Tensor) -> None:
        self.first_push_step[env_ids] = -1
        if self.beyondmimic_global_push_timer:
            return
        min_push, max_push = self.push_interval_step_range
        self.next_push_step[env_ids] = torch.randint(
            min_push,
            max_push + 1,
            (env_ids.numel(),),
            dtype=torch.long,
            device=self.device,
        )
