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
        physics_substep_actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        previous_action = self.last_action.clone()
        if physics_substep_actions is not None:
            expected_shape = (self.decimation, self.num_envs, self.action_dim)
            if tuple(physics_substep_actions.shape) != expected_shape:
                raise ValueError(
                    f"Expected physics_substep_actions {expected_shape}, "
                    f"got {tuple(physics_substep_actions.shape)}"
                )
            if getattr(self, "_strict_action_contract", False):
                self.validate_policy_actions(actions)
                applied_actions = actions
            else:
                # Preserve 2db: reward and last_action use the requested
                # high-level action; only substep target writes are clamped.
                applied_actions = actions
        else:
            maybe_applied_actions = self._apply_action_targets(actions)
            # Lightweight test/adaptor fixtures historically returned None.
            applied_actions = (
                actions
                if maybe_applied_actions is None
                else maybe_applied_actions
            )
        for substep in range(self.decimation):
            if physics_substep_actions is not None:
                self._apply_action_targets(physics_substep_actions[substep])
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
        self.episode_steps += 1
        self._motion_end_mask = next_phase_steps >= (
            self.motion.num_frames - 1
        )
        reference_phase_steps = torch.clamp(
            next_phase_steps, max=self.motion.num_frames - 1
        )
        self.phase_steps = reference_phase_steps

        termination_phase_steps = reference_phase_steps.clone()
        reward, reward_terms = self.compute_reward(
            applied_actions, previous_action
        )
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
        # From this point onward every post-action observation must expose the
        # target that was actually executed. Done envs overwrite this state
        # with their reconstructed reset target inside ``_reset_env_state``.
        self.last_action.copy_(applied_actions)
        # Capture the true post-action state before any optional reset.  FCAMP
        # normally disables auto-reset within a chunk, while this also makes the
        # semantics correct for evaluation code that uses auto_reset=True.
        imitation_frame = self.get_imitation_policy_frame()
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

        # External impulses are interventions on the edge leading to the next
        # returned observation. Never mutate an already terminal transition.
        intervention_edge_mask = self._apply_interval_pushes(
            eligible_mask=(
                ~done.bool()
                if getattr(self, "_strict_action_contract", False)
                else None
            )
        )
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
            "intervention_edge_mask": intervention_edge_mask,
            "imitation_frame": imitation_frame,
        }
        if terminal_observation is not None:
            info["final_observation"] = terminal_observation
        if terminal_critic_observation is not None:
            info["final_critic_observation"] = terminal_critic_observation
        return observation, reward, done, info

    def _apply_interval_pushes(
        self,
        eligible_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._last_interval_push_mask.zero_()
        if not self.interval_pushes:
            return self._last_interval_push_mask.clone()
        if eligible_mask is None:
            eligible_mask = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        if eligible_mask.shape != (self.num_envs,):
            raise ValueError(
                f"eligible_mask must have shape {(self.num_envs,)}, "
                f"got {tuple(eligible_mask.shape)}"
            )
        eligible_mask = eligible_mask.to(device=self.device, dtype=torch.bool)
        due_env_ids = torch.where(self.episode_steps >= self.next_push_step)[0]
        if due_env_ids.numel() > 0:
            due_env_ids = due_env_ids[
                eligible_mask.index_select(0, due_env_ids)
            ]
        if due_env_ids.numel() == 0:
            return self._last_interval_push_mask.clone()
        self._last_interval_push_mask[due_env_ids] = True
        # record each env's first-push episode-step for validation diagnostics.
        first_timers = due_env_ids[self.first_push_step[due_env_ids] < 0]
        if first_timers.numel() > 0:
            self.first_push_step[first_timers] = self.episode_steps[first_timers]

        velocity_range = torch.tensor(VELOCITY_RANGE, dtype=torch.float32, device=self.device)
        low = velocity_range[:, 0].unsqueeze(0)
        high = velocity_range[:, 1].unsqueeze(0)
        velocity_delta = low + (high - low) * torch.rand((due_env_ids.numel(), 6), device=self.device)
        root_velocity = (
            self.get_mimic_root_velocity_w().index_select(0, due_env_ids)
            + velocity_delta
        )
        write_velocity = getattr(self, "write_mimic_root_velocity_to_sim", None)
        if callable(write_velocity):
            write_velocity(root_velocity, due_env_ids)
        elif self.config.root_velocity_mode == "link":
            self.robot.write_root_link_velocity_to_sim(
                root_velocity, env_ids=due_env_ids
            )
        else:
            self.robot.write_root_velocity_to_sim(
                root_velocity, env_ids=due_env_ids
            )

        min_interval, max_interval = self.push_interval_step_range
        self.next_push_step[due_env_ids] = self.episode_steps[due_env_ids] + torch.randint(
            min_interval,
            max_interval + 1,
            (due_env_ids.numel(),),
            dtype=torch.long,
            device=self.device,
        )
        return self._last_interval_push_mask.clone()

    def _reset_interval_push_schedule(self, env_ids: torch.Tensor) -> None:
        self.first_push_step[env_ids] = -1
        min_push, max_push = self.push_interval_step_range
        self.next_push_step[env_ids] = (
            self.episode_steps.index_select(0, env_ids)
            + torch.randint(
                min_push,
                max_push + 1,
                (env_ids.numel(),),
                dtype=torch.long,
                device=self.device,
            )
        )

    def set_episode_age(
        self,
        env_ids: torch.Tensor,
        episode_age: torch.Tensor,
        *,
        reschedule_interval_pushes: bool = True,
    ) -> None:
        """Set synthetic episode ages without leaving event clocks stale."""

        if env_ids.ndim != 1 or episode_age.shape != env_ids.shape:
            raise ValueError(
                "env_ids and episode_age must be matching 1-D tensors"
            )
        if torch.is_floating_point(episode_age):
            raise ValueError("episode_age must use an integer dtype")
        episode_age = episode_age.to(
            device=self.device, dtype=self.episode_steps.dtype
        )
        if bool((episode_age < 0).any()):
            raise ValueError("episode_age must be non-negative")
        self.episode_steps.index_copy_(0, env_ids, episode_age)
        if reschedule_interval_pushes:
            self._reset_interval_push_schedule(env_ids)
