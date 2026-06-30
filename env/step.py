from __future__ import annotations

import torch

from .config import PUSH_INTERVAL_STEP_RANGE, VELOCITY_RANGE


class MimicStepMixin:
    def step(
        self,
        actions: torch.Tensor,
        auto_reset: bool = False,
        reset_horizon: int = 1,
        loop_motion: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        previous_action = self.last_action.clone()
        previous_previous_action = self.prev_action.clone()
        self._apply_action_targets(actions)
        for _ in range(self.decimation):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
        if self.cfg.render:
            self._render_step_index += 1
            render_every = max(1, int(getattr(self.cfg, "render_every", 1)))
            if self._render_step_index % render_every == 0:
                self.sim.render()

        self.episode_steps += 1
        # Holosoma phase timing: reward/termination are evaluated against the CURRENT phase
        # ref(t) (the frame the policy observed when it chose this action). The phase is only
        # advanced to t+1 AFTER reward/termination/reset, so the NEXT observation reads ref(t+1).
        # Motion-end: the last valid reference frame (num_frames - 1) is the terminal frame. In
        # training this is NOT a done -- the survivor is teleported back into the clip below. It
        # only becomes a (motion_complete) done when terminate_on_motion_end is set (validation).
        self._motion_end_mask = self.phase_steps >= (self.motion.num_frames - 1)

        # Death frame = the phase that reward/termination are scored at (pre-advance).
        termination_phase_steps = self.phase_steps.clone()
        reward, reward_terms = self.compute_reward(actions, previous_action, previous_previous_action)
        done, done_terms, debug_terms = self.compute_termination()
        terminal_observation = None
        terminal_critic_observation = None

        # The env owns the adaptive sampler: record this step's tracking-failure death frames
        # BEFORE reset (guarded to count each episode once). A tracking failure is counted even
        # when it coincides with a timeout on the same step; a pure timeout is not a failure. The
        # EMA is folded only at the end of the step so the reset below samples from the OLD EMA
        # (official order). All algorithms simply consume the env -- none write to the sampler.
        tracking_failure = done_terms["anchor_pos_bad"] | done_terms["anchor_ori_bad"] | done_terms["ee_body_bad"]
        self._record_adaptive_failures(tracking_failure, termination_phase_steps)

        if auto_reset and bool(done.any()):
            # Capture the terminal observation at ref(t) (robot's post-physics death state)
            # before reset, then reset done envs back into the clip.
            terminal_observation = self.get_observation().clone()
            terminal_critic_observation = self.get_critic_observation().clone()
            env_ids = done.nonzero(as_tuple=False).squeeze(-1)
            reset_phases = self.sample_phase_indices(env_ids.numel(), horizon=max(1, reset_horizon))
            self.reset_envs(env_ids, phase_indices=reset_phases)

        # Advance the phase for ALL envs (reset envs go from their sampled frame k to k+1, so
        # the returned observation references ref(k+1) just like a surviving env references
        # ref(t+1)). Done in loop_motion too, then finished envs are teleported back in-clip.
        self.phase_steps += 1
        # Training roll-in (and play.py loop_motion): a surviving env that walked off the end of
        # the clip is teleported back to a freshly sampled start frame with NO done and the
        # episode timer preserved. Skipped only when motion end is a real termination
        # (validation, terminate_on_motion_end=True), where the env stops at the final frame.
        if not getattr(self, "terminate_on_motion_end", False):
            self._resample_finished_motions()

        # Fold this step's recorded failures into the sampler EMA AFTER reset/phase-advance, so
        # the reset above used the old EMA (official update order).
        self._fold_adaptive_sampler()

        self.prev_action = previous_action.clone()
        self.last_action = actions.clone()
        if auto_reset and bool(done.any()):
            self.prev_action[done] = 0.0
            self.last_action[done] = 0.0
        self._apply_interval_pushes()
        observation = self.get_observation()
        info = {
            "phase_steps": self.phase_steps.clone(),
            "reward_terms": reward_terms,
            "done_terms": done_terms,
            "debug_terms": debug_terms,
            "termination_phase_steps": termination_phase_steps,
        }
        if terminal_observation is not None:
            info["final_observation"] = terminal_observation
            info["_final_observation"] = done.clone()
        if terminal_critic_observation is not None:
            info["final_critic_observation"] = terminal_critic_observation
        return observation, reward, done, info

    def _apply_interval_pushes(self) -> None:
        if hasattr(self, "task_cfg") and not self.task_cfg.interval_pushes:
            return
        if not hasattr(self, "next_push_step"):
            return
        due_env_ids = torch.where(self.episode_steps >= self.next_push_step)[0]
        if due_env_ids.numel() == 0:
            return

        velocity_range = torch.tensor(VELOCITY_RANGE, dtype=torch.float32, device=self.device)
        low = velocity_range[:, 0].unsqueeze(0)
        high = velocity_range[:, 1].unsqueeze(0)
        velocity_delta = low + (high - low) * torch.rand((due_env_ids.numel(), 6), device=self.device)
        root_velocity = self.robot.data.root_vel_w.index_select(0, due_env_ids) + velocity_delta
        self.robot.write_root_velocity_to_sim(root_velocity, env_ids=due_env_ids)

        min_interval, max_interval = PUSH_INTERVAL_STEP_RANGE
        self.next_push_step[due_env_ids] = self.episode_steps[due_env_ids] + torch.randint(
            min_interval,
            max_interval + 1,
            (due_env_ids.numel(),),
            dtype=torch.long,
            device=self.device,
        )
