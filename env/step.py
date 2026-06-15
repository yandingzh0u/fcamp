from __future__ import annotations

import torch

from .config import PUSH_INTERVAL_STEP_RANGE, VELOCITY_RANGE


class MimicStepMixin:
    def step(
        self,
        action_offsets: torch.Tensor,
        auto_reset: bool = False,
        reset_horizon: int = 1,
        loop_motion: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        previous_action = self.last_action.clone()
        self._apply_action_targets(action_offsets)
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
        # Advance phase BEFORE reward/termination so robot(t+1) is compared against ref(t+1).
        # _apply_action_targets uses phase+1 to set the PD target, and after the sim step the
        # robot state corresponds to that next frame. Keeping phase at t (the old behaviour)
        # introduced a 1-frame mismatch in the reward/termination signal.
        self.phase_steps += 1
        # Motion-end handling. Capture which envs reached the clip end this step.
        self._motion_end_mask = self.phase_steps >= self.motion.num_frames
        if loop_motion:
            # Explicit infinite-playback mode only: silently teleport finished envs back into
            # the clip so the rollout never stops. NOT used during training or validation,
            # where reaching the clip end must register as a (timeout-style) done so episodes
            # terminate cleanly and survival is measured against the real clip length.
            self._motion_end_mask = torch.zeros_like(self._motion_end_mask)
            self._resample_finished_motions()

        termination_phase_steps = self.phase_steps.clone()
        reward, reward_terms = self.compute_reward(action_offsets, previous_action)
        done, done_terms, debug_terms = self.compute_termination()
        terminal_observation = None

        if auto_reset:
            if bool(done.any()):
                terminal_observation = self.get_observation().clone()
                env_ids = done.nonzero(as_tuple=False).squeeze(-1)
                reset_phases = self.sample_phase_indices(env_ids.numel(), horizon=max(1, reset_horizon))
                self.reset_envs(env_ids, phase_indices=reset_phases)

        self.last_action = action_offsets.clone()
        if auto_reset and bool(done.any()):
            self.last_action[done] = 0.0
        # Maintain prev_action (the action one step before last_action) so the actor can observe
        # the last inter-frame velocity (last_action - prev_action), the C1 boundary state the
        # action parametrization integrates from. On reset both collapse to 0 (zero velocity).
        self.prev_action = previous_action
        if auto_reset and bool(done.any()):
            self.prev_action[done] = 0.0
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
        velocity_delta = low + (high - low) * self._group_shared_push_rand(due_env_ids, 6)
        root_velocity = self.robot.data.root_vel_w.index_select(0, due_env_ids) + velocity_delta
        self.robot.write_root_velocity_to_sim(root_velocity, env_ids=due_env_ids)

        min_interval, max_interval = PUSH_INTERVAL_STEP_RANGE
        next_interval = self._group_shared_push_interval(due_env_ids, min_interval, max_interval)
        self.next_push_step[due_env_ids] = self.episode_steps[due_env_ids] + next_interval

    def _group_shared_push_interval(
        self,
        due_env_ids: torch.Tensor,
        min_interval: int,
        max_interval: int,
    ) -> torch.Tensor:
        """Next push interval (in steps), shared across a GRPO group.

        The branches of a group are pushed in lockstep (their `next_push_step` is replicated
        from the carrier and they share the impulse), so the *next* interval must also be
        shared or the groups would desynchronize after the first push — which matters for long
        rollouts / the tail window. With group_size <= 1 this is plain independent per-env
        sampling.
        """
        group_size = int(getattr(self, "group_size", 1))
        if group_size <= 1:
            return torch.randint(
                min_interval,
                max_interval + 1,
                (due_env_ids.numel(),),
                dtype=torch.long,
                device=self.device,
            )
        num_groups = (self.num_envs + group_size - 1) // group_size
        group_interval = torch.randint(
            min_interval,
            max_interval + 1,
            (num_groups,),
            dtype=torch.long,
            device=self.device,
        )
        group_of_due = (due_env_ids // group_size).long()
        return group_interval.index_select(0, group_of_due)

    def _group_shared_push_rand(self, due_env_ids: torch.Tensor, dim: int) -> torch.Tensor:
        """Per-due-env uniform[0,1) push samples that are shared across a GRPO group.

        The branches of a group are kept on an identical physical state and are pushed in
        lockstep (their `next_push_step` is replicated from the group leader), so a push is
        due for either all branches of a group or none of them. To keep the same-state GRPO
        comparison clean, every branch in a group must receive the *same* push impulse, so we
        draw one sample per group index and gather it back to the due envs. With group_size
        <= 1 this is plain independent per-env noise.
        """
        group_size = int(getattr(self, "group_size", 1))
        if group_size <= 1:
            return torch.rand((due_env_ids.numel(), dim), device=self.device)
        num_groups = (self.num_envs + group_size - 1) // group_size
        group_rand = torch.rand((num_groups, dim), device=self.device)
        group_of_due = (due_env_ids // group_size).long()
        return group_rand.index_select(0, group_of_due)
