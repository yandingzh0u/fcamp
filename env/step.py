from __future__ import annotations

import torch

from .config import PUSH_INTERVAL_STEP_RANGE, VELOCITY_RANGE


class MimicStepMixin:
    def step(
        self,
        action_offsets: torch.Tensor,
        auto_reset: bool = False,
        reset_horizon: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        previous_action = self.last_action.clone()
        self._apply_action_targets(action_offsets)
        for _ in range(self.decimation):
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
        if self.cfg.render:
            self.sim.render()

        self.episode_steps += 1
        # Advance phase BEFORE reward/termination so robot(t+1) is compared against ref(t+1).
        # _apply_action_targets uses phase+1 to set the PD target, and after the sim step the
        # robot state corresponds to that next frame. Keeping phase at t (the old behaviour)
        # introduced a 1-frame mismatch in the reward/termination signal.
        self.phase_steps += 1
        self._resample_finished_motions()

        termination_phase_steps = self.phase_steps.clone()
        reward, reward_terms = self.compute_reward(action_offsets, previous_action)
        done, done_terms, debug_terms = self.compute_termination()
        terminal_observation = None

        if auto_reset:
            failure_mask = done & (~done_terms["time_out"])
            failed_env_ids = failure_mask.nonzero(as_tuple=False).squeeze(-1)
            self._record_adaptive_motion_failures(failed_env_ids, termination_phase_steps)

            if bool(done.any()):
                terminal_observation = self.get_observation().clone()
                env_ids = done.nonzero(as_tuple=False).squeeze(-1)
                reset_phases = self.sample_phase_indices(env_ids.numel(), horizon=max(1, reset_horizon))
                self.reset_envs(env_ids, phase_indices=reset_phases)

        self.last_action = action_offsets.clone()
        if auto_reset and bool(done.any()):
            self.last_action[done] = 0.0
        self._update_adaptive_motion_sampling()
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
